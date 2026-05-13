import time
from langchain_core.prompts import ChatPromptTemplate
from rag_logic import retrieve_documents
from sql_logic import execute_sql_query
from safety_utils import check_safety
from web_search import perform_web_search
from evaluator import grade_faithfulness

def route_intent(query, groq_client):
    """
    Uses the LLM to determine if the query requires:
    - 'DOCS' (Local Documents only)
    - 'SQL' (Database only)
    - 'BOTH' (Needs cross-referencing)
    """
    router_prompt = f"""
    You are an intelligent router for an enterprise query system.
    The system has two data sources:
    1. DOCS: A repository of unstructured text documents (PDFs, PPTXs, DOCXs) containing company policies, guidelines, and manuals.
    2. SQL: A SQLite database named 'company_data.db' with tables 'employees' and 'sales', containing structured data like employee salaries, departments, and sales amounts.
    
    Given the user's question, which data source(s) should be queried to provide a complete answer?
    Respond with EXACTLY ONE word: 'DOCS', 'SQL', or 'BOTH'.
    
    User Question: "{query}"
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0,
            max_tokens=10
        )
        intent = response.choices[0].message.content.strip().upper()
        if "BOTH" in intent:
            return "BOTH"
        elif "SQL" in intent:
            return "SQL"
        else:
            return "DOCS"
    except Exception:
        # Default to BOTH if routing fails just to be safe
        return "BOTH"

def get_unified_answer(query, vectorstore, query_vectorstore, local_docs, llm, groq_client, db_path='company_data.db'):
    """
    Orchestrates the unified RAG pipeline.
    Routes the query, retrieves necessary context from SQL/Docs/Web,
    generates a single final answer, and evaluates its faithfulness.
    """
    start_time = time.time()
    total_tokens = 0
    contexts = []
    
    # 0. Safety Check Node
    is_safe = check_safety(query, groq_client)
    if not is_safe:
        return "⚠️ **Security Alert:** Your query was flagged as unsafe (e.g., contains sensitive PII or malicious instructions) and has been blocked.", 0, ["Blocked by Safety Node"], 0, None

    # 1. Routing Node
    intent = route_intent(query, groq_client)
    
    pdf_context = ""
    sql_context = ""
    web_context = ""
    
    # 2. Retrieve Documents (if needed)
    if intent in ["DOCS", "BOTH"]:
        reranked_results, doc_latency = retrieve_documents(query, vectorstore, local_docs)
        if reranked_results:
            pdf_context = "\n\n".join([res['text'] for res in reranked_results])
            contexts.extend([res['text'] for res in reranked_results])
    
    # 3. Retrieve SQL Data (if needed)
    if intent in ["SQL", "BOTH"]:
        db_results, raw_sql, sql_latency, sql_tokens = execute_sql_query(query, groq_client, db_path)
        total_tokens += sql_tokens
        if not (isinstance(db_results, str) and db_results.startswith("Error:")):
            sql_context = f"SQL Query Executed: {raw_sql}\nDatabase Results: {db_results}"
            contexts.append(sql_context)
            
    # 3.5. Live Web Search Node (Always run as supplementary)
    web_results, web_latency = perform_web_search(query)
    if web_results and not web_results.startswith("Failed"):
        web_context = f"Live Web Context: {web_results}"
        contexts.append(web_context)
            
    # 4. Generate Final Answer
    final_prompt_template = """
    You are an intelligent enterprise assistant answering a user's question.
    
    You have retrieved information from the following sources (if any):
    
    --- DOCUMENT CONTEXT ---
    {pdf_context}
    
    --- DATABASE CONTEXT ---
    {sql_context}
    
    --- WEB CONTEXT ---
    {web_context}
    
    ---
    Answer the user's question based ONLY on the provided context. 
    If you use information from the Document Context, try to include a brief citation if metadata is present.
    If you use information from the Database Context or Web Context, formulate it naturally.
    If the context does not contain the answer, say "I don't have enough information to answer that."
    
    User Question: {query}
    """
    
    prompt = ChatPromptTemplate.from_template(final_prompt_template)
    chain = prompt | llm
    
    res = chain.invoke({
        "pdf_context": pdf_context if pdf_context else "No document context retrieved.",
        "sql_context": sql_context if sql_context else "No database context retrieved.",
        "web_context": web_context if web_context else "No web context retrieved.",
        "query": query
    })
    
    if hasattr(res, 'response_metadata') and 'token_usage' in res.response_metadata:
        total_tokens += res.response_metadata['token_usage'].get('total_tokens', 0)
        
    retrieval_latency = time.time() - start_time
    
    final_answer = res.content
    is_fallback = False
    
    # 5. Fallback Search for Past Queries
    if "I don't have enough information" in final_answer:
        # Check if we have past queries
        similar_queries = []
        if query_vectorstore:
            try:
                # K=3 to get top 3 similar past queries
                similar_queries = query_vectorstore.similarity_search(query, k=3)
            except Exception:
                pass
                
        if similar_queries:
            suggestions = "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(similar_queries)])
            final_answer = f"I couldn't find an exact answer to your specific question. Did you mean one of these previous queries?\n\n{suggestions}"
        else:
            final_answer = "I couldn't find an exact answer to your question, and I don't have any similar past queries to suggest yet."
            
        is_fallback = True

    # 6. Faithfulness Critic Node
    if is_fallback:
        evaluation = {"score": 100, "is_faithful": True, "reasoning": "Fallback response triggered due to lack of context. Suggesting past queries."}
    else:
        evaluation = grade_faithfulness(final_answer, contexts, groq_client)
    
    # Add a note about the intent chosen
    contexts.insert(0, f"System routed intent as: {intent}")
    if is_fallback:
        contexts.insert(1, "Fallback Triggered: Semantic search used on past queries collection.")
    
    return final_answer, retrieval_latency, contexts, total_tokens, evaluation, is_fallback
