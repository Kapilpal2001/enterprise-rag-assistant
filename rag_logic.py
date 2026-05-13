import time
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

def retrieve_documents(query, vectorstore, local_docs):
    start_time = time.time()
    
    # 1. DENSE SEARCH (Semantic meaning via Qdrant)
    dense_docs = vectorstore.similarity_search(query, k=5)
    
    # 2. SPARSE SEARCH (Exact keyword match via pure Python BM25)
    tokenized_corpus = [doc.page_content.lower().split() for doc in local_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    sparse_docs = bm25.get_top_n(tokenized_query, local_docs, n=5)
    
    # 3. COMBINE & DEDUPLICATE
    unique_docs = {}
    for doc in dense_docs + sparse_docs:
        unique_docs[doc.page_content] = doc
    all_docs = list(unique_docs.values())
    
    # 4. RERANK (Score and sort using native FlashRank model)
    ranker = Ranker()
    passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(all_docs)]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerank_request)[:3] # Keep only the absolute best 3
    
    retrieval_latency = time.time() - start_time
    
    return reranked_results, retrieval_latency

def get_answer(query, vectorstore, local_docs, llm):
    reranked_results, retrieval_latency = retrieve_documents(query, vectorstore, local_docs)
    
    if len(reranked_results) == 0:
        return "I could not find any relevant information in the document.", retrieval_latency, [], 0

    # 5. GENERATE THE ANSWER
    context_text = "\n\n".join([res['text'] for res in reranked_results])
    contexts = [res['text'] for res in reranked_results]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using the context provided. Context: {context}"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    res = chain.invoke({"context": context_text, "input": query})
    
    tokens = 0
    if hasattr(res, 'response_metadata') and 'token_usage' in res.response_metadata:
        tokens = res.response_metadata['token_usage'].get('total_tokens', 0)
        
    return res.content, retrieval_latency, contexts, tokens