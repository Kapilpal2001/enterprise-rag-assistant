import json

def grade_faithfulness(answer, contexts, groq_client):
    """
    Acts as a 'Critic' to evaluate if the generated answer is faithful to the provided context.
    Returns a dictionary with 'score' (0-100), 'is_faithful' (bool), and 'reasoning' (str).
    """
    
    # If no contexts were used, we can't really grade faithfulness to context
    if not contexts or (len(contexts) == 1 and "No document context" in contexts[0] and "No database context" in contexts[0]):
        return {"score": 100, "is_faithful": True, "reasoning": "No context provided to check against."}
        
    combined_context = "\n---\n".join(contexts)
    
    eval_prompt = f"""
    You are an expert evaluator for an enterprise RAG system.
    Your task is to determine if the given ANSWER is faithful to the provided CONTEXT.
    An answer is "faithful" if it does not hallucinate information and all of its claims can be reasonably inferred from the context.
    
    CONTEXT:
    {combined_context}
    
    ANSWER:
    {answer}
    
    Respond strictly in JSON format with three keys:
    1. "score": An integer from 0 to 100 representing how faithful the answer is.
    2. "is_faithful": A boolean (true or false). Consider it false if the score is below 70.
    3. "reasoning": A brief 1-2 sentence explanation of why you gave this score.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result_str = response.choices[0].message.content.strip()
        result = json.loads(result_str)
        return result
    except Exception as e:
        print(f"Grader Failed: {e}")
        return {"score": 0, "is_faithful": False, "reasoning": f"Evaluation failed: {e}"}
