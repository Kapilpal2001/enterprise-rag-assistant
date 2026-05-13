import sqlite3
import time

def get_db_schema(db_path='company_data.db'):
    """Extracts the schema from the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = []
    for table_name in tables:
        table_name = table_name[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_details = [f"{col[1]} ({col[2]})" for col in columns]
        schema_info.append(f"Table: {table_name} | Columns: {', '.join(column_details)}")
        
    conn.close()
    return "\n".join(schema_info)

def execute_sql_query(query, groq_client, db_path='company_data.db'):
    """Generates and executes SQL, returning raw data."""
    start_time = time.time()
    
    # 1. Get Schema
    schema = get_db_schema(db_path)
    
    # 2. Ask LLM to generate SQL
    sql_prompt = f"""
You are an expert SQLite developer.
Here is the database schema:
{schema}

Write a valid SQLite query to answer the following question: "{query}"

Rules:
- Return ONLY the raw SQL query.
- Do not include markdown blocks, backticks, or the word 'sql'.
- Do not include any explanations.
- Ensure the query is valid and matches the schema exactly.
"""
    
    try:
        messages = [{"role": "user", "content": sql_prompt}]
        max_retries = 3
        attempt = 0
        raw_sql = ""
        db_results = None
        total_tokens = 0
        
        while attempt < max_retries:
            attempt += 1
            sql_response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0
            )
            total_tokens += sql_response.usage.total_tokens
            raw_sql = sql_response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting just in case
            raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
            
            # 3. Execute SQL
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            try:
                cursor.execute(raw_sql)
                db_results = cursor.fetchall()
                conn.close()
                break  # Success! Break out of the retry loop
                
            except sqlite3.Error as e:
                conn.close()
                if attempt == max_retries:
                    raise Exception(f"Failed to generate valid SQL after {max_retries} attempts. Last error: {e}")
                
                # Self-Correction: Append the bad attempt and the error to the chat history
                messages.append({"role": "assistant", "content": raw_sql})
                messages.append({
                    "role": "user", 
                    "content": f"The query failed with the following SQLite error: {e}\nPlease fix the query and try again. Return ONLY the raw SQL query without explanations."
                })
                print(f"⚠️ SQL Error Caught (Attempt {attempt}): {e}. Retrying autonomously...")
        
        retrieval_latency = time.time() - start_time
        return db_results, raw_sql, retrieval_latency, total_tokens
        
    except Exception as e:
        return f"Error: {str(e)}", "", time.time() - start_time, 0

def get_sql_answer(query, groq_client, db_path='company_data.db'):
    """Legacy wrapper for isolated SQL mode."""
    start_time = time.time()
    
    db_results, raw_sql, retrieval_latency, total_tokens = execute_sql_query(query, groq_client, db_path)
    
    if isinstance(db_results, str) and db_results.startswith("Error:"):
        return f"An error occurred while querying the database: {db_results}", retrieval_latency, [], total_tokens

    # 4. Generate Final Answer
    answer_prompt = f"""
The user asked: "{query}"
The SQL query executed was: {raw_sql}
The database returned the following results: {db_results}

Formulate a concise, friendly, and natural human-sounding answer based on these results.
Do not mention the SQL query in your response unless necessary.
"""
    try:
        final_response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.3
        )
        
        final_answer = final_response.choices[0].message.content.strip()
        tokens = total_tokens + final_response.usage.total_tokens
        
        return final_answer, retrieval_latency, [f"SQL Query Executed: {raw_sql}\nDatabase Results: {db_results}"], tokens
    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}", time.time() - start_time, [], total_tokens
