import time
from langchain_community.tools import DuckDuckGoSearchRun

def perform_web_search(query):
    """
    Performs a live internet search using DuckDuckGo to gather supplementary context.
    Returns the search results as a string and the latency.
    """
    start_time = time.time()
    try:
        search = DuckDuckGoSearchRun()
        results = search.invoke(query)
        latency = time.time() - start_time
        return results, latency
    except Exception as e:
        print(f"Web Search Failed: {e}")
        return f"Failed to retrieve web context: {e}", time.time() - start_time
