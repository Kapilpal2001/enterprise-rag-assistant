import streamlit as st
import time
import asyncio
import os
import tempfile
from langchain_qdrant import QdrantVectorStore

from engine import load_stable_engine, get_database_connection, setup_llm_and_groq
from document_processor import process_uploaded_files, wipe_database, load_local_documents
from rag_logic import get_answer
from sql_logic import get_sql_answer
from voice_utils import talk, transcribe_audio

# Advanced features
from monitoring import get_memory_usage
from visualization import visualize_vectors_tsne

st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")
st.title("🚀 Enterprise AI Assistant (Advanced Edition)")

# --- 1. AUTHENTICATION & LANGSMITH ---
st.sidebar.header("1. Authentication")
groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
langsmith_api_key = st.sidebar.text_input("Enter LangSmith API Key (Optional):", type="password")

if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = "Enterprise-RAG"

if not groq_api_key:
    st.warning("👈 Please enter your free Groq API key to continue.")
    st.stop()

# --- 2. ENGINE SETUP ---
try:
    embeddings = load_stable_engine()
except Exception:
    st.error("Environment setup is still finalizing. Please run: pip install torchvision")
    st.stop()

llm, groq_client = setup_llm_and_groq(groq_api_key)
db_client = get_database_connection()

# --- 3. PERSISTENT MEMORY LOADING ---
if "vectorstore" not in st.session_state:
    if db_client.collection_exists(collection_name="my_documents"):
        st.session_state.vectorstore = QdrantVectorStore(
            client=db_client,
            collection_name="my_documents",
            embedding=embeddings
        )
        st.sidebar.success("💾 Loaded existing database from disk!")
    else:
        st.sidebar.info("📂 No existing database found. Please upload a PDF.")

# --- 4. DOCUMENT PROCESSING ---
st.sidebar.header("2. Knowledge Base")

if st.sidebar.button("🗑️ Wipe Entire Brain"):
    wipe_database(db_client)
    if "vectorstore" in st.session_state:
        del st.session_state.vectorstore
    st.sidebar.success("Brain wiped clean! Ready for new files.")
    time.sleep(1)
    st.rerun()

uploaded_files = st.sidebar.file_uploader("Upload PDF", accept_multiple_files=True, type=['pdf'])

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        with st.spinner("Analyzing PDF and adding to your Knowledge Base..."):
            st.session_state.vectorstore = process_uploaded_files(uploaded_files, db_client, embeddings)
            st.sidebar.success("✅ Added to Knowledge Base!")
    else:
        st.sidebar.warning("Upload a file first.")

# --- NEW: SYSTEM FOOTPRINT ---
st.sidebar.header("3. System Footprint")
if st.sidebar.button("Check Memory Footprint"):
    ram_mb, ram_percent = get_memory_usage()
    st.sidebar.metric("RAM Usage", f"{ram_mb:.2f} MB", f"{ram_percent}%")

# --- NEW: VECTOR MAP ---
st.sidebar.header("4. Vector Visualization")
if st.sidebar.button("Generate t-SNE Map"):
    with st.spinner("Reducing dimensions..."):
        fig = visualize_vectors_tsne(db_client)
        if fig:
            st.sidebar.plotly_chart(fig, use_container_width=True)
        else:
            st.sidebar.error("No vectors found.")

# --- 5. UI INTERACTION & RAG ---
st.header("5. Ask Your Assistant")

data_source = st.radio("Select Knowledge Source:", ["PDF Documents", "Structured SQL Database"], horizontal=True)

t_query = st.text_input("Type here:")
a_query = st.audio_input("Or speak:")

final_query = None

if t_query:
    final_query = t_query
elif a_query:
    with st.spinner("Listening..."):
        try:
            final_query = transcribe_audio(a_query, groq_client)
        except Exception as e:
            st.error(f"Voice Error: {e}. Try typing your question instead!")

if final_query:
    if data_source == "PDF Documents" and "vectorstore" not in st.session_state:
        st.error("Process your PDF first!")
    else:
        try:
            if data_source == "PDF Documents":
                local_docs = load_local_documents()
                if not local_docs:
                    st.error("Local documents missing for BM25. Please re-upload your PDF.")
                    st.stop()
                
                with st.spinner("Executing Hybrid Search & Reranking..."):
                    # NO MORE LANGCHAIN RETRIEVERS! Direct function call.
                    ans, latency, contexts, tokens = get_answer(final_query, st.session_state.vectorstore, local_docs, llm)
                    st.success(f"**AI:** {ans}")
            else:
                with st.spinner("Executing Text-to-SQL Query..."):
                    ans, latency, contexts, tokens = get_sql_answer(final_query, groq_client)
                    st.success(f"**AI:** {ans}")
            
            # Observability Dashboard
            st.subheader("Observability Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Retrieval Latency", f"{latency*1000:.2f} ms")
            col2.metric("Cost Tracker (Tokens)", f"{tokens}")
            
            st.session_state.last_qa = {
                "question": final_query,
                "answer": ans,
                "contexts": contexts
            }
            
            p = os.path.join(tempfile.gettempdir(), f"v_{int(time.time())}.mp3")
            asyncio.run(talk(ans, p))
            st.audio(p, format="audio/mp3", autoplay=True)
            
        except Exception as e:
            st.error(f"Logic Error: {e}")