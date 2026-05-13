import streamlit as st
import time
import asyncio
import os
import tempfile
from langchain_qdrant import QdrantVectorStore

from engine import load_stable_engine, get_database_connection, setup_llm_and_groq
from document_processor import process_uploaded_files, wipe_database, load_local_documents, process_structured_data
from rag_logic import get_answer
from sql_logic import get_sql_answer
from unified_logic import get_unified_answer
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
        st.sidebar.info("📂 No existing database found. Please upload a document.")

if "query_vectorstore" not in st.session_state:
    from qdrant_client.models import VectorParams, Distance
    if not db_client.collection_exists(collection_name="past_queries"):
        db_client.create_collection(
            collection_name="past_queries",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    st.session_state.query_vectorstore = QdrantVectorStore(
        client=db_client,
        collection_name="past_queries",
        embedding=embeddings
    )

# --- 4. DOCUMENT PROCESSING ---
st.sidebar.header("2. Knowledge Base")

if st.sidebar.button("🗑️ Wipe Entire Brain"):
    wipe_database(db_client)
    if "vectorstore" in st.session_state:
        del st.session_state.vectorstore
    st.sidebar.success("Brain wiped clean! Ready for new files.")
    time.sleep(1)
    st.rerun()

# 1. UNSTRUCTURED DATA (PDFs, Docs, PPTs)
uploaded_docs = st.sidebar.file_uploader(
    "Upload Documents (PDF, DOCX, TXT, PPTX)", 
    accept_multiple_files=True, 
    type=['pdf', 'docx', 'txt', 'pptx']
)

if st.sidebar.button("Process Documents"):
    if uploaded_docs:
        with st.spinner("Analyzing documents and adding to Knowledge Base..."):
            st.session_state.vectorstore = process_uploaded_files(uploaded_docs, db_client, embeddings)
            st.sidebar.success("✅ Documents Added!")
    else:
        st.sidebar.warning("Upload a document first.")

st.sidebar.divider() # Adds a nice visual line between the two sections

# 2. STRUCTURED DATA (CSVs, Excel)
uploaded_structured_file = st.sidebar.file_uploader(
    "Upload Structured Data (CSV/Excel)",
    accept_multiple_files=False,
    type=['csv', 'xlsx', 'xls']
)

# --- 5. STRUCTURED DATA PROCESSING ---
if st.sidebar.button("Process Structured Data"):
    if uploaded_structured_file:
        with st.spinner("Loading structured data..."):
            try:
                msg = process_structured_data(uploaded_structured_file)
                st.session_state.active_db = "custom_data.db"
                st.sidebar.success(msg)
            except Exception as e:
                st.error(f"Failed to load structured data: {e}")
    else:
        st.sidebar.warning("Upload a CSV or Excel file first.")


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

data_source = "Combined Knowledge Base" # Replaced the radio button

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
    if "vectorstore" not in st.session_state:
        st.error("Please upload and process your documents first to fully enable the unified assistant.")
    else:
        try:
            local_docs = load_local_documents()
            if not local_docs:
                st.error("Local documents missing for BM25. Please re-upload your documents.")
                st.stop()
            
            with st.spinner("Analyzing intent and searching knowledge bases..."):
                current_db = st.session_state.get("active_db", "company_data.db")
                ans, latency, contexts, tokens, evaluation, is_fallback = get_unified_answer(
                    query=final_query, 
                    vectorstore=st.session_state.vectorstore, 
                    query_vectorstore=st.session_state.get("query_vectorstore"),
                    local_docs=local_docs, 
                    llm=llm, 
                    groq_client=groq_client,
                    db_path=current_db
                )
                
                # Save successful query for future fallbacks
                if not is_fallback and evaluation and evaluation.get('is_faithful', False):
                    if "query_vectorstore" in st.session_state:
                        st.session_state.query_vectorstore.add_texts([final_query])
                        
                st.success(f"**AI:** {ans}")
            
            # --- NEW: ADVANCED METRICS DASHBOARD ---
            st.subheader("Advanced RAG Metrics")
            
            # Safety Status
            if evaluation is None:
                st.error("🚨 Query Blocked: Failed Safety Check.")
                st.stop()
            else:
                st.success("✅ Safety Check Passed")
            
            # Metrics Columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Retrieval Latency", f"{latency*1000:.2f} ms")
            col2.metric("Cost Tracker (Tokens)", f"{tokens}")
            
            # Faithfulness Critic Score
            faith_score = evaluation.get('score', 0)
            is_faithful = evaluation.get('is_faithful', False)
            reasoning = evaluation.get('reasoning', '')
            
            col3.metric(
                "Faithfulness Score", 
                f"{faith_score}/100", 
                "Faithful" if is_faithful else "Unfaithful",
                delta_color="normal" if is_faithful else "inverse"
            )
            
            with st.expander("Critic Reasoning"):
                st.write(reasoning)
                
            with st.expander("View Retrieved Contexts (Docs/SQL/Web)"):
                for i, ctx in enumerate(contexts):
                    st.markdown(f"**Context {i+1}:**\n```\n{ctx}\n```")
            
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