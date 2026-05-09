import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from groq import Groq
from qdrant_client import QdrantClient

@st.cache_resource
def load_stable_engine():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_database_connection():
    return QdrantClient(path="./qdrant_storage")

def setup_llm_and_groq(groq_api_key):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0)
    groq_client = Groq(api_key=groq_api_key)
    return llm, groq_client
