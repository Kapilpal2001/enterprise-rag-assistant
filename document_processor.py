import os
import tempfile
import pickle
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

def load_local_documents():
    docs_path = "local_docs.pkl"
    if os.path.exists(docs_path):
        with open(docs_path, "rb") as f:
            return pickle.load(f)
    return []

def process_uploaded_files(uploaded_files, db_client, embeddings):
    docs = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(tmp_path)
        elif file_extension == '.txt':
            loader = TextLoader(tmp_path, encoding='utf-8')
        elif file_extension == '.docx':
            loader = Docx2txtLoader(tmp_path)
        elif file_extension == '.pptx':
            loader = UnstructuredPowerPointLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            os.unlink(tmp_path)
            continue
            
        try:
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            
        os.unlink(tmp_path)
    
    splits = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50).split_documents(docs)
    
    with open("local_docs.pkl", "wb") as f:
        pickle.dump(splits, f)
    
    if not db_client.collection_exists(collection_name="my_documents"):
        db_client.create_collection(
            collection_name="my_documents",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        
    vector_store = QdrantVectorStore(
        client=db_client,
        collection_name="my_documents",
        embedding=embeddings
    )
    
    vector_store.add_documents(splits)
    return vector_store

def wipe_database(db_client):
    if db_client.collection_exists(collection_name="my_documents"):
        db_client.delete_collection(collection_name="my_documents")
    if os.path.exists("local_docs.pkl"):
        os.remove("local_docs.pkl")
