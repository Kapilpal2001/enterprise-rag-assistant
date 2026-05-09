# Implement Hybrid Search and FlashRank

This plan details the implementation of Hybrid Search (Dense + Sparse) and FlashRank (Reranking) into the RAG Assistant, keeping them highly modular as requested.

## User Review Required

- **Dependencies**: This requires installing two new packages: `rank_bm25` (for BM25 sparse search) and `flashrank` (for reranking).
- **Persistent Storage**: To enable BM25 search across application restarts without re-uploading PDFs, we need to save the raw text document chunks locally. I will implement a lightweight `pickle` based storage (`local_docs.pkl`) alongside the Qdrant database.

## Proposed Changes

### Requirements
#### [MODIFY] `requirements.txt`
- Add `rank_bm25`
- Add `flashrank`

### New Modules
#### [NEW] `hybrid_search.py`
- Implements `get_hybrid_retriever(vectorstore, documents, k=3)`.
- Uses `langchain_community.retrievers.BM25Retriever` to create a sparse keyword search index over the documents.
- Uses Langchain's `EnsembleRetriever` to seamlessly combine the Qdrant Vector search (dense) with the BM25 search (sparse) at a 60/40 weight ratio.

#### [NEW] `reranker.py`
- Implements `get_flashrank_retriever(base_retriever, top_n=3)`.
- Uses `FlashrankRerank` as a document compressor.
- Wraps the Hybrid `EnsembleRetriever` with a `ContextualCompressionRetriever` to re-score and re-order the combined search results for maximum accuracy.

### Application Logic Updates
#### [MODIFY] `document_processor.py`
- Update `process_uploaded_files` to save the generated document chunks (`splits`) to a file named `local_docs.pkl` using Python's `pickle` module.
- Update `wipe_database` to also delete `local_docs.pkl` when the user clears their knowledge base.
- Add a helper function `load_local_documents()` to retrieve these chunks on app startup.

#### [MODIFY] `rag_logic.py`
- Change `get_answer(query, vectorstore, llm)` to `get_answer(query, retriever, llm)`.
- Remove the internal retriever creation step, allowing the application to inject our powerful, pre-configured Hybrid + FlashRank retriever directly into the logic.

#### [MODIFY] `app.py`
- Import the new functionalities.
- During application state initialization, if the vector database exists, load the `local_docs.pkl`.
- Build the pipeline: `VectorStore` + `Local Docs` -> `Hybrid Retriever` -> `FlashRank Retriever`.
- Pass this advanced pipeline into `get_answer()`.

## Verification Plan

### Automated Tests
- N/A

### Manual Verification
- Install the new dependencies via `pip install -r requirements.txt`.
- Upload a PDF to trigger the creation of `local_docs.pkl` and the Qdrant DB.
- Ask a highly specific keyword question to verify that BM25 catches it even if the semantic vector search struggles.
- Check the Streamlit UI to ensure no errors occur during retrieval and that answers remain fast and accurate.
