# Enterprise RAG Voice Assistant 🚀

An advanced, voice-activated Retrieval-Augmented Generation (RAG) system tailored for high-precision information extraction from complex documents like PDFs. This project leverages the Groq API, Qdrant Vector Database, and Streamlit to provide an interactive, hands-free conversational AI experience.

## 🌟 Advanced Enterprise Features

This system has been upgraded with a powerful suite of advanced RAG capabilities:

*   **📊 Observability & Tracing (LangSmith)**: Seamlessly integrates with LangSmith for complete execution tracing, allowing you to debug and monitor every LLM call and retrieval step.
*   **⚖️ RAGAS Evaluation Framework**: Includes built-in evaluation to automatically score answers on **Faithfulness** and **Context Precision** using the `ragas` library.
*   **🗺️ Vector Visualization**: Uses `scikit-learn` (t-SNE) and `plotly` to render an interactive 2D map of where your document chunks "live" in the semantic vector space.
*   **🧠 Custom Hybrid Search**: Replaced standard LangChain retrievers with a highly optimized custom algorithm performing parallel Dense (Qdrant) and Sparse (BM25) searches.
*   **🥇 FlashRank Reranking**: Automatically deduplicates and reranks hybrid search results using FlashRank to ensure the absolute most relevant context is sent to the LLM.
*   **⏱️ Retrieval Latency & Cost Tracking**: Displays real-time database retrieval latency (in milliseconds) and precisely tracks the number of tokens consumed by the LLM per query.
*   **💾 Memory Footprint Monitoring**: Integrates `psutil` to track and display active RAM usage directly in the Streamlit UI.
*   **🧩 Modular Architecture**: The codebase is cleanly split into highly specialized modules (`engine.py`, `document_processor.py`, `rag_logic.py`, `voice_utils.py`, `visualization.py`, `monitoring.py`, and `evaluator.py`).

## 📖 What is RAG? (Retrieval-Augmented Generation)

Retrieval-Augmented Generation (RAG) is a technique that enhances the accuracy and reliability of generative AI models by fetching relevant facts from an external knowledge base before generating a response. 

### Minute Details of the RAG Process:

1.  **Ingestion & Document Loading**: 
    *   Documents (e.g., PDFs, text files) are ingested into the system. 
    *   *In this project*: `PyPDFLoader` is used to read text from uploaded PDF files.
2.  **Chunking (Text Splitting)**: 
    *   Large documents are broken down into smaller, manageable pieces (chunks) so that they fit into the context window of LLMs and capture specific meanings.
    *   *In this project*: `RecursiveCharacterTextSplitter` with a chunk size of 600 and overlap of 50 characters is used.
3.  **Embedding**: 
    *   Text chunks are converted into numerical representations (vectors) that capture their semantic meaning. 
    *   *In this project*: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Embeddings creates 384-dimensional vectors.
4.  **Vector Storage**: 
    *   These vectors are stored in a specialized database optimized for fast similarity searches.
    *   *In this project*: **Qdrant** is used locally (`./qdrant_storage`) using `Cosine` distance to measure similarity.
5.  **Retrieval & Reranking**: 
    *   When a user asks a question, the system executes a dense semantic search (via Qdrant) and a sparse keyword search (via BM25). The results are combined, deduplicated, and reranked using a local FlashRank model to identify the most relevant context.
    *   *In this project*: Combines top 5 dense and top 5 sparse results, reranks them, and keeps the top 3 absolute best chunks.
6.  **Generation (Augmentation)**: 
    *   The retrieved text chunks (context) and the original user query are passed together to a Large Language Model (LLM) as a combined prompt. The LLM generates a well-reasoned, highly accurate answer based *strictly* on the provided context.
    *   *In this project*: **Llama 3.1 8B Instant** (via Groq API) is prompted to answer using only the provided context.

## ✨ Project Features

*   **🎙️ Voice Interaction**: Built-in voice input capabilities using Groq's Whisper API (`whisper-large-v3`) for seamless, hands-free questioning.
*   **🔊 Audio Responses**: Reads out the AI-generated responses automatically using `edge-tts` (Aria Neural Voice).
*   **⚡ Blazing Fast Generation**: Powered by the Groq API for near-instantaneous LLM inference using LLaMA 3.1 8B.
*   **💾 Persistent Local Storage**: Uses Qdrant for local vector storage. Your knowledge base persists across sessions so you don't have to re-upload PDFs unless you want to.
*   **🗑️ Brain Wipe Feature**: Easily clear the Qdrant database with a single click to start fresh.
*   **📄 Multi-Document Support**: Upload and process multiple PDFs concurrently.

## 🛠️ Technology Stack

*   **Frontend UI**: [Streamlit](https://streamlit.io/)
*   **LLM & Inference**: [Groq API](https://groq.com/) (`llama-3.1-8b-instant`)
*   **Embeddings**: [HuggingFace](https://huggingface.co/) (`sentence-transformers/all-MiniLM-L6-v2`)
*   **Vector Database**: [Qdrant](https://qdrant.tech/) (Local client)
*   **Orchestration**: [LangChain](https://www.langchain.com/) (Document loaders, splitters, prompts)
*   **Search & Reranking**: `rank_bm25` (Sparse), `flashrank` (Reranking)
*   **Evaluation & Observability**: RAGAS, LangSmith, psutil
*   **Visualization**: Plotly, Scikit-Learn (t-SNE)
*   **Speech-to-Text (ASR)**: Groq API (`whisper-large-v3`)
*   **Text-to-Speech (TTS)**: `edge-tts`

## ⚙️ Architecture Flow

1.  **User uploads PDFs** via the Streamlit sidebar.
2.  The application processes PDFs -> Splits into chunks -> Generates Embeddings -> Saves to local **Qdrant Database**.
3.  **User asks a question** via Text Input or Voice Recording.
4.  If Voice is used, it's transcribed using **Groq Whisper API**.
5.  The system executes a custom hybrid search (Qdrant + BM25) and uses **FlashRank** to rerank the combined results to find the **top 3 document chunks**.
6.  The retrieved context + user query are sent to **Groq Llama 3.1 8B**.
7.  The LLM generates a contextual answer.
8.  The text answer is displayed on the UI alongside **Retrieval Latency** and **Token Cost**, and converted to audio via **edge-tts**.

## 🚀 Installation & Setup

### 1. Prerequisites
*   Python 3.8+
*   A free API key from [Groq Console](https://console.groq.com/keys)
*   *(Optional)* A LangSmith API Key for tracing.

### 2. Clone/Setup Directory
Navigate to your project directory:
```bash
cd path/to/PROJECTS/RAG
```

### 3. Install Dependencies
Install the required packages. (Note: Ensure you have `torch` or `torchvision` installed if required by HuggingFace transformers depending on your OS).
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the Streamlit server:
```bash
streamlit run app.py
```

## 🎮 How to Use

1.  **Authenticate**: Enter your Groq API Key (and optional LangSmith API Key) in the sidebar.
2.  **Build Knowledge Base**: Upload one or more PDF files in the sidebar and click "Process Documents". Wait for the success message.
3.  **Monitor Footprint**: Use the sidebar buttons to track your active RAM usage and visualize your vector space on a 2D map.
4.  **Ask Questions**: 
    *   **Type**: Use the text box labeled "Type here:".
    *   **Speak**: Click the microphone icon labeled "Or speak:", record your question, and stop the recording.
5.  **Evaluate**: Click the "Evaluate Last Answer" button to trigger the RAGAS framework.

## ⚠️ Troubleshooting

*   **"Environment setup is still finalizing"**: This means the `sentence-transformers` library lacks its underlying dependencies like `torch` or `torchvision`. Run `pip install torch torchvision`.
*   **Voice Error**: Ensure your microphone permissions are granted in the browser. 
*   **Database Lock Issues**: The application uses a single global `QdrantClient` connection to prevent SQLite locking issues when using Streamlit's reruns.

---
*Developed for advanced document querying, rigorous evaluation, and hands-free accessibility.*
