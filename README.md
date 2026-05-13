# Enterprise RAG Voice Assistant 🚀

An advanced, voice-activated Retrieval-Augmented Generation (RAG) system tailored for high-precision information extraction from complex documents like PDFs, Word files, PowerPoints, and plain text. This project leverages the Groq API, Qdrant Vector Database, and Streamlit to provide an interactive, hands-free conversational AI experience.

## 🌟 Advanced Enterprise Features

This system has been upgraded with a powerful suite of advanced RAG capabilities:

*   **🛡️ Safety-First Pipeline**: Every query passes through a strict safety node before processing to prevent malicious or unsafe prompts.
*   **🔀 Autonomous Intent Router**: Automatically determines if a user's question requires information from Local Documents, the SQL Database, or Both, bypassing manual UI toggles and heavy agent frameworks like LangGraph.
*   **🌐 Live Web Search Node**: Supplements internal knowledge with real-time web context using DuckDuckGo to ensure answers are always up-to-date.
*   **🎯 Semantic Query Fallback**: Maintains a vector "memory" of past successful queries. If the system cannot answer a new question (or if there's a typo), it seamlessly searches past queries to suggest a helpful "Did you mean...?" alternative.
*   **⚖️ Native Faithfulness Critic**: A custom lightweight Python grader node that acts as a "critic" to verify response accuracy against retrieved context, replacing heavy dependencies like RAGAS.
*   **📊 Observability & Tracing (LangSmith)**: Seamlessly integrates with LangSmith for complete execution tracing, allowing you to debug and monitor every LLM call and retrieval step.
*   **🗺️ Vector Visualization**: Uses `scikit-learn` (t-SNE) and `plotly` to render an interactive 2D map of where your document chunks "live" in the semantic vector space.
*   **🧠 Custom Hybrid Search**: Uses a highly optimized custom algorithm performing parallel Dense (Qdrant) and Sparse (BM25) searches.
*   **🥇 FlashRank Reranking**: Automatically deduplicates and reranks hybrid search results using FlashRank to ensure the absolute most relevant context is sent to the LLM.
*   **🗄️ Native Text-to-SQL**: Built-in capability to query structured SQLite databases natively using pure Python and LLM prompting.
*   **⏱️ Retrieval Latency & Cost Tracking**: Displays real-time database retrieval latency and precisely tracks token consumption.
*   **🧩 Modular Architecture**: Cleanly split into highly specialized modules (`engine.py`, `document_processor.py`, `unified_logic.py`, `rag_logic.py`, `sql_logic.py`, `web_search.py`, `safety_utils.py`, `evaluator.py`).

## 📖 What is RAG? (Retrieval-Augmented Generation)

Retrieval-Augmented Generation (RAG) is a technique that enhances the accuracy and reliability of generative AI models by fetching relevant facts from an external knowledge base before generating a response. 

### Minute Details of the RAG Process:

1.  **Ingestion & Document Loading**: 
    *   Documents (e.g., PDFs, text files) are ingested into the system. 
    *   *In this project*: Automatically routes to `PyPDFLoader`, `TextLoader`, `Docx2txtLoader`, or `UnstructuredPowerPointLoader` based on the file extension.
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
*   **💾 Persistent Local Storage**: Uses Qdrant for local vector storage. Your knowledge base persists across sessions so you don't have to re-upload documents unless you want to.
*   **🔄 Dual Knowledge Sources**: Seamlessly toggle between querying unstructured documents and a structured SQL database right from the UI.
*   **🗑️ Brain Wipe Feature**: Easily clear the Qdrant database with a single click to start fresh.
*   **📄 Multi-Document Support**: Upload and process multiple PDF, Word (.docx), PowerPoint (.pptx), and Text (.txt) files concurrently.

## 🛠️ Technology Stack

*   **Frontend UI**: [Streamlit](https://streamlit.io/)
*   **LLM & Inference**: [Groq API](https://groq.com/) (`llama-3.1-8b-instant`, `llama3-70b-8192`)
*   **Embeddings**: [HuggingFace](https://huggingface.co/) (`sentence-transformers/all-MiniLM-L6-v2`)
*   **Vector Database**: [Qdrant](https://qdrant.tech/) (Local client)
*   **Orchestration**: Custom Lightweight Python Architecture
*   **Search & Reranking**: `rank_bm25` (Sparse), `flashrank` (Reranking)
*   **Web Search**: DuckDuckGo
*   **Evaluation & Observability**: Native Faithfulness Critic, LangSmith, psutil
*   **Visualization**: Plotly, Scikit-Learn (t-SNE)
*   **Speech-to-Text (ASR)**: Groq API (`whisper-large-v3`)
*   **Text-to-Speech (TTS)**: `edge-tts`

## ⚙️ Architecture Flow

1.  **User uploads documents** via the Streamlit sidebar.
2.  The application processes documents -> Splits into chunks -> Generates Embeddings -> Saves to local **Qdrant Database**.
3.  **User asks a question** via Text Input or Voice Recording.
4.  If Voice is used, it's transcribed using **Groq Whisper API**.
5.  **Unified Processing Pipeline**:
    *   **Safety Check**: The query is validated for safety.
    *   **Intent Routing**: An LLM router autonomously decides if the query needs `DOCS`, `SQL`, or `BOTH`.
    *   **Parallel Retrieval**: Context is pulled from Qdrant/BM25, the SQLite database, and optionally a Live Web Search via DuckDuckGo.
    *   **Generation**: The LLM synthesizes a unified answer based strictly on the retrieved context.
    *   **Faithfulness Critic**: The generated answer is graded against the context to prevent hallucinations.
    *   **Query Fallback**: If the system cannot answer the question, it searches a vector database of past successful queries and suggests alternatives.
6.  The text answer, Critic Reasoning, Retrieval Latency, and Token Cost are displayed on the UI, and the answer is converted to audio via **edge-tts**.

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
2.  **Build Knowledge Base**: Upload one or more document files in the sidebar and click "Process Documents". Wait for the success message.
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
