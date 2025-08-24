# Streamlit RAG LLM Chatbot

A lightweight Retrieval-Augmented Generation (RAG) chatbot built with Streamlit + LangChain.  
It loads a pre-built Chroma vector database of thesis/document content and allows you to query it using multiple LLM provider backends.

The current thesis title surfaced in the UI:  
“Determining Effects Of A Web-Based Teachers’ Professional Development Programme On Teaching Self-Efficacy Beliefs And Classroom Practice - Ketan Satish Deshmukh”

---

## Table of Contents
1. Features
2. Quick Start
3. Application Overview
4. Architecture & Data Flow
5. Retrieval & Prompting Details
6. Supported LLM Providers
7. Vector Store & Embeddings
8. Project Structure
9. Configuration & Environment
10. Running the App
11. Usage Walkthrough
12. Troubleshooting
13. Extending the Project
14. Roadmap
15. Limitations
16. Contributing
17. License
18. Acknowledgements

---

## 1. Features
- Streamlit chat UI with persistent session message history.
- Multi-provider LLM selection (OpenAI, Together, Groq, Hugging Face Hub, Anthropic, Perplexity).
- On-the-fly provider/API key entry (no local secret storage required).
- Chroma vector store (persisted locally) loaded at runtime (no regeneration each launch).
- Hugging Face MiniLM embedding model for retrieval (all-MiniLM-L6-v2).
- Simple RAG chain: similarity search → prompt assembly → model call → answer display.
- Adjustable model name input (manual override per provider).
- Graceful error handling and visible tracebacks on failure.

---

## 2. Quick Start

```
git clone https://github.com/ketand114/Streamlit_RAG_LLM_Chatbot.git
cd Streamlit_RAG_LLM_Chatbot
pip install -r requirements.txt
streamlit run RAG_LLM_VecDB.py
```

Open http://localhost:8501 and supply an API key + select provider in the sidebar.

---

## 3. Application Overview
All logic currently resides in a single script: `RAG_LLM_VecDB.py`.  
The script:
1. Renders a Streamlit UI (title, provider controls, chat interface).
2. Initializes an LLM wrapper depending on provider selection and key validation.
3. Loads a persisted Chroma DB from `./chroma_db_v2`.
4. Performs a similarity search (k=5) for each user question.
5. Builds a prompt concatenating retrieved document chunks.
6. Calls the selected chat model, then displays the answer.

---

## 4. Architecture & Data Flow

```
User Input (Chat) 
    ↓
Vector Store Similarity Search (Chroma; k=5)
    ↓
Context Assembly (join page_content)
    ↓
Prompt Template (single user role message)
    ↓
LLM Generation (selected provider model)
    ↓
Answer Display (Streamlit chat_message)
```

Session memory is an in-memory list (`st.session_state.messages`) of user/assistant turns.  
No long-term conversation memory or summarization is implemented yet.

---

## 5. Retrieval & Prompting Details
- Retrieval function: `vectorstore.similarity_search(query, k=5)`
- Embeddings: `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`
- Prompt style: A single message instructing model to answer only from provided context.
- No explicit source citation formatting yet (chunks are merged plain, no IDs).

Potential enhancements:
- Include chunk metadata and cite sources (e.g., [S1], [S2]).
- Add top_k selector in UI sidebar.
- Add answer confidence indicators.

---

## 6. Supported LLM Providers

| Provider     | Validation Heuristic       | Default Model (if blank)                  |
|--------------|----------------------------|-------------------------------------------|
| OpenAI       | Key starts with `sk-`      | gpt-4o-mini                               |
| Together     | Provided as-is             | mistralai/Mistral-7B-Instruct-v0.2        |
| Groq         | Provided as-is             | llama3-8b-8192                            |
| Hugging Face | Provided as-is             | HuggingFaceH4/zephyr-7b-beta              |
| Anthropic    | Key starts with `sk-ant-`  | claude-3-haiku-20240307                   |
| Perplexity   | Key starts with `pplx-`    | sonar-pro                                 |

Temperature is fixed at 0.7 in current implementation across providers.

---

## 7. Vector Store & Embeddings
- Directory: `./chroma_db_v2`
- Backend: Chroma (LangChain community wrapper)
- Embedding model: `all-MiniLM-L6-v2` (Hugging Face)
- Creation: Not performed dynamically in the script; the index must already exist locally.
- If the folder is missing or empty, retrieval errors will occur.

To rebuild (conceptual steps – not yet scripted):
1. Gather raw documents.
2. Split into chunks with a text splitter (e.g., RecursiveCharacterTextSplitter).
3. Embed chunks with same embedding model.
4. Persist to Chroma directory with `persist_directory=./chroma_db_v2`.

Consider adding a future `build_index.py` for reproducibility.

---

## 8. Project Structure (Current)
```
.
├─ RAG_LLM_VecDB.py          # Main Streamlit + RAG logic
├─ requirements.txt
├─ chroma_db_v2/             # Persisted Chroma vector store (binary/index files)
└─ .devcontainer/            # (Dev container configuration if used)
```

---

## 9. Configuration & Environment
No `.env` file is required; API keys are entered manually in the UI and not persisted.  
To enable optional automation or CI deployments, a future enhancement could load environment variables (e.g., `OPENAI_API_KEY`) if present and pre-fill the sidebar.

---

## 10. Running the App

Basic:
```
streamlit run RAG_LLM_VecDB.py
```

Optional (if GPU + local models are introduced):
- Install appropriate torch build.
- Swap to a local model integration (not currently done).

---

## 11. Usage Walkthrough
1. Launch app.
2. Choose a provider in the sidebar.
3. Paste an API key matching expected format (for providers with prefix validation).
4. (Optional) Enter a custom model name.
5. Type a question about the underlying thesis/documents.
6. Receive an answer synthesized from retrieved Chroma chunks.
7. Scroll to view history; state persists until page refresh or rerun.

---

## 12. Troubleshooting

| Symptom | Likely Cause | Suggested Fix |
|---------|--------------|---------------|
| “Unsupported provider or invalid API key format.” | Prefix mismatch | Check key format / provider choice |
| Empty / low-quality answers | Retrieval mismatch | Ensure `chroma_db_v2` exists and is populated |
| Exception with Chroma | Missing sqlite lib or db folder | Reinstall dependencies; confirm persistence directory |
| Slow response | Network latency / large model | Try a smaller provider model |
| Key accepted but model fails | Wrong default model name | Specify model_name manually in sidebar |

---

## 13. Extending the Project
Ideas:
- Add ingestion script to regenerate embeddings.
- Add metadata-based filtering (e.g., section headings).
- Implement streaming tokens in UI.
- Add citation rendering: show which chunks contributed.
- Add an adjustable `k` slider.
- Incorporate conversation memory summarization.
- Add evaluation scripts (retrieval recall, answer faithfulness).

---

## 14. Roadmap
- [ ] Add index build script (`build_index.py`)
- [ ] Add model streaming responses
- [ ] Add citation & source metadata in answers
- [ ] Parameter controls: top_k, temperature
- [ ] Dockerfile for containerized deployment
- [ ] Basic test suite (retrieval + prompt formatting)
- [ ] Optional environment variable auto-load
- [ ] Chunk metadata display toggle

---

## 15. Limitations
- Single-file architecture limits modularity.
- No guardrails: potential hallucination if retrieval is weak.
- No rate limit handling across providers.
- No persistent chat history beyond session state.
- Assumes prior existence of vector DB (no reproducible pipeline included yet).

---

## 16. Contributing
1. Fork repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes; add or update documentation/tests
4. Open a pull request with a clear description of rationale

Recommended tooling (suggested):
- Code style: `black`, `ruff` or `flake8`
- Testing: `pytest`

---

## 17. License
If a LICENSE file is absent, the project is effectively “all rights reserved.” Adding a standard OSI-approved license (e.g., MIT or Apache-2.0) is recommended.

---

## 18. Acknowledgements
- Streamlit for rapid UI.
- LangChain community integrations.
- Chroma for vector storage.
- Hugging Face for embedding model (all-MiniLM-L6-v2).
- LLM providers: OpenAI, Together AI, Groq, Anthropic, Perplexity, Hugging Face Hub.

---

### Appendix: Prompt Template (Conceptual)
```
You are a helpful assistant.
Use the following context (which may include text, summary of tables, and image descriptions) to answer:
Answer the question based only on the following context

Context:
{joined chunk texts}

Question: {user_question}
```

### Appendix: Regenerating Index (Future Script Sketch)
```
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

loader = DirectoryLoader("docs", glob="**/*.txt")  # adapt patterns
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
Chroma.from_documents(chunks, emb, persist_directory="./chroma_db_v2").persist()
```
