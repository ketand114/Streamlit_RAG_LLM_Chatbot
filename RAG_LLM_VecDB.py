
import streamlit as st
import json
import uuid
from base64 import b64decode
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_together import ChatTogether
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatHuggingFace
from langchain_anthropic import ChatAnthropic
from langchain_perplexity import ChatPerplexity
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

st.title("🧪 Test Chat with Thesis Document")

# Sidebar: Choose provider & keys
provider = st.sidebar.selectbox(
    "Choose LLM Provider:",
    ("OpenAI", "Together", "Groq", "Hugging Face", "Anthropic", "Perplexity")
)
api_key = st.sidebar.text_input(f"{provider} API Key", type="password")
model_name = st.sidebar.text_input("Model name (optional)", "")
#json_file = st.sidebar.file_uploader("Upload your JSON data", type="json")

json_file = "rag_input.json"

model = None
#embedding_model = None
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

if api_key:
    try:
        if provider == "OpenAI" and api_key.startswith("sk-"):
            model = ChatOpenAI(
                api_key=api_key,
                model=model_name or "gpt-4o-mini",
                temperature=0.7
            )

        elif provider == "Together":
            model = ChatTogether(
                together_api_key=api_key,
                model=model_name or "mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.7
            )

        elif provider == "Groq":
            model = ChatGroq(
                groq_api_key=api_key,
                model_name=model_name or "llama3-8b-8192",
                temperature=0.7
            )

        elif provider == "Hugging Face":
            # Typical model e.g. "HuggingFaceH4/zephyr-7b-beta"
            model = ChatHuggingFace(
                huggingfacehub_api_token=api_key,
                repo_id=model_name or "HuggingFaceH4/zephyr-7b-beta",
                temperature=0.7
            )

        elif provider == "Anthropic" and api_key.startswith("sk-ant-"):
            model = ChatAnthropic(
                anthropic_api_key=api_key,
                model_name=model_name or "claude-3-haiku-20240307",
                temperature=0.7
            )

        elif provider == "Perplexity" and api_key.startswith("pplx-"):
            model = ChatPerplexity(
                perplexity_api_key=api_key,
                model=model_name or "pplx-7b-online",
                temperature=0.7
            )
        else:
            st.error("Unsupported provider or invalid API key format.")
    except Exception as e:
        st.error(f"Error initializing model: {e}")

# Helper to index items
def index_items(items, summary_key, content_key=None):
    ids = [str(uuid.uuid4()) for _ in items]
    # Summaries for vector search
    summaries = [
        Document(
            page_content=str(item.get(summary_key, "")),  # Ensure string
            metadata={id_key: ids[i]}
        )
        for i, item in enumerate(items)
    ]
    retriever.vectorstore.add_documents(summaries)

    # Full docs for retrieval
    full_docs = []
    for item in items:
        raw_content = item.get(content_key, item.get(summary_key, ""))
        if not isinstance(raw_content, str):
            raw_content = json.dumps(raw_content, ensure_ascii=False)  # Convert lists/dicts to JSON string
        full_docs.append(
            Document(
                page_content=raw_content,
                metadata=item
            )
        )
    retriever.docstore.mset(list(zip(ids, full_docs)))


if model and json_file:
    # Load JSON data
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = data.get("texts", [])
    tables = data.get("tables", [])
    images = data.get("images", [])
    
    # Local embedding model (no API key needed)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create Chroma vector store & retriever
    vectorstore = Chroma(collection_name="rag_demo", embedding_function=embedding_model)
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
        
    # Index each data type
    index_items(texts, summary_key="summary", content_key="text")
    index_items(tables, summary_key="summary", content_key="rows")
    index_items(images, summary_key="description", content_key="description")

   
    # Build RAG chain
    def parse_docs(docs):
        """Separate texts and base64 images"""
        b64, text = [], []
        for doc in docs:
            try:
                # See if content is a base64 string; if yes, treat as image
                b64decode(doc.page_content)
                b64.append(doc.page_content)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}

    def build_prompt(kwargs):
        ctx = kwargs["context"]
        question = kwargs["question"]

        context_text = "\n".join([d.page_content for d in ctx["texts"]])
        prompt_template = f"""
        You are a helpful assistant.
        Use the following context (which may include text, tables, and image descriptions) to answer:
        Context:
        {context_text}

        Question: {question}
        """

        prompt_content = [{"type": "text", "text": prompt_template}]
        for image in ctx["images"]:
            prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})

        template = ChatPromptTemplate.from_messages([{"role": "user", "content": prompt_content}])
        return template.format_messages()

    chain = (
        {"context": retriever | RunnableLambda(parse_docs), "question": RunnablePassthrough()}
        | RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        try:
            answer = chain.invoke(user_input)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        except Exception as e:
            st.error(f"Error running RAG chain: {e}")
else:
    if not json_file:
        st.info("Error in RAG data file.")
    else:
        st.warning("Please enter your API key and choose a provider.", icon="⚠")
