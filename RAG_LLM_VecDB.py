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
from langchain.schema import Document
import traceback
from operator import itemgetter
import torch
import os

os.environ["PYTORCH_ENABLE_META_TENSOR"] = "0"

st.title("🧪 Test Chat with Thesis Document")

# Sidebar: Choose provider & keys
provider = st.sidebar.selectbox(
    "Choose LLM Provider:",
    ("OpenAI", "Together", "Groq", "Hugging Face", "Anthropic", "Perplexity")
)
api_key = st.sidebar.text_input(f"{provider} API Key", type="password")
model_name = st.sidebar.text_input("Model name (optional)", "")

# Load prebuilt chroma DB path (you must download it from GitHub locally)
PERSIST_DIRECTORY = "./chroma_db"

model = None

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

if model:
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load vectorstore from disk instead of recreating it
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

    # Cleaner parse_docs with expander
    def parse_docs(docs):
        return {"texts": docs}

    # Replace retriever with a RunnableLambda that does similarity_search
    def run_similarity_search(query):
        # k=5 to get top 5 docs, adjust as needed
        results = vectorstore.similarity_search(query, k=5)
        return results

    # Build prompt with expander
    def build_prompt(kwargs):
        ctx = kwargs["context"]
        question = kwargs["question"]
        context_text = "\n".join([d.page_content for d in ctx["texts"]])
        prompt_template = f"""
            You are a helpful assistant.
            Use the following context (which may include text, summary of tables, and image descriptions) to answer:
            Context:
            {context_text}

            Question: {question}
            """

        return ChatPromptTemplate.from_messages(
            [{"role": "user", "content": prompt_template}]
        ).format_messages()

    # Compose chain using RunnableLambda for similarity_search + parse_docs
    chain = (
        {
            "context": itemgetter("question") | RunnableLambda(run_similarity_search) | RunnableLambda(parse_docs),
            "question": itemgetter("question")
        }
        | RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )

    # Streamlit chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        try:
            answer = chain.invoke({"question": user_input})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        except Exception as e:
            st.error(f"Error running RAG chain: {e}")
            st.error(traceback.format_exc())

else:
    st.warning("Please enter your API key and choose a provider.", icon="⚠")
