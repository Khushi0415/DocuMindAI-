import streamlit as st
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

st.title("DocuMind AI – Intelligent Document Assistant")

# ===== API KEY =====
api = st.text_input("Enter LLM API Key", type="password")

# ===== FILE =====
file = st.file_uploader("Upload PDF", type="pdf")


# ===== LLM FUNCTION =====
def ask_llm(prompt, key):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "rag-demo"
    }

    data = {
        "model": "openrouter/auto",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        res = requests.post(url, headers=headers, json=data)

        result = res.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]

        elif "error" in result:
            return "API ERROR → " + str(result["error"])

        else:
            return "Unexpected Response → " + str(result)

    except Exception as e:
        return "Request Failed → " + str(e)

# ===== RAG FLOW =====
if api and file:

    # save pdf
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())

    # load
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # split
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(docs)

    # embeddings + vector db
    embed = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embed)

    # question
    q = st.text_input("Ask question")

    if q:
        res = db.similarity_search(q)
        context = res[0].page_content

        prompt = f"""
Answer ONLY from the given context.

Context:
{context}

Question:
{q}
"""

        answer = ask_llm(prompt, api)

        st.subheader("Answer")
        st.write(answer)