import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS  # ✅ Use FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama  # 🧠 Use Ollama LLM

# Load documents
documents = []
folder_path = "documents"
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(folder_path, file))
        documents.extend(loader.load())

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# HuggingFace Embeddings (offline)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Use FAISS instead of Chroma
vectorstore = FAISS.from_documents(chunks, embedding)

# 🧠 Use Ollama's LLaMA3 model
llm = ChatOllama(model="llama3")  # Make sure `ollama run llama3` is running in background

# Retrieval-based QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Streamlit UI
st.title("🤖 InfoFlow AI Chatbot (Understands Your Company Environment)")
question = st.text_input("Ask your company question:")

if question:
    answer = qa.run(question)
    st.success(answer)
