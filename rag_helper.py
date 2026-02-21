from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def create_vectorstore():
    # Load documents
    leave_loader = TextLoader("documents/leave_policy.txt")
    onboarding_loader = TextLoader("documents/onboarding.txt")

    docs = leave_loader.load() + onboarding_loader.load()

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="db"
    )

    vectorstore.persist()
    print("Vector DB created ✅")


if __name__ == "__main__":
    create_vectorstore()