import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(doc_path="docs"):
    """Load documents from the docs directory"""
    print(f"Loading documents from {doc_path}...")

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Directory {doc_path} does not exist")

    loader = DirectoryLoader(
        path=doc_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {doc_path}")

    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}:")
        print(f" metadata: {doc.metadata}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into chunks with overlap"""
    print(f"Splitting {len(documents)} documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 100)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks)-5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB Vector Store"""

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=os.getenv("OPENAI_API_KEY"),
        )

    print("--- Creating ChromaDB Vector Store ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"Vector store created and stored in {persist_directory}")

    return vector_store


def main():
    print("Main Function")

    # Load files
    documents = load_documents(doc_path="docs")

    # Chunking files
    chunks = split_documents(documents)

    # Embedding and storing in Vector DB
    create_vector_store(chunks)


if __name__ == "__main__":
    main()