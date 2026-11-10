import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persisted_db_path = "db/chroma_db"

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

db = Chroma(
    persist_directory=persisted_db_path,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

query= "Which island does SpaceX lease for its launches in the Pacific?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3,
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


"""
Prompt for claude to use it as "LLM as a judge" for the quality of the retrieved documents

prompt:

I'm testing my RAG system, I'll give you the following information:
User Query:
Context:

I want you to tell me if the context is good and has the answer to the user's question in 1-2 lines.

"""