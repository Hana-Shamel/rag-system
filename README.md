# rag-system

A small retrieval-augmented generation (RAG) pipeline. It loads company profiles from `docs/`, splits them into chunks, embeds those chunks with OpenAI, and stores them in a local Chroma database. A second script retrieves the chunks most similar to a question.

## Layout

- `docs/` holds the source text files: Google, Microsoft, Nvidia, SpaceX, and Tesla.
- `ingestion_pipeline.py` loads every `*.txt` file as UTF-8, splits each document into 1000-character chunks with no overlap, embeds the chunks with `text-embedding-3-small`, and writes a cosine-distance Chroma collection to `db/chroma_db`.
- `retrieval_pipeline.py` opens that collection and returns up to 5 chunks whose similarity score is at least 0.3. The question is set in the `query` variable near the top of the file.
- `.env` supplies `OPENAI_API_KEY`. This file is gitignored.

## Setup

Create a virtual environment and install the packages with that environment's Python:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install langchain-community langchain-text-splitters langchain-openai langchain-chroma python-dotenv
```

Add your OpenAI key to `.env`:

```text
OPENAI_API_KEY=your-key-here
```

## Run

Build the vector store first:

```powershell
py ingestion_pipeline.py
```

Then retrieve chunks for the question in `retrieval_pipeline.py`:

```powershell
py retrieval_pipeline.py
```

Change `query` in `retrieval_pipeline.py` to ask something else. Run the ingestion script again after you add or edit files in `docs/`.
