
# Local RAG Study Assistant

A document-grounded Retrieval-Augmented Generation (RAG) application that allows users to upload PDF study materials and ask questions through a conversational interface.

The system combines semantic and keyword-based retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and LLM-based answer generation to produce context-aware responses with source attribution.

It uses LangGraph to orchestrate retrieval, generation, answer validation, and conditional retries, with FastAPI providing the backend and Streamlit serving as the frontend.

## Architecture

```text
                 User
                  |
             Streamlit UI
                  |
               FastAPI
                  |
             LangGraph
                  |
          Query Processing
                  |
          Hybrid Retrieval
           /           \
        FAISS          BM25
           \           /
          Reciprocal Rank
             Fusion
                  |
          Cross-Encoder
            Reranking
                  |
          Context Assembly
                  |
          Gemini 2.5 Flash
                  |
          Answer Generation
                  |
          Answer Validation
             /       \
          Pass       Retry
             \       /
          Grounded Answer
                  |
          Source Attribution
                  |
                User
```

## Key Features

- **PDF Ingestion:** Extracts text and page-level information from uploaded PDF documents.
- **Semantic Retrieval:** Uses Sentence Transformers embeddings and FAISS vector search.
- **Keyword Retrieval:** Uses BM25 to retrieve relevant documents based on lexical matching.
- **Hybrid Retrieval:** Combines semantic and keyword rankings using Reciprocal Rank Fusion.
- **Cross-Encoder Reranking:** Reorders retrieved passages based on query-document relevance.
- **LLM Integration:** Uses Gemini 2.5 Flash for context-grounded answer generation.
- **LangGraph Orchestration:** Implements a stateful workflow for retrieval, generation, validation, and conditional retries.
- **Answer Validation:** Evaluates generated answers for unsupported claims and missing information.
- **Source Attribution:** Returns document and page metadata alongside generated answers.
- **FastAPI Backend:** Exposes endpoints for querying, retrieval, health checks, document uploads, and document listing.
- **Streamlit Frontend:** Provides a user interface for asking questions, uploading PDFs, viewing results, and tracking question history.
- **Duplicate Upload Protection:** Uses SHA-256 document hashing to detect repeated PDF uploads.
- **Docker Support:** Provides containerized backend and frontend services using Docker Compose.

## Tech Stack

| Component | Technologies |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| RAG Orchestration | LangGraph |
| LLM Integration | LangChain, Gemini 2.5 Flash |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| Keyword Retrieval | BM25 |
| Ranking | Reciprocal Rank Fusion, Cross-Encoder |
| PDF Processing | PyPDF |
| Machine Learning | PyTorch |
| Evaluation | Custom retrieval and answer evaluation scripts |
| Containerization | Docker, Docker Compose |

## Project Structure

```text
Local-RAG-Study-Assistant/
│
├── app.py
├── api.py
├── config.py
├── ingestion.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── retrieval.py
├── reranker.py
├── rag_utils.py
├── rag_pipeline.py
├── llm_provider.py
├── generate.py
│
├── evaluation/
│   ├── questions.json
│   ├── evaluate.py
│   ├── answer_evaluate.py
│   ├── results.csv
│   └── answer_results.csv
│
├── data/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Docker Desktop (optional)
- Gemini API key

### 1. Clone the Repository

```bash
git clone https://github.com/SachinR2004/Local-RAG-Study-Assistant.git

cd Local-RAG-Study-Assistant
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv

source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

You can use `.env.example` as a reference.

### 5. Start the FastAPI Backend

```bash
uvicorn api:app --reload
```

Backend API:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the Streamlit Frontend

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run app.py
```

Frontend:

```text
http://localhost:8501
```

## Running with Docker

Make sure Docker Desktop is running.

Configure the `.env` file with your Gemini API key, then execute:

```bash
docker compose up --build -d
```

Access the application:

| Service | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |

To stop the services:

```bash
docker compose down
```

## Evaluation

The project includes a custom evaluation framework for measuring retrieval relevance and generated-answer grounding.

### Retrieval Evaluation

Initial evaluation on a six-question dataset produced:

| Metric | Result |
|---|---:|
| Mean Precision@3 | 0.944 |
| Mean Recall@3 | 1.000 |
| Mean Reciprocal Rank (MRR) | 1.000 |
| Mean Page Coverage | 1.000* |

*Page coverage was calculated over the three questions with applicable page-level ground truth.*

### Answer Evaluation

Generated answers are evaluated for:

- Context support
- Unsupported claims
- Missing information
- Overall answer validity

The initial evaluation had four successful answer validations, all passing. Two additional questions could not be evaluated because of API quota and timeout errors.

**Note:** These are preliminary results from a small, project-specific dataset, not results from a standardized benchmark.

## Limitations

- Supports digitally generated or typed PDFs; scanned documents and handwritten notes are not currently supported through OCR.
- Answer generation depends on Gemini API availability and quota limits.
- Answer validation uses an LLM-based evaluator and should not be interpreted as independent proof of factual correctness.
- Evaluation currently uses a small, manually curated question dataset.
- First-time model downloads and initialization can increase startup latency.

## Future Improvements

- Expand the evaluation dataset and introduce more comprehensive retrieval metrics.
- Add OCR support for scanned documents.
- Explore alternative LLM providers and fallback strategies.
- Improve retrieval evaluation using larger and more diverse document collections.
- Add persistent cloud storage and public deployment.

## License

This project is intended for educational and research purposes.
