
import os
import tempfile
import threading
import hashlib
import json

from contextlib import asynccontextmanager
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from config import TOP_K
from ingestion import extract_pdf_pages
from rag_utils import RAGIndex


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "data" / "vectorstore"

DOCUMENT_REGISTRY_PATH = (
    BASE_DIR / "data" / "document_hashes.json"
)

# Shared application state
rag_index = RAGIndex()
rag_pipeline = None

# Maps SHA-256 hashes to original filenames.
document_hashes = {}

# Protect shared index operations.
index_lock = threading.RLock()


# --------------------------------------------------
# DOCUMENT REGISTRY
# --------------------------------------------------

def load_document_registry():
    """
    Load previously recorded document hashes.
    """

    global document_hashes

    if not DOCUMENT_REGISTRY_PATH.exists():
        document_hashes = {}
        return

    try:
        with open(
            DOCUMENT_REGISTRY_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            document_hashes = json.load(file)

        if not isinstance(document_hashes, dict):
            document_hashes = {}

        print(
            f"Loaded document registry: "
            f"{len(document_hashes)} documents"
        )

    except (json.JSONDecodeError, OSError) as exc:
        print(f"Document registry loading failed: {exc}")
        document_hashes = {}


def save_document_registry():
    """
    Persist document hashes safely using atomic replacement.
    """

    DOCUMENT_REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=DOCUMENT_REGISTRY_PATH.parent,
            delete=False
        ) as temp_file:

            json.dump(
                document_hashes,
                temp_file,
                indent=4
            )

            temp_path = temp_file.name

        os.replace(
            temp_path,
            DOCUMENT_REGISTRY_PATH
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def calculate_file_hash(file_content: bytes) -> str:
    """
    Generate a SHA-256 fingerprint for an uploaded PDF.
    """

    return hashlib.sha256(file_content).hexdigest()


# --------------------------------------------------
# LAZY PIPELINE INITIALIZATION
# --------------------------------------------------

def get_pipeline():
    """
    Initialize LangGraph and Gemini only when a query arrives.
    Reuse the same pipeline for subsequent queries.
    """

    global rag_pipeline

    if rag_pipeline is None:
        from rag_pipeline import RAGPipeline

        rag_pipeline = RAGPipeline(
            rag_index,
            top_k=TOP_K
        )

    return rag_pipeline


# --------------------------------------------------
# APPLICATION LIFECYCLE
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the persisted vector store and document registry.
    """

    if INDEX_PATH.exists():

        try:
            rag_index.load(str(INDEX_PATH))

            print(
                f"Loaded vector store: "
                f"{len(rag_index.chunks)} chunks"
            )

        except Exception as exc:

            print(f"Vector store loading failed: {exc}")

            raise RuntimeError(
                "Could not load the existing vector store."
            ) from exc

    else:
        print("No existing vector store found. Starting empty.")

    load_document_registry()

    yield

    print("FastAPI application shutting down.")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="RAG Study Assistant API",
    description=(
        "Backend API for a hybrid RAG study assistant "
        "using FAISS, BM25, reranking, LangGraph and Gemini."
    ),
    version="1.1.0",
    lifespan=lifespan
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class QueryRequest(BaseModel):

    question: str = Field(
        min_length=2,
        max_length=2000
    )


# --------------------------------------------------
# 1. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "RAG Study Assistant API",
        "index_loaded": rag_index._index is not None,
        "total_chunks": len(rag_index.chunks),
        "registered_documents": len(document_hashes)
    }


# --------------------------------------------------
# 2. QUERY ENDPOINT
# --------------------------------------------------

@app.post("/query")
def query_documents(request: QueryRequest):

    if not rag_index.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a PDF first."
        )

    try:

        with index_lock:

            pipeline = get_pipeline()

            result = pipeline.ask(request.question)

        return {
            "question": request.question,
            "answer": result["answer"],
            "validation_status": result["validation_status"],
            "validation_report": result["validation_report"],
            "attempts": result["attempts"],
            "error": result["error"],
            "sources": result["sources"]
        }

    except HTTPException:
        raise

    except Exception as exc:

        error_message = str(exc)
        error_upper = error_message.upper()

        print(f"RAG query error: {error_message}")

        # Gemini quota or rate limit.
        if (
            "RESOURCE_EXHAUSTED" in error_upper
            or "429" in error_upper
            or "QUOTA" in error_upper
        ):

            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exceeded. "
                    "Please retry after the quota resets."
                )
            )

        # Gemini temporary service unavailability.
        elif (
            "UNAVAILABLE" in error_upper
            or "503" in error_upper
        ):

            raise HTTPException(
                status_code=503,
                detail="The AI service is temporarily unavailable."
            )

        # Request timeout.
        elif "TIMEOUT" in error_upper:

            raise HTTPException(
                status_code=504,
                detail="The AI service request timed out."
            )

        # Unexpected internal error.
        else:

            raise HTTPException(
                status_code=500,
                detail=(
                    "An internal error occurred "
                    "while processing the query."
                )
            )


# --------------------------------------------------
# 3. PDF UPLOAD AND INDEXING
# --------------------------------------------------

@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    filename = Path(file.filename).name

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    temp_path = None

    try:

        # Read uploaded PDF content.
        file_content = file.file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty."
            )

        # Calculate SHA-256 fingerprint.
        file_hash = calculate_file_hash(file_content)

        # Check for duplicate documents.
        with index_lock:

            if file_hash in document_hashes:

                original_filename = document_hashes[file_hash]

                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Duplicate PDF detected. "
                        f"This file has already been indexed as "
                        f"'{original_filename}'."
                    )
                )

        # Save PDF temporarily for extraction.
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp_file:

            temp_file.write(file_content)
            temp_path = temp_file.name

        # Extract PDF pages.
        pages = extract_pdf_pages(temp_path)

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in the PDF."
            )

        # Index document and persist changes.
        with index_lock:

            # Recheck inside the lock before modifying the index.
            if file_hash in document_hashes:

                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Duplicate PDF detected. "
                        "This document has already been indexed."
                    )
                )

            chunks_added = rag_index.add_pages(
                pages,
                filename
            )

            if chunks_added == 0:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No usable chunks extracted from the PDF."
                    )
                )

            # Persist FAISS index and metadata.
            INDEX_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            rag_index.save(str(INDEX_PATH))

            # Register the successfully indexed document.
            document_hashes[file_hash] = filename

            save_document_registry()

        return {
            "message": "Document uploaded and indexed successfully.",
            "filename": filename,
            "pages_extracted": len(pages),
            "chunks_added": chunks_added,
            "total_chunks": len(rag_index.chunks),
            "duplicate_check": "passed"
        }

    except HTTPException:
        raise

    except Exception as exc:

        print(f"Document upload error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(exc)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        file.file.close()


# --------------------------------------------------
# 4. RETRIEVAL-ONLY ENDPOINT
# --------------------------------------------------

@app.post("/retrieve")
def retrieve_documents(request: QueryRequest):

    if not rag_index.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a PDF first."
        )

    try:

        with index_lock:

            results = rag_index.query_with_metadata(
                request.question,
                top_k=TOP_K
            )

        return {
            "question": request.question,
            "mode": "retrieval_only",
            "total_results": len(results),
            "sources": results
        }

    except Exception as exc:

        print(f"Retrieval error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Document retrieval failed."
        )


# --------------------------------------------------
# 5. LIST INDEXED DOCUMENTS
# --------------------------------------------------

@app.get("/documents")
def list_documents():

    if not rag_index.chunks:

        return {
            "total_documents": 0,
            "total_chunks": 0,
            "documents": []
        }

    with index_lock:

        source_counts = Counter(rag_index.sources)

        documents = [
            {
                "filename": filename,
                "chunks": count
            }
            for filename, count in source_counts.items()
        ]

    return {
        "total_documents": len(documents),
        "total_chunks": len(rag_index.chunks),
        "documents": documents
    }
