import streamlit as st
import tempfile
import os
from rag_utils import extract_text_from_pdf, chunk_text, RAGIndex
from generate import generate_answer

st.set_page_config(page_title="Local RAG Study Assistant", layout="wide")

st.title(" Local RAG Study Assistant")
st.caption("Upload typed note PDFs and query them semantically without external APIs.")

# Initialize or retrieve index from session state
if "rag_index" not in st.session_state:
    st.session_state.rag_index = RAGIndex()

# Sidebar for PDF Uploads
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload Typed Notes (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Extracting text and indexing..."):
            total_chunks = 0
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily to process with pypdf
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                text = extract_text_from_pdf(tmp_path)
                os.remove(tmp_path)

                if not text.strip():
                    st.warning(f"No extractable text found in '{uploaded_file.name}'. (Scanned/Handwritten PDFs are not supported).")
                    continue

                chunks = chunk_text(text)
                sources = [uploaded_file.name] * len(chunks)
                st.session_state.rag_index.add(chunks, sources)
                total_chunks += len(chunks)
            
            st.success(f"Successfully processed! Total chunks indexed: {total_chunks}")

# Main Query Interface
query = st.text_input("Ask a question about your uploaded notes:")
use_generation = st.checkbox("Generate synthesized answer using Flan-T5 (Local LLM)")

if query:
    index = st.session_state.rag_index
    if index._index is None or index._index.ntotal == 0:
        st.error("Please upload and process at least one typed PDF document first.")
    else:
        results = index.query(query, top_k=3)
        
        if not results:
            st.warning("I couldn't find this in your notes.")
        else:
            if use_generation:
                with st.spinner("Generating synthesized answer..."):
                    context_chunks = [c[0] for c in results]
                    answer = generate_answer(query, context_chunks)
                    st.subheader("Synthesized Answer")
                    st.info(answer)

            st.subheader("Retrieved Context Chunks")
            for chunk, source, score in results:
                with st.expander(f"Source: {source} (Similarity Score: {score:.3f})"):
                    st.write(chunk)