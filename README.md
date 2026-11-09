# Local RAG Study Assistant

An offline, privacy-first Retrieval-Augmented Generation (RAG) system built with Python, Streamlit, FAISS, and Hugging Face transformers. It allows users to upload typed lecture notes (PDFs) and perform semantic queries to receive grounded answers with exact source attribution without relying on external cloud APIs or paid keys.

---

```

------

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Embeddings & Vector Search:** `sentence-transformers`, `faiss-cpu`
- **Text Extraction:** `pypdf`
- **Generative Model:** `transformers` (`google/flan-t5-base`), `torch`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- macOS / Linux / Windows

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/SachinR2004/Local-RAG-Study-Assistant.git](https://github.com/SachinR2004/Local-RAG-Study-Assistant.git)
   cd Local-RAG-Study-Assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## 📌 Limitations & Future Work

- **Typed PDFs Only:** Currently supports digitally generated or typed text PDFs. Scanned image PDFs and handwritten notes are excluded to prevent degraded OCR text from corrupting downstream vector quality.
- **Future Extension:** Integrating Handwritten Text Recognition (HTR) models (e.g., Hugging Face TrOCR) to process handwritten lecture material.
