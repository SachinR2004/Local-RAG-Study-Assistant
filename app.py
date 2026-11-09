
import os
import html
import requests
import streamlit as st


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        /* Dashboard metric cards */
        .metric-card {
            background-color: #f5f6f8;
            border: 1px solid #e1e4e8;
            border-radius: 10px;
            padding: 20px;
            min-height: 112px;
        }

        .metric-label {
            color: #606975 !important;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 10px;
        }

        .metric-value {
            color: #17202b !important;
            font-size: 28px;
            font-weight: 650;
            line-height: 1.2;
        }

        .metric-description {
            color: #737d89 !important;
            font-size: 12px;
            margin-top: 7px;
        }

        /* General layout */
        .section-caption {
            color: #737d89 !important;
            font-size: 14px;
        }

        .answer-box {
            background-color: #f5f6f8;
            border: 1px solid #e1e4e8;
            border-radius: 9px;
            padding: 18px;
        }

        .status-label {
            color: #606975 !important;
            font-size: 12px;
        }

        /* Buttons */
        div.stButton > button {
            border-radius: 7px;
            min-height: 38px;
            font-weight: 500;
        }

        div[data-testid="stFormSubmitButton"] button {
            border-radius: 7px;
            min-height: 42px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }

        /* Expanders */
        div[data-testid="stExpander"] {
            border-radius: 8px;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            border-radius: 8px;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# SESSION STATE
# ==================================================

DEFAULTS = {
    "api_url": os.getenv(
        "API_BASE_URL",
        "http://127.0.0.1:8000"
    ),
    "last_result": None,
    "last_mode": None,
    "question_history": [],
    "backend_health": None,
    "document_data": None,
    "upload_summary": None
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# API HELPER
# ==================================================

def api_request(method, endpoint, **kwargs):

    url = (
        st.session_state.api_url.rstrip("/")
        + endpoint
    )

    timeout = kwargs.pop("timeout", 180)

    try:
        response = requests.request(
            method,
            url,
            timeout=timeout,
            **kwargs
        )

        if not response.ok:

            try:
                error_data = response.json()
                detail = error_data.get(
                    "detail",
                    response.text
                )

            except ValueError:
                detail = response.text

            raise RuntimeError(
                f"API error ({response.status_code}): {detail}"
            )

        return response.json()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to FastAPI. "
            "Check that the backend container is running."
        )

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "The request timed out. "
            "Check the backend and try again."
        )


# ==================================================
# DASHBOARD DATA
# ==================================================

def refresh_dashboard():

    health = api_request(
        "GET",
        "/health",
        timeout=15
    )

    documents = api_request(
        "GET",
        "/documents",
        timeout=15
    )

    st.session_state.backend_health = health
    st.session_state.document_data = documents

    return health, documents


# ==================================================
# DASHBOARD METRIC CARD
# ==================================================

def metric_card(label, value, description):

    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_description = html.escape(str(description))

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {safe_label}
            </div>
            <div class="metric-value">
                {safe_value}
            </div>
            <div class="metric-description">
                {safe_description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# QUERY EXECUTION
# ==================================================

def execute_question(question, mode):

    if not question or not question.strip():
        st.warning("Please enter a question.")
        return

    question = question.strip()

    try:

        if mode == "Retrieval Only":
            endpoint = "/retrieve"
            payload = {"question": question}
            spinner_text = "Searching indexed documents..."

        else:
            endpoint = "/query"
            payload = {"question": question}
            spinner_text = (
                "Retrieving evidence and generating "
                "your answer..."
            )

        with st.spinner(spinner_text):

            result = api_request(
                "POST",
                endpoint,
                json=payload,
                timeout=300
            )

        st.session_state.last_result = result
        st.session_state.last_mode = mode

        st.session_state.question_history.insert(
            0,
            {
                "question": question,
                "mode": mode,
                "status": result.get(
                    "validation_status",
                    "retrieved"
                    if mode == "Retrieval Only"
                    else "unknown"
                )
            }
        )

        st.session_state.question_history = (
            st.session_state.question_history[:20]
        )

    except Exception as exc:
        st.error(str(exc))


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("RAG Study Assistant")

    st.caption(
        "Document-grounded AI study support"
    )

    st.divider()

    # ----------------------------------------------
    # BACKEND CONNECTION
    # ----------------------------------------------

    st.subheader("Backend Connection")

    st.text_input(
        "FastAPI Base URL",
        key="api_url"
    )

    if st.button(
        "Check Connection",
        use_container_width=True
    ):

        try:

            health, documents = refresh_dashboard()

            st.success("Backend connected.")

            st.caption(
                f"Indexed chunks: {health.get('total_chunks', 0)}"
            )

        except Exception as exc:
            st.error(str(exc))

    st.divider()

    # ----------------------------------------------
    # DOCUMENT UPLOAD
    # ----------------------------------------------

    st.subheader("Document Management")

    uploaded_files = st.file_uploader(
        "Upload study notes",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF study materials."
    )

    if st.button(
        "Upload and Process",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files
    ):

        successful = 0
        failed = 0
        skipped = 0

        progress = st.progress(0)

        status_placeholder = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):

            try:

                status_placeholder.info(
                    f"Processing {uploaded_file.name}..."
                )

                response = api_request(
                    "POST",
                    "/documents/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    },
                    timeout=300
                )

                successful += 1

                chunks_added = response.get(
                    "chunks_added",
                    0
                )

                st.success(
                    f"{uploaded_file.name}: "
                    f"{chunks_added} chunks indexed."
                )

            except Exception as exc:

                error_message = str(exc)

                if "409" in error_message:
                    skipped += 1

                    st.warning(
                        f"{uploaded_file.name}: "
                        "This document is already indexed."
                    )

                else:
                    failed += 1

                    st.error(
                        f"{uploaded_file.name}: {error_message}"
                    )

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        status_placeholder.empty()

        st.session_state.upload_summary = {
            "successful": successful,
            "failed": failed,
            "skipped": skipped
        }

        try:
            refresh_dashboard()
        except Exception as exc:
            st.warning(
                f"Could not refresh document statistics: {exc}"
            )

        st.session_state.last_result = None

        st.success(
            "Upload operation completed."
        )

        st.write(
            f"Successfully indexed: {successful}"
        )

        st.write(
            f"Already indexed: {skipped}"
        )

        st.write(
            f"Failed: {failed}"
        )

    # ----------------------------------------------
    # DOCUMENT LIBRARY
    # ----------------------------------------------

    st.divider()

    st.subheader("Document Library")

    if st.button(
        "Refresh Documents",
        use_container_width=True
    ):

        try:
            refresh_dashboard()
            st.success("Document list refreshed.")

        except Exception as exc:
            st.error(str(exc))

    documents = st.session_state.document_data

    if documents:

        st.caption(
            f"{documents.get('total_documents', 0)} documents | "
            f"{documents.get('total_chunks', 0)} chunks"
        )

        for document in documents.get("documents", []):

            with st.expander(
                document.get("filename", "Unknown document")
            ):

                st.write(
                    f"Indexed chunks: "
                    f"{document.get('chunks', 0)}"
                )

    elif documents is not None:

        st.info("No documents indexed yet.")

    st.divider()

    st.caption(
        "Frontend: Streamlit\n\n"
        "Backend: FastAPI"
    )


# ==================================================
# MAIN HEADER
# ==================================================

header_col1, header_col2 = st.columns(
    [5, 1]
)

with header_col1:

    st.title("Study Assistant")

    st.markdown(
        """
        <p class="section-caption">
            Search your study materials, retrieve relevant evidence,
            and generate answers grounded in your uploaded documents.
        </p>
        """,
        unsafe_allow_html=True
    )

with header_col2:

    if st.button(
        "Refresh",
        use_container_width=True
    ):

        try:
            refresh_dashboard()
            st.rerun()

        except Exception as exc:
            st.error(str(exc))


st.divider()


# ==================================================
# DASHBOARD METRICS
# ==================================================

try:

    health, documents = refresh_dashboard()

    backend_status = (
        "Connected"
        if health.get("status") == "healthy"
        else "Unavailable"
    )

    document_count = documents.get(
        "total_documents",
        0
    )

    chunk_count = documents.get(
        "total_chunks",
        0
    )

except Exception:

    backend_status = "Disconnected"
    document_count = 0
    chunk_count = 0


metric1, metric2, metric3 = st.columns(3)

with metric1:

    metric_card(
        "Backend Status",
        backend_status,
        "FastAPI service"
    )

with metric2:

    metric_card(
        "Indexed Documents",
        document_count,
        "Available PDF documents"
    )

with metric3:

    metric_card(
        "Indexed Chunks",
        chunk_count,
        "Searchable text segments"
    )


st.divider()


# ==================================================
# MAIN TABS
# ==================================================

ask_tab, history_tab, about_tab = st.tabs(
    [
        "Ask Assistant",
        "Question History",
        "About Project"
    ]
)


# ==================================================
# ASK ASSISTANT
# ==================================================

with ask_tab:

    st.subheader("Ask Your Study Notes")

    st.caption(
        "Choose an answer mode and enter your question."
    )

    # ----------------------------------------------
    # ANSWER MODE
    # ----------------------------------------------

    mode = st.radio(
        "Answer Mode",
        [
            "Gemini 2.5 Flash (Cloud)",
            "Retrieval Only"
        ],
        horizontal=True
    )

    selected_mode = (
        "Retrieval Only"
        if mode == "Retrieval Only"
        else "Gemini"
    )

    if selected_mode == "Retrieval Only":

        st.info(
            "Retrieval Only searches indexed documents "
            "without calling Gemini. It does not consume "
            "Gemini API quota."
        )

    else:

        st.caption(
            "Uses Gemini for answer generation and "
            "evidence validation. Requires available API quota."
        )

    # ----------------------------------------------
    # QUESTION INPUT
    # ----------------------------------------------

    with st.form("question_form"):

        question = st.text_area(
            "Your question",
            placeholder=(
                "Example: Explain the sender and receiver "
                "workflow in Activity 3."
            ),
            height=110,
            max_chars=2000
        )

        submit = st.form_submit_button(
            "Submit Question",
            type="primary",
            use_container_width=True
        )

    if submit:

        if chunk_count == 0:

            st.warning(
                "No indexed documents found. Upload a PDF first."
            )

        else:

            execute_question(
                question,
                selected_mode
            )

    # ----------------------------------------------
    # EXAMPLE QUESTIONS
    # ----------------------------------------------

    st.markdown("**Example Questions**")

    examples = [
        "What is S/MIME?",
        "Compare Activity 2 and Activity 4.",
        "Explain the purpose of X.509 certificates."
    ]

    example_columns = st.columns(3)

    for i, example in enumerate(examples):

        with example_columns[i]:

            if st.button(
                example,
                key=f"example_{i}",
                use_container_width=True
            ):

                if chunk_count == 0:

                    st.warning(
                        "Upload a PDF before asking questions."
                    )

                else:

                    execute_question(
                        example,
                        selected_mode
                    )

    # ----------------------------------------------
    # RESULT DISPLAY
    # ----------------------------------------------

    result = st.session_state.last_result
    result_mode = st.session_state.last_mode

    if result:

        st.divider()

        st.subheader("Results")

        if result_mode == "Retrieval Only":

            st.success("Retrieval completed.")

            st.caption(
                "The following passages were retrieved "
                "from your indexed study materials."
            )

        else:

            st.subheader("Generated Answer")

            st.markdown(
                result.get(
                    "answer",
                    "No answer returned."
                )
            )

            st.divider()

            # --------------------------------------
            # VALIDATION
            # --------------------------------------

            st.subheader("Evidence Validation")

            status = result.get(
                "validation_status",
                "unknown"
            )

            col1, col2 = st.columns(2)

            with col1:

                if status == "passed":
                    st.success(
                        "Evidence validation passed."
                    )

                elif status == "failed":
                    st.error(
                        "Evidence validation failed."
                    )

                else:
                    st.warning(
                        f"Validation status: {status}"
                    )

            with col2:

                st.metric(
                    "Generation Attempts",
                    result.get("attempts", 0)
                )

            with st.expander("Validation Report"):

                report = result.get(
                    "validation_report"
                )

                if report:
                    st.json(report)
                else:
                    st.info(
                        "No validation report available."
                    )

            error = result.get("error")

            if error:
                st.warning(error)

        # ------------------------------------------
        # SOURCES
        # ------------------------------------------

        sources = result.get("sources", [])

        st.divider()

        st.subheader("Retrieved Sources")

        st.caption(
            f"{len(sources)} relevant chunk(s) retrieved."
        )

        if not sources:

            st.info(
                "No relevant source chunks were returned."
            )

        else:

            for i, item in enumerate(sources, start=1):

                source = item.get(
                    "source",
                    "Unknown source"
                )

                page = item.get("page")
                score = item.get("score")
                activity = item.get("activity")
                section = item.get("section")

                with st.expander(
                    f"Source {i}: {source} | "
                    f"Page {page if page is not None else 'N/A'}"
                ):

                    st.markdown("**Retrieved Content**")

                    st.write(
                        item.get(
                            "text",
                            item.get("content", "")
                        )
                    )

                    st.divider()

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.caption("Page")

                        st.write(
                            page if page is not None else "N/A"
                        )

                    with c2:

                        st.caption("Similarity Score")

                        if isinstance(score, (int, float)):
                            st.write(f"{score:.3f}")
                        else:
                            st.write("Unavailable")

                    with c3:

                        st.caption("Activity")

                        st.write(activity or "N/A")

                    st.caption(
                        f"Source: {source} | "
                        f"Section: {section or 'N/A'} | "
                        f"Chunk: {item.get('chunk_index', 'N/A')}"
                    )

        # ------------------------------------------
        # CLEAR RESULT
        # ------------------------------------------

        if st.button("Clear Current Result"):

            st.session_state.last_result = None
            st.session_state.last_mode = None

            st.rerun()


# ==================================================
# QUESTION HISTORY
# ==================================================

with history_tab:

    st.subheader("Question History")

    history = st.session_state.question_history

    if not history:

        st.info(
            "Questions submitted during this session "
            "will appear here."
        )

    else:

        if st.button("Clear History"):

            st.session_state.question_history = []

            st.rerun()

        for i, item in enumerate(history, start=1):

            with st.expander(
                f"{i}. {item['question']}"
            ):

                st.write(
                    f"Mode: {item['mode']}"
                )

                st.write(
                    f"Status: {item['status']}"
                )


# ==================================================
# ABOUT PROJECT
# ==================================================

with about_tab:

    st.subheader("About RAG Study Assistant")

    st.write(
        """
        RAG Study Assistant is a document-grounded
        question-answering application that helps students
        retrieve and understand information from their
        own study materials.
        """
    )

    st.markdown("**Technology Stack**")

    technologies = [
        ("Frontend", "Streamlit"),
        ("Backend", "FastAPI"),
        ("Orchestration", "LangGraph"),
        ("LLM Integration", "LangChain and Gemini"),
        ("Vector Database", "FAISS"),
        ("Keyword Retrieval", "BM25"),
        ("Hybrid Ranking", "Reciprocal Rank Fusion"),
        ("Reranking", "CrossEncoder"),
        ("Document Processing", "PDF text extraction")
    ]

    for category, technology in technologies:

        st.write(
            f"**{category}:** {technology}"
        )

    st.divider()

    st.markdown("**System Workflow**")

    st.code(
        """
PDF Upload
    |
Text Extraction and Chunking
    |
Embedding Generation
    |
FAISS + BM25
    |
Hybrid Retrieval and Reranking
    |
LangGraph Orchestration
    |
Gemini Generation
    |
Evidence Validation
    |
Grounded Answer and Source References
        """,
        language="text"
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "RAG Study Assistant | Streamlit + FastAPI | "
    "Hybrid Retrieval and LangGraph"
)
