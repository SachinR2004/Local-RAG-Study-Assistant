
import re
from functools import lru_cache

from transformers import pipeline

from config import GENERATION_MODEL


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MAX_INPUT_TOKENS = 480

REQUIRED_CHUNK_TOKEN_ALLOWANCE = 110
OPTIONAL_CHUNK_TOKEN_ALLOWANCE = 75


# ---------------------------------------------------------
# 1. LOAD GENERATION MODEL
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def _get_generator():
    """Load Flan-T5 once and reuse it."""

    return pipeline(
        "text2text-generation",
        model=GENERATION_MODEL,
        device=-1
    )


# ---------------------------------------------------------
# 2. FORMAT CONTEXT
# ---------------------------------------------------------

def _format_context(context_chunks: list) -> str:

    formatted_sources = []

    for i, chunk in enumerate(context_chunks, start=1):

        if isinstance(chunk, str):
            text = chunk.strip()
            metadata = ""

        elif isinstance(chunk, dict):

            text = chunk.get("text", "").strip()

            if not text:
                continue

            details = []

            if chunk.get("activity"):
                details.append(chunk["activity"])

            if chunk.get("page") is not None:
                details.append(f"Page {chunk['page']}")

            if chunk.get("section"):
                details.append(chunk["section"])

            metadata = " | ".join(details)

        else:
            continue

        if not text:
            continue

        header = f"Source {i}"

        if metadata:
            header += f" ({metadata})"

        formatted_sources.append(
            f"{header}:\n{text}"
        )

    return "\n\n".join(formatted_sources)


# ---------------------------------------------------------
# 3. QUESTION CLASSIFICATION
# ---------------------------------------------------------

def _get_question_type(question: str) -> str:

    q = question.lower()

    if any(
        word in q
        for word in ["compare", "difference", "differentiate"]
    ):
        return "comparison"

    if any(
        word in q
        for word in ["workflow", "steps", "process", "procedure"]
    ):
        return "workflow"

    if any(
        phrase in q
        for phrase in ["what is", "define", "meaning of"]
    ):
        return "definition"

    return "general"


# ---------------------------------------------------------
# 4. EXTRACT ACTIVITY NUMBERS
# ---------------------------------------------------------

def _get_requested_activities(question: str) -> list:

    return list(
        dict.fromkeys(
            re.findall(
                r"\bactivit(?:y|ies)\s*(\d+)\b",
                question.lower()
            )
        )
    )


# ---------------------------------------------------------
# 5. BUILD A SHORT, QUESTION-AWARE PROMPT
# ---------------------------------------------------------

def build_prompt(question: str, context_chunks: list) -> str:

    context = _format_context(context_chunks)

    question_type = _get_question_type(question)

    if question_type == "comparison":

        instruction = """
Compare the requested activities.

Explain the purpose and working of each activity.
Then explain their main differences.
Cover both activities.
"""

    elif question_type == "workflow":

        instruction = """
Explain the actual actions, not just section headings.

Use separate Sender and Receiver headings.
Describe each role in the correct sequence.
Do not repeat steps.
"""

    elif question_type == "definition":

        instruction = """
Give the full form or definition first.
Briefly explain its purpose using the notes.
"""

    else:

        instruction = """
Answer directly using the relevant facts in the notes.
"""

    return f"""
Answer using ONLY the study notes.

Do not invent facts.
Do not copy headings instead of explaining them.
Do not repeat sentences.

{instruction}

NOTES:
{context}

QUESTION:
{question}

ANSWER:
""".strip()


# ---------------------------------------------------------
# 6. TOKEN COUNTING
# ---------------------------------------------------------

def _encode_tokens(text: str, tokenizer):

    backend = getattr(tokenizer, "backend_tokenizer", None)

    if backend is not None:
        return backend.encode(
            text,
            add_special_tokens=True
        ).ids

    return tokenizer.tokenize(text)


def _count_tokens(text: str, tokenizer) -> int:

    return len(_encode_tokens(text, tokenizer))


# ---------------------------------------------------------
# 7. TRUNCATE CHUNKS SAFELY
# ---------------------------------------------------------

def _truncate_chunk(
    chunk,
    tokenizer,
    max_text_tokens: int
):

    if isinstance(chunk, str):

        text = chunk.strip()
        metadata = None

    elif isinstance(chunk, dict):

        text = chunk.get("text", "").strip()
        metadata = chunk.copy()

    else:
        return None

    if not text:
        return None

    encoded = _encode_tokens(text, tokenizer)

    if len(encoded) <= max_text_tokens:

        shortened_text = text

    else:

        shortened_text = tokenizer.decode(
            encoded[:max_text_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

    if not shortened_text:
        return None

    if metadata is None:
        return shortened_text

    metadata["text"] = shortened_text

    return metadata


# ---------------------------------------------------------
# 8. CONTENT-BASED DUPLICATE DETECTION
# ---------------------------------------------------------

def _chunk_key(chunk):
    """
    Identify duplicate chunks by their content and metadata.

    Unlike object IDs, this works even when chunks have
    been copied or truncated into new dictionaries.
    """

    if isinstance(chunk, str):
        return chunk.strip().lower()

    if isinstance(chunk, dict):

        return (
            str(chunk.get("activity", "")).lower().strip(),
            str(chunk.get("page", "")).strip(),
            str(chunk.get("section", "")).lower().strip(),
            re.sub(
                r"\s+",
                " ",
                chunk.get("text", "").lower()
            ).strip()
        )

    return str(chunk)


# ---------------------------------------------------------
# 9. IDENTIFY SENDER / RECEIVER
# ---------------------------------------------------------

def _get_workflow_role(chunk) -> str:

    if isinstance(chunk, str):
        text = chunk.lower()

    elif isinstance(chunk, dict):
        text = chunk.get("text", "").lower()

    else:
        return "other"

    sender = re.search(
        r"(?:^|\n)\s*(?:\d+[.)]\s*)?sender\s*[:—-]",
        text
    )

    receiver = re.search(
        r"(?:^|\n)\s*(?:\d+[.)]\s*)?(?:receiver|recipient)\s*[:—-]",
        text
    )

    if sender and receiver:
        return "both"

    if sender:
        return "sender"

    if receiver:
        return "receiver"

    return "other"


# ---------------------------------------------------------
# 10. COMPARISON CANDIDATES
# ---------------------------------------------------------

def _get_comparison_candidates(
    question: str,
    context_chunks: list
) -> list:

    requested = _get_requested_activities(question)

    if len(requested) < 2:
        return context_chunks

    groups = {}

    for number in requested:

        activity_name = f"activity {number}"

        groups[number] = [
            chunk
            for chunk in context_chunks
            if isinstance(chunk, dict)
            and str(chunk.get("activity", "")).lower().strip()
            == activity_name
        ]

    if not all(groups.values()):
        return context_chunks

    ordered = []

    max_chunks = max(
        len(group)
        for group in groups.values()
    )

    for i in range(max_chunks):

        for number in requested:

            if i < len(groups[number]):
                ordered.append(groups[number][i])

    return ordered


# ---------------------------------------------------------
# 11. WORKFLOW CANDIDATES
# ---------------------------------------------------------

def _get_workflow_candidates(context_chunks: list) -> list:

    sender = []
    receiver = []
    both = []
    other = []

    for chunk in context_chunks:

        role = _get_workflow_role(chunk)

        if role == "sender":
            sender.append(chunk)

        elif role == "receiver":
            receiver.append(chunk)

        elif role == "both":
            both.append(chunk)

        else:
            other.append(chunk)

    return sender + receiver + both + other


# ---------------------------------------------------------
# 12. IDENTIFY REQUIRED CONTEXT
# ---------------------------------------------------------

def _get_required_context(
    question: str,
    context_chunks: list
) -> list:

    question_type = _get_question_type(question)

    # Comparison: reserve one chunk per activity.
    if question_type == "comparison":

        requested = _get_requested_activities(question)

        if len(requested) < 2:
            return []

        required = []

        for number in requested:

            activity_name = f"activity {number}"

            matching = [
                chunk
                for chunk in context_chunks
                if isinstance(chunk, dict)
                and str(chunk.get("activity", "")).lower().strip()
                == activity_name
            ]

            if not matching:
                return []

            required.append(matching[0])

        return required

    # Workflow: reserve sender and receiver.
    if question_type == "workflow":

        sender = []
        receiver = []
        both = []

        for chunk in context_chunks:

            role = _get_workflow_role(chunk)

            if role == "sender":
                sender.append(chunk)

            elif role == "receiver":
                receiver.append(chunk)

            elif role == "both":
                both.append(chunk)

        required = []

        if sender:
            required.append(sender[0])

        if receiver:
            required.append(receiver[0])

        if not required and both:
            required.append(both[0])

        return required

    return []


# ---------------------------------------------------------
# 13. FIT REQUIRED CONTEXT
# ---------------------------------------------------------

def _fit_required_context(
    question: str,
    required_chunks: list,
    tokenizer,
    max_tokens: int
) -> list:

    if not required_chunks:
        return []

    allowance = REQUIRED_CHUNK_TOKEN_ALLOWANCE

    while allowance >= 30:

        compacted = []

        for chunk in required_chunks:

            shortened = _truncate_chunk(
                chunk,
                tokenizer,
                allowance
            )

            if shortened is not None:
                compacted.append(shortened)

        prompt = build_prompt(
            question,
            compacted
        )

        if (
            len(compacted) == len(required_chunks)
            and _count_tokens(prompt, tokenizer) <= max_tokens
        ):
            return compacted

        allowance -= 10

    return []


# ---------------------------------------------------------
# 14. SELECT CONTEXT WITHIN TOKEN BUDGET
# ---------------------------------------------------------

def _select_context_within_budget(
    question: str,
    context_chunks: list,
    tokenizer,
    max_tokens: int = MAX_INPUT_TOKENS
) -> list:

    if not context_chunks:
        return []

    question_type = _get_question_type(question)

    # First reserve the required chunks.
    required = _get_required_context(
        question,
        context_chunks
    )

    selected = _fit_required_context(
        question,
        required,
        tokenizer,
        max_tokens
    )

    if required and not selected:
        return []

    # Arrange optional candidates.
    if question_type == "comparison":

        candidates = _get_comparison_candidates(
            question,
            context_chunks
        )

    elif question_type == "workflow":

        candidates = _get_workflow_candidates(
            context_chunks
        )

    else:
        candidates = context_chunks

    # Content-based deduplication.
    selected_keys = {
        _chunk_key(chunk)
        for chunk in selected
    }

    # Add additional context only if it fits.
    for chunk in candidates:

        key = _chunk_key(chunk)

        if key in selected_keys:
            continue

        trial = selected + [chunk]

        if _count_tokens(
            build_prompt(question, trial),
            tokenizer
        ) <= max_tokens:

            selected.append(chunk)
            selected_keys.add(key)
            continue

        # Try a shortened version.
        compacted = _truncate_chunk(
            chunk,
            tokenizer,
            OPTIONAL_CHUNK_TOKEN_ALLOWANCE
        )

        if compacted is None:
            continue

        compacted_key = _chunk_key(compacted)

        if compacted_key in selected_keys:
            continue

        trial = selected + [compacted]

        if _count_tokens(
            build_prompt(question, trial),
            tokenizer
        ) <= max_tokens:

            selected.append(compacted)
            selected_keys.add(compacted_key)

    # Final token-budget validation.
    final_prompt = build_prompt(
        question,
        selected
    )

    if _count_tokens(final_prompt, tokenizer) > max_tokens:
        return []

    return selected


# ---------------------------------------------------------
# 15. GENERATE ANSWER
# ---------------------------------------------------------

def generate_answer(
    question: str,
    context_chunks: list
) -> str:

    if not isinstance(question, str) or not question.strip():
        return "Please enter a valid question."

    if not isinstance(context_chunks, list) or not context_chunks:
        return "I couldn't find this in your notes."

    if not _format_context(context_chunks):
        return "I couldn't find this in your notes."

    generator = _get_generator()
    tokenizer = generator.tokenizer

    effective_budget = min(
        MAX_INPUT_TOKENS,
        tokenizer.model_max_length - 16
    )

    selected_chunks = _select_context_within_budget(
        question,
        context_chunks,
        tokenizer,
        max_tokens=effective_budget
    )

    if not selected_chunks:

        return (
            "I couldn't find enough relevant information "
            "in your notes to answer this question."
        )

    prompt = build_prompt(
        question,
        selected_chunks
    )

    prompt_tokens = _count_tokens(
        prompt,
        tokenizer
    )

    if prompt_tokens > tokenizer.model_max_length:

        raise ValueError(
            f"Prompt has {prompt_tokens} tokens, exceeding "
            f"the model limit of {tokenizer.model_max_length}."
        )

    output = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        truncation=False
    )

    answer = output[0]["generated_text"].strip()

    if not answer:
        return "I couldn't generate an answer from the available notes."

    return answer
