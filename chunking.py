
import re
import unicodedata

from config import CHUNK_SIZE, CHUNK_OVERLAP


# --------------------------------------------------
# 1. Text normalization
# --------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize Unicode whitespace while preserving lines."""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u202f", " ")

    normalized_lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


# --------------------------------------------------
# 2. Sentence-aware splitting
# --------------------------------------------------

def _split_long_sentence(
    sentence: str,
    chunk_size: int
) -> list[str]:
    """Split exceptionally long sentences by word count."""

    words = sentence.split()

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def _split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int
) -> list[str]:
    """Split text at sentence boundaries where possible."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    text = _normalize_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.replace("\n", " ")
    )

    units = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        words = sentence.split()

        if len(words) > chunk_size:
            units.extend(
                _split_long_sentence(sentence, chunk_size)
            )
        else:
            units.append(sentence)

    chunks = []
    current_words = []

    for unit in units:
        unit_words = unit.split()

        if len(current_words) + len(unit_words) <= chunk_size:
            current_words.extend(unit_words)

        else:
            if current_words:
                chunks.append(" ".join(current_words))

            current_words = (
                current_words[-overlap:] if overlap > 0 else []
            )

            if len(current_words) + len(unit_words) > chunk_size:
                if current_words:
                    chunks.append(" ".join(current_words))

                current_words = unit_words[:]

            else:
                current_words.extend(unit_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# --------------------------------------------------
# 3. Backward-compatible text chunking
# --------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Backward-compatible sentence-aware text chunking."""

    return _split_into_chunks(text, chunk_size, overlap)


# --------------------------------------------------
# 4. Page-based chunking
# --------------------------------------------------

def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """Split PDF pages independently."""

    document_chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        page_text = page_data["text"]

        page_chunks = _split_into_chunks(
            page_text,
            chunk_size,
            overlap
        )

        for chunk_index, chunk in enumerate(page_chunks):
            document_chunks.append({
                "text": chunk,
                "page": page_number,
                "chunk_index": chunk_index,
                "section": None,
                "activity": None
            })

    return document_chunks


# --------------------------------------------------
# 5. Activity heading detection
# --------------------------------------------------

def _is_activity_heading(line: str):
    """Detect Activity 1, Activity 2, Activity 3 or Activity 4."""

    pattern = (
        r"^\s*[●•\-–—]?\s*"
        r"Activity\s*([1-4])\s*"
        r"(?:[—–\-:.)]\s*|$)"
    )

    match = re.match(pattern, line, re.IGNORECASE)

    if match:
        return f"Activity {match.group(1)}"

    return None


def _is_activity_summary(line: str) -> bool:
    """
    Identify bullet-point activity summaries.

    These mention activities but should not start
    new activity sections.
    """

    return bool(
        re.match(
            r"^\s*[●•]\s*Activity\s*[1-4]\s*[—–-]",
            line,
            re.IGNORECASE
        )
    )


# --------------------------------------------------
# 6. General heading detection
# --------------------------------------------------

def _is_heading(line: str):
    line = line.strip()

    if not line or len(line) > 100:
        return False

    words = line.split()

    if len(words) > 12:
        return False

    # Numbered headings: 1. Introduction, 2.1 Overview, etc.
    if re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S", line):
        return True

    # Explicit structural headings
    if re.match(
        r"^(chapter|section|unit|module)\s+\w+",
        line,
        re.IGNORECASE
    ):
        return True

    # Multi-word ALL-CAPS headings
    letters = [char for char in line if char.isalpha()]

    if letters and len(letters) >= 3:
        uppercase_ratio = (
            sum(char.isupper() for char in letters) / len(letters)
        )

        if uppercase_ratio > 0.85 and len(words) >= 2:
            return True

    # Allow only recognized single-word section headings.
    # This avoids treating names and sentence beginnings as headings.
    single_word_headings = {
        "introduction",
        "conclusion",
        "overview",
        "background",
        "references",
        "summary",
        "objectives",
        "methodology",
        "results",
        "discussion",
        "appendix",
        "acknowledgements",
        "acknowledgments",
    }

    if len(words) == 1:
        return line.lower().rstrip(":") in single_word_headings

    # Short multi-word Title Case headings
    if 2 <= len(words) <= 8 and not re.search(r"[.!?;:]$", line):
        capitalized_words = sum(
            word[0].isupper()
            for word in words
            if word[0].isalpha()
        )

        if capitalized_words >= max(2, len(words) - 1):
            return True

    return False

# --------------------------------------------------
# 7. Section and activity extraction
# --------------------------------------------------

def _extract_sections(
    text: str,
    current_activity: str = None
) -> list[dict]:
    """
    Group text under headings and detect activity labels.

    Activity labels continue across PDF pages.
    """

    text = _normalize_text(text)

    sections = []

    current_heading = None
    current_paragraphs = []
    active_activity = current_activity

    def save_section():
        if current_paragraphs or current_heading:
            sections.append({
                "heading": current_heading,
                "text": "\n".join(current_paragraphs).strip(),
                "activity": active_activity
            })

    for line in text.splitlines():

        # Ignore summary-list entries as activity transitions.
        if _is_activity_summary(line):
            current_paragraphs.append(line)
            continue

        activity = _is_activity_heading(line)

        if activity:
            save_section()

            active_activity = activity
            current_heading = line
            current_paragraphs = []

        elif _is_heading(line):
            save_section()

            current_heading = line
            current_paragraphs = []

        else:
            current_paragraphs.append(line)

    save_section()

    if not sections and text.strip():
        sections.append({
            "heading": None,
            "text": text,
            "activity": active_activity
        })

    return sections


# --------------------------------------------------
# 8. Section-aware and activity-aware chunking
# --------------------------------------------------

def chunk_pages_by_section(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Create sentence-aware chunks within sections.

    Preserves:
    - PDF page number
    - Section heading
    - Activity label
    - Page-local chunk index

    Activity labels continue across pages until
    another activity heading is detected.
    """

    document_chunks = []

    active_activity = None

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        sections = _extract_sections(
            page_text,
            current_activity=active_activity
        )

        page_chunk_index = 0

        for section in sections:

            heading = section["heading"]
            section_text = section["text"]
            activity = section["activity"]

            if heading:
                section_text = heading + "\n" + section_text

            section_chunks = _split_into_chunks(
                section_text,
                chunk_size,
                overlap
            )

            for chunk in section_chunks:
                document_chunks.append({
                    "text": chunk,
                    "page": page_number,
                    "chunk_index": page_chunk_index,
                    "section": heading,
                    "activity": activity
                })

                page_chunk_index += 1

        # Carry activity into the next page.
        if sections:
            active_activity = sections[-1]["activity"]

    return document_chunks
