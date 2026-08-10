"""
generate.py

Stretch goal: turn retrieved chunks into a synthesized answer instead of
just showing raw chunks. Uses a small local flan-t5 model via
transformers, so it needs no API key.
"""

from functools import lru_cache
from transformers import pipeline


@lru_cache(maxsize=1)
def _get_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Generate a short answer to `question`, grounded only in the given chunks."""
    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say "
        "\"I couldn't find this in your notes.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    generator = _get_generator()
    output = generator(prompt, max_new_tokens=150, do_sample=False)
    return output[0]["generated_text"].strip()