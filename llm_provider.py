
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


MODEL_NAME = "gemini-2.5-flash"


def get_llm():
    """
    Initialize the Gemini chat model.

    The API key is loaded from the environment,
    never hardcoded in the source code.
    """

    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. "
            "Add GOOGLE_API_KEY to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.2,
        max_retries=2,
        timeout=60,
    )


def test_llm():
    """Run a simple connection test."""

    llm = get_llm()

    response = llm.invoke(
        "Reply with exactly: Gemini connection successful."
    )

    return response.content
