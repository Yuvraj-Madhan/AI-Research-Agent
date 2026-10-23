

from langchain_ollama import ChatOllama
from src.logger import get_logger

logger = get_logger("ai_research_agent")

Client = ChatOllama(
    model="gpt-oss:20b",
    temperature=0
)


def rewritten_query(user_query: str) -> str:
    """Rewrite a query for clearer keyword and semantic retrieval."""

    if not user_query or not user_query.strip():
        raise ValueError("User query cannot be empty.")

    system_prompt = """
    You are a query rewriting assistant for a RAG system.

    Rewrite the user's query to make document retrieval more effective.

    Rules:
    1. Preserve the original meaning and intent.
    2. Do not answer the question.
    3. Do not introduce new facts.
    4. Expand abbreviations only when their meaning is obvious.
    5. Preserve important keywords from the original query.
    6. Make the query suitable for semantic and keyword search.
    7. Return only the rewritten query.
    """

    try:
        response = Client.invoke([

            {"role":"system", "content":system_prompt},
            {"role" :"user","content" :user_query.strip()},
        ])

        return response.content.strip()

    except Exception:
        logger.exception("Query rewriting failed.")
        raise


def hyde_query(user_query: str) -> str:
    """Generate a hypothetical passage for semantic retrieval."""

    if not user_query or not user_query.strip():
        raise ValueError("User query cannot be empty.")

    system_prompt = """
    You are a document-generation assistant for a RAG system.

    Given a user query, generate a short hypothetical passage
    that could realistically appear in a document containing
    information relevant to the query.

    Rules:
    1. Write in the style of an informative document.
    2. Include terminology relevant to the query.
    3. Keep the passage concise.
    4. Do not add citations or page numbers.
    5. Do not answer outside the scope of the query.
    6. Return only the passage.
    """

    try:
        response = Client.invoke([
            {"role":"system", "content":system_prompt},
            {"role" :"user","content" :user_query.strip()},
        ])

        return response.content.strip()

    except Exception:
        logger.exception("HyDE generation failed.")
        raise
