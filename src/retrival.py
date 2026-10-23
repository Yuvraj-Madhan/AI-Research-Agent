
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)
from langchain_ollama import OllamaEmbeddings

from src.logger import get_logger

logger = get_logger("ai_research_agent")


def retrieve_documents(
    query: str,
    k: int = 4,
    retrieval_mode: str =None
):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if retrieval_mode == "sparse":
      mode = RetrievalMode.SPARSE
    elif retrieval_mode == "dense":
        mode = RetrievalMode.DENSE
    elif retrieval_mode == "hybrid":
        mode = RetrievalMode.HYBRID
    else:
        raise ValueError(
            "retrieval_mode must be 'sparse', 'dense', or 'hybrid'."
        )

    embedding = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    sparse_embedding = FastEmbedSparse(
        model_name="Qdrant/bm25",
    )

    try:
        vector_store = QdrantVectorStore.from_existing_collection(
        collection_name="ai_agent_research",
        url="http://localhost:6333",
        embedding=embedding,
        sparse_embedding=sparse_embedding,
        retrieval_mode=mode,
    )

        documents = vector_store.similarity_search(
            query=query,
            k=k,
        )

        logger.info(
            "Retrieved %d documents using %s retrieval.",
            len(documents),
            retrieval_mode,
        )
        for i, doc in enumerate(documents, start=1):
            logger.info(
            "Chunk %d | Source: %s | Page: %s",
            i,
            doc.metadata.get("source", "Unknown"),
            doc.metadata.get("page", "Unknown"),
        )

        return documents

    except Exception as e:
        logger.exception("Document retrieval failed.")
        raise RuntimeError(
            f"Error retrieving documents from Qdrant: {e}"
        ) from e
