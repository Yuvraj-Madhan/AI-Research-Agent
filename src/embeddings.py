
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)

from src.logger import get_logger

logger = get_logger("ai_research_agent")


def create_embeddings(chunks):
    if not chunks:
        raise ValueError("No chunks provided for embedding.")

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
            retrieval_mode=RetrievalMode.HYBRID,
        )

        vector_store.add_documents(chunks)

        logger.info(
            "Successfully added %d chunks to the existing Qdrant collection.",
            len(chunks),
        )

        return vector_store

    except Exception as e:
        logger.exception("Error storing document chunks in Qdrant.")
        raise RuntimeError(
            f"Error storing data in vector DB: {e}"
        ) from e
