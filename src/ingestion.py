
from pathlib import Path
from src.chunking import split_documents
from src.embeddings import create_embeddings

from langchain_community.document_loaders import PyPDFLoader
from src.logger import get_logger

logger = get_logger("ai_research_agent")


def load_pdf(pdf_path: str):
    try:
        file_path = Path(pdf_path)

        # Check whether the PDF exists
        if not file_path.is_file():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError("The provided file must be a PDF.")

        logger.info("Loading PDF: %s", file_path)

        # Extract pages from the PDF
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        if not documents:
            raise ValueError("No pages were extracted from the PDF.")

        logger.info("Successfully loaded %d pages", len(documents))

        return documents

    except Exception:
        logger.exception("PDF loading failed")
        raise


from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

print(client.get_collections())


def ingest_pdf(pdf_path: str) -> dict:
    documents = load_pdf(pdf_path)

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were generated from the PDF.")

    create_embeddings(chunks)

    return {
        "pages": len(documents),
        "chunks": len(chunks),
    }