
from pathlib import Path

from src.ingestion import load_pdf
from src.embeddings import create_embeddings
from src.chunking import split_documents

from src.logger import get_logger

logger = get_logger("ai_research_agent")


def main():
    pdf_path = (
        Path(__file__).resolve().parent
        / "data"
        / "Ace the Data Science Interview (Nick Singh  Kevin Huo).pdf"
    )

    logger.info("Starting PDF ingestion: %s", pdf_path)

    # 1. Load the PDF
    documents = load_pdf(str(pdf_path))
    print(f"Pages loaded: {len(documents)}")

    # 2. Split pages into chunks
    chunks = split_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    # 3. Embed chunks and store them in Qdrant
    create_embeddings(chunks)

    print("Ingestion completed successfully.")
    print("Qdrant collection: ai_agent_research")


if __name__ == "__main__":
    main()
