import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore,FastEmbedSparse,RetrievalMode
from  dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from src.logger import get_logger
from src.exception import AgenticException


logger = get_logger("ai_research_agent")
logger.info("Chunking Started")

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents=documents)

    return chunks

logger.info("Chunking done")
