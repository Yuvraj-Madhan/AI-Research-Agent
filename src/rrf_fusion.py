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



def rrf_fusion(sparse_result, dense_result, k: int = 60):
    rrf_score = {}
    documents = {}

    for rank, doc in enumerate(sparse_result, start=1):
        doc_id = doc.page_content
        documents[doc_id] = doc
        rrf_score[doc_id] = (
            rrf_score.get(doc_id, 0) + 1 / (rank + k)
        )

    for rank, doc in enumerate(dense_result, start=1):
        doc_id = doc.page_content
        documents[doc_id] = doc
        rrf_score[doc_id] = (
            rrf_score.get(doc_id, 0) + 1 / (rank + k)
        )

    sorted_results = sorted(
        rrf_score.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (documents[doc_id], score)
        for doc_id, score in sorted_results
    ]
