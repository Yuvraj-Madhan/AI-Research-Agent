
from sentence_transformers import CrossEncoder

from src.logger import get_logger

logger = get_logger("ai_research_agent")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(query: str, fused_results, top_k: int = 5):
    if not fused_results:
        return []

    pairs = [
        (query, doc.page_content)
        for doc, rrf_score in fused_results
    ]

    scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(scores, fused_results),
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        {
            "document": doc,
            "rrf_score": float(rrf_score),
            "reranker_score": float(score),
        }
        for score, (doc, rrf_score) in reranked_results[:top_k]
    ]
