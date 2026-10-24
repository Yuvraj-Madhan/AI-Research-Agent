
from src.query_enhancments import rewritten_query, hyde_query
from src.retrival import retrieve_documents,format_documents_as_context
from src.rrf_fusion import rrf_fusion
from src.reranker import rerank_documents
from src.generate_answer import generate_answer
from src.logger import get_logger

logger = get_logger("ai_research_agent")


def rag_node(state: dict) -> dict:
    user_query = state.get("user_query", "").strip()

    if not user_query:
        return {
            "llm_result": "Please provide a valid question."
        }

    try:
        # 1. Rewrite the original query
        improved_query = rewritten_query(user_query)
        

        # 2. Generate a hypothetical passage for dense retrieval
        hypothetical_passage = hyde_query(improved_query)
        

        # 3. Retrieve sparse candidates
        sparse_results = retrieve_documents(
            query=improved_query,
            k=20,
            retrieval_mode="sparse",
        )

        

        # 4. Retrieve dense candidates using the HyDE passage
        dense_results = retrieve_documents(
            query=hypothetical_passage,
            k=20,
            retrieval_mode="dense",
        )
      

        # 5. Fuse the retrieval results
        fused_results = rrf_fusion(
            sparse_result=sparse_results,
            dense_result=dense_results,
        )[:20]

       

        


        # 6. Rerank against the original question
        final_results = rerank_documents(
            query=user_query,
            fused_results=fused_results,
            top_k=5,
        )

        # 7. Format the reranked documents with source metadata
        documents = [
            result["document"]
            for result in final_results
        ]

        context = format_documents_as_context(documents)

        # 8. Generate an answer grounded in the retrieved context
        answer = generate_answer(
            user_query=user_query,
            context=context,
        )

        logger.info("RAG node completed successfully.")

        return {
            "llm_result": answer,
            "query_category": "rag",
        }



    except Exception:
        logger.exception("RAG node failed.")
        raise
