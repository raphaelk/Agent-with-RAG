"""
Document Search Tool.
Queries the document vector database for text chunks matching a query.
"""
from typing import Any, Dict, List, Optional
from config import MIN_RAG_DOC_SCORE
from services.vector_store import doc_vector_store
from services.log_service import audit_logger


def search_documents(
    query: str,
    top_k: int = 5,
    min_score: float = MIN_RAG_DOC_SCORE,
    conversation_id: Optional[str] = None,
    invoker: str = "tool"
) -> List[Dict[str, Any]]:
    """
    Get the list of text chunks from the document vector database matching the query.

    Args:
        query: The search query or question.
        top_k: Maximum number of text chunks to retrieve (default: 5).
        min_score: Minimum cosine similarity score threshold (default: 0.3).
        conversation_id: Optional conversation ID for audit logging.
        invoker: Component invoking the search (default: 'tool').

    Returns:
        List of matching document text chunks with content, scores, and metadata.
    """
    audit_logger.log_call(
        event_type="document search",
        call_type="invocation",
        invoker=invoker,
        recipient="document search",
        payload={"query": query, "top_k": top_k, "min_score": min_score},
        description=f"Document search invoked for query: '{query}' by {invoker}",
        conversation_id=conversation_id
    )

    results = doc_vector_store.query_similar(
        query_text=query,
        top_k=top_k,
        min_score=min_score,
        conversation_id=conversation_id,
        invoker="document search"
    )

    formatted = []
    for r in results:
        meta = r.get("metadata", {})
        formatted.append({
            "text": r.get("text", ""),
            "score": r.get("score", 0.0),
            "document_name": meta.get("document_name", "Unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "source": meta.get("source", "")
        })

    audit_logger.log_call(
        event_type="document search",
        call_type="response",
        invoker="document search",
        recipient=invoker,
        payload={"count": len(formatted), "results": formatted},
        description=f"Document search retrieved {len(formatted)} chunks",
        conversation_id=conversation_id
    )

    return formatted


# Alias for flexible tool invocation
get_document_chunks = search_documents
