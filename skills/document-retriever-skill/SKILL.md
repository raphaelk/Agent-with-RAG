---
name: Document Vector Database Retriever Skill
description: Get the list of text chunks from the document vector database. The Agent must use the search_documents function in skills/document-retriever-skill/tools/document_search_tool.py to retrieve chunks of documents from the document vector database about Agent and RAG technology, marketing strategy, and financial reports.
Trigger Queries:
  - What does our company marketing strategy say about enterprise acquisition?
  - Retrieve details from the annual financial report about revenue and operating expenses
  - Explain the architecture of Agent and RAG technology from our documents
  - Search internal documents for financial projections and ARR growth
  - What are our customer retention metrics according to the marketing documentation?
---

# Document Vector Database Retriever Skill

## Overview
This skill performs semantic vector similarity search against the document vector database (`database/doc_vectors.json`). The Agent must use the `search_documents` function located in `skills/document-retriever-skill/tools/document_search_tool.py` to retrieve text chunks whose cosine similarity score exceeds `MIN_RAG_DOC_SCORE` (default: 0.3), supplying verifiable context evidence to ground LLM reasoning.

## Procedural Tool
- **File**: `skills/document-retriever-skill/tools/document_search_tool.py`
- **Function**: `search_documents(query: str, top_k: int = 5, min_score: float = 0.3, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]`
- **Description**: Invokes document vector store similarity search and returns structured text chunks with content, score, document title, and chunk index.

## Standard Operating Procedure (SOP)
1. **Analyze User Inquiry**: Formulate dense semantic search representation from the user's question.
2. **Execute Vector Search via Tool**:
   - Call `search_documents` in `skills/document-retriever-skill/tools/document_search_tool.py` with `query`, `top_k` (from RAG max chunks), and `min_score` (from document threshold).
   - Filter chunks where similarity score >= threshold.
3. **Assemble Evidence**:
   - Package matched chunks with source document title, chunk index, character length, and similarity score.
   - Ground the final answer directly in this retrieved evidence.

