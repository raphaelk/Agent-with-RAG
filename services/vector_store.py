"""
Vector Store Engine.
Supports JSON-backed persistence, cosine similarity search, chunk deduplication,
URL web scraping & local directory recursive ingestion with customizable chunking parameters.
"""
import hashlib
import json
import os
import re
import threading
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from config import (
    DOC_VECTOR_DB_FILE,
    SKILL_VECTOR_DB_FILE,
    DATABASE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_RAG_DOC_SCORE,
    MIN_SKILL_SCORE
)
from services.ollama_service import ollama_service
from services.log_service import audit_logger

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks of specified character length with overlap.
    """
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    step = max(1, chunk_size - overlap)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += step

    return chunks

class VectorStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        self.is_ingesting = False
        self._ensure_db_file()

    def _ensure_db_file(self):
        with self.lock:
            if not self.db_path.exists() or self.db_path.stat().st_size == 0:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.db_path, "w", encoding="utf-8") as f:
                    json.dump({"embedding_model": ollama_service.current_model, "documents": {}, "chunks": []}, f, indent=2)

    def _load(self) -> Dict[str, Any]:
        with self.lock:
            try:
                if self.db_path.exists() and self.db_path.stat().st_size > 0:
                    with open(self.db_path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as e:
                print(f"[VectorStore Load Error] {e}")
            return {"embedding_model": ollama_service.current_model, "documents": {}, "chunks": []}

    def _save(self, data: Dict[str, Any]):
        with self.lock:
            if "embedding_model" not in data:
                data["embedding_model"] = ollama_service.current_model
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def reset(self, embedding_model: Optional[str] = None):
        """Clear all indexed records and record active embedding model."""
        model = embedding_model or ollama_service.current_model
        with self.lock:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump({"embedding_model": model, "documents": {}, "chunks": []}, f, indent=2)

    def delete_document(self, doc_name: str) -> Dict[str, Any]:
        """
        Delete any document from the DB and all its associated chunks.
        Per SPECIFICATION.md: 'Allow the user to delete any document from the DB by using the Delete button on the right side of the document row'
        """
        with self.lock:
            data = self._load()
            docs = data.get("documents", {})
            chunks = data.get("chunks", [])

            if doc_name not in docs and not any(c.get("metadata", {}).get("document_name") == doc_name for c in chunks):
                return {
                    "status": "error",
                    "message": f"Document '{doc_name}' not found in database."
                }

            # Remove doc from documents dict
            docs.pop(doc_name, None)

            # Filter out chunks
            original_count = len(chunks)
            remaining_chunks = [c for c in chunks if c.get("metadata", {}).get("document_name") != doc_name]
            deleted_chunks_count = original_count - len(remaining_chunks)

            data["documents"] = docs
            data["chunks"] = remaining_chunks

            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        audit_logger.log_event(
            event_type="Document Deleted",
            invoker="User",
            target="VectorStore",
            payload={"document_name": doc_name},
            response={"deleted_chunks": deleted_chunks_count, "remaining_chunks": len(remaining_chunks)},
            description=f"Deleted document '{doc_name}' ({deleted_chunks_count} chunks removed)"
        )

        return {
            "status": "success",
            "message": f"Document '{doc_name}' successfully deleted.",
            "deleted_document": doc_name,
            "deleted_chunks": deleted_chunks_count,
            "remaining_chunks": len(remaining_chunks)
        }

    def get_stats(self) -> Dict[str, Any]:
        data = self._load()
        chunks = data.get("chunks", [])
        docs = data.get("documents", {})

        file_size_mb = 0.0
        if self.db_path.exists():
            file_size_mb = round(self.db_path.stat().st_size / (1024 * 1024), 3)

        doc_summaries = []
        for doc_name, meta in docs.items():
            doc_summaries.append({
                "name": doc_name,
                "chunks_count": meta.get("chunks_count", 0),
                "total_characters": meta.get("total_characters", 0),
                "source": meta.get("source", "file")
            })

        return {
            "embedding_model": data.get("embedding_model", ollama_service.current_model),
            "total_chunks": len(chunks),
            "total_documents": len(docs),
            "db_size_mb": file_size_mb,
            "is_ingesting": self.is_ingesting,
            "documents": doc_summaries
        }

    def add_chunk(self, chunk_text_content: str, metadata: Dict[str, Any]) -> bool:
        """
        Add a single chunk to the store if not duplicated (checked by SHA-256 hash).
        """
        content_hash = hashlib.sha256(chunk_text_content.strip().encode("utf-8")).hexdigest()
        data = self._load()

        # Check deduplication
        existing_hashes = {c.get("content_hash") for c in data.get("chunks", [])}
        if content_hash in existing_hashes:
            return False

        # Generate embedding
        vector = ollama_service.generate_embedding(chunk_text_content)

        new_chunk = {
            "id": f"chunk-{len(data.get('chunks', [])) + 1}",
            "content_hash": content_hash,
            "text": chunk_text_content,
            "vector": vector,
            "metadata": metadata
        }

        data["chunks"].append(new_chunk)
        doc_name = metadata.get("document_name", "unknown")
        if doc_name not in data["documents"]:
            data["documents"][doc_name] = {
                "chunks_count": 0,
                "total_characters": 0,
                "source": metadata.get("source", "unknown")
            }

        data["documents"][doc_name]["chunks_count"] += 1
        data["documents"][doc_name]["total_characters"] += len(chunk_text_content)

        self._save(data)
        return True

    def ingest_text_document(
        self,
        doc_name: str,
        text_content: str,
        source: str = "text",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> Tuple[int, int]:
        """
        Chunk and index text content. Returns (added_chunks, total_chars).
        Ensures no duplicate chunks are added.
        """
        chunks = chunk_text(text_content, chunk_size=chunk_size, overlap=overlap)
        added_count = 0
        total_chars = len(text_content)

        for idx, chk in enumerate(chunks):
            meta = {
                "document_name": doc_name,
                "source": source,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            }
            if self.add_chunk(chk, meta):
                added_count += 1

        return added_count, total_chars

    def ingest_url(
        self,
        url: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> Dict[str, Any]:
        """
        Fetch HTML from URL, strip tags, and ingest chunks.
        """
        self.is_ingesting = True
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgentWithRAG/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8", errors="ignore")

            # Basic HTML text extraction
            text = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = ' '.join(text.split())

            doc_name = url.split("//")[-1].replace("/", "_")[:60]
            added_chunks, total_chars = self.ingest_text_document(
                doc_name=doc_name,
                text_content=clean_text,
                source=url,
                chunk_size=chunk_size,
                overlap=overlap
            )

            audit_logger.log_event(
                event_type="URL Ingestion",
                invoker="User",
                target="VectorStore",
                payload={"url": url, "chunk_size": chunk_size, "overlap": overlap},
                response={"doc_name": doc_name, "added_chunks": added_chunks, "characters": total_chars},
                description=f"Ingested URL: {url}"
            )

            return {
                "status": "success",
                "document_name": doc_name,
                "added_chunks": added_chunks,
                "total_characters": total_chars
            }
        except Exception as e:
            audit_logger.log_event(
                event_type="Ingestion Error",
                invoker="User",
                target="VectorStore",
                payload={"url": url},
                response={"error": str(e)},
                description=f"Failed to ingest URL {url}: {e}",
                status="error"
            )
            return {"status": "error", "message": str(e)}
        finally:
            self.is_ingesting = False

    def ingest_local_path(
        self,
        path_str: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> Dict[str, Any]:
        """
        Recursively ingest files from a local directory or file.
        """
        self.is_ingesting = True
        try:
            target_path = Path(path_str).resolve()
            if not target_path.exists():
                return {"status": "error", "message": f"Path '{path_str}' does not exist."}

            files_to_read = []
            if target_path.is_file():
                files_to_read.append(target_path)
            else:
                for ext in ["*.txt", "*.md", "*.csv", "*.json", "*.py", "*.html"]:
                    files_to_read.extend(list(target_path.rglob(ext)))

            total_added = 0
            total_chars = 0
            processed_docs = []

            for file_path in files_to_read:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    if not content.strip():
                        continue
                    doc_name = file_path.name
                    added, chars = self.ingest_text_document(
                        doc_name=doc_name,
                        text_content=content,
                        source=str(file_path),
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                    total_added += added
                    total_chars += chars
                    processed_docs.append(doc_name)
                except Exception as fe:
                    print(f"Failed to read file {file_path}: {fe}")

            audit_logger.log_event(
                event_type="Local Path Ingestion",
                invoker="User",
                target="VectorStore",
                payload={"path": path_str, "files_found": len(files_to_read)},
                response={"added_chunks": total_added, "total_characters": total_chars, "docs": processed_docs},
                description=f"Ingested {len(processed_docs)} local documents"
            )

            return {
                "status": "success",
                "documents_processed": processed_docs,
                "added_chunks": total_added,
                "total_characters": total_chars
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            self.is_ingesting = False

    def query_similar(self, query_text: str, top_k: int = 5, min_score: float = MIN_RAG_DOC_SCORE, conversation_id: Optional[str] = None, invoker: str = "vector database") -> List[Dict[str, Any]]:
        """
        Retrieve chunks exceeding min_score, sorted by cosine similarity descending.
        """
        query_vec = ollama_service.generate_embedding(query_text, conversation_id=conversation_id, invoker=invoker)
        data = self._load()
        chunks = data.get("chunks", [])

        scored_chunks = []
        for c in chunks:
            c_vec = c.get("vector")
            if not c_vec:
                continue
            score = cosine_similarity(query_vec, c_vec)
            if score >= min_score:
                scored_chunks.append({
                    "id": c.get("id"),
                    "score": round(score, 4),
                    "text": c.get("text"),
                    "metadata": c.get("metadata", {})
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

# Global singletons for Documents and Skills
doc_vector_store = VectorStore(DOC_VECTOR_DB_FILE)
skill_vector_store = VectorStore(SKILL_VECTOR_DB_FILE)
