"""
Tests for Vector Store and Cosine Similarity.
"""
import pytest
from pathlib import Path
from services.vector_store import cosine_similarity, chunk_text, VectorStore
import config

def test_cosine_similarity():
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    vec3 = [0.0, 1.0, 0.0]

    assert pytest.approx(cosine_similarity(vec1, vec2), 0.001) == 1.0
    assert pytest.approx(cosine_similarity(vec1, vec3), 0.001) == 0.0

def test_chunk_text():
    sample = "A" * 1200
    chunks = chunk_text(sample, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert len(chunks[0]) == 500

def test_vector_store_deduplication(tmp_path):
    test_db_file = tmp_path / "test_vectors.json"
    store = VectorStore(test_db_file)

    meta = {"document_name": "test.txt", "source": "test"}
    added_first = store.add_chunk("This is a unique chunk of information.", meta)
    added_duplicate = store.add_chunk("This is a unique chunk of information.", meta)

    assert added_first is True
    assert added_duplicate is False

    stats = store.get_stats()
    assert stats["total_chunks"] == 1
    assert stats["total_documents"] == 1

def test_vector_store_reset(tmp_path):
    test_db_file = tmp_path / "test_vectors.json"
    store = VectorStore(test_db_file)
    store.add_chunk("Chunk 1", {"document_name": "doc1"})
    assert store.get_stats()["total_chunks"] == 1

    store.reset()
    assert store.get_stats()["total_chunks"] == 0
