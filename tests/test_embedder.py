"""Tests for the FinancialEmbedder."""

import numpy as np
import pytest
from embeddings.embedder import FinancialEmbedder


@pytest.fixture(scope="module")
def embedder():
    return FinancialEmbedder()


def test_single_embedding_shape(embedder):
    emb = embedder.generate_embedding("Apple reported strong earnings.")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (384,), f"Expected (384,), got {emb.shape}"


def test_batch_embedding_shape(embedder):
    texts = ["Revenue grew 10%.", "Tesla delivered 500k vehicles."]
    embs = embedder.generate_embeddings_batch(texts)
    assert embs.shape == (2, 384)


def test_embeddings_are_normalized(embedder):
    emb = embedder.generate_embedding("Test normalization.")
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 0.01, f"Embedding norm {norm} is not ~1.0"


def test_embed_document_chunks(embedder):
    chunks = [
        {"text": "Revenue was $10 billion.", "company": "ACME"},
        {"text": "Net income rose 5%.", "company": "ACME"},
    ]
    result = embedder.embed_document_chunks(chunks)
    assert len(result) == 2
    assert "embedding" in result[0]
    assert len(result[0]["embedding"]) == 384


def test_similar_texts_closer_than_dissimilar(embedder):
    emb_a = embedder.generate_embedding("Apple's quarterly revenue increased.")
    emb_b = embedder.generate_embedding("Apple reported higher sales this quarter.")
    emb_c = embedder.generate_embedding("The weather today is sunny and warm.")
    sim_ab = float(np.dot(emb_a, emb_b))
    sim_ac = float(np.dot(emb_a, emb_c))
    assert sim_ab > sim_ac, "Semantically similar texts should have higher similarity"
