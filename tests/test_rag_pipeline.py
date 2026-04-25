"""Tests for the FinancialRAGPipeline."""

import pytest
from retrieval.rag_pipeline import FinancialRAGPipeline


SAMPLE_CHUNKS = [
    {
        "text": "Apple reported Q4 2023 revenue of $94.8 billion.",
        "company": "Apple",
        "quarter": "Q4 2023",
        "source": "earnings_call",
    },
    {
        "text": "Tesla delivered 484,507 vehicles in Q3 2023.",
        "company": "Tesla",
        "quarter": "Q3 2023",
        "source": "earnings_call",
    },
    {
        "text": "Microsoft Azure revenue grew 29% year-over-year.",
        "company": "Microsoft",
        "quarter": "Q3 2023",
        "source": "earnings_call",
    },
]


@pytest.fixture(scope="module")
def rag():
    pipeline = FinancialRAGPipeline()
    embedded = pipeline.embedder.embed_document_chunks(SAMPLE_CHUNKS)
    pipeline.build_index(embedded)
    return pipeline


def test_index_built(rag):
    assert rag.index is not None
    assert rag.index.ntotal == len(SAMPLE_CHUNKS)


def test_retrieve_returns_results(rag):
    results = rag.retrieve_relevant_chunks("Apple revenue", top_k=2)
    assert len(results) == 2
    assert "similarity_score" in results[0]


def test_top_result_is_correct_company(rag):
    results = rag.retrieve_relevant_chunks("How many vehicles did Tesla deliver?", top_k=1)
    assert results[0]["company"] == "Tesla"


def test_query_returns_answer(rag):
    response = rag.query("What was Apple's revenue?", top_k=2)
    assert "answer" in response
    assert "sources" in response
    assert len(response["answer"]) > 0


def test_query_with_no_index():
    """Query on an empty pipeline should return empty sources gracefully."""
    pipeline = FinancialRAGPipeline()
    results = pipeline.retrieve_relevant_chunks("anything", top_k=3)
    assert results == []
