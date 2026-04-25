"""
RAG Evaluation Framework
Computes faithfulness, answer relevance, and context recall metrics
against a golden QA set — no external API required.
"""

import json
import re
import sys
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embedder import FinancialEmbedder
from retrieval.rag_pipeline import FinancialRAGPipeline

GOLDEN_QA_PATH = Path(__file__).parent / "golden_qa.json"
RESULTS_DIR = Path(__file__).parent / "results"

SAMPLE_CHUNKS = [
    {
        "text": "Apple reported Q4 2023 revenue of $94.8 billion, up 1% year-over-year, driven by strong iPhone sales.",
        "company": "Apple",
        "quarter": "Q4 2023",
        "source": "earnings_call",
        "type": "earnings",
    },
    {
        "text": "Tesla delivered 484,507 vehicles in Q3 2023, exceeding expectations and showing strong demand for electric vehicles.",
        "company": "Tesla",
        "quarter": "Q3 2023",
        "source": "earnings_call",
        "type": "earnings",
    },
    {
        "text": "Microsoft Azure revenue grew 29% year-over-year in the latest quarter, driven by increased cloud adoption.",
        "company": "Microsoft",
        "quarter": "Q3 2023",
        "source": "earnings_call",
        "type": "earnings",
    },
    {
        "text": "Amazon Web Services reported operating income of $7.0 billion, up 12% year-over-year.",
        "company": "Amazon",
        "quarter": "Q3 2023",
        "source": "earnings_call",
        "type": "earnings",
    },
    {
        "text": "The Federal Reserve raised interest rates by 0.25% to combat inflation, affecting tech stock valuations.",
        "company": "Market",
        "quarter": "Q3 2023",
        "source": "news",
        "type": "news",
    },
]


def _tokenize(text: str) -> set:
    """Lowercase word-level tokenization for overlap metrics."""
    return set(re.findall(r"\w+", text.lower()))


class RAGEvaluator:
    """Evaluates a FinancialRAGPipeline on a golden QA set."""

    def __init__(self, rag: FinancialRAGPipeline, embedder: FinancialEmbedder):
        self.rag = rag
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def faithfulness(self, answer: str, context_chunks: List[Dict]) -> float:
        """
        Token-overlap between answer and concatenated retrieved context.
        High overlap => answer is grounded in the context (less hallucination).
        Returns a score in [0, 1].
        """
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0
        context_tokens: set = set()
        for chunk in context_chunks:
            context_tokens |= _tokenize(chunk.get("text", ""))
        overlap = answer_tokens & context_tokens
        return len(overlap) / len(answer_tokens)

    def answer_relevance(self, question: str, answer: str) -> float:
        """
        Cosine similarity between question embedding and answer embedding.
        Measures whether the answer is on-topic.
        Returns a score in [0, 1] (clamped from cosine range).
        """
        q_emb = self.embedder.generate_embedding(question)
        a_emb = self.embedder.generate_embedding(answer)
        sim = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9))
        return max(0.0, min(1.0, sim))

    def context_recall(self, expected_answer: str, context_chunks: List[Dict]) -> float:
        """
        Keyword overlap between the *expected* answer and retrieved chunks.
        High recall => the retriever surfaced the right information.
        Returns a score in [0, 1].
        """
        expected_tokens = _tokenize(expected_answer)
        if not expected_tokens:
            return 0.0
        context_tokens: set = set()
        for chunk in context_chunks:
            context_tokens |= _tokenize(chunk.get("text", ""))
        overlap = expected_tokens & context_tokens
        return len(overlap) / len(expected_tokens)

    # ------------------------------------------------------------------
    # Full evaluation run
    # ------------------------------------------------------------------

    def evaluate(self, golden_qa: List[Dict], top_k: int = 3) -> Dict:
        """
        Run all metrics on each golden QA pair and return aggregate results.
        """
        per_question: List[Dict] = []

        for qa in golden_qa:
            question = qa["question"]
            expected = qa["expected_answer"]

            response = self.rag.query(question, top_k=top_k)
            answer = response["answer"]
            sources = response["sources"]

            faith = self.faithfulness(answer, sources)
            relevance = self.answer_relevance(question, answer)
            recall = self.context_recall(expected, sources)

            per_question.append({
                "question": question,
                "answer": answer,
                "faithfulness": round(faith, 4),
                "answer_relevance": round(relevance, 4),
                "context_recall": round(recall, 4),
                "num_sources": len(sources),
            })

        avg = lambda key: round(np.mean([r[key] for r in per_question]), 4)
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(per_question),
            "avg_faithfulness": avg("faithfulness"),
            "avg_answer_relevance": avg("answer_relevance"),
            "avg_context_recall": avg("context_recall"),
            "per_question": per_question,
        }
        return summary


def run_evaluation() -> Dict:
    """End-to-end: build pipeline, load golden set, evaluate, save results."""

    # Build RAG pipeline with sample data
    rag = FinancialRAGPipeline()
    embedded = rag.embedder.embed_document_chunks(SAMPLE_CHUNKS)
    rag.build_index(embedded)

    # Load golden QA
    with open(GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        golden_qa = json.load(f)

    evaluator = RAGEvaluator(rag, rag.embedder)
    results = evaluator.evaluate(golden_qa, top_k=3)

    # Print summary table
    print("\n" + "=" * 60)
    print("  RAG Evaluation Results")
    print("=" * 60)
    print(f"  Questions evaluated : {results['num_questions']}")
    print(f"  Avg Faithfulness    : {results['avg_faithfulness']:.4f}")
    print(f"  Avg Answer Relevance: {results['avg_answer_relevance']:.4f}")
    print(f"  Avg Context Recall  : {results['avg_context_recall']:.4f}")
    print("=" * 60)

    print("\nPer-question breakdown:")
    for i, r in enumerate(results["per_question"], 1):
        print(f"\n  Q{i}: {r['question']}")
        print(f"      faith={r['faithfulness']:.3f}  rel={r['answer_relevance']:.3f}  recall={r['context_recall']:.3f}")

    # Save to disk
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
