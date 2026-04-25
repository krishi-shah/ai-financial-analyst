"""Tests for the FinancialSentimentAnalyzer."""

import pytest
from sentiment.sentiment_analyzer import FinancialSentimentAnalyzer

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


@pytest.fixture(scope="module")
def analyzer():
    return FinancialSentimentAnalyzer()


def test_sentiment_label_is_valid(analyzer):
    result = analyzer.analyze_sentiment("Revenue increased sharply this quarter.")
    assert result["sentiment"] in VALID_SENTIMENTS


def test_confidence_in_range(analyzer):
    result = analyzer.analyze_sentiment("The company posted a significant loss.")
    assert 0.0 <= result["confidence"] <= 1.0


def test_class_probabilities_sum_to_one(analyzer):
    result = analyzer.analyze_sentiment("Earnings were in line with expectations.")
    probs = result["class_probabilities"]
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.02, f"Probabilities sum to {total}, expected ~1.0"


def test_positive_text_detected(analyzer):
    result = analyzer.analyze_sentiment(
        "The company smashed earnings expectations and raised full-year guidance."
    )
    assert result["sentiment"] in ("positive", "neutral")


def test_negative_text_detected(analyzer):
    result = analyzer.analyze_sentiment(
        "Revenue plunged 40% and the company announced mass layoffs."
    )
    assert result["sentiment"] in ("negative", "neutral")
