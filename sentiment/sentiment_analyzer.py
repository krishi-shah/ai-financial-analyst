"""
Sentiment Analysis Module
Analyzes financial text sentiment using FinBERT and other models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Union
import re


class FinancialSentimentAnalyzer:
    """
    Analyzes sentiment of financial text using specialized models.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """
        Load the pre-trained model and tokenizer.
        """
        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to basic sentiment analysis")
            self.model = None
            self.tokenizer = None
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary with sentiment analysis results
        """
        if self.model is None:
            return self._basic_sentiment_analysis(text)
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predicted class and confidence
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = torch.max(predictions, dim=-1).values.item()
        
        # Map class to sentiment
        sentiment_labels = self._get_sentiment_labels()
        predicted_sentiment = sentiment_labels[predicted_class]
        
        # Get all class probabilities
        class_probabilities = predictions[0].cpu().numpy()
        
        return {
            'text': text,
            'sentiment': predicted_sentiment,
            'confidence': confidence,
            'class_probabilities': {
                label: float(prob) for label, prob in zip(sentiment_labels, class_probabilities)
            },
            'model_used': self.model_name
        }
    
    def analyze_batch_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            List of sentiment analysis results
        """
        if self.model is None:
            return [self._basic_sentiment_analysis(text) for text in texts]
        
        # Preprocess texts
        cleaned_texts = [self._preprocess_text(text) for text in texts]
        
        # Tokenize batch
        inputs = self.tokenizer(
            cleaned_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Process results
        results = []
        sentiment_labels = self._get_sentiment_labels()
        
        for i, text in enumerate(texts):
            predicted_class = torch.argmax(predictions[i]).item()
            confidence = torch.max(predictions[i]).item()
            predicted_sentiment = sentiment_labels[predicted_class]
            
            class_probabilities = predictions[i].cpu().numpy()
            
            results.append({
                'text': text,
                'sentiment': predicted_sentiment,
                'confidence': confidence,
                'class_probabilities': {
                    label: float(prob) for label, prob in zip(sentiment_labels, class_probabilities)
                },
                'model_used': self.model_name
            })
        
        return results
    
    def _get_sentiment_labels(self) -> List[str]:
        """
        Return sentiment labels in the order the model's output logits use.
        Read directly from model.config.id2label so we never hard-code the
        wrong order (ProsusAI/finbert uses {0: positive, 1: negative, 2: neutral}).
        """
        if self.model is not None and hasattr(self.model, "config"):
            id2label = self.model.config.id2label
            return [id2label[i].lower() for i in range(len(id2label))]
        return ["positive", "negative", "neutral"]
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        text = re.sub(r'\s+', ' ', text)
        # Preserve $, %, numbers — important in financial text
        text = re.sub(r'[^\w\s.,!?$%]', '', text)
        return text.strip()
    
    def _basic_sentiment_analysis(self, text: str) -> Dict:
        """
        Basic sentiment analysis using keyword matching.
        
        Args:
            text: Input text
        
        Returns:
            Basic sentiment analysis results
        """
        # Define sentiment keywords
        positive_keywords = [
            'positive', 'growth', 'increase', 'strong', 'excellent', 'outstanding',
            'bullish', 'optimistic', 'confident', 'successful', 'profitable',
            'beat', 'exceed', 'surpass', 'outperform', 'gain', 'rise', 'up'
        ]
        
        negative_keywords = [
            'negative', 'decline', 'decrease', 'weak', 'poor', 'disappointing',
            'bearish', 'pessimistic', 'concerned', 'struggling', 'loss',
            'miss', 'fall', 'drop', 'underperform', 'challenge', 'risk'
        ]
        
        text_lower = text.lower()
        
        # Count positive and negative keywords
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.8, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'class_probabilities': {
                'negative': 0.33 if sentiment == "negative" else 0.2,
                'neutral': 0.33 if sentiment == "neutral" else 0.2,
                'positive': 0.33 if sentiment == "positive" else 0.2
            },
            'model_used': 'basic_keyword_analysis'
        }
    
    def analyze_financial_document(self, document: Dict) -> Dict:
        """
        Analyze sentiment of a financial document.
        
        Args:
            document: Document dictionary with 'text' field
        
        Returns:
            Document with added sentiment analysis
        """
        text = document.get('text', '')
        
        # Analyze overall sentiment
        sentiment_result = self.analyze_sentiment(text)
        
        # Add sentiment to document
        document_with_sentiment = document.copy()
        document_with_sentiment['sentiment'] = sentiment_result['sentiment']
        document_with_sentiment['sentiment_confidence'] = sentiment_result['confidence']
        document_with_sentiment['sentiment_analysis'] = sentiment_result
        
        return document_with_sentiment
    
    def analyze_document_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for document chunks.
        
        Args:
            chunks: List of document chunks
        
        Returns:
            List of chunks with added sentiment analysis
        """
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Analyze sentiment for all texts
        sentiment_results = self.analyze_batch_sentiment(texts)
        
        # Add sentiment to chunks
        chunks_with_sentiment = []
        for chunk, sentiment_result in zip(chunks, sentiment_results):
            chunk_with_sentiment = chunk.copy()
            chunk_with_sentiment['sentiment'] = sentiment_result['sentiment']
            chunk_with_sentiment['sentiment_confidence'] = sentiment_result['confidence']
            chunk_with_sentiment['sentiment_analysis'] = sentiment_result
            chunks_with_sentiment.append(chunk_with_sentiment)
        
        return chunks_with_sentiment
    
    def get_sentiment_summary(self, chunks: List[Dict]) -> Dict:
        """
        Get sentiment summary for a collection of chunks.
        
        Args:
            chunks: List of chunks with sentiment analysis
        
        Returns:
            Sentiment summary statistics
        """
        if not chunks:
            return {}
        
        # Count sentiments
        sentiment_counts = {}
        total_confidence = 0
        
        for chunk in chunks:
            sentiment = chunk.get('sentiment', 'neutral')
            confidence = chunk.get('sentiment_confidence', 0.5)
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_confidence += confidence
        
        # Calculate percentages
        total_chunks = len(chunks)
        sentiment_percentages = {
            sentiment: (count / total_chunks) * 100 
            for sentiment, count in sentiment_counts.items()
        }
        
        # Calculate average confidence
        avg_confidence = total_confidence / total_chunks if total_chunks > 0 else 0
        
        # Determine overall sentiment
        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'average_confidence': avg_confidence,
            'total_chunks': total_chunks
        }


def main():
    """Sample usage of the sentiment analyzer."""
    print("Financial Sentiment Analyzer")
    print("This module analyzes sentiment of financial text using FinBERT and other models.")
    
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer()
    
    # Sample financial texts
    sample_texts = [
        "Apple reported strong Q4 earnings with revenue growth of 8% year-over-year, exceeding analyst expectations.",
        "Tesla's stock price declined significantly due to concerns about production delays and increased competition.",
        "Microsoft's cloud computing division showed robust growth, but the company faces challenges in the mobile market.",
        "Amazon's e-commerce business continues to dominate, with record-breaking sales during the holiday season."
    ]
    
    print(f"\nAnalyzing sentiment for {len(sample_texts)} sample texts...")
    
    # Analyze sentiment
    results = analyzer.analyze_batch_sentiment(sample_texts)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\nText {i}: {result['text'][:80]}...")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print(f"Model used: {result['model_used']}")
    
    # Test with document chunks
    print("\n" + "="*50)
    print("Testing with document chunks...")
    
    sample_chunks = [
        {'text': text, 'source': 'sample', 'type': 'news'} 
        for text in sample_texts
    ]
    
    # Analyze chunks
    chunks_with_sentiment = analyzer.analyze_document_chunks(sample_chunks)
    
    # Get summary
    summary = analyzer.get_sentiment_summary(chunks_with_sentiment)
    
    print(f"\nSentiment Summary:")
    print(f"Overall sentiment: {summary['overall_sentiment']}")
    print(f"Distribution: {summary['sentiment_distribution']}")
    print(f"Average confidence: {summary['average_confidence']:.3f}")


if __name__ == "__main__":
    main()