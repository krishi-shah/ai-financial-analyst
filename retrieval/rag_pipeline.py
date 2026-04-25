"""
RAG Pipeline Module
Implements Retrieval-Augmented Generation for financial Q&A.
"""

import numpy as np
import json
from typing import List, Dict
from pathlib import Path
from openai import OpenAI
from config import OPENAI_API_KEY

# Handle imports with error handling
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: faiss not available. Please install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False

try:
    from embeddings.embedder import FinancialEmbedder
    EMBEDDER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FinancialEmbedder not available: {e}")
    EMBEDDER_AVAILABLE = False
    FinancialEmbedder = None

try:
    from retrieval.local_llm import LocalFinancialLLM
    LOCAL_LLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LocalFinancialLLM not available: {e}")
    LOCAL_LLM_AVAILABLE = False
    LocalFinancialLLM = None


class FinancialRAGPipeline:
    """
    RAG pipeline for financial document Q&A.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_name: Name of the embedding model to use
        """
        if not EMBEDDER_AVAILABLE or not FAISS_AVAILABLE:
            raise ImportError("Required dependencies not available. Please install tf-keras and faiss-cpu")
        
        self.embedder = FinancialEmbedder(model_name)
        self.index = None
        self.chunks = []
        self.openai_client = None
        self.local_llm = None
        
        if OPENAI_API_KEY and OPENAI_API_KEY.strip() not in ("", "your_openai_api_key_here", "OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize local LLM as backup
        if LOCAL_LLM_AVAILABLE:
            self.local_llm = LocalFinancialLLM()
    
    def build_index(self, embedded_chunks: List[Dict]):
        """
        Build FAISS index from embedded chunks.
        
        Args:
            embedded_chunks: List of chunks with embeddings
        """
        if not embedded_chunks:
            print("No embedded chunks provided")
            return
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks])
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks for reference
        self.chunks = embedded_chunks
        
        print(f"Built FAISS index with {len(embedded_chunks)} chunks")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
        
        Returns:
            List of relevant chunks with similarity scores
        """
        if self.index is None:
            print("Index not built. Please call build_index() first.")
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search index
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Get relevant chunks
        relevant_chunks = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(similarity)
                chunk['rank'] = i + 1
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """
        Generate answer using retrieved chunks and LLM.
        
        Args:
            query: User query
            relevant_chunks: Retrieved relevant chunks
        
        Returns:
            Generated answer
        """
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context from chunks
        context = self._prepare_context(relevant_chunks)
        
        # Generate answer using OpenAI (if available), local LLM, or fallback
        if self.openai_client:
            try:
                return self._generate_openai_answer(query, context)
            except Exception as e:
                print(f"OpenAI API error: {e}")
                # Fall back to local LLM or basic fallback
                if self.local_llm:
                    return self._generate_local_llm_answer(query, context)
                else:
                    return self._generate_fallback_answer(query, context)
        elif self.local_llm:
            return self._generate_local_llm_answer(query, context)
        else:
            return self._generate_fallback_answer(query, context)
    
    def _generate_local_llm_answer(self, query: str, context: str) -> str:
        """
        Generate answer using local LLM.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Generated answer using local LLM
        """
        if self.local_llm:
            return self.local_llm.generate_answer(query, context)
        else:
            return self._generate_fallback_answer(query, context)
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """
        Prepare context string from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source_info = self._get_source_info(chunk)
            context_parts.append(f"[Source {i}] {source_info}\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def _get_source_info(self, chunk: Dict) -> str:
        """
        Get source information for a chunk.
        
        Args:
            chunk: Document chunk
        
        Returns:
            Source information string
        """
        source_parts = []
        
        if 'company' in chunk:
            source_parts.append(f"Company: {chunk['company']}")
        
        if 'quarter' in chunk:
            source_parts.append(f"Quarter: {chunk['quarter']}")
        
        if 'date' in chunk:
            source_parts.append(f"Date: {chunk['date']}")
        
        if 'source' in chunk:
            source_parts.append(f"Source: {chunk['source']}")
        
        if 'file_path' in chunk:
            source_parts.append(f"File: {Path(chunk['file_path']).name}")
        
        return " | ".join(source_parts) if source_parts else "Unknown source"
    
    def _generate_openai_answer(self, query: str, context: str) -> str:
        """
        Generate answer using OpenAI GPT.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Generated answer
        """
        try:
            prompt = (
                "You are a financial analyst AI. Answer the question using ONLY "
                "the provided context. Cite sources as [Source N]. If the context "
                "is insufficient, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\nAnswer:"
            )

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst AI. Only answer from the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating OpenAI answer: {e}")
            return self._generate_fallback_answer(query, context)
    
    def _generate_fallback_answer(self, query: str, context: str) -> str:
        """
        Generate fallback answer without LLM.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Fallback answer
        """
        # Enhanced keyword-based answer generation
        query_lower = query.lower()
        query_words = [word.strip('?.,!') for word in query_lower.split() if len(word) > 2]
        
        # Find relevant sentences with scoring
        relevant_sentences = []
        for chunk in context.split('\n'):
            if chunk.strip():
                chunk_lower = chunk.lower()
                score = sum(1 for word in query_words if word in chunk_lower)
                if score > 0:
                    relevant_sentences.append((chunk.strip(), score))
        
        # Sort by relevance score
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_sentences:
            answer = "Based on the available information:\n\n"
            for sentence, score in relevant_sentences[:3]:
                answer += f"• {sentence}\n"
            
            answer += f"\n📊 **Found {len(relevant_sentences)} relevant sources**\n"
            answer += "💡 **Note**: This is an AI-enhanced answer. For more sophisticated responses, check your OpenAI API quota."
        else:
            answer = "I found some relevant information, but couldn't generate a specific answer based on the query."
        
        return answer
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k)
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)
        
        # Prepare response
        response = {
            'question': question,
            'answer': answer,
            'sources': relevant_chunks,
            'num_sources': len(relevant_chunks)
        }
        
        return response
    
    def save_index(self, file_path: str):
        """
        Save FAISS index to file.
        
        Args:
            file_path: Path to save the index
        """
        if self.index is None:
            print("No index to save")
            return
        
        # Save FAISS index
        faiss.write_index(self.index, file_path)
        
        # Save chunks metadata
        chunks_path = file_path.replace('.index', '_chunks.json')
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        print(f"Saved index to {file_path}")
    
    def load_index(self, file_path: str):
        """
        Load FAISS index from file.
        
        Args:
            file_path: Path to load the index from
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(file_path)
            
            # Load chunks metadata
            chunks_path = file_path.replace('.index', '_chunks.json')
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            print(f"Loaded index with {len(self.chunks)} chunks from {file_path}")
            
        except Exception as e:
            print(f"Error loading index: {e}")


def main():
    """Sample usage of the RAG pipeline."""
    print("Financial RAG Pipeline")
    print("This module implements Retrieval-Augmented Generation for financial Q&A.")
    
    # Initialize RAG pipeline
    rag = FinancialRAGPipeline()
    
    # Sample chunks (in practice, these would come from your data ingestion)
    sample_chunks = [
        {
            'text': 'Apple reported Q4 2023 revenue of $94.8 billion, up 1% year-over-year.',
            'company': 'Apple',
            'quarter': 'Q4 2023',
            'source': 'earnings_call'
        },
        {
            'text': 'Tesla delivered 484,507 vehicles in Q3 2023, exceeding expectations.',
            'company': 'Tesla',
            'quarter': 'Q3 2023',
            'source': 'earnings_call'
        },
        {
            'text': 'Microsoft Azure revenue grew 29% year-over-year in the latest quarter.',
            'company': 'Microsoft',
            'quarter': 'Q3 2023',
            'source': 'earnings_call'
        }
    ]
    
    # Embed chunks
    embedded_chunks = rag.embedder.embed_document_chunks(sample_chunks)
    
    # Build index
    rag.build_index(embedded_chunks)
    
    # Test queries
    test_queries = [
        "What was Apple's revenue in Q4 2023?",
        "How many vehicles did Tesla deliver?",
        "What was Microsoft's Azure growth rate?"
    ]
    
    print("\nTesting RAG pipeline with sample queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = rag.query(query)
        print(f"Answer: {response['answer']}")
        print(f"Sources: {response['num_sources']} chunks retrieved")


if __name__ == "__main__":
    main()