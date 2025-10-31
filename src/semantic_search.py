"""
Semantic Search Engine
Uses sentence-transformers + FAISS for fast similarity search
"""

import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer
from src.utils import get_logger

logger = get_logger()

@dataclass
class SearchResult:
    """Single search result"""
    filename: str
    chunk_text: str
    similarity_score: float
    chunk_index: int

class SemanticSearchEngine:
    """Fast semantic search using sentence embeddings and FAISS"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", offline_mode: bool = True):
        """
        Initialize semantic search engine
        
        Args:
            model_name: SentenceTransformer model to use (default: all-MiniLM-L6-v2)
                - 22MB model, 384-dimensional embeddings
                - Fast inference (~50ms for 128 docs)
            offline_mode: If True, only use local cached models
        """
        logger.info(f"Initializing SemanticSearchEngine with model: {model_name}")
        
        try:
            # Set offline environment variables
            if offline_mode:
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.expanduser('~/.cache/torch/sentence_transformers')
            
            self.model = SentenceTransformer(model_name, cache_folder=None if not offline_mode else os.path.expanduser('~/.cache/torch/sentence_transformers'))
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.success(f"Loaded model: {model_name} (dim={self.embedding_dim})")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Storage for documents and embeddings
        self.documents = []  # List of (filename, chunk_text, chunk_index, category)
        self.index = None  # FAISS index
        self.is_built = False
    
    def build_index(self, documents: List[Dict]) -> None:
        """
        Build FAISS index from documents
        
        Args:
            documents: List of document dicts with 'filename', 'chunks', 'category'
        """
        logger.info("Building semantic search index...")
        
        all_chunks = []
        chunk_metadata = []
        
        # Collect all chunks from all documents
        for doc in documents:
            filename = doc['filename']
            chunks = doc.get('chunks', [])
            category = doc.get('category', 'Unknown')
            
            for idx, chunk in enumerate(chunks):
                if chunk and len(chunk.strip()) > 10:  # Skip empty/tiny chunks
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        'filename': filename,
                        'chunk_text': chunk,
                        'chunk_index': idx,
                        'category': category
                    })
        
        if not all_chunks:
            logger.warning("No chunks found to index")
            return
        
        logger.info(f"Encoding {len(all_chunks)} chunks...")
        
        # Generate embeddings
        embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        # Build FAISS index (IndexFlatL2 for exact nearest neighbor search)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.documents = chunk_metadata
        self.is_built = True
        
        logger.success(f"Index built with {len(all_chunks)} chunks")
        logger.info(f"Memory usage: ~{len(all_chunks) * self.embedding_dim * 4 / 1024 / 1024:.2f} MB")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents
        
        Args:
            query: Search query string
            top_k: Number of results to return
            category_filter: Optional category to filter results (e.g., "Invoice")
        
        Returns:
            List of SearchResult objects sorted by similarity
        """
        if not self.is_built:
            logger.error("Index not built. Call build_index() first.")
            return []
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in FAISS index (get more results if filtering)
        search_k = top_k * 3 if category_filter else top_k
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Convert distances to similarity scores (cosine similarity)
        # Since embeddings are normalized, L2 distance relates to cosine similarity:
        # similarity = 1 - (L2_distance^2 / 2)
        similarities = 1 - (distances[0] ** 2 / 2)
        
        # Build results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            doc_meta = self.documents[idx]
            
            # Apply category filter
            if category_filter and doc_meta['category'] != category_filter:
                continue
            
            result = SearchResult(
                filename=doc_meta['filename'],
                chunk_text=doc_meta['chunk_text'],
                similarity_score=float(similarity),
                chunk_index=doc_meta['chunk_index']
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if not self.is_built:
            return {"status": "not_built"}
        
        unique_files = set(doc['filename'] for doc in self.documents)
        categories = {}
        for doc in self.documents:
            cat = doc['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "status": "ready",
            "total_chunks": len(self.documents),
            "unique_documents": len(unique_files),
            "embedding_dimension": self.embedding_dim,
            "categories": categories,
            "memory_mb": len(self.documents) * self.embedding_dim * 4 / 1024 / 1024
        }


def test_semantic_search():
    """Test the semantic search engine"""
    from src.document_processor import DocumentProcessor
    from src.classifier import ThreeLayerClassifier
    
    logger.section("TESTING SEMANTIC SEARCH ENGINE")
    
    # Load and classify documents
    processor = DocumentProcessor()
    documents = processor.process_folder("data/input_documents")
    
    classifier = ThreeLayerClassifier(offline_mode=True)
    
    # Prepare documents for indexing
    indexed_docs = []
    for doc in documents:
        if not doc.error:
            classification = classifier.classify(doc.text, doc.filename)
            indexed_docs.append({
                'filename': doc.filename,
                'chunks': doc.chunks,
                'category': classification.category
            })
    
    # Build search index
    search_engine = SemanticSearchEngine(offline_mode=True)
    search_engine.build_index(indexed_docs)
    
    # Display stats
    stats = search_engine.get_stats()
    print(f"\n{'='*80}")
    print("INDEX STATISTICS")
    print(f"{'='*80}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Unique Documents: {stats['unique_documents']}")
    print(f"Embedding Dimension: {stats['embedding_dimension']}")
    print(f"Memory Usage: {stats['memory_mb']:.2f} MB")
    print(f"Categories: {stats['categories']}")
    
    # Test queries
    test_queries = [
        ("Find invoices with amounts", None),
        ("Show me resumes with experience", "Resume"),
        ("electricity usage", "Utility Bill"),
        ("payment information", None),
    ]
    
    print(f"\n{'='*80}")
    print("SEARCH RESULTS")
    print(f"{'='*80}\n")
    
    for query, category_filter in test_queries:
        filter_str = f" (filter: {category_filter})" if category_filter else ""
        print(f"\nQuery: '{query}'{filter_str}")
        print("-" * 80)
        
        results = search_engine.search(query, top_k=3, category_filter=category_filter)
        
        if not results:
            print("  No results found")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n  {i}. {result.filename} (similarity: {result.similarity_score:.3f})")
                preview = result.chunk_text[:100] + "..." if len(result.chunk_text) > 100 else result.chunk_text
                print(f"     {preview}")


if __name__ == "__main__":
    test_semantic_search()
