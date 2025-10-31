"""
Three-Layer Document Classification System
Layer 1: Rule-based (fast keyword matching)
Layer 2: TF-IDF similarity (template matching)
Layer 3: Zero-shot classification (deep learning)
"""

import re
import time
import os
from typing import Dict, Tuple, List
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
from src.utils import get_logger

logger = get_logger()

@dataclass
class ClassificationResult:
    """Classification result with confidence score"""
    category: str
    confidence: float
    method: str  # Which layer made the decision
    layer_scores: Dict[str, float]  # Scores from each layer

class ThreeLayerClassifier:
    """Advanced three-layer classification with early termination"""
    
    CATEGORIES = ["Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"]
    
    def __init__(self, offline_mode: bool = True):
        """
        Initialize all three classification layers
        
        Args:
            offline_mode: If True, uses local cache only (no downloads)
        """
        logger.info("Initializing ThreeLayerClassifier...")
        
        self.offline_mode = offline_mode
        if offline_mode:
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        # Layer 1: Rule-based keywords
        self.keywords = {
            "Invoice": [
                "invoice", "invoice #", "invoice number", "inv #", "inv-",
                "total amount", "amount due", "payment", "bill to", "date:",
                "thank you for your business", "subtotal", "tax", "grand total"
            ],
            "Resume": [
                "email:", "phone:", "experience:", "years", "summary:",
                "skills:", "education:", "work experience", "professional",
                "objective:", "qualifications", "cv", "curriculum vitae"
            ],
            "Utility Bill": [
                "account number", "acc-", "billing date", "usage", "kwh",
                "amount due", "utility provider", "electric", "water", "gas",
                "meter reading", "previous balance", "current charges"
            ]
        }
        
        # Layer 2: TF-IDF with template documents
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words='english'
        )
        self.tfidf_trained = False
        self.template_vectors = None
        self.template_labels = None
        
        # Layer 3: Zero-shot classification (lazy loaded)
        self.zero_shot_classifier = None
        self.zero_shot_labels = ["invoice", "resume", "utility bill", "general document"]
        
        logger.success("ThreeLayerClassifier initialized")
    
    def classify(self, text: str, filename: str = "") -> ClassificationResult:
        """
        Classify document using three-layer approach with early termination
        
        Args:
            text: Document text
            filename: Original filename (for logging)
        
        Returns:
            ClassificationResult with category and confidence
        """
        if not text or len(text.strip()) < 10:
            return ClassificationResult(
                category="Unclassifiable",
                confidence=1.0,
                method="empty_check",
                layer_scores={}
            )
        
        text_lower = text.lower()
        layer_scores = {}
        
        # Layer 1: Rule-based (2ms)
        start_time = time.time()
        rule_category, rule_confidence = self._layer1_rule_based(text_lower)
        layer1_time = (time.time() - start_time) * 1000
        layer_scores["rule_based"] = rule_confidence
        logger.info(f"  Layer 1 (Rule): {rule_category} ({rule_confidence:.2f}) [{layer1_time:.1f}ms]")
        
        # Early termination if high confidence
        if rule_confidence >= 0.85:
            return ClassificationResult(
                category=rule_category,
                confidence=rule_confidence,
                method="Layer1_RuleBased",
                layer_scores=layer_scores
            )
        
        # Layer 2: TF-IDF (15ms)
        start_time = time.time()
        tfidf_category, tfidf_confidence = self._layer2_tfidf(text)
        layer2_time = (time.time() - start_time) * 1000
        layer_scores["tfidf"] = tfidf_confidence
        logger.info(f"  Layer 2 (TF-IDF): {tfidf_category} ({tfidf_confidence:.2f}) [{layer2_time:.1f}ms]")
        
        # Early termination if moderate confidence
        if tfidf_confidence >= 0.7:
            return ClassificationResult(
                category=tfidf_category,
                confidence=tfidf_confidence,
                method="Layer2_TFIDF",
                layer_scores=layer_scores
            )
        
        # Layer 3: Zero-shot classification (85ms) - only if not in offline mode or model available
        start_time = time.time()
        try:
            zeroshot_category, zeroshot_confidence = self._layer3_zero_shot(text)
            layer3_time = (time.time() - start_time) * 1000
            layer_scores["zero_shot"] = zeroshot_confidence
            logger.info(f"  Layer 3 (Zero-shot): {zeroshot_category} ({zeroshot_confidence:.2f}) [{layer3_time:.1f}ms]")
        except Exception as e:
            logger.warning(f"  Layer 3 unavailable (offline mode or model not cached): {str(e)[:50]}")
            zeroshot_category, zeroshot_confidence = "Other", 0.3
            layer_scores["zero_shot"] = 0.0
        
        # Ensemble voting (weighted combination)
        final_category, final_confidence = self._ensemble_vote(
            rule_category, rule_confidence,
            tfidf_category, tfidf_confidence,
            zeroshot_category, zeroshot_confidence
        )
        
        # Mark as unclassifiable if confidence too low
        if final_confidence < 0.5:
            final_category = "Unclassifiable"
        
        return ClassificationResult(
            category=final_category,
            confidence=final_confidence,
            method="Layer3_Ensemble",
            layer_scores=layer_scores
        )
    
    def _layer1_rule_based(self, text_lower: str) -> Tuple[str, float]:
        """
        Layer 1: Fast keyword-based classification
        
        Returns:
            Tuple of (category, confidence)
        """
        scores = {}
        for category, keywords in self.keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            score = matches / len(keywords)
            scores[category] = score
        
        if not scores or max(scores.values()) == 0:
            return "Other", 0.3
        
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]
        
        return best_category, confidence
    
    def _layer2_tfidf(self, text: str) -> Tuple[str, float]:
        """
        Layer 2: TF-IDF based template matching
        
        Returns:
            Tuple of (category, confidence)
        """
        # Create simple templates if not trained
        if not self.tfidf_trained:
            templates = [
                ("Invoice #1001 Date: 2025-01-01 Company: ABC Corp Total Amount: $1000 Thank you for your business", "Invoice"),
                ("John Doe Email: john@email.com Phone: 555-1234 Experience: 5 years Summary: Professional", "Resume"),
                ("Account Number: ACC-123 Billing Date: 2025-01-01 Usage: 500 kWh Amount Due: $100 Utility Provider", "Utility Bill"),
                ("This is a general document with random information", "Other")
            ]
            
            texts = [t[0] for t in templates]
            self.template_labels = [t[1] for t in templates]
            self.template_vectors = self.tfidf_vectorizer.fit_transform(texts)
            self.tfidf_trained = True
        
        # Transform input text
        try:
            text_vector = self.tfidf_vectorizer.transform([text])
            similarities = cosine_similarity(text_vector, self.template_vectors)[0]
            best_idx = np.argmax(similarities)
            confidence = float(similarities[best_idx])
            category = self.template_labels[best_idx]
            return category, confidence
        except:
            return "Other", 0.3
    
    def _layer3_zero_shot(self, text: str) -> Tuple[str, float]:
        """
        Layer 3: Zero-shot classification using transformers
        
        Returns:
            Tuple of (category, confidence)
        """
        # Lazy load the model (only when needed)
        if self.zero_shot_classifier is None:
            logger.info("  Loading zero-shot model (first time only)...")
            device = 0 if torch.cuda.is_available() else -1
            
            # Use local_files_only for offline mode
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="typeform/distilbert-base-uncased-mnli",
                device=device,
                local_files_only=self.offline_mode
            )
        
        try:
            # Truncate text if too long
            text_truncated = text[:512]
            result = self.zero_shot_classifier(
                text_truncated,
                candidate_labels=self.zero_shot_labels,
                multi_label=False
            )
            
            # Map to our categories
            label_map = {
                "invoice": "Invoice",
                "resume": "Resume",
                "utility bill": "Utility Bill",
                "general document": "Other"
            }
            
            predicted_label = result['labels'][0]
            confidence = result['scores'][0]
            category = label_map.get(predicted_label, "Other")
            
            return category, confidence
        
        except Exception as e:
            logger.warning(f"  Zero-shot classification failed: {str(e)[:50]}")
            return "Other", 0.3
    
    def _ensemble_vote(
        self,
        cat1: str, conf1: float,
        cat2: str, conf2: float,
        cat3: str, conf3: float
    ) -> Tuple[str, float]:
        """
        Ensemble voting: weighted combination of all three layers
        Weights: 30% Layer1 + 30% Layer2 + 40% Layer3
        """
        votes = {}
        
        # Weighted votes
        weights = [0.3, 0.3, 0.4]
        categories = [cat1, cat2, cat3]
        confidences = [conf1, conf2, conf3]
        
        for cat, conf, weight in zip(categories, confidences, weights):
            if cat not in votes:
                votes[cat] = 0
            votes[cat] += conf * weight
        
        best_category = max(votes, key=votes.get)
        final_confidence = votes[best_category]
        
        return best_category, final_confidence


def test_classifier():
    """Test the three-layer classifier"""
    from src.document_processor import DocumentProcessor
    
    logger.section("TESTING THREE-LAYER CLASSIFIER")
    
    # Load documents
    processor = DocumentProcessor()
    documents = processor.process_folder("data/input_documents")
    
    # Initialize classifier
    classifier = ThreeLayerClassifier(offline_mode=True)
    
    # Classify each document
    results = []
    for doc in documents:
        if doc.error:
            result = ClassificationResult(
                category="Unclassifiable",
                confidence=1.0,
                method="error",
                layer_scores={}
            )
        else:
            logger.info(f"\nClassifying: {doc.filename}")
            result = classifier.classify(doc.text, doc.filename)
        
        results.append((doc.filename, result))
        logger.success(f"  → {result.category} ({result.confidence:.2f}) via {result.method}")
    
    # Summary
    print(f"\n{'='*80}")
    print("CLASSIFICATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Filename':<30} {'Category':<15} {'Confidence':<12} {'Method'}")
    print(f"{'-'*80}")
    
    for filename, result in results:
        print(f"{filename:<30} {result.category:<15} {result.confidence:<12.2f} {result.method}")


if __name__ == "__main__":
    test_classifier()
