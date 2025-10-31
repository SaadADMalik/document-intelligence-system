"""
Main Pipeline Integration
Orchestrates all components: processor → classifier → extractor → search
Generates final output.json
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import time

from src.document_processor import DocumentProcessor
from src.classifier import ThreeLayerClassifier
from src.extractor import DataExtractor
from src.semantic_search import SemanticSearchEngine
from src.utils import get_logger

logger = get_logger()

class DocumentPipeline:
    """Complete document processing pipeline"""
    
    def __init__(self, offline_mode: bool = True):
        """
        Initialize all pipeline components
        
        Args:
            offline_mode: If True, uses only local cached models (no downloads)
        """
        logger.section("INITIALIZING DOCUMENT PIPELINE")
        
        self.processor = DocumentProcessor()
        self.classifier = ThreeLayerClassifier(offline_mode=offline_mode)
        self.extractor = DataExtractor(offline_mode=offline_mode)
        self.search_engine = SemanticSearchEngine(offline_mode=offline_mode)
        
        logger.success("Pipeline initialized")
    
    def process(self, input_folder: str, output_file: str = "output.json") -> Dict:
        """
        Process all documents in folder
        
        Args:
            input_folder: Path to folder with PDF files
            output_file: Output JSON file path
        
        Returns:
            Pipeline results dictionary
        """
        logger.section("PROCESSING DOCUMENTS")
        start_time = time.time()
        
        # Step 1: Process documents
        logger.info("Step 1/4: Processing documents...")
        documents = self.processor.process_folder(input_folder)
        
        # Step 2: Classify documents
        logger.info("Step 2/4: Classifying documents...")
        classified_docs = self._classify_documents(documents)
        
        # Step 3: Extract data
        logger.info("Step 3/4: Extracting structured data...")
        extracted_docs = self._extract_data(classified_docs)
        
        # Step 4: Build search index
        logger.info("Step 4/4: Building semantic search index...")
        indexed_docs = self._build_search_index(extracted_docs)
        
        # Generate output
        output = self._generate_output(extracted_docs, indexed_docs)
        
        # Save to JSON
        self._save_output(output, output_file)
        
        elapsed_time = time.time() - start_time
        logger.success(f"Pipeline completed in {elapsed_time:.2f} seconds")
        
        return output
    
    def _classify_documents(self, documents: List) -> List[Dict]:
        """Classify all documents"""
        classified = []
        
        for doc in documents:
            if doc.error:
                classification = {
                    'filename': doc.filename,
                    'category': 'Unclassifiable',
                    'confidence': 1.0,
                    'method': 'error',
                    'layer_scores': {}
                }
            else:
                result = self.classifier.classify(doc.text, doc.filename)
                classification = {
                    'filename': doc.filename,
                    'category': result.category,
                    'confidence': result.confidence,
                    'method': result.method,
                    'layer_scores': result.layer_scores
                }
            
            classified.append({
                **doc.__dict__,
                **classification
            })
        
        return classified
    
    def _extract_data(self, classified_docs: List[Dict]) -> List[Dict]:
        """Extract structured data from classified documents"""
        extracted = []
        
        for doc in classified_docs:
            if doc.get('error') or doc.get('category') == 'Unclassifiable':
                extraction = {
                    'extracted_fields': {},
                    'extraction_confidence': 0.0
                }
            else:
                result = self.extractor.extract(doc['text'], doc['category'])
                extraction = {
                    'extracted_fields': {
                        field: {
                            'value': field_data.value,
                            'confidence': field_data.confidence
                        }
                        for field, field_data in result.fields.items()
                    },
                    'extraction_confidence': result.overall_confidence
                }
            
            extracted.append({
                **doc,
                **extraction
            })
        
        return extracted
    
    def _build_search_index(self, extracted_docs: List[Dict]) -> Dict:
        """Build semantic search index"""
        indexed_docs = []
        
        for doc in extracted_docs:
            if not doc.get('error') and doc.get('category') != 'Unclassifiable':
                indexed_docs.append({
                    'filename': doc['filename'],
                    'chunks': doc['chunks'],
                    'category': doc['category']
                })
        
        # Build index
        if indexed_docs:
            self.search_engine.build_index(indexed_docs)
            stats = self.search_engine.get_stats()
        else:
            stats = {}
        
        return stats
    
    def _generate_output(self, extracted_docs: List[Dict], search_stats: Dict) -> Dict:
        """Generate final output dictionary"""
        # Document summaries
        document_results = []
        
        for doc in extracted_docs:
            doc_summary = {
                'filename': doc['filename'],
                'text_length': len(doc.get('text', '')),
                'classification': {
                    'category': doc.get('category', 'Unknown'),
                    'confidence': doc.get('confidence', 0.0),
                    'method': doc.get('method', 'unknown'),
                    'layer_scores': doc.get('layer_scores', {})
                },
                'extraction': {
                    'fields': doc.get('extracted_fields', {}),
                    'confidence': doc.get('extraction_confidence', 0.0)
                },
                'processing': {
                    'error': doc.get('error'),
                    'chunks': len(doc.get('chunks', []))
                }
            }
            document_results.append(doc_summary)
        
        # Summary statistics
        total_docs = len(extracted_docs)
        successful_docs = sum(1 for d in extracted_docs if not d.get('error') and d.get('category') != 'Unclassifiable')
        failed_docs = total_docs - successful_docs
        
        category_counts = {}
        for doc in extracted_docs:
            cat = doc.get('category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Exclude unclassifiable from confidence calculations
        classifiable_docs = [d for d in extracted_docs if d.get('category') != 'Unclassifiable']
        
        avg_classification_conf = sum(
            d.get('confidence', 0) for d in classifiable_docs
        ) / max(1, len(classifiable_docs))
        
        avg_extraction_conf = sum(
            d.get('extraction_confidence', 0) for d in extracted_docs if not d.get('error')
        ) / max(1, successful_docs)
        
        # Final output
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': '1.0',
                'total_documents': total_docs,
                'successful_documents': successful_docs,
                'failed_documents': failed_docs
            },
            'summary': {
                'document_categories': category_counts,
                'avg_classification_confidence': round(avg_classification_conf, 3),
                'avg_extraction_confidence': round(avg_extraction_conf, 3),
                'search_index': search_stats
            },
            'documents': document_results
        }
    
    def _save_output(self, output: Dict, output_file: str) -> None:
        """Save output to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        file_size = Path(output_file).stat().st_size / 1024  # KB
        logger.success(f"Output saved to {output_file} ({file_size:.2f} KB)")


def main():
    """Main entry point"""
    logger.section("DOCUMENT PROCESSING PIPELINE")
    
    # Create pipeline with offline mode enabled
    pipeline = DocumentPipeline(offline_mode=True)
    
    # Process documents
    output = pipeline.process(
        input_folder="data/input_documents",
        output_file="output.json"
    )
    
    # Display summary
    print(f"\n{'='*80}")
    print("PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total Documents: {output['metadata']['total_documents']}")
    print(f"Successful: {output['metadata']['successful_documents']}")
    print(f"Failed: {output['metadata']['failed_documents']}")
    print(f"\nCategories: {output['summary']['document_categories']}")
    print(f"Avg Classification Confidence: {output['summary']['avg_classification_confidence']:.3f}")
    print(f"Avg Extraction Confidence: {output['summary']['avg_extraction_confidence']:.3f}")
    
    if output['summary']['search_index']:
        print(f"\nSearch Index:")
        print(f"  - Total Chunks: {output['summary']['search_index'].get('total_chunks', 0)}")
        print(f"  - Memory: {output['summary']['search_index'].get('memory_mb', 0):.2f} MB")


if __name__ == "__main__":
    main()
