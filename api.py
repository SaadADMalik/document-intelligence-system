"""
Flask REST API for Document Processing Pipeline
"""

import os

# Get absolute path to frontend
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json
from pathlib import Path

from main import DocumentPipeline
from src.utils import get_logger

logger = get_logger()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize pipeline with offline mode
pipeline = DocumentPipeline(offline_mode=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve main page"""
    return send_from_directory('frontend/templates', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.join(FRONTEND_PATH, 'static'), path)

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Document Processing API',
        'version': '1.0',
        'offline_mode': True
    })

@app.route('/api/process', methods=['POST'])
def process_documents():
    """Process documents from folder"""
    try:
        data = request.get_json() or {}
        input_folder = data.get('input_folder', 'data/input_documents')
        
        if not os.path.exists(input_folder):
            return jsonify({
                'error': f'Input folder not found: {input_folder}'
            }), 400
        
        logger.info(f"Processing documents from {input_folder}")
        output = pipeline.process(
            input_folder=input_folder,
            output_file='output.json'
        )
        
        return jsonify(output), 200
    
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload and process PDF files"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                saved_files.append(filename)
                logger.info(f"Uploaded: {filename}")
        
        if not saved_files:
            return jsonify({'error': 'No valid PDF files uploaded'}), 400
        
        output = pipeline.process(
            input_folder=app.config['UPLOAD_FOLDER'],
            output_file='output.json'
        )
        
        # Clean up uploaded files
        for filename in saved_files:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            except:
                pass
        
        return jsonify({
            'status': 'success',
            'files_processed': len(saved_files),
            'results': output
        }), 200
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify_document():
    """Classify a single document"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        result = pipeline.classifier.classify(text)
        
        return jsonify({
            'category': result.category,
            'confidence': result.confidence,
            'method': result.method,
            'layer_scores': result.layer_scores
        }), 200
    
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract', methods=['POST'])
def extract_data():
    """Extract structured data from document"""
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'category' not in data:
            return jsonify({'error': 'Missing text or category field'}), 400
        
        text = data['text']
        category = data['category']
        
        result = pipeline.extractor.extract(text, category)
        
        return jsonify({
            'fields': {
                field: {
                    'value': field_data.value,
                    'confidence': field_data.confidence
                }
                for field, field_data in result.fields.items()
            },
            'overall_confidence': result.overall_confidence
        }), 200
    
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """Semantic search across documents"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
        
        query = data['query']
        top_k = data.get('top_k', 5)
        category_filter = data.get('category_filter')
        
        if not pipeline.search_engine.is_built:
            documents = pipeline.processor.process_folder('data/input_documents')
            indexed_docs = []
            
            for doc in documents:
                if not doc.error:
                    classification = pipeline.classifier.classify(doc.text, doc.filename)
                    indexed_docs.append({
                        'filename': doc.filename,
                        'chunks': doc.chunks,
                        'category': classification.category
                    })
            
            pipeline.search_engine.build_index(indexed_docs)
        
        results = pipeline.search_engine.search(
            query,
            top_k=top_k,
            category_filter=category_filter
        )
        
        return jsonify({
            'query': query,
            'results': [
                {
                    'filename': r.filename,
                    'chunk_text': r.chunk_text,
                    'similarity_score': r.similarity_score,
                    'chunk_index': r.chunk_index
                }
                for r in results
            ]
        }), 200
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/output', methods=['GET'])
def get_output():
    """Get latest pipeline output"""
    try:
        if os.path.exists('output.json'):
            with open('output.json', 'r', encoding='utf-8') as f:
                output = json.load(f)
            return jsonify(output), 200
        else:
            return jsonify({'error': 'output.json not found. Run /api/process first'}), 404
    
    except Exception as e:
        logger.error(f"Output retrieval error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.section("STARTING DOCUMENT PROCESSING API")
    logger.info("=" * 60)
    logger.info("Flask API Server Starting...")
    logger.info("=" * 60)
    logger.info("\n📍 Web UI: http://localhost:5000")
    logger.info("\n🔗 API Endpoints:")
    logger.info("  GET  /api/health")
    logger.info("  POST /api/process")
    logger.info("  POST /api/upload")
    logger.info("  POST /api/classify")
    logger.info("  POST /api/extract")
    logger.info("  POST /api/search")
    logger.info("  GET  /api/output")
    logger.info("\n⚙️  Offline Mode: ENABLED")
    logger.info("\n" + "=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
