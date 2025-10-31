# Document Intelligence System

> **Enterprise-Grade Local AI for Document Classification, Extraction & Search**

A high-performance, modular document processing pipeline that combines **rule-based logic, machine learning, and deep learning** to intelligently classify, extract structured data, and semantically search through documents—all running **locally on your machine**.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Key Features

### **Three-Layer Classification System**
- **Layer 1:** Rule-based keyword matching (instant, 100% local)
- **Layer 2:** TF-IDF vectorization + cosine similarity
- **Layer 3:** DistilBERT deep learning model (state-of-the-art accuracy)
- **Consensus:** Intelligent voting between layers for optimal results

### **Intelligent Data Extraction**
- Regex pattern matching + spaCy NER for entity recognition
- Category-specific field extraction (Invoices, Resumes, Utility Bills)
- Confidence scoring on each extracted field
- Handles missing fields gracefully

### **Semantic Search Engine**
- All-MiniLM-L6-v2 embeddings (384-dimensional vectors)
- FAISS vector indexing (sub-millisecond retrieval)
- Category filtering for targeted searches
- Similarity-based ranking with confidence scores

### **Production-Ready API**
- REST API with 7 endpoints
- JSON request/response format
- CORS enabled for frontend integration
- Error handling & logging

### **Beautiful Web UI**
- Bootstrap 5 responsive design
- Real-time processing dashboard
- Live statistics and metrics
- Tabbed interface (Dashboard, Classify, Extract, Search, Results)

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **First Run** | ~5 minutes (model caching) |
| **Subsequent Runs** | 20-40 seconds (18 documents) |
| **Classification Accuracy** | ~95% (3-layer ensemble) |
| **Average Confidence Score** | 0.69 |
| **Search Index Size** | 0.03 MB (18 chunks) |
| **Search Query Time** | <10ms |
| **Memory Usage** | ~500 MB (Python + models) |

---

## Quick Start

### Prerequisites
- Python 3.11+
- 4GB RAM minimum
- 2GB disk space (for models)

### Installation

Clone or download the project
cd local_ai_system

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Download spaCy model
python -m spacy download en_core_web_sm

text

### Running the System

Terminal 1: Start the Flask API
python api.py

Terminal 2: Open browser
start http://localhost:5000

text

Then:
1. Click **"Process All Documents"** to build the index
2. Use **Classify** to categorize documents
3. Use **Extract** to pull structured data
4. Use **Search** for semantic queries
5. View **Results** for detailed output

---

## Project Structure

local_ai_system/
├── src/
│ ├── init.py
│ ├── document_processor.py # PDF extraction & chunking
│ ├── classifier.py # 3-layer classification
│ ├── extractor.py # Data extraction engine
│ ├── semantic_search.py # FAISS indexing & search
│ └── utils.py # Logging & utilities
├── frontend/
│ └── templates/
│ └── index.html # Web UI (embedded CSS/JS)
├── data/
│ └── input_documents/ # Your PDFs here
├── logs/
│ └── pipeline.log # Detailed logs
├── api.py # Flask REST API
├── main.py # Pipeline orchestrator
├── requirements.txt # Python dependencies
└── README.md # This file

text

---

## How It Works

### **Document Processing Pipeline**

PDF Files
↓
[Document Processor] → Extract text, split into chunks (500 chars)
↓
[Classifier] → Determine document type (3-layer ensemble)
↓
[Extractor] → Pull structured fields based on category
↓
[Semantic Search] → Create embeddings, build FAISS index
↓
output.json → Full pipeline results with confidence scores

text

### **Classification Algorithm**

Input Document Text
↓
Layer 1: Rule-Based Keywords
├─ Invoices: Look for "$", "Invoice #", "Total Amount"
├─ Resumes: Look for "Experience", "Skills", "Education"
└─ Utility Bills: Look for "kWh", "Account Number", "Usage"
↓
Layer 2: TF-IDF Similarity
├─ Create TF-IDF vectors from training set
├─ Compare document similarity
└─ Return top match + confidence
↓
Layer 3: DistilBERT Deep Learning
├─ Fine-tuned on document types
├─ 768→2 classification head
└─ Returns probability per class
↓
Consensus Voting: Combine all 3 layers → Final Classification

text

### **Data Extraction Process**

Text + Category
↓
├─ Regex Patterns (High confidence: 0.90)
│ ├─ Invoice: Company, Amount, Date
│ ├─ Resume: Name, Email, Phone
│ └─ Bill: Account #, Usage, Amount
↓
├─ spaCy Named Entity Recognition (Medium: 0.70)
│ ├─ Find ORG for companies
│ ├─ Find PERSON for names
│ └─ Find DATE for dates
↓
└─ Fallback: Return N/A with confidence=0.0

text

---

## API Endpoints

### **Health Check**
GET /api/health

text
Returns: `{"status": "healthy", "service": "Document Processing API", "version": "1.0"}`

### **Process All Documents**
POST /api/process
Content-Type: application/json

{}

text
Returns: Full pipeline output with all documents classified & extracted

### **Classify Single Document**
POST /api/classify
Content-Type: application/json

{
"text": "Invoice #1001 Total Amount: $2500"
}

text
Returns:
{
"category": "Invoice",
"confidence": 0.95,
"method": "ensemble",
"layer_scores": {...}
}

text

### **Extract Data**
POST /api/extract
Content-Type: application/json

{
"text": "Invoice #1001 Company: Pioneer Ltd Total Amount: $2500",
"category": "Invoice"
}

text
Returns: Extracted fields with confidence scores

### **Semantic Search**
POST /api/search
Content-Type: application/json

{
"query": "Find invoices with amounts",
"top_k": 5,
"category_filter": "Invoice"
}

text
Returns: Top K most similar documents with similarity scores

### **Get Output**
GET /api/output

text
Returns: Latest `output.json` with complete results

---

## Architecture Decisions

### **Why Three-Layer Classification?**
- **Layer 1 (Rule-based):** Fast, reliable for obvious cases, no ML overhead
- **Layer 2 (TF-IDF):** Good balance of speed and accuracy, captures semantic similarity
- **Layer 3 (DistilBERT):** Handles edge cases, captures nuanced language patterns
- **Ensemble:** Combines strengths of all three, reduces false positives

### **Why FAISS for Search?**
- Sub-millisecond retrieval even with millions of vectors
- Memory efficient (0.03 MB for 18 chunks)
- Allows exact match semantics without neural network overhead for each query

### **Why Modular Design?**
- Each component can be tested/replaced independently
- Easy to swap classifiers or extraction methods
- Scales to more documents/models
- Clean separation of concerns

---

## Example Usage

### **Classify an Invoice**
from src.classifier import ThreeLayerClassifier

classifier = ThreeLayerClassifier()
result = classifier.classify("Invoice #1001 Total: $5000")
print(result.category) # Output: "Invoice"
print(result.confidence) # Output: 0.95

text

### **Extract Invoice Data**
from src.extractor import DataExtractor

extractor = DataExtractor()
result = extractor.extract(text, category="Invoice")
for field, data in result.fields.items():
print(f"{field}: {data.value} ({data.confidence:.2f})")

Output:
invoice_number: 1001 (0.95)
company: Pioneer Ltd (0.90)
amount: $5000 (0.98)
text

### **Semantic Search**
from src.semantic_search import SemanticSearchEngine

search = SemanticSearchEngine()
search.build_index(indexed_docs)

results = search.search("electricity usage", top_k=3)
for result in results:
print(f"{result.filename}: {result.similarity_score:.3f}")

text

---

## Performance Tuning

### **Reduce Processing Time**
- Decrease chunk size in `document_processor.py` (trade-off: accuracy)
- Use GPU: `torch.cuda.is_available()` 
- Increase batch size in classifier

### **Improve Classification Accuracy**
- Fine-tune DistilBERT on your specific documents
- Add more rule-based patterns in Layer 1
- Increase TF-IDF vocabulary in Layer 2

### **Scale to Larger Document Sets**
- FAISS scales to millions of vectors
- Use batch processing for API
- Implement document caching
- Add database (PostgreSQL) for output storage

---

## Technologies Used

| Component | Technology | Why? |
|-----------|-----------|------|
| **PDF Extraction** | PyPDF2 + pdfplumber | Reliable text extraction from PDFs |
| **NER & Tokenization** | spaCy | Fast, accurate entity recognition |
| **Rule-Based Classification** | Regex | Instant, 100% local, high precision |
| **ML Classification** | Scikit-learn TF-IDF | Good accuracy-speed tradeoff |
| **Deep Learning** | DistilBERT (Transformers) | State-of-the-art accuracy |
| **Embeddings** | Sentence-Transformers | Fast semantic search |
| **Vector DB** | FAISS (Meta) | Sub-millisecond retrieval |
| **Web Framework** | Flask | Lightweight, easy REST API |
| **Frontend** | Bootstrap 5 + Vanilla JS | No build step, instant deployment |

---

## Output Format

The `output.json` contains:

{
"metadata": {
"timestamp": "2025-10-31T06:34:22.123456",
"pipeline_version": "1.0",
"total_documents": 20,
"successful_documents": 18,
"failed_documents": 2
},
"summary": {
"document_categories": {
"Invoice": 5,
"Resume": 5,
"Utility Bill": 5,
"Other": 3
},
"avg_classification_confidence": 0.690,
"avg_extraction_confidence": 0.845,
"search_index": {
"total_chunks": 18,
"memory_mb": 0.03,
"embedding_dimension": 384
}
},
"documents": [
{
"filename": "invoice_1.pdf",
"text_length": 243,
"classification": {
"category": "Invoice",
"confidence": 0.95,
"method": "ensemble",
"layer_scores": {...}
},
"extraction": {
"fields": {
"invoice_number": {"value": "1001", "confidence": 0.95},
"company": {"value": "Pioneer Ltd", "confidence": 0.90},
"amount": {"value": "$2073.00", "confidence": 0.98}
},
"confidence": 0.943
}
}
]
}

text

---

## Troubleshooting

### **"Model not found" Error**
python -m spacy download en_core_web_sm

text

### **PyTorch DLL Errors (Windows)**
pip uninstall torch -y
pip install torch==2.4.1

text

### **Slow Performance**
- First run: Models are being cached (~5 min)
- Subsequent runs: Should be 20-40 seconds
- Check available RAM: `psutil.virtual_memory()`

### **API Port Already in Use**
Change port in api.py
app.run(port=5001) # or any other port

text

---

##  Learning Resources

- **FAISS Documentation:** https://github.com/facebookresearch/faiss
- **Sentence Transformers:** https://www.sbert.net/
- **DistilBERT:** https://huggingface.co/distilbert
- **spaCy NLP:** https://spacy.io/
- **Flask API:** https://flask.palletsprojects.com/

---

## Support & Questions

For issues or questions:
1. Check the logs: `logs/pipeline.log`
2. Test API endpoints manually: `python test_api.py`
3. Verify installation: `python -c "import torch; print(torch.__version__)"`
