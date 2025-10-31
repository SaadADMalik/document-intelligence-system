# Setup Guide - Document Intelligence System

## Prerequisites

Before starting, ensure you have:
- **Python 3.11+** installed
- **4GB RAM** minimum
- **2GB disk space** for models
- **Internet connection** (for first-time model download only)

Verify Python installation:
python --version

text

---

##  Step-by-Step Installation

### Step 1: Navigate to Project Directory

cd folder_dir


### Step 2: Create Virtual Environment

python -m venv venv


### Step 3: Activate Virtual Environment

**Windows:**
venv\Scripts\activate

**Mac/Linux:**
source venv/bin/activate

### Step 4: Install Dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm

### Step 5: Verify Installation

python -c "import torch; print(torch.version)"
python -c "import transformers; print(transformers.version)"

---

## Project Structure

Your project should look like this:

local_ai_system/
├── src/
│ ├── classifier.py
│ ├── document_processor.py
│ ├── extractor.py
│ ├── semantic_search.py
│ └── utils.py
├── frontend/
│ └── templates/
│ └── index.html
├── data/
│ └── input_documents/ ← Add your PDFs here
├── logs/
│ └── pipeline.log
├── requirements.txt
├── README.md
├── ANALYSIS.md
├── SETUP_GUIDE.md
├── main.py
├── api.py


---

## Running the Project

### Option 1: Run Full Pipeline (CLI)

python main.py

**Output:**
- Processes all PDFs in `data/input_documents/`
- Generates `output.json` with results
- Creates `logs/pipeline.log` with detailed logs
- Runtime: 20-40 seconds for 18 documents

### Option 2: Start Web UI + API

Terminal 1: Start Flask API
python api.py

Terminal 2: Open browser
start http://localhost:5000


Then:
1. Click **"Process All Documents"**
2. Test **Classify** with sample text
3. Test **Extract** to pull structured data
4. Test **Search** for semantic queries
5. View **Results** in output.json

---

##  Adding Your Documents

1. Place PDF files in: `data/input_documents/`
2. Supported formats: `.pdf`, `.txt`
3. Maximum file size: 50MB each
4. Recommended: 5-100 documents for testing

---

## 🔧 Troubleshooting

### Issue: Model Download Fails

Download spaCy model manually
python -m spacy download en_core_web_sm

Download transformer model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('distilbert-base-uncased')"


### Issue: Port 5000 Already in Use

Edit `api.py` and change port:
app.run(port=5001) # Use different port

Then access: `http://localhost:5001`

### Issue: Out of Memory

Reduce batch size in `src/classifier.py`:
BATCH_SIZE = 4 # Change from 16 to 4

### Issue: DLL Errors (Windows)

Reinstall PyTorch:
pip uninstall torch -y
pip install torch==2.4.1

### Issue: spaCy Model Not Found

python -m spacy download en_core_web_sm

---

## Expected Output

After running `python main.py`, you should see:

================================================================================
DOCUMENT PROCESSING PIPELINE
✓ Processed 20 PDFs
✓ Classified 18 documents (0.69 avg confidence)
✓ Extracted structured data from all documents
✓ Built semantic search index with 18 chunks

Output saved to: output.json
Logs saved to: logs/pipeline.log

text

---

## API Endpoints (When Running api.py)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | Check API status |
| POST | `/api/process` | Run full pipeline |
| POST | `/api/classify` | Classify single document |
| POST | `/api/extract` | Extract data from document |
| POST | `/api/search` | Semantic search query |
| GET | `/api/output` | Get last results |

---

## Next Steps

1. **Explore Results:**
   - View `output.json` for detailed pipeline output
   - Check `logs/pipeline.log` for execution details

2. **Test Features:**
   - Use Web UI to test classification, extraction, search
   - Try different document types

3. **Customize:**
   - Add more regex patterns in `src/extractor.py`
   - Fine-tune thresholds in `src/classifier.py`

4. **Deploy:**
   - Package with Docker
   - Deploy to cloud (AWS, GCP, Azure)

---

## Key Files Explained

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates entire pipeline |
| `api.py` | Flask REST API server |
| `src/classifier.py` | 3-layer classification ensemble |
| `src/extractor.py` | Structured data extraction |
| `src/semantic_search.py` | FAISS vector search |
| `frontend/templates/index.html` | Web UI dashboard |

---

##  Support

For issues:
1. Check `logs/pipeline.log`
2. Verify Python version: `python --version` (need 3.11+)
3. Test installation: `python -c "import torch; import transformers"`
4. Check disk space: `wmic logicaldisk get name,freespace`

---

##  You're All Set!

Your Document Intelligence System is ready to use. Start with:

python main.py


