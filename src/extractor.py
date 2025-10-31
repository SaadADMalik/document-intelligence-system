"""
Data Extraction Module
Extract structured fields from classified documents with confidence scores
"""
import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import spacy
from src.utils import get_logger

logger = get_logger()


@dataclass
class ExtractedField:
    """Single extracted field with confidence"""
    value: Optional[str]
    confidence: float


@dataclass
class ExtractionResult:
    """Complete extraction result for a document"""
    fields: Dict[str, ExtractedField]
    overall_confidence: float


class DataExtractor:
    """Extract structured data from documents based on type"""
    
    def __init__(self):
        """Initialize extractor with spaCy NER model"""
        logger.info("Initializing DataExtractor...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.success("DataExtractor initialized with spaCy NER")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.nlp = None
    
    def extract(self, text: str, category: str) -> ExtractionResult:
        """
        Extract fields based on document category
        
        Args:
            text: Document text
            category: Document category (Invoice, Resume, Utility Bill, etc.)
            
        Returns:
            ExtractionResult with extracted fields and confidence
        """
        if category == "Invoice":
            return self._extract_invoice(text)
        elif category == "Resume":
            return self._extract_resume(text)
        elif category == "Utility Bill":
            return self._extract_utility_bill(text)
        else:
            return ExtractionResult(fields={}, overall_confidence=0.0)
    
    # ========================================================================
    # INVOICE EXTRACTION
    # ========================================================================
    
    def _extract_invoice(self, text: str) -> ExtractionResult:
        """Extract invoice fields: invoice_number, date, company, total_amount"""
        fields = {}
        
        # Invoice Number
        invoice_num = self._extract_invoice_number(text)
        fields["invoice_number"] = invoice_num
        
        # Date
        date = self._extract_date(text)
        fields["date"] = date
        
        # Company (using spaCy NER)
        company = self._extract_company(text)
        fields["company"] = company
        
        # Total Amount
        total_amount = self._extract_total_amount(text)
        fields["total_amount"] = total_amount
        
        # Calculate overall confidence
        confidences = [f.confidence for f in fields.values() if f.value]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ExtractionResult(fields=fields, overall_confidence=overall_confidence)
    
    def _extract_invoice_number(self, text: str) -> ExtractedField:
        """Extract invoice number using regex patterns"""
        patterns = [
            r'Invoice\s*#\s*(\d+)',
            r'INV[-#]?\s*(\d+)',
            r'Invoice\s+Number[:\s]+(\d+)',
            r'#\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                invoice_num = match.group(1)
                confidence = 0.95 if 'invoice' in pattern.lower() else 0.85
                return ExtractedField(value=invoice_num, confidence=confidence)
        
        return ExtractedField(value=None, confidence=0.0)
    
    def _extract_total_amount(self, text: str) -> ExtractedField:
        """Extract total amount using regex"""
        patterns = [
            r'Total\s+Amount[:\s]+\$?([\d,]+\.?\d*)',
            r'Amount\s+Due[:\s]+\$?([\d,]+\.?\d*)',
            r'Grand\s+Total[:\s]+\$?([\d,]+\.?\d*)',
            r'\$\s*([\d,]+\.\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1).replace(',', '')
                confidence = 0.95 if 'total' in pattern.lower() else 0.80
                return ExtractedField(value=f"${amount}", confidence=confidence)
        
        return ExtractedField(value=None, confidence=0.0)
    
    # ========================================================================
    # RESUME EXTRACTION
    # ========================================================================
    
    def _extract_resume(self, text: str) -> ExtractionResult:
        """Extract resume fields: name, email, phone, experience_years"""
        fields = {}
        
        # Name (using spaCy NER from first 200 chars)
        name = self._extract_name(text)
        fields["name"] = name
        
        # Email
        email = self._extract_email(text)
        fields["email"] = email
        
        # Phone
        phone = self._extract_phone(text)
        fields["phone"] = phone
        
        # Experience Years
        experience = self._extract_experience_years(text)
        fields["experience_years"] = experience
        
        # Calculate overall confidence
        confidences = [f.confidence for f in fields.values() if f.value]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ExtractionResult(fields=fields, overall_confidence=overall_confidence)
    
    def _extract_name(self, text: str) -> ExtractedField:
        """Extract name using spaCy PERSON entity (first 200 chars)"""
        if not self.nlp:
            return ExtractedField(value=None, confidence=0.0)
        
        # Focus on first 200 characters where name usually appears
        text_snippet = text[:200]
        doc = self.nlp(text_snippet)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ExtractedField(value=ent.text, confidence=0.85)
        
        return ExtractedField(value=None, confidence=0.0)
    
    def _extract_email(self, text: str) -> ExtractedField:
        """Extract email using regex"""
        pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        match = re.search(pattern, text)
        
        if match:
            return ExtractedField(value=match.group(0), confidence=0.95)
        
        return ExtractedField(value=None, confidence=0.0)
    
    def _extract_phone(self, text: str) -> ExtractedField:
        """Extract phone number using multiple regex patterns"""
        patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # +1-555-123-4567
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # (555) 123-4567
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # 555-123-4567
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return ExtractedField(value=match.group(0), confidence=0.90)
        
        return ExtractedField(value=None, confidence=0.0)
    
    def _extract_experience_years(self, text: str) -> ExtractedField:
        """Extract years of experience"""
        patterns = [
            r'Experience[:\s]+(\d+)\s*years?',
            r'(\d+)\s*years?\s+(?:of\s+)?experience',
            r'(\d+)\+?\s*years?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                years = match.group(1)
                confidence = 0.90 if 'experience' in pattern.lower() else 0.75
                return ExtractedField(value=years, confidence=confidence)
        
        return ExtractedField(value=None, confidence=0.0)
    
    # ========================================================================
    # UTILITY BILL EXTRACTION
    # ========================================================================
    
    def _extract_utility_bill(self, text: str) -> ExtractionResult:
        """Extract utility bill fields: account_number, billing_date, usage_kwh, amount_due"""
        fields = {}
        
        # Account Number
        account_num = self._extract_account_number(text)
        fields["account_number"] = account_num
        
        # Billing Date
        billing_date = self._extract_date(text, context="billing")
        fields["billing_date"] = billing_date
        
        # Usage (kWh)
        usage = self._extract_usage_kwh(text)
        fields["usage_kwh"] = usage
        
        # Amount Due
        amount_due = self._extract_amount_due(text)
        fields["amount_due"] = amount_due
        
        # Calculate overall confidence
        confidences = [f.confidence for f in fields.values() if f.value]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ExtractionResult(fields=fields, overall_confidence=overall_confidence)
    
    def _extract_account_number(self, text: str) -> ExtractedField:
        """Extract account number"""
        patterns = [
            r'Account\s+Number[:\s]+(ACC-\d+)',
            r'Account[:\s]+(ACC-\d+)',
            r'ACC-(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                acc_num = match.group(1) if 'ACC-' in match.group(0) else f"ACC-{match.group(1)}"
                confidence = 0.95 if 'account' in pattern.lower() else 0.85
                return ExtractedField(value=acc_num, confidence=confidence)
        
        return ExtractedField(value=None, confidence=0.0)
    
    def _extract_usage_kwh(self, text: str) -> ExtractedField:
        """Extract electricity usage in kWh"""
        patterns = [
            r'Usage[:\s]+(\d+)\s*kWh',
            r'(\d+)\s*kWh',
            r'Consumption[:\s]+(\d+)\s*kWh',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                usage = match.group(1)
                confidence = 0.90 if 'usage' in pattern.lower() else 0.80
                return ExtractedField(value=f"{usage} kWh", confidence=confidence)
        
        return ExtractedField(value=None, confidence=0.0)
    
    def _extract_amount_due(self, text: str) -> ExtractedField:
        """Extract amount due"""
        patterns = [
            r'Amount\s+Due[:\s]+\$?([\d,]+\.?\d*)',
            r'Total\s+Due[:\s]+\$?([\d,]+\.?\d*)',
            r'Balance\s+Due[:\s]+\$?([\d,]+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1).replace(',', '')
                return ExtractedField(value=f"${amount}", confidence=0.95)
        
        return ExtractedField(value=None, confidence=0.0)
    
    # ========================================================================
    # SHARED HELPER METHODS
    # ========================================================================
    
    def _extract_date(self, text: str, context: str = None) -> ExtractedField:
        """Extract date using regex and spaCy"""
        # Try regex patterns first
        patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2025-01-15
            r'\d{2}/\d{2}/\d{4}',  # 01/15/2025
            r'\d{2}-\d{2}-\d{4}',  # 01-15-2025
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return ExtractedField(value=match.group(0), confidence=0.90)
        
        # Fallback to spaCy DATE entities
        if self.nlp:
            doc = self.nlp(text[:300])  # First 300 chars
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    return ExtractedField(value=ent.text, confidence=0.75)
        
        return ExtractedField(value=None, confidence=0.0)
    
    def _extract_company(self, text: str) -> ExtractedField:
        """Extract company name from invoice"""
    
    # Simple direct pattern that works for your invoice format
        pattern = r'Company:\s*([^\n]+)'
        match = re.search(pattern, text)
    
        if match:
            company = match.group(1).strip()
            if company and company.lower() != 'n/a':
                return ExtractedField(value=company, confidence=0.95)
    
    # If direct pattern fails, return N/A
        return ExtractedField(value=None, confidence=0.0)


def test_extractor():
    """Test the data extractor"""
    from src.document_processor import DocumentProcessor
    from src.classifier import ThreeLayerClassifier
    
    logger.section("TESTING DATA EXTRACTOR")
    
    # Load and classify documents
    processor = DocumentProcessor()
    documents = processor.process_folder("data/input_documents")
    
    classifier = ThreeLayerClassifier()
    extractor = DataExtractor()
    
    # Test extraction
    print(f"\n{'='*80}")
    print("EXTRACTION RESULTS")
    print(f"{'='*80}\n")
    
    for doc in documents[:5]:  # Test first 5 documents
        if doc.error:
            continue
        
        # Classify
        classification = classifier.classify(doc.text, doc.filename)
        
        # Extract
        extraction = extractor.extract(doc.text, classification.category)
        
        print(f"File: {doc.filename}")
        print(f"Category: {classification.category}")
        print(f"Extraction Confidence: {extraction.overall_confidence:.2f}")
        print("Fields:")
        for field_name, field_data in extraction.fields.items():
            if field_data.value:
                print(f"  - {field_name}: {field_data.value} (conf: {field_data.confidence:.2f})")
        print()


if __name__ == "__main__":
    test_extractor()
