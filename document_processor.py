import os
import requests
import tempfile
from typing import Optional
import logging
from PyPDF2 import PdfReader
from docx import Document
import re

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx']
    
    def download_document(self, url: str) -> str:
        """Download document from URL and save to temporary file"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Get file extension from URL or content type
            ext = os.path.splitext(url)[1].lower()
            if not ext:
                content_type = response.headers.get('content-type', '')
                if 'pdf' in content_type:
                    ext = '.pdf'
                elif 'word' in content_type or 'docx' in content_type:
                    ext = '.docx'
                else:
                    ext = '.pdf'  # default assumption
            
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        except Exception as e:
            logger.error(f"Error downloading document: {str(e)}")
            raise
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error reading DOCX: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        return text
    
    def process_document(self, document_url: str) -> str:
        """Process document from URL and return extracted text"""
        try:
            # Download document
            temp_file = self.download_document(document_url)
            
            # Process based on file type
            if temp_file.endswith('.pdf'):
                text = self.extract_text_from_pdf(temp_file)
            elif temp_file.endswith('.docx'):
                text = self.extract_text_from_docx(temp_file)
            else:
                raise ValueError(f"Unsupported file format: {temp_file}")
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
            
            return text
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise