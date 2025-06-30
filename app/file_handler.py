import os
import PyPDF2
import docx
import pandas as pd
from werkzeug.utils import secure_filename
import magic
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FileHandler:
    ALLOWED_EXTENSIONS = {
        'txt': ['text/plain'],
        'pdf': ['application/pdf'],
        'docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        'doc': ['application/msword'],
        'csv': ['text/csv', 'application/csv'],
        'xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
    }

    def __init__(self, upload_folder='uploads'):
        self.upload_folder = upload_folder
        if not os.path.exists(upload_folder):
            try:
                os.makedirs(upload_folder)
                logger.info(f"Created upload folder: {upload_folder}")
            except Exception as e:
                logger.error(f"Failed to create upload folder: {str(e)}")
                raise

    @staticmethod
    def allowed_file(filename):
        if not filename:
            logger.error("No filename provided")
            return False
            
        logger.debug(f"Checking file: {filename}")
        
        # Get file extension
        if '.' not in filename:
            logger.error(f"No extension found in filename: {filename}")
            return False
            
        ext = filename.rsplit('.', 1)[1].lower()
        logger.debug(f"File extension: {ext}")
        
        # Check if extension is allowed
        if ext not in FileHandler.ALLOWED_EXTENSIONS:
            logger.error(f"Extension {ext} not in allowed extensions: {list(FileHandler.ALLOWED_EXTENSIONS.keys())}")
            return False
            
        logger.debug(f"File {filename} is allowed")
        return True

    @staticmethod
    def get_file_type(file_path):
        try:
            mime_type = magic.from_file(file_path, mime=True)
            logger.debug(f"Detected MIME type: {mime_type}")
            return mime_type
        except Exception as e:
            logger.error(f"Error detecting file type: {str(e)}")
            return None

    @staticmethod
    def extract_text_from_pdf(file_path):
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            logger.debug(f"Successfully extracted text from PDF: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error reading PDF file: {str(e)}")
            raise Exception(f"Error reading PDF file: {str(e)}")

    @staticmethod
    def extract_text_from_docx(file_path):
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.debug(f"Successfully extracted text from DOCX: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX file: {str(e)}")
            raise Exception(f"Error reading DOCX file: {str(e)}")

    @staticmethod
    def extract_text_from_csv(file_path):
        try:
            df = pd.read_csv(file_path)
            text = df.to_string()
            logger.debug(f"Successfully extracted text from CSV: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise Exception(f"Error reading CSV file: {str(e)}")

    @staticmethod
    def extract_text_from_excel(file_path):
        try:
            df = pd.read_excel(file_path)
            text = df.to_string()
            logger.debug(f"Successfully extracted text from Excel: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise Exception(f"Error reading Excel file: {str(e)}")

    @staticmethod
    def read_text_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                logger.debug(f"Successfully read text file: {file_path}")
                return text
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            raise Exception(f"Error reading text file: {str(e)}")

    def process_file(self, file_path):
        try:
            if not os.path.exists(file_path):
                raise Exception(f"File not found: {file_path}")
                
            # Get file extension
            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            logger.debug(f"Processing file with extension: {ext}")
            
            if ext == 'pdf':
                return self.extract_text_from_pdf(file_path)
            elif ext == 'docx':
                return self.extract_text_from_docx(file_path)
            elif ext == 'csv':
                return self.extract_text_from_csv(file_path)
            elif ext == 'xlsx':
                return self.extract_text_from_excel(file_path)
            elif ext == 'txt':
                return self.read_text_file(file_path)
            else:
                raise Exception(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def save_file(self, file):
        """Save uploaded file to the upload folder"""
        if not file:
            logger.error("No file provided for saving")
            return None
        
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(self.upload_folder, filename)
            file.save(file_path)
            logger.debug(f"Successfully saved file: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise

    def read_file(self, file_path):
        """Read content from a file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
                
            text = self.process_file(file_path)
            if not text:
                logger.error(f"No content extracted from file: {file_path}")
                return None
                
            return text
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return None

    def delete_file(self, file_path):
        """Delete a file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Successfully deleted file: {file_path}")
                return True
            logger.warning(f"File not found for deletion: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False 