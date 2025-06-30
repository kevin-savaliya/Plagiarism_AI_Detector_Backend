import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import logging
import numpy as np
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create NLTK data directory if it doesn't exist
nltk_data_dir = Path.home() / 'nltk_data'
nltk_data_dir.mkdir(parents=True, exist_ok=True)

# Add the NLTK data directory to NLTK's search path
nltk.data.path.insert(0, str(nltk_data_dir))

def ensure_nltk_data():
    """Download required NLTK data if not already present"""
    required_packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
            logger.info(f"Found NLTK data: {package}")
        except LookupError:
            try:
                logger.info(f"Downloading NLTK data: {package}")
                nltk.download(package, download_dir=str(nltk_data_dir), quiet=True)
                logger.info(f"Successfully downloaded {package}")
            except Exception as e:
                logger.error(f"Error downloading {package}: {str(e)}")
                raise RuntimeError(f"Failed to download required NLTK data: {package}")

class TextPreprocessor:
    def __init__(self):
        try:
            # First ensure all required NLTK data is available
            ensure_nltk_data()
            
            # Now initialize the components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("TextPreprocessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing TextPreprocessor: {str(e)}")
            raise RuntimeError(f"Failed to initialize TextPreprocessor: {str(e)}")
    
    def clean_text(self, text):
        try:
            if not isinstance(text, str):
                text = str(text)
            # Convert to lowercase
            text = text.lower()
            # Remove special characters but keep periods and basic punctuation
            text = re.sub(r'[^a-zA-Z\s\.,!?]', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            logger.error(f"Error in clean_text: {str(e)}")
            return ""
    
    def tokenize(self, text):
        """Tokenize text into words"""
        try:
            return word_tokenize(text)
        except Exception as e:
            logger.error(f"Error in tokenize: {str(e)}")
            return []
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokenized text"""
        try:
            return [token for token in tokens if token.lower() not in self.stop_words]
        except Exception as e:
            logger.error(f"Error in remove_stopwords: {str(e)}")
            return tokens
    
    def lemmatize(self, tokens):
        """Lemmatize tokens"""
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception as e:
            logger.error(f"Error in lemmatize: {str(e)}")
            return tokens
    
    def preprocess(self, text):
        """Complete text preprocessing pipeline"""
        try:
            if not text or len(text.strip()) == 0:
                raise ValueError("Empty text provided")
                
            # Clean text
            cleaned_text = self.clean_text(text)
            if len(cleaned_text.strip()) == 0:
                logger.warning("Text became empty after cleaning, using original text")
                cleaned_text = text
                
            # Tokenize
            tokens = self.tokenize(cleaned_text)
            if len(tokens) == 0:
                logger.warning("No tokens generated, using simple split")
                tokens = cleaned_text.split()
                
            # Remove stopwords
            tokens = self.remove_stopwords(tokens)
            
            # Lemmatize
            tokens = self.lemmatize(tokens)
            
            # Join tokens back into text
            processed_text = ' '.join(tokens)
            if len(processed_text.strip()) == 0:
                logger.warning("Processing resulted in empty text, using original")
                return text
                
            return processed_text
            
        except Exception as e:
            logger.error(f"Error in preprocess: {str(e)}")
            return text
    
    def get_doc_vectors(self, texts):
        """Convert texts to document vectors using word frequencies"""
        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
                
            # Preprocess all texts
            processed_texts = [self.preprocess(text) for text in texts]
            
            # Create vocabulary
            all_words = set()
            for text in processed_texts:
                all_words.update(text.split())
            vocabulary = list(all_words)
            
            # Create document vectors
            vectors = []
            for text in processed_texts:
                words = text.split()
                vector = np.zeros(len(vocabulary))
                for word in words:
                    if word in vocabulary:
                        vector[vocabulary.index(word)] += 1
                vectors.append(vector)
            
            return np.array(vectors)
            
        except Exception as e:
            logger.error(f"Error in get_doc_vectors: {str(e)}")
            # Return empty vectors if vectorization fails
            return np.zeros((len(texts), 1)) 