import re
import logging
from collections import Counter
from statistics import mean, stdev

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDetector:
    def __init__(self):
        try:
            # Basic AI patterns
            self.ai_patterns = [
                r'\b(however|moreover|furthermore|therefore|thus|consequently)\b',
                r'\b(in conclusion|to summarize|in summary|ultimately)\b',
                r'\b(it is worth noting|it should be noted|research suggests)\b',
                r'\b(analysis|research|study|data|results|methodology)\b',
                r'\b(firstly|secondly|finally|in addition|furthermore)\b'
            ]
            
            # Basic human patterns
            self.human_patterns = [
                r'\b(i think|i feel|in my opinion|i believe)\b',
                r'\b(kind of|sort of|basically|literally|actually)\b',
                r'\b(like|you know|well|anyway|honestly)\b',
                r'\b(maybe|probably|possibly|perhaps|seems)\b',
                r'\b(gonna|wanna|gotta|kinda|sorta)\b'
            ]
            
            logger.info("AIDetector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AIDetector: {str(e)}")
            raise

    def analyze_text(self, text):
        try:
            # Input validation
            if not isinstance(text, str):
                text = str(text)
            
            if not text or len(text.strip()) == 0:
                return {
                    'ai_probability': 0.0,
                    'is_ai_generated': False,
                    'confidence': 0.0,
                    'message': 'Empty text provided',
                    'details': {
                        'pattern_score': 0.0,
                        'structure_score': 0.0,
                        'style_score': 0.0
                    }
                }
            
            # Get pattern score
            pattern_score = self._analyze_patterns(text)
            
            # Get structure score
            structure_score = self._analyze_structure(text)
            
            # Get style score
            style_score = self._analyze_style(text)
            
            # Calculate final score (weighted average)
            final_score = (pattern_score * 0.4) + (structure_score * 0.3) + (style_score * 0.3)
            
            # Convert to percentage
            ai_probability = round(final_score * 100, 1)
            ai_probability = max(0.0, min(100.0, ai_probability))
            
            # Calculate confidence
            confidence = self._calculate_confidence([pattern_score, structure_score, style_score])
            
            # Determine if AI-generated
            is_ai_generated = ai_probability > 50.0
            
            # Prepare response
            return {
                'ai_probability': ai_probability,
                'is_ai_generated': is_ai_generated,
                'confidence': confidence,
                'message': self._generate_message(ai_probability, confidence),
                'details': {
                    'pattern_score': round(pattern_score * 100, 1),
                    'structure_score': round(structure_score * 100, 1),
                    'style_score': round(style_score * 100, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {
                'error': str(e),
                'ai_probability': 50.0,
                'is_ai_generated': False,
                'confidence': 0.0,
                'message': "Unable to determine",
                'details': {
                    'pattern_score': 50.0,
                    'structure_score': 50.0,
                    'style_score': 50.0
                }
            }

    def _analyze_patterns(self, text):
        try:
            text = text.lower()
            ai_matches = sum(len(re.findall(pattern, text)) for pattern in self.ai_patterns)
            human_matches = sum(len(re.findall(pattern, text)) for pattern in self.human_patterns)
            
            total_matches = ai_matches + human_matches
            if total_matches == 0:
                return 0.5
                
            score = ai_matches / total_matches
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return 0.5

    def _analyze_structure(self, text):
        try:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            words = [w.lower() for w in re.findall(r'\b\w+\b', text.lower())]
            
            if not sentences or not words:
                return 0.5
                
            # Calculate average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Score based on sentence length (AI tends to have longer sentences)
            if avg_sentence_length < 10:
                return 0.3
            elif avg_sentence_length < 20:
                return 0.5
            else:
                return 0.7
                
        except Exception as e:
            logger.error(f"Error in structure analysis: {str(e)}")
            return 0.5

    def _analyze_style(self, text):
        """Analyze text style"""
        try:
            words = [w.lower() for w in re.findall(r'\b\w+\b', text.lower())]
            if not words:
                return 0.5
                
            # Calculate word diversity
            unique_words = len(set(words))
            word_diversity = unique_words / len(words)
            
            # Score based on word diversity (AI tends to have higher diversity)
            if word_diversity < 0.4:
                return 0.3
            elif word_diversity < 0.6:
                return 0.5
            else:
                return 0.7
                
        except Exception as e:
            logger.error(f"Error in style analysis: {str(e)}")
            return 0.5

    def _calculate_confidence(self, scores):
        try:
            if not scores:
                return 0.0
                
            # Calculate standard deviation
            if len(scores) > 1:
                std_dev = stdev(scores)
                # Lower standard deviation means higher confidence
                confidence = 100 - (std_dev * 50)
                return max(0.0, min(100.0, confidence))
            else:
                return 50.0
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 50.0

    def _generate_message(self, probability, confidence):
        try:
            if confidence < 50:
                return "Low confidence in analysis"
                
            if probability < 30:
                return "Likely human-written"
            elif probability < 70:
                return "Uncertain - could be either human or AI"
            else:
                return "Likely AI-generated"
                
        except Exception as e:
            logger.error(f"Error generating message: {str(e)}")
            return "Unable to determine" 