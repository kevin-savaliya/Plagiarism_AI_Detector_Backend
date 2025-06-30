import numpy as np
from .preprocessing import TextPreprocessor

class SimilarityAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        
    def calculate_jaccard_similarity(self, text1, text2):
        # Preprocess texts
        preprocessed_text1 = self.preprocessor.preprocess(text1)
        preprocessed_text2 = self.preprocessor.preprocess(text2)
        
        # Convert to sets of words
        set1 = set(preprocessed_text1.split())
        set2 = set(preprocessed_text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
            
        return float(intersection / union)
    
        """
            This measures the intersection over union of word sets. Range: 0 (no overlap) to 1 (identical sets).
            
            Text 1: "artificial intelligence machine learning"
            Text 2: "machine learning deep learning"
            
            Formula: J(A,B) = |A∩B| / |AUB|
            
            Jaccard similarity would be:
            Intersection: {"machine", "learning"} (2 words)
            Union: {"artificial", "intelligence", "machine", "learning", "deep"} (5 words)
            Similarity = 2/5 = 0.4
        """
        
    def calculate_cosine_similarity(self, text1, text2):
        # Preprocess texts
        preprocessed_text1 = self.preprocessor.preprocess(text1)
        preprocessed_text2 = self.preprocessor.preprocess(text2)
        
        # Get word frequencies
        words1 = preprocessed_text1.split()
        words2 = preprocessed_text2.split()
        
        # Create vocabulary
        vocabulary = list(set(words1 + words2))
        
        # Create vectors
        vector1 = np.zeros(len(vocabulary))
        vector2 = np.zeros(len(vocabulary))
        
        # Fill vectors with word frequencies
        for word in words1:
            if word in vocabulary:
                vector1[vocabulary.index(word)] += 1
                
        for word in words2:
            if word in vocabulary:
                vector2[vocabulary.index(word)] += 1
        
        # Calculate cosine similarity
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))
        """
            This measures the cosine of the angle between two text vectors. Range: 0 (different) to 1 (identical).
            
            Text 1: "The cat sat on the mat"
            Text 2: "The cat sat on the rug"
            
            Formula : cos(θ) = (A·B)/(||A||·||B||)
            
            These texts would have high cosine similarity (around 0.8-0.9) because most words are identical.
        """
    
    def calculate_tfidf_similarity(self, text1, text2):
        """Calculate TF-IDF similarity between two texts"""
        # Preprocess texts
        preprocessed_text1 = self.preprocessor.preprocess(text1)
        preprocessed_text2 = self.preprocessor.preprocess(text2)
        
        # Get word frequencies
        words1 = preprocessed_text1.split()
        words2 = preprocessed_text2.split()
        
        # Create vocabulary
        vocabulary = list(set(words1 + words2))
        
        # Calculate term frequencies
        tf1 = np.zeros(len(vocabulary))
        tf2 = np.zeros(len(vocabulary))
        
        for word in words1:
            if word in vocabulary:
                tf1[vocabulary.index(word)] += 1
                
        for word in words2:
            if word in vocabulary:
                tf2[vocabulary.index(word)] += 1
        
        # Calculate inverse document frequency
        idf = np.ones(len(vocabulary))
        for i, word in enumerate(vocabulary):
            if word in words1:
                idf[i] += 1
            if word in words2:
                idf[i] += 1
        idf = np.log(2 / idf)
        
        # Calculate TF-IDF vectors
        tfidf1 = tf1 * idf
        tfidf2 = tf2 * idf
        
        # Calculate cosine similarity
        dot_product = np.dot(tfidf1, tfidf2)
        norm1 = np.linalg.norm(tfidf1)
        norm2 = np.linalg.norm(tfidf2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))
        """
        
            Text 1: "The cat and dog play together in the garden"
            Text 2: "The cat plays with the mouse in the garden"

            Step 1: Preprocessing
                    Convert to lowercase
                    Remove punctuation
                    Split into words
            
            Text 1 words: ["the", "cat", "and", "dog", "play", "together", "in", "the", "garden"]
            Text 2 words: ["the", "cat", "plays", "with", "the", "mouse", "in", "the", "garden"]
            
            Step 2: Create Vocabulary
            
            vocabulary = ["the", "cat", "and", "dog", "play", "together", "in", "garden", "plays", "with", "mouse"]
            
            Step 3: Calculate Term Frequency (TF)
            
            # How many times each word appears in each text

                Text 1 TF:
                - the: 2
                - cat: 1
                - and: 1
                - dog: 1
                - play: 1
                - together: 1
                - in: 1
                - garden: 1
                - plays: 0
                - with: 0
                - mouse: 0

                Text 2 TF:
                - the: 3
                - cat: 1
                - and: 0
                - dog: 0
                - play: 0
                - together: 0
                - in: 1
                - garden: 1
                - plays: 1
                - with: 1
                - mouse: 1
                
            Step 4: Calculate Inverse Document Frequency (IDF)
            
            # IDF = log(total documents / number of documents containing the word)

                IDF values:
                - the: log(2/2) = 0
                - cat: log(2/2) = 0
                - and: log(2/1) = 0.301
                - dog: log(2/1) = 0.301
                - play: log(2/1) = 0.301
                - together: log(2/1) = 0.301
                - in: log(2/2) = 0
                - garden: log(2/2) = 0
                - plays: log(2/1) = 0.301
                - with: log(2/1) = 0.301
                - mouse: log(2/1) = 0.301
                
            Step 5: Calculate TF-IDF Vectors
            
            # Multiply TF * IDF for each term

                Text 1 TF-IDF:
                [0, 0, 0.301, 0.301, 0.301, 0.301, 0, 0, 0, 0, 0]

                Text 2 TF-IDF:
                [0, 0, 0, 0, 0, 0, 0, 0, 0.301, 0.301, 0.301]
                
            Step 6: Calculate Cosine Similarity
            
            similarity = dot_product(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
            
            
            Term Frequency (TF): count(term) / total_terms

            Inverse Document Frequency (IDF): log(N/df)
            
            The similarity score would be around 0.4-0.5 because:
            Common words like "the", "cat", "in", "garden" contribute less (low IDF)
            Unique words like "dog", "play", "mouse" contribute more (high IDF)
            Some word forms are different but related ("play" vs "plays")
            
            These texts would have high cosine similarity (around 0.8-0.9) because most words are identical.
        """
    
    
    def analyze(self, text1, text2):
        """Perform comprehensive similarity analysis"""
        cosine = self.calculate_cosine_similarity(text1, text2)
        jaccard = self.calculate_jaccard_similarity(text1, text2)
        tfidf = self.calculate_tfidf_similarity(text1, text2)
        
        results = {
            'cosine_similarity': cosine,
            'jaccard_similarity': jaccard,
            'tfidf_similarity': tfidf,
            'average_similarity': float((cosine + jaccard + tfidf) / 3)
        }
        
        return results 