from flask import Flask, request, jsonify
from flask_cors import CORS
from app.similarity import SimilarityAnalyzer
from app.ai_detection import AIDetector
from app.file_handler import FileHandler
import os
import logging
from dotenv import load_dotenv
import json
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Configure CORS to allow requests from your frontend
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "https://plagiarism-ai-detector-frontend-164h2zfle.vercel.app", "https://plagiarism-ai-detector-frontend.vercel.app", "https://*.vercel.app",],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# File to store reports
REPORTS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'reports.json')

# Create data directory if it doesn't exist
os.makedirs(os.path.dirname(REPORTS_FILE), exist_ok=True)

# Thread lock for file operations
file_lock = threading.Lock()

# Initialize handlers
file_handler = FileHandler()
similarity_analyzer = SimilarityAnalyzer()
ai_detector = AIDetector()

def load_reports():
    try:
        with file_lock:
            if os.path.exists(REPORTS_FILE):
                with open(REPORTS_FILE, 'r') as f:
                    return json.load(f)
            return []
    except Exception as e:
        logger.error(f"Error loading reports: {str(e)}")
        return []

def save_report(report):
    try:
        reports = load_reports()
        with file_lock:
            # Generate ID if not present
            if not reports:
                report['id'] = 1
            else:
                report['id'] = max(r['id'] for r in reports) + 1
            
            report['date'] = datetime.now().isoformat()
            reports.append(report)
            
            with open(REPORTS_FILE, 'w') as f:
                json.dump(reports, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        raise

@app.route('/api/reports', methods=['GET'])
def get_reports():
    try:
        reports = load_reports()
        return jsonify(reports)
    except Exception as e:
        logger.error(f"Error retrieving reports: {str(e)}")
        return jsonify({'error': 'Failed to retrieve reports'}), 500

@app.route('/api/detect-ai', methods=['POST'])
def detect_ai():
    try:
        # Check if request contains a file
        if 'file' in request.files:
            file = request.files['file']
            if not file:
                logger.error("No file provided in request")
                return jsonify({'error': 'No file provided'}), 400
                
            if not file_handler.allowed_file(file.filename):
                logger.error(f"Invalid file type: {file.filename}")
                return jsonify({'error': 'Invalid file type'}), 400
            
            try:
                # Save the uploaded file
                file_path = file_handler.save_file(file)
                if not file_path:
                    logger.error("Failed to save file")
                    return jsonify({'error': 'Failed to save file'}), 500
                
                # Extract text from the file
                text = file_handler.read_file(file_path)
                if not text:
                    logger.error("Failed to read file content")
                    return jsonify({'error': 'Failed to read file content'}), 500
                
                # Analyze the text
                result = ai_detector.analyze_text(text)
                if not result:
                    logger.error("Failed to analyze text")
                    return jsonify({'error': 'Failed to analyze text'}), 500
                
                # Save the report
                report = {
                    'type': 'ai_detection',
                    'text': text,
                    'result': result
                }
                save_report(report)
                
                # Clean up the file
                file_handler.delete_file(file_path)
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                return jsonify({'error': f'Error processing file: {str(e)}'}), 500
                
        # Check if request contains text
        elif request.is_json:
            data = request.get_json()
            if not data:
                logger.error("No JSON data in request")
                return jsonify({'error': 'Invalid request format'}), 400
                
            text = data.get('text')
            if not text:
                logger.error("No text provided in request")
                return jsonify({'error': 'No text provided'}), 400
            
            try:
                result = ai_detector.analyze_text(text)
                if not result:
                    logger.error("Failed to analyze text")
                    return jsonify({'error': 'Failed to analyze text'}), 500
                
                # Save the report
                report = {
                    'type': 'ai_detection',
                    'text': text,
                    'result': result
                }
                save_report(report)
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error analyzing text: {str(e)}")
                return jsonify({'error': f'Error analyzing text: {str(e)}'}), 500
                
        else:
            logger.error("Invalid request format")
            return jsonify({'error': 'Invalid request format'}), 400
            
    except Exception as e:
        logger.error(f"Error in AI detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-similarity', methods=['POST'])
def analyze_similarity():
    try:
        if 'file1' in request.files and 'file2' in request.files:
            file1 = request.files['file1']
            file2 = request.files['file2']
            
            if not (file1 and file2 and 
                   file_handler.allowed_file(file1.filename) and 
                   file_handler.allowed_file(file2.filename)):
                return jsonify({'error': 'Invalid file(s)'}), 400
            
            # Save and read files
            file1_path = file_handler.save_file(file1)
            file2_path = file_handler.save_file(file2)
            
            text1 = file_handler.read_file(file1_path)
            text2 = file_handler.read_file(file2_path)
            
            if not (text1 and text2):
                return jsonify({'error': 'Failed to read file content'}), 500
            
            # Analyze similarity
            result = similarity_analyzer.analyze(text1, text2)
            
            # Save the report
            report = {
                'type': 'similarity_analysis',
                'text1': text1,
                'text2': text2,
                'result': result
            }
            save_report(report)
            
            # Clean up files
            file_handler.delete_file(file1_path)
            file_handler.delete_file(file2_path)
            
            return jsonify(result)
        else:
            data = request.json
            text1 = data.get('text1')
            text2 = data.get('text2')
            
            if not (text1 and text2):
                return jsonify({'error': 'Both texts are required'}), 400
            
            result = similarity_analyzer.analyze(text1, text2)
            
            # Save the report
            report = {
                'type': 'similarity_analysis',
                'text1': text1,
                'text2': text2,
                'result': result
            }
            save_report(report)
            
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error in similarity analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>AI Detector System API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f0f2f5;
                }
                .container {
                    text-align: center;
                    padding: 40px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #1a73e8;
                    margin-bottom: 20px;
                }
                p {
                    color: #5f6368;
                    font-size: 18px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome to AI Detector System</h1>
                <p>Server is running successfully!</p>
            </div>
        </body>
    </html>
    """

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Not Found",
        "message": "The requested resource was not found on this server",
        "status_code": 404
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }), 500

if __name__ == '__main__':
    app.run(debug=True)