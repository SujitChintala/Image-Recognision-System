"""
Flask API for Image Recognition
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from image_recognition import ImageRecognition

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the model
print("Loading model...")
recognizer = ImageRecognition(model_name='alexnet')
print("Model loaded!")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, webp'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        results = recognizer.predict(filepath, top_k=5)
        
        # Clean up - delete the uploaded file
        os.remove(filepath)
        
        # Format results
        predictions = [
            {
                'class': class_name,
                'confidence': round(confidence, 2)
            }
            for class_name, confidence in results
        ]
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_prediction': {
                'class': predictions[0]['class'],
                'confidence': predictions[0]['confidence']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
