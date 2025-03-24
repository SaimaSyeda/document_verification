# app/routes/api.py (simplified)
import os
import uuid
from flask import Blueprint, request, jsonify, render_template, current_app
import re
from werkzeug.utils import secure_filename

api = Blueprint('api', __name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@api.route('/')
def index():
    return render_template('index.html')

@api.route('/api/verify', methods=['POST'])
def verify_document():
    # Check if file was uploaded
    if 'document' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['document']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # For demo purposes, use filename to determine document type
        doc_type = "unknown"
        if "land" in file.filename.lower():
            doc_type = "land_record"
        elif "caste" in file.filename.lower():
            doc_type = "caste_certificate"
        elif "property" in file.filename.lower():
            doc_type = "property_registration"
        
        # Process file (simplified for demo)
        result = {
            'document_type': doc_type,
            'document_type_confidence': 0.85,
            'is_authentic': True,
            'extracted_data': {
                'document_id': 'SAMPLE-' + str(uuid.uuid4())[:8],
                'name': 'Sample Name',
                'date': '2023-01-01'
            },
            'verification_status': 'verified'
        }
        
        # Clean up file
        os.remove(filepath)
        
        return jsonify(result)
    
    return jsonify({'error': 'File type not allowed'}), 400