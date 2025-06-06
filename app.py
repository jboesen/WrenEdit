import os
import logging
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import uuid
from pathlib import Path
import threading
import json
from video_processor import VideoProcessor

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
UPLOAD_FOLDER = Path('uploads')
PROCESSED_FOLDER = Path('processed')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
PROCESSED_FOLDER.mkdir(exist_ok=True)

# Store active processing sessions
processing_sessions = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported formats: MP4, MOV, AVI, MKV, WEBM'}), 400
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        file_extension = filename.rsplit('.', 1)[1].lower()
        upload_path = UPLOAD_FOLDER / f"{file_id}.{file_extension}"
        
        file.save(upload_path)
        
        # Store session info
        processing_sessions[session_id] = {
            'upload_path': upload_path,
            'filename': filename,
            'status': 'uploaded',
            'progress': 0,
            'current_step': 'Uploaded',
            'metadata': {}
        }
        
        return jsonify({
            'session_id': session_id,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_video():
    session_id = session.get('session_id')
    if not session_id or session_id not in processing_sessions:
        return jsonify({'error': 'No valid session found'}), 400
    
    try:
        # Start processing in background
        thread = threading.Thread(
            target=process_video_background,
            args=(session_id,)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Processing started'})
        
    except Exception as e:
        logging.error(f"Process start error: {str(e)}")
        return jsonify({'error': f'Failed to start processing: {str(e)}'}), 500

def process_video_background(session_id):
    """Background video processing with progress updates"""
    try:
        session_data = processing_sessions[session_id]
        upload_path = session_data['upload_path']
        
        # Create output path
        output_filename = f"processed_{uuid.uuid4().hex}.mp4"
        output_path = PROCESSED_FOLDER / output_filename
        
        # Initialize video processor with callback
        def progress_callback(step, progress, metadata=None):
            session_data['current_step'] = step
            session_data['progress'] = progress
            if metadata:
                session_data['metadata'].update(metadata)
            
            socketio.emit('processing_update', {
                'session_id': session_id,
                'step': step,
                'progress': progress,
                'metadata': session_data['metadata']
            })
        
        processor = VideoProcessor(progress_callback)
        
        # Process video
        processor.process(str(upload_path), str(output_path))
        
        # Update session with completion
        session_data['status'] = 'completed'
        session_data['output_path'] = output_path
        session_data['output_filename'] = output_filename
        session_data['progress'] = 100
        session_data['current_step'] = 'Completed'
        
        socketio.emit('processing_complete', {
            'session_id': session_id,
            'output_filename': output_filename,
            'metadata': session_data['metadata']
        })
        
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        session_data = processing_sessions.get(session_id, {})
        session_data['status'] = 'error'
        session_data['error'] = str(e)
        
        socketio.emit('processing_error', {
            'session_id': session_id,
            'error': str(e)
        })

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = PROCESSED_FOLDER / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"edited_{filename}"
        )
        
    except Exception as e:
        logging.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/preview/<path:filename>')
def preview_file(filename):
    """Serve files for preview"""
    try:
        # Check if it's an upload or processed file
        upload_path = UPLOAD_FOLDER / filename
        processed_path = PROCESSED_FOLDER / filename
        
        if upload_path.exists():
            return send_file(upload_path)
        elif processed_path.exists():
            return send_file(processed_path)
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        logging.error(f"Preview error: {str(e)}")
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500

@app.route('/status')
def get_status():
    session_id = session.get('session_id')
    if not session_id or session_id not in processing_sessions:
        return jsonify({'error': 'No valid session found'}), 400
    
    session_data = processing_sessions[session_id]
    return jsonify({
        'status': session_data.get('status', 'unknown'),
        'progress': session_data.get('progress', 0),
        'current_step': session_data.get('current_step', ''),
        'metadata': session_data.get('metadata', {}),
        'error': session_data.get('error')
    })

@socketio.on('connect')
def handle_connect():
    emit('connected', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
