"""
Flask-based web interface for ebook2audiobook
Replaces the Gradio interface while maintaining all functionality
"""

import os
import json
import uuid
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room
from werkzeug.utils import secure_filename

from lib import *
from lib.functions import (
    SessionContext, convert_ebook, convert_ebook_batch,
    extract_custom_model, prepare_dirs, show_alert, 
    get_all_ip_addresses
)


class FlaskInterface:
    def __init__(self, args, ctx):
        self.app = Flask(__name__, 
                        template_folder='../templates',
                        static_folder='../static')
        self.app.secret_key = 'ebook2audiobook_' + str(uuid.uuid4())
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.args = args
        self.ctx = ctx
        self.script_mode = args['script_mode']
        self.is_gui_process = args['is_gui_process']
        self.is_gui_shared = args['share']
        
        # Thread management
        self.conversion_threads = {}
        self.log_buffers = {}
        
        # Setup routes
        self.setup_routes()
        self.setup_socket_handlers()
        
    def setup_routes(self):
        """Setup all Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main interface page"""
            try:
                return render_template('index.html', 
                                     version=prog_version,
                                     language_options=self.get_language_options(),
                                     tts_engine_options=self.get_tts_engine_options(),
                                     device_options=self.get_device_options(),
                                     output_format_options=self.get_output_format_options())
            except Exception as e:
                return f"Error loading interface: {e}", 500
        
        @self.app.route('/upload/ebook', methods=['POST'])
        def upload_ebook():
            """Handle ebook file upload"""
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not self.allowed_ebook_file(file.filename):
                return jsonify({'error': 'Invalid file format'}), 400
            
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(tmp_dir, 'uploads', filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                file.save(filepath)
                
                session['ebook_path'] = filepath
                session['ebook_filename'] = filename
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'path': filepath
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/upload/voice', methods=['POST'])
        def upload_voice():
            """Handle voice file upload"""
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not self.allowed_voice_file(file.filename):
                return jsonify({'error': 'Invalid file format'}), 400
            
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(tmp_dir, 'voices', filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                file.save(filepath)
                
                session['voice_path'] = filepath
                session['voice_filename'] = filename
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'path': filepath
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/upload/custom_model', methods=['POST'])
        def upload_custom_model():
            """Handle custom model upload"""
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not file.filename.lower().endswith('.zip'):
                return jsonify({'error': 'Only ZIP files are allowed'}), 400
            
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(tmp_dir, 'models', filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                file.save(filepath)
                
                # Extract and validate the model
                model_info = extract_custom_model(filepath, self.ctx)
                
                session['custom_model_path'] = filepath
                session['custom_model_filename'] = filename
                session['custom_model_info'] = model_info
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'path': filepath,
                    'model_info': model_info
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/convert', methods=['POST'])
        def start_conversion():
            """Start the ebook conversion process"""
            try:
                # Get conversion parameters from form
                params = self.get_conversion_params()
                
                # Create a unique session ID for this conversion
                conversion_id = str(uuid.uuid4())
                
                # Start conversion in a separate thread
                thread = threading.Thread(
                    target=self.run_conversion,
                    args=(conversion_id, params)
                )
                thread.daemon = True
                thread.start()
                
                self.conversion_threads[conversion_id] = thread
                
                return jsonify({
                    'success': True,
                    'conversion_id': conversion_id
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/download/<conversion_id>')
        def download_audiobook(conversion_id):
            """Download the generated audiobook"""
            try:
                # Get the output file path from session or database
                output_path = session.get(f'output_path_{conversion_id}')
                if not output_path or not os.path.exists(output_path):
                    return jsonify({'error': 'File not found'}), 404
                
                return send_file(output_path, as_attachment=True)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/preview/voice')
        def preview_voice():
            """Preview uploaded voice file"""
            voice_path = session.get('voice_path')
            if not voice_path or not os.path.exists(voice_path):
                return jsonify({'error': 'Voice file not found'}), 404
            
            return send_file(voice_path)
        
        @self.app.route('/voices/list')
        def list_voices():
            """List available voices"""
            voices = []
            voices_directory = os.path.join(os.getcwd(), 'voices')
            if os.path.exists(voices_directory):
                for file in os.listdir(voices_directory):
                    if any(file.lower().endswith(ext) for ext in voice_formats):
                        voices.append({
                            'name': file,
                            'path': os.path.join(voices_directory, file)
                        })
            return jsonify({'voices': voices})
        
        @self.app.route('/models/list')
        def list_models():
            """List available custom models"""
            models = []
            models_directory = os.path.join(os.getcwd(), 'models')
            if os.path.exists(models_directory):
                for file in os.listdir(models_directory):
                    if file.endswith('.zip'):
                        models.append({
                            'name': file,
                            'path': os.path.join(models_directory, file)
                        })
            return jsonify({'models': models})
    
    def setup_socket_handlers(self):
        """Setup SocketIO event handlers for real-time communication"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            emit('connected', {'status': 'Connected to ebook2audiobook'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            print('Client disconnected')
        
        @self.socketio.on('join_conversion')
        def handle_join_conversion(data):
            """Join a conversion room for real-time updates"""
            conversion_id = data.get('conversion_id')
            if conversion_id:
                join_room(conversion_id)
    
    def get_conversion_params(self):
        """Extract conversion parameters from request"""
        data = request.get_json() or request.form
        
        params = {
            'ebook': session.get('ebook_path'),
            'language': data.get('language', default_language_code),
            'voice': session.get('voice_path'),
            'device': data.get('device', default_device),
            'tts_engine': data.get('tts_engine'),
            'custom_model': session.get('custom_model_path'),
            'fine_tuned': data.get('fine_tuned'),
            'output_format': data.get('output_format', default_output_format),
            'temperature': data.get('temperature'),
            'length_penalty': data.get('length_penalty'),
            'num_beams': data.get('num_beams'),
            'repetition_penalty': data.get('repetition_penalty'),
            'top_k': data.get('top_k'),
            'top_p': data.get('top_p'),
            'speed': data.get('speed'),
            'enable_text_splitting': data.get('enable_text_splitting', False),
            'text_temp': data.get('text_temp'),
            'waveform_temp': data.get('waveform_temp'),
            'output_dir': data.get('output_dir') or audiobooks_gradio_dir,
            'session': data.get('session_id') or self.ctx.get_session_id(),
            'is_gui_process': True,
            'audiobooks_dir': audiobooks_gradio_dir
        }
        
        # Clean up None values
        return {k: v for k, v in params.items() if v is not None}
    
    def run_conversion(self, conversion_id, params):
        """Run the conversion process in a separate thread"""
        try:
            # Emit conversion start
            self.socketio.emit('conversion_start', 
                             {'conversion_id': conversion_id}, 
                             room=conversion_id)
            
            # Setup progress callback
            def progress_callback(message, percentage=None):
                self.socketio.emit('conversion_progress', {
                    'conversion_id': conversion_id,
                    'message': message,
                    'percentage': percentage
                }, room=conversion_id)
            
            # Run conversion
            progress_status, passed = convert_ebook(params, self.ctx)
            
            if passed:
                # Store output path for download
                session[f'output_path_{conversion_id}'] = progress_status
                
                self.socketio.emit('conversion_complete', {
                    'conversion_id': conversion_id,
                    'success': True,
                    'output_path': progress_status
                }, room=conversion_id)
            else:
                self.socketio.emit('conversion_complete', {
                    'conversion_id': conversion_id,
                    'success': False,
                    'error': progress_status
                }, room=conversion_id)
                
        except Exception as e:
            self.socketio.emit('conversion_error', {
                'conversion_id': conversion_id,
                'error': str(e)
            }, room=conversion_id)
        finally:
            # Clean up thread reference
            if conversion_id in self.conversion_threads:
                del self.conversion_threads[conversion_id]
    
    def allowed_ebook_file(self, filename):
        """Check if uploaded file is an allowed ebook format"""
        return any(filename.lower().endswith(ext) for ext in ebook_formats)
    
    def allowed_voice_file(self, filename):
        """Check if uploaded file is an allowed voice format"""
        return any(filename.lower().endswith(ext) for ext in voice_formats)
    
    def get_language_options(self):
        """Get available language options"""
        return [
            {
                'code': lang,
                'name': f"{details['name']} - {details['native_name']}" 
                       if details['name'] != details['native_name'] 
                       else details['name']
            }
            for lang, details in language_mapping.items()
        ]
    
    def get_tts_engine_options(self):
        """Get available TTS engine options"""
        return [
            {'value': key, 'label': key} 
            for key in TTS_ENGINES.keys()
        ]
    
    def get_device_options(self):
        """Get available device options"""
        return [
            {'value': 'cpu', 'label': 'CPU'},
            {'value': 'cuda', 'label': 'GPU'},
            {'value': 'mps', 'label': 'MPS'}
        ]
    
    def get_output_format_options(self):
        """Get available output format options"""
        return [
            {'value': fmt, 'label': fmt.upper()} 
            for fmt in output_formats
        ]
    
    def run(self, host='0.0.0.0', port=7860, debug=False):
        """Run the Flask application"""
        try:
            all_ips = get_all_ip_addresses()
            print(f'IPs available for connection:\n{all_ips}')
            print(f'Note: 0.0.0.0 is not the IP to connect. Instead use an IP above to connect.')
            
            # Handle public sharing
            if self.is_gui_shared:
                self.setup_public_sharing(host, port)
            
            self.socketio.run(self.app, 
                            host=host, 
                            port=port, 
                            debug=debug, 
                            allow_unsafe_werkzeug=True)
        except Exception as e:
            print(f'Error starting Flask server: {e}')
            raise
    
    def setup_public_sharing(self, host, port):
        """Setup public sharing using ngrok or similar"""
        try:
            # Try to import and use pyngrok for public sharing
            from pyngrok import ngrok
            
            # Start ngrok tunnel
            public_url = ngrok.connect(port)
            print(f'Public URL: {public_url}')
            print(f'Share this URL to access the interface publicly')
            
        except ImportError:
            print('Warning: pyngrok not installed. Public sharing not available.')
            print('To enable public sharing, install pyngrok: pip install pyngrok')
        except Exception as e:
            print(f'Warning: Could not setup public sharing: {e}')


def web_interface_flask(args, ctx):
    """Flask-based web interface replacement for Gradio"""
    interface = FlaskInterface(args, ctx)
    interface.run(
        host=interface_host,
        port=interface_port,
        debug=debug_mode
    )