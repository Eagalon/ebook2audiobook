"""
Flask-based web interface for ebook2audiobook
Complete replacement for Gradio interface with all original functionality
"""

import os
import json
import uuid
import tempfile
import threading
import time
import shutil
import glob
import re
from pathlib import Path
from queue import Queue

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room
from werkzeug.utils import secure_filename

from lib import *
from lib.functions import (
    SessionContext, convert_ebook, convert_ebook_batch,
    show_alert, get_all_ip_addresses, get_compatible_tts_engines,
    get_sanitized
)
from lib.classes.voice_extractor import VoiceExtractor


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
                # Initialize session if not exists
                if 'session_id' not in session:
                    session['session_id'] = str(uuid.uuid4())
                    
                session_id = session['session_id']
                
                # Create context session if not exists
                if not hasattr(self.ctx, 'sessions') or session_id not in self.ctx.sessions:
                    self.ctx.get_session(session_id)
                
                return render_template('index.html', 
                                     version=prog_version,
                                     session_id=session_id,
                                     language_options=self.get_language_options(),
                                     tts_engine_options=self.get_tts_engine_options(),
                                     device_options=self.get_device_options(),
                                     output_format_options=self.get_output_format_options(),
                                     voice_options=self.get_voice_options(session_id),
                                     custom_model_options=self.get_custom_model_options(session_id),
                                     fine_tuned_options=self.get_fine_tuned_options(session_id),
                                     audiobook_options=self.get_audiobook_options(session_id),
                                     xtts_params=self.get_xtts_params(),
                                     bark_params=self.get_bark_params())
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
                session_id = session.get('session_id')
                ctx_session = self.ctx.get_session(session_id)
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(ctx_session['temp_dir'], 'uploads', filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                file.save(filepath)
                
                ctx_session['ebook'] = filepath
                ctx_session['ebook_mode'] = 'single'
                
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
            
            session_id = session.get('session_id')
            if not session_id:
                return jsonify({'error': 'No session found'}), 400
                
            ctx_session = self.ctx.get_session(session_id)
            
            # Check voice limit
            voice_options = self.get_voice_options(session_id)
            custom_voices = [v for v in voice_options if not v.get('builtin', False)]
            if len(custom_voices) >= max_custom_voices:
                return jsonify({'error': f'Maximum {max_custom_voices} custom voices allowed'}), 400
            
            if not self.allowed_voice_file(file.filename):
                return jsonify({'error': 'Invalid voice file format'}), 400
            
            try:
                filename = secure_filename(file.filename)
                temp_path = os.path.join(tempfile.gettempdir(), filename)
                file.save(temp_path)
                
                # Extract voice using VoiceExtractor
                voice_name = os.path.splitext(filename)[0].replace('&', 'And')
                voice_name = get_sanitized(voice_name)
                
                extractor = VoiceExtractor(ctx_session, temp_path, voice_name)
                status, msg = extractor.extract_voice()
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if status:
                    final_voice_file = os.path.join(ctx_session['voice_dir'], f'{voice_name}_24000.wav')
                    ctx_session['voice'] = final_voice_file
                    
                    return jsonify({
                        'success': True,
                        'message': f'Voice {voice_name} added successfully',
                        'voice_path': final_voice_file,
                        'voice_name': voice_name,
                        'voice_options': self.get_voice_options(session_id)
                    })
                else:
                    return jsonify({'error': f'Voice extraction failed: {msg}'}), 400
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/upload/custom_model', methods=['POST'])
        def upload_custom_model():
            """Handle custom model file upload"""
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
                
            session_id = session.get('session_id')
            if not session_id:
                return jsonify({'error': 'No session found'}), 400
                
            ctx_session = self.ctx.get_session(session_id)
            
            if not file.filename.lower().endswith('.zip'):
                return jsonify({'error': 'Only .zip files are allowed for custom models'}), 400
            
            try:
                filename = secure_filename(file.filename)
                temp_path = os.path.join(tempfile.gettempdir(), filename)
                file.save(temp_path)
                
                # Extract and validate custom model
                model_name = os.path.splitext(filename)[0]
                model_dir = os.path.join(ctx_session['custom_model_dir'], model_name)
                
                # Extract the zip file
                import zipfile
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Validate model files
                required_files = ['config.json', 'model.pth', 'vocab.json']
                missing_files = []
                for req_file in required_files:
                    if not any(os.path.exists(os.path.join(root, req_file)) 
                             for root, dirs, files in os.walk(model_dir)):
                        missing_files.append(req_file)
                
                if missing_files:
                    # Clean up invalid model
                    shutil.rmtree(model_dir, ignore_errors=True)
                    return jsonify({'error': f'Invalid model: missing files {missing_files}'}), 400
                
                ctx_session['custom_model'] = model_dir
                
                return jsonify({
                    'success': True,
                    'message': f'Custom model {model_name} uploaded successfully',
                    'model_path': model_dir,
                    'model_name': model_name,
                    'custom_model_options': self.get_custom_model_options(session_id)
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/upload/directory', methods=['POST'])
        def upload_directory():
            """Handle directory upload for batch processing"""
            if 'files' not in request.files:
                return jsonify({'error': 'No files provided'}), 400
            
            files = request.files.getlist('files')
            if not files:
                return jsonify({'error': 'No files selected'}), 400
                
            session_id = session.get('session_id')
            if not session_id:
                return jsonify({'error': 'No session found'}), 400
                
            ctx_session = self.ctx.get_session(session_id)
            
            try:
                ebook_files = []
                upload_dir = os.path.join(ctx_session['temp_dir'], 'batch_uploads')
                os.makedirs(upload_dir, exist_ok=True)
                
                for file in files:
                    if file.filename and self.allowed_ebook_file(file.filename):
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(upload_dir, filename)
                        file.save(filepath)
                        ebook_files.append(filepath)
                
                if not ebook_files:
                    return jsonify({'error': 'No valid ebook files found'}), 400
                
                ctx_session['ebook_list'] = ebook_files
                ctx_session['ebook_mode'] = 'directory'
                
                return jsonify({
                    'success': True,
                    'message': f'Uploaded {len(ebook_files)} ebook files',
                    'file_count': len(ebook_files),
                    'files': [os.path.basename(f) for f in ebook_files]
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/voices')
        def get_voices():
            """Get list of available voices"""
            session_id = session.get('session_id')
            if not session_id:
                return jsonify({'error': 'No session found'}), 400
            
            return jsonify({'voices': self.get_voice_options(session_id)})
        
        @self.app.route('/api/voices/<path:voice_path>/delete', methods=['POST'])
        def delete_voice(voice_path):
            """Delete a voice file"""
            session_id = session.get('session_id')
            if not session_id:
                return jsonify({'error': 'No session found'}), 400
                
            ctx_session = self.ctx.get_session(session_id)
            
            try:
                # Verify voice exists and is deleteable
                voice_name = re.sub(r'_(24000|16000)\.wav$|\.npz$', '', os.path.basename(voice_path))
                
                # Check if it's a builtin voice
                is_builtin = (voice_name in default_engine_settings[TTS_ENGINES['XTTSv2']]['voices'].keys() or
                             voice_name in default_engine_settings[TTS_ENGINES['BARK']]['voices'].keys() or
                             voice_name in default_engine_settings[TTS_ENGINES['YOURTTS']]['voices'].keys())
                
                if is_builtin:
                    return jsonify({'error': f'Voice {voice_name} is builtin and cannot be deleted'}), 400
                
                # Check if voice is in user's directory
                voice_path_obj = Path(voice_path).resolve()
                user_voice_dir = Path(ctx_session['voice_dir']).parent.resolve()
                
                if user_voice_dir not in voice_path_obj.parents:
                    return jsonify({'error': 'Only custom uploaded voices can be deleted'}), 400
                
                # Delete related files
                pattern = re.sub(r'_(24000|16000)\.wav$', '_*.wav', voice_path)
                files_to_remove = glob.glob(pattern)
                for file in files_to_remove:
                    if os.path.exists(file):
                        os.remove(file)
                
                # Remove bark subdirectory if exists
                bark_dir = os.path.join(os.path.dirname(voice_path), 'bark', voice_name)
                if os.path.exists(bark_dir):
                    shutil.rmtree(bark_dir, ignore_errors=True)
                
                # Clear session voice if it was the deleted one
                if ctx_session.get('voice') == voice_path:
                    ctx_session['voice'] = None
                
                return jsonify({
                    'success': True,
                    'message': f'Voice {voice_name} deleted successfully',
                    'voice_options': self.get_voice_options(session_id)
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/custom_models/<path:model_path>/delete', methods=['POST'])
        def delete_custom_model(model_path):
            """Delete a custom model"""
            session_id = session.get('session_id')
            if not session_id:
                return jsonify({'error': 'No session found'}), 400
                
            ctx_session = self.ctx.get_session(session_id)
            
            try:
                model_name = os.path.basename(model_path)
                
                # Verify model directory exists
                if not os.path.exists(model_path):
                    return jsonify({'error': 'Model not found'}), 404
                
                # Remove model directory
                shutil.rmtree(model_path, ignore_errors=True)
                
                # Clear session model if it was the deleted one
                if ctx_session.get('custom_model') == model_path:
                    ctx_session['custom_model'] = None
                
                return jsonify({
                    'success': True,
                    'message': f'Custom model {model_name} deleted successfully',
                    'custom_model_options': self.get_custom_model_options(session_id)
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/audiobooks/<path:audiobook_path>/delete', methods=['POST'])
        def delete_audiobook(audiobook_path):
            """Delete an audiobook"""
            session_id = session.get('session_id')
            if not session_id:
                return jsonify({'error': 'No session found'}), 400
                
            ctx_session = self.ctx.get_session(session_id)
            
            try:
                audiobook_name = os.path.basename(audiobook_path)
                
                # Remove audiobook file or directory
                if os.path.isdir(audiobook_path):
                    shutil.rmtree(audiobook_path, ignore_errors=True)
                elif os.path.exists(audiobook_path):
                    os.remove(audiobook_path)
                else:
                    return jsonify({'error': 'Audiobook not found'}), 404
                
                # Clear session audiobook if it was the deleted one
                if ctx_session.get('audiobook') == audiobook_path:
                    ctx_session['audiobook'] = None
                
                return jsonify({
                    'success': True,
                    'message': f'Audiobook {audiobook_name} deleted successfully',
                    'audiobook_options': self.get_audiobook_options(session_id)
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/audiobooks/<path:audiobook_path>/download')
        def download_audiobook(audiobook_path):
            """Download an audiobook"""
            try:
                if not os.path.exists(audiobook_path):
                    return jsonify({'error': 'Audiobook not found'}), 404
                
                return send_file(audiobook_path, as_attachment=True, 
                               download_name=os.path.basename(audiobook_path))
                               
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/tts_engines')
        def get_tts_engines():
            """Get available TTS engines with ratings"""
            session_id = session.get('session_id')
            ctx_session = self.ctx.get_session(session_id) if session_id else None
            language = ctx_session.get('language', default_language_code) if ctx_session else default_language_code
            
            compatible_engines = get_compatible_tts_engines(language)
            engines_with_ratings = []
            
            for engine in compatible_engines:
                rating = default_engine_settings.get(engine, {}).get('rating', {})
                engines_with_ratings.append({
                    'value': engine,
                    'label': engine,
                    'rating': rating
                })
            
            return jsonify({'engines': engines_with_ratings})
        
        @self.app.route('/api/fine_tuned_models')
        def get_fine_tuned_models():
            """Get fine-tuned models for selected TTS engine"""
            session_id = session.get('session_id')
            if not session_id:
                return jsonify({'error': 'No session found'}), 400
                
            ctx_session = self.ctx.get_session(session_id)
            tts_engine = ctx_session.get('tts_engine', TTS_ENGINES['XTTSv2'])
            
            return jsonify({'fine_tuned_models': self.get_fine_tuned_options(session_id, tts_engine)})
        
        @self.app.route('/convert', methods=['POST'])
        def start_conversion():
            """Start the ebook conversion process"""
            try:
                session_id = session.get('session_id')
                if not session_id:
                    return jsonify({'error': 'No session found'}), 400
                
                # Get conversion parameters
                params = self.get_conversion_params(session_id)
                
                # Create a unique conversion ID
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
    
    def get_conversion_params(self, session_id):
        """Extract conversion parameters from request and session"""
        data = request.get_json() or request.form
        ctx_session = self.ctx.get_session(session_id)
        
        # Prepare parameters for convert_ebook function
        params = {
            'ebook': ctx_session.get('ebook'),
            'ebook_list': ctx_session.get('ebook_list'),
            'ebook_mode': ctx_session.get('ebook_mode', 'single'),
            'language': data.get('language', default_language_code),
            'voice': ctx_session.get('voice'),
            'device': data.get('device', default_device),
            'tts_engine': data.get('tts_engine'),
            'custom_model': ctx_session.get('custom_model'),
            'fine_tuned': data.get('fine_tuned'),
            'output_format': data.get('output_format', default_output_format),
            'temperature': data.get('temperature'),
            'repetition_penalty': data.get('repetition_penalty'),
            'top_k': data.get('top_k'),
            'top_p': data.get('top_p'),
            'speed': data.get('speed'),
            'enable_text_splitting': data.get('enable_text_splitting', False),
            'text_temp': data.get('text_temp'),
            'waveform_temp': data.get('waveform_temp'),
            'session': session_id,
            'is_gui_process': True,
            'audiobooks_dir': ctx_session.get('audiobook_dir', audiobooks_gradio_dir)
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
            
            # Run the actual conversion
            if params.get('ebook_mode') == 'directory' and params.get('ebook_list'):
                # Batch conversion
                progress_status, passed = convert_ebook_batch(params, self.ctx)
            else:
                # Single ebook conversion
                progress_status, passed = convert_ebook(params, self.ctx)
            
            if passed:
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
    
    def get_language_options(self):
        """Get list of available languages"""
        return [
            {
                'code': lang,
                'name': f"{details['name']} - {details['native_name']}" if details['name'] != details['native_name'] else details['name']
            }
            for lang, details in language_mapping.items()
        ]
    
    def get_device_options(self):
        """Get list of available devices"""
        return [
            {'value': 'cpu', 'label': 'CPU'},
            {'value': 'cuda', 'label': 'GPU'},
            {'value': 'mps', 'label': 'MPS'}
        ]
    
    def get_output_format_options(self):
        """Get list of available output formats"""
        return [
            {'value': fmt, 'label': fmt.upper()}
            for fmt in output_formats
        ]
    
    def get_tts_engine_options(self, language=None):
        """Get list of available TTS engines"""
        if language:
            compatible_engines = get_compatible_tts_engines(language)
        else:
            compatible_engines = list(TTS_ENGINES.values())
        
        engines_with_ratings = []
        for engine in compatible_engines:
            rating = default_engine_settings.get(engine, {}).get('rating', {})
            engines_with_ratings.append({
                'value': engine,
                'label': engine,
                'rating': self.format_engine_rating(rating)
            })
        
        return engines_with_ratings
    
    def format_engine_rating(self, rating):
        """Format engine rating display"""
        if not rating:
            return ""
        
        def yellow_stars(n):
            return "★" * n
        
        def color_box(value, label):
            if value <= 4:
                color = "#4CAF50"  # Green = low
            elif value <= 8:
                color = "#FF9800"  # Orange = medium
            else:
                color = "#F44336"  # Red = high
            return f'<span style="background:{color};color:white;padding:1px 5px;border-radius:3px;font-size:11px">{value} GB</span>'
        
        return f"""
        <div style="font-size:12px; display:inline; gap:10px;">
            <span><b>GPU VRAM:</b> {color_box(rating.get("GPU VRAM", 0), "GPU VRAM")}</span>
            <span><b>CPU:</b> {yellow_stars(rating.get("CPU", 0))}</span>
            <span><b>RAM:</b> {color_box(rating.get("RAM", 0), "RAM")}</span>
            <span><b>Realism:</b> {yellow_stars(rating.get("Realism", 0))}</span>
        </div>
        """
    
    def get_voice_options(self, session_id):
        """Get list of available voices for session"""
        ctx_session = self.ctx.get_session(session_id)
        voices = []
        
        # Add builtin voices for each engine
        for engine, settings in default_engine_settings.items():
            if 'voices' in settings:
                for voice_name, voice_path in settings['voices'].items():
                    voices.append({
                        'value': voice_path,
                        'label': voice_name,
                        'builtin': True,
                        'engine': engine
                    })
        
        # Add custom voices from session directory
        voice_dir = ctx_session.get('voice_dir')
        if voice_dir and os.path.exists(voice_dir):
            for file in os.listdir(voice_dir):
                if file.endswith('_24000.wav'):
                    voice_name = file.replace('_24000.wav', '')
                    voice_path = os.path.join(voice_dir, file)
                    voices.append({
                        'value': voice_path,
                        'label': voice_name,
                        'builtin': False,
                        'engine': 'custom'
                    })
        
        return voices
    
    def get_custom_model_options(self, session_id):
        """Get list of available custom models for session"""
        ctx_session = self.ctx.get_session(session_id)
        models = []
        
        model_dir = ctx_session.get('custom_model_dir')
        if model_dir and os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path):
                    # Check if it's a valid model directory
                    config_path = os.path.join(item_path, 'config.json')
                    if os.path.exists(config_path):
                        models.append({
                            'value': item_path,
                            'label': item,
                            'path': item_path
                        })
        
        return models
    
    def get_fine_tuned_options(self, session_id, tts_engine=None):
        """Get list of fine-tuned models for TTS engine"""
        if not tts_engine:
            ctx_session = self.ctx.get_session(session_id)
            tts_engine = ctx_session.get('tts_engine', TTS_ENGINES['XTTSv2'])
        
        presets = []
        
        # Get presets for the selected engine
        engine_settings = default_engine_settings.get(tts_engine, {})
        if 'presets' in engine_settings:
            for preset_name, preset_path in engine_settings['presets'].items():
                presets.append({
                    'value': preset_path,
                    'label': preset_name,
                    'engine': tts_engine
                })
        
        return presets
    
    def get_audiobook_options(self, session_id):
        """Get list of generated audiobooks for session"""
        ctx_session = self.ctx.get_session(session_id)
        audiobooks = []
        
        audiobook_dir = ctx_session.get('audiobook_dir', audiobooks_gradio_dir)
        if audiobook_dir and os.path.exists(audiobook_dir):
            for item in os.listdir(audiobook_dir):
                item_path = os.path.join(audiobook_dir, item)
                if os.path.isfile(item_path) and any(item.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.flac']):
                    audiobooks.append({
                        'value': item_path,
                        'label': item,
                        'path': item_path,
                        'size': os.path.getsize(item_path)
                    })
                elif os.path.isdir(item_path):
                    # Directory of audio files
                    audiobooks.append({
                        'value': item_path,
                        'label': item,
                        'path': item_path,
                        'type': 'directory'
                    })
        
        return audiobooks
    
    def get_xtts_params(self):
        """Get XTTS parameter defaults"""
        xtts_settings = default_engine_settings.get(TTS_ENGINES['XTTSv2'], {})
        return {
            'temperature': {
                'min': 0.1, 'max': 10.0, 'step': 0.1,
                'value': float(xtts_settings.get('temperature', 0.7)),
                'info': 'Higher values lead to more creative, unpredictable outputs. Lower values make it more monotone.'
            },
            'repetition_penalty': {
                'min': 1.0, 'max': 10.0, 'step': 0.1,
                'value': float(xtts_settings.get('repetition_penalty', 5.0)),
                'info': 'Penalizes repeated phrases. Higher values reduce repetition.'
            },
            'top_k': {
                'min': 10, 'max': 100, 'step': 1,
                'value': int(xtts_settings.get('top_k', 50)),
                'info': 'Lower values restrict outputs to more likely words and increase speed.'
            },
            'top_p': {
                'min': 0.1, 'max': 1.0, 'step': 0.01,
                'value': float(xtts_settings.get('top_p', 0.8)),
                'info': 'Controls cumulative probability for word selection. Lower values make output more predictable.'
            },
            'speed': {
                'min': 0.5, 'max': 3.0, 'step': 0.1,
                'value': float(xtts_settings.get('speed', 1.0)),
                'info': 'Adjusts how fast the narrator will speak.'
            }
        }
    
    def get_bark_params(self):
        """Get BARK parameter defaults"""
        bark_settings = default_engine_settings.get(TTS_ENGINES['BARK'], {})
        return {
            'text_temp': {
                'min': 0.0, 'max': 1.0, 'step': 0.01,
                'value': float(bark_settings.get('text_temp', 0.7)),
                'info': 'Higher values lead to more creative, unpredictable outputs. Lower values make it more conservative.'
            },
            'waveform_temp': {
                'min': 0.0, 'max': 1.0, 'step': 0.01,
                'value': float(bark_settings.get('waveform_temp', 0.7)),
                'info': 'Higher values lead to more creative, unpredictable outputs. Lower values make it more conservative.'
            }
        }
    
    def allowed_ebook_file(self, filename):
        """Check if file is an allowed ebook format"""
        return any(filename.lower().endswith(ext) for ext in ebook_formats)
    
    def allowed_voice_file(self, filename):
        """Check if file is an allowed voice format"""
        return any(filename.lower().endswith(ext) for ext in voice_formats)
    
    def allowed_model_file(self, filename):
        """Check if file is an allowed model format"""
        return filename.lower().endswith('.zip')


def web_interface_flask(args, ctx):
    """Start the Flask web interface"""
    interface = FlaskInterface(args, ctx)
    
    # Get available IPs
    ips = get_all_ip_addresses()
    print("IPs available for connection:")
    print(ips)
    print("Note: 0.0.0.0 is not the IP to connect. Instead use an IP above to connect.")
    
    # Start the server
    try:
        if args['share']:
            # Enable public sharing with ngrok
            try:
                from pyngrok import ngrok
                public_url = ngrok.connect(interface_port)
                print(f"Public URL: {public_url}")
            except ImportError:
                print("Warning: pyngrok not installed. Install with 'pip install pyngrok' for public sharing.")
            except Exception as e:
                print(f"Warning: Could not create public URL: {e}")
        
        # Don't use reloader when called from main app.py to prevent port conflicts
        use_reloader = not args.get('script_mode') == NATIVE
        
        interface.socketio.run(
            interface.app,
            host='0.0.0.0',
            port=interface_port,
            debug=True,
            use_reloader=use_reloader,
            allow_unsafe_werkzeug=True
        )
        
    except Exception as e:
        print(f"Error starting Flask interface: {e}")
        raise