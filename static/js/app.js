/**
 * JavaScript for ebook2audiobook Flask interface
 * Handles file uploads, real-time updates, and user interactions
 */

class Ebook2AudiobookApp {
    constructor() {
        this.socket = null;
        this.currentConversionId = null;
        this.uploadedFiles = {
            ebook: null,
            voice: null,
            customModel: null
        };
        
        this.init();
    }
    
    init() {
        this.initSocketIO();
        this.setupEventListeners();
        this.loadAvailableVoices();
        this.loadAvailableModels();
        this.hideGlassMask();
    }
    
    initSocketIO() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        this.socket.on('conversion_start', (data) => {
            this.handleConversionStart(data);
        });
        
        this.socket.on('conversion_progress', (data) => {
            this.handleConversionProgress(data);
        });
        
        this.socket.on('conversion_complete', (data) => {
            this.handleConversionComplete(data);
        });
        
        this.socket.on('conversion_error', (data) => {
            this.handleConversionError(data);
        });
    }
    
    setupEventListeners() {
        // File upload handlers
        this.setupFileUpload('ebook_file', 'ebook', this.handleEbookUpload.bind(this));
        this.setupFileUpload('voice_file', 'voice', this.handleVoiceUpload.bind(this));
        this.setupFileUpload('custom_model_file', 'customModel', this.handleCustomModelUpload.bind(this));
        
        // Form submission
        document.getElementById('conversionForm').addEventListener('submit', this.handleFormSubmit.bind(this));
        
        // Input mode switching
        document.getElementsByName('input_mode').forEach(radio => {
            radio.addEventListener('change', this.handleInputModeChange.bind(this));
        });
        
        // Voice list selection
        document.getElementById('voice_list').addEventListener('change', this.handleVoiceSelection.bind(this));
        
        // Custom model list selection
        document.getElementById('custom_model_list').addEventListener('change', this.handleCustomModelSelection.bind(this));
        
        // TTS engine change
        document.getElementById('tts_engine').addEventListener('change', this.handleTTSEngineChange.bind(this));
    }
    
    setupFileUpload(inputId, fileType, callback) {
        const input = document.getElementById(inputId);
        const uploadArea = document.getElementById(`${fileType === 'ebook' ? 'ebook' : 
                                                   fileType === 'voice' ? 'voice' : 'model'}_upload_area`);
        
        input.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                callback(e.target.files[0]);
            }
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length > 0) {
                input.files = e.dataTransfer.files;
                callback(e.dataTransfer.files[0]);
            }
        });
    }
    
    async handleEbookUpload(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/upload/ebook', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.uploadedFiles.ebook = result;
                this.showFilePreview('ebook', result.filename, file.size);
                this.showSuccess(`Ebook uploaded: ${result.filename}`);
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Failed to upload ebook: ' + error.message);
        }
    }
    
    async handleVoiceUpload(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/upload/voice', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.uploadedFiles.voice = result;
                this.showFilePreview('voice', result.filename, file.size);
                this.showVoicePreview(result.path);
                this.showSuccess(`Voice file uploaded: ${result.filename}`);
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Failed to upload voice file: ' + error.message);
        }
    }
    
    async handleCustomModelUpload(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/upload/custom_model', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.uploadedFiles.customModel = result;
                this.showFilePreview('model', result.filename, file.size);
                this.showSuccess(`Custom model uploaded: ${result.filename}`);
                this.loadAvailableModels(); // Refresh model list
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Failed to upload custom model: ' + error.message);
        }
    }
    
    showFilePreview(type, filename, size) {
        const container = document.getElementById(`${type}_filename`);
        container.innerHTML = `
            <div class="file-preview">
                <div class="file-info">
                    <i class="fas fa-file file-icon"></i>
                    <span class="file-name">${filename}</span>
                    <span class="file-size">(${this.formatFileSize(size)})</span>
                </div>
                <button type="button" class="remove-file" onclick="app.removeFile('${type}')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        container.classList.remove('d-none');
        
        const uploadArea = document.getElementById(`${type === 'model' ? 'model' : 
                                                   type === 'voice' ? 'voice' : 'ebook'}_upload_area`);
        uploadArea.classList.add('has-file');
    }
    
    removeFile(type) {
        this.uploadedFiles[type] = null;
        const container = document.getElementById(`${type}_filename`);
        container.classList.add('d-none');
        container.innerHTML = '';
        
        const uploadArea = document.getElementById(`${type === 'model' ? 'model' : 
                                                   type === 'voice' ? 'voice' : 'ebook'}_upload_area`);
        uploadArea.classList.remove('has-file');
        
        const input = document.getElementById(`${type === 'voice' ? 'voice' : 
                                              type === 'model' ? 'custom_model' : 'ebook'}_file`);
        input.value = '';
        
        if (type === 'voice') {
            document.getElementById('voice_preview').classList.add('d-none');
        }
    }
    
    showVoicePreview(path) {
        const preview = document.getElementById('voice_preview');
        const source = document.getElementById('voice_audio_source');
        source.src = '/preview/voice';
        preview.classList.remove('d-none');
    }
    
    async loadAvailableVoices() {
        try {
            const response = await fetch('/voices/list');
            const result = await response.json();
            
            const select = document.getElementById('voice_list');
            select.innerHTML = '<option value="">Select from available voices...</option>';
            
            result.voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.path;
                option.textContent = voice.name;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load voices:', error);
        }
    }
    
    async loadAvailableModels() {
        try {
            const response = await fetch('/models/list');
            const result = await response.json();
            
            const select = document.getElementById('custom_model_list');
            select.innerHTML = '<option value="">Select from uploaded models...</option>';
            
            result.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = model.name;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }
    
    handleInputModeChange(e) {
        const fileUpload = document.getElementById('file_upload');
        
        if (e.target.value === 'file') {
            fileUpload.style.display = 'block';
        } else {
            fileUpload.style.display = 'none';
        }
    }
    
    handleVoiceSelection(e) {
        if (e.target.value) {
            // Use selected voice from list
            this.showSuccess(`Selected voice: ${e.target.options[e.target.selectedIndex].text}`);
        }
    }
    
    handleCustomModelSelection(e) {
        if (e.target.value) {
            // Use selected model from list
            this.showSuccess(`Selected model: ${e.target.options[e.target.selectedIndex].text}`);
        }
    }
    
    handleTTSEngineChange(e) {
        const engine = e.target.value;
        const ratingDiv = document.getElementById('tts_rating');
        
        // Show different parameter tabs based on engine
        const xttsTab = document.getElementById('xtts-tab');
        const barkTab = document.getElementById('bark-tab');
        
        if (engine && (engine.toLowerCase().includes('xtts') || engine.toLowerCase().includes('vits'))) {
            xttsTab.style.display = 'block';
        } else {
            xttsTab.style.display = 'none';
        }
        
        if (engine && engine.toLowerCase().includes('bark')) {
            barkTab.style.display = 'block';
        } else {
            barkTab.style.display = 'none';
        }
        
        // Add engine rating/info
        const engineInfo = this.getEngineInfo(engine);
        ratingDiv.innerHTML = engineInfo;
    }
    
    getEngineInfo(engine) {
        const engineMap = {
            'XTTSv2': '<div class="alert alert-info p-2"><small><strong>XTTS v2:</strong> High quality, supports voice cloning</small></div>',
            'BARK': '<div class="alert alert-warning p-2"><small><strong>BARK:</strong> Creative voices, requires GPU</small></div>',
            'VITS': '<div class="alert alert-info p-2"><small><strong>VITS:</strong> Fast synthesis, good quality</small></div>',
            'FAIRSEQ': '<div class="alert alert-secondary p-2"><small><strong>FAIRSEQ:</strong> Research model</small></div>',
            'TACOTRON2': '<div class="alert alert-secondary p-2"><small><strong>TACOTRON2:</strong> Classic TTS</small></div>',
            'YOURTTS': '<div class="alert alert-info p-2"><small><strong>YourTTS:</strong> Multi-speaker synthesis</small></div>'
        };
        
        return engineMap[engine] || '';
    }
    
    async handleFormSubmit(e) {
        e.preventDefault();
        
        // Validate required fields
        if (!this.uploadedFiles.ebook && document.querySelector('input[name="input_mode"]:checked').value === 'file') {
            this.showError('Please upload an ebook file');
            return;
        }
        
        // Prepare form data
        const formData = this.getFormData();
        
        try {
            this.showProgress();
            this.disableForm();
            
            const response = await fetch('/convert', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentConversionId = result.conversion_id;
                this.socket.emit('join_conversion', { conversion_id: this.currentConversionId });
                this.showSuccess('Conversion started successfully');
            } else {
                this.showError(result.error);
                this.enableForm();
                this.hideProgress();
            }
        } catch (error) {
            this.showError('Failed to start conversion: ' + error.message);
            this.enableForm();
            this.hideProgress();
        }
    }
    
    getFormData() {
        const form = document.getElementById('conversionForm');
        const formData = new FormData(form);
        const data = {};
        
        // Get all form values
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        // Add file paths from uploads
        if (this.uploadedFiles.ebook) {
            data.ebook = this.uploadedFiles.ebook.path;
        }
        if (this.uploadedFiles.voice) {
            data.voice = this.uploadedFiles.voice.path;
        }
        if (this.uploadedFiles.customModel) {
            data.custom_model = this.uploadedFiles.customModel.path;
        }
        
        // Handle voice selection
        const voiceList = document.getElementById('voice_list');
        if (voiceList.value && !data.voice) {
            data.voice = voiceList.value;
        }
        
        // Handle custom model selection
        const modelList = document.getElementById('custom_model_list');
        if (modelList.value && !data.custom_model) {
            data.custom_model = modelList.value;
        }
        
        return data;
    }
    
    showProgress() {
        document.getElementById('progress_card').style.display = 'block';
        document.getElementById('progress_card').scrollIntoView({ behavior: 'smooth' });
    }
    
    hideProgress() {
        document.getElementById('progress_card').style.display = 'none';
    }
    
    showDownload() {
        document.getElementById('download_card').style.display = 'block';
        document.getElementById('download_card').scrollIntoView({ behavior: 'smooth' });
    }
    
    hideDownload() {
        document.getElementById('download_card').style.display = 'none';
    }
    
    disableForm() {
        const form = document.getElementById('conversionForm');
        const inputs = form.querySelectorAll('input, select, button');
        inputs.forEach(input => input.disabled = true);
    }
    
    enableForm() {
        const form = document.getElementById('conversionForm');
        const inputs = form.querySelectorAll('input, select, button');
        inputs.forEach(input => input.disabled = false);
    }
    
    handleConversionStart(data) {
        this.updateProgress(0, 'Starting conversion...');
        this.logMessage('Conversion started');
    }
    
    handleConversionProgress(data) {
        const percentage = data.percentage || 0;
        const message = data.message || 'Processing...';
        
        this.updateProgress(percentage, message);
        this.logMessage(message);
    }
    
    handleConversionComplete(data) {
        if (data.success) {
            this.updateProgress(100, 'Conversion completed successfully!');
            this.logMessage('Conversion completed successfully');
            this.showDownload();
            
            // Setup download button
            const downloadBtn = document.getElementById('download_btn');
            downloadBtn.onclick = () => {
                window.location.href = `/download/${data.conversion_id}`;
            };
        } else {
            this.showError(`Conversion failed: ${data.error}`);
            this.logMessage(`Error: ${data.error}`);
        }
        
        this.enableForm();
    }
    
    handleConversionError(data) {
        this.showError(`Conversion error: ${data.error}`);
        this.logMessage(`Error: ${data.error}`);
        this.enableForm();
    }
    
    updateProgress(percentage, message) {
        const progressBar = document.getElementById('progress_bar');
        const progressMessage = document.getElementById('progress_message');
        
        progressBar.style.width = `${percentage}%`;
        progressBar.textContent = `${Math.round(percentage)}%`;
        progressMessage.textContent = message;
        
        if (percentage === 100) {
            progressBar.classList.remove('progress-bar-animated');
        }
    }
    
    logMessage(message) {
        const logDiv = document.getElementById('progress_log');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.innerHTML = `<span class="text-muted">[${timestamp}]</span> ${message}`;
        logDiv.appendChild(logEntry);
        logDiv.scrollTop = logDiv.scrollHeight;
    }
    
    showSuccess(message) {
        this.showAlert(message, 'success');
    }
    
    showError(message) {
        this.showAlert(message, 'danger');
    }
    
    showAlert(message, type) {
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alert.style.cssText = 'top: 20px; right: 20px; z-index: 1050; max-width: 400px;';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert && alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    showGlassMask(message = 'Please wait...') {
        const mask = document.getElementById('glass-mask');
        mask.querySelector('div').textContent = message;
        mask.classList.remove('d-none');
    }
    
    hideGlassMask() {
        const mask = document.getElementById('glass-mask');
        mask.classList.add('d-none');
    }
}

// Initialize the application
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new Ebook2AudiobookApp();
});