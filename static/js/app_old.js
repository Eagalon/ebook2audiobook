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
        
        // Load data asynchronously but don't block initialization
        Promise.allSettled([
            this.loadAvailableVoices(),
            this.loadAvailableModels()
        ]).finally(() => {
            // Always hide the glass mask after a short delay, regardless of loading success
            setTimeout(() => {
                this.hideGlassMask();
            }, 500);
        });
        
        // Fallback: hide glass mask after 3 seconds no matter what
        setTimeout(() => {
            this.hideGlassMask();
        }, 3000);
    }
    
    initSocketIO() {
        try {
            // Check if Socket.IO is available
            if (typeof io === 'undefined') {
                console.warn('Socket.IO not available, continuing without real-time updates');
                return;
            }
            
            this.socket = io({
                timeout: 10000,
                transports: ['websocket', 'polling']
            });
            
            this.socket.on('connect', () => {
                console.log('Connected to server');
            });
            
            this.socket.on('connect_error', (error) => {
                console.warn('Socket connection error:', error);
                // Continue anyway - the interface can work without real-time updates
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
        } catch (error) {
            console.error('Failed to initialize SocketIO:', error);
            // Continue without socket - basic functionality will still work
        }
    }
    
    setupEventListeners() {
        // File upload handlers
        this.setupFileUpload('ebook_file', 'ebook', this.handleEbookUpload.bind(this));
        this.setupFileUpload('voice_file', 'voice', this.handleVoiceUpload.bind(this));
        this.setupFileUpload('custom_model_file', 'customModel', this.handleCustomModelUpload.bind(this));
        
        // Form submission
        const form = document.getElementById('conversionForm');
        if (form) {
            form.addEventListener('submit', this.handleFormSubmit.bind(this));
        }
        
        // Input mode switching
        const inputModeRadios = document.getElementsByName('input_mode');
        if (inputModeRadios) {
            inputModeRadios.forEach(radio => {
                radio.addEventListener('change', this.handleInputModeChange.bind(this));
            });
        }
        
        // Voice list selection
        const voiceList = document.getElementById('voice_list');
        if (voiceList) {
            voiceList.addEventListener('change', this.handleVoiceSelection.bind(this));
        }
        
        // Custom model list selection
        const modelList = document.getElementById('custom_model_list');
        if (modelList) {
            modelList.addEventListener('change', this.handleCustomModelSelection.bind(this));
        }
        
        // TTS engine change
        const ttsEngine = document.getElementById('tts_engine');
        if (ttsEngine) {
            ttsEngine.addEventListener('change', this.handleTTSEngineChange.bind(this));
        }
        
        // Parameter sliders
        this.setupParameterSliders();
    }
    
    setupFileUpload(inputId, fileType, callback) {
        const input = document.getElementById(inputId);
        const uploadArea = document.getElementById(`${fileType === 'ebook' ? 'ebook' : 
                                                   fileType === 'voice' ? 'voice' : 'model'}_upload_area`);
        
        if (!input || !uploadArea) {
            console.warn(`Missing elements for file upload: ${inputId}, uploadArea for ${fileType}`);
            return;
        }
        
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
                this.enableConvertButton();
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
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
            
            const response = await fetch('/api/voices', {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            const result = await response.json();
            
            const select = document.getElementById('voice_list');
            if (select && result.voices) {
                select.innerHTML = '<option value="">Select from available voices...</option>';
                result.voices.forEach(voice => {
                    const option = document.createElement('option');
                    option.value = voice.value;
                    option.textContent = voice.label + (voice.builtin ? ' (built-in)' : ' (custom)');
                    option.dataset.builtin = voice.builtin;
                    option.dataset.engine = voice.engine;
                    select.appendChild(option);
                });
            }
        } catch (error) {
            console.warn('Failed to load voices:', error);
            // Don't show error to user - this is not critical for startup
        }
    }
    
    async loadAvailableModels() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch('/api/custom_models', {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            const result = await response.json();
            
            const select = document.getElementById('custom_model_list');
            if (select && result.custom_models) {
                select.innerHTML = '<option value="">Select from uploaded models...</option>';
                result.custom_models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.value;
                    option.textContent = model.label;
                    select.appendChild(option);
                });
            }
        } catch (error) {
            console.warn('Failed to load custom models:', error);
        }
    }
    
    async loadAudiobooks() {
        try {
            const response = await fetch('/api/audiobooks');
            const result = await response.json();
            
            const select = document.getElementById('audiobook_list');
            if (select && result.audiobooks) {
                select.innerHTML = '<option value="">Select audiobook...</option>';
                result.audiobooks.forEach(audiobook => {
                    const option = document.createElement('option');
                    option.value = audiobook.value;
                    option.textContent = audiobook.label;
                    select.appendChild(option);
                });
                
                // Show audiobook section if there are audiobooks
                const audiobookSection = document.getElementById('audiobook_section');
                if (audiobookSection) {
                    audiobookSection.classList.toggle('d-none', result.audiobooks.length === 0);
                }
            }
        } catch (error) {
            console.warn('Failed to load audiobooks:', error);
        }
    }
    
    async loadTTSEngines() {
        try {
            const response = await fetch('/api/tts_engines');
            const result = await response.json();
            
            const select = document.getElementById('tts_engine');
            if (select && result.engines) {
                select.innerHTML = '';
                result.engines.forEach(engine => {
                    const option = document.createElement('option');
                    option.value = engine.value;
                    option.textContent = engine.label;
                    option.dataset.rating = JSON.stringify(engine.rating);
                    select.appendChild(option);
                });
                
                // Update rating display for first engine
                if (result.engines.length > 0) {
                    this.updateTTSRating(result.engines[0].rating);
                }
            }
        } catch (error) {
            console.warn('Failed to load TTS engines:', error);
        }
    }
    
    async loadFineTunedModels() {
        try {
            const response = await fetch('/api/fine_tuned_models');
            const result = await response.json();
            
            const select = document.getElementById('fine_tuned');
            if (select && result.fine_tuned_models) {
                select.innerHTML = '<option value="">Select preset...</option>';
                result.fine_tuned_models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.value;
                    option.textContent = model.label;
                    select.appendChild(option);
                });
            }
        } catch (error) {
            console.warn('Failed to load fine-tuned models:', error);
        }
    }
    
    handleVoiceSelection(event) {
        const selectedPath = event.target.value;
        const selectedOption = event.target.options[event.target.selectedIndex];
        
        if (selectedPath) {
            // Show voice preview
            this.showVoicePreview(selectedPath);
            
            // Show delete button for custom voices
            const deleteBtn = document.getElementById('voice_delete_btn');
            if (deleteBtn) {
                const isBuiltin = selectedOption.dataset.builtin === 'true';
                deleteBtn.classList.toggle('d-none', isBuiltin);
                deleteBtn.onclick = () => this.deleteVoice(selectedPath);
            }
        } else {
            // Hide preview and delete button
            document.getElementById('voice_preview').classList.add('d-none');
            document.getElementById('voice_delete_btn').classList.add('d-none');
        }
    }
    
    handleCustomModelSelection(event) {
        const selectedPath = event.target.value;
        
        const deleteBtn = document.getElementById('model_delete_btn');
        if (deleteBtn) {
            deleteBtn.classList.toggle('d-none', !selectedPath);
            deleteBtn.onclick = () => this.deleteCustomModel(selectedPath);
        }
    }
    
    handleTTSEngineChange(event) {
        const selectedEngine = event.target.value;
        const selectedOption = event.target.options[event.target.selectedIndex];
        
        // Update rating display
        if (selectedOption.dataset.rating) {
            const rating = JSON.parse(selectedOption.dataset.rating);
            this.updateTTSRating(rating);
        }
        
        // Reload fine-tuned models for this engine
        this.loadFineTunedModels();
        
        // Show/hide parameter tabs based on engine
        this.updateParameterTabs(selectedEngine);
    }
    
    updateTTSRating(rating) {
        const ratingDiv = document.getElementById('tts_rating');
        if (ratingDiv && rating) {
            const gpuVramColor = rating['GPU VRAM'] <= 4 ? '#4CAF50' : rating['GPU VRAM'] <= 8 ? '#FF9800' : '#F44336';
            const ramColor = rating['RAM'] <= 4 ? '#4CAF50' : rating['RAM'] <= 8 ? '#FF9800' : '#F44336';
            const stars = (n) => '★'.repeat(n || 0);
            
            ratingDiv.innerHTML = `
                <div style="font-size:12px; display:flex; gap:10px; flex-wrap:wrap;">
                    <span><b>GPU VRAM:</b> <span style="background:${gpuVramColor};color:white;padding:1px 5px;border-radius:3px;">${rating['GPU VRAM'] || 0} GB</span></span>
                    <span><b>CPU:</b> <span style="color:#FFD700;">${stars(rating['CPU'])}</span></span>
                    <span><b>RAM:</b> <span style="background:${ramColor};color:white;padding:1px 5px;border-radius:3px;">${rating['RAM'] || 0} GB</span></span>
                    <span><b>Realism:</b> <span style="color:#FFD700;">${stars(rating['Realism'])}</span></span>
                </div>
            `;
        }
    }
    
    updateParameterTabs(engine) {
        const xttsTab = document.getElementById('xtts-tab');
        const barkTab = document.getElementById('bark-tab');
        
        if (xttsTab && barkTab) {
            const isXTTS = engine.includes('XTTSv2');
            const isBark = engine.includes('BARK');
            
            xttsTab.style.display = isXTTS ? 'block' : 'none';
            barkTab.style.display = isBark ? 'block' : 'none';
        }
    }
    
    async deleteVoice(voicePath) {
        if (!confirm('Are you sure you want to delete this voice?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/voices/${encodeURIComponent(voicePath)}/delete`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess(result.message);
                this.loadAvailableVoices(); // Refresh voice list
                
                // Clear selection
                document.getElementById('voice_list').value = '';
                document.getElementById('voice_preview').classList.add('d-none');
                document.getElementById('voice_delete_btn').classList.add('d-none');
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Failed to delete voice: ' + error.message);
        }
    }
    
    async deleteCustomModel(modelPath) {
        if (!confirm('Are you sure you want to delete this custom model?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/custom_models/${encodeURIComponent(modelPath)}/delete`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess(result.message);
                this.loadAvailableModels(); // Refresh model list
                
                // Clear selection
                document.getElementById('custom_model_list').value = '';
                document.getElementById('model_delete_btn').classList.add('d-none');
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Failed to delete custom model: ' + error.message);
        }
    }
    
    async deleteAudiobook(audiobookPath) {
        if (!confirm('Are you sure you want to delete this audiobook?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/audiobooks/${encodeURIComponent(audiobookPath)}/delete`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess(result.message);
                this.loadAudiobooks(); // Refresh audiobook list
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Failed to delete audiobook: ' + error.message);
        }
    }
    
    setupParameterSliders() {
        // XTTS Parameter sliders
        this.setupSlider('temperature', 0.1, 10.0, 0.1, 0.7);
        this.setupSlider('repetition_penalty', 1.0, 10.0, 0.1, 5.0);
        this.setupSlider('top_k', 10, 100, 1, 50);
        this.setupSlider('top_p', 0.1, 1.0, 0.01, 0.8);
        this.setupSlider('speed', 0.5, 3.0, 0.1, 1.0);
        
        // BARK Parameter sliders
        this.setupSlider('text_temp', 0.0, 1.0, 0.01, 0.7);
        this.setupSlider('waveform_temp', 0.0, 1.0, 0.01, 0.7);
    }
    
    setupSlider(name, min, max, step, defaultValue) {
        const slider = document.getElementById(name);
        const valueDisplay = document.getElementById(`${name}_value`);
        
        if (slider) {
            slider.min = min;
            slider.max = max;
            slider.step = step;
            slider.value = defaultValue;
            
            if (valueDisplay) {
                valueDisplay.textContent = defaultValue;
            }
            
            slider.addEventListener('input', (e) => {
                if (valueDisplay) {
                    valueDisplay.textContent = e.target.value;
                }
            });
        }
    }
                result.voices.forEach(voice => {
                    const option = document.createElement('option');
                    option.value = voice.path;
                    option.textContent = voice.name;
                    select.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Failed to load voices:', error);
            // Don't throw - just log and continue
        }
    }
    
    async loadAvailableModels() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
            
            const response = await fetch('/models/list', {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            const result = await response.json();
            
            const select = document.getElementById('custom_model_list');
            if (select) {
                select.innerHTML = '<option value="">Select from uploaded models...</option>';
                
                result.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.path;
                    option.textContent = model.name;
                    select.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Failed to load models:', error);
            // Don't throw - just log and continue
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
    
    setupParameterSliders() {
        // XTTS parameter sliders
        const xttsSliders = [
            'temperature', 'repetition_penalty', 'top_k', 'top_p', 'speed'
        ];
        
        xttsSliders.forEach(sliderId => {
            const slider = document.getElementById(sliderId);
            const valueDisplay = document.getElementById(`${sliderId}_value`);
            
            if (slider && valueDisplay) {
                slider.addEventListener('input', (e) => {
                    valueDisplay.textContent = e.target.value;
                });
            }
        });
        
        // BARK parameter sliders
        const barkSliders = ['text_temp', 'waveform_temp'];
        
        barkSliders.forEach(sliderId => {
            const slider = document.getElementById(sliderId);
            const valueDisplay = document.getElementById(`${sliderId}_value`);
            
            if (slider && valueDisplay) {
                slider.addEventListener('input', (e) => {
                    valueDisplay.textContent = e.target.value;
                });
            }
        });
    }
    
    showProgress() {
        // Progress is now shown inline in the main form
        const progressTextarea = document.getElementById('conversion_progress');
        if (progressTextarea) {
            progressTextarea.scrollIntoView({ behavior: 'smooth' });
        }
    }
    
    hideProgress() {
        // Progress textarea is always visible, just clear it
        const progressTextarea = document.getElementById('conversion_progress');
        if (progressTextarea) {
            progressTextarea.value = '';
        }
    }
    
    showDownload() {
        // Show the audiobook results section
        const audiobookResults = document.getElementById('audiobook_results');
        if (audiobookResults) {
            audiobookResults.classList.remove('d-none');
            audiobookResults.scrollIntoView({ behavior: 'smooth' });
        }
    }
    
    hideDownload() {
        // Hide the audiobook results section
        const audiobookResults = document.getElementById('audiobook_results');
        if (audiobookResults) {
            audiobookResults.classList.add('d-none');
        }
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
    
    enableConvertButton() {
        const convertBtn = document.getElementById('convert_btn');
        if (convertBtn) {
            convertBtn.disabled = false;
        }
    }
    
    disableConvertButton() {
        const convertBtn = document.getElementById('convert_btn');
        if (convertBtn) {
            convertBtn.disabled = true;
        }
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
            
            // Update audiobook display
            const audiobookText = document.getElementById('audiobook_text');
            if (audiobookText && data.filename) {
                audiobookText.value = data.filename;
            }
            
            // Setup audiobook player
            if (data.preview_url) {
                const audiobookPlayer = document.getElementById('audiobook_player');
                const audiobookSource = document.getElementById('audiobook_audio_source');
                if (audiobookPlayer && audiobookSource) {
                    audiobookSource.src = data.preview_url;
                    audiobookPlayer.load();
                }
            }
            
            // Setup download button
            const downloadBtn = document.getElementById('audiobook_download_btn');
            if (downloadBtn && data.conversion_id) {
                downloadBtn.onclick = () => {
                    window.location.href = `/download/${data.conversion_id}`;
                };
            }
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
        const progressTextarea = document.getElementById('conversion_progress');
        if (progressTextarea) {
            const timestamp = new Date().toLocaleTimeString();
            const progressLine = `[${timestamp}] ${Math.round(percentage)}% - ${message}\n`;
            progressTextarea.value += progressLine;
            progressTextarea.scrollTop = progressTextarea.scrollHeight;
        }
    }
    
    logMessage(message) {
        const progressTextarea = document.getElementById('conversion_progress');
        if (progressTextarea) {
            const timestamp = new Date().toLocaleTimeString();
            const logLine = `[${timestamp}] ${message}\n`;
            progressTextarea.value += logLine;
            progressTextarea.scrollTop = progressTextarea.scrollHeight;
        }
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