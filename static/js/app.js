class VideoEditor {
    constructor() {
        this.socket = io();
        this.currentSession = null;
        this.uploadedFile = null;
        
        this.initializeElements();
        this.bindEvents();
        this.setupSocketListeners();
    }
    
    initializeElements() {
        // Upload elements
        this.uploadArea = document.getElementById('upload-area');
        this.videoInput = document.getElementById('video-input');
        this.fileInfo = document.getElementById('file-info');
        this.fileName = document.getElementById('file-name');
        
        // Process elements
        this.processControls = document.getElementById('process-controls');
        this.processBtn = document.getElementById('process-btn');
        this.progressSection = document.getElementById('progress-section');
        this.progressBar = document.getElementById('progress-bar');
        this.currentStep = document.getElementById('current-step');
        
        // Download elements
        this.downloadSection = document.getElementById('download-section');
        this.downloadBtn = document.getElementById('download-btn');
        
        // Error display
        this.errorDisplay = document.getElementById('error-display');
        
        // Preview elements
        this.previewEmpty = document.getElementById('preview-empty');
        this.videoPreview = document.getElementById('video-preview');
        this.originalVideo = document.getElementById('original-video');
        this.processedVideo = document.getElementById('processed-video');
        this.processedContainer = document.getElementById('processed-container');
        this.analysisResults = document.getElementById('analysis-results');
    }
    
    bindEvents() {
        // Upload area events
        this.uploadArea.addEventListener('click', () => this.videoInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        this.videoInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Process button
        this.processBtn.addEventListener('click', this.startProcessing.bind(this));
        
        // Download button
        this.downloadBtn.addEventListener('click', this.downloadVideo.bind(this));
    }
    
    setupSocketListeners() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        this.socket.on('processing_update', (data) => {
            this.updateProgress(data);
        });
        
        this.socket.on('processing_complete', (data) => {
            this.handleProcessingComplete(data);
        });
        
        this.socket.on('processing_error', (data) => {
            this.handleProcessingError(data);
        });
    }
    
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('drag-over');
    }
    
    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }
    
    async handleFile(file) {
        // Validate file
        const allowedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm'];
        if (!allowedTypes.includes(file.type)) {
            this.showError('Invalid file type. Please select a video file (MP4, MOV, AVI, MKV, WEBM).');
            return;
        }
        
        const maxSize = 500 * 1024 * 1024; // 500MB
        if (file.size > maxSize) {
            this.showError('File too large. Maximum size is 500MB.');
            return;
        }
        
        this.uploadedFile = file;
        this.uploadFile(file);
    }
    
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            this.showUploading();
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.currentSession = result.session_id;
                this.showUploadSuccess(result.filename);
                this.setupVideoPreview(file);
            } else {
                this.showError(result.error || 'Upload failed');
            }
        } catch (error) {
            this.showError('Upload failed: ' + error.message);
        }
    }
    
    showUploading() {
        this.uploadArea.innerHTML = `
            <div class="upload-content">
                <div class="spinner-border text-primary" role="status"></div>
                <h6 class="mt-3">Uploading...</h6>
            </div>
        `;
    }
    
    showUploadSuccess(filename) {
        this.hideError();
        this.fileInfo.classList.remove('d-none');
        this.processControls.classList.remove('d-none');
        this.fileName.textContent = filename;
        
        // Reset upload area
        this.uploadArea.innerHTML = `
            <div class="upload-content">
                <i class="fas fa-check-circle text-success upload-icon"></i>
                <h6>Upload Complete</h6>
                <p class="text-muted">Click to upload another video</p>
            </div>
        `;
    }
    
    setupVideoPreview(file) {
        this.previewEmpty.classList.add('d-none');
        this.videoPreview.classList.remove('d-none');
        
        // Set up original video preview
        const videoUrl = URL.createObjectURL(file);
        this.originalVideo.src = videoUrl;
    }
    
    async startProcessing() {
        if (!this.currentSession) {
            this.showError('No file uploaded');
            return;
        }
        
        try {
            const response = await fetch('/process', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showProcessing();
            } else {
                this.showError(result.error || 'Failed to start processing');
            }
        } catch (error) {
            this.showError('Processing failed: ' + error.message);
        }
    }
    
    showProcessing() {
        this.processControls.classList.add('d-none');
        this.progressSection.classList.remove('d-none');
        this.hideError();
    }
    
    updateProgress(data) {
        // Update progress bar
        this.progressBar.style.width = data.progress + '%';
        this.progressBar.textContent = data.progress + '%';
        
        // Update current step
        this.currentStep.textContent = data.step;
        
        // Update step indicators
        this.updateStepIndicators(data.step, data.progress);
        
        // Update metadata if available
        if (data.metadata) {
            this.updateAnalysisResults(data.metadata);
        }
    }
    
    updateStepIndicators(stepName, progress) {
        const stepMap = {
            'Loading emotion analysis model': 'loading',
            'Loading speech recognition model': 'loading',
            'Transcribing audio': 'transcription',
            'Transcription complete': 'transcription',
            'Analyzing emotions for hook detection': 'hook',
            'Hook detection complete': 'hook',
            'Extracting hook clip': 'editing',
            'Hook clip extracted': 'editing',
            'Removing silence': 'editing',
            'Silence removal complete': 'editing',
            'Generating and burning captions': 'captions',
            'Captions burned successfully': 'captions',
            'Combining hook and main video': 'finalizing',
            'Video processing complete': 'finalizing',
            'Processing complete': 'finalizing'
        };
        
        const stepKey = stepMap[stepName];
        if (stepKey) {
            const stepElement = document.querySelector(`[data-step="${stepKey}"]`);
            if (stepElement) {
                stepElement.classList.add('active');
                
                // Mark previous steps as completed
                const allSteps = document.querySelectorAll('.step');
                const currentIndex = Array.from(allSteps).indexOf(stepElement);
                
                for (let i = 0; i < currentIndex; i++) {
                    allSteps[i].classList.add('completed');
                    allSteps[i].querySelector('.step-check').classList.remove('d-none');
                }
                
                // Mark current step as completed if progress indicates so
                if (progress >= 100 || stepName.includes('complete')) {
                    stepElement.classList.add('completed');
                    stepElement.querySelector('.step-check').classList.remove('d-none');
                }
            }
        }
    }
    
    updateAnalysisResults(metadata) {
        if (metadata.hook_segment) {
            this.updateHookInfo(metadata.hook_segment);
        }
        
        if (metadata.segments_count) {
            this.updateTranscriptionInfo(metadata);
        }
        
        if (metadata.analyzed_segments) {
            this.updateEmotionSegments(metadata.analyzed_segments);
        }
        
        this.analysisResults.classList.remove('d-none');
    }
    
    updateHookInfo(hookSegment) {
        const hookDetails = document.getElementById('hook-details');
        const startTime = this.formatTime(hookSegment.start);
        const endTime = this.formatTime(hookSegment.end);
        const duration = this.formatTime(hookSegment.end - hookSegment.start);
        
        hookDetails.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <small class="text-muted">Time Range:</small>
                    <div>${startTime} - ${endTime} (${duration})</div>
                </div>
                <div class="col-md-6">
                    <small class="text-muted">Emotion Score:</small>
                    <div>${(hookSegment.emotion_score * 100).toFixed(1)}%</div>
                </div>
            </div>
            <div class="mt-2">
                <small class="text-muted">Text:</small>
                <div class="text-break">"${hookSegment.text}"</div>
            </div>
        `;
    }
    
    updateTranscriptionInfo(metadata) {
        const transcriptionDetails = document.getElementById('transcription-details');
        transcriptionDetails.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <small class="text-muted">Total Segments:</small>
                    <div>${metadata.segments_count}</div>
                </div>
                <div class="col-md-6">
                    <small class="text-muted">Duration:</small>
                    <div>${this.formatTime(metadata.total_duration || 0)}</div>
                </div>
            </div>
        `;
    }
    
    updateEmotionSegments(segments) {
        const emotionDetails = document.getElementById('emotion-details');
        
        if (segments.length === 0) {
            emotionDetails.innerHTML = '<p class="text-muted">No emotional segments analyzed yet.</p>';
            return;
        }
        
        const segmentsList = segments.map(segment => `
            <div class="emotion-segment mb-2">
                <div class="d-flex justify-content-between align-items-center">
                    <small class="text-muted">${this.formatTime(segment.start)} - ${this.formatTime(segment.end)}</small>
                    <span class="badge bg-secondary">${(segment.emotion_score * 100).toFixed(1)}%</span>
                </div>
                <div class="text-break small">"${segment.text}"</div>
            </div>
        `).join('');
        
        emotionDetails.innerHTML = segmentsList;
    }
    
    handleProcessingComplete(data) {
        this.processedContainer.classList.remove('d-none');
        this.processedVideo.src = `/preview/${data.output_filename}`;
        
        this.downloadSection.classList.remove('d-none');
        this.progressSection.classList.add('d-none');
        
        // Store the output filename for download
        this.outputFilename = data.output_filename;
    }
    
    handleProcessingError(data) {
        this.showError('Processing failed: ' + data.error);
        this.progressSection.classList.add('d-none');
        this.processControls.classList.remove('d-none');
    }
    
    downloadVideo() {
        if (this.outputFilename) {
            window.location.href = `/download/${this.outputFilename}`;
        }
    }
    
    showError(message) {
        this.errorDisplay.textContent = message;
        this.errorDisplay.classList.remove('d-none');
    }
    
    hideError() {
        this.errorDisplay.classList.add('d-none');
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VideoEditor();
});
