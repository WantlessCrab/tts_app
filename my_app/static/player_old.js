// ~/TTS/my_app/static/player.js (Phase 2 Complete - E2 Syntax Fixes Integrated)

class TTSAudioPlayer {
    constructor() {
        this.audio = document.getElementById('player');
        this.currentFile = null;
        this.isLooping = false;
        this.files = [];

        this.audioSources = [];
        this.selectedSource = "audiobooks";
        this.sourceLabels = {
            "audiobooks": "Audiobooks",
            "obsidian": "Obsidian Notes",
            "standalone": "Audio Files"
        };

        this.pollingIntervalId = null;
        this.pollCount = 0;
        this.currentBookId = null;
        this.currentBookManifest = null;
        this.processingPdfFilename = null;

        // Bind methods
        this.init = this.init.bind(this);
        this.loadFileList = this.loadFileList.bind(this);
        this.loadAudioSources = this.loadAudioSources.bind(this);
        this.handleSourceChange = this.handleSourceChange.bind(this);
        this.startPollingStatus = this.startPollingStatus.bind(this);
        this.stopPollingStatus = this.stopPollingStatus.bind(this);
        this.pollStatus = this.pollStatus.bind(this);
        this.updateUiWithStatus = this.updateUiWithStatus.bind(this);
        this.triggerPdfProcessing = this.triggerPdfProcessing.bind(this);
        this.fetchAndPlayAudiobook = this.fetchAndPlayAudiobook.bind(this);
        this.playNextChunk = this.playNextChunk.bind(this);
        this.loadAvailablePdfs = this.loadAvailablePdfs.bind(this);

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', this.init);
        } else {
            this.init();
        }
    }

    init() {
        this.setupAudioControls();
        this.setupButtonControls();
        this.setupKeyboardShortcuts();
        this.loadAudioSources().then(() => {
             this.loadFileList();
        });
        this.loadSettings();
        document.getElementById('source-select').addEventListener('change', this.handleSourceChange);
        this.loadAvailablePdfs();
        document.getElementById('process-pdf-button')?.addEventListener('click', this.triggerPdfProcessing);
    }

    setupAudioControls() {
        const speedControl = document.getElementById('speed');
        speedControl.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.audio.playbackRate = value;
            // E2#1 Fix Applied
            document.getElementById('speed-value').textContent = `${value}x`;
            this.saveSettings();
        });

        const volumeControl = document.getElementById('volume');
        volumeControl.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.audio.volume = value / 100;
            // E2#1 Fix Applied
            document.getElementById('volume-value').textContent = `${value}%`;
            this.saveSettings();
        });

        const seekControl = document.getElementById('seek');
        seekControl.addEventListener('input', (e) => {
            if (this.audio.duration) {
                const percent = parseFloat(e.target.value);
                this.audio.currentTime = (percent / 100) * this.audio.duration;
            }
        });

        this.audio.addEventListener('timeupdate', () => {
            this.updateTimeDisplay();
            this.updateSeekBar();
        });

        this.audio.addEventListener('ended', () => {
            if (this.isLooping && this.selectedSource !== 'audiobooks') {
                this.audio.play();
            }
            // Audiobook end handled in fetchAndPlayAudiobook
        });

        this.audio.addEventListener('error', (e) => {
            // E2#1 Fix Applied
            this.logError(`Audio playback error: ${e.message || 'Unknown error'}`);
        });
    }

    setupButtonControls() {
        document.getElementById('skip-back-10').addEventListener('click', () => this.skip(-10));
        document.getElementById('skip-back-5').addEventListener('click', () => this.skip(-5));
        document.getElementById('skip-forward-5').addEventListener('click', () => this.skip(5));
        document.getElementById('skip-forward-10').addEventListener('click', () => this.skip(10));
        document.getElementById('play-pause-button').addEventListener('click', () => this.togglePlay());
        document.getElementById('loop-button').addEventListener('click', () => this.toggleLoop());
        document.getElementById('reset-settings-button').addEventListener('click', () => this.resetSettings());
        document.getElementById('download-button').addEventListener('click', () => this.downloadCurrent());
        document.getElementById('refresh-button').addEventListener('click', () => {
            this.loadFileList();
            this.loadAvailablePdfs();
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

            switch(e.key) {
                case ' ': e.preventDefault(); this.togglePlay(); break;
                case 'ArrowLeft': e.preventDefault(); this.skip(-5); break;
                case 'ArrowRight': e.preventDefault(); this.skip(5); break;
                case 'ArrowUp': e.preventDefault(); this.changeVolume(5); break;
                case 'ArrowDown': e.preventDefault(); this.changeVolume(-5); break;
                case 'l': this.toggleLoop(); break;
            }
        });
    }

    async loadAudioSources() {
        const sourceSelect = document.getElementById('source-select');
        try {
            // E2#1 Fix Applied
            const response = await fetch(`/api/audio_sources`);
            // E2#1 Fix Applied
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();

            this.audioSources = data.sources || [];
            sourceSelect.innerHTML = '';

            if (this.audioSources.length === 0) {
                 sourceSelect.innerHTML = '<option value="">No sources found</option>';
                 return;
            }

            this.audioSources.forEach(sourceKey => {
                const option = document.createElement('option');
                option.value = sourceKey;
                option.textContent = this.sourceLabels[sourceKey] || sourceKey;
                sourceSelect.appendChild(option);
            });

            // E2 Undefined Variable Fix Applied
            this.selectedSource = this.audioSources.includes("audiobooks")
                                   ? "audiobooks"
                                   : this.audioSources[0];
            sourceSelect.value = this.selectedSource;
            this.updateSourceLabel();

        } catch (error) {
            console.error('Error loading audio sources:', error);
            sourceSelect.innerHTML = '<option value="">Error loading</option>';
            this.logError('Failed to load audio sources');
        }
    }

    handleSourceChange(event) {
        this.selectedSource = event.target.value;
        // E2#1 Fix Applied
        console.log(`Source changed to: ${this.selectedSource}`);
        this.updateSourceLabel();
        this.stopPollingStatus();
        this.currentBookManifest = null;
        this.currentFile = null;
        this.audio.src = '';
        document.getElementById('current-file').textContent = 'None';
        this.loadFileList();
    }

     updateSourceLabel() {
        const labelElement = document.getElementById('current-source-label');
        if (labelElement && this.selectedSource) {
             labelElement.textContent = this.sourceLabels[this.selectedSource] || this.selectedSource;
        } else if (labelElement) {
            labelElement.textContent = '...';
        }
    }

    async loadAvailablePdfs() {
        const pdfSelect = document.getElementById('pdf-select');
        if (!pdfSelect) return;

        try {
            // E2#1 Fix Applied
            const response = await fetch(`/api/available_pdfs`);
            // E2#1 Fix Applied
             if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
             const data = await response.json();

             pdfSelect.innerHTML = '<option value="">Select PDF to Process</option>';

             if (data.available_pdfs && data.available_pdfs.length > 0) {
                 data.available_pdfs.forEach(pdf => {
                     const option = document.createElement('option');
                     option.value = pdf.filename;
                      // E2#1 Fix Applied
                     option.textContent = `${pdf.filename} (${pdf.size_mb} MB)`;
                     pdfSelect.appendChild(option);
                 });
             } else {
                 pdfSelect.innerHTML = '<option value="">No PDFs found in input</option>';
             }

        } catch (error) {
             console.error("Error loading available PDFs:", error);
             pdfSelect.innerHTML = '<option value="">Error loading PDFs</option>';
        }
    }

    async triggerPdfProcessing() {
        const pdfSelect = document.getElementById('pdf-select');
        const selectedPdf = pdfSelect?.value;
        const statusDisplay = document.getElementById('processing-status');

        if (!selectedPdf) {
            this.logError("Please select a PDF file to process.");
            return;
        }

        // E2#1 Fix Applied
        if (statusDisplay) statusDisplay.textContent = `Starting processing for ${selectedPdf}...`;
        this.logError('');
        this.processingPdfFilename = selectedPdf;

        try {
            // E2#1 & E2#3 Fix Applied
            const response = await fetch(`/api/process_pdf?filename=${encodeURIComponent(selectedPdf)}`, {
                method: 'POST'
            });
             if (!response.ok) {
                 const errorData = await response.json();
                 // E2#1 Fix Applied
                 throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
             }
             const result = await response.json();

             if (result.status === "processing_started") {
                 // E2#1 Fix Applied
                 if (statusDisplay) statusDisplay.textContent = `Processing started for ${selectedPdf}. Refreshing lists...`;

                 // E2#4 Comment added
                 // TODO: Backend should return book_id immediately after starting processing,
                 // or we need a separate endpoint to map PDF filename to book_id reliably.
                 // Current approach relies on refreshing the list and hoping the title matches.

                 setTimeout(() => {
                     this.loadAvailablePdfs();
                     if(this.selectedSource === 'audiobooks') {
                        this.loadFileList();
                     }
                     this.processingPdfFilename = null;
                 }, 5000);
             } else {
                  throw new Error("Unexpected response from server.");
             }

        } catch (error) {
             console.error("Error triggering PDF processing:", error);
             // E2#1 Fix Applied
             this.logError(`Failed to start processing: ${error.message}`);
             // E2#1 Fix Applied
             if (statusDisplay) statusDisplay.textContent = `Error: ${error.message}`;
             this.processingPdfFilename = null;
        }
    }


    async loadFileList() {
        if (!this.selectedSource) {
            console.warn("No source selected, cannot load file list.");
            document.getElementById('files').innerHTML = '<p>Select an audio source above.</p>';
            return;
        }
        const statusDisplay = document.getElementById('processing-status');
        if (statusDisplay && !this.processingPdfFilename) {
             statusDisplay.textContent = '';
        }
        const filesContainer = document.getElementById('files');
        filesContainer.innerHTML = '<p>Loading files...</p>';
        try {
            let apiUrl = '';
            let isAudiobookList = false;
            if (this.selectedSource === 'audiobooks') {
                apiUrl = `/api/audiobooks`;
                isAudiobookList = true;
            } else {
                apiUrl = `/api/list_audio?source=${encodeURIComponent(this.selectedSource)}`;
                isAudiobookList = false;
            }
            console.log(`Fetching file list from: ${apiUrl}`);
            const response = await fetch(apiUrl);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            filesContainer.innerHTML = '';
            let filesToDisplay = [];
            if (isAudiobookList) {
                filesToDisplay = (data.audiobooks || []).map(book => ({
                    name: book.book_id,
                    display_name: book.title || book.book_id,
                    size_bytes: null,
                    is_complete: book.is_complete
                }));
                 this.files = filesToDisplay;
            } else {
                 filesToDisplay = data.files || [];
                 this.files = filesToDisplay;
            }
            if (filesToDisplay.length === 0) {
                filesContainer.innerHTML = `<p>No audio files found in ${this.sourceLabels[this.selectedSource] || this.selectedSource}.</p>`;
                return;
            }
            filesToDisplay.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                let sizeText = file.size_bytes !== null ? this.formatFileSize(file.size_bytes) : '';
                let statusText = file.is_complete === false ? ' (Processing)' : (file.is_complete === true ? ' (Complete)' : '');
                let displayName = file.display_name || file.name;
                fileItem.innerHTML = `
                    <span>${displayName}${statusText}</span>
                    <span>${sizeText}</span>
                `;
                fileItem.addEventListener('click', () => this.loadFile(file.name));
                filesContainer.appendChild(fileItem);
            });
            this.clearError();
        } catch (error) {
            console.error(`Error loading file list for source ${this.selectedSource}:`, error);
            this.logError(`Failed to load audio files for ${this.sourceLabels[this.selectedSource] || this.selectedSource}`);
            filesContainer.innerHTML = `<p>Error loading files.</p>`;
        }
    }


    loadFile(filename) {
         if (!this.selectedSource) {
            this.logError("Cannot load file: No audio source selected.");
            return;
        }

        this.stopPollingStatus();
        this.currentBookManifest = null;

        this.currentFile = filename;

         const isAudiobook = this.selectedSource === 'audiobooks';
         this.currentBookId = isAudiobook ? filename : null;

        if (isAudiobook) {
            this.fetchAndPlayAudiobook(filename);
        } else {
             // E2#1 Fix Applied
             this.audio.src = `/api/audio/${encodeURIComponent(filename)}?source=${encodeURIComponent(this.selectedSource)}`;
             document.getElementById('current-file').textContent = filename;
             this.updateFileListHighlighting(filename);
             this.audio.play().catch(e => console.log('Auto-play prevented:', e));
        }
    }

    async fetchAndPlayAudiobook(bookId) {
         // E2#1 Fix Applied
         console.log(`Fetching status for audiobook: ${bookId}`);
         // E2#1 Fix Applied
         document.getElementById('current-file').textContent = `${bookId} (Loading...)`;
         this.updateFileListHighlighting(bookId);

        try {
            // E2#1 Fix Applied
             const response = await fetch(`/api/audiobook/${encodeURIComponent(bookId)}/status`);
             if (!response.ok) {
                const errorData = await response.json();
                // E2#1 Fix Applied
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
             }
             const manifest = await response.json();
             this.currentBookManifest = manifest;

             document.getElementById('current-file').textContent = manifest.metadata?.title || bookId;

            if (manifest.ready_chunks && manifest.ready_chunks.length > 0) {
                const firstChunk = manifest.ready_chunks[0];
                // E2#1 Fix Applied
                this.audio.src = `/api/audiobook/${encodeURIComponent(bookId)}/play/${encodeURIComponent(firstChunk.filename)}`;

                // E2#5 Fix Applied
                this.audio.onended = null;
                this.audio.onended = () => this.playNextChunk();

                this.audio.play().catch(e => console.log('Auto-play prevented:', e));
                this.updateUiWithStatus(manifest);

                if (!manifest.is_complete) {
                    this.startPollingStatus(bookId);
                }
            } else {
                this.logError("Audiobook processing started, but no audio chunks are ready yet.");
                this.updateUiWithStatus(manifest);
                this.startPollingStatus(bookId);
            }

        } catch (error) {
             // E2#1 Fix Applied
             console.error(`Error loading audiobook ${bookId}:`, error);
             // E2#1 Fix Applied
             this.logError(`Failed to load audiobook: ${error.message}`);
             // E2#1 Fix Applied
             document.getElementById('current-file').textContent = `${bookId} (Error)`;
        }
    }

    updateFileListHighlighting(identifier) {
        document.querySelectorAll('.file-item').forEach(item => {
            const span = item.querySelector('span:first-child');
            if (span && span.textContent === identifier) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }

    playNextChunk() {
        if (!this.currentBookManifest || !this.currentBookId || this.selectedSource !== 'audiobooks') return;

        const currentSrcPath = new URL(this.audio.currentSrc).pathname;
        const currentFilename = currentSrcPath.split('/').pop();

        const currentChunkIndex = this.currentBookManifest.ready_chunks.findIndex(chunk =>
             chunk.filename === currentFilename
        );

        if (currentChunkIndex !== -1 && currentChunkIndex + 1 < this.currentBookManifest.ready_chunks.length) {
            const nextChunk = this.currentBookManifest.ready_chunks[currentChunkIndex + 1];
            // E2#1 Fix Applied
            console.log(`Playing next chunk: ${nextChunk.filename}`);
            // E2#1 Fix Applied
            this.audio.src = `/api/audiobook/${encodeURIComponent(this.currentBookId)}/play/${encodeURIComponent(nextChunk.filename)}`;
            this.audio.play();
        } else {
             console.log("End of available chunks or manifest error.");
             if(this.currentBookManifest.is_complete) {
                 this.stopPollingStatus();
             }
        }
    }

    // E2#2 Fix Applied: Simplified polling logic
    startPollingStatus(bookId) {
        this.stopPollingStatus();
        this.currentBookId = bookId;
        this.pollCount = 0;

        // E2#1 Fix Applied
        console.log(`Starting polling for ${bookId} every 3s`);

        this.pollStatus(); // Immediate first poll

        this.pollingIntervalId = setInterval(() => {
            this.pollStatus();
            this.pollCount++;
            if (this.pollCount > 100) { // E2: Add max poll limit
                console.warn("Max poll count reached, stopping polling.");
                this.stopPollingStatus();
            }
        }, 3000); // 3 seconds
    }


    stopPollingStatus() {
        if (this.pollingIntervalId) {
            console.log("Stopping polling.");
            clearInterval(this.pollingIntervalId);
            this.pollingIntervalId = null;
            this.pollCount = 0;
            this.processingPdfFilename = null;
             const statusDisplay = document.getElementById('processing-status');
             if (statusDisplay && this.currentBookManifest?.is_complete) {
                // Keep the "Complete" message for a bit
                setTimeout(() => {
                   if (statusDisplay.textContent.includes("Complete")) statusDisplay.textContent = '';
                }, 5000);
             }
        }
    }

   async pollStatus() {
        if (!this.currentBookId) return;

        try {
            // E2#1 Fix Applied
            const response = await fetch(`/api/audiobook/${encodeURIComponent(this.currentBookId)}/status`);
            if (!response.ok) {
                // E2#1 Fix Applied
                console.error(`Polling error: ${response.status}. Stopping polling.`);
                this.stopPollingStatus();
                this.logError("Polling failed: Audiobook status not found or error.");
                return;
            }
            const manifest = await response.json();

            const previousChunkCount = this.currentBookManifest?.ready_chunks?.length || 0;
            const newChunkCount = manifest.ready_chunks?.length || 0;
            this.currentBookManifest = manifest;

            this.updateUiWithStatus(manifest);

            if (manifest.is_complete) {
                console.log("Audiobook processing complete. Stopping polling.");
                this.stopPollingStatus();
            } else if (newChunkCount > previousChunkCount) {
                  // E2#1 Fix Applied
                 console.log(`Polling found new chunks for ${this.currentBookId}. Total ready: ${newChunkCount}`);
                 if (this.audio.ended && previousChunkCount > 0) {
                    console.log("Audio ended, attempting to play newly available chunk.");
                    this.playNextChunk();
                 }
            }

        } catch (error) {
            console.error("Error during polling:", error);
        }
    }

    updateUiWithStatus(manifest) {
        const statusDisplay = document.getElementById('processing-status');
        const playPauseButton = document.getElementById('play-pause-button');

        if (!manifest) return;

         if (statusDisplay) {
             // E2#1 Fix Applied
             statusDisplay.textContent = `Status: ${manifest.progress_percentage}% complete (${manifest.ready_chunks.length}/${manifest.total_chunks} chunks ready).`;
             if(manifest.is_complete) {
                 statusDisplay.textContent += " Processing Complete.";
             }
         }

        if (playPauseButton) {
             playPauseButton.disabled = !(manifest.ready_chunks && manifest.ready_chunks.length > 0);
        }
    }


    togglePlay() {
        if (!this.audio.src) {
            this.logError("No audio file loaded.");
            return;
        }
        if (this.audio.paused) {
            this.audio.play();
        } else {
            this.audio.pause();
        }
    }

    skip(seconds) {
        if (!this.audio.src) return;
        this.audio.currentTime = Math.max(0, Math.min(
            this.audio.currentTime + seconds,
            this.audio.duration || 0
        ));
    }

    toggleLoop() {
        this.isLooping = !this.isLooping;
        document.getElementById('loop-status').textContent = this.isLooping ? 'On' : 'Off';
    }

    changeVolume(delta) {
        const volumeControl = document.getElementById('volume');
        const newValue = Math.max(0, Math.min(100,
            parseInt(volumeControl.value) + delta));
        volumeControl.value = newValue;
        volumeControl.dispatchEvent(new Event('input'));
    }

    updateTimeDisplay() {
        const current = this.formatTime(this.audio.currentTime);
        const duration = this.formatTime(this.audio.duration || 0);
        // E2#1 Fix Applied
        document.getElementById('time-display').textContent = `${current} / ${duration}`;
    }

    updateSeekBar() {
        if (this.audio.duration && isFinite(this.audio.duration)) {
            const percent = (this.audio.currentTime / this.audio.duration) * 100;
            document.getElementById('seek').value = percent;
        } else {
             document.getElementById('seek').value = 0;
        }
    }

    resetSettings() {
        document.getElementById('speed').value = 1.0;
        document.getElementById('volume').value = 100;
        document.getElementById('seek').value = 0;
        document.getElementById('speed').dispatchEvent(new Event('input'));
        document.getElementById('volume').dispatchEvent(new Event('input'));
        this.isLooping = false;
        document.getElementById('loop-status').textContent = 'Off';
        localStorage.removeItem('ttsPlayerSettings');
    }

    downloadCurrent() {
        if (!this.currentFile && !this.currentBookId) {
            this.logError('No file or audiobook loaded to download');
            return;
        }
        if (!this.selectedSource) {
             this.logError('Cannot download: No audio source selected.');
            return;
        }

        const fileIdentifier = this.selectedSource === 'audiobooks' ? this.currentBookId : this.currentFile;
        if (!fileIdentifier) {
             this.logError('Cannot download: Missing file identifier.');
            return;
        }

        const link = document.createElement('a');
        if (this.selectedSource === 'audiobooks') {
            this.logError('Downloading entire audiobooks not yet implemented.');
             return;
        } else {
            // E2#1 Fix Applied
            link.href = `/api/audio/${encodeURIComponent(fileIdentifier)}?source=${encodeURIComponent(this.selectedSource)}`;
            link.download = fileIdentifier;
        }

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    saveSettings() {
        const settings = {
            speed: this.audio.playbackRate,
            volume: this.audio.volume
        };
        localStorage.setItem('ttsPlayerSettings', JSON.stringify(settings));
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('ttsPlayerSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                if (settings.speed) {
                    document.getElementById('speed').value = settings.speed;
                    this.audio.playbackRate = settings.speed;
                     // E2#1 Fix Applied
                    document.getElementById('speed-value').textContent = `${settings.speed}x`;
                }
                if (settings.volume !== undefined) {
                    const volumePercent = Math.round(settings.volume * 100);
                    document.getElementById('volume').value = volumePercent;
                    this.audio.volume = settings.volume;
                     // E2#1 Fix Applied
                    document.getElementById('volume-value').textContent = `${volumePercent}%`;
                }
            }
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    formatTime(seconds) {
        if (!seconds || !isFinite(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        // E2#1 Fix Applied (Was already correct)
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatFileSize(bytes) {
        if (!bytes || bytes < 0) return '0 B';
        if (bytes < 1024) return `${bytes} B`; // E2#1 Fix Applied
        const kb = bytes / 1024;
        // E2#1 Fix Applied
        return kb < 1024 ? `${kb.toFixed(1)} KB` : `${(kb / 1024).toFixed(1)} MB`;
    }

    logError(message) {
        const errorLog = document.getElementById('error-log'); // Assume <div id="error-log"> exists
        if (errorLog) errorLog.textContent = message;
        if(message) console.error(message);
    }

    clearError() {
        const errorLog = document.getElementById('error-log');
        if (errorLog) errorLog.textContent = '';
    }
}

// Initialize player
const player = new TTSAudioPlayer();