//
// my_app/static/player.js
// Phase 0B.5: TTSAudioPlayer Class Shell
// This file REPLACES the old player.js
// It depends on event-emitter.js, audio-backend.js, and native-audio-backend.js
//

/**
 * TTSAudioPlayer
 *
 * The main application class that manages all player state,
 * DOM elements, and logic for the audio player.
 */
class TTSAudioPlayer {
  /**
   * Part 2: Player Shell (State)
   * Centralized state object. This is the single source of truth.
   * Based on Contract 1.
   */
  state = {
    audio: {
      backend: null,
      backendType: 'native',
      currentSource: 'audiobooks',
      isPlaying: false,
      currentTime: 0,
      duration: 0,
      volume: 1.0,
      playbackRate: 1.0,
      isLooping: false,
    },
    audiobook: {
      mode: 'standalone',
      bookId: null,
      manifest: null,
      currentChunkIndex: 0,
      totalChunks: 0,
      readyChunks: [],
      isProcessing: false,
      processingProgress: 0,
    },
    pdf: {
      documentUrl: null,
      pdfDocument: null,
      currentPageNum: 1,
      totalPages: 0,
      scale: 1.0,
      fitMode: 'height', //  'width' or 'height'
      isPageRendering: false,
      pendingPageNum: null,
      currentTextLayer: null,
      viewport: null,
      clickSeekEnabled: false,
    },
    polling: {
      intervalId: null,
      isActive: false,
      frequency: 2000,
      backoffMultiplier: 1.5,
      maxFrequency: 10000,
      lastUpdate: null,
    },
    ui: {
      fileListSource: 'audiobooks',
      selectedFile: null,
      citationDisplay: null,
      processingStatus: '',
      errorMessage: '',
    },
  };

  /**
   * Part 2: Player Shell (Elements)
   * Centralized object for all DOM references.
   * Populated by _queryDOMElements() during init().
   * Based on Deliverable B.
   */
  elements = {
    // Audio
    audioElement: null,
    waveformContainer: null,
    // PDF
    pdfViewerContainer: null,
    pdfCanvas: null,
    textLayerDiv: null,
    // Controls
    playPauseButton: null,
    speedSlider: null,
    volumeSlider: null,
    seekSlider: null,
    loopButton: null,
    skipBack5: null,
    skipBack10: null,
    skipForward5: null,
    skipForward10: null,
    resetButton: null,
    downloadButton: null,
    refreshButton: null,
    // Display
    speedValue: null,
    volumeValue: null,
    loopStatus: null,
    timeDisplay: null,
    currentFileDisplay: null,
    // File Management
    sourceSelect: null,
    fileList: null,
    currentSourceLabel: null,
    // PDF Processing
    pdfSelect: null,
    processPdfButton: null,
    processingStatusDiv: null,
    // Citation
    getCitationButton: null,
    citationDisplayDiv: null,
    // Error
    errorLog: null,
  };

  /**
   * Class constructor.
   */
  constructor() {
    console.log('TTSAudioPlayer constructed. Ready to init.');
  }

  // ===========================================
  // Part 4: Infrastructure (IMPLEMENTATION)
  // Core initialization and event binding logic.
  // ===========================================

  /**
   * Initialize player with DOM bindings and backend selection.
   * This is the main entry point after construction.
   * @param {string} containerId - Root container element ID (not used in Phase 0)
   * @param {object} options - Configuration overrides
   * @param {string} options.backend - 'native' | 'wavesurfer' (default: 'native')
   * @returns {Promise<void>}
   */
  async init(containerId = null, options = {}) {
    console.log('TTSAudioPlayer initializing...');

    try {
      // STEP 1: Query all DOM elements
      console.log('Step 1: Querying DOM elements...');
      this._queryDOMElements();

      // STEP 2: Initialize audio backend
      console.log('Step 2: Initializing audio backend...');
      const backendType = options.backend || 'native';
      await this._initBackend(backendType);

      // STEP 3: Bind backend event listeners
      console.log('Step 3: Binding backend event listeners...');
      this._bindBackendEvents();

      // STEP 4: Bind DOM event listeners
      console.log('Step 4: Binding DOM event listeners...');
      this._bindDOMEvents();
      // Hide PDF viewer initially
      if (this.elements.pdfViewerContainer) {
        this.elements.pdfViewerContainer.style.display = 'none';
      }
      if (this.elements.pdfCanvas) {
        this.elements.pdfCanvas.style.display = 'none';
      }

      // STEP 5: Load initial data
      console.log('Step 5: Loading initial data...');
      try {
        await this.loadAudioSources();
      } catch (error) {
        console.error('Failed to load audio sources:', error);
        // Non-fatal, continue initialization
      }

      // STEP 6: Sync UI to initial state
      console.log('Step 6: Setting initial UI state...');
      this._syncUIToState();

      console.log('TTSAudioPlayer initialization complete.');

    } catch (error) {
      console.error('TTSAudioPlayer initialization failed:', error);
      this.logError('Failed to initialize player: ' + error.message);
      throw error;
    }
  }

  /**
   * Cleanup all resources, stop polling, destroy backend.
   * @returns {Promise<void>}
   */
  async destroy() {
    console.log('TTSAudioPlayer destroying...');

    // Stop polling
    this.stopPolling();

    // Destroy backend
    if (this.state.audio.backend) {
      this.state.audio.backend.destroy();
      this.state.audio.backend = null;
    }

    // Clear all state
    this.state.audio.isPlaying = false;
    this.state.audio.currentTime = 0;
    this.state.audio.duration = 0;

    console.log('TTSAudioPlayer destroyed.');
  }

  async _initBackend(backendType) {
    if (backendType === 'native') {
      this.state.audio.backend = new NativeAudioBackend();
      await this.state.audio.backend.init(document.body, {
        audioElementId: 'audio-element'
      });
    } else if (backendType === 'wavesurfer') {
      this.state.audio.backend = new WavesurferBackend();
      await this.state.audio.backend.init(document.body, {
        waveformContainerId: 'waveform'
      });
    } else {
      throw new Error(`Unknown backend type: ${backendType}`);
    }

    this.state.audio.backendType = backendType;
    console.log(`  ✓ ${backendType} backend initialized`);
  }

  /**
   * Query and store all DOM element references.
   * Throws error if critical elements are missing.
   * @private
   */
  _queryDOMElements() {
    // AUDIO ELEMENT (for backend initialization)
    this.elements.audioElement = document.getElementById('audio-element');
    if (!this.elements.audioElement) {
      throw new Error('Critical: <audio id="audio-element"> element not found');
    }
    // PDF VIEWER
    this.elements.pdfViewerContainer = document.getElementById('pdf-viewer-container');
    this.elements.pdfCanvas = document.getElementById('pdf-canvas');
    this.elements.pageIndicator = document.getElementById('page-indicator');
    this.elements.prevPageButton = document.getElementById('prev-page-button');
    this.elements.nextPageButton = document.getElementById('next-page-button');
    this.elements.audioPlayerComponent = document.getElementById('audio-player-component');

    // PDF controls (enhanced)
    this.elements.pageJumpInput = document.getElementById('page-jump-input');
    this.elements.zoomInButton = document.getElementById('zoom-in-button');
    this.elements.zoomOutButton = document.getElementById('zoom-out-button');
    this.elements.zoomFitWidthButton = document.getElementById('zoom-fit-width-button');  // NEW
    this.elements.zoomFitHeightButton = document.getElementById('zoom-fit-height-button');
    this.elements.zoomLevel = document.getElementById('zoom-level');
    this.elements.clickSeekToggle = document.getElementById('click-seek-toggle');
    this.elements.pdfPageControls = document.getElementById('pdf-page-controls');

    // PLAYBACK CONTROLS
    this.elements.playPauseButton = document.getElementById('play-pause-button');
    this.elements.skipBack5 = document.getElementById('skip-back-5');
    this.elements.skipBack10 = document.getElementById('skip-back-10');
    this.elements.skipForward5 = document.getElementById('skip-forward-5');
    this.elements.skipForward10 = document.getElementById('skip-forward-10');
    this.elements.loopButton = document.getElementById('loop-button');

    // RANGE SLIDERS
    this.elements.speedSlider = document.getElementById('speed');
    this.elements.volumeSlider = document.getElementById('volume');
    this.elements.seekSlider = document.getElementById('seek');

    // DISPLAY ELEMENTS
    this.elements.speedValue = document.getElementById('speed-value');
    this.elements.volumeValue = document.getElementById('volume-value');
    this.elements.loopStatus = document.getElementById('loop-status');
    this.elements.timeDisplay = document.getElementById('time-display');
    this.elements.currentFileDisplay = document.getElementById('current-file');

    // UTILITY BUTTONS
    this.elements.resetButton = document.getElementById('reset-settings-button');
    this.elements.downloadButton = document.getElementById('download-button');
    this.elements.refreshButton = document.getElementById('refresh-button');

    // SOURCE/FILE MANAGEMENT
    this.elements.sourceSelect = document.getElementById('source-select');
    this.elements.currentSourceLabel = document.getElementById('current-source-label');
    this.elements.fileList = document.getElementById('files');

    // ERROR DISPLAY
    this.elements.errorLog = document.getElementById('error-log');

    // VALIDATION: Check critical elements exist
    const criticalElements = [
      'playPauseButton', 'speedSlider', 'volumeSlider',
      'sourceSelect', 'fileList', 'timeDisplay'
    ];

    for (const elementName of criticalElements) {
      if (!this.elements[elementName]) {
        throw new Error(`Critical element missing: ${elementName}`);
      }
    }

    console.log('  ✓ DOM elements queried');
  }

  /**
   * Bind backend event listeners.
   * Subscribes to all backend events and updates state/UI accordingly.
   * @private
   */
  _bindBackendEvents() {
    const backend = this.state.audio.backend;

    // READY: Audio loaded and playable
    backend.on(AudioBackend.EVENTS.READY, (data) => {
      console.log('Backend ready, duration:', data.duration);
      this.state.audio.duration = data.duration;
      this._updateTimeDisplay();

      // --- BEGIN FIX ---
      // Re-apply cached state to the newly loaded audio,
      // fixing the .load() reset bug.
      this.state.audio.backend.setPlaybackRate(this.state.audio.playbackRate);
      this.state.audio.backend.setVolume(this.state.audio.volume);
      this.state.audio.backend.setLoop(this.state.audio.isLooping);
      // --- END FIX ---

      // Enable play button (may be disabled during load)
      if (this.elements.playPauseButton) {
        this.elements.playPauseButton.disabled = false;
      }
    });

    // PLAY: Playback started
    backend.on(AudioBackend.EVENTS.PLAY, () => {
      this.state.audio.isPlaying = true;
      if (this.elements.playPauseButton) {
        this.elements.playPauseButton.textContent = 'Pause';
      }
    });

    // PAUSE: Playback paused
    backend.on(AudioBackend.EVENTS.PAUSE, () => {
      this.state.audio.isPlaying = false;
      if (this.elements.playPauseButton) {
        this.elements.playPauseButton.textContent = 'Play';
      }
    });

    // TIMEUPDATE: Time position changed
    backend.on(AudioBackend.EVENTS.TIMEUPDATE, (data) => {
      this.state.audio.currentTime = data.currentTime;
      this._updateTimeDisplay();
      this._updateSeekSlider();
    });
    // AUDIOPROCESS: High-frequency time updates (60Hz)
    backend.on(AudioBackend.EVENTS.AUDIOPROCESS, (data) => {
      // Sync PDF page
      if (this.state.audiobook.mode === 'audiobook' && this.state.pdf.pdfDocument) {
        const newPage = this.getPageForTimestamp(data.currentTime);

        // Only trigger re-render if page actually changed (debouncing)
        if (newPage && newPage !== this.state.pdf.currentPageNum) {
          this.queueRenderPage(newPage);
        }
      }
    });

    backend.on(AudioBackend.EVENTS.FINISH, () => {
      console.log('Playback finished');

      // Check mode and advance if audiobook
      if (this.state.audiobook.mode === 'audiobook') {
        console.log('Chunk finished, advancing to next...');
        this.playNextChunk();
      } else {
        // Standalone mode - just stop
        this.state.audio.isPlaying = false;
        if (this.elements.playPauseButton) {
          this.elements.playPauseButton.textContent = 'Play';
        }
      }
    });

    // SEEKING: User seeked to new position
    backend.on(AudioBackend.EVENTS.SEEKING, (data) => {
      this.state.audio.currentTime = data.currentTime;
      this._updateTimeDisplay();
      this._updateSeekSlider();
    });

    // ERROR: Backend error occurred
    backend.on(AudioBackend.EVENTS.ERROR, (data) => {
      console.error('Backend error:', data.error);
      this.logError(data.error.message);
    });

    console.log('  ✓ Backend event listeners bound');
  }

  /**
   * Bind DOM event listeners.
   * Subscribes to user interactions with UI controls.
   * PHASE 0B.5: Only play/pause button bound.
   * Additional bindings will be added incrementally in Part 5.
   * @private
   */
  _bindDOMEvents() {
    // CRITICAL: All handlers use arrow functions to preserve 'this' context

    // === PLAYBACK CONTROLS ===
    // Phase 5.1: Play/Pause (bound now for initial testing)
    this.elements.playPauseButton.addEventListener('click', () => {
      this.playPause();
    });

    // Phase 5.2: Skip controls (to be uncommented in Part 5.2)

    this.elements.skipBack5.addEventListener('click', () => {
      this.skip(-5);
    });

    this.elements.skipBack10.addEventListener('click', () => {
      this.skip(-10);
    });

    this.elements.skipForward5.addEventListener('click', () => {
      this.skip(5);
    });

    this.elements.skipForward10.addEventListener('click', () => {
      this.skip(10);
    });


    // Phase 5.3: Sliders (to be uncommented in Part 5.3)

    this.elements.speedSlider.addEventListener('input', (e) => {
      const speed = parseFloat(e.target.value);
      this.setSpeed(speed);
    });

    this.elements.volumeSlider.addEventListener('input', (e) => {
      const volume = parseFloat(e.target.value) / 100;
      this.setVolume(volume);
    });

    this.elements.seekSlider.addEventListener('input', (e) => {
      const percentage = parseFloat(e.target.value);
      const time = (percentage / 100) * this.state.audio.duration;
      this.seekTo(time);
    });

    this.elements.loopButton.addEventListener('click', () => {
      this.toggleLoop();
    });

    this.elements.resetButton.addEventListener('click', () => {
      this.resetSettings();
    });


    // Phase 5.4: File management (to be uncommented in Part 5.4)

    this.elements.sourceSelect.addEventListener('change', (e) => {
      const source = e.target.value;
      this.changeSource(source);
    });

    this.elements.refreshButton.addEventListener('click', () => {
      this.loadFileList();
    });

    // Phase 5.5: Download (to be uncommented in Part 5.5)
    /*
    this.elements.downloadButton.addEventListener('click', () => {
      this.downloadCurrentFile();
    });
    */


    // PDF Navigation
    this.elements.nextPageButton.addEventListener('click', () => {
      this.nextPage();
    });

    this.elements.prevPageButton.addEventListener('click', () => {
      this.previousPage();
    });

    // PDF zoom controls
    this.elements.zoomInButton.addEventListener('click', () => {
      this.zoomIn();
    });

    this.elements.zoomOutButton.addEventListener('click', () => {
      this.zoomOut();
    });

    this.elements.zoomFitWidthButton.addEventListener('click', () => {
      this.zoomFitWidth();
    });

    this.elements.zoomFitHeightButton.addEventListener('click', () => {
      this.zoomFitHeight();
    });

    // Page jump input (on Enter key)
    this.elements.pageJumpInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        this.jumpToPage(e.target.value);
      }
    });

    // Click-to-seek toggle
    this.elements.clickSeekToggle.addEventListener('click', () => {
      this.toggleClickSeekMode();
    });

    // PDF canvas click handler
    this.elements.pdfCanvas.addEventListener('click', (e) => {
      this.handlePdfClick(e);
    });

    console.log('  ✓ All DOM event listeners bound');
  }

  /**
   * Sync UI elements to current state values.
   * Sets initial display values and control positions.
   * @private
   */
  _syncUIToState() {
    // Set slider positions
    this.elements.speedSlider.value = this.state.audio.playbackRate;
    this.elements.volumeSlider.value = this.state.audio.volume * 100;
    this.elements.seekSlider.value = 0;

    // Set display values
    this.elements.speedValue.textContent = this.state.audio.playbackRate.toFixed(1) + 'x';
    this.elements.volumeValue.textContent = Math.round(this.state.audio.volume * 100) + '%';
    this.elements.loopStatus.textContent = this.state.audio.isLooping ? 'On' : 'Off';

    // Set initial time display
    this._updateTimeDisplay();

    // Set play/pause button text
    this.elements.playPauseButton.textContent = this.state.audio.isPlaying ? 'Pause' : 'Play';

    console.log('  ✓ UI synced to initial state');
  }

  /**
   * Update time display element.
   * Shows current time / total duration.
   * @private
   */
  _updateTimeDisplay() {
    if (!this.elements.timeDisplay) return;

    const current = this.formatTime(this.state.audio.currentTime);
    const total = this.formatTime(this.state.audio.duration);
    this.elements.timeDisplay.textContent = `${current} / ${total}`;
  }

  /**
   * Update seek slider position based on current time.
   * @private
   */
  _updateSeekSlider() {
    if (!this.elements.seekSlider) return;

    if (this.state.audio.duration > 0) {
      const percentage = (this.state.audio.currentTime / this.state.audio.duration) * 100;
      this.elements.seekSlider.value = percentage;
    } else {
      this.elements.seekSlider.value = 0;
    }
  }

  // ===========================================
  // Part 5: Controls (STUBS)
  // We will port logic into these.
  // ===========================================

  /**
   * Toggle play/pause
   * @returns {Promise<void>}
   */
  async playPause() {
    // Backend will emit PLAY or PAUSE event
    // Those events will update UI
    // We just call the backend method
    if (this.state.audio.isPlaying) {
      this.pause();
    } else {
      await this.play();
    }
  }

  /**
   * Start playback
   * @returns {Promise<void>}
   */
  async play() {
    try {
      await this.state.audio.backend.play();
      // State updated by backend PLAY event handler
    } catch (error) {
      this.logError('Playback failed: ' + error.message);  // ← Use '+'
    }
  }

  /**
   * Pause playback
   */
  pause() {
    this.state.audio.backend.pause();
    // State updated by backend PAUSE event handler
  }

  /**
     * Skip forward/backward by offset.
     * @param {number} seconds - Offset (positive = forward, negative = back)
     */
    skip(seconds) {
      const currentTime = this.state.audio.backend.getCurrentTime();
      const duration = this.state.audio.backend.getDuration();

      // Calculate and clamp new time
      const newTime = currentTime + seconds;
      const clampedTime = Math.max(0, Math.min(newTime, duration));

      // Backend handles actual seek
      this.state.audio.backend.setTime(clampedTime);
      // UI updated by backend SEEKING event handler
    }

    /**
     * Seek to absolute time position.
     * @param {number} seconds - Target time in seconds
     */
    seekTo(seconds) {
      const duration = this.state.audio.backend.getDuration();

      // Clamp to valid range
      const clampedTime = Math.max(0, Math.min(seconds, duration));

      // Backend handles actual seek
      this.state.audio.backend.setTime(clampedTime);
      // UI updated by backend SEEKING event handler
    }

  /**
   * Set playback speed.
   * Updates backend, state cache, and display.
   * @param {number} rate - Speed (0.5 - 2.0)
   */

  setSpeed(rate) {
    // Clamp to valid range
    const clampedRate = Math.max(0.5, Math.min(rate, 2.0));

    // Update backend
    this.state.audio.backend.setPlaybackRate(clampedRate);

    // Cache in state for quick access
    this.state.audio.playbackRate = clampedRate;

    // Update display
    this.elements.speedValue.textContent = clampedRate.toFixed(1) + 'x';
  }

  /**
   * Set volume.
   * Updates backend, state cache, and display.
   * @param {number} volume - Volume (0.0 - 1.0)
   */
  setVolume(volume) {
    // Clamp to valid range
    const clampedVolume = Math.max(0.0, Math.min(volume, 1.0));

    // Update backend
    this.state.audio.backend.setVolume(clampedVolume);

    // Cache in state
    this.state.audio.volume = clampedVolume;

    // Update display (convert to percentage)
    const percentage = Math.round(clampedVolume * 100);
    this.elements.volumeValue.textContent = percentage + '%';
  }

  /**
   * Toggle loop mode.
   * Uses backend abstraction instead of direct element access.
   */
  toggleLoop() {
    // Flip state
    this.state.audio.isLooping = !this.state.audio.isLooping;

    // Apply to backend
    this.state.audio.backend.setLoop(this.state.audio.isLooping);

    // Update display
    this.elements.loopStatus.textContent = this.state.audio.isLooping ? 'On' : 'Off';
  }

  /**
   * Reset all settings to defaults.
   * Speed → 1.0x, Volume → 100%, Loop → Off
   */
  resetSettings() {
    // Reset speed to 1.0x
    this.setSpeed(1.0);
    this.elements.speedSlider.value = 1.0;

    // Reset volume to 100%
    this.setVolume(1.0);
    this.elements.volumeSlider.value = 100;

    // Reset loop to off
    this.state.audio.isLooping = false;
    this.state.audio.backend.setLoop(false);
    this.elements.loopStatus.textContent = 'Off';
  }

  downloadCurrentFile() {
    console.warn('downloadCurrentFile() not yet implemented.');
  }

  // ===========================================
  // Feature 1: Source Selection (IMPLEMENTATION)
  // ===========================================

  /**
   * Load available audio sources from backend.
   */
  async loadAudioSources() {
    try {
      const response = await fetch('/api/audio_sources');
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);  // ← FIXED
      }

      const data = await response.json();
      this.elements.sourceSelect.innerHTML = '';

      if (data.sources && data.sources.length > 0) {
        const sourceLabels = {
          "audiobooks": "Audiobooks",
          "obsidian": "Obsidian Notes",
          "standalone": "Standalone Files"
        };

        data.sources.forEach(sourceName => {
          const option = document.createElement('option');
          option.value = sourceName;
          option.textContent = sourceLabels[sourceName] || sourceName;
          this.elements.sourceSelect.appendChild(option);
        });

        this.state.ui.fileListSource = data.sources[0];
        this.elements.sourceSelect.value = data.sources[0];
        await this.loadFileList();
      } else {
        this.elements.sourceSelect.innerHTML = '<option value="">No sources found</option>';
      }
    } catch (error) {
      this.logError('Failed to load audio sources: ' + error.message);
    }
  }

  /**
   * Change active source
   * @param {string} sourceName - Source identifier
   */
  async changeSource(sourceName) {
    this.state.ui.fileListSource = sourceName;
    await this.loadFileList();
  }

  /**
   * Load file list for current source
   */
  async loadFileList() {
    try {
      const source = this.state.ui.fileListSource;
      this.clearError();
      this.elements.fileList.innerHTML = '<p>Loading files...</p>';

      let apiUrl = `/api/list_audio?source=${source}`;
      if (source === 'audiobooks') {
        apiUrl = '/api/audiobooks';
      }

      const response = await fetch(apiUrl);
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);  // ← FIXED
      }

      const data = await response.json();
      this.elements.fileList.innerHTML = '';

      let filesToDisplay = [];
      if (source === 'audiobooks' && data.audiobooks) {
        filesToDisplay = data.audiobooks.map(book => ({
          id: book.book_id,
          name: book.title || book.book_id,
          isAudiobook: true
        }));
      } else if (data.files) {
        filesToDisplay = data.files.map(file => ({
          id: file.name,
          name: file.name,
          isAudiobook: false
        }));
      }

      if (filesToDisplay.length > 0) {
        filesToDisplay.forEach(file => {
          const div = document.createElement('div');
          div.className = 'file-item';
          div.textContent = file.name;
          div.dataset.filename = file.id;

          div.onclick = () => {
            document.querySelectorAll('.file-item').forEach(el => el.classList.remove('active'));
            div.classList.add('active');

            // Route to correct loader based on file type
            if (file.isAudiobook) {
              this.loadAudiobook(file.id);
            } else {
              this.loadFile(file.id, source);
            }
          };

          this.elements.fileList.appendChild(div);
        });
      } else {
        this.elements.fileList.innerHTML = '<p>No files found in this source.</p>';
      }

      const sourceLabels = {
        "audiobooks": "Audiobooks",
        "obsidian": "Obsidian Notes",
        "standalone": "Standalone Files"
      };
      this.elements.currentSourceLabel.textContent = sourceLabels[source] || source;
    } catch (error) {
      this.logError('Failed to load file list: ' + error.message);
    }
  }

  /**
   * Load single file (standalone mode)
   * @param {string} filename - File to load
   * @param {string} source - Source identifier
   */
  async loadFile(filename, source) {
    try {
      this.clearError();

      let url = `/api/audio/${this.sanitizeFilename(filename)}?source=${source}`;

      if (source === 'audiobooks') {
        console.warn('loadFile running for an audiobook. This is temporary.');
        url = `/api/audio/${this.sanitizeFilename(filename)}?source=standalone`;
        console.warn(`Temporary URL override: ${url}`);  // ← FIXED
      }

      this.state.audio.currentSource = source;
      this.state.ui.selectedFile = filename;
      this.state.audiobook.mode = 'standalone';

      this.elements.currentFileDisplay.textContent = filename;
      this.elements.playPauseButton.disabled = true;

      await this.state.audio.backend.load(url);
    } catch (error) {
      this.logError('Failed to load file: ' + error.message);
    }
  }

  // ===========================================
  // Feature 2: PDF Processing (STUBS)
  // ===========================================

  async loadAvailablePdfs() {
    console.warn('loadAvailablePdfs() not yet implemented.');
  }

  async startPdfProcessing(pdfFilename) {
    console.warn('startPdfProcessing() not yet implemented.');
  }

  // ===========================================
  // Feature 3: Polling (STUBS)
  // ===========================================

  startPolling(bookId) {
    console.warn('startPolling() not yet implemented.');
  }

  stopPolling() {
    console.warn('stopPolling() not yet implemented.');
  }

  async _pollStatus() {
    console.warn('_pollStatus() not yet implemented.');
  }

  _updateStatusUI(status) {
    console.warn('_updateStatusUI() not yet implemented.');
  }

  // ===========================================
  // Feature 4: Audiobook Playback (IMPLEMENTATION)
  // ===========================================

  /**
   * Load audiobook and start playback
   * @param {string} bookId - Audiobook identifier
   */
  async loadAudiobook(bookId) {
    try {
      this.clearError();

      // Set mode
      this.state.audiobook.mode = 'audiobook';
      this.state.audiobook.bookId = bookId;

      // Fetch manifest from status endpoint (Contract 4)
      console.log(`Loading audiobook: ${bookId}`);
      const response = await fetch(`/api/audiobook/${bookId}/status`);

      if (!response.ok) {
        throw new Error(`Failed to load audiobook manifest: ${response.status}`);
      }

      const data = await response.json();

      // Store manifest
      this.state.audiobook.manifest = data;
      this.state.audiobook.totalChunks = data.total_chunks || 0;
      this.state.audiobook.readyChunks = data.ready_chunks || [];
      this.state.audiobook.currentChunkIndex = 0;

      // Update UI
      this.elements.currentFileDisplay.textContent = data.metadata?.title || bookId;

      console.log(`Audiobook loaded: ${this.state.audiobook.totalChunks} total chunks.`);

      // --- ADD THIS (Phase 2B / E2's Step 7) ---
      // Load the corresponding PDF
      // The source_filename is in the metadata
      const pdfFilename = data.metadata?.source_filename;
      if (pdfFilename) {
        await this.loadPdf(pdfFilename);
      } else {
        console.warn('No source_filename found in manifest. PDF will not be loaded.');
        if (this.elements.pdfViewerComponent) {
          this.elements.pdfViewerComponent.style.display = 'none'; // Hide if no PDF
        }
      }
      // --- END ADDITION ---

      // Start playing first chunk
      await this.playChunk(0);

      // Tell the backend to play the chunk it just loaded.
      await this.play();

    } catch (error) {
      this.logError('Failed to load audiobook: ' + error.message);
      console.error('Audiobook load error:', error);
    }
  }

  /**
   * Advance to next chunk
   */
  async playNextChunk() {
    const nextIndex = this.state.audiobook.currentChunkIndex + 1;

    if (nextIndex < this.state.audiobook.readyChunks.length) {
      console.log(`Advancing to chunk ${nextIndex + 1}/${this.state.audiobook.totalChunks}`);
      await this.playChunk(nextIndex);

      // Tell the backend to play the new chunk.
      await this.play();
    } else {
      console.log('Reached end of available audiobook chunks');

      this.state.audio.isPlaying = false;
      if (this.elements.playPauseButton) {
        this.elements.playPauseButton.textContent = 'Play';
      }
    }
  }

  /**
   * Play specific chunk
   * @param {number} chunkIndex - Chunk index (0-based)
   */
  async playChunk(chunkIndex) {
    try {
      // Get ready chunks from state
      const chunks = this.state.audiobook.readyChunks || [];

      // Validate chunk index
      if (chunkIndex < 0 || chunkIndex >= chunks.length) {
        console.warn(`Invalid chunk index or chunk not ready: ${chunkIndex}`);
        this.logError(`Chunk ${chunkIndex} is not available or not ready.`);
        return;
      }

      // Update state
      this.state.audiobook.currentChunkIndex = chunkIndex;

      const chunk = chunks[chunkIndex];

      // Construct URL
      const bookId = this.state.audiobook.bookId;
      const chunkFilename = chunk.filename;
      const url = `/api/audiobook/${bookId}/play/${chunkFilename}`;

      console.log(`Loading chunk ${chunkIndex + 1}/${this.state.audiobook.totalChunks}: ${chunkFilename}`);

      // Disable play button during load
      this.elements.playPauseButton.disabled = true;

      // Load chunk
      await this.state.audio.backend.load(url);

    } catch (error) {
      this.logError('Failed to play chunk: ' + error.message);
      console.error('Chunk playback error:', error);
    }
  }

  async playPreviousChunk() {
    console.warn('playPreviousChunk() not yet implemented.');
  }

  getChunkForTimestamp(timestamp) {
    console.warn('getChunkForTimestamp() not yet implemented.');
    return null;
  }

  // ===========================================
  // Feature 5 & 6: PDF Sync (some STUBS)
  // ===========================================

  /**
   * Zoom in on PDF
   */
  zoomIn() {
    if (!this.state.pdf.pdfDocument) return;

    // Increase scale by 25%
    const newScale = this.state.pdf.scale * 1.25;
    const maxScale = 3.0;  // 300% max zoom

    if (newScale <= maxScale) {
      this.state.pdf.scale = newScale;
      this.renderPage(this.state.pdf.currentPageNum);
      this.updateZoomDisplay();
    }
  }

  /**
   * Zoom out on PDF
   */
  zoomOut() {
    if (!this.state.pdf.pdfDocument) return;

    // Decrease scale by 25%
    const newScale = this.state.pdf.scale * 0.75;
    const minScale = 0.5;  // 50% min zoom

    if (newScale >= minScale) {
      this.state.pdf.scale = newScale;
      this.renderPage(this.state.pdf.currentPageNum);
      this.updateZoomDisplay();
    }
  }

  /**
   * Fit PDF to width of container
   */
  zoomFitWidth() {
    if (!this.state.pdf.pdfDocument) return;

    // Set scale to null to trigger width calculation
    this.state.pdf.fitMode = 'width';  // Track fit mode
    this.state.pdf.scale = null;
    this.renderPage(this.state.pdf.currentPageNum);

    console.log('Fit to width');
  }

  /**
   * Fit PDF to height of container
   */
  zoomFitHeight() {
    if (!this.state.pdf.pdfDocument) return;

    this.state.pdf.fitMode = 'height';  // Track fit mode

    // Calculate fit-to-height scale
    // Need to get page first to know its dimensions
    this._calculateFitHeight(this.state.pdf.currentPageNum);
  }

  /**
   * Calculate and apply fit-to-height scale
   * @param {number} pageNum - Page to render
   * @private
   */
  async _calculateFitHeight(pageNum) {
    try {
      const page = await this.state.pdf.pdfDocument.getPage(pageNum);

      // Get desired height (container height minus padding)
      const desiredHeight = this.elements.pdfViewerContainer.clientHeight - 40;

      // Get page viewport at scale 1.0 to know natural dimensions
      const viewportDefault = page.getViewport({scale: 1.0});

      // Calculate scale to fit height
      const scale = desiredHeight / viewportDefault.height;

      // Store and render
      this.state.pdf.scale = scale;
      await this.renderPage(pageNum);

      console.log(`Fit to height: ${Math.round(scale * 100)}%`);

    } catch (error) {
      console.error('Failed to calculate fit-to-height:', error);
    }
  }

  /**
   * Update zoom level display
   */
  updateZoomDisplay() {
    if (!this.elements.zoomLevel) return;

    const percentage = Math.round(this.state.pdf.scale * 100);
    this.elements.zoomLevel.textContent = `${percentage}%`;
  }

  /**
   * Jump to specific page number
   * @param {number} pageNum - Page number (1-indexed)
   */
  jumpToPage(pageNum) {
    const page = parseInt(pageNum, 10);

    if (isNaN(page)) {
      this.logError('Invalid page number');
      return;
    }

    if (page < 1 || page > this.state.pdf.totalPages) {
      this.logError(`Page must be between 1 and ${this.state.pdf.totalPages}`);
      return;
    }

    this.queueRenderPage(page);

    // Clear input after successful jump
    if (this.elements.pageJumpInput) {
      this.elements.pageJumpInput.value = '';
    }
  }

  /**
   * Toggle click-to-seek mode
   */
  toggleClickSeekMode() {
    this.state.pdf.clickSeekEnabled = !this.state.pdf.clickSeekEnabled;

    // Update button appearance
    if (this.elements.clickSeekToggle) {
      if (this.state.pdf.clickSeekEnabled) {
        this.elements.clickSeekToggle.classList.add('active');
        this.elements.clickSeekToggle.textContent = '🎯 Seek: ON';
      } else {
        this.elements.clickSeekToggle.classList.remove('active');
        this.elements.clickSeekToggle.textContent = '🎯 Seek Mode';
      }
    }

    console.log(`Click-to-seek mode: ${this.state.pdf.clickSeekEnabled ? 'ON' : 'OFF'}`);
  }

  /**
   * Handle click on PDF canvas to seek audio
   * @param {MouseEvent} event - Click event
   */
  async handlePdfClick(event) {
    if (!this.state.pdf.clickSeekEnabled) return;
    if (!this.state.audiobook.manifest) return;

    // Get current page
    const currentPage = this.state.pdf.currentPageNum;

    // Find chunk for this page
    const chunk = this.state.audiobook.readyChunks.find(c => c.page === currentPage);

    if (!chunk) {
      console.warn(`No audio chunk found for page ${currentPage}`);
      this.logError(`Page ${currentPage} has no associated audio`);
      return;
    }

    // Seek to start of this chunk
    const seekTime = chunk.start_time || 0;

    console.log(`Seeking to page ${currentPage} at ${seekTime}s`);
    this.seekTo(seekTime);

    // Optional: Auto-play if paused
    if (!this.state.audio.isPlaying) {
      await this.play();
    }
  }

  /**
     * Load PDF document
     * @param {string} pdfSourceFilename - PDF file URL (just the filename)
     * @returns {Promise<void>}
     */
    async loadPdf(pdfSourceFilename) {
      try {
        // Use the existing sanitize utility
        const safeFilename = this.sanitizeFilename(pdfSourceFilename);
        const pdfUrl = `/api/pdf/${safeFilename}`;

        this.clearError();

        // Set loading state
        this.state.pdf.documentUrl = pdfUrl;

        // Load PDF (uses pdfjsLib from the script tag)
        console.log(`Loading PDF: ${pdfUrl}`);
        const loadingTask = pdfjsLib.getDocument(pdfUrl);
        this.state.pdf.pdfDocument = await loadingTask.promise;

        // Get total pages
        this.state.pdf.totalPages = this.state.pdf.pdfDocument.numPages;

        console.log(`PDF loaded: ${this.state.pdf.totalPages} pages`);

        // Show PDF viewer
        if (this.elements.pdfViewerContainer) {
          this.elements.pdfViewerContainer.style.display = 'flex';
        }
        if (this.elements.pdfCanvas) {
          this.elements.pdfCanvas.style.display = 'block';
        }

        // Render first page
        await this.renderPage(1);

        // Show PDF controls
        if (this.elements.pdfPageControls) {
          this.elements.pdfPageControls.style.display = 'flex';
        }

        // Update zoom display
        this.updateZoomDisplay();

        // Enable click-to-seek if in audiobook mode
        if (this.state.audiobook.mode === 'audiobook') {
          if (this.elements.clickSeekToggle) {
            this.elements.clickSeekToggle.disabled = false;
          }
        }

      } catch (error) {
        this.logError('Failed to load PDF: ' + error.message);
        console.error('PDF load error:', error);
        // Hide PDF viewer on error
        if (this.elements.pdfViewerContainer) {
          this.elements.pdfViewerContainer.style.display = 'none';
        }
      }
    }

    /**
     * Render specific PDF page with HiDPI support and Fit-to-Width
     * @param {number} pageNum - Page number (1-indexed)
     * @returns {Promise<void>}
     */
    async renderPage(pageNum) {
      // Validate
      if (!this.state.pdf.pdfDocument) {
        console.warn('No PDF loaded');
        return;
      }

      if (pageNum < 1 || pageNum > this.state.pdf.totalPages) {
        console.warn(`Invalid page number: ${pageNum}`);
        return;
      }

      // Set render lock (prevents simultaneous renders)
      if (this.state.pdf.isPageRendering) {
        this.state.pdf.pendingPageNum = pageNum;
        return;
      }

      this.state.pdf.isPageRendering = true;
      this.state.pdf.currentPageNum = pageNum;

      // Disable buttons during render
      if (this.elements.prevPageButton) this.elements.prevPageButton.disabled = true;
      if (this.elements.nextPageButton) this.elements.nextPageButton.disabled = true;

      try {
        // Get page
        const page = await this.state.pdf.pdfDocument.getPage(pageNum);

        // Calculate viewport with HiDPI support
        const outputScale = window.devicePixelRatio || 1;

        // ZOOM LOGIC: Use existing scale or calculate based on fit mode
        let scale = this.state.pdf.scale;

        // If scale is null/undefined, calculate based on fit mode
        if (scale === null || scale === undefined) {
          const fitMode = this.state.pdf.fitMode || 'width';  // Default to width

          if (fitMode === 'height') {
            // Calculate fit-to-height scale
            const desiredHeight = this.elements.pdfViewerContainer.clientHeight - 40;
            const viewportDefault = page.getViewport({scale: 1.0});
            scale = desiredHeight / viewportDefault.height;
          } else {
            // Calculate fit-to-width scale (default)
            const desiredWidth = this.elements.pdfViewerContainer.clientWidth - 40;
            const viewportDefault = page.getViewport({scale: 1.0});
            scale = desiredWidth / viewportDefault.width;
          }

          this.state.pdf.scale = scale;
        }
        // Otherwise, use the existing zoom scale (from zoom in/out)

        const viewport = page.getViewport({scale: scale});
        this.state.pdf.viewport = viewport;

        // Setup canvas
        const canvas = this.elements.pdfCanvas;
        const context = canvas.getContext('2d');

        // Set internal resolution (quality)
        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);

        // Set display size (layout)
        canvas.style.width = Math.floor(viewport.width) + 'px';
        canvas.style.height = Math.floor(viewport.height) + 'px';

        // Render with transform for HiDPI
        const transform = outputScale !== 1
            ? [outputScale, 0, 0, outputScale, 0, 0]
            : null;

        const renderContext = {
          canvasContext: context,
          viewport: viewport,
          transform: transform
        };

        const renderTask = page.render(renderContext);
        await renderTask.promise;

        // Update UI
        this.elements.pageIndicator.textContent =
            `Page: ${pageNum} / ${this.state.pdf.totalPages}`;

        // Update zoom display (if zoom controls exist)
        if (this.elements.zoomLevel) {
          this.updateZoomDisplay();
        }

        console.log(`Rendered page ${pageNum}`);

      } catch (error) {
        console.error('Page render error:', error);
        this.logError('Failed to render page: ' + error.message);
      } finally {
        // Release lock
        this.state.pdf.isPageRendering = false;

        // Re-enable buttons based on page number
        if (this.elements.prevPageButton) this.elements.prevPageButton.disabled = (pageNum === 1);
        if (this.elements.nextPageButton) this.elements.nextPageButton.disabled = (pageNum === this.state.pdf.totalPages);

        // Process pending page request
        if (this.state.pdf.pendingPageNum !== null) {
          const pending = this.state.pdf.pendingPageNum;
          this.state.pdf.pendingPageNum = null;
          this.renderPage(pending);
        }
      }
    }

    /**
     * Queue page render (safe for rapid calls)
     * @param {number} pageNum - Page number to render
     */
    queueRenderPage(pageNum) {
      if (this.state.pdf.isPageRendering) {
        // Canvas busy, queue request
        this.state.pdf.pendingPageNum = pageNum;
      } else {
        // Canvas free, render immediately
        this.renderPage(pageNum);
      }
    }

    /**
     * Go to next page
     */
    nextPage() {
      const nextPageNum = this.state.pdf.currentPageNum + 1;
      if (nextPageNum <= this.state.pdf.totalPages) {
        this.queueRenderPage(nextPageNum);
      }
    }

    /**
     * Go to previous page
     */
    previousPage() {
      const prevPageNum = this.state.pdf.currentPageNum - 1;
      if (prevPageNum >= 1) {
        this.queueRenderPage(prevPageNum);
      }
    }


  async _renderTextLayer(page, viewport) {
    console.warn('_renderTextLayer() not yet implemented.');
  }

  syncPdfToAudio(currentTime) {
    console.warn('syncPdfToAudio() not yet implemented.');
  }

  /**
   * Get page number for audio timestamp
   * @param {number} timestamp - Current audio time in seconds
   * @returns {number|null} Page number or null if not found
   */
  getPageForTimestamp(timestamp) {
    if (!this.state.audiobook.manifest || !this.state.audiobook.readyChunks) {
      return null;
    }

    const chunks = this.state.audiobook.readyChunks;

    // Find the chunk where the current time falls within its start/end
    // We use the global 'start_time' and 'duration_seconds' from the manifest
    const chunk = chunks.find(c =>
      timestamp >= c.start_time && timestamp < (c.start_time + c.duration_seconds)
    );

    if (chunk) {
      return chunk.page || null; // 'page' is the correct field from the manifest
    }

    // Fallback for edge cases (e.g., end of book)
    if (chunks.length > 0 && timestamp >= chunks[chunks.length - 1].start_time) {
        return chunks[chunks.length - 1].page;
    }

    return null;
  }

  async highlightAtTimestamp(timestamp) {
    console.warn('highlightAtTimestamp() not yet implemented.');
  }

  // ===========================================
  // Feature 8: Citation (STUBS)
  // ===========================================

  async getCitationAtCurrentTime() {
    console.warn('getCitationAtCurrentTime() not yet implemented.');
  }

  async fetchCitation(bookId, timestamp) {
    console.warn('fetchCitation() not yet implemented.');
  }

  displayCitation(citation) {
    console.warn('displayCitation() not yet implemented.');
  }

  // ===========================================
  // Part 3: Ported Utility Methods
  // Logic from old player.js is ported here.
  // ===========================================

  /**
   * Format seconds to MM:SS or HH:MM:SS display.
   * Handles durations over 1 hour.
   * @param {number} seconds - Time in seconds
   * @returns {string} Formatted time
   */
  formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return '0:00';

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  }

  /**
   * Log error to UI and console.
   * @param {string} message - Error message
   */
  logError(message) {
    console.error('TTSAudioPlayer Error:', message);
    this.state.ui.errorMessage = message;

    if (this.elements.errorLog) {
      this.elements.errorLog.textContent = message;
      this.elements.errorLog.style.display = 'block';
    }
  }

  /**
   * Clear error display.
   */
  clearError() {
    this.state.ui.errorMessage = '';

    if (this.elements.errorLog) {
      this.elements.errorLog.textContent = '';
      this.elements.errorLog.style.display = 'none';
    }
  }

  /**
   * Sanitize filename for API calls.
   * Removes special characters except word chars, hyphens, and dots.
   * @param {string} filename - Raw filename
   * @returns {string} Sanitized filename
   */
  sanitizeFilename(filename) {
    return filename.replace(/[^\w\-\.]/g, '').trim();
  }
} // ← CLASS ENDS HERE

// ===========================================
// Part 6: Integration (Entry Point)
// ===========================================

/**
 * Wait for the DOM to be fully loaded before initializing the player.
 */
document.addEventListener('DOMContentLoaded', async () => {
  console.log('=== TTSAudioPlayer Initialization ===');

  // Create global player instance
  window.player = new TTSAudioPlayer();

  try {
    // Initialize player with wavesurfer backend
    await window.player.init(null, { backend: 'wavesurfer' });
    console.log('=== TTSAudioPlayer Ready ===');
  } catch (error) {
    console.error('=== TTSAudioPlayer Initialization Failed ===');
    console.error(error);
    alert('Failed to initialize audio player. Check console for details.');
  }
});