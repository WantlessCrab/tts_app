//
// my_app/static/player.js
// TTSAudioPlayer Class Shell
// depends on event-emitter.js, audio-backend.js, and native-audio-backend.js
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
      coordinateData: null,
      hasCoordinateIndex: false,
      currentChunkIndex: 0,
      totalChunks: 0,
      readyChunks: [],
      chunks: null, // ← FINAL CONSISTENCY FIX: Declares property used later
      isProcessing: false,
      processingProgress: 0,
    },
    pdf: {
      documentUrl: null,
      pdfDocument: null,
      currentPageNum: 1,
      totalPages: 0,
      scale: 1.0,
      fitMode: 'height',
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
      lastCitation: null,
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
    this.elements.audioElement = document.getElementById('audio-element');
    if (!this.elements.audioElement) {
      throw new Error('Critical: <audio id="audio-element"> element not found');
    }

    // PDF VIEWER
    this.elements.pdfViewerContainer = document.getElementById('pdf-viewer-container');
    this.elements.pdfCanvas = document.getElementById('pdf-canvas');
    this.elements.highlightContainer = document.getElementById('highlight-container'); // 1.1 FIX
    this.elements.pageIndicator = document.getElementById('page-indicator');
    this.elements.prevPageButton = document.getElementById('prev-page-button');
    this.elements.nextPageButton = document.getElementById('next-page-button');
    this.elements.audioPlayerComponent = document.getElementById('audio-player-component');

    // PDF controls (enhanced)
    this.elements.pageJumpInput = document.getElementById('page-jump-input');
    this.elements.zoomInButton = document.getElementById('zoom-in-button');
    this.elements.zoomOutButton = document.getElementById('zoom-out-button');
    this.elements.zoomFitWidthButton = document.getElementById('zoom-fit-width-button');
    this.elements.zoomFitHeightButton = document.getElementById('zoom-fit-height-button');
    this.elements.zoomResetButton = document.getElementById('zoom-reset-button'); // 3.3 FIX
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

    const criticalElements = [
      'playPauseButton', 'speedSlider', 'volumeSlider',
      'sourceSelect', 'fileList', 'timeDisplay', 'highlightContainer'
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

      this.state.audio.backend.setPlaybackRate(this.state.audio.playbackRate);
      this.state.audio.backend.setVolume(this.state.audio.volume);
      this.state.audio.backend.setLoop(this.state.audio.isLooping);

      if (this.elements.playPauseButton) {
        this.elements.playPauseButton.disabled = false;
      }
    });

    // PLAY/PAUSE/TIMEUPDATE/ERROR events (Unchanged)
    backend.on(AudioBackend.EVENTS.PLAY, () => {
      this.state.audio.isPlaying = true;
      if (this.elements.playPauseButton) {
        this.elements.playPauseButton.textContent = 'Pause';
      }
    });

    backend.on(AudioBackend.EVENTS.PAUSE, () => {
      this.state.audio.isPlaying = false;
      if (this.elements.playPauseButton) {
        this.elements.playPauseButton.textContent = 'Play';
      }
    });

    backend.on(AudioBackend.EVENTS.TIMEUPDATE, (data) => {
      this.state.audio.currentTime = data.currentTime;
      this._updateTimeDisplay();
      this._updateSeekSlider();
    });

    // 2.1 M2 FIX: Create throttled function for highlighting only
    this._throttledHighlightUpdate = this.throttle(async (currentTime) => {
      const bookId = this.state.audiobook.bookId;

      if (this.state.audiobook.mode !== 'audiobook' || !bookId || !this.state.pdf.pdfDocument) {
        return;
      }

      this.clearHighlights(); // Clear BEFORE fetching

      try {
        const citation = await this.fetchCitation(bookId, currentTime);

        if (citation && citation.highlighting_enabled) {
          // NOTE: Page sync logic is REMOVED from here (it's now 60Hz independent)
          this.highlightAtTimestamp(citation);
        }
      } catch (error) {
        console.warn('Highlight update failed (non-fatal):', error);
      }
    }, 100); // 100ms throttle

    // Bind AUDIOPROCESS (60Hz) with SEPARATED concerns
    backend.on(AudioBackend.EVENTS.AUDIOPROCESS, (data) => {
      // PRIORITY 1: Page Sync (Must run at 60Hz and be API-independent)
      if (this.state.audiobook.mode === 'audiobook' && this.state.pdf.pdfDocument) {
        const newPage = this.getPageForTimestamp(data.currentTime); // Local lookup
        if (newPage && newPage !== this.state.pdf.currentPageNum) {
          this.queueRenderPage(newPage); // NO await here
        }
      }

      // PRIORITY 2: Highlighting (Throttled, API-dependent)
      this._throttledHighlightUpdate(data.currentTime);
    });

    backend.on(AudioBackend.EVENTS.FINISH, () => {
      if (this.state.audiobook.mode === 'audiobook') {
        this.playNextChunk();
      } else {
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
      this.clearHighlights(); // Clear stale highlights immediately
    });

    backend.on(AudioBackend.EVENTS.ERROR, (data) => {
      console.error('Backend error:', data.error);
      this.logError(data.error.message);
    });
  }

  highlightAtTimestamp(citation) {
    // Defensive check: Highlighting requires a valid index pointer and coordinate data cache
    if (!citation || citation.span_start_index === -1 || !this.state.audiobook.coordinateData) {
      return;
    }

    const currentPage = citation.page;

    // ✅ FIX 2: Page Guard (Rendering Stability)
    // Only draw the highlight if the citation page matches the currently rendered page.
    if (currentPage !== this.state.pdf.currentPageNum) {
      return;
    }

    // Use the coordinateData cache (which is the content of the _raw.json file)
    const pageCoordData = this.state.audiobook.coordinateData[currentPage - 1];

    if (!pageCoordData || !pageCoordData.coordinate_blocks) {
      return;
    }

    const allSpans = pageCoordData.coordinate_blocks;
    const startIdx = citation.span_start_index;
    const endIdx = citation.span_end_index;

    // Slice the spans array using the indices provided by the backend
    const sentenceSpans = allSpans.slice(startIdx, endIdx + 1);

    if (sentenceSpans.length === 0) {
      return;
    }

    // Aggregate the span bounding boxes into rectangular lines
    const lineBoxes = this._aggregateSpanBounds(sentenceSpans);
    const container = this.elements.highlightContainer;

    // Draw rectangles for each line box
    lineBoxes.forEach((lineBox) => {
      const screenCoords = this.calculateCanvasCoordinates(lineBox);
      if (!screenCoords) return;

      const highlightDiv = document.createElement('div');
      highlightDiv.className = 'highlight-rect sentence current';

      highlightDiv.style.left = `${screenCoords.x}px`;

      // ✅ FINAL SPATIAL FIX: Subtract the height to correct the Y-axis inversion
      // screenCoords.y represents the bottom edge in flipped space. Subtracting height gives the true top edge.
      highlightDiv.style.top = `${screenCoords.y - screenCoords.height}px`;

      highlightDiv.style.width = `${screenCoords.width}px`;
      highlightDiv.style.height = `${screenCoords.height}px`;

      container.appendChild(highlightDiv);
    });
  }

  /**
   * Bind DOM event listeners.
   * Subscribes to user interactions with UI controls.
   * PHASE 0B.5: Only play/pause button bound.
   * Additional bindings will be added incrementally in Part 5.
   * @private
   */
  _bindDOMEvents() {
    this.elements.playPauseButton.addEventListener('click', () => {
      this.playPause();
    });
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
    this.elements.speedSlider.addEventListener('input', (e) => {
      this.setSpeed(parseFloat(e.target.value));
    });
    this.elements.volumeSlider.addEventListener('input', (e) => {
      this.setVolume(parseFloat(e.target.value) / 100);
    });
    this.elements.seekSlider.addEventListener('input', (e) => {
      this.seekTo((parseFloat(e.target.value) / 100) * this.state.audio.duration);
    });
    this.elements.loopButton.addEventListener('click', () => {
      this.toggleLoop();
    });
    this.elements.resetButton.addEventListener('click', () => {
      this.resetSettings();
    });
    this.elements.sourceSelect.addEventListener('change', (e) => {
      this.changeSource(e.target.value);
    });
    this.elements.refreshButton.addEventListener('click', () => {
      this.loadFileList();
    });
    this.elements.nextPageButton.addEventListener('click', () => {
      this.nextPage();
    });
    this.elements.prevPageButton.addEventListener('click', () => {
      this.previousPage();
    });
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

    // 3.3 FIX: Bind reset zoom button
    if (this.elements.zoomResetButton) {
      this.elements.zoomResetButton.addEventListener('click', () => {
        this.resetZoom();
      });
    }

    this.elements.pageJumpInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        this.jumpToPage(e.target.value);
      }
    });

    this.elements.clickSeekToggle.addEventListener('click', () => {
      this.toggleClickSeekMode();
    });

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

      // point to the file-server on port 8003
      let url = `http://localhost:8003/${this.sanitizeFilename(filename)}?source=${source}`;

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

      // Clear old state
      this.state.audiobook.coordinateData = null;
      this.coordIndex = null;
      this.state.audiobook.hasCoordinateIndex = false;
      this.state.audiobook.chunks = null; // Clear out old chunks

      this.state.audiobook.mode = 'audiobook';
      this.state.audiobook.bookId = bookId;

      console.log(`Loading audiobook and fetching contract data for: ${bookId}`);

      // Fetch status (lightweight) and full chunk data (heavy)
      const [statusResponse, coordResponse, fullChunksResponse] = await Promise.all([
        fetch(`/api/audiobook/${bookId}/status`),
        fetch(`/api/audiobook/${bookId}/coordinates`),
        fetch(`/api/audiobook/${bookId}/chunks`) // Full objects (with sentences)
      ]);

      if (!statusResponse.ok) {
        throw new Error(`Failed to load audiobook status: ${statusResponse.status}`);
      }

      const statusData = await statusResponse.json();
      this.state.audiobook.manifest = statusData;
      this.state.audiobook.totalChunks = statusData.total_chunks || 0;
      this.state.audiobook.readyChunks = statusData.ready_chunks || [];
      this.state.audiobook.currentChunkIndex = 0;

      let sentenceChunks = [];
      if (fullChunksResponse.ok) {
        sentenceChunks = await fullChunksResponse.json().then(d => d.chunks);
      }

      // ✅ CRITICAL FIX: Merge filename (from status/readyChunks) with sentence data (from fullChunksResponse)
      const finalChunks = this.state.audiobook.readyChunks.map(readyChunk => {
        // Find the matching sentence chunk (they share chunk_id)
        const sentenceChunk = sentenceChunks.find(sc => sc.chunk_id === readyChunk.chunk_id);

        if (sentenceChunk) {
          // Merge the required properties from sentenceChunk (sentences, end_time, etc.)
          return {
            ...readyChunk, // Base properties: chunk_id, filename, start_time, duration_seconds
            sentences: sentenceChunk.sentences, // Add sentence data
            // Ensure end_time is present for the find function
            end_time: sentenceChunk.end_time || (readyChunk.start_time + readyChunk.duration_seconds)
          };
        }
        return readyChunk; // Keep lightweight chunk if no sentence data is available
      });

      this.state.audiobook.chunks = finalChunks; // This is the new, merged list used by playChunk and indexing.

      // Load PDF coordinates and build the index (if we have coordinate data)
      if (coordResponse.ok) {
        const coordData = await coordResponse.json();
        this.state.audiobook.coordinateData = coordData.content;

        // buildCoordinateIndex now uses the merged this.state.audiobook.chunks
        this.buildCoordinateIndex();
      } else {
        console.warn(`Coordinate data fetch failed (${coordResponse.status}). Highlighting disabled.`);
        this.state.audiobook.coordinateData = null;
      }

      this.elements.currentFileDisplay.textContent = statusData.metadata?.title || bookId;

      const pdfFilename = statusData.metadata?.source_filename;
      if (pdfFilename) {
        await this.loadPdf(pdfFilename);
      }

      await this.playChunk(0);
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
      // ✅ FIX A: Use the definitive 'chunks' list (full objects) for reliability.
      // This list is populated from the new /chunks endpoint and contains all metadata.
      const chunks = this.state.audiobook.chunks || [];

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
      // FIX A: chunk.filename is now reliably available from the full chunk object
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
      this.clearHighlights();
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
      this.clearHighlights();
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
    this.clearHighlights();
    this.renderPage(this.state.pdf.currentPageNum);

    console.log('Fit to width');
  }

  /**
   * Fit PDF to height of container
   */
  zoomFitHeight() {
    if (!this.state.pdf.pdfDocument) return;

    this.state.pdf.fitMode = 'height';  // Track fit mode
    this.clearHighlights();

    // Calculate fit-to-height scale
    // Need to get page first to know its dimensions
    this._calculateFitHeight(this.state.pdf.currentPageNum);
  }

  resetZoom() {
    if (!this.state.pdf.pdfDocument) return;

    // Reset to default scale and fit mode
    this.state.pdf.scale = 1.0;
    this.state.pdf.fitMode = 'height';

    this.clearHighlights();
    this.renderPage(this.state.pdf.currentPageNum);
    this.updateZoomDisplay();
  }

  /**
   * Calculate and apply fit-to-height scale
   * @param {number} pageNum - Page to render
   * @private
   */
  async _calculateFitHeight(pageNum) {
    try {
      const page = await this.state.pdf.pdfDocument.getPage(pageNum);

      // CRITICAL FIX: The padding compensation must be removed.
      // const desiredHeight = this.elements.pdfViewerContainer.clientHeight - 40; // <-- OLD LINE
      const desiredHeight = this.elements.pdfViewerContainer.clientHeight; // <-- CORRECTED LINE

      const viewportDefault = page.getViewport({scale: 1.0});
      const scale = desiredHeight / viewportDefault.height;

      this.state.pdf.scale = scale;
      await this.renderPage(pageNum);

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

    // Update button appearance (per e2)
    if (this.elements.clickSeekToggle) {
      if (this.state.pdf.clickSeekEnabled) {
        this.elements.clickSeekToggle.classList.add('active');
        this.elements.clickSeekToggle.textContent = 'Seek: ON';
        // Update cursor for PDF canvas
        if (this.elements.pdfCanvas) {
          this.elements.pdfCanvas.style.cursor = 'pointer';
        }
      } else {
        this.elements.clickSeekToggle.classList.remove('active');
        this.elements.clickSeekToggle.textContent = 'Seek Mode';
        // Reset cursor
        if (this.elements.pdfCanvas) {
          this.elements.pdfCanvas.style.cursor = 'default';
        }
      }
    }

    console.log(`Click-to-seek mode: ${this.state.pdf.clickSeekEnabled ? 'ON' : 'OFF'}`);
  }

  /**
   * Handle click on PDF canvas to seek audio
   * @param {MouseEvent} event - Click event
   */
  /**
   * MILESTONE 3: Handle click on PDF canvas to seek audio
   * @param {MouseEvent} event - Click event
   */
  async handlePdfClick(event) {
    if (!this.state.pdf.clickSeekEnabled || !this.state.audiobook.hasCoordinateIndex) return;

    const canvas = this.elements.pdfCanvas;
    const rect = canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    const pdfCoords = this.screenToPdfCoordinates(clickX, clickY);
    if (!pdfCoords) {
      return;
    }

    // Find the sentence using the spatial index
    const sentence = this.findSentenceAtCoordinates(pdfCoords.x, pdfCoords.y);

    if (sentence && sentence.chunk) {
      const seekTime = sentence.chunk.start_time;
      this.seekTo(seekTime);

      if (!this.state.audio.isPlaying) {
        await this.play();
      }
    }
  }

  findSentenceAtCoordinates(pdfX, pdfY) {
    const currentPage = this.state.pdf.currentPageNum;
    const pageData = this.state.audiobook.coordinateData?.find(p => p.page_number === currentPage);

    // This check is now redundant due to hasCoordinateIndex guard in handlePdfClick, but kept for function integrity.
    if (!pageData || !this.coordIndex?.byPage[currentPage]) return null;

    // 1. Find the coordinate block that contains the click (spatial search)
    const allBlocks = pageData.coordinate_blocks;
    const clickedBlock = allBlocks.find(block => {
      const bbox = block.bbox;
      return pdfX >= bbox.x &&
          pdfX <= bbox.x + bbox.width &&
          pdfY >= bbox.y &&
          pdfY <= bbox.y + bbox.height;
    });

    if (!clickedBlock) return null;

    // 2. Find the index of the clicked span
    const clickedSpanIndex = allBlocks.indexOf(clickedBlock);

    // ✅ FIX 5: The index now contains full chunk objects.
    const chunks = this.coordIndex.byPage[currentPage];

    // 3. Search the index for the chunk/sentence that owns this span index.
    for (const chunk of chunks) {
      // chunk is now the full chunk object (containing .sentences)
      if (chunk.sentences) {
        for (const sentence of chunk.sentences) {
          if (clickedSpanIndex >= sentence.span_start_index && clickedSpanIndex <= sentence.span_end_index) {
            return {
              chunk: chunk,
              page: currentPage,
              sentence: sentence
            };
          }
        }
      }
    }

    return null;
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
            // MILESTONE 3: Make button visible (per e2)
            this.elements.clickSeekToggle.style.opacity = '1';
            this.elements.clickSeekToggle.style.cursor = 'pointer';
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
      if (!this.state.pdf.pdfDocument) {
        console.warn('No PDF loaded');
        return;
      }
      this.clearHighlights();

      if (pageNum < 1 || pageNum > this.state.pdf.totalPages) {
        console.warn(`Invalid page number: ${pageNum}`);
        return;
      }

      // ACTION 2.4: Clear highlights immediately after validation
      this.clearHighlights();

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
        const page = await this.state.pdf.pdfDocument.getPage(pageNum);
        const outputScale = window.devicePixelRatio || 1;

        // ZOOM LOGIC
        let scale = this.state.pdf.scale;

        if (scale === null || scale === undefined) {
          const fitMode = this.state.pdf.fitMode || 'width';
          const viewportDefault = page.getViewport({scale: 1.0});

          if (fitMode === 'height') {
            // Removed padding compensation (CSS fix handles padding externally)
            const desiredHeight = this.elements.pdfViewerContainer.clientHeight;
            scale = desiredHeight / viewportDefault.height;
          } else {
            // Removed padding compensation (CSS fix handles padding externally)
            const desiredWidth = this.elements.pdfViewerContainer.clientWidth;
            scale = desiredWidth / viewportDefault.width;
          }

          this.state.pdf.scale = scale;
        }

        const viewport = page.getViewport({scale: scale});
        this.state.pdf.viewport = viewport;

        const canvas = this.elements.pdfCanvas;
        const context = canvas.getContext('2d');

        // Set internal resolution (quality)
        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);

        // Set display size (layout)
        canvas.style.width = Math.floor(viewport.width) + 'px';
        canvas.style.height = Math.floor(viewport.height) + 'px';

        // 4.3 Highlight Scroll Fix: Set container size to match canvas
        this.elements.highlightContainer.style.width = canvas.style.width;
        this.elements.highlightContainer.style.height = canvas.style.height;

        // CRITICAL: Calculate the offset caused by the flexbox centering the canvas
        const canvasRect = canvas.getBoundingClientRect();
        const containerRect = this.elements.pdfViewerContainer.getBoundingClientRect();

        // Calculate the offset (distance from container's edge to canvas's edge)
        const offsetLeft = canvasRect.left - containerRect.left;
        const offsetTop = canvasRect.top - containerRect.top;

        // Position highlight container to match canvas's offset from the parent container
        this.elements.highlightContainer.style.left = `${offsetLeft}px`;
        this.elements.highlightContainer.style.top = `${offsetTop}px`;

        // Render with transform for HiDPI
        const transform = outputScale !== 1
            ? [outputScale, 0, 0, outputScale, 0, 0]
            : null;

        const renderContext = {
          canvasContext: context,
          viewport: viewport,
          transform: transform
        };

        await page.render(renderContext).promise;

        // Update UI
        this.elements.pageIndicator.textContent =
            `Page: ${pageNum} / ${this.state.pdf.totalPages}`;

        if (this.elements.zoomLevel) {
          this.updateZoomDisplay();
        }

      } catch (error) {
        console.error('Page render error:', error);
        this.logError('Failed to render page: ' + error.message);
      } finally {
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
    // FIX: Use the definitive 'chunks' list (populated with full objects from API)
    const chunks = this.state.audiobook.chunks || [];

    if (chunks.length === 0) {
      return null;
    }

    // Find the chunk where the current time falls within its start/end
    const chunk = chunks.find(c =>
        // Use the end_time property now correctly enforced by process.py
        timestamp >= c.start_time && timestamp < c.end_time
    );

    if (chunk) {
      return chunk.page || null; // 'page' is the correct field from the manifest/chunk data
    }

    // Fallback for edge cases (e.g., end of book)
    // Use the last chunk's page if time exceeds the known end time
    if (chunks.length > 0 && timestamp >= chunks[chunks.length - 1].end_time) {
      return chunks[chunks.length - 1].page;
    }

    return null;
  }

  calculateCanvasCoordinates(pdfCoords) {
    const viewport = this.state.pdf.viewport;

    if (!viewport) {
      return null;
    }

    // ✅ FINAL SPATIAL FIX (Simple Fix): Bypass the full pdfjsLib.Util.applyTransform
    // The backend coordinates are already in Y-DOWN page space (top-left origin).
    // We only need to scale them to screen pixels using viewport.scale.
    const scale = viewport.scale;

    // NOTE: This assumes no page rotation/skew, which is true for standard documents.
    return {
      x: pdfCoords.x * scale,
      y: pdfCoords.y * scale,
      width: pdfCoords.width * scale,
      height: pdfCoords.height * scale
    };
  }

  _aggregateSpanBounds(spans) {
    if (spans.length === 0) return [];

    // FIX B: Determine the minimum X-offset to normalize the coordinates
    // This value represents the fixed left margin of the PDF page body (368.16 in your test)
    const minXOffset = spans.reduce((min, span) => Math.min(min, span.bbox.x), Infinity);

    const lineBoxes = [];
    let currentLine = null;

    // Calculate tolerance dynamically based on average font size
    const avgFontSize = spans.reduce((sum, s) => sum + s.font_size, 0) / spans.length;
    const yTolerance = Math.max(3, avgFontSize * 0.4);

    spans.forEach(span => {
      const bbox = span.bbox;

      // Calculate normalized X-coordinate for this span
      const normalizedX = bbox.x - minXOffset; // Apply normalization

      const span_top = bbox.y;
      const span_bottom = bbox.y + bbox.height;

      if (!currentLine) {
        // Initialize line using normalized X
        currentLine = {
          x0: normalizedX,
          y0: span_top,
          x1: normalizedX + bbox.width, // Use normalized X + original width
          y_top: span_top,
          y_bottom: span_bottom,
          count: 1
        };
      } else {
        // Check vertical proximity using the line's top edge (y_top) for stability
        if (Math.abs(span_top - currentLine.y_top) < yTolerance) {
          // Merge horizontally
          currentLine.x0 = Math.min(currentLine.x0, normalizedX);
          currentLine.x1 = Math.max(currentLine.x1, normalizedX + bbox.width);
          currentLine.y_top = Math.min(currentLine.y_top, span_top);
          currentLine.y_bottom = Math.max(currentLine.y_bottom, span_bottom);
          currentLine.count++;
        } else {
          // Span is on a new line, finalize the old one
          lineBoxes.push({
            x: currentLine.x0,
            y: currentLine.y_top,
            width: currentLine.x1 - currentLine.x0,
            height: currentLine.y_bottom - currentLine.y_top,
          });

          // Start new line with normalized X
          currentLine = {
            x0: normalizedX,
            y0: span_top,
            x1: normalizedX + bbox.width,
            y_top: span_top,
            y_bottom: span_bottom,
            count: 1
          };
        }
      }
    });

    if (currentLine) {
      lineBoxes.push({
        x: currentLine.x0,
        y: currentLine.y_top,
        width: currentLine.x1 - currentLine.x0,
        height: currentLine.y_bottom - currentLine.y_top,
      });
    }

    return lineBoxes;
  }

  screenToPdfCoordinates(screenX, screenY) {
    const viewport = this.state.pdf.viewport;
    if (!viewport) return null;

    const inverseTransform = viewport.getInverseTransform();
    const [pdfX, pdfY] = pdfjsLib.Util.applyTransform([screenX, screenY], inverseTransform);

    return {x: pdfX, y: pdfY};
  }

  /**
   * MILESTONE 3: Find chunk that contains the given PDF coordinates
   * (Correct, high-performance version - uses local index)
   * @param {number} pdfX - X coordinate in PDF space
   * @param {number} pdfY - Y coordinate in PDF space
   * @returns {object|null} Chunk data or null if not found
   */
  findChunkAtCoordinates(pdfX, pdfY) {
    const currentPage = this.state.pdf.currentPageNum;
    const manifest = this.state.audiobook.manifest;

    if (!manifest || !manifest.content) return null;

    const pageIdx = currentPage - 1;
    if (pageIdx < 0 || pageIdx >= manifest.content.length) return null;

    // Get all coordinates for the current page
    const coordinate_blocks = manifest.content[pageIdx].coordinate_blocks || [];

    // Find the first block that contains the click
    const clickedBlock = coordinate_blocks.find(block => {
      const bbox = block.bbox;
      return pdfX >= bbox.x &&
          pdfX <= bbox.x + bbox.width &&
          pdfY >= bbox.y &&
          pdfY <= bbox.y + bbox.height;
    });

    if (!clickedBlock) {
      return null; // No block found at this location
    }

    // Now, find which audio chunk this block's text belongs to.
    // (This is a simpler, good-enough approximation for now)
    const chunks = this.state.audiobook.chunks || [];
    const clickedText = clickedBlock.text.trim().toLowerCase();

    for (const chunk of chunks) {
      if (chunk.page !== currentPage) continue;

      const chunkText = chunk.text.trim().toLowerCase();
      if (chunkText.includes(clickedText)) {
        return chunk; // Found the matching chunk
      }
    }

    return null; // No chunk contained that text
  }

  // ===========================================
  // Feature 8: Citation (STUBS)
  // ===========================================

  async getCitationAtCurrentTime() {
    console.warn('getCitationAtCurrentTime() not yet implemented.');
  }

  async fetchCitation(bookId, timestamp) {
    // Check cache first
    const cacheKey = `${bookId}_${Math.floor(timestamp / 3)}`;
    if (this.state.ui.lastCitation?.cacheKey === cacheKey) {
      return this.state.ui.lastCitation.data; // Return the cached data payload
    }

    try {
      // FIX: Corrected syntax for fetch URL construction
      const response = await fetch(`/api/audiobook/${bookId}/citation?timestamp=${timestamp}`);

      if (!response.ok) {
        throw new Error(`Citation fetch failed: ${response.status}`);
      }

      const data = await response.json();

      // Store latest citation for caching
      this.state.ui.lastCitation = {
        data: data, // Store the API payload
        cacheKey: cacheKey // Store the timestamp key
      };

      return data;

    } catch (error) {
      console.warn('Citation fetch error:', error);
      return null; // Graceful degradation
    }
  }

  buildCoordinateIndex() {
    // Initialize structure
    this.coordIndex = {
      byPage: {},     // Page -> chunks mapping (stores full chunk objects)
      byTime: [],     // Sorted array of {startTime, endTime, chunkId}
      spatial: {}     // Page -> coordinate_blocks for click lookup
    };

    // ✅ FIX 5: Use this.state.audiobook.chunks (full objects with sentences)
    const chunks = this.state.audiobook.chunks || [];
    let hasSentences = false;

    chunks.forEach(chunk => {
      const pageNum = chunk.page;

      // Time-based index
      this.coordIndex.byTime.push({
        startTime: chunk.start_time,
        endTime: chunk.end_time || (chunk.start_time + chunk.duration_seconds), // Fallback for safety
        chunkId: chunk.chunk_id,
        page: pageNum
      });

      // ✅ FIX 5: Page-based index: Store the full chunk object reference.
      if (!this.coordIndex.byPage[pageNum]) {
        this.coordIndex.byPage[pageNum] = [];
      }
      this.coordIndex.byPage[pageNum].push(chunk); // Store full chunk object with sentences

      if (chunk.sentences && chunk.sentences.length > 0) {
        hasSentences = true;
      }
    });

    // Sort byTime for efficient binary search
    this.coordIndex.byTime.sort((a, b) => a.startTime - b.startTime);

    // Populate spatial index from coordinateData
    const coordData = this.state.audiobook.coordinateData || [];
    coordData.forEach(pageData => {
      const pageNum = pageData.page_number;
      this.coordIndex.spatial[pageNum] = pageData.coordinate_blocks || [];
    });

    // ✅ FIX 5: CRITICAL: Set the flag now that the index is built and data is available.
    this.state.audiobook.hasCoordinateIndex = (coordData.length > 0 && hasSentences);

    console.log(`Index built: ${chunks.length} chunks, ${Object.keys(this.coordIndex.spatial).length} pages with coordinates. Index ready: ${this.state.audiobook.hasCoordinateIndex}`);
  }

  /**
   * Clear all highlight rectangles from the PDF overlay.
   * Called before rendering new highlights or changing pages/zoom.
   */
  clearHighlights() {
    const container = this.elements.highlightContainer;
    if (!container) {
      console.warn('Highlight container not found');
      return;
    }

    // Remove all child divs (highlight rectangles)
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
  }

  /**
   * MILESTONE 2: Throttle function to limit update frequency
   * Preserves 'this' context and handles async functions
   */
  throttle(func, delay) {
    let timeoutId;
    let lastExecTime = 0;

    return (...args) => {  // ← CHANGE: Arrow function preserves context
      const currentTime = Date.now();

      // Execute immediately if enough time has passed
      if (currentTime - lastExecTime > delay) {
        lastExecTime = currentTime;
        func.apply(this, args);  // 'this' now correctly refers to TTSAudioPlayer
      } else {
        // Schedule for later
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          lastExecTime = Date.now();
          func.apply(this, args);
        }, delay - (currentTime - lastExecTime));
      }
    };
  }

  // Find chunk by timestamp using binary search
  findChunkByTime(timestamp) {
    if (!this.coordIndex || !this.coordIndex.byTime) return null;

    const times = this.coordIndex.byTime;
    let left = 0, right = times.length - 1;

    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      const chunk = times[mid];

      if (timestamp >= chunk.startTime && timestamp < chunk.endTime) {
        return chunk;
      } else if (timestamp < chunk.startTime) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    return null;
  }

  // Find chunks on current page
  getChunksForPage(pageNum) {
    if (!this.coordIndex || !this.coordIndex.byPage) return [];
    return this.coordIndex.byPage[pageNum] || [];
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