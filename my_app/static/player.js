//
// my_app/static/player.js
// TTSAudioPlayer Class Shell
// depends on event-emitter.js, audio-backend.js, and native-audio-backend.js
//

/**
 * ─────────────────────────────────────────────────────────────
 * PIPELINE CONSUMER CONTRACT (Frontend Authority)
 * ─────────────────────────────────────────────────────────────
 *
 * This player assumes the following backend pipeline semantics:
 *
 * PIPELINE STAGES
 *   - stage_1_complete:
 *       • PDF parsed
 *       • coordinate_blocks available
 *       • NO audio yet
 *
 *   - stage_2_complete:
 *       • chunks + sentences finalized
 *       • still NO audio
 *
 *   - stage_3_started:
 *       • audio generation in progress
 *
 *   - stage_3_partial:
 *       • some audio chunks available
 *       • playback is allowed
 *       • highlighting & click-seek are allowed
 *
 *   - stage_3_complete:
 *       • all chunks available
 *       • full playback enabled
 *
 * PLAYABILITY RULE
 *   Playback is allowed if:
 *     processing_status ∈ {stage_3_partial, stage_3_complete}
 *     AND ready_chunks.length > 0
 *
 * HIGHLIGHTING RULE
 *   Highlighting is enabled only if:
 *     • coordinate data exists
 *     • sentence span indices are available
 *
 * RETRY SEMANTICS
 *   - retry(bookId, force=false): resume pipeline
 *   - retry(bookId, force=true): wipe caches + rebuild
 *
 * NON-GOALS
 *   This file does NOT:
 *     • interpret extraction logic
 *     • compute alignment
 *     • modify backend artifacts
 *
 * The backend is authoritative for all timing, spans, and pages.
 */

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
            volume: 0.1,
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
            // === P1: Playback Epoch ===
            // Incremented on book load and chunk switch
            // Used to invalidate stale AUDIOPROCESS events
            playbackEpoch: 0,

            // === P4: Loading State Gate ===
            // True during loadAudiobook() execution
            // Blocks AUDIOPROCESS handling to prevent race conditions
            isLoading: false,
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
            lastHighlightedSpan: null,
            lastAutoTurnedPage: null,
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

        // === P1: Active Epoch Tracker ===
        // Snapshot of playbackEpoch when current chunk started
        // Compared against state.audiobook.playbackEpoch in AUDIOPROCESS
        this._activeEpoch = 0;

        // === P2: Warning Deduplication ===
        // Prevents log spam if sentence.page_number missing
        // Reset on each book load
        this._loggedMissingPageNumber = false;
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

        // STAGE 5: STATUS/ERROR BANNER
        this.elements.statusContainer = document.getElementById('status-container');
        this.elements.playerContainer = document.getElementById('player-container');

        const criticalElements = [
            'playPauseButton', 'speedSlider', 'volumeSlider',
            'sourceSelect', 'fileList', 'timeDisplay', 'highlightContainer',
            'statusContainer', 'playerContainer'
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
            // === P4: Block during loading ===
            if (this.state.audiobook.isLoading) return;

            // === P1: Ignore stale events from prior epoch ===
            if (this._activeEpoch !== this.state.audiobook.playbackEpoch) return;

            this.state.audio.currentTime = data.currentTime;
            this._updateTimeDisplay();
            this._updateSeekSlider();
        });

        // Bind AUDIOPROCESS (60Hz) with SEPARATED concerns
        backend.on(AudioBackend.EVENTS.AUDIOPROCESS, (data) => {
            // === P4: Block during loading ===
            if (this.state.audiobook.isLoading) return;

            // === P1: Ignore stale events from prior epoch ===
            if (this._activeEpoch !== this.state.audiobook.playbackEpoch) return;

            // PRIORITY 1: Page Sync (Smart Auto-Turn)
            if (this.state.audiobook.mode === 'audiobook' && this.state.pdf.pdfDocument) {
                const activeChunkPage = this.getPageForTimestamp(data.currentTime);

            // Only turn page if the AUDIO has moved to a new page.
            if (activeChunkPage && activeChunkPage !== this.state.ui.lastAutoTurnedPage) {
              this.queueRenderPage(activeChunkPage);
              this.state.ui.lastAutoTurnedPage = activeChunkPage;
            }
          }

          // PRIORITY 2: Highlighting (Smart Zero-Latency)
          if (this.state.audiobook.mode === 'audiobook') {
            const sentenceData = this.getSentenceAtTimestamp(data.currentTime);

            if (sentenceData) {
              const lastSpan = this.state.ui.lastHighlightedSpan;

              // ✅ SMART FIX: Only redraw if the sentence (or page) actually changed
              if (!lastSpan ||
                  sentenceData.page !== lastSpan.page || // Safety check
                  sentenceData.span_start_index !== lastSpan.start ||
                  sentenceData.span_end_index !== lastSpan.end) {

                this.clearHighlights();
                this.highlightAtTimestamp(sentenceData);

                // Update cache
                this.state.ui.lastHighlightedSpan = {
                  page: sentenceData.page,
                  start: sentenceData.span_start_index,
                  end: sentenceData.span_end_index
                };
              }
            }
          }
        });

        backend.on(AudioBackend.EVENTS.FINISH, () => {
            // === P1: Ignore stale FINISH from prior epoch ===
            if (this._activeEpoch !== this.state.audiobook.playbackEpoch) return;

            if (this.state.audiobook.mode === 'audiobook') {
                this.playNextChunk();
            } else {
                this.state.audio.isPlaying = false;
                if (this.elements.playPauseButton) {
                    this.elements.playPauseButton.textContent = 'Play';
                }
            }
        });

        backend.on(AudioBackend.EVENTS.SEEKING, (data) => {
            // === P4: Block during loading ===
            if (this.state.audiobook.isLoading) return;

            // === P1: Ignore stale events from prior epoch ===
            if (this._activeEpoch !== this.state.audiobook.playbackEpoch) return;

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

  getSentenceAtTimestamp(timestamp) {
    // ✅ FIX 2: SYNC ACCURACY
    // instead of searching by time, we use the current chunk directly.
    const chunkIndex = this.state.audiobook.currentChunkIndex;
    const chunks = this.state.audiobook.chunks || [];

    if (chunkIndex < 0 || chunkIndex >= chunks.length) return null;

    const chunk = chunks[chunkIndex];
    if (!chunk || !chunk.sentences || chunk.sentences.length === 0) return null;

    // Use REAL audio time and REAL audio duration to calculate ratio.
    // This auto-corrects any estimation errors from the backend.
    const currentTime = this.state.audio.currentTime; // From <audio> tag
    const realDuration = this.state.audio.duration;   // From <audio> metadata

    if (realDuration <= 0) return chunk.sentences[0];

    // Calculate exact percentage through the file
    const ratio = Math.min(Math.max(currentTime / realDuration, 0), 0.999);

    // Map percentage to sentence index
    const index = Math.floor(ratio * chunk.sentences.length);
    const sentence = chunk.sentences[index];

    // === P2: Use sentence-level page with fallback ===
    let page = sentence.page_number;

    if (page == null) {
      page = chunk.page;

      // Warn once per book if page_number missing (processor contract issue)
      if (!this._loggedMissingPageNumber) {
          const traceId = this.state.audiobook.manifest?.trace_id || 'unknown';
          console.warn(
              `[${traceId}] sentence.page_number missing; falling back to chunk.page`
          );
          this._loggedMissingPageNumber = true;
      }
    }

    return {
      page: page,
      span_start_index: sentence.span_start_index,
      span_end_index: sentence.span_end_index,
      highlighting_enabled: true
    };
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
            // ✅ FIX: Force transparency and blend mode (Foolproof Styling)
            highlightDiv.style.opacity = '0.4';           // 40% visible (Adjust 0.1 - 1.0)
            highlightDiv.style.mixBlendMode = 'multiply'; // Darkens text underneath (Highlighter effect)

            highlightDiv.style.left = `${screenCoords.x}px`;

            highlightDiv.style.top = `${screenCoords.y}px`;


            highlightDiv.style.width = `${screenCoords.width}px`;
            highlightDiv.style.height = `${screenCoords.height * 0.6}px`;
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
    // REPLACE existing loadAudiobook (Full code block)
    async loadAudiobook(bookId) {
        // === P4: Block event processing during load ===
        this.state.audiobook.isLoading = true;

        // === P1: Increment epoch, capture snapshot ===
        const localEpoch = ++this.state.audiobook.playbackEpoch;
        this._activeEpoch = localEpoch;

        // === P1: Clear UI caches tied to previous epoch ===
        this.state.ui.lastHighlightedSpan = null;
        this.state.ui.lastAutoTurnedPage = null;

        // === P2: Reset per-book warning flag ===
        this._loggedMissingPageNumber = false;

        try {
            this.clearError();
            this.hideBanner();
            // Clear old state (EXISTING CODE)
            this.state.audiobook.coordinateData = null;
            this.coordIndex = null;
            this.state.audiobook.hasCoordinateIndex = false;
            this.state.audiobook.chunks = null;
            this.state.audiobook.mode = 'audiobook';
            this.state.audiobook.bookId = bookId;

            console.log(`Loading audiobook and fetching contract data for: ${bookId}`);

            // Fetch status (lightweight) and full chunk data (heavy)
            const [statusResponse, coordResponse, fullChunksResponse] = await Promise.all([
                fetch(`/api/audiobook/${bookId}/status`),
                fetch(`/api/audiobook/${bookId}/coordinates`),
                fetch(`/api/audiobook/${bookId}/chunks`)
            ]);

            if (!statusResponse.ok) {
                throw new Error(`Failed to load audiobook status: ${statusResponse.status}`);
            }

            const statusData = await statusResponse.json();
            this.state.audiobook.manifest = statusData;

            // --- NEW: Status Handling and UI Decision ---
            const status = statusData.processing_status || 'unknown';

            // FIX: Allow both complete AND partial books to be playable
            const isPlayable = (status === 'stage_3_complete' || status === 'stage_3_partial');

            // Set new contract fields from backend
            this.state.audiobook.processingStatus = status;
            this.state.audiobook.errorMessage = statusData.error_message || null;
            this.state.audiobook.traceId = statusData.trace_id || null;
            this.state.audiobook.isStale = statusData.is_stale || false;
            this.state.audiobook.totalChunks = statusData.total_chunks || 0;
            this.state.audiobook.readyChunks = statusData.ready_chunks || [];
            this.state.audiobook.currentChunkIndex = 0;

            if (!isPlayable) {
                // Non-playable states: Show status banner instead of player
                // Includes: processing_started, stage_1_complete, stage_2_complete, stage_3_started, failed, unknown
                this.updateStatusBanner();
                this.elements.playerContainer.style.display = 'none';
                this.elements.statusContainer.style.display = 'block';
                return; // CRITICAL: Halts processing flow
            }

            // --- Player Initialization (Runs for stage_3_complete AND stage_3_partial) ---
            this.elements.playerContainer.style.display = 'block';
            this.elements.statusContainer.style.display = 'none';

            // Guard: Check if we have any chunks to play
            if (this.state.audiobook.readyChunks.length === 0) {
                this.logError('No audio chunks available yet. Please wait for processing.');
                this.updateStatusBanner();
                this.elements.playerContainer.style.display = 'none';
                this.elements.statusContainer.style.display = 'block';
                return;
            }

            let sentenceChunks = [];
            if (fullChunksResponse.ok) {
                sentenceChunks = await fullChunksResponse.json().then(d => d.chunks);
            }

            // CRITICAL FIX: Merge filename (EXISTING CODE)
            const finalChunks = this.state.audiobook.readyChunks.map(readyChunk => {
                const sentenceChunk = sentenceChunks.find(sc => sc.chunk_id === readyChunk.chunk_id);
                if (sentenceChunk) {
                    return {
                        ...readyChunk,
                        sentences: sentenceChunk.sentences,
                        end_time: sentenceChunk.end_time || (readyChunk.start_time + readyChunk.duration_seconds)
                    };
                }
                return readyChunk;
            });

            this.state.audiobook.chunks = finalChunks;

            // Load PDF coordinates and build the index (EXISTING CODE)
            if (coordResponse.ok) {
                const coordData = await coordResponse.json();
                this.state.audiobook.coordinateData = coordData.content;
                this.buildCoordinateIndex();
            } else {
                console.warn(`Coordinate data fetch failed (${coordResponse.status}). Highlighting disabled.`);
                this.state.audiobook.coordinateData = null;
            }

            // Update title display - include partial indicator if applicable
            const titleText = statusData.metadata?.title || bookId;
            const partialIndicator = (status === 'stage_3_partial') ? ' ⚠️ (Partial)' : '';
            this.elements.currentFileDisplay.textContent = titleText + partialIndicator;

            const pdfFilename = statusData.metadata?.source_filename;
            if (pdfFilename) {
                await this.loadPdf(pdfFilename);
            }

            // Log partial status for user awareness
            if (status === 'stage_3_partial') {
                const ready = this.state.audiobook.readyChunks.length;
                const total = this.state.audiobook.totalChunks;
                console.log(`Playing partial audiobook: ${ready}/${total} chunks available (${Math.round((ready / total) * 100)}%)`);
            }

            // === P1: Guard play() against superseded loads (epoch-consistent) ===
            // playChunk(0) may or may not increment epoch depending on currentChunkIndex,
            // so we predict the next epoch if a chunk switch occurs.
            const expectedEpoch = (0 !== this.state.audiobook.currentChunkIndex)
                ? (this.state.audiobook.playbackEpoch + 1)
                : this.state.audiobook.playbackEpoch;

            await this.playChunk(0);

            // Only play if this transition was not superseded
            if (this.state.audiobook.playbackEpoch === expectedEpoch) {
                await this.play();
            }

        } catch (error) {
            this.logError('Failed to load audiobook: ' + error.message);
            console.error('Audiobook load error:', error);
            // === P4: Disable highlighting on error ===
            this.state.audiobook.hasCoordinateIndex = false;
        } finally {
            // === P4: ALWAYS re-enable event processing ===
            this.state.audiobook.isLoading = false;
        }
    }

    hideBanner() {
        const banner = this.elements.statusContainer;
        if (banner) {
            banner.style.display = 'none';
        }
    }

    mapStatusToUI(status) {
        const statusMap = {
            'processing_started': {text: 'Starting Pipeline...', className: 'info', icon: '⚙️', buttons: ['Delete']},
            'stage_1_complete': {
                text: 'Stage 1: Text & Coordinate Indexing Complete',
                className: 'processing',
                icon: '📝',
                buttons: ['Delete']
            },
            'stage_2_complete': {
                text: 'Stage 2: Chunking Complete',
                className: 'processing',
                icon: '🧩',
                buttons: ['Delete']
            },
            'stage_3_started': {
                text: 'Stage 3: Generating Audio...',
                className: 'processing',
                icon: '🎙️',
                buttons: ['Delete']
            },
            'stage_3_partial': {
                text: 'Generation Interrupted (Resumable)',
                className: 'warning',
                icon: '⏸️',
                buttons: ['Retry', 'Delete']
            },
            'stage_3_complete': {
                text: 'Audiobook Complete',
                className: 'success',
                icon: '✅',
                buttons: ['Play', 'Delete']
            },
            'failed': {text: 'Pipeline Failed', className: 'error', icon: '❌', buttons: ['Retry', 'Delete']},
            'unknown': {
                text: 'Status Unknown (Manifest Read Error)',
                className: 'error',
                icon: '❓',
                buttons: ['Delete']
            }
        };
        return statusMap[status] || statusMap['unknown'];
    }

    updateStatusBanner() {
        const statusData = this.state.audiobook.manifest;
        if (!statusData) return;

        const status = this.state.audiobook.processingStatus;
        const uiMap = this.mapStatusToUI(status);
        const container = this.elements.statusContainer;

        let html = '';

        // 1. Primary Status Header (Top Banner Style)
        html += `<div class="status-banner ${uiMap.className}">
                <div class="banner-content">
                    <span class="banner-icon">${uiMap.icon}</span>
                    <div class="banner-text">
                        <div class="banner-message">${uiMap.text}</div>
                    </div>
                </div>
             </div>`;

        // 2. Progress, Details, and Error (Main Card Body)
        const total = statusData.total_chunks || '...';
        const ready = this.state.audiobook.readyChunks.length;
        const progress = statusData.progress_percentage || 0;
        const errorMsg = this.state.audiobook.errorMessage;
        const traceId = this.state.audiobook.traceId;

        html += `<div class="card-body">`;
        html += `<h3>${statusData.metadata?.title || statusData.book_id}</h3>`;

        // Progress Display (FIX #2: Added stage_2_complete case)
        if (status === 'stage_3_started' || status === 'stage_3_partial') {
            html += `<p>Progress: ${progress}% (${ready} of ${total} chunks)</p>`;
        } else if (status === 'processing_started' || status === 'stage_1_complete') {
            html += `<p>Status: Initializing pipeline steps...</p>`;
        } else if (status === 'stage_2_complete') {
            html += `<p>Status: Chunks prepared. Starting audio generation...</p>`;
        } else if (status === 'stage_3_complete') {
            html += `<p>Status: All ${total} chunks ready. Click Play to listen.</p>`;
        }

        // Error Message / Trace ID
        if (status === 'failed') {
            html += `<p class="error-detail">❌ Error: ${errorMsg || 'A permanent pipeline error occurred.'}</p>`;
            html += `<p class="trace-info">Trace ID: ${traceId || 'N/A'}</p>`;
        } else if (this.state.audiobook.isStale) {
            html += `<p class="warning-detail">⚠️ Warning: Job has not updated for ${statusData.stale_threshold_minutes} minutes.</p>`;
        }
        html += `</div>`;

        // 3. Action Buttons (FIX #1: Added Play case, FIX #3: Pass bookId to triggerDelete)
        html += `<div class="banner-actions">`;
        uiMap.buttons.forEach(buttonText => {
            if (buttonText === 'Retry') {
                html += `<button onclick="player.retryProcessing('${statusData.book_id}', false)" class="banner-btn primary">Resume</button>`;
                html += `<button onclick="player.retryProcessing('${statusData.book_id}', true)" class="banner-btn secondary">Force Rebuild</button>`;
            } else if (buttonText === 'Play') {
                html += `<button onclick="player.playFromBanner('${statusData.book_id}')" class="banner-btn primary">▶ Play Audiobook</button>`;
            } else if (buttonText === 'Delete') {
                html += `<button onclick="player.triggerDelete('${statusData.book_id}')" class="banner-btn secondary">Delete Job</button>`;
            }
        });
        html += `</div>`;

        container.innerHTML = html;
        container.style.display = 'block';
    }

    async retryProcessing(bookId, fullRebuild = false) {
        if (!bookId) return;

        this.state.audiobook.manifest = {
            book_id: bookId,
            processing_status: 'processing_started',
            trace_id: 'Pending...',
            total_chunks: 0,
            ready_chunks: [],
            metadata: this.state.audiobook.manifest?.metadata || {title: bookId}
        };
        this.state.audiobook.processingStatus = 'processing_started';
        this.state.audiobook.errorMessage = null;
        this.state.audiobook.bookId = bookId;

        this.updateStatusBanner();
        this.elements.playerContainer.style.display = 'none';
        this.elements.statusContainer.style.display = 'block';

        try {
            const response = await fetch(`/api/retry/${bookId}?force_rebuild=${fullRebuild}`, {
                method: 'POST'
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Retry failed: ${response.status}`);
            }

            const data = await response.json();

            this.state.audiobook.traceId = data.trace_id || 'N/A';
            this.state.audiobook.manifest.trace_id = data.trace_id || 'N/A';
            this.updateStatusBanner();

            setTimeout(async () => {
                await this.loadAudiobook(bookId);
            }, 2000);

        } catch (error) {
            this.state.audiobook.processingStatus = 'failed';
            this.state.audiobook.errorMessage = error.message;
            this.state.audiobook.manifest.processing_status = 'failed';
            this.state.audiobook.manifest.error_message = error.message;

            this.updateStatusBanner();
            console.error('Retry error:', error);
        }
    }

    playFromBanner(bookId) {
        this.hideBanner();
        this.elements.playerContainer.style.display = 'block';
        this.elements.statusContainer.style.display = 'none';
        this.loadAudiobook(bookId);
    }

    triggerDelete(bookId) {
        if (!confirm(`Are you sure you want to delete "${bookId}" and all its files? (Backend delete is currently UNIMPLEMENTED)`)) return;
        this.logError(`Delete functionality pending for ${bookId}. Please remove files manually on the host.`);
    }

    /* Advance to next chunk */
    async playNextChunk() {
        const nextIndex = this.state.audiobook.currentChunkIndex + 1;

        if (nextIndex < this.state.audiobook.readyChunks.length) {
            console.log(`Advancing to chunk ${nextIndex + 1}/${this.state.audiobook.totalChunks}`);

            // === P1: Epoch-consistent play guard ===
            // playChunk(nextIndex) increments playbackEpoch when the chunk actually changes.
            // Capture expected epoch AFTER the chunk switch.
            const epochBefore = this.state.audiobook.playbackEpoch;

            await this.playChunk(nextIndex);

            // playChunk should advance epoch by exactly 1 when chunk changes
            if (this.state.audiobook.playbackEpoch === (epochBefore + 1)) {
                await this.play();
            }

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
            const chunks = this.state.audiobook.chunks || [];

            // === P1: Epoch increment ONLY if chunk actually changes ===
            if (chunkIndex !== this.state.audiobook.currentChunkIndex) {
                const localEpoch = ++this.state.audiobook.playbackEpoch;
                this._activeEpoch = localEpoch;

                // Clear UI caches tied to previous chunk
                this.state.ui.lastHighlightedSpan = null;
                this.state.ui.lastAutoTurnedPage = null;
            }

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
      if (!pdfCoords) return;

      // Find which sentence/chunk was clicked
      const result = this.findSentenceAtCoordinates(pdfCoords.x, pdfCoords.y);

      if (result && result.chunk) {
        const targetChunk = result.chunk;

        // ✅ FIX: Logic to handle File Switching vs. Local Seeking
        // We check if the clicked chunk ID matches the currently playing chunk ID
        const currentChunk = this.state.audiobook.chunks[this.state.audiobook.currentChunkIndex];

        if (currentChunk && targetChunk.chunk_id !== currentChunk.chunk_id) {
          // Case A: Clicked a different chunk -> Load it!
          const newIndex = this.state.audiobook.chunks.indexOf(targetChunk);
          if (newIndex !== -1) {
            console.log(`Seek: Switching to Chunk ${newIndex}`);
            await this.playChunk(newIndex);
          }
        } else {
          // Case B: Clicked inside current chunk -> Replay from start
          console.log("Seek: Replaying current chunk");
          this.seekTo(0);
          if (!this.state.audio.isPlaying) {
            await this.play();
          }
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
      const chunkIndex = this.state.audiobook.currentChunkIndex;
      const chunks = this.state.audiobook.chunks || [];

      if (chunkIndex < 0 || chunkIndex >= chunks.length) {
        return null;
      }

      const chunk = chunks[chunkIndex];
      return chunk.page || null;
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

        // FIX 4.1: REMOVED minXOffset calculation.
        // Use absolute PDF space coordinates (bbox.x) directly.

        const lineBoxes = [];
        let currentLine = null;

        const avgFontSize = spans.reduce((sum, s) => sum + s.font_size, 0) / spans.length;
        const yTolerance = Math.max(3, avgFontSize * 0.4);

        spans.forEach(span => {
            const bbox = span.bbox;
            const absX = bbox.x;  // ✅ Use absolute X position
            const span_top = bbox.y;
            const span_bottom = bbox.y + bbox.height;

            if (!currentLine) {
                currentLine = {
                    x0: absX,
                    y0: span_top,
                    x1: absX + bbox.width,
                    y_top: span_top,
                    y_bottom: span_bottom,
                    count: 1
                };
            } else {
                if (Math.abs(span_top - currentLine.y_top) < yTolerance) {
                    currentLine.x0 = Math.min(currentLine.x0, absX);
                    currentLine.x1 = Math.max(currentLine.x1, absX + bbox.width);
                    currentLine.y_top = Math.min(currentLine.y_top, span_top);
                    currentLine.y_bottom = Math.max(currentLine.y_bottom, span_bottom);
                    currentLine.count++;
                } else {
                    lineBoxes.push({
                        x: currentLine.x0,
                        y: currentLine.y_top,
                        width: currentLine.x1 - currentLine.x0,
                        height: currentLine.y_bottom - currentLine.y_top,
                    });
                    currentLine = {
                        x0: absX,
                        y0: span_top,
                        x1: absX + bbox.width,
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

      const scale = viewport.scale;

      return {
        x: screenX / scale,
        y: screenY / scale
      };
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
        // FIX 5.2: Reduce cache bucket from 3s to 0.5s
        const cacheKey = `${bookId}_${Math.floor(timestamp * 2)}`;

        // FIX 5.1: Self-contained scrub/seek detection
        const lastTimestamp = this.state.ui.lastCitation?.timestamp ?? 0;
        const timeDelta = timestamp - lastTimestamp;
        const isNonLinearSeek = timeDelta < -0.1 || timeDelta > 1.0;

        // Return cached data ONLY if not scrubbing AND same bucket
        if (!isNonLinearSeek && this.state.ui.lastCitation?.cacheKey === cacheKey) {
            return this.state.ui.lastCitation.data;
        }

        try {
            const response = await fetch(`/api/audiobook/${bookId}/citation?timestamp=${timestamp}`);

            if (!response.ok) {
                throw new Error(`Citation fetch failed: ${response.status}`);
            }

            const data = await response.json();

            // Store with timestamp for future scrub detection
            this.state.ui.lastCitation = {
                data: data,
                cacheKey: cacheKey,
                timestamp: timestamp  // ✅ NEW: Enable scrub detection
            };

            return data;

        } catch (error) {
            console.warn('Citation fetch error:', error);
            return null;
        }
    }

    buildCoordinateIndex() {
        // === P3: Preflight validation of sentence span indices ===
        const traceId = this.state.audiobook.manifest?.trace_id || 'unknown';
        const chunks = this.state.audiobook.chunks || [];
        const coordData = this.state.audiobook.coordinateData || [];
        let validationPassed = true;

        for (const chunk of chunks) {
            const pageIndex = (chunk.page || 0) - 1;

            // Page bounds validation
            if (pageIndex < 0 || pageIndex >= coordData.length) {
                console.warn(
                    `[${traceId}] chunk.page ${chunk.page} out of range for coordinateData length ${coordData.length}`
                );
                validationPassed = false;
                continue;
            }

            const pageData = coordData[pageIndex];

            if (!pageData || !Array.isArray(pageData.coordinate_blocks)) {
                continue; // No coordinates for this page, skip validation
            }

            const maxIndex = pageData.coordinate_blocks.length - 1;

            for (const sentence of (chunk.sentences || [])) {
                const start = sentence.span_start_index;
                const end = sentence.span_end_index;

                // Skip sentences with missing indices
                if (start == null || end == null) continue;

                if (start < 0) {
                    console.warn(
                        `[${traceId}] P${chunk.page}: span_start_index ${start} is negative`
                    );
                    validationPassed = false;
                }

                if (end < start) {
                    console.warn(
                        `[${traceId}] P${chunk.page}: span_end_index ${end} < span_start_index ${start}`
                    );
                    validationPassed = false;
                }

                if (end > maxIndex) {
                    console.warn(
                        `[${traceId}] P${chunk.page}: span_end_index ${end} exceeds coordinate_blocks length ${maxIndex + 1}`
                    );
                    validationPassed = false;
                }
            }
        }

        if (!validationPassed) {
            console.error(
                `[${traceId}] Coordinate index validation failed — highlighting disabled`
            );
            this.state.audiobook.hasCoordinateIndex = false;
            return;
        }
        // === END P3 VALIDATION ===

        // Initialize structure
        this.coordIndex = {
            byPage: {},     // Page -> chunks mapping (stores full chunk objects)
            byTime: [],     // Sorted array of {startTime, endTime, chunkId}
            spatial: {}     // Page -> coordinate_blocks for click lookup
        };
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
}

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
        await window.player.init(null, {backend: 'wavesurfer'});
        console.log('=== TTSAudioPlayer Ready ===');
    } catch (error) {
        console.error('=== TTSAudioPlayer Initialization Failed ===');
        console.error(error);
        alert('Failed to initialize audio player. Check console for details.');
    }
});