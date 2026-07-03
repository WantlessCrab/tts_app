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
 *       • PDF geometry extracted
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

/******************************************************************************
 * SECTION 0 — GLOBAL STATE & CONSTANTS
 ******************************************************************************/

/**
 * @typedef {Object} NormalizedBbox
 * @property {number} x      - Left edge (PDF units)
 * @property {number} y      - Top edge (PDF units)
 * @property {number} width  - Box width
 * @property {number} height - Box height
 */

/**
 * @typedef {Object} ReadyChunk
 * @property {number} chunk_id           - Unique chunk identifier
 * @property {number} start_time         - Timeline start (seconds)
 * @property {number} duration_seconds   - Chunk duration
 * @property {string} [filename]         - Audio filename
 * @property {number} page               - Primary page number
 * @property {number[]} [pages]          - All pages spanned (multi-page chunks)
 */

/**
 * @typedef {Object} BookMetadata
 * @property {string} [title]           - Display title
 * @property {string} [source_filename] - Original PDF filename
 */

/**
 * @typedef {Object} AudiobookStatusResponse
 * @property {string} processing_status       - Pipeline stage identifier
 * @property {string} [error_message]         - Error details if failed
 * @property {string} [trace_id]              - Observability trace ID
 * @property {boolean} [is_stale]             - True if job stalled
 * @property {number} [stale_threshold_minutes] - Minutes before stale warning
 * @property {number} [progress_percentage]   - Completion percentage (0-100)
 * @property {number} total_chunks            - Expected chunk count
 * @property {BookMetadata} [metadata]        - Book metadata
 * @property {string} [book_id]               - Audiobook identifier
 * @property {boolean} [is_complete]          - True if all chunks ready
 * @property {boolean} [ui_ready]             - True if UI JSON is available
 * @property {ReadyChunk[]} [ready_chunks]    - Available audio chunks
 */

/**
 * @typedef {Object} UISentencesResponse
 * @property {UISentence[]} sentences           - All sentences in audiobook
 * @property {Object<string, number[]>} [page_index] - Page number to sentence indices mapping
 */

/**
 * @typedef {Object} AudioSourcesResponse
 * @property {string[]} sources - Available source identifiers
 */

/**
 * @typedef {Object} UISentence
 * @property {number} global_index       - Unique sentence index across audiobook
 * @property {number} chunk_id           - Parent chunk identifier
 * @property {string} text               - Sentence text content
 * @property {number[]} pages            - Pages this sentence spans
 * @property {string[]} cids             - Canonical span IDs
 * @property {{start: number, end: number}} timing - Global timeline timing
 * @property {Object<string, Array<{cid: string, bbox: number[]}>>} [geometry] - Page-keyed geometry
 * @property {{turn_time: number, to_page: number, from_page: number}} [page_turn] - Page turn marker
 */

// Skip duration presets (seconds) - mapped to slider positions 0-10
const SKIP_PRESETS = [1, 2, 5, 10, 15, 30, 45, 60, 120, 300, 600];
const SKIP_STORAGE_KEY = 'tts-player-skip-amount';
const VOLUME_STORAGE_KEY = 'tts_volume';
const FULL_UI_JSON_PAGE_LIMIT = 150;


class TTSAudioPlayer {
    /******************************************************************************
     * SECTION 1: Class Declaration & State
     ******************************************************************************/

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
            currentChunkIndex: 0,
            totalChunks: 0,
            readyChunks: [],
            chunks: null,
            nextPageTurnIndex: 0,
            isProcessing: false,
            processingProgress: 0,
            // === Playback Epoch ===
            // Incremented on book load and chunk switch
            // Used to invalidate stale AUDIOPROCESS events
            playbackEpoch: 0,

            // === Loading State Gate ===
            // True during loadAudiobook() execution
            // Blocks AUDIOPROCESS handling to prevent race conditions
            isLoading: false,
            // === Chunk Transition Mutex ===
            isTransitioningChunk: false,

            // === UI JSON Contract ===
            // Raw UI sentences data from /api/v1/audiobooks/{id}/ui-sentences
            uiSentences: null,
            audioTiming: null,
            uiPageTurns: [],
            shards: {
                uiSentencesIndex: null,
                semanticIndex: null,
                uiPageCache: new Map(),
                semanticPageCache: new Map(),
                usePageShards: false,
            },
            lastLoadedAudioUrl: null,
            lastLoadedAudioFilename: null,
            // UI readiness flag
            uiReady: false,
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
            pendingHighlightSentence: null,
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
            processingStatus: '',
            errorMessage: '',
            lastAutoTurnedPage: null,
            activeHighlightSentenceKey: null,
            dependencies: {
                pdfjs: false,
                wavesurfer: false,
                errors: [],
            },
        },
    };

    /* Player Shell (Elements) */
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
        skipBackward: null,
        skipForward: null,
        skipAmountSlider: null,
        skipValue: null,
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
        // Error
        errorLog: null,
    };

    // === TIMELINE STATE MACHINE ===
    // Authoritative internal state for page/sentence coordination.
    // Separates PAGE_TURN events (intra-sentence) from SENTENCE_BOUNDARY events.
    timeline = {
        t: 0,                              // audiobook timeline seconds
        page: null,                        // current resolved display page
        sentenceKey: null,                 // stable key for active sentence
        // Policy: suppress highlight after mid-sentence page turn
        // When set, highlight updates blocked until sentenceKey changes
        suppressedUntilSentenceKey: null,
        // Debug tracing (optional)
        lastEvent: null,
        // Render coordination: true while PDF page render in progress
        pageRenderPending: false
    };

    /* Class constructor */
    constructor() {
        console.log('TTSAudioPlayer constructed. Ready to init.');

        // === Active Epoch Tracker ===
        // Snapshot of playbackEpoch when current chunk started
        // Compared against state.audiobook.playbackEpoch in AUDIOPROCESS
        this._activeEpoch = 0;

        // === UI JSON Index ===
        // Derived lookup structures built from uiSentences
        // Not stored in state (rebuilt on each load)
        this.uiIndex = null;
        this._highlightRafId = null;
        this._highlightLastFrameAt = 0;
    }

    /******************************************************************************
     * SECTION A - Application Orchestration
     ******************************************************************************/

    /**
     * Initialize the TTS Audio Player
     * @param {string|null} containerId - Optional container element ID
     * @param {Object} options - Initialization options
     */
    async init(containerId = null, options = {}) {
        console.log('TTSAudioPlayer initializing...');

        try {
            // STEP 1: Resolve local browser-side dependencies
            console.log('Step 1: Resolving local browser dependencies...');
            await this._resolveRuntimeDependencies();

            // STEP 2: Query all DOM elements
            console.log('Step 2: Querying DOM elements...');
            this._queryDOMElements();

            // STEP 3: Initialize audio backend
            console.log('Step 3: Initializing audio backend...');
            const backendType = options.backend || 'native';
            await this._initBackend(backendType);

            // STEP 4: Bind backend event listeners
            console.log('Step 4: Binding backend event listeners...');
            await this._bindBackendEvents();

            // STEP 5: Bind DOM event listeners
            console.log('Step 5: Binding DOM event listeners...');
            await this._bindDOMEvents();
            // Hide PDF viewer initially
            if (this.elements.pdfViewerContainer) {
                this.elements.pdfViewerContainer.style.display = 'none';
            }
            if (this.elements.pdfCanvas) {
                this.elements.pdfCanvas.style.display = 'none';
            }

            // STEP 6: Load initial data
            console.log('Step 6: Loading initial data...');
            try {
                await this.loadAudioSources();
            } catch (error) {
                console.error('Failed to load audio sources:', error);
                // Non-fatal, continue initialization
            }

            // STEP 7: Sync UI to initial state
            console.log('Step 7: Setting initial UI state...');
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

        // Destroy backend
        if (this.state.audio.backend) {
            this.state.audio.backend.destroy();
            this.state.audio.backend = null;
        }

        // Clear all states
        this.state.audio.isPlaying = false;
        this.state.audio.currentTime = 0;
        this.state.audio.duration = 0;

        console.log('TTSAudioPlayer destroyed.');
    }

    async _resolveRuntimeDependencies() {
        const loader = window.TTSLocalDependencies;
        if (loader && loader.ready && typeof loader.ready.then === 'function') {
            try {
                await loader.ready;
            } catch (error) {
                console.warn('[dependencies] Local dependency loader failed:', error);
            }
        }

        this.state.ui.dependencies = {
            pdfjs: typeof window.pdfjsLib !== 'undefined',
            wavesurfer: typeof window.WaveSurfer !== 'undefined',
            errors: Array.isArray(loader?.errors) ? loader.errors : [],
        };

        if (!this.state.ui.dependencies.pdfjs) {
            console.warn(
                '[dependencies] PDF.js local assets unavailable. ' +
                'Audio playback will continue; PDF rendering/highlighting is disabled until ' +
                '/static/vendor/pdfjs/pdf.min.js and pdf.worker.min.js are installed.'
            );
        }
        if (!this.state.ui.dependencies.wavesurfer) {
            console.info('[dependencies] WaveSurfer local asset unavailable. Native audio backend will be used.');
        }
    }

    async _initBackend(backendType) {
        if (backendType === 'wavesurfer' && typeof window.WaveSurfer === 'undefined') {
            console.warn('[audio] WaveSurfer requested but local asset is unavailable; falling back to native backend.');
            backendType = 'native';
        }
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
        this.elements.highlightContainer = document.getElementById('highlight-container');
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
        this.elements.zoomResetButton = document.getElementById('zoom-reset-button');
        this.elements.zoomLevel = document.getElementById('zoom-level');
        this.elements.clickSeekToggle = document.getElementById('click-seek-toggle');
        this.elements.pdfPageControls = document.getElementById('pdf-page-controls');

        // PLAYBACK CONTROLS
        this.elements.playPauseButton = document.getElementById('play-pause-button');
        this.elements.skipBackward = document.getElementById('skip-backward');
        this.elements.skipForward = document.getElementById('skip-forward');
        this.elements.skipAmountSlider = document.getElementById('skip-amount');
        this.elements.skipValue = document.getElementById('skip-value');
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
     * Bind DOM event listeners.
     * Subscribes to user interactions with UI controls
     * @private
     */
    async _bindDOMEvents() {
        this.elements.playPauseButton.addEventListener('click', async () => {
            await this.playPause();
        });
        this.elements.skipBackward.addEventListener('click', () => {
            this.skipCustom(-1);
        });
        this.elements.skipForward.addEventListener('click', () => {
            this.skipCustom(1);
        });
        this.elements.skipAmountSlider.addEventListener('input', (e) => {
            this.updateSkipAmount(parseInt(e.target.value, 10));
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
        this.elements.sourceSelect.addEventListener('change', async (e) => {
            await this.changeSource(e.target.value);
        });
        this.elements.refreshButton.addEventListener('click', async () => {
            await this.loadFileList();
        });
        if (this.elements.downloadButton) {
            this.elements.downloadButton.addEventListener('click', () => {
                this.downloadCurrentAudio();
            });
        }
        this.elements.nextPageButton.addEventListener('click', () => {
            this.nextPage();
        });
        this.elements.prevPageButton.addEventListener('click', () => {
            this.previousPage();
        });
        this.elements.zoomInButton.addEventListener('click', async () => {
            await this.zoomIn();
        });
        this.elements.zoomOutButton.addEventListener('click', async () => {
            await this.zoomOut();
        });
        this.elements.zoomFitWidthButton.addEventListener('click', async () => {
            await this.zoomFitWidth();
        });
        this.elements.zoomFitHeightButton.addEventListener('click', async () => {
            await this.zoomFitHeight();
        });

        // Bind reset zoom button
        if (this.elements.zoomResetButton) {
            this.elements.zoomResetButton.addEventListener('click', async () => {
                await this.resetZoom();
            });
        }

        this.elements.pageJumpInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.jumpToPage(e.target.value);
            }
        });

        if (this.elements.clickSeekToggle) {
            this.elements.clickSeekToggle.addEventListener('click', () => {
                this.toggleClickSeekMode();
            });
        }

        this.elements.pdfCanvas.addEventListener('click', async (e) => {
            await this.handlePdfClick(e);
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

        // Restore skip amount from localStorage
        const savedSkipIndex = localStorage.getItem(SKIP_STORAGE_KEY);
        if (savedSkipIndex !== null) {
            const index = parseInt(savedSkipIndex, 10);
            if (index >= 0 && index < SKIP_PRESETS.length) {
                this.elements.skipAmountSlider.value = index;
                this.updateSkipAmount(index, false);
            }
        }

        // Restore volume from localStorage
        const savedVolume = localStorage.getItem(VOLUME_STORAGE_KEY);
        if (savedVolume !== null) {
            const vol = parseFloat(savedVolume);
            if (!isNaN(vol) && vol >= 0 && vol <= 1) {
                this.state.audio.volume = vol;
                this.elements.volumeSlider.value = vol * 100;
                this.elements.volumeValue.textContent = Math.round(vol * 100) + '%';
            }
        }

        console.log('  ✓ UI synced to initial state');
    }

    /******************************************************************************
     * SECTION A2: Audiobook Orchestration
     *******************************************************************************/

    /**
     * Load audiobook and start playback
     * @param {string} bookId - Audiobook identifier
     */
    async loadAudiobook(bookId) {
        // === Block event processing during load ===
        this.state.audiobook.isLoading = true;

        // === Increment epoch, capture snapshot ===
        this._activeEpoch = ++this.state.audiobook.playbackEpoch;

        this.state.ui.lastAutoTurnedPage = null;

        // === Reset timeline state machine ===
        this.timeline.t = 0;
        this.timeline.page = null;
        this.timeline.sentenceKey = null;
        this.timeline.suppressedUntilSentenceKey = null;
        this.timeline.lastEvent = null;
        this.timeline.pageRenderPending = false;

        try {
            this.clearError();

            // === Clear old state ===
            this.state.audiobook.chunks = null;
            this.clearHighlights();
            this.state.audiobook.mode = 'audiobook';
            this.state.audiobook.bookId = bookId;

            // === Clear UI JSON state ===
            this.state.audiobook.uiSentences = null;
            this.state.audiobook.audioTiming = null;
            this.state.audiobook.uiPageTurns = [];
            this.state.audiobook.uiReady = false;
            this.state.audiobook.shards = {
                uiSentencesIndex: null,
                semanticIndex: null,
                uiPageCache: new Map(),
                semanticPageCache: new Map(),
                usePageShards: false,
            };
            this.uiIndex = null;

            console.log(`[loadAudiobook] Loading: ${bookId}`);

            // =====================================================================
            // Fetch /status first (audio lifecycle + ui_ready gate)
            // =====================================================================
            const statusResponse = await fetch(`/api/v1/audiobooks/${bookId}/status`);

            if (!statusResponse.ok) {
                const msg = `Failed to load audiobook status: ${statusResponse.status}`;
                console.error(msg);           // surfaces in container logs
                this.logError(msg);           // surfaces in UI
                throw new Error(msg);          // aborts execution path
            }

            /** @type {AudiobookStatusResponse} */
            const statusData = await statusResponse.json();
            this.state.audiobook.manifest = statusData;

            // === Status handling and playability check ===
            const status = statusData.processing_status || 'unknown';
            const isPlayable = (status === 'stage_3_complete' || status === 'stage_3_partial');

            this.state.audiobook.processingStatus = status;
            this.state.audiobook.errorMessage = statusData.error_message || null;
            this.state.audiobook.traceId = statusData.trace_id || null;
            this.state.audiobook.isStale = statusData.is_stale || false;
            this.state.audiobook.totalChunks = statusData.total_chunks || 0;
            this.state.audiobook.readyChunks = statusData.ready_chunks || [];
            this.state.audiobook.progress_percentage = statusData.progress_percentage || 0;
            this.state.audiobook.processingProgress = statusData.progress_percentage || 0;
            this.state.audiobook.currentChunkIndex = 0;

            await this.loadArtifactShardIndexes(bookId);
            this.state.audiobook.shards.usePageShards = this.shouldPreferPageShards(statusData);

            if (!isPlayable) {
                this.updateStatusBanner();
                this.elements.playerContainer.style.display = 'none';
                this.elements.statusContainer.style.display = 'block';
                return;
            }

            // === Player initialization (runs for stage_3_complete AND stage_3_partial) ===
            this.elements.playerContainer.style.display = 'block';
            this.elements.statusContainer.style.display = 'none';

            if (this.state.audiobook.readyChunks.length === 0) {
                this.logError('No audio chunks available yet. Please wait for processing.');
                this.updateStatusBanner();
                this.elements.playerContainer.style.display = 'none';
                this.elements.statusContainer.style.display = 'block';
                return;
            }

            // =====================================================================
            // Chunks are now AUDIO-ONLY metadata
            // Sentences are accessed via uiIndex.byChunkId, NOT chunk.sentences
            // =====================================================================
            this.state.audiobook.chunks = this.state.audiobook.readyChunks.map((readyChunk) => ({
                chunk_id: readyChunk.chunk_id,
                filename: readyChunk.filename,
                start_time: readyChunk.start_time,
                duration_seconds: readyChunk.duration_seconds,
                page: readyChunk.page,
                pages: readyChunk.pages || [readyChunk.page],
                end_time: readyChunk.start_time + readyChunk.duration_seconds
            }));

            await this.loadAudioTiming(bookId, statusData);

            // Conditionally fetch UI data based on full-artifact and page-shard availability.
            if (statusData.ui_ready && !this.state.audiobook.shards.usePageShards) {
                try {
                    const uiResponse = await fetch(`/api/v1/audiobooks/${bookId}/ui-sentences`);

                    if (uiResponse.ok) {
                        this.state.audiobook.uiSentences = await uiResponse.json();
                        // === UI schema sanity (sentence-level contract) ===
                        const ui = this.state.audiobook.uiSentences;
                        const sentenceList = (ui && Array.isArray(ui.sentences)) ? ui.sentences : null;

                        if (!sentenceList) {
                            console.error('[loadAudiobook] ui_sentences schema invalid: missing top-level sentences[]');
                            this.state.audiobook.uiSentences = null;
                            this.state.audiobook.uiReady = false;
                        } else {
                            // Persist canonical sentence list handle for later logic (no behavior change)
                            this.state.audiobook._uiSentenceCount = sentenceList.length;
                            this.state.audiobook._uiSchemaVersion = ui.schema_version || null;
                            if (this.state.audiobook.audioTiming) {
                                this.applyAudioTimingToSentenceList(sentenceList, this.state.audiobook.audioTiming);
                            }
                        }

                        // Build indexes (returns readiness boolean)
                        const uiReady = this.buildUiIndex();
                        // === CID reuse flag (proven true in somatosensory) ===
                        // IMPORTANT: CIDs may appear in multiple sentences; do not treat CID as sentence identity.
                        try {
                            const ui = this.state.audiobook.uiSentences;
                            const sents = ui && Array.isArray(ui.sentences) ? ui.sentences : [];
                            const cidToCount = new Map();
                            for (const s of sents) {
                                for (const cid of (s.cids || [])) {
                                    cidToCount.set(cid, (cidToCount.get(cid) || 0) + 1);
                                }
                            }
                            const sharedCids = new Set();
                            for (const [cid, count] of cidToCount.entries()) {
                                if (count > 1) sharedCids.add(cid);
                            }
                            this.state.audiobook._sharedCidCount = sharedCids.size;
                            this.state.audiobook._sharedCids = sharedCids;
                            const sharedCidCount = sharedCids.size;
                            if (sharedCidCount > 0) {
                                console.log(`[loadAudiobook] NOTE: shared CIDs detected across sentences: ${sharedCidCount}`);
                            }
                        } catch (e) {
                            console.warn('[loadAudiobook] shared CID analysis failed:', e);
                        }
                        // === Load semantic.json and build spanIndex for progressive highlighting ===
                        try {
                            const semanticResponse = await fetch(`/api/v1/audiobooks/${bookId}/semantic`);
                            if (semanticResponse.ok) {
                                const semanticData = await semanticResponse.json();
                                this.uiIndex.spanIndex = this.buildSpanIndex(semanticData);

                                // === Explicit authority markers (no behavior change) ===
                                // UI sentence geometry is the only sentence-scoped geometry source.
                                // spanIndex is span-scoped and may cross sentence boundaries (CID reuse).
                                this.uiIndex.geometryAuthority = 'ui_sentences';
                                this.uiIndex.spanIndexAuthority = 'semantic_spans';
                                console.log(`[loadAudiobook] spanIndex built: ${this.uiIndex.spanIndex.size} spans`);
                            } else {
                                console.warn(`[loadAudiobook] semantic.json not available (${semanticResponse.status})`);
                                this.uiIndex.spanIndex = null;
                            }
                        } catch (e) {
                            console.warn('[loadAudiobook] Error loading semantic.json:', e);
                            this.uiIndex.spanIndex = null;
                        }

                        // Wire state from built indexes
                        this.state.audiobook.uiPageTurns = this.uiIndex?.pageTurnsByTime || [];
                        this.state.audiobook.uiReady = uiReady;
                        this._resetPageTurnIndexForTime(this.timeline?.t ?? 0);

                        // === Step 2.2: Verification logging ===
                        if (uiReady) {
                            console.log(`[loadAudiobook] UI JSON loaded successfully for: ${bookId}`);
                            console.log(`[loadAudiobook] Index verification:`, {
                                byGlobalIndex: this.uiIndex.byGlobalIndex.size,
                                byChunkId: this.uiIndex.byChunkId.size,
                                byPage: Object.keys(this.uiIndex.byPage).length,
                                byTimeByChunk: this.uiIndex.byTimeByChunk.size,
                                pageTurnsByTime: this.uiIndex.pageTurnsByTime.length,
                                geometryByPage: Object.keys(this.uiIndex.geometryByPage).length
                            });
                        }
                    } else {
                        // ui_ready was true but fetch failed — unexpected error
                        console.error(
                            `[loadAudiobook] UI sentences fetch failed unexpectedly ` +
                            `(status indicated ui_ready=true): ${uiResponse.status}`
                        );
                        this.state.audiobook.uiSentences = null;
                        this.state.audiobook.uiReady = false;
                    }
                } catch (parseError) {
                    console.error(`[loadAudiobook] Failed to parse UI JSON for ${bookId}:`, parseError);
                    this.state.audiobook.uiSentences = null;
                    this.state.audiobook.uiReady = false;
                }
            } else if (this.state.audiobook.shards.usePageShards) {
                const firstChunk = this.state.audiobook.chunks[0];
                await this.ensureUiDataForChunk(firstChunk);
                console.log(
                    `[loadAudiobook] Using page-sharded UI data for ${bookId}. ` +
                    `Loaded initial chunk pages; additional pages load on chunk transitions.`
                );
            } else {
                // UI JSON not ready yet — audio-only mode (expected during processing)
                console.log(
                    `[loadAudiobook] UI sentences not ready yet (ui_ready=false). ` +
                    `Highlighting and click-seek disabled.`
                );
                this.state.audiobook.uiSentences = null;
                this.state.audiobook.uiReady = false;
                // Ensure no stale spanIndex survives into audio-only mode
                if (this.uiIndex) this.uiIndex.spanIndex = null;
            }

            // === Update title display ===
            const titleText = statusData.metadata?.title || bookId;
            const partialIndicator = (status === 'stage_3_partial') ? ' ⚠️ (Partial)' : '';
            const uiIndicator = this.state.audiobook.uiReady
                ? (this.state.audiobook.shards.usePageShards ? ' [Page Shards]' : '')
                : ' [Audio Only]';
            this.elements.currentFileDisplay.textContent = titleText + partialIndicator + uiIndicator;

            // === Load PDF ===
            const pdfFilename = statusData.metadata?.source_filename;
            if (pdfFilename) {
                await this.loadPdf(pdfFilename);
            }

            // === Log partial status ===
            if (status === 'stage_3_partial') {
                const ready = this.state.audiobook.readyChunks.length;
                const total = this.state.audiobook.totalChunks;
                console.log(`[loadAudiobook] Partial: ${ready}/${total} chunks (${Math.round((ready / total) * 100)}%)`);
            }

            if (status === 'stage_3_partial' || statusData.is_stale || statusData.job_status === 'failed') {
                this.updateStatusBanner({allowPlayer: true});
            }

            // === Guard play() against superseded loads ===
            const expectedEpoch = (0 !== this.state.audiobook.currentChunkIndex)
                ? (this.state.audiobook.playbackEpoch + 1)
                : this.state.audiobook.playbackEpoch;

            await this.playChunk(0);

            if (this.state.audiobook.playbackEpoch === expectedEpoch) {
                await this.play();
            }

        } catch (error) {
            this.logError('Failed to load audiobook: ' + error.message);
            console.error('[loadAudiobook] Error:', error);
            this.state.audiobook.uiReady = false;
        } finally {
            // === ALWAYS re-enable event processing ===
            this.state.audiobook.isLoading = false;
        }
    }

    /******************************************************************************
     * SECTION B: Audio Backend (TTS/Playback)
     ******************************************************************************/

    /**
     * Bind backend event listeners.
     * @private
     */
    async _bindBackendEvents() {
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

        // PLAY/PAUSE/TIMEUPDATE/ERROR events
        backend.on(AudioBackend.EVENTS.PLAY, () => {
            this.state.audio.isPlaying = true;
            if (this.elements.playPauseButton) {
                this.elements.playPauseButton.textContent = 'Pause';
            }
            this.startHighlightAnimationLoop();
        });

        backend.on(AudioBackend.EVENTS.PAUSE, () => {
            this.state.audio.isPlaying = false;
            if (this.elements.playPauseButton) {
                this.elements.playPauseButton.textContent = 'Play';
            }
            this.stopHighlightAnimationLoop();
        });

        backend.on(AudioBackend.EVENTS.TIMEUPDATE, (data) => {
            // === Block during loading ===
            if (this.state.audiobook.isLoading) return;

            // === Ignore stale events from prior epoch ===
            if (this._activeEpoch !== this.state.audiobook.playbackEpoch) return;

            this.state.audio.currentTime = data.currentTime;
            this._updateTimeDisplay();
            this._updateSeekSlider();
        });

        // Bind AUDIOPROCESS (60Hz) with SEPARATED concerns
        backend.on(AudioBackend.EVENTS.AUDIOPROCESS, (data) => {
            // === Runtime gates ===
            if (this.state.audiobook.isLoading) return;
            if (this._activeEpoch !== this.state.audiobook.playbackEpoch) return;
            if (this.state.audiobook.mode !== 'audiobook') return;

            // === Resolve timeline state (pure) ===
            const resolution = this._resolveTimelineState(data.currentTime);

            // === Apply policy events (UI) ===
            this._applyTimelineEvents(resolution);
        });

        backend.on(AudioBackend.EVENTS.FINISH, async () => {
            // === Block during loading ===
            if (this.state.audiobook.isLoading) return;
            // === Ignore stale FINISH from prior epoch ===
            if (this._activeEpoch !== this.state.audiobook.playbackEpoch) return;
            // === Prevent re-entrant chunk transitions ===
            if (this.state.audiobook.isTransitioningChunk) return;
            if (this.state.audiobook.mode === 'audiobook') {
                await this.playNextChunk();
            } else {
                this.state.audio.isPlaying = false;
                if (this.elements.playPauseButton) {
                    this.elements.playPauseButton.textContent = 'Play';
                }
                this.stopHighlightAnimationLoop();
            }
        });

        backend.on(AudioBackend.EVENTS.SEEKING, (data) => {
            if (this.state.audiobook.isLoading) return;
            if (this._activeEpoch !== this.state.audiobook.playbackEpoch) return;

            this.state.audio.currentTime = data.currentTime;

            const t = this._getAudiobookTimelineTime(data.currentTime);
            this._resetPageTurnIndexForTime((t != null) ? t : 0);

            this._updateTimeDisplay();
            this._updateSeekSlider();
            this.clearHighlights();

            // Reset timeline suppression on seek (user intent = start fresh)
            this.timeline.suppressedUntilSentenceKey = null;
            this.timeline.sentenceKey = null; // Force re-evaluation
        });

        backend.on(AudioBackend.EVENTS.ERROR, (data) => {
            console.error('Backend error:', data.error);
            this.logError(data.error.message);
        });
    }

    startHighlightAnimationLoop() {
        if (this._highlightRafId != null) return;
        const tick = (now) => {
            this._highlightRafId = null;
            if (!this.state.audio.isPlaying) return;
            if (!this.state.audiobook || this.state.audiobook.isLoading) return;
            if (this.state.audiobook.mode === 'audiobook' && this.state.audio.backend) {
                const localTime = this.state.audio.backend.getCurrentTime();
                if (typeof localTime === 'number') {
                    this.state.audio.currentTime = localTime;
                    const resolution = this._resolveTimelineState(localTime);
                    this._applyTimelineEvents(resolution);
                }
            }
            this._highlightRafId = requestAnimationFrame(tick);
        };
        this._highlightRafId = requestAnimationFrame(tick);
    }

    stopHighlightAnimationLoop() {
        if (this._highlightRafId != null) {
            cancelAnimationFrame(this._highlightRafId);
            this._highlightRafId = null;
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
            this.logError('Playback failed: ' + error.message);
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

        // Reset marker cursor using global audiobook time.
        const tGlobal = this._getAudiobookTimelineTime(clampedTime);
        this._resetPageTurnIndexForTime(tGlobal ?? 0);

        // UI updated by backend SEEKING event handler
    }

    /**
     * Get current skip duration in seconds from slider position.
     * @returns {number} Skip duration in seconds
     */
    getSkipDuration() {
        const index = parseInt(this.elements.skipAmountSlider?.value ?? 3, 10);
        return SKIP_PRESETS[index] ?? 10;
    }

    /**
     * Update skip amount display and persist to localStorage.
     * @param {number} index - Slider position (0-10)
     * @param {boolean} save - Whether to persist to localStorage (default: true)
     */
    updateSkipAmount(index, save = true) {
        const seconds = SKIP_PRESETS[index] ?? 10;

        // Format display value
        let display;
        if (seconds < 60) {
            display = `${seconds}s`;
        } else {
            const mins = seconds / 60;
            display = `${mins}m`;
        }

        if (this.elements.skipValue) {
            this.elements.skipValue.textContent = display;
        }

        // Persist preference
        if (save) {
            localStorage.setItem(SKIP_STORAGE_KEY, index.toString());
        }
    }

    /**
     * Skip forward or backward by custom amount.
     * @param {number} direction - 1 for forward, -1 for backward
     */
    skipCustom(direction) {
        const duration = this.getSkipDuration();
        this.skip(duration * direction);
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

        // Reset marker cursor for seek safety (slider + click-seek)
        this._resetPageTurnIndexForTime(clampedTime);

        // UI updated by backend SEEKING event handler
    }

    /**
     * Play specific chunk
     * @param {number} chunkIndex - Chunk index (0-based)
     */
    async playChunk(chunkIndex) {
        try {
            const chunks = this.state.audiobook.chunks || [];

            // === Epoch increment ONLY if chunk actually changes ===
            if (chunkIndex !== this.state.audiobook.currentChunkIndex) {
                this._activeEpoch = ++this.state.audiobook.playbackEpoch;

                // Clear UI caches tied to previous chunk
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
            await this.ensureUiDataForChunk(chunk);

            // Construct URL
            const bookId = this.state.audiobook.bookId;
            // chunk.filename is now reliably available from the full chunk object
            const chunkFilename = chunk.filename;
            const url = `/api/v1/audiobooks/${bookId}/chunks/${chunkFilename}/audio`;
            this.state.audiobook.lastLoadedAudioUrl = url;
            this.state.audiobook.lastLoadedAudioFilename = chunkFilename;

            console.log(`Loading chunk ${chunkIndex + 1}/${this.state.audiobook.totalChunks}: ${chunkFilename}`);

            // Disable play button during load
            this.elements.playPauseButton.disabled = true;

            // === Acquire transition mutex for duration of backend.load ===
            this.state.audiobook.isTransitioningChunk = true;
            try {
                // Load chunk
                await this.state.audio.backend.load(url);

                // ============================================================
                // PAGE TURN CURSOR INITIALIZATION (CRITICAL)
                //
                // uiPageTurns.turn_time is in GLOBAL audiobook time.
                // Cursor MUST be aligned whenever a chunk is loaded.
                // This guarantees page turns occur at the last spoken word.
                // ============================================================
                const globalStartTime = this._getAudiobookTimelineTime(0);
                this._resetPageTurnIndexForTime(globalStartTime);

            } finally {
                // === Always release mutex, even on AbortError ===
                this.state.audiobook.isTransitioningChunk = false;
            }
        } catch (error) {
            this.logError('Failed to play chunk: ' + error.message);
            console.error('Chunk playback error:', error);
        }
    }

    async playNextChunk() {
        const chunks = this.state.audiobook.chunks || [];
        const currentIndex = this.state.audiobook.currentChunkIndex;

        // No chunks or invalid state
        if (!Array.isArray(chunks) || chunks.length === 0) {
            console.warn('[playNextChunk] No chunks available');
            this.state.audio.isPlaying = false;
            return;
        }

        const nextIndex = currentIndex + 1;

        // End of audiobook
        if (nextIndex >= chunks.length) {
            console.log('[playNextChunk] End of audiobook reached');
            this.state.audio.isPlaying = false;

            if (this.elements.playPauseButton) {
                this.elements.playPauseButton.textContent = 'Play';
            }
            return;
        }

        // Prevent re-entrant transitions
        if (this.state.audiobook.isTransitioningChunk) {
            console.warn('[playNextChunk] Transition already in progress');
            return;
        }

        console.log(`[playNextChunk] Advancing from chunk ${currentIndex} → ${nextIndex}`);

        await this.playChunk(nextIndex);

        // Resume playback only if epoch still valid
        if (this._activeEpoch === this.state.audiobook.playbackEpoch) {
            await this.play();
        }
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
     * Set playback speed.
     * Updates backend, state cache, and display.
     * @param {number} rate - Speed (0.5-2.0)
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
     * @param {number} volume - Volume (0.0-1.0)
     */
    setVolume(volume) {
        // Clamp to valid range
        const clampedVolume = Math.max(0.0, Math.min(volume, 1.0));

        // Update backend
        this.state.audio.backend.setVolume(clampedVolume);

        // Cache in state
        this.state.audio.volume = clampedVolume;

        // Persist to localStorage
        localStorage.setItem(VOLUME_STORAGE_KEY, clampedVolume.toString());

        // Update display (convert to percentage)
        const percentage = Math.round(clampedVolume * 100);
        this.elements.volumeValue.textContent = percentage + '%';
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

    /******************************************************************************
     * SECTION C — Timing / Timeline
     ******************************************************************************/

    /**
     * Convert backend-provided local playback time into audiobook timeline seconds.
     */
    _getAudiobookTimelineTime(localTime) {
        if (typeof localTime !== 'number') return null;


        const chunk = this.state.audiobook.chunks?.[this.state.audiobook.currentChunkIndex];
        const chunkStart = (typeof chunk?.start_time === 'number') ? chunk.start_time : 0;


        const t = chunkStart + localTime;


// Clamp if chunk bounds are available (prevents wrong-window resolution on transitions)
        if (chunk && typeof chunk.start_time === 'number' && typeof chunk.end_time === 'number') {
            return Math.max(chunk.start_time, Math.min(chunk.end_time, t));
        }


        return t;
    }

    getSentenceAtTimestamp(timestamp) {
        // ================================================================
        // SENTENCE LOCATOR (Pure Function)
        // Returns UI JSON sentence object directly, unmodified.
        // MUST NOT: decide page turns, interpret CIDs, mutate state.
        // Callers handle all policy decisions.
        // ================================================================

        // === Hard gate: UI JSON must be ready ===
        if (!this.uiIndex || !this.state.audiobook.uiReady) {
            return null;
        }

        const chunkIndex = this.state.audiobook.currentChunkIndex;
        const chunks = this.state.audiobook.chunks || [];

        if (chunkIndex == null || chunkIndex < 0 || chunkIndex >= chunks.length) {
            return null;
        }

        const chunk = chunks[chunkIndex];
        const chunkId = chunk.chunk_id;

        // Get pre-sorted sentences for this chunk from UI index
        const sentences = this.uiIndex.byTimeByChunk.get(chunkId);
        if (!sentences || sentences.length === 0) {
            return null;
        }

        const currentTime = timestamp;

        // === Binary search for sentence containing timestamp ===
        // Sentences sorted by timing.start ascending
        // UI JSON contract guarantees timing.start and timing.end exist
        let sentence = null;
        let low = 0;
        let high = sentences.length - 1;

        while (low <= high) {
            const mid = Math.floor((low + high) / 2);
            const s = sentences[mid];

            if (currentTime < s.timing.start) {
                high = mid - 1;
            } else if (currentTime >= s.timing.end) {
                low = mid + 1;
            } else {
                sentence = s;
                break;
            }
        }

        // Clamp to boundary if outside all windows
        if (!sentence) {
            sentence = (currentTime < sentences[0].timing.start)
                ? sentences[0]
                : sentences[sentences.length - 1];
        }

        // === Sentence contract declaration (UI layer) ===
        // NOTE: Returned sentence is UI JSON–scoped.
        // It may include reused CIDs and CID-level bboxes.
        // It does NOT include semantic span ownership, char offsets,
        // or sentence-bounded span projections.
        // Downstream logic MUST NOT assume CID == sentence.
        sentence._uiContract = 'ui_sentence';

        // V2 UI contract can carry backend-owned sentence coverage.
        sentence._hasSemanticProjection = Boolean(
            Array.isArray(sentence.source_cids) || Array.isArray(sentence.geometry_cids)
        );

        return sentence;
    }

    /**
     * Reset page turn marker cursor for seek safety.
     * Finds first marker with turn_time > t, so playback resumes correctly.
     */
    _resetPageTurnIndexForTime(t) {
        // ================================================================
        // PAGE TURN CURSOR RESET
        // Uses uiPageTurns exclusively (UI JSON contract).
        // ================================================================
        const pageTurns = this.state.audiobook.uiPageTurns || [];

        if (!pageTurns.length || typeof t !== 'number') {
            this.state.audiobook.nextPageTurnIndex = 0;
            return;
        }

        // Find first marker with turn_time > t
        let idx = 0;
        while (
            idx < pageTurns.length &&
            typeof pageTurns[idx]?.turn_time === 'number' &&
            pageTurns[idx].turn_time <= t
            ) {
            idx++;
        }
        this.state.audiobook.nextPageTurnIndex = idx;
    }

    /**
     * === Timeline Resolution (Pure) ===
     * REVISION: Uses UI JSON exclusively.
     * Uses:
     *   - getSentenceAtTimestamp() → UI JSON sentence
     *   - uiPageTurns cursor → word-accurate page turns
     *
     * @param {number} localTime - Chunk-local playback time from backend
     * @returns {Object|null} Resolution object or null if not resolvable
     */
    _resolveTimelineState(localTime) {
        const timelineTime = this._getAudiobookTimelineTime(localTime);
        if (timelineTime == null) return null;

        // === Hard gate: UI JSON must be ready ===
        if (!this.uiIndex || !this.state.audiobook.uiReady) {
            return null;
        }

        const sentence = this.getSentenceAtTimestamp(timelineTime);
        if (!sentence) return null;

        // === SENTENCE KEY: UI JSON guarantees global_index ===
        const sentenceKey = sentence.global_index;

        // ------------------------------------------------------------
        // CONTRACT GATE:
        // If this is a UI sentence without semantic projection authority,
        // do not use semantic span text (cleanedText) for boundary inference.
        // ------------------------------------------------------------
        const isUiSentence = (sentence && sentence._uiContract === 'ui_sentence');
        const hasSemanticProjection = !!(sentence && sentence._hasSemanticProjection);

        // Reset cached CID weights when sentence changes
        if (this.timeline.sentenceKey !== sentenceKey) {
            this.timeline.cachedCidLens = null;
            this.timeline.activeCidIndex = null;
        }

        // ================================================================
        // PAGE RESOLUTION: CID-transition based (per-word accurate)
        //
        // Invariant:
        //   Page turns when the ACTIVE CID changes to one on a different page.
        //   This is the closest possible proxy for "last spoken word on page"
        //   without backend word-level timestamps.
        // ================================================================
        let resolvedPage = null;

        const spanIndex = this.uiIndex?.spanIndex;
        const cids = sentence.cids || [];
        const currentPage = this.timeline.page || sentence.pages?.[0] || 1;

        // ------------------------------------------------------------
        // UI AUTHORITY: derive CID→page mapping from sentence.geometry
        // Avoid relying on semantic spanIndex.page when CID is reused.
        // ------------------------------------------------------------
        const cidToPage = new Map();
        try {
            const geom = sentence.geometry || {};
            for (const [pageStr, items] of Object.entries(geom)) {
                const p = Number(pageStr);
                if (!Array.isArray(items)) continue;
                for (const it of items) {
                    if (it && it.cid != null && !cidToPage.has(it.cid)) {
                        cidToPage.set(it.cid, p);
                    }
                }
            }
        } catch (e) {
            // If geometry is absent/malformed, fall back to spanIndex.page/currentPage
        }

        if (spanIndex && cids.length > 0) {
            // ------------------------------------------------------------
            // Step 1: Compute total effective chars (cached per sentence)
            // Uses sentence span fractions to count only the portion of
            // each CID that belongs to THIS sentence.
            // ------------------------------------------------------------
            let totalEffectiveChars;
            let cidEffectiveLens;
            let cidSentenceFracs;
            let alignmentUsable;
            let hasSubSpan;

            if (this.timeline.cachedCidLens) {
                ({
                    totalEffectiveChars,
                    cidEffectiveLens,
                    cidSentenceFracs,
                    alignmentUsable,
                    hasSubSpan
                } = this.timeline.cachedCidLens);
            } else {
                totalEffectiveChars = 0;
                cidEffectiveLens = [];
                cidSentenceFracs = [];

                const spanFractions = this.computeSentenceSpanFractions(sentence, spanIndex);

                // "Usable" means we got any mapping at all.
                // (Even full-span mappings can be valid for many sentences.)
                alignmentUsable = spanFractions && spanFractions.size > 0;

                // In UI sentence mode without semantic projection, treat alignment as non-authoritative for text slicing
                if (isUiSentence && !hasSemanticProjection) alignmentUsable = false;

                // "Has sub-span" means we can safely slice a CID's text portion.
                hasSubSpan = false;

                for (const cid of cids) {
                    const meta = spanIndex.get(cid);
                    const fullLen = meta?.len || meta?.cleanedText?.length || 0;

                    const frac = spanFractions.get(cid) || {startFrac: 0, endFrac: 1};
                    const owned = Math.max(0, Math.min(1, frac.endFrac - frac.startFrac));

                    if (frac.startFrac !== 0 || frac.endFrac !== 1) hasSubSpan = true;

                    // Keep float precision; prevent zero-length with tiny epsilon
                    const effectiveLen = Math.max(1e-6, fullLen * owned);

                    cidEffectiveLens.push(effectiveLen);
                    cidSentenceFracs.push(frac);
                    totalEffectiveChars += effectiveLen;
                }

                this.timeline.cachedCidLens = {
                    totalEffectiveChars,
                    cidEffectiveLens,
                    cidSentenceFracs,
                    alignmentUsable,
                    hasSubSpan
                };
            }

            // ------------------------------------------------------------
            // Step 2: Compute sentence progress [0,1]
            // ------------------------------------------------------------
            const duration = sentence.timing.end - sentence.timing.start;
            let sentenceProgress = duration > 0
                ? (timelineTime - sentence.timing.start) / duration
                : 0;

            // Clamp for safety
            sentenceProgress = Math.max(0, Math.min(1, sentenceProgress));

            // ------------------------------------------------------------
            // Step 3: Compute CID boundaries (start/end fractions)
            // ------------------------------------------------------------
            let cumulative = 0;
            const cidBounds = []; // [{ start, end, page }]

            for (let i = 0; i < cids.length; i++) {
                const cid = cids[i];
                const meta = spanIndex.get(cid);
                const page = cidToPage.get(cid) ?? meta?.page ?? currentPage;

                const startFrac = totalEffectiveChars > 0
                    ? cumulative / totalEffectiveChars
                    : 0;

                cumulative += cidEffectiveLens[i];

                const endFrac = totalEffectiveChars > 0
                    ? cumulative / totalEffectiveChars
                    : 1;

                cidBounds.push({
                    start: startFrac,
                    end: endFrac,
                    page
                });
            }

            // ------------------------------------------------------------
            // Step 4: Determine current page using LAST-WORD-END rule
            //
            // Invariant:
            //   Turn the page when the LAST SPOKEN WORD on the current
            //   page FINISHES speaking. This avoids comma / pause lag
            //   without flipping early.
            // ------------------------------------------------------------
            resolvedPage = currentPage;

            for (let i = 0; i < cidBounds.length; i++) {
                const curr = cidBounds[i];
                const next = cidBounds[i + 1];

                // This CID belongs to the current page
                if (Number(curr.page) === Number(currentPage)) {

                    // If the next CID is on a different page,
                    // curr is the LAST CID on this page
                    if (next && Number(next.page) !== Number(currentPage)) {

                        const cid = cids[i];
                        const meta = spanIndex.get(cid);

                        let triggerFrac = curr.end;

                        // Use semantic cleanedText only when semantic projection authority exists AND we have real sub-span boundaries
                        if ((!isUiSentence || hasSemanticProjection) && alignmentUsable && hasSubSpan && meta?.cleanedText && cidSentenceFracs?.[i]) {
                            const text = meta.cleanedText;
                            const frac = cidSentenceFracs[i];

                            // Extract only the sentence-owned portion of the CID text
                            const startChar = Math.max(0, Math.min(text.length, Math.floor(frac.startFrac * text.length)));
                            const endChar = Math.max(0, Math.min(text.length, Math.ceil(frac.endFrac * text.length)));

                            const sentencePortion = text.slice(startChar, endChar).trimEnd();

                            if (sentencePortion.length > 0) {
                                const match = sentencePortion.match(/^(.*?)([\p{L}\p{N}]+)[^\p{L}\p{N}]*$/u);
                                if (match) {
                                    const wordEndIdx = match[1].length + match[2].length;
                                    const wordEndFracInPortion = wordEndIdx / sentencePortion.length;
                                    const cidSpan = curr.end - curr.start;


                                    // Guard against FP amplification on tiny spans
                                    if (cidSpan < 0.05) {
                                        triggerFrac = curr.end;
                                    } else {
                                        triggerFrac = curr.start + (wordEndFracInPortion * cidSpan);
                                    }
                                }
                            } else {
                                const fullText = meta.cleanedText.trimEnd();
                                const match = fullText.match(/^(.*?)([\p{L}\p{N}]+)[^\p{L}\p{N}]*$/u);
                                if (match && fullText.length > 0) {
                                    const wordEndIdx = match[1].length + match[2].length;
                                    const wordEndFracInCid = wordEndIdx / fullText.length;
                                    const cidSpan = curr.end - curr.start;

                                    // Guard against FP amplification on tiny spans
                                    if (cidSpan < 0.05) {
                                        triggerFrac = curr.end;
                                    } else {
                                        triggerFrac = curr.start + (wordEndFracInCid * cidSpan);
                                    }
                                }
                            }
                        }
                        // If alignment not usable, triggerFrac stays at curr.end (CID-END fallback)
                        // Turn page when the LAST WORD finishes speaking (with hysteresis)
                        const HYSTERESIS_FRAC = 0.02; // fraction of sentence progress [0..1]
                        if (sentenceProgress >= triggerFrac + HYSTERESIS_FRAC) {
                            resolvedPage = next.page;
                        }
                        break;
                    }
                }
            }

            // ------------------------------------------------------------
            // Step 5: Maintain CID index for debugging / continuity
            // ------------------------------------------------------------
            let activeCidIndex = 0;
            for (let i = 0; i < cidBounds.length; i++) {
                if (sentenceProgress >= cidBounds[i].start) {
                    activeCidIndex = i;
                } else {
                    break;
                }
            }

            this.timeline.activeCidIndex = activeCidIndex;

        } else {
            // No span data — fall back to sentence page
            resolvedPage = sentence.pages?.[0] || currentPage;
        }

        // ------------------------------------------------------------
        // Final fallback: chunk-level page
        // ------------------------------------------------------------
        if (resolvedPage == null) {
            const chunk = this.state.audiobook.chunks?.[this.state.audiobook.currentChunkIndex];
            if (typeof chunk?.page === 'number') {
                resolvedPage = chunk.page;
            }
        }

        // ------------------------------------------------------------
        // Cache invalidation expansion (page/zoom aware)
        // ------------------------------------------------------------
        const currentScale = this.state.pdf.viewport?.scale;
        const cachePageChanged = (this.timeline.cachedPage !== resolvedPage);
        const cacheScaleChanged = (this.timeline.cachedViewportScale !== currentScale);


        if (cachePageChanged || cacheScaleChanged) {
            this.timeline.cachedCidLens = null;
            this.timeline.activeCidIndex = null;
            this.timeline.cachedPage = resolvedPage;
            this.timeline.cachedViewportScale = currentScale;
        }

        // === CHANGE DETECTION ===
        const pageChanged =
            typeof resolvedPage === 'number' &&
            resolvedPage !== this.timeline.page;

        const sentenceChanged =
            sentenceKey !== this.timeline.sentenceKey;

        return {
            timelineTime,
            sentence,
            sentenceKey,
            resolvedPage,
            pageChanged,
            sentenceChanged
        };
    }

    /**
     * Apply timeline events to UI state.
     *
     * Design contract:
     *   - SENTENCE_BOUNDARY: New sentence started. Clear old highlights, apply new.
     *   - PAGE_TURN: Mid-sentence page change. Clear highlights, show sentence's
     *                geometry on the new page (word-accurate continuation).
     *   - GEOMETRY_REFRESH: Same sentence, but highlights need recalculation
     *                       (zoom, viewport resize, cache invalidation).
     *                       Routes through SENTENCE_BOUNDARY path.
     *
     * Rendering contract:
     *   - If page render required: set pending highlight, let renderPage apply it
     *   - If no render required: apply highlight via RAF directly
     *   - Pending highlights only apply when rendered page matches timeline.page
     *
     * Invariant: timeline.page is ALWAYS updated to resolvedPage on any event.
     *
     * @param {Object} resolution - Timeline resolution from _resolveTimelineState
     * @private
     */
    _applyTimelineEvents(resolution) {
        if (!resolution) return;

        this.timeline.t = resolution.timelineTime;

        const {
            sentence,
            sentenceKey,
            resolvedPage,
            pageChanged,
            sentenceChanged
        } = resolution;

        // ================================================================
        // GEOMETRY REFRESH GATE:
        // Detect conditions requiring highlight recalculation without
        // sentence or page change (e.g., zoom, viewport resize, cache miss).
        // ================================================================
        const geometryRefreshRequired =
            !sentenceChanged &&
            !this.state.pdf.isPageRendering &&
            !this.timeline.pageRenderPending &&
            (
                this.timeline.lastEvent === 'PAGE_TURN' ||
                this.timeline.cachedCidLens === null ||
                this.timeline.activeCidIndex === null ||
                this.timeline.cachedViewportScale !== this.state.pdf.viewport?.scale
            );

        // ================================================================
        // CASE 1: PAGE_TURN (mid-sentence, word-accurate)
        // MUST run before sentence boundary logic
        // ================================================================
        if (pageChanged) {
            this.timeline.lastEvent = 'PAGE_TURN';
            this.timeline.page = resolvedPage;
            this.timeline.sentenceKey = sentenceKey;

            this.clearHighlights();
            // PAGE_TURN ALWAYS INVALIDATES GEOMETRY
            // Required to bypass idempotency guard in highlightSentenceProgress
            this.state.ui._highlightGeometryInvalidated = true;

            const needsRender = this.state.pdf.pdfDocument && (
                resolvedPage !== this.state.pdf.currentPageNum ||
                this.state.pdf.isPageRendering === true ||
                this.timeline.pageRenderPending === true
            );

            this.state.pdf.pendingHighlightSentence = sentence;
            this.state.pdf.pendingHighlightTime = resolution.timelineTime;

            if (needsRender) {
                this.queueRenderPage(resolvedPage).then(() => {
                    if (this.state.pdf.pendingHighlightSentence) {
                        const s = this.state.pdf.pendingHighlightSentence;
                        const t = this.state.pdf.pendingHighlightTime;
                        this.state.pdf.pendingHighlightSentence = null;
                        this.state.pdf.pendingHighlightTime = null;
                        this.highlightSentenceProgress(s, t);
                    }
                });
            } else {
                requestAnimationFrame(() => {
                    if (this.state.pdf.pendingHighlightSentence) {
                        const s = this.state.pdf.pendingHighlightSentence;
                        const t = this.state.pdf.pendingHighlightTime;
                        this.state.pdf.pendingHighlightSentence = null;
                        this.state.pdf.pendingHighlightTime = null;
                        this.highlightSentenceProgress(s, t);
                    }
                });
            }

            this.state.ui.lastAutoTurnedPage = resolvedPage;
            return; // PAGE_TURN consumes this tick
        }

        // ================================================================
        // CASE 2: SENTENCE_BOUNDARY (new sentence) or GEOMETRY_REFRESH
        // Only fires if no page turn happened.
        // GEOMETRY_REFRESH reuses this path when highlights need
        // recalculation without an actual sentence change.
        // ================================================================
        if (sentenceChanged || geometryRefreshRequired) {
            // Distinguish event type for debugging/tracing
            if (!sentenceChanged && geometryRefreshRequired) {
                this.timeline.lastEvent = 'GEOMETRY_REFRESH';
            } else {
                this.timeline.lastEvent = 'SENTENCE_BOUNDARY';
            }
            this.timeline.sentenceKey = sentenceKey;
            this.timeline.suppressedUntilSentenceKey = null;
            this.timeline.activeCidIndex = null;

            this.clearHighlights();

            // SENTENCE/GEOMETRY change ALWAYS INVALIDATES GEOMETRY
            this.state.ui._highlightGeometryInvalidated = true;

            const needsRender = this.state.pdf.pdfDocument && (
                resolvedPage !== this.state.pdf.currentPageNum ||
                this.state.pdf.isPageRendering === true ||
                this.timeline.pageRenderPending === true
            );

            this.state.pdf.pendingHighlightSentence = sentence;
            this.state.pdf.pendingHighlightTime = resolution.timelineTime;

            if (needsRender) {
                this.queueRenderPage(resolvedPage).then(() => {
                    if (this.state.pdf.pendingHighlightSentence) {
                        const s = this.state.pdf.pendingHighlightSentence;
                        const t = this.state.pdf.pendingHighlightTime;
                        this.state.pdf.pendingHighlightSentence = null;
                        this.state.pdf.pendingHighlightTime = null;
                        this.highlightSentenceProgress(s, t);
                    }
                });
            } else {
                requestAnimationFrame(() => {
                    if (this.state.pdf.pendingHighlightSentence) {
                        const s = this.state.pdf.pendingHighlightSentence;
                        const t = this.state.pdf.pendingHighlightTime;
                        this.state.pdf.pendingHighlightSentence = null;
                        this.state.pdf.pendingHighlightTime = null;
                        this.highlightSentenceProgress(s, t);
                    }
                });
            }

            this.state.ui.lastAutoTurnedPage = resolvedPage;
        }
    }

    /******************************************************************************
     * SECTION D — UI JSON - Data Normalization & Indexing
     ******************************************************************************/


    async loadAudioTiming(bookId, statusData = {}) {
        this.state.audiobook.audioTiming = null;
        if (!bookId || statusData.audio_timing_ready === false) return null;
        try {
            const encoded = encodeURIComponent(bookId);
            const response = await fetch(`/api/v1/audiobooks/${encoded}/audio-timing`);
            if (!response.ok) {
                if (response.status !== 404) {
                    console.warn(`[loadAudioTiming] audio timing unavailable: ${response.status}`);
                }
                return null;
            }
            const timing = await response.json();
            this.state.audiobook.audioTiming = timing;
            this.applyAudioTimingToChunksAndSentences(timing);
            console.log(`[loadAudioTiming] Loaded ${timing?.summary?.timing_chunk_count || 0} timing chunks`);
            return timing;
        } catch (error) {
            console.warn('[loadAudioTiming] failed:', error);
            return null;
        }
    }

    applyAudioTimingToChunksAndSentences(timing) {
        if (!timing || !Array.isArray(timing.chunks)) return;
        const chunkMap = new Map();
        for (const chunk of timing.chunks) {
            if (chunk && chunk.chunk_id != null) chunkMap.set(chunk.chunk_id, chunk);
        }
        for (const chunk of (this.state.audiobook.chunks || [])) {
            const timingChunk = chunkMap.get(chunk.chunk_id);
            if (!timingChunk) continue;
            if (typeof timingChunk.actual_start === 'number') chunk.start_time = timingChunk.actual_start;
            if (typeof timingChunk.actual_duration_seconds === 'number') chunk.duration_seconds = timingChunk.actual_duration_seconds;
            if (typeof timingChunk.actual_end === 'number') chunk.end_time = timingChunk.actual_end;
            chunk.timing_basis = timingChunk.timing_basis || chunk.timing_basis;
            chunk.timing_confidence = timingChunk.confidence || chunk.timing_confidence;
        }
        if (this.state.audiobook.uiSentences && Array.isArray(this.state.audiobook.uiSentences.sentences)) {
            this.applyAudioTimingToSentenceList(this.state.audiobook.uiSentences.sentences, timing);
        }
        if (this.uiIndex && this.uiIndex.byGlobalIndex) {
            for (const chunk of timing.chunks) {
                for (const sentTiming of (chunk.sentences || [])) {
                    const idx = sentTiming.global_index;
                    if (idx == null) continue;
                    const sentence = this.uiIndex.byGlobalIndex.get(idx);
                    if (sentence) this.applySentenceTiming(sentence, sentTiming);
                }
            }
        }
    }

    applyAudioTimingToSentenceList(sentences, timing) {
        const byGlobal = new Map();
        for (const chunk of (timing?.chunks || [])) {
            for (const sentTiming of (chunk.sentences || [])) {
                if (sentTiming.global_index != null) byGlobal.set(sentTiming.global_index, sentTiming);
            }
        }
        for (const sentence of (sentences || [])) {
            const sentTiming = byGlobal.get(sentence.global_index);
            if (sentTiming) this.applySentenceTiming(sentence, sentTiming);
        }
    }

    applySentenceTiming(sentence, sentTiming) {
        if (!sentence || !sentTiming) return;
        sentence.timing = sentence.timing || {};
        if (typeof sentTiming.actual_start === 'number') {
            sentence.timing.start = sentTiming.actual_start;
            sentence.timing.actual_start = sentTiming.actual_start;
        }
        if (typeof sentTiming.actual_end === 'number') {
            sentence.timing.end = sentTiming.actual_end;
            sentence.timing.actual_end = sentTiming.actual_end;
        }
        if (typeof sentTiming.actual_duration_seconds === 'number') {
            sentence.timing.actual_duration_seconds = sentTiming.actual_duration_seconds;
        }
        sentence.timing.basis = sentTiming.timing_basis || sentence.timing.basis || 'estimated_text';
        sentence.timing.confidence = sentTiming.confidence || sentence.timing.confidence || 'low';
    }

    async loadArtifactShardIndexes(bookId) {
        const shards = this.state.audiobook.shards;
        if (!shards) return;

        async function fetchIndex(url) {
            try {
                const response = await fetch(url);
                if (!response.ok) return null;
                const data = await response.json();
                return (data && Array.isArray(data.pages)) ? data : null;
            } catch (error) {
                console.warn(`[shards] Failed to load index ${url}:`, error);
                return null;
            }
        }

        const encoded = encodeURIComponent(bookId);
        shards.uiSentencesIndex = await fetchIndex(`/api/v1/audiobooks/${encoded}/ui-sentences/pages/index`);
        shards.semanticIndex = await fetchIndex(`/api/v1/audiobooks/${encoded}/semantic/pages/index`);

        const uiPages = shards.uiSentencesIndex?.pages?.length || 0;
        const semanticPages = shards.semanticIndex?.pages?.length || 0;
        if (uiPages || semanticPages) {
            console.log(`[shards] Available indexes: ui=${uiPages} pages, semantic=${semanticPages} pages`);
        }
    }

    shouldPreferPageShards(statusData) {
        const uiIndex = this.state.audiobook.shards?.uiSentencesIndex;
        if (!uiIndex || !Array.isArray(uiIndex.pages) || uiIndex.pages.length === 0) {
            return false;
        }

        if (!statusData.ui_ready) {
            return true;
        }

        const pageCount = uiIndex.summary?.total_pages || uiIndex.pages.length;
        return pageCount > FULL_UI_JSON_PAGE_LIMIT;
    }

    _ensureIncrementalUiIndex() {
        if (this.uiIndex) {
            if (!this.uiIndex._loadedShardPages) this.uiIndex._loadedShardPages = new Set();
            if (!this.uiIndex._loadedSemanticPages) this.uiIndex._loadedSemanticPages = new Set();
            if (!this.uiIndex._indexedSentenceIds) {
                this.uiIndex._indexedSentenceIds = new Set(this.uiIndex.byGlobalIndex?.keys?.() || []);
            }
            return;
        }

        this.uiIndex = {
            byGlobalIndex: new Map(),
            byChunkId: new Map(),
            byPage: {},
            byTimeByChunk: new Map(),
            pageTurnsByTime: [],
            geometryByPage: {},
            spanIndex: null,
            geometryAuthority: 'ui_sentences_page_shards',
            spanIndexAuthority: null,
            _loadedShardPages: new Set(),
            _loadedSemanticPages: new Set(),
            _indexedSentenceIds: new Set(),
            _shardMode: true,
        };
    }

    _indexUiSentence(sentence) {
        if (!sentence || typeof sentence !== 'object') return;
        const gIdx = sentence.global_index;
        const chunkId = sentence.chunk_id;
        if (gIdx == null || chunkId == null) return;
        if (this.uiIndex._indexedSentenceIds.has(gIdx)) return;

        this.uiIndex._indexedSentenceIds.add(gIdx);
        this.uiIndex.byGlobalIndex.set(gIdx, sentence);

        if (!this.uiIndex.byChunkId.has(chunkId)) {
            this.uiIndex.byChunkId.set(chunkId, []);
        }
        this.uiIndex.byChunkId.get(chunkId).push(sentence);

        const pages = Array.isArray(sentence.pages) ? sentence.pages : [];
        for (const page of pages) {
            const pageNum = Number(page);
            if (!Number.isFinite(pageNum) || pageNum < 1) continue;
            const key = String(pageNum);
            if (!Array.isArray(this.uiIndex.byPage[key])) this.uiIndex.byPage[key] = [];
            this.uiIndex.byPage[key].push(gIdx);
        }

        if (sentence.page_turn && typeof sentence.page_turn.turn_time === 'number') {
            this.uiIndex.pageTurnsByTime.push({
                turn_time: sentence.page_turn.turn_time,
                to_page: sentence.page_turn.to_page,
                from_page: sentence.page_turn.from_page,
                globalIndex: gIdx
            });
        }

        if (sentence.geometry && typeof sentence.geometry === 'object') {
            for (const [pageStr, geoList] of Object.entries(sentence.geometry)) {
                const pageNum = parseInt(pageStr, 10);
                if (Number.isNaN(pageNum) || !Array.isArray(geoList)) continue;
                if (!this.uiIndex.geometryByPage[pageNum]) {
                    this.uiIndex.geometryByPage[pageNum] = [];
                }
                for (const geo of geoList) {
                    if (!geo?.bbox || !geo?.cid) continue;
                    this.uiIndex.geometryByPage[pageNum].push({
                        bbox: geo.bbox,
                        cid: geo.cid,
                        globalIndex: gIdx,
                        chunkId: chunkId,
                        timingStart: sentence.timing?.start ?? null,
                        timingEnd: sentence.timing?.end ?? null
                    });
                }
            }
        }
    }

    _finalizeIncrementalUiIndex() {
        if (!this.uiIndex) return false;

        this.uiIndex.pageTurnsByTime.sort((a, b) => a.turn_time - b.turn_time);
        this.uiIndex.byTimeByChunk = new Map();
        for (const [chunkId, chunkSentences] of this.uiIndex.byChunkId.entries()) {
            const sorted = [...chunkSentences].sort((a, b) => {
                const aStart = a.timing?.start ?? 0;
                const bStart = b.timing?.start ?? 0;
                return aStart - bStart;
            });
            this.uiIndex.byTimeByChunk.set(chunkId, sorted);
        }

        const ready = (
            this.uiIndex.byGlobalIndex.size > 0 &&
            Object.keys(this.uiIndex.byPage).length > 0 &&
            Object.keys(this.uiIndex.geometryByPage).length > 0
        );
        this.state.audiobook.uiPageTurns = this.uiIndex.pageTurnsByTime || [];
        this.state.audiobook.uiReady = ready;
        this._resetPageTurnIndexForTime(this.timeline?.t ?? 0);
        return ready;
    }

    async ensureUiDataForChunk(chunk) {
        if (!chunk || !this.state.audiobook.shards?.usePageShards) return;
        const pages = new Set();
        for (const value of (chunk.pages || [chunk.page])) {
            const page = Number(value);
            if (Number.isFinite(page) && page > 0) pages.add(page);
        }
        for (const page of pages) {
            await this.ensureUiPageShard(page);
            await this.ensureSemanticPageShard(page);
        }
    }

    async ensureUiPageShard(pageNumber) {
        const shards = this.state.audiobook.shards;
        const bookId = this.state.audiobook.bookId;
        if (!bookId || !shards?.uiSentencesIndex) return false;
        if (shards.uiPageCache.has(pageNumber)) return true;

        try {
            const encoded = encodeURIComponent(bookId);
            const response = await fetch(`/api/v1/audiobooks/${encoded}/ui-sentences/pages/${pageNumber}`);
            if (!response.ok) return false;
            const pageData = await response.json();
            const sentences = Array.isArray(pageData?.sentences) ? pageData.sentences : [];
            if (this.state.audiobook.audioTiming) {
                this.applyAudioTimingToSentenceList(sentences, this.state.audiobook.audioTiming);
            }
            shards.uiPageCache.set(pageNumber, pageData);

            this._ensureIncrementalUiIndex();
            for (const sentence of sentences) {
                this._indexUiSentence(sentence);
            }
            this.uiIndex._loadedShardPages.add(pageNumber);
            this._finalizeIncrementalUiIndex();
            return true;
        } catch (error) {
            console.warn(`[shards] Failed to load UI sentence page ${pageNumber}:`, error);
            return false;
        }
    }

    async ensureSemanticPageShard(pageNumber) {
        const shards = this.state.audiobook.shards;
        const bookId = this.state.audiobook.bookId;
        if (!bookId || !shards?.semanticIndex) return false;
        if (shards.semanticPageCache.has(pageNumber)) return true;

        try {
            const encoded = encodeURIComponent(bookId);
            const response = await fetch(`/api/v1/audiobooks/${encoded}/semantic/pages/${pageNumber}`);
            if (!response.ok) return false;
            const pageData = await response.json();
            shards.semanticPageCache.set(pageNumber, pageData);

            this._ensureIncrementalUiIndex();
            const pageSpanIndex = this.buildSpanIndex({spans: pageData.spans || {}});
            if (!this.uiIndex.spanIndex) {
                this.uiIndex.spanIndex = new Map();
                this.uiIndex.spanIndex._authority = 'semantic_page_shard';
                this.uiIndex.spanIndex._sentenceSafe = false;
            }
            for (const [cid, entry] of pageSpanIndex.entries()) {
                this.uiIndex.spanIndex.set(cid, entry);
            }
            this.uiIndex.spanIndexAuthority = 'semantic_page_shards';
            this.uiIndex._loadedSemanticPages.add(pageNumber);
            return true;
        } catch (error) {
            console.warn(`[shards] Failed to load semantic page ${pageNumber}:`, error);
            return false;
        }
    }

    /**
     * === UI JSON Index Builder
     *
     * Builds derived lookup structures from ui_sentences.json.
     * This is the SOLE source of UI data
     *
     * Produces (on this.uiIndex):
     *   - byGlobalIndex: Map<number, UISentence> — O(1) sentence lookup
     *   - byChunkId: Map<number, UISentence[]> — chunk-scoped arrays
     *   - byPage: Object<page, number[]> — page → sentence indices (from backend)
     *   - byTimeByChunk: Map<chunkId, sorted UISentence[]> — binary search within chunk
     *   - pageTurnsByTime: Array<{turn_time, to_page, from_page, globalIndex}> — cursor-based
     *   - geometryByPage: Object<page, Array<{bbox, cid, globalIndex, chunkId}>> — hit-testing
     *
     * NOTE: This method builds indexes only. State wiring (uiReady, uiPageTurns)
     * is handled by the caller (loadAudiobook) to maintain separation.
     * * NOTE on geometryByPage:
     *  *   - Same CID may appear multiple times with different globalIndex values (shared CIDs)
     *  *   - Hit-testing should prefer the entry whose timing window contains current playback time
     *  *   - If no timing match, prefer the entry with lowest globalIndex (earliest sentence)
     *
     * @returns {boolean} True if index built successfully with valid data
     */
    buildUiIndex() {
        const uiSentences = this.state.audiobook.uiSentences;

        if (!uiSentences || !Array.isArray(uiSentences.sentences)) {
            console.warn('[buildUiIndex] No valid UI sentences data available');
            this.uiIndex = null;
            return false;
        }

        const sentences = uiSentences.sentences;
        const pageIndex = uiSentences.page_index || {};

        // Initialize index structure
        // NOTE: Map used for numeric keys (iteration order, proper equality)
        //       Object used for page-based lookups (JSON-origin keys)
        this.uiIndex = {
            byGlobalIndex: new Map(),
            byChunkId: new Map(),
            byPage: {},
            byTimeByChunk: new Map(),
            pageTurnsByTime: [],
            geometryByPage: {}
        };

        // === Build indexes from sentences ===
        for (const sentence of sentences) {
            const gIdx = sentence.global_index;
            const chunkId = sentence.chunk_id;

            // 1. byGlobalIndex: O(1) sentence lookup
            this.uiIndex.byGlobalIndex.set(gIdx, sentence);

            // 2. byChunkId: chunk-scoped sentence arrays
            if (!this.uiIndex.byChunkId.has(chunkId)) {
                this.uiIndex.byChunkId.set(chunkId, []);
            }
            this.uiIndex.byChunkId.get(chunkId).push(sentence);

            // 3. pageTurnsByTime: collect page turns for cursor-based resolution
            if (sentence.page_turn && typeof sentence.page_turn.turn_time === 'number') {
                this.uiIndex.pageTurnsByTime.push({
                    turn_time: sentence.page_turn.turn_time,
                    to_page: sentence.page_turn.to_page,
                    from_page: sentence.page_turn.from_page,
                    globalIndex: gIdx
                });
            }

            // 4. geometryByPage: flattened geometry for hit-testing
            if (sentence.geometry && typeof sentence.geometry === 'object') {
                for (const [pageStr, geoList] of Object.entries(sentence.geometry)) {
                    const pageNum = parseInt(pageStr, 10);
                    if (isNaN(pageNum)) continue;

                    if (!this.uiIndex.geometryByPage[pageNum]) {
                        this.uiIndex.geometryByPage[pageNum] = [];
                    }

                    for (const geo of geoList) {
                        if (!geo.bbox || !geo.cid) continue;
                        this.uiIndex.geometryByPage[pageNum].push({
                            bbox: geo.bbox,
                            cid: geo.cid,
                            globalIndex: gIdx,
                            chunkId: chunkId,
                            // Timing context for temporal hit-testing disambiguation
                            timingStart: sentence.timing?.start ?? null,
                            timingEnd: sentence.timing?.end ?? null
                        });
                    }
                }
            }
        }

        // === Compute shared CID diagnostics for geometryByPage ===
        // This helps downstream methods understand index characteristics
        const cidOccurrences = new Map();
        for (const pageEntries of Object.values(this.uiIndex.geometryByPage)) {
            for (const entry of pageEntries) {
                cidOccurrences.set(entry.cid, (cidOccurrences.get(entry.cid) || 0) + 1);
            }
        }
        this.uiIndex._sharedGeometryCidCount = Array.from(cidOccurrences.values()).filter(n => n > 1).length;


        // Sort pageTurnsByTime ascending by turn_time
        this.uiIndex.pageTurnsByTime.sort((a, b) => a.turn_time - b.turn_time);

        // Sort byTimeByChunk: each chunk's sentences by timing.start
        for (const [chunkId, chunkSentences] of this.uiIndex.byChunkId.entries()) {
            const sorted = [...chunkSentences].sort((a, b) => {
                const aStart = a.timing?.start ?? 0;
                const bStart = b.timing?.start ?? 0;
                return aStart - bStart;
            });
            this.uiIndex.byTimeByChunk.set(chunkId, sorted);
        }

        // === Copy page_index from UI JSON ===
        // byPage is taken directly from UI JSON page_index (authoritative backend mapping)
        this.uiIndex.byPage = pageIndex;

        // === Compute readiness (returned to caller for state wiring) ===
        const sentenceCount = sentences.length;
        const pageCount = Object.keys(this.uiIndex.byPage).length;
        const geometryPageCount = Object.keys(this.uiIndex.geometryByPage).length;

        const uiReady = (
            sentenceCount > 0 &&
            pageCount > 0 &&
            geometryPageCount > 0
        );

        this.uiIndex._loadedShardPages = new Set();
        this.uiIndex._loadedSemanticPages = new Set();
        this.uiIndex._indexedSentenceIds = new Set(this.uiIndex.byGlobalIndex.keys());
        this.uiIndex._shardMode = false;

        console.log(
            `[buildUiIndex] Complete: ${sentenceCount} sentences, ` +
            `${pageCount} pages, ${geometryPageCount} geometry pages, ` +
            `${this.uiIndex.pageTurnsByTime.length} page turns, ` +
            `${this.uiIndex._sharedGeometryCidCount || 0} shared geometry CIDs. ` +
            `Ready=${uiReady}`
        );

        return uiReady;
    }

    /******************************************************************************
     * SECTION E — PDF Rendering
     ******************************************************************************/

    /**
     * Load PDF document
     * @param {string} pdfSourceFilename - PDF file URL (just the filename)
     * @returns {Promise<void>}
     */
    async loadPdf(pdfSourceFilename) {
        try {
            if (typeof window.pdfjsLib === 'undefined') {
                this.state.pdf.documentUrl = null;
                if (this.elements.pdfViewerContainer) {
                    this.elements.pdfViewerContainer.style.display = 'none';
                }
                if (this.elements.pdfPageControls) {
                    this.elements.pdfPageControls.style.display = 'none';
                }
                if (this.elements.clickSeekToggle) {
                    this.elements.clickSeekToggle.disabled = true;
                    this.elements.clickSeekToggle.textContent = 'Seek: PDF unavailable';
                }
                console.warn('[loadPdf] Local PDF.js assets unavailable; PDF viewer disabled for this session.');
                return;
            }

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

            // Enable click-to-seek only when backend-authoritative UI geometry is loaded.
            if (this.state.audiobook.mode === 'audiobook' && this.state.audiobook.uiReady) {
                if (this.elements.clickSeekToggle) {
                    this.elements.clickSeekToggle.disabled = false;
                    this.elements.clickSeekToggle.style.opacity = '1';
                    this.elements.clickSeekToggle.style.cursor = 'pointer';
                    this.elements.clickSeekToggle.textContent = 'Seek: OFF';
                }
            } else if (this.elements.clickSeekToggle) {
                this.elements.clickSeekToggle.disabled = true;
                this.elements.clickSeekToggle.style.opacity = '0.6';
                this.elements.clickSeekToggle.textContent = 'Seek: UI unavailable';
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

        // Set render lock (prevents simultaneous renders)
        if (this.state.pdf.isPageRendering) {
            this.state.pdf.pendingPageNum = pageNum;
            return;
        }

        this.state.pdf.isPageRendering = true;
        this.timeline.pageRenderPending = true;  // Signal to highlight logic
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
                    const desiredHeight = this.elements.pdfViewerContainer.clientHeight;
                    scale = desiredHeight / viewportDefault.height;
                } else {
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

            // Set highlight container size to match canvas
            this.elements.highlightContainer.style.width = canvas.style.width;
            this.elements.highlightContainer.style.height = canvas.style.height;

            // Render with transform for HiDPI
            const transform = outputScale !== 1
                ? [outputScale, 0, 0, outputScale, 0, 0]
                : null;

            const renderContext = {
                canvasContext: context,
                viewport: viewport,
                transform: transform
            };

            // Render PDF
            await page.render(renderContext).promise;

            // Defer offset calculation until layout reflow completes
            requestAnimationFrame(() => {
                const canvasEl = this.elements.pdfCanvas;
                const containerEl = this.elements.pdfViewerContainer;

                if (!canvasEl || !containerEl) return;

                const canvasBounds = canvasEl.getBoundingClientRect();
                const containerBounds = containerEl.getBoundingClientRect();

                // Account for scroll position within container
                const scrollLeft = containerEl.scrollLeft;
                const scrollTop = containerEl.scrollTop;

                const offsetLeft = (canvasBounds.left - containerBounds.left) + scrollLeft;
                const offsetTop = (canvasBounds.top - containerBounds.top) + scrollTop;

                this.elements.highlightContainer.style.left = `${offsetLeft}px`;
                this.elements.highlightContainer.style.top = `${offsetTop}px`;
                this.elements.highlightContainer.style.pointerEvents = 'none';
                this.elements.highlightContainer.style.position = 'absolute';
                this.elements.highlightContainer.style.zIndex = '10';
            });

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
            this.timeline.pageRenderPending = false;  // Clear render signal

            // === APPLY PENDING HIGHLIGHT ===
            // Only apply if:
            //   1. Rendered page matches timeline's expected page
            //   2. Pending sentence is still the active sentence (staleness check)
            //
            // State synchronization:
            //   - pendingHighlightSentence: Set by _applyTimelineEvents before render
            //   - timeline.page: Authoritative expected page from timeline state machine
            //   - timeline.sentenceKey: Authoritative current sentence (global_index)
            //   - currentPageNum: Actual rendered page
            //
            // If any mismatch detected, pending highlight is discarded to prevent
            // incorrect highlights. The polling loop will apply the correct highlight.
            if (this.state.pdf.pendingHighlightSentence) {
                const renderedPage = this.state.pdf.currentPageNum;
                const expectedPage = this.timeline.page;
                const pendingSentenceKey = this.state.pdf.pendingHighlightSentence?.global_index;
                const currentSentenceKey = this.timeline.sentenceKey;

                if (renderedPage === expectedPage && pendingSentenceKey === currentSentenceKey) {
                    // Correct page rendered: apply and clear pending
                    const pendingSentence = this.state.pdf.pendingHighlightSentence;
                    const pendingTime = this.state.pdf.pendingHighlightTime;
                    this.state.pdf.pendingHighlightSentence = null;
                    this.state.pdf.pendingHighlightTime = null;

                    requestAnimationFrame(() => {
                        this.highlightSentenceProgress(pendingSentence, pendingTime);
                    });
                } else {
                    // Pending highlight is stale (page or sentence mismatch)
                    // Drop it safely; a newer resolution will reapply if needed
                    console.debug(
                        `[renderPage] Discarding stale pending highlight: ` +
                        `pendingSentence=${pendingSentenceKey}, currentSentence=${currentSentenceKey}, ` +
                        `renderedPage=${renderedPage}, expectedPage=${expectedPage}`
                    );
                    this.state.pdf.pendingHighlightSentence = null;
                    this.state.pdf.pendingHighlightTime = null;
                }
            }

            // Re-enable buttons based on page number
            if (this.elements.prevPageButton) {
                this.elements.prevPageButton.disabled = (pageNum === 1);
            }
            if (this.elements.nextPageButton) {
                this.elements.nextPageButton.disabled = (pageNum === this.state.pdf.totalPages);
            }

            // Process pending page request
            if (this.state.pdf.pendingPageNum !== null) {
                const pending = this.state.pdf.pendingPageNum;
                this.state.pdf.pendingPageNum = null;
                await this.renderPage(pending);
            }
        }
    }

    /**
     * Queue page render (safe for rapid calls)
     * @param {number} pageNum - Page number to render
     */
    async queueRenderPage(pageNum) {
        if (this.state.pdf.isPageRendering) {
            // Canvas busy, queue request
            this.state.pdf.pendingPageNum = pageNum;
        } else {
            // Canvas free, render immediately
            await this.renderPage(pageNum);
        }
    }

    /**
     * Calculate and apply fit-to-height scale
     * @param {number} pageNum - Page to render
     * @private
     */
    async _calculateFitHeight(pageNum) {
        try {
            const page = await this.state.pdf.pdfDocument.getPage(pageNum);

            const desiredHeight = this.elements.pdfViewerContainer.clientHeight;
            const viewportDefault = page.getViewport({scale: 1.0});
            this.state.pdf.scale = desiredHeight / viewportDefault.height;
            await this.renderPage(pageNum);

        } catch (error) {
            console.error('Failed to calculate fit-to-height:', error);
        }
    }

    /******************************************************************************
     * SECTION E2 — PDF Navigation and Zoom
     ******************************************************************************/

    /**
     * Go to next page
     */
    async nextPage() {
        const nextPageNum = this.state.pdf.currentPageNum + 1;
        if (nextPageNum <= this.state.pdf.totalPages) {
            await this.queueRenderPage(nextPageNum);
        }
    }

    /**
     * Go to previous page
     */
    async previousPage() {
        const prevPageNum = this.state.pdf.currentPageNum - 1;
        if (prevPageNum >= 1) {
            await this.queueRenderPage(prevPageNum);
        }
    }

    /**
     * Jump to specific page number
     * @param {string|number} pageNum - Page number (1-indexed)
     */
    async jumpToPage(pageNum) {
        const page =
            typeof pageNum === 'number'
                ? pageNum
                : parseInt(pageNum, 10);

        if (isNaN(page)) {
            this.logError('Invalid page number');
            return;
        }

        if (page < 1 || page > this.state.pdf.totalPages) {
            this.logError(`Page must be between 1 and ${this.state.pdf.totalPages}`);
            return;
        }

        await this.queueRenderPage(page);

        // Clear input after successful jump
        if (this.elements.pageJumpInput) {
            this.elements.pageJumpInput.value = '';
        }
    }

    /**
     * Zoom in on PDF
     */
    async zoomIn() {
        if (!this.state.pdf.pdfDocument) return;

        const newScale = this.state.pdf.scale * 1.25;
        const maxScale = 3.0;

        if (newScale <= maxScale) {
            this.state.pdf.scale = newScale;
            this.clearHighlights();
            await this.renderPage(this.state.pdf.currentPageNum);
            this.updateZoomDisplay();
        }
    }

    /**
     * Zoom out on PDF
     */
    async zoomOut() {
        if (!this.state.pdf.pdfDocument) return;

        // Decrease scale by 25%
        const newScale = this.state.pdf.scale * 0.75;
        const minScale = 0.5;  // 50% min zoom

        if (newScale >= minScale) {
            this.state.pdf.scale = newScale;
            this.clearHighlights();
            await this.renderPage(this.state.pdf.currentPageNum);
            this.updateZoomDisplay();
        }
    }

    /**
     * Fit PDF to width of container
     */
    async zoomFitWidth() {
        if (!this.state.pdf.pdfDocument) return;

        this.state.pdf.fitMode = 'width';
        this.state.pdf.scale = null;
        this.clearHighlights();

        await this.renderPage(this.state.pdf.currentPageNum);

        console.log('Fit to width');
    }

    /**
     * Fit PDF to height of container
     */
    async zoomFitHeight() {
        if (!this.state.pdf.pdfDocument) return;

        this.state.pdf.fitMode = 'height';
        this.clearHighlights();

        await this._calculateFitHeight(this.state.pdf.currentPageNum);
    }

    async resetZoom() {
        if (!this.state.pdf.pdfDocument) return;

        this.state.pdf.scale = 1.0;
        this.state.pdf.fitMode = 'height';

        this.clearHighlights();
        await this.renderPage(this.state.pdf.currentPageNum);
        this.updateZoomDisplay();
    }

    /**
     * Update zoom level display
     */
    updateZoomDisplay() {
        if (!this.elements.zoomLevel) return;

        const percentage = Math.round(this.state.pdf.scale * 100);
        this.elements.zoomLevel.textContent = `${percentage}%`;
    }

    /******************************************************************************
     * SECTION F: Sentence Visualization (GEOMETRY & HIGHLIGHTING)
     ******************************************************************************/


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

        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }

        // Sync logical state with visual state
        this.state.ui.activeHighlightSentenceKey = null;
        this.state.ui.activeHighlightKey = null;

        // === Geometry invalidation marker ===
        // Indicates that all highlight geometry has been cleared and
        // must be recomputed before any new highlight is applied.
        this.state.ui._highlightGeometryInvalidated = true;

        // Diagnostic: record last clear time (for debugging / audit)
        this.state.ui._lastHighlightClearTs = performance.now?.() ?? Date.now();
    }

    /**
     * Compute sentence-owned start/end fractions for each CID.
     *
     * Returns a Map<cid, {startFrac, endFrac}> with an additional property:
     *   - _authoritative: true if structural ownership was available,
     *                     false if fractions are full-span fallbacks
     *
     * Callers should check fractions._authoritative before using fractions
     * for precise boundary inference (text slicing, page turn timing).
     *
     * @param {Object} sentence - UI sentence with cids, text, contract markers
     * @param {Map} spanIndex - Map of CID → span metadata
     * @returns {Map} Fractions map with _authoritative marker
     */
    computeSentenceSpanFractions(sentence, spanIndex) {
        const fractions = new Map();

        if (!sentence || !spanIndex) return fractions;

        const cids = Array.isArray(sentence.cids) ? sentence.cids : [];
        if (cids.length === 0) return fractions;

        // Default: full span (safe)
        for (const cid of cids) {
            fractions.set(cid, {startFrac: 0, endFrac: 1});
        }

        // Initialize authority markers
        fractions._authoritative = false;
        fractions._textRefined = false;

        // ─────────────────────────────────────────────────────────────────
        // PHASE 1: STRUCTURAL OWNERSHIP (authoritative)
        // ─────────────────────────────────────────────────────────────────
        const sourceCidsRaw =
            sentence.source_cids ||
            sentence._source_span_ids ||
            sentence._semantic_projection?.source_span_ids ||
            null;

        if (Array.isArray(sourceCidsRaw) && sourceCidsRaw.length > 0) {
            // Structural ownership data exists — attempt Phase 1
            const ownedSet = new Set(sourceCidsRaw);

            let firstIdx = -1;
            let lastIdx = -1;

            for (let i = 0; i < cids.length; i++) {
                if (ownedSet.has(cids[i])) {
                    if (firstIdx === -1) firstIdx = i;
                    lastIdx = i;
                }
            }

            if (firstIdx !== -1) {
                // Structural ownership validated — fractions are authoritative
                fractions._authoritative = true;
                fractions._authoritySource = 'structural';
                // Clamp non-owned CIDs to zero width
                for (let i = 0; i < cids.length; i++) {
                    if (i < firstIdx) {
                        fractions.set(cids[i], {startFrac: 1, endFrac: 1});
                    } else if (i > lastIdx) {
                        fractions.set(cids[i], {startFrac: 0, endFrac: 0});
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // PHASE 2: TEXT-ANCHORED SENTENCE BOUNDARY PROJECTION (SAFE)
        // ------------------------------------------------------------
        // Goal: If structural ownership is missing, derive per-CID start/end
        // fractions by locating sentence.text within the concatenated CID text.
        //
        // IMPORTANT SAFETY RULES:
        //   1) Only run when structural ownership is NOT authoritative.
        //   2) Use stable normalization and preserve word boundaries.
        //   3) If we cannot prove a unique match, fall back safely.
        //   4) If match succeeds, mark fractions as authoritative for clipping.
        // ─────────────────────────────────────────────────────────────────
        if (!sentence.text) return fractions;

        // Skip text anchoring only when structural authority is sufficient
        if (fractions._authoritative === true && fractions._authoritySource === 'structural') {
            fractions._textRefined = false;
            return fractions;
        }

        const normalize = (s) => (s || '').replace(/\s+/g, ' ').trim();

        // Build concatenated stream WITH SEPARATORS to preserve word boundaries
        const cidTextInfo = [];
        let combined = '';
        for (const cid of cids) {
            const meta = spanIndex.get(cid);
            const raw = meta?.cleanedText || '';
            const txt = normalize(raw);

            const startOffset = combined.length;
            combined += txt;

            const endOffset = combined.length;

            // Add a single space separator BETWEEN cids (not after last)
            combined += ' ';

            cidTextInfo.push({
                cid,
                text: txt,
                startOffset,
                endOffset
            });
        }

        // Remove trailing separator space (keeps offsets valid up to endOffset)
        combined = combined.trimEnd();

        const needle = normalize(sentence.text);
        if (!needle || needle.length < 3) return fractions;

        // Find match - try exact first, then partial if CID coverage is incomplete
        let matchIdx = combined.indexOf(needle);
        let matchedNeedle = needle;

        if (matchIdx === -1) {
            // Exact match failed - CIDs may not cover full sentence text
            // Strategy: Find longest matching prefix (sentence may extend beyond CID coverage)
            const minMatchLen = Math.max(30, Math.floor(needle.length * 0.4));

            // Binary search for longest matching prefix
            let lo = minMatchLen, hi = needle.length, bestLen = 0;
            while (lo <= hi) {
                const mid = Math.floor((lo + hi) / 2);
                const prefix = needle.slice(0, mid);
                if (combined.includes(prefix)) {
                    bestLen = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }

            if (bestLen >= minMatchLen) {
                matchedNeedle = needle.slice(0, bestLen);
                matchIdx = combined.indexOf(matchedNeedle);
            }
        }

        if (matchIdx === -1) {
            // Cannot prove boundaries → remain non-authoritative full-span defaults
            fractions._authoritative = false;
            return fractions;
        }

        // Uniqueness check on matched portion
        const secondIdx = combined.indexOf(matchedNeedle, matchIdx + 1);
        if (secondIdx !== -1) {
            fractions._authoritative = false;
            return fractions;
        }

        const sentenceStart = matchIdx;
        const sentenceEnd = matchIdx + matchedNeedle.length;

        // Map overlap back to each CID as fractions
        let hasZeroFractionFromTruncation = false;

        for (const info of cidTextInfo) {
            const cid = info.cid;
            const txt = info.text;
            const cidStart = info.startOffset;
            const cidEnd = info.endOffset;

            if (!txt || txt.length === 0) {
                fractions.set(cid, {startFrac: 0, endFrac: 0});
                continue;
            }

            const overlapStart = Math.max(cidStart, sentenceStart);
            const overlapEnd = Math.min(cidEnd, sentenceEnd);

            if (overlapEnd <= overlapStart) {
                // CID lies outside matched range
                if (matchedNeedle.length < needle.length) {
                    // Truncation → cannot prove exclusion
                    fractions.set(cid, {startFrac: 0, endFrac: 1});
                    hasZeroFractionFromTruncation = true;
                } else {
                    fractions.set(cid, {startFrac: 0, endFrac: 0});
                }
                continue;
            }

            const localStart = (overlapStart - cidStart) / txt.length;
            const localEnd = (overlapEnd - cidStart) / txt.length;

            fractions.set(cid, {
                startFrac: Math.max(0, Math.min(1, localStart)),
                endFrac: Math.max(0, Math.min(1, localEnd))
            });
        }

        if (matchedNeedle.length < needle.length) {
            for (const info of cidTextInfo) {
                const frac = fractions.get(info.cid);
                const owned = frac.endFrac - frac.startFrac;

                // CID extends beyond truncation point but got partial credit
                if (info.endOffset > sentenceEnd && owned > 0 && owned < 0.5) {
                    fractions.set(info.cid, {startFrac: 0, endFrac: 1});
                    hasZeroFractionFromTruncation = true;
                }
            }
        }

        // Do NOT claim authority if truncation caused incomplete coverage
        if (hasZeroFractionFromTruncation) {
            fractions._authoritative = false;
            fractions._textRefined = false;
            return fractions;
        }

        // Proven full sentence boundaries
        fractions._authoritative = true;
        fractions._textRefined = true;
        fractions._authoritySource = 'text';

        return fractions;
    }


    /**
     * Build array of {cid, bbox, fraction} allocations for rendering highlights.
     *
     * Geometry sources (preference order):
     *   1. sentence.geometry[page][cid].bbox (sentence-scoped, preferred)
     *   2. spanIndex[cid].bbox (only when fractions are authoritative)
     *
     * Enforcement:
     *   When fractions._authoritative === false (UI sentence without semantic
     *   projection), full-span CIDs are SKIPPED to prevent highlighting text
     *   beyond sentence boundaries. Under-highlighting is preferred over
     *   over-highlighting.
     *
     * @param {Object} sentence - Sentence object with .cids, .geometry
     * @param {number} pageNum - Current page number
     * @param {number} progress - Speech progress [0,1], typically 1 for static mode
     * @returns {Array} Array of {cid, bbox, fraction} allocations
     */
    allocateProgressAcrossSpans(sentence, pageNum, progress) {
        if (!sentence || progress <= 0) {
            return [];
        }

        const spanIndex = this.uiIndex?.spanIndex || null;

        const allCids = sentence.geometry_cids || sentence.source_cids || sentence.cids || [];

        const pageGeom = sentence.geometry || {};
        const currentPageGeometry = Array.isArray(pageGeom[String(pageNum)])
            ? pageGeom[String(pageNum)]
            : [];
        const backendCoverageUsable = currentPageGeometry.some(g =>
            g && typeof g.coverage_start_ratio === 'number' && typeof g.coverage_end_ratio === 'number'
        );
        if (backendCoverageUsable) {
            const pageEntries = currentPageGeometry
                .filter(g => g && g.bbox && g.cid != null)
                .sort((a, b) => allCids.indexOf(a.cid) - allCids.indexOf(b.cid));
            const weights = pageEntries.map((g) => {
                const start = Math.max(0, Math.min(1, Number(g.coverage_start_ratio ?? 0)));
                const end = Math.max(start, Math.min(1, Number(g.coverage_end_ratio ?? 1)));
                const charLen = (typeof g.char_start === 'number' && typeof g.char_end === 'number')
                    ? Math.max(0, g.char_end - g.char_start)
                    : Math.max(1, Math.round((end - start) * 100));
                return Math.max(0, charLen);
            });
            const total = weights.reduce((a, b) => a + b, 0);
            if (total > 0) {
                const spokenBudget = total * Math.max(0, Math.min(1, progress));
                let cursor = 0;
                const allocations = [];
                for (let i = 0; i < pageEntries.length; i++) {
                    const entry = pageEntries[i];
                    const len = weights[i];
                    if (len <= 0) continue;
                    if (spokenBudget <= cursor) break;
                    const inEntry = Math.min(len, spokenBudget - cursor);
                    const frac = Math.max(0, Math.min(1, inEntry / len));
                    const bbox = this.normalizeBbox(entry.bbox);
                    if (bbox) allocations.push({cid: entry.cid, bbox, fraction: frac});
                    cursor += len;
                    if (inEntry < len) break;
                }
                return allocations;
            }
        }

        if (!spanIndex) return [];

        // ─────────────────────────────────────────────────────────────────
        // SENTENCE-LOCAL MEDIAN HEIGHT (for multi-line detection)
        // Avoids headings/footers skewing the baseline.
        // ─────────────────────────────────────────────────────────────────
        const MULTI_LINE_THRESHOLD = 1.7;


        // PAGE-WIDE median height (for multi-line detection)
        // Sentence-local median fails when multi-line span IS the median
        const pageHeights = [];
        for (const [, meta] of spanIndex.entries()) {
            if (Number(meta.page) === Number(pageNum) && meta.bbox) {
                pageHeights.push(meta.bbox[3] - meta.bbox[1]);
            }
        }
        pageHeights.sort((a, b) => a - b);
        const medianHeight = pageHeights.length > 0
            ? pageHeights[Math.floor(pageHeights.length / 2)]
            : 15;
        const saneMedian = Math.max(12, medianHeight);

        // ─────────────────────────────────────────────────────────────────
        // Compute sentence-to-span alignment fractions
        // ─────────────────────────────────────────────────────────────────
        const spanFractions = this.computeSentenceSpanFractions(sentence, spanIndex);

        // ------------------------------------------------------------
        // AUTHORITY CHECK:
        // Only trust fractions when explicitly marked authoritative.
        // When non-authoritative, full-span CIDs will be skipped to
        // prevent highlighting beyond sentence boundaries.
        // ------------------------------------------------------------
        const fractionsAuthoritative = (spanFractions && spanFractions._authoritative === true);

        // Explicit geometry authority detection
        const sentenceGeo = sentence.geometry?.[String(pageNum)];
        const hasSentenceGeometry = Array.isArray(sentenceGeo) && sentenceGeo.length > 0;

        const geometryAuthority =
            hasSentenceGeometry
                ? 'geometry'
                : fractionsAuthoritative
                    ? 'text'
                    : 'none';

        // ─────────────────────────────────────────────────────────────────
        // Build cumulative offsets
        // ─────────────────────────────────────────────────────────────────
        let totalWeight = 0;
        let cumulative = 0;
        const spanOffsets = new Map();

        for (const cid of allCids) {
            const meta = spanIndex.get(cid);
            const frac = spanFractions.get(cid) || {startFrac: 0, endFrac: 1};

            const fullLen = meta?.len || meta?.cleanedText?.length || 0;
            const owned = Math.max(0, Math.min(1, frac.endFrac - frac.startFrac));

            // Preserve true zeros (prevents cumulative boundary drift).
            // Only guard tiny positive values.
            let effectiveLen = fullLen * owned;
            if (owned > 0 && effectiveLen <= 0) effectiveLen = 1e-6;

            spanOffsets.set(cid, {
                start: cumulative,
                end: cumulative + effectiveLen,
                len: effectiveLen,
                startFrac: frac.startFrac,
                endFrac: frac.endFrac
            });

            cumulative += effectiveLen;
            totalWeight += effectiveLen;
        }

        if (totalWeight <= 0) {
            return [];
        }

        const spokenBudget = totalWeight * progress;
        let allocations = [];


        // ─────────────────────────────────────────────────────────────────
        // Allocate with sentence-local geometry
        // ─────────────────────────────────────────────────────────────────
        for (const cid of allCids) {
            const meta = spanIndex.get(cid);
            if (!meta) continue;
            if (Number(meta.page) !== Number(pageNum)) continue;

            const offset = spanOffsets.get(cid);
            if (!offset || offset.len <= 0) continue;

            // ─────────────────────────────────────────────────────────────
            // ALWAYS USE ORIGINAL PDF GEOMETRY
            //
            // This is the core v4 invariant. We never use computedBbox
            // because it's computed from page-global clustering that can
            // include non-sentence spans, corrupting the geometry.
            // ─────────────────────────────────────────────────────────────
            let fullBbox = null;

            if (Array.isArray(sentenceGeo)) {
                const match = sentenceGeo.find(g => g.cid === cid);
                if (match?.bbox) {
                    fullBbox = match.bbox;
                }
            }

            // ─────────────────────────────────────────────────────────────
            // FALLBACK: Use spanIndex original PDF bbox when sentence
            // geometry is incomplete. This is safe because meta.bbox is
            // original PDF extraction, not computed from clustering.
            // ─────────────────────────────────────────────────────────────
            if (!fullBbox && meta?.bbox) {
                fullBbox = meta.bbox;
            }

            if (!fullBbox) {
                continue;
            }

            if (geometryAuthority === 'geometry') {
                let useBbox = fullBbox;

                // ─────────────────────────────────────────────────────────────────
                // GEOMETRY MODE: Use text fractions for CLIPPING (not allocation)
                //
                // Even though we're not in text-authority mode, the computed
                // fractions tell us which portion of this CID belongs to this
                // sentence. Use them to clip the bbox appropriately.
                // ─────────────────────────────────────────────────────────────────
                const frac = spanFractions.get(cid) || {startFrac: 0, endFrac: 1};
                const fracOwned = frac.endFrac - frac.startFrac;

                // Only apply clipping if we have partial ownership
                if (fracOwned > 0.01 && fracOwned < 0.99) {
                    const bboxHeight = fullBbox[3] - fullBbox[1];
                    const bboxWidth = fullBbox[2] - fullBbox[0];
                    const isMultiLine = bboxHeight > saneMedian * MULTI_LINE_THRESHOLD;

                    if (isMultiLine) {
                        // Multi-line: apply Y-clipping based on fractions
                        const baseLineHeight = Math.max(10, Math.min(28, saneMedian));
                        const maxLines = Math.max(1, Math.min(8, Math.round(bboxHeight / baseLineHeight)));
                        const quantizedLineHeight = bboxHeight / maxLines;

                        const startLine = Math.floor(frac.startFrac * maxLines);
                        const endLine = Math.max(startLine, Math.ceil(frac.endFrac * maxLines) - 1);

                        const y0 = fullBbox[1] + startLine * quantizedLineHeight;
                        const y1 = Math.min(fullBbox[3], fullBbox[1] + (endLine + 1) * quantizedLineHeight);

                        useBbox = [fullBbox[0], y0, fullBbox[2], y1];
                    } else {
                        // Single line: apply X-clipping based on fractions
                        useBbox = [
                            fullBbox[0] + bboxWidth * frac.startFrac,
                            fullBbox[1],
                            fullBbox[0] + bboxWidth * frac.endFrac,
                            fullBbox[3]
                        ];
                    }
                }
                // If fracOwned is ~0 or ~1, use fullBbox as-is (no clipping needed)

                const normalizedBbox = this.normalizeBbox(useBbox);
                if (!normalizedBbox) continue;

                allocations.push({
                    cid,
                    bbox: normalizedBbox,
                    fraction: 1  // Still use 1 for allocation (geometry mode)
                });
                continue;
            }


            if (spokenBudget <= offset.start) {
                continue;
            }

            // ─────────────────────────────────────────────────────────────
            // MULTI-LINE DETECTION
            // ─────────────────────────────────────────────────────────────
            const bboxHeight = fullBbox[3] - fullBbox[1];
            const isMultiLine = bboxHeight > saneMedian * MULTI_LINE_THRESHOLD;


            // ─────────────────────────────────────────────────────────────
            // CLIPPING STRATEGY
            // ─────────────────────────────────────────────────────────────
            let clippedBbox;

            if (isMultiLine) {
                // ─────────────────────────────────────────────────────────
                // TEXT-AUTHORITY: MULTI-LINE Y-CLIPPING + X-CLIPPING
                // ─────────────────────────────────────────────────────────

                // Guardrail: sane line height bounds
                const baseLineHeight = Math.max(10, Math.min(28, saneMedian));

                // Clamp line count to avoid wild swings
                const maxLines = Math.max(1, Math.min(8, Math.round(bboxHeight / baseLineHeight)));

                // Quantized line height derived from bounded line count
                const quantizedLineHeight = bboxHeight / maxLines;
                const lineWidth = fullBbox[2] - fullBbox[0];

                // Convert text fractions to line indices
                const startLine = Math.floor(offset.startFrac * maxLines);
                const endLine = Math.max(startLine, Math.ceil(offset.endFrac * maxLines) - 1);

                // Y boundaries for occupied lines
                let y0 = fullBbox[1] + startLine * quantizedLineHeight;
                let y1 = Math.min(fullBbox[3], fullBbox[1] + (endLine + 1) * quantizedLineHeight);

                // X boundaries - apply horizontal clipping for partial fractions
                let x0 = fullBbox[0];
                let x1 = fullBbox[2];

                if (startLine === endLine) {
                    // Single line occupied: apply X-clipping within that line
                    const lineTextStart = startLine / maxLines;
                    const lineTextEnd = (startLine + 1) / maxLines;
                    const lineTextSpan = lineTextEnd - lineTextStart;

                    if (lineTextSpan > 0) {
                        const withinLineStart = Math.max(0, (offset.startFrac - lineTextStart) / lineTextSpan);
                        const withinLineEnd = Math.min(1, (offset.endFrac - lineTextStart) / lineTextSpan);

                        x0 = fullBbox[0] + lineWidth * withinLineStart;
                        x1 = fullBbox[0] + lineWidth * withinLineEnd;
                    }
                } else if (offset.startFrac > 0.01) {
                    // Multiple lines: X-clip the first line start
                    const lineTextStart = startLine / maxLines;
                    const lineTextEnd = (startLine + 1) / maxLines;
                    const lineTextSpan = lineTextEnd - lineTextStart;

                    if (lineTextSpan > 0) {
                        const withinLineStart = Math.max(0, (offset.startFrac - lineTextStart) / lineTextSpan);
                        x0 = fullBbox[0] + lineWidth * withinLineStart;
                    }
                }

                clippedBbox = [x0, y0, x1, y1];

                console.debug(
                    `[allocateProgressAcrossSpans] Multi-line clip:`,
                    cid,
                    `lines=${startLine}-${endLine}`,
                    `y=${y0.toFixed(1)}-${y1.toFixed(1)}`,
                    `x=${x0.toFixed(1)}-${x1.toFixed(1)}`,
                    `(frac=[${offset.startFrac.toFixed(2)},${offset.endFrac.toFixed(2)}])`
                );

            } else if (offset.startFrac === 0 && offset.endFrac === 1) {
                // ─────────────────────────────────────────────────────────
                // TEXT-AUTHORITY: SINGLE-LINE FULL SPAN
                // ─────────────────────────────────────────────────────────
                clippedBbox = [...fullBbox];

            } else {
                // ─────────────────────────────────────────────────────────
                // TEXT-AUTHORITY: SINGLE-LINE PARTIAL X-CLIPPING
                // ─────────────────────────────────────────────────────────
                const bboxWidth = fullBbox[2] - fullBbox[0];
                clippedBbox = [
                    fullBbox[0] + bboxWidth * offset.startFrac,
                    fullBbox[1],
                    fullBbox[0] + bboxWidth * offset.endFrac,
                    fullBbox[3]
                ];
            }

            const normalizedBbox = this.normalizeBbox(clippedBbox);
            if (!normalizedBbox) continue;

            // Fully spoken span
            if (spokenBudget >= offset.end) {
                allocations.push({
                    cid,
                    bbox: normalizedBbox,
                    fraction: 1
                });
                continue;
            }

            // Partially spoken
            const spokenInSpan = spokenBudget - offset.start;
            const fraction = offset.len > 0 ? spokenInSpan / offset.len : 1;

            allocations.push({
                cid,
                bbox: normalizedBbox,
                fraction: Math.min(1, Math.max(0, fraction))
            });

            break;
        }
        // ─────────────────────────────────────────────────────────────────
        // CONTAINMENT DEDUPLICATION (text-authority mode only)
        //
        // Runs ONLY on allocations actually emitted (post spokenBudget break).
        // This is intentional to preserve progressive highlighting semantics.
        // ─────────────────────────────────────────────────────────────────
        if (fractionsAuthoritative && allocations.length > 1) {
            const dominated = new Set();

            for (let i = 0; i < allocations.length; i++) {
                const a = allocations[i];

                for (let j = 0; j < allocations.length; j++) {
                    if (i === j) continue;

                    const b = allocations[j];

                    // Identify which is larger by area
                    const aArea = a.bbox.width * a.bbox.height;
                    const bArea = b.bbox.width * b.bbox.height;

                    let large, small, largeIdx;
                    if (aArea >= bArea) {
                        large = a;
                        small = b;
                        largeIdx = i;
                    } else {
                        large = b;
                        small = a;
                        largeIdx = j;
                    }

                    // Require same visual line band (strong Y overlap)
                    const largeTop = large.bbox.y;
                    const largeBot = large.bbox.y + large.bbox.height;
                    const smallTop = small.bbox.y;
                    const smallBot = small.bbox.y + small.bbox.height;

                    const yOverlap = Math.max(
                        0,
                        Math.min(largeBot, smallBot) - Math.max(largeTop, smallTop)
                    );

                    const smallHeight = small.bbox.height;

                    // Guard: must be essentially same line, not multi-line paragraph
                    const sameLineBand =
                        smallHeight > 0 &&
                        yOverlap > smallHeight * 0.8 &&
                        large.bbox.height < smallHeight * 3 &&
                        smallHeight < saneMedian * 1.2;

                    if (!sameLineBand) continue;

                    // Strict containment check (with small tolerance)
                    const tol = 2;
                    const contains =
                        large.bbox.x <= small.bbox.x + tol &&
                        large.bbox.y <= small.bbox.y + tol &&
                        (large.bbox.x + large.bbox.width) >=
                        (small.bbox.x + small.bbox.width) - tol &&
                        (large.bbox.y + large.bbox.height) >=
                        (small.bbox.y + small.bbox.height) - tol;

                    if (!contains) continue;
                    // Only suppress if clearly a superset
                    const largeArea = large.bbox.width * large.bbox.height;
                    const smallArea = small.bbox.width * small.bbox.height;
                    if (largeArea > smallArea * 1.5) {
                        dominated.add(largeIdx);
                    }
                }
            }

            if (dominated.size > 0) {
                const filtered = [];
                for (let i = 0; i < allocations.length; i++) {
                    if (!dominated.has(i)) {
                        filtered.push(allocations[i]);
                    }
                }
                allocations = filtered;
            }
        }

        return allocations;
    }


    /**
     * Render clipped highlight boxes for allocated spans.
     *
     * This is a PURE RENDERER with no sentence exclusivity enforcement.
     * It faithfully renders whatever geometry it receives.
     *
     * Contract:
     *   - Caller is responsible for sentence-correct allocations
     *   - Caller must call clearHighlights() before invoking this method
     *   - allocateProgressAcrossSpans() is the expected allocation source
     *
     * Input format:
     *   - cid: Span identifier (for debugging only, not used in rendering)
     *   - bbox: {x0, y0, x1, y1} in PDF coordinate space
     *   - fraction: [0,1] horizontal fill ratio (1 = full width)
     *
     * @param {Array<{ cid: string, bbox: Object, fraction: number }>} spanAllocations
     */
    renderClippedSpanHighlights(spanAllocations) {
        const container = this.elements.highlightContainer;
        if (!container) return;

        for (const entry of spanAllocations) {
            const screen = this.calculateCanvasCoordinates(entry.bbox);
            if (!screen) continue;

            const clippedWidth = screen.width * entry.fraction;
            if (clippedWidth <= 0 || screen.height <= 0) continue;

            const el = document.createElement('div');
            el.className = 'tts-highlight';

            el.style.position = 'absolute';
            el.style.left = `${screen.x}px`;
            el.style.top = `${screen.y}px`;
            el.style.width = `${clippedWidth}px`;
            el.style.height = `${screen.height}px`;
            el.style.backgroundColor = '#FFEB3B';
            el.style.opacity = '0.35';
            el.style.mixBlendMode = 'multiply';
            el.style.borderRadius = '2px';
            el.style.pointerEvents = 'none';

            container.appendChild(el);
        }
    }

    /**
     * Primary progressive highlight entry point.
     *
     * Idempotency:
     *   - Skips rendering if same sentence is already highlighted
     *   - UNLESS _highlightGeometryInvalidated is true (zoom, viewport change)
     *   - clearHighlights() also bypasses idempotency by resetting the key
     *
     * Callers:
     *   - _applyTimelineEvents (PAGE_TURN, SENTENCE_BOUNDARY, GEOMETRY_REFRESH)
     *   - renderPage (pending highlight application)
     *
     * @param {Object} sentence - UISentence selected by timeline engine
     * @param {number} timelineTime - Global audiobook time (seconds)
     */
    highlightSentenceProgress(sentence, timelineTime) {
        if (!sentence || typeof timelineTime !== 'number') {
            return;
        }
        // === HARD GUARD: never highlight during page render ===
        if (this.timeline.pageRenderPending) {
            return;
        }

        const pageNum = this.state?.pdf?.currentPageNum;
        if (typeof pageNum !== 'number') {
            return;
        }

        // Render-safe queueing (preserve existing page-render contract)
        if (this.state?.pdf?.isPageRendering) {
            this.state.pdf.pendingHighlightSentence = sentence;
            this.state.pdf.pendingHighlightTime = timelineTime;
            return;
        }

        this.state.pdf.pendingHighlightSentence = null;
        this.state.pdf.pendingHighlightTime = null;

        // ───────────────────────────────────────────────────────────────
        // IDEMPOTENT: Skip if already highlighting this sentence
        // UNLESS geometry was invalidated (zoom, viewport change, etc.)
        // ───────────────────────────────────────────────────────────────
        const sentenceKey = sentence.global_index;
        const geometryInvalidated = this.state.ui._highlightGeometryInvalidated === true;

        const duration = (sentence.timing && typeof sentence.timing.end === 'number' && typeof sentence.timing.start === 'number')
            ? Math.max(0.001, sentence.timing.end - sentence.timing.start)
            : 0.001;
        const progress = Math.max(0, Math.min(1, (timelineTime - sentence.timing.start) / duration));
        const progressBucket = Math.floor(progress * 80); // ~1.25% buckets prevent over-rendering while preserving motion
        const highlightKey = `${sentenceKey}:${progressBucket}`;

        if (this.state.ui.activeHighlightSentenceKey === highlightKey && !geometryInvalidated) {
            return;
        }

        // New sentence or material progress movement: clear old highlights and update tracking
        this.clearHighlights();
        this.state.ui.activeHighlightSentenceKey = highlightKey;

        const allocations = this.allocateProgressAcrossSpans(
            sentence,
            pageNum,
            progress
        );

        if (allocations.length === 0) {
            return;
        }

        this.renderClippedSpanHighlights(allocations);

        // Clear geometry invalidation marker after successful render
        this.state.ui._highlightGeometryInvalidated = false;
    }

    /**
     * Build span index from semantic.json data.
     * Called once during loadAudiobook().
     *
     * CONTRACT:
     *   - SpanIndex is SEMANTIC/SPAN-CENTRIC, not sentence-centric.
     *   - cleanedText, bbox, and computedBbox may span multiple sentences.
     *   - Span geometry MUST NOT be used for sentence highlighting
     *     unless bounded by authoritative sentence fractions.
     *
     * Index-level markers:
     *   - _authority: 'semantic_span'
     *   - _sentenceSafe: false
     *
     * Entry-level markers:
     *   - _sentenceBounded: false (on each entry)
     *
     * @param {Object} semanticData - Parsed semantic.json
     * @returns {Map<string, Object>} Span-centric metadata index
     */
    buildSpanIndex(semanticData) {
        const spanIndex = new Map();
        // ------------------------------------------------------------
        // SPAN INDEX CONTRACT:
        // Entries are span-centric and NOT sentence-bounded.
        // Any sentence-level geometry usage MUST be constrained by
        // sentence-owned fractions and authority checks.
        // ------------------------------------------------------------
        spanIndex._authority = 'semantic_span';
        spanIndex._sentenceSafe = false;

        const spans = semanticData?.spans || {};

        // ─────────────────────────────────────────────────────────────────
        // PHASE 1: Initial population + group by PAGE
        // ─────────────────────────────────────────────────────────────────
        const pageGroups = new Map(); // page -> [{cid, entry, bbox}]

        for (const [cid, spanData] of Object.entries(spans)) {
            const cleanedText = spanData.cleaned_text || '';
            const lineId = spanData.line_id;
            const spanIdxInLine = spanData.span_index_in_line ?? 0;
            const pageNum = spanData.page_number;
            const bbox = spanData.bbox;

            const entry = {
                len: cleanedText.length,
                cleanedText: cleanedText,
                bbox: bbox,
                page: pageNum,
                lineId: lineId,
                spanIdxInLine: spanIdxInLine,
                computedBbox: null,
                // Explicitly mark as non-sentence geometry
                _sentenceBounded: false
            };

            spanIndex.set(cid, entry);

            // Group by page for visual line clustering
            if (bbox && bbox.length === 4 && pageNum != null) {
                if (!pageGroups.has(pageNum)) {
                    pageGroups.set(pageNum, []);
                }
                pageGroups.get(pageNum).push({
                    cid,
                    entry,
                    y0: bbox[1],
                    y1: bbox[3],
                    x0: bbox[0],
                    x1: bbox[2],
                    height: bbox[3] - bbox[1]
                });
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // PHASE 2: Visual line clustering via Y-overlap
        //
        // INVARIANT: Two spans belong to the same visual line if their
        // vertical ranges overlap by at least 50% of the smaller height.
        //
        // Algorithm:
        //   1. Sort spans by y0 (top edge)
        //   2. Sweep top → bottom
        //   3. Extend current cluster while Y-overlap condition holds
        //   4. Start new cluster when no overlap
        // ─────────────────────────────────────────────────────────────────
        const OVERLAP_THRESHOLD = 0.5; // 50% of smaller span height

        let totalClusters = 0;

        for (const [pageNum, pageSpans] of pageGroups.entries()) {
            // Sort by y0 (top edge), then x0 for stability
            pageSpans.sort((a, b) => {
                const yDiff = a.y0 - b.y0;
                return Math.abs(yDiff) > 0.5 ? yDiff : a.x0 - b.x0;
            });

            // Cluster spans into visual lines
            const clusters = [];
            let currentCluster = null;

            for (const span of pageSpans) {
                if (!currentCluster) {
                    // Start first cluster
                    currentCluster = {
                        spans: [span],
                        y0: span.y0,
                        y1: span.y1
                    };
                    continue;
                }

                // Check Y-overlap with current cluster
                const overlapTop = Math.max(currentCluster.y0, span.y0);
                const overlapBottom = Math.min(currentCluster.y1, span.y1);
                const overlapHeight = overlapBottom - overlapTop;

                const minHeight = Math.min(
                    currentCluster.y1 - currentCluster.y0,
                    span.height
                );

                const hasOverlap = minHeight > 0 &&
                    (overlapHeight / minHeight) >= OVERLAP_THRESHOLD;

                if (hasOverlap) {
                    // Extend current cluster
                    currentCluster.spans.push(span);
                    // Expand cluster bounds
                    currentCluster.y0 = Math.min(currentCluster.y0, span.y0);
                    currentCluster.y1 = Math.max(currentCluster.y1, span.y1);
                } else {
                    // Finalize current cluster, start new one
                    clusters.push(currentCluster);
                    currentCluster = {
                        spans: [span],
                        y0: span.y0,
                        y1: span.y1
                    };
                }
            }

            // Don't forget last cluster
            if (currentCluster) {
                clusters.push(currentCluster);
            }

            totalClusters += clusters.length;

            // ─────────────────────────────────────────────────────────────
            // PHASE 3: Horizontal distribution within each cluster
            // ─────────────────────────────────────────────────────────────
            for (const cluster of clusters) {
                const clusterSpans = cluster.spans;

                if (clusterSpans.length === 1) {
                    // Single span - use original bbox
                    const span = clusterSpans[0];
                    span.entry.computedBbox = [...span.entry.bbox];
                    continue;
                }

                // Sort by X position (left to right)
                clusterSpans.sort((a, b) => a.x0 - b.x0);

                // Get line bounds
                const lineX0 = clusterSpans[0].x0;
                const lineX1 = clusterSpans[clusterSpans.length - 1].x1;
                const lineY0 = cluster.y0;
                const lineY1 = cluster.y1;
                const lineWidth = lineX1 - lineX0;

                if (lineWidth <= 0) {
                    // Fallback: use original bboxes
                    for (const span of clusterSpans) {
                        span.entry.computedBbox = [...span.entry.bbox];
                    }
                    continue;
                }

                const totalChars = clusterSpans.reduce((sum, s) => sum + s.entry.len, 0);

                if (totalChars === 0) {
                    for (const span of clusterSpans) {
                        span.entry.computedBbox = [...span.entry.bbox];
                    }
                    continue;
                }

                // Distribute width proportionally by character count
                const pxPerChar = lineWidth / totalChars;
                let cursorX = lineX0;

                for (const span of clusterSpans) {
                    const spanWidth = span.entry.len * pxPerChar;

                    // Keep original Y bounds, only redistribute X
                    span.entry.computedBbox = [
                        cursorX,
                        span.y0,              // Original span Y
                        cursorX + spanWidth,
                        span.y1               // Original span Y
                    ];

                    cursorX += spanWidth;
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // PHASE 4: Fallback for any ungrouped spans
        // ─────────────────────────────────────────────────────────────────
        for (const [cid, entry] of spanIndex.entries()) {
            if (!entry.computedBbox && entry.bbox) {
                entry.computedBbox = [...entry.bbox];
            }
        }

        console.log(`[buildSpanIndex] Indexed ${spanIndex.size} spans into ${totalClusters} visual line clusters`);
        return spanIndex;
    }

    /******************************************************************************
     * SECTION F0: PDF Coordinate Utilities
     ******************************************************************************/

    /**
     * === Geometry Normalizer (Backend Contract Adapter) ===
     * Backend emits bbox as [x0,y0,x1,y1]. Frontend geometry expects {x,y,width,height}.
     * @param {number[]|NormalizedBbox|null} bbox - Bounding box in either format
     * @returns {NormalizedBbox|null} Normalized {x, y, width, height} or null if invalid
     */
    normalizeBbox(bbox) {
        if (!bbox) return null;
        // Array form: [x0, y0, x1, y1]
        if (Array.isArray(bbox) && bbox.length === 4) {
            const x0 = bbox[0], y0 = bbox[1], x1 = bbox[2], y1 = bbox[3];
            const width = x1 - x0;
            const height = y1 - y0;

            // Guard against malformed bbox (negative dimensions)
            if (width <= 0 || height <= 0) {
                console.warn('[normalizeBbox] Invalid bbox dimensions:', bbox);
                return null;
            }

            return {
                x: x0,
                y: y0,
                width: width,
                height: height
            };
        }
        // Object form: {x,y,width,height}
        if (typeof bbox === 'object' && bbox.x != null && bbox.y != null) {
            // Also validate object form
            if (bbox.width <= 0 || bbox.height <= 0) {
                console.warn('[normalizeBbox] Invalid bbox dimensions:', bbox);
                return null;
            }
            return bbox;
        }
        return null;
    }

    calculateCanvasCoordinates(pdfCoords) {
        const viewport = this.state.pdf.viewport;

        if (!viewport) {
            return null;
        }

        // Bypass the full pdfjsLib.Util.applyTransform
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

    screenToPdfCoordinates(screenX, screenY) {
        const viewport = this.state.pdf.viewport;
        if (!viewport) return null;

        const scale = viewport.scale;

        return {
            x: screenX / scale,
            y: screenY / scale
        };
    }

    /******************************************************************************
     * SECTION G: SPATIAL INTERACTION & HIT-TESTING
     *******************************************************************************/

    /**
     * Click-to-seek handler for PDF canvas.
     *
     * TIMEBASE CONVERSION:
     *   - sentence.timing uses GLOBAL audiobook time
     *   - seekTo() expects CHUNK-LOCAL time
     *   - Conversion: localTime = globalTime - chunk.start_time
     *
     * AUTHORITY REQUIREMENT:
     *   CID-based proportional seek (seeking into sentence based on
     *   clicked CID position) is DISABLED unless semantic projection
     *   authority exists. Without authority, seek defaults to sentence start.
     *
     * KNOWN LIMITATION (shared CIDs):
     *   When a click lands on a CID shared by multiple sentences,
     *   findSentenceAtCoordinates() returns the first match.
     *   Future enhancement: use timing context for disambiguation.
     */
    async handlePdfClick(event) {
        // ================================================================
        // CLICK-SEEK HANDLER
        // Uses UI JSON sentence and uiIndex exclusively.
        // TIMEBASE: sentence.timing is GLOBAL; seekTo expects CHUNK-LOCAL.
        // ================================================================

        // === Hard gate: click-seek enabled and UI ready ===
        if (!this.state.pdf.clickSeekEnabled || !this.state.audiobook.uiReady) {
            return;
        }

        const canvas = this.elements.pdfCanvas;
        const rect = canvas.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const clickY = event.clientY - rect.top;

        const pdfCoords = this.screenToPdfCoordinates(clickX, clickY);
        if (!pdfCoords) return;

        // === Find clicked sentence via geometry hit-test ===
        const result = this.findSentenceAtCoordinates(pdfCoords.x, pdfCoords.y);

        if (!result || !result.sentence) {
            return;
        }

        const {sentence, chunkId, clickedCid} = result;

        // === Contract markers ===
        const isUiSentence = (sentence && sentence._uiContract === 'ui_sentence');
        const hasSemanticProjection = !!(sentence && sentence._hasSemanticProjection);

        // === Calculate GLOBAL seek time ===
        // Default: seek to sentence start (global timeline)
        let globalSeekTime = sentence.timing.start;

        // CID-based proportional seek within sentence
        // GATED: Only safe when semantic projection authority exists
        if (
            clickedCid &&
            Array.isArray(sentence.cids) &&
            sentence.cids.length > 0 &&
            hasSemanticProjection
        ) {
            const cidIndex = sentence.cids.indexOf(clickedCid);

            if (cidIndex !== -1) {
                const duration = sentence.timing.end - sentence.timing.start;

                if (duration > 0) {
                    const ratio = Math.min(
                        Math.max(cidIndex / sentence.cids.length, 0),
                        0.999
                    );
                    globalSeekTime = sentence.timing.start + (ratio * duration);
                }
            }
        } else if (clickedCid && !hasSemanticProjection) {
            // Diagnostic: proportional seek disabled due to missing authority
            console.debug(
                `[Click-Seek] CID-based proportional seek disabled for sentence ${sentence.global_index}: ` +
                `no semantic projection authority. Falling back to sentence start.`
            );
        }

        // === Get target chunk for timebase conversion ===
        const targetChunk = this.state.audiobook.chunks.find(c => c.chunk_id === chunkId);
        if (!targetChunk) {
            return;
        }
        const chunkStartTime = targetChunk.start_time ?? 0;

        // === Convert GLOBAL → CHUNK-LOCAL for seekTo ===
        const localSeekTime = globalSeekTime - chunkStartTime;

        // === Reset page turn cursor (uses GLOBAL time) ===
        this._resetPageTurnIndexForTime(globalSeekTime);

        // === Determine if chunk switch is needed ===
        const currentChunk = this.state.audiobook.chunks?.[this.state.audiobook.currentChunkIndex];
        const needsChunkSwitch = !currentChunk || chunkId !== currentChunk.chunk_id;

        // === Execute seek ===
        if (needsChunkSwitch) {
            const newIndex = this.state.audiobook.chunks.findIndex(c => c.chunk_id === chunkId);

            if (newIndex !== -1) {
                console.log(`[Click-Seek] Switching to chunk ${newIndex} (id=${chunkId}), local=${localSeekTime.toFixed(2)}s, global=${globalSeekTime.toFixed(2)}s`);
                await this.playChunk(newIndex);
                this.seekTo(localSeekTime);
            }
        } else {
            console.log(`[Click-Seek] Within-chunk seek, local=${localSeekTime.toFixed(2)}s, global=${globalSeekTime.toFixed(2)}s`);
            this.seekTo(localSeekTime);
        }

        // Resume playback if paused
        if (!this.state.audio.isPlaying) {
            await this.play();
        }
    }

    findSentenceAtCoordinates(pdfX, pdfY) {
        // ================================================================
        // CLICK-SEEK HIT TEST
        // Uses uiIndex.geometryByPage for spatial lookup.
        // Returns UI JSON sentence + clicked CID for interpolation.
        // ================================================================

        // === Hard gate: UI JSON must be ready ===
        if (!this.uiIndex || !this.state.audiobook.uiReady) {
            return null;
        }

        const currentPage = this.state.pdf.currentPageNum;
        if (typeof currentPage !== 'number') {
            return null;
        }

        // === Get flattened geometry for current page ===
        const pageGeometry = this.uiIndex.geometryByPage[currentPage];
        if (!Array.isArray(pageGeometry) || pageGeometry.length === 0) {
            return null;
        }

        // === Hit-test: find geometry entry containing click point ===
        let hitEntry = null;
        for (const entry of pageGeometry) {
            const bbox = this.normalizeBbox(entry.bbox);
            if (!bbox) continue;

            if (
                pdfX >= bbox.x &&
                pdfX <= bbox.x + bbox.width &&
                pdfY >= bbox.y &&
                pdfY <= bbox.y + bbox.height
            ) {
                hitEntry = entry;
                break;
            }
        }

        if (!hitEntry) {
            return null;
        }

        // === Look up the UI JSON sentence ===
        const sentence = this.uiIndex.byGlobalIndex.get(hitEntry.globalIndex);
        if (!sentence) {
            return null;
        }

        // === Return click-seek result ===
        return {
            sentence: sentence,          // UI JSON sentence object
            chunkId: hitEntry.chunkId,   // For chunk-switch detection
            clickedCid: hitEntry.cid,    // For within-sentence interpolation
            page: currentPage            // Clicked page
        };
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
                this.elements.clickSeekToggle.textContent = 'Seek: ON';
                // Update cursor for PDF canvas
                if (this.elements.pdfCanvas) {
                    this.elements.pdfCanvas.style.cursor = 'pointer';
                    this.elements.pdfCanvas.classList.add('seek-enabled');
                }
            } else {
                this.elements.clickSeekToggle.classList.remove('active');
                this.elements.clickSeekToggle.textContent = 'Seek: OFF';
                // Reset cursor
                if (this.elements.pdfCanvas) {
                    this.elements.pdfCanvas.style.cursor = 'default';
                    this.elements.pdfCanvas.classList.remove('seek-enabled');
                }
            }
        }

        console.log(`Click-to-seek mode: ${this.state.pdf.clickSeekEnabled ? 'ON' : 'OFF'}`);
    }

    /******************************************************************************
     * SECTION H: FILE & SOURCE MANAGEMENT
     *******************************************************************************/

    /**
     * Load available audio sources from backend.
     */
    async loadAudioSources() {
        let response;

        try {
            response = await fetch('/api/audio_sources');
        } catch (error) {
            this.logError('Failed to fetch audio sources: ' + error.message);
            throw error;
        }

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        try {
            /** @type {AudioSourcesResponse} */
            const data = await response.json();
            this.elements.sourceSelect.innerHTML = '';

            if (data.sources && data.sources.length > 0) {
                const sourceLabels = {
                    audiobooks: 'Audiobooks',
                    obsidian: 'Obsidian Notes',
                    standalone: 'Standalone Files'
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
                this.elements.sourceSelect.innerHTML =
                    '<option value="">No sources found</option>';
            }
        } catch (error) {
            this.logError('Failed to parse audio sources: ' + error.message);
            throw error;
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
        const source = this.state.ui.fileListSource;
        let response;

        this.clearError();
        this.elements.fileList.innerHTML = '<p>Loading files...</p>';

        let apiUrl = `/api/list_audio?source=${source}`;
        if (source === 'audiobooks') {
            apiUrl = '/api/audiobooks';
        }

        // ─────────────────────────────────────────────
        // STEP 1: Fetch (network boundary)
        // ─────────────────────────────────────────────
        try {
            response = await fetch(apiUrl);
        } catch (error) {
            this.logError('Failed to fetch file list: ' + error.message);
            throw error; // propagate to container
        }

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        // ─────────────────────────────────────────────
        // STEP 2: Parse + render (data boundary)
        // ─────────────────────────────────────────────
        try {
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

                    div.onclick = async () => {
                        document
                            .querySelectorAll('.file-item')
                            .forEach(el => el.classList.remove('active'));

                        div.classList.add('active');

                        if (file.isAudiobook) {
                            await this.loadAudiobook(file.id);
                        } else {
                            await this.loadFile(file.id, source);
                        }
                    };

                    this.elements.fileList.appendChild(div);
                });
            } else {
                this.elements.fileList.innerHTML =
                    '<p>No files found in this source.</p>';
            }

            const sourceLabels = {
                audiobooks: 'Audiobooks',
                obsidian: 'Obsidian Notes',
                standalone: 'Standalone Files'
            };

            this.elements.currentSourceLabel.textContent =
                sourceLabels[source] || source;

        } catch (error) {
            this.logError('Failed to process file list: ' + error.message);
            throw error;
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

            const safeFilename = this.sanitizeFilename(filename);
            let url = `/api/audio/${safeFilename}?source=${encodeURIComponent(source)}`;

            if (source === 'audiobooks') {
                const bookId = this.state.audiobook.bookId;
                if (!bookId) {
                    throw new Error('Audiobook chunk loads require an active book_id.');
                }
                url = `/api/audio/${safeFilename}?source=audiobooks&book_id=${encodeURIComponent(bookId)}`;
            }

            this.state.audio.currentSource = source;
            this.state.ui.selectedFile = filename;
            this.state.audiobook.mode = 'standalone';

            this.elements.currentFileDisplay.textContent = filename;
            this.elements.playPauseButton.disabled = true;
            this.state.audiobook.lastLoadedAudioUrl = url;
            this.state.audiobook.lastLoadedAudioFilename = safeFilename;

            await this.state.audio.backend.load(url);
        } catch (error) {
            this.logError('Failed to load file: ' + error.message);
        }
    }

    /******************************************************************************
     * SECTION I: UI STATUS, ERRORS & BANNERS
     *******************************************************************************/

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

    updateStatusBanner({allowPlayer = false} = {}) {
        const banner = this.elements.statusContainer;
        const player = this.elements.playerContainer;

        if (!banner || !player) {
            console.warn('[updateStatusBanner] Missing banner or player container');
            return;
        }

        const {
            processingStatus,
            errorMessage,
            isStale,
            progress_percentage
        } = this.state.audiobook;

        const ui = this.mapStatusToUI(processingStatus, {
            is_stale: isStale,
            progress_percentage,
            error_message: errorMessage
        });

        // Show banner; hide player only for non-playable states.
        banner.style.display = 'block';
        player.style.display = allowPlayer ? 'block' : 'none';

        // Clear existing banner content
        banner.innerHTML = '';

        const msg = document.createElement('div');
        msg.className = `status-message ${ui.severity}`;
        msg.textContent = ui.message;
        banner.appendChild(msg);

        const details = this.buildStatusDetails();
        if (details) {
            const detailNode = document.createElement('div');
            detailNode.className = 'status-detail';
            detailNode.textContent = details;
            banner.appendChild(detailNode);
        }

        if (ui.showRetry) {
            const retryBtn = document.createElement('button');
            retryBtn.textContent = 'Retry';
            retryBtn.onclick = () => {
                this.retryProcessing(this.state.audiobook.bookId, false);
            };
            banner.appendChild(retryBtn);
        }
    }

    mapStatusToUI(status, options = {}) {
        const {
            is_stale = false,
            progress_percentage = 0,
            error_message = null
        } = options;

        // Explicit error always wins
        if (error_message) {
            return {
                mode: 'error',
                severity: 'error',
                message: error_message,
                showRetry: true
            };
        }

        switch (status) {
            case 'queued':
            case 'running':
            case 'processing_started':
            case 'stage_1_extracting':
            case 'stage_1_complete':
            case 'stage_2_semantic':
            case 'stage_2_complete':
            case 'stage_25_ui':
            case 'stage_3_audio':
            case 'stage_3_started':
                return {
                    mode: 'waiting',
                    severity: is_stale ? 'warning' : 'info',
                    message: is_stale
                        ? `Processing stalled (${progress_percentage}%)`
                        : `Processing… (${progress_percentage}%)`,
                    showRetry: true
                };

            case 'stage_3_partial':
            case 'degraded':
                return {
                    mode: 'partial',
                    severity: is_stale ? 'warning' : 'info',
                    message: is_stale
                        ? `Partial audio available (stalled)`
                        : `Partial audio available`,
                    showRetry: true
                };

            case 'completed':
            case 'stage_3_complete':
                return {
                    mode: 'ready',
                    severity: 'info',
                    message: 'Audiobook ready',
                    showRetry: false
                };

            case 'failed':
                return {
                    mode: 'error',
                    severity: 'error',
                    message: 'Processing failed',
                    showRetry: true
                };

            case 'cancelled':
                return {
                    mode: 'cancelled',
                    severity: 'warning',
                    message: 'Processing cancelled',
                    showRetry: true
                };

            default:
                return {
                    mode: 'unknown',
                    severity: 'warning',
                    message: 'Unknown processing state',
                    showRetry: true
                };
        }
    }

    buildStatusDetails() {
        const manifest = this.state.audiobook.manifest || {};
        const ready = Array.isArray(manifest.ready_chunks) ? manifest.ready_chunks.length : this.state.audiobook.readyChunks.length;
        const total = manifest.total_chunks || this.state.audiobook.totalChunks || 0;
        const parts = [];

        if (total) parts.push(`${ready}/${total} chunks`);
        const progress = this.state.audiobook.progress_percentage ?? manifest.progress_percentage;
        if (typeof progress === 'number') parts.push(`${Math.round(progress)}%`);
        if (manifest.job_status) parts.push(`job: ${manifest.job_status}`);
        if (manifest.job_stage) parts.push(`stage: ${manifest.job_stage}`);
        if (manifest.ui_shards_ready || manifest.semantic_shards_ready) {
            parts.push(`shards: ui=${manifest.ui_shards_ready ? 'ready' : 'missing'}, semantic=${manifest.semantic_shards_ready ? 'ready' : 'missing'}`);
        }
        if (this.state.audiobook.traceId) parts.push(`trace: ${this.state.audiobook.traceId}`);
        return parts.join(' · ');
    }

    downloadCurrentAudio() {
        const url = this.state.audiobook.lastLoadedAudioUrl;
        const filename = this.state.audiobook.lastLoadedAudioFilename || this.state.ui.selectedFile;
        if (!url || !filename) {
            this.logError('No loaded audio is available to download.');
            return;
        }

        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.rel = 'noopener';
        document.body.appendChild(link);
        link.click();
        link.remove();
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
            const response = await fetch(`/api/v1/audiobooks/${bookId}/retry?force_rebuild=${fullRebuild}`, {
                method: 'POST'
            });

            if (!response.ok) {
                const errorData = await response.json();
                const msg = errorData.detail || `Retry failed: ${response.status}`;
                console.error(msg);   // container-visible
                throw new Error(msg); // abort retry flow
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

    /******************************************************************************
     * SECTION I2: Playback Display Helpers
     *******************************************************************************/

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
            this.elements.seekSlider.value = (this.state.audio.currentTime / this.state.audio.duration) * 100;
        } else {
            this.elements.seekSlider.value = 0;
        }
    }

    /******************************************************************************
     * SECTION J: General Utilities
     *******************************************************************************/

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
     * Sanitize filename for API calls.
     * Removes special characters except word chars, hyphens, and dots.
     * @param {string} filename - Raw filename
     * @returns {string} Sanitized filename
     */
    sanitizeFilename(filename) {
        return filename.replace(/[^\w\-.]/g, '').trim();
    }
}

/**
 * Wait for the DOM to be fully loaded before initializing the player.
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('=== TTSAudioPlayer Initialization ===');

    // Create global player instance
    window.player = new TTSAudioPlayer();

    try {
        // Initialize player with wavesurfer backend
        await window.player.init(null, {backend: 'native'});
        console.log('=== TTSAudioPlayer Ready ===');
    } catch (error) {
        console.error('=== TTSAudioPlayer Initialization Failed ===');
        console.error(error);
        alert('Failed to initialize audio player. Check console for details.');
    }
});