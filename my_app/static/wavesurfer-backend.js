/**
 * WAVESURFER AUDIO BACKEND
 *
 * Wraps Wavesurfer.js library with AudioBackend contract.
 * Provides high-precision timing (60Hz) and waveform visualization.
 *
 * Features:
 *   - 60Hz audioprocess events via requestAnimationFrame
 *   - Waveform visualization
 *   - Precise seeking
 *   - Better browser compatibility
 *
 * Use Cases:
 *   - Phase 2+ (PDF sync with highlighting)
 *   - Primary backend for production
 */
class WavesurferBackend extends AudioBackend {
  constructor() {
    super();

    // Wavesurfer instance
    this.wavesurfer = null;

    // Container element reference
    this.container = null;

    // 60Hz timer for precise audioprocess events
    this._animationFrameId = null;
    this._lastEmittedTime = 0;

    // Track if we're currently playing (for audioprocess emission)
    this._isPlayingForTimer = false;
  }

  /**
   * Initialize with waveform container
   * @param {HTMLElement} container - Parent container (not used, we query by ID)
   * @param {object} options - Configuration
   * @param {string} options.waveformContainerId - ID of waveform div (default: 'waveform')
   * @param {string} options.audioElementId - Not used (Wavesurfer creates its own)
   */
  async init(container, options = {}) {
    const waveformContainerId = options.waveformContainerId || 'waveform';
    this.container = document.getElementById(waveformContainerId);

    if (!this.container) {
      throw new Error(`WavesurferBackend: Waveform container #${waveformContainerId} not found`);
    }

    // Wait for Wavesurfer library to load (with timeout)
    console.log('Waiting for WaveSurfer library...');
    let attempts = 0;
    while (typeof WaveSurfer === 'undefined' && attempts < 50) {
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;
    }

    if (typeof WaveSurfer === 'undefined') {
      throw new Error('WavesurferBackend: WaveSurfer library failed to load after 5 seconds. Check network connection.');
    }

    console.log('✓ WaveSurfer library loaded');

    // Create Wavesurfer instance
    this.wavesurfer = WaveSurfer.create({
      container: this.container,
      waveColor: '#D06C9B',
      progressColor: '#691D40',
      cursorColor: '#ffffff',
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      height: 128,
      normalize: true,
      backend: 'WebAudio', // Use Web Audio API backend for best performance
    });

    // Bind event listeners
    this._bindWavesurferEvents();

    this._isInitialized = true;
    console.log('✓ WavesurferBackend initialized');
  }

  /**
   * Load audio file
   * @param {string} url - Audio file URL
   */
  async load(url) {
    if (!this._isInitialized) {
      throw new Error('WavesurferBackend: init() must be called before load()');
    }

    this._isLoading = true;
    this.emit(AudioBackend.EVENTS.LOADING);

    try {
      await this.wavesurfer.load(url);
      // READY event will fire from wavesurfer 'ready' handler
    } catch (error) {
      this.emit(AudioBackend.EVENTS.ERROR, {
        error: new Error('Failed to load audio: ' + error.message)
      });
      throw error;
    }
  }

  /**
   * Start playback
   */
  async play() {
    if (!this._isInitialized) {
      throw new Error('WavesurferBackend: init() must be called before play()');
    }

    try {
      await this.wavesurfer.play();
      // PLAY event will fire from wavesurfer 'play' handler
    } catch (error) {
      this.emit(AudioBackend.EVENTS.ERROR, {
        error: new Error('Playback failed: ' + error.message)
      });
      throw error;
    }
  }

  /**
   * Pause playback
   */
  pause() {
    if (!this._isInitialized) {
      throw new Error('WavesurferBackend: init() must be called before pause()');
    }

    this.wavesurfer.pause();
    // PAUSE event will fire from wavesurfer 'pause' handler
  }

  /**
   * Seek to time
   * @param {number} seconds - Target time
   */
  setTime(seconds) {
    if (!this._isInitialized) {
      throw new Error('WavesurferBackend: init() must be called before setTime()');
    }

    this.wavesurfer.setTime(seconds);
    // SEEKING event will fire from our manual emission
    this.emit(AudioBackend.EVENTS.SEEKING, {
      currentTime: seconds
    });
  }

  /**
   * Set playback speed
   * @param {number} rate - Speed (0.5 - 2.0)
   */
  setPlaybackRate(rate) {
    if (!this._isInitialized) {
      throw new Error('WavesurferBackend: init() must be called before setPlaybackRate()');
    }

    const clampedRate = Math.max(0.5, Math.min(rate, 2.0));

    // --- BEGIN FIX ---
    // Get the underlying HTML5 <audio> element that Wavesurfer is using
    const mediaElement = this.wavesurfer.getMediaElement();
    if (mediaElement) {
        // This is the magic property that fixes the pitch
        mediaElement.preservesPitch = true;
    }
    // --- END FIX ---

    // Now, set the rate
    this.wavesurfer.setPlaybackRate(clampedRate);
  }

  /**
   * Set volume
   * @param {number} volume - Volume (0.0 - 1.0)
   */
  setVolume(volume) {
    if (!this._isInitialized) {
      throw new Error('WavesurferBackend: init() must be called before setVolume()');
    }

    const clampedVolume = Math.max(0.0, Math.min(volume, 1.0));
    this.wavesurfer.setVolume(clampedVolume);
  }

  /**
   * Set loop state
   * @param {boolean} isLooping - True to loop, false to not
   */
  setLoop(isLooping) {
    if (!this._isInitialized) {
      throw new Error('WavesurferBackend: init() must be called before setLoop()');
    }

    // Wavesurfer doesn't have built-in loop, we handle it in finish event
    this._shouldLoop = isLooping;
  }

  /**
   * Get current time
   * @returns {number} Current time in seconds
   */
  getCurrentTime() {
    return this.wavesurfer ? this.wavesurfer.getCurrentTime() : 0;
  }

  /**
   * Get duration
   * @returns {number} Duration in seconds
   */
  getDuration() {
    return this.wavesurfer ? this.wavesurfer.getDuration() : 0;
  }

  /**
   * Cleanup
   */
  destroy() {
    if (!this.wavesurfer) return;

    // Stop 60Hz timer
    this._stop60HzTimer();

    // Destroy wavesurfer instance
    this.wavesurfer.destroy();
    this.wavesurfer = null;

    // Clear all event subscribers
    this.removeAllListeners();

    this._isInitialized = false;
    console.log('✓ WavesurferBackend destroyed');
  }

  /**
   * INTERNAL: Bind Wavesurfer event listeners
   * @private
   */
  _bindWavesurferEvents() {
    // READY: Audio loaded and decoded
    this.wavesurfer.on('ready', () => {
      this._isLoading = false;
      this.emit(AudioBackend.EVENTS.READY, {
        duration: this.getDuration()
      });
    });

    // PLAY: Playback started
    this.wavesurfer.on('play', () => {
      this._isPlaying = true;
      this._isPlayingForTimer = true;
      this.emit(AudioBackend.EVENTS.PLAY);

      // Start 60Hz timer
      this._start60HzTimer();
    });

    // PAUSE: Playback paused
    this.wavesurfer.on('pause', () => {
      this._isPlaying = false;
      this._isPlayingForTimer = false;
      this.emit(AudioBackend.EVENTS.PAUSE);

      // Stop 60Hz timer
      this._stop60HzTimer();
    });

    // FINISH: Playback ended
    this.wavesurfer.on('finish', () => {
      this._isPlaying = false;
      this._isPlayingForTimer = false;

      // Stop 60Hz timer
      this._stop60HzTimer();

      // Handle loop
      if (this._shouldLoop) {
        this.wavesurfer.setTime(0);
        this.play();
      } else {
        this.emit(AudioBackend.EVENTS.FINISH);
      }
    });

    // TIMEUPDATE: Wavesurfer provides this, but we supplement with 60Hz
    this.wavesurfer.on('timeupdate', (currentTime) => {
      this.emit(AudioBackend.EVENTS.TIMEUPDATE, { currentTime });
    });

    // SEEKING: Wavesurfer provides 'seeking' event
    this.wavesurfer.on('seeking', (currentTime) => {
      this.emit(AudioBackend.EVENTS.SEEKING, { currentTime });
    });

    // ERROR: Wavesurfer error handling
    this.wavesurfer.on('error', (error) => {
      this.emit(AudioBackend.EVENTS.ERROR, {
        error: new Error('Wavesurfer error: ' + error)
      });
    });
  }

  /**
   * INTERNAL: Start 60Hz precision timer
   * Emits AUDIOPROCESS event at ~60Hz for smooth highlighting
   * @private
   */
  _start60HzTimer() {
    // Don't start if already running
    if (this._animationFrameId !== null) return;

    const emit60HzEvent = () => {
      if (!this._isPlayingForTimer) {
        this._stop60HzTimer();
        return;
      }

      const currentTime = this.getCurrentTime();

      // Emit audioprocess event
      this.emit(AudioBackend.EVENTS.AUDIOPROCESS, { currentTime });

      this._lastEmittedTime = currentTime;

      // Schedule next frame
      this._animationFrameId = requestAnimationFrame(emit60HzEvent);
    };

    // Start loop
    this._animationFrameId = requestAnimationFrame(emit60HzEvent);
  }

  /**
   * INTERNAL: Stop 60Hz precision timer
   * @private
   */
  _stop60HzTimer() {
    if (this._animationFrameId !== null) {
      cancelAnimationFrame(this._animationFrameId);
      this._animationFrameId = null;
    }
  }
}