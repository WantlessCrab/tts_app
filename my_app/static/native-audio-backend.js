/**
 * NATIVE AUDIO BACKEND
 *
 * Wraps HTML5 <audio> element with AudioBackend contract.
 * Uses standard browser audio APIs.
 *
 * Limitations:
 *   - Low timing precision (~250ms between timeupdate events)
 *   - No waveform visualization
 *   - Limited seeking precision
 *
 * Use Cases:
 *   - Phase 0-3 (before wavesurfer integration)
 *   - Fallback if wavesurfer fails to load
 */
class NativeAudioBackend extends AudioBackend {
  constructor() {
    super();

    // HTML5 audio element reference
    this.audioElement = null;

    // Bound event handlers (for cleanup)
    this._boundHandlers = {
      onLoadStart: null,
      onCanPlayThrough: null,
      onPlay: null,
      onPause: null,
      onEnded: null,
      onTimeUpdate: null,
      onSeeking: null,
      onError: null
    };
  }

  /**
   * Initialize with existing audio element
   * @param {HTMLElement} container - Not used (audio element must already exist)
   * @param {object} options - Configuration
   * @param {string} options.audioElementId - ID of <audio> element (default: 'player')
   */
  async init(container, options = {}) {
    const audioElementId = options.audioElementId || 'player';
    this.audioElement = document.getElementById(audioElementId);

    if (!this.audioElement) {
      throw new Error(`NativeAudioBackend: Audio element #${audioElementId} not found`);
    }

    // Bind event handlers
    this._boundHandlers.onLoadStart = () => this._onLoadStart();
    this._boundHandlers.onCanPlayThrough = () => this._onCanPlayThrough();
    this._boundHandlers.onPlay = () => this._onPlay();
    this._boundHandlers.onPause = () => this._onPause();
    this._boundHandlers.onEnded = () => this._onEnded();
    this._boundHandlers.onTimeUpdate = () => this._onTimeUpdate();
    this._boundHandlers.onSeeking = () => this._onSeeking();
    this._boundHandlers.onError = (e) => this._onError(e);

    // Attach event listeners
    this.audioElement.addEventListener('loadstart', this._boundHandlers.onLoadStart);
    this.audioElement.addEventListener('canplaythrough', this._boundHandlers.onCanPlayThrough);
    this.audioElement.addEventListener('play', this._boundHandlers.onPlay);
    this.audioElement.addEventListener('pause', this._boundHandlers.onPause);
    this.audioElement.addEventListener('ended', this._boundHandlers.onEnded);
    this.audioElement.addEventListener('timeupdate', this._boundHandlers.onTimeUpdate);
    this.audioElement.addEventListener('seeking', this._boundHandlers.onSeeking);
    this.audioElement.addEventListener('error', this._boundHandlers.onError);

    this._isInitialized = true;
  }

  /**
   * Load audio file
   * @param {string} url - Audio file URL
   */
  async load(url) {
    if (!this._isInitialized) {
      throw new Error('NativeAudioBackend: init() must be called before load()');
    }

    this._isLoading = true;

    // Set source and trigger load
    this.audioElement.src = url;
    this.audioElement.load();

    // Note: 'loading' event will be emitted by onLoadStart handler
  }

  /**
   * Start playback
   * Returns promise to handle autoplay policy
   */
  async play() {
    if (!this._isInitialized) {
      throw new Error('NativeAudioBackend: init() must be called before play()');
    }

    try {
      // play() returns a promise (autoplay policy)
      await this.audioElement.play();
    } catch (error) {
      // Autoplay blocked by browser
      this.emit(AudioBackend.EVENTS.ERROR, {
        error: new Error('Autoplay blocked by browser: ' + error.message)
      });
      throw error;
    }
  }

  /**
   * Pause playback
   */
  pause() {
    if (!this._isInitialized) {
      throw new Error('NativeAudioBackend: init() must be called before pause()');
    }

    this.audioElement.pause();
  }

  /**
   * Seek to time
   * @param {number} seconds - Target time
   */
  setTime(seconds) {
    if (!this._isInitialized) {
      throw new Error('NativeAudioBackend: init() must be called before setTime()');
    }

    // Clamp to valid range
    const duration = this.getDuration();
    const clampedTime = Math.max(0, Math.min(seconds, duration));

    this.audioElement.currentTime = clampedTime;
  }

  /**
   * Set playback speed
   * @param {number} rate - Speed (0.5 - 2.0)
   */
  setPlaybackRate(rate) {
    if (!this._isInitialized) {
      throw new Error('NativeAudioBackend: init() must be called before setPlaybackRate()');
    }

    // Clamp to reasonable range
    const clampedRate = Math.max(0.5, Math.min(rate, 2.0));
    this.audioElement.playbackRate = clampedRate;
  }

  /**
   * Set volume
   * @param {number} volume - Volume (0.0 - 1.0)
   */
  setVolume(volume) {
    if (!this._isInitialized) {
      throw new Error('NativeAudioBackend: init() must be called before setVolume()');
    }

    // Clamp to valid range
    const clampedVolume = Math.max(0.0, Math.min(volume, 1.0));
    this.audioElement.volume = clampedVolume;
  }

  /**
   * Get current time
   * @returns {number} Current time in seconds
   */
  getCurrentTime() {
    return this.audioElement ? this.audioElement.currentTime : 0;
  }

  /**
   * Get duration
   * @returns {number} Duration in seconds (0 if not loaded)
   */
  getDuration() {
    if (!this.audioElement) return 0;

    // duration is NaN if not loaded
    const duration = this.audioElement.duration;
    return isNaN(duration) ? 0 : duration;
  }

  /**
   * Cleanup
   */
  destroy() {
    if (!this.audioElement) return;

    // Remove all event listeners
    this.audioElement.removeEventListener('loadstart', this._boundHandlers.onLoadStart);
    this.audioElement.removeEventListener('canplaythrough', this._boundHandlers.onCanPlayThrough);
    this.audioElement.removeEventListener('play', this._boundHandlers.onPlay);
    this.audioElement.removeEventListener('pause', this._boundHandlers.onPause);
    this.audioElement.removeEventListener('ended', this._boundHandlers.onEnded);
    this.audioElement.removeEventListener('timeupdate', this._boundHandlers.onTimeUpdate);
    this.audioElement.removeEventListener('seeking', this._boundHandlers.onSeeking);
    this.audioElement.removeEventListener('error', this._boundHandlers.onError);

    // Pause and clear source
    this.audioElement.pause();
    this.audioElement.src = '';
    this.audioElement.load(); // Clear buffered data

    // Clear reference
    this.audioElement = null;

    // Clear all event subscribers
    this.removeAllListeners();

    this._isInitialized = false;
  }

  /**
   * INTERNAL EVENT HANDLERS
   * Map HTML5 events to contract events
   */

  _onLoadStart() {
    this.emit(AudioBackend.EVENTS.LOADING);
  }

  _onCanPlayThrough() {
    this._isLoading = false;
    this.emit(AudioBackend.EVENTS.READY, {
      duration: this.getDuration()
    });
  }

  _onPlay() {
    this._isPlaying = true;
    this.emit(AudioBackend.EVENTS.PLAY);
  }

  _onPause() {
    this._isPlaying = false;
    this.emit(AudioBackend.EVENTS.PAUSE);
  }

  _onEnded() {
    this._isPlaying = false;
    // CRITICAL: HTML5 uses 'ended', contract uses 'finish'
    this.emit(AudioBackend.EVENTS.FINISH);
  }

  _onTimeUpdate() {
    const currentTime = this.getCurrentTime();

    // Emit both events for compatibility
    this.emit(AudioBackend.EVENTS.TIMEUPDATE, { currentTime });

    // For HTML5, AUDIOPROCESS is alias of TIMEUPDATE (~4Hz)
    // Only emit if actually playing (not paused)
    if (this._isPlaying) {
      this.emit(AudioBackend.EVENTS.AUDIOPROCESS, { currentTime });
    }
  }

  _onSeeking() {
    this.emit(AudioBackend.EVENTS.SEEKING, {
      currentTime: this.getCurrentTime()
    });
  }

  _onError(event) {
    const audioElement = this.audioElement;
    let errorMessage = 'Unknown audio error';

    // Decode HTML5 MediaError codes
    if (audioElement && audioElement.error) {
      switch (audioElement.error.code) {
        case 1:
          errorMessage = 'Audio loading aborted';
          break;
        case 2:
          errorMessage = 'Network error while loading audio';
          break;
        case 3:
          errorMessage = 'Audio decoding failed';
          break;
        case 4:
          errorMessage = 'Audio format not supported';
          break;
      }
    }

    this.emit(AudioBackend.EVENTS.ERROR, {
      error: new Error(errorMessage)
    });
  }
}