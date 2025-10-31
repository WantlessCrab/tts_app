/**
 * AUDIO BACKEND BASE CLASS
 *
 * Abstract interface for audio playback engines.
 * All methods must be implemented by subclasses.
 *
 * Implementations:
 *   - NativeAudioBackend (HTML5 <audio>)
 *   - WavesurferBackend (wavesurfer.js) [Future: Phase 4]
 *
 * Event Contract:
 *   Subclasses must emit these standardized events via EventEmitter.
 */
class AudioBackend extends EventEmitter {
  constructor() {
    super(); // Initialize EventEmitter

    // State flags (subclasses may add more)
    this._isInitialized = false;
    this._isLoading = false;
    this._isPlaying = false;
  }

  /**
   * Initialize backend with container
   * @param {HTMLElement} container - DOM container for audio UI
   * @param {object} options - Backend-specific configuration
   * @returns {Promise<void>}
   */
  async init(container, options = {}) {
    throw new Error('AudioBackend.init() must be implemented by subclass');
  }

  /**
   * Load audio from URL
   * MUST emit: 'loading', then 'ready' or 'error'
   * @param {string} url - Audio file URL
   * @returns {Promise<void>}
   */
  async load(url) {
    throw new Error('AudioBackend.load() must be implemented by subclass');
  }

  /**
   * Start playback
   * MUST emit: 'play'
   * @returns {Promise<void>}
   */
  async play() {
    throw new Error('AudioBackend.play() must be implemented by subclass');
  }

  /**
   * Pause playback
   * MUST emit: 'pause'
   */
  pause() {
    throw new Error('AudioBackend.pause() must be implemented by subclass');
  }

  /**
   * Seek to absolute time
   * MUST emit: 'seeking' with new time
   * @param {number} seconds - Target time in seconds
   */
  setTime(seconds) {
    throw new Error('AudioBackend.setTime() must be implemented by subclass');
  }

  /**
   * Set playback speed
   * @param {number} rate - Speed (0.5 - 2.0)
   */
  setPlaybackRate(rate) {
    throw new Error('AudioBackend.setPlaybackRate() must be implemented by subclass');
  }

  /**
   * Set volume
   * @param {number} volume - Volume (0.0 - 1.0)
   */
  setVolume(volume) {
    throw new Error('AudioBackend.setVolume() must be implemented by subclass');
  }

  /**
   * Get current playback time
   * @returns {number} Current time in seconds
   */
  getCurrentTime() {
    throw new Error('AudioBackend.getCurrentTime() must be implemented by subclass');
  }

  /**
   * Get total duration
   * @returns {number} Duration in seconds (0 if not loaded)
   */
  getDuration() {
    throw new Error('AudioBackend.getDuration() must be implemented by subclass');
  }

  /**
   * Check if currently playing
   * @returns {boolean} True if playing
   */
  isPlaying() {
    return this._isPlaying;
  }

  /**
   * Cleanup resources
   * MUST remove all event listeners and free memory
   */
  destroy() {
    throw new Error('AudioBackend.destroy() must be implemented by subclass');
  }
}

/**
 * STANDARDIZED EVENT NAMES
 *
 * All AudioBackend implementations must emit these events.
 * Use these constants for type safety.
 */
AudioBackend.EVENTS = {
  LOADING: 'loading',       // Emitted when load() starts
  READY: 'ready',           // Emitted when audio decoded and playable
  PLAY: 'play',             // Emitted when playback starts
  PAUSE: 'pause',           // Emitted when playback pauses
  FINISH: 'finish',         // Emitted when playback reaches end
  TIMEUPDATE: 'timeupdate', // Emitted during playback (frequency varies)
  AUDIOPROCESS: 'audioprocess', // Emitted during playback (high frequency)
  SEEKING: 'seeking',       // Emitted when user seeks
  ERROR: 'error'            // Emitted on any error
};