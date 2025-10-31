/**
 * EVENT EMITTER
 *
 * Lightweight observer pattern implementation.
 * Provides event subscription mechanism for AudioBackend classes.
 *
 * Usage:
 *   const emitter = new EventEmitter();
 *   emitter.on('ready', (data) => console.log('Ready:', data));
 *   emitter.emit('ready', { duration: 120 });
 */
class EventEmitter {
  constructor() {
    // Internal event registry
    // Structure: { eventName: [callback1, callback2, ...] }
    this._events = {};
  }

  /**
   * Subscribe to event
   * @param {string} event - Event name
   * @param {function} callback - Handler function
   */
  on(event, callback) {
    if (typeof callback !== 'function') {
      throw new Error('EventEmitter.on() requires callback to be a function');
    }

    // Initialize event array if first subscriber
    if (!this._events[event]) {
      this._events[event] = [];
    }

    // Add callback to subscribers list
    this._events[event].push(callback);
  }

  /**
   * Unsubscribe from event
   * @param {string} event - Event name
   * @param {function} callback - Handler function to remove
   */
  off(event, callback) {
    if (!this._events[event]) {
      return; // No subscribers for this event
    }

    // Filter out the specified callback
    this._events[event] = this._events[event].filter(cb => cb !== callback);

    // Cleanup empty event arrays
    if (this._events[event].length === 0) {
      delete this._events[event];
    }
  }

  /**
   * Emit event to all subscribers
   * @param {string} event - Event name
   * @param {*} data - Data to pass to handlers
   */
  emit(event, data) {
    if (!this._events[event]) {
      return; // No subscribers for this event
    }

    // Call all subscribers with data
    // Use slice() to prevent issues if callback modifies the array
    this._events[event].slice().forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        // Prevent one subscriber error from blocking others
        console.error(`EventEmitter: Error in ${event} handler:`, error);
      }
    });
  }

  /**
   * Remove all event listeners
   * Critical for cleanup/destroy operations
   */
  removeAllListeners() {
    this._events = {};
  }

  /**
   * Get subscriber count for event (debugging)
   * @param {string} event - Event name
   * @returns {number} Number of subscribers
   */
  listenerCount(event) {
    return this._events[event] ? this._events[event].length : 0;
  }
}