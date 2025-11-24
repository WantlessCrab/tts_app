// background.js
// Handles persistent communication if needed.
// For this MVP, we are delegating the heavy lifting to content.js
// to ensure direct access to the DOM.

chrome.runtime.onInstalled.addListener(() => {
    console.log("LLM Bridge Installed.");
});

// Listener for future background tasks (e.g. keeping the connection alive)
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "LOG") {
        console.log("[Bridge Log]:", message.payload);
    }
});