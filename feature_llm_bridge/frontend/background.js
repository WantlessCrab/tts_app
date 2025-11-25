// background.js - The Privileged Network Handler

const SERVER_URL = "https://localhost:8005/api/v1";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    // 1. PROXY: Handle Status Polling (Bypasses CSP)
    if (request.action === "PROXY_STATUS") {
        fetch(`${SERVER_URL}/status`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(request.payload)
        })
            .then(res => res.json())
            .then(data => sendResponse({success: true, data: data}))
            .catch(err => sendResponse({success: false, error: err.toString()}));

        return true; // Keep channel open for async response
    }

    // 2. PROXY: Handle Capture (Bypasses CSP)
    if (request.action === "PROXY_CAPTURE") {
        fetch(`${SERVER_URL}/capture`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(request.payload)
        })
            .then(res => res.json())
            .then(data => sendResponse({success: true, data: data}))
            .catch(err => sendResponse({success: false, error: err.toString()}));

        return true;
    }
});