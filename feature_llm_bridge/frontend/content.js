const SERVER_URL = "https://localhost:8005/api/v1";
let POLL_INTERVAL = null;
let UI_WATCHDOG = null;
let AGENT_ID = null;

console.log("[Bridge] 🚀 Content Script Loaded.");

// --- CSS SELECTORS ---
const SELECTORS = {
    "Claude": {
        lastMessage: ".standard-markdown",
        input: "div[contenteditable='true']",
        sendBtn: "button[aria-label='Send message']"
    },
    "Gemini": {
        lastMessage: ".message-content",
        input: "div[role='textbox']",
        sendBtn: "button[aria-label='Send message']"
    },
    "ChatGPT": {
        lastMessage: ".markdown",
        input: "#prompt-textarea",
        sendBtn: "button[data-testid='send-button']"
    }
};

// --- UI OVERLAY ---

function injectOverlay() {
    if (document.getElementById('llm-bridge-overlay')) return;

    if (!document.body) {
        console.warn("[Bridge] ⏳ Body not ready. Retrying injection in 500ms...");
        setTimeout(injectOverlay, 500);
        return;
    }

    const div = document.createElement('div');
    div.id = 'llm-bridge-overlay';
    div.innerHTML = `
    <div class="bridge-agent-name">${AGENT_ID}</div>
    <div class="bridge-status-container">
        <div id="bridge-status-dot" class="online"></div>
        <span id="llm-bridge-status-text">Connected</span>
    </div>
  `;

    // Append to documentElement to survive React root wipes
    document.documentElement.appendChild(div);
    console.log("[Bridge] ✅ UI Injected successfully.");
}

function updateOverlayStatus(status) {
    const dot = document.getElementById('bridge-status-dot');
    const text = document.getElementById('llm-bridge-status-text');

    if (dot && text) {
        if (status === 'online') {
            dot.className = 'online';
            text.innerText = 'Online';
        } else {
            dot.className = 'offline';
            text.innerText = 'Offline';
        }
    }
}

// --- CORE LOOPS ---

function startBridge(agentId) {
    if (POLL_INTERVAL) clearInterval(POLL_INTERVAL);
    if (UI_WATCHDOG) clearInterval(UI_WATCHDOG);

    AGENT_ID = agentId;
    console.log(`[Bridge] Starting as agent: ${AGENT_ID}`);

    injectOverlay();
    POLL_INTERVAL = setInterval(pollServer, 2000);

    UI_WATCHDOG = setInterval(() => {
        if (!document.getElementById('llm-bridge-overlay')) {
            console.log("[Bridge] 🛡️ Watchdog: UI was removed. Re-injecting...");
            injectOverlay();
        }
    }, 1000);
}

function stopBridge() {
    if (POLL_INTERVAL) clearInterval(POLL_INTERVAL);
    if (UI_WATCHDOG) clearInterval(UI_WATCHDOG);
    POLL_INTERVAL = null;
    UI_WATCHDOG = null;

    const overlay = document.getElementById('llm-bridge-overlay');
    if (overlay) overlay.remove();
    console.log("[Bridge] Stopped.");
}

async function pollServer() {
    if (!AGENT_ID) return;

    try {
        const response = await fetch(`${SERVER_URL}/status`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                agent_id: AGENT_ID,
                state: "idle",
                active_url: window.location.href,
                tab_id: 0
            })
        });

        const instruction = await response.json();
        handleInstruction(instruction);
        scrapeAndReport();

    } catch (err) {
        updateOverlayStatus("offline");
    }
}

// --- INSTRUCTION HANDLER ---

async function handleInstruction(instr) {
    updateOverlayStatus("online");

    if (instr.command === "inject_content" && instr.content_payload) {
        console.log("[Bridge] 💉 Injecting content...");

        const selector = SELECTORS[AGENT_ID];
        const inputBox = document.querySelector(selector.input);

        if (inputBox) {
            inputBox.focus();
            const success = document.execCommand('insertText', false, instr.content_payload);
            if (!success || inputBox.innerText.trim() === "") {
                simulateUserInput(inputBox, instr.content_payload);
            }
            console.log("[Bridge] ✅ Injection complete. Sending...");

            setTimeout(() => {
                const sendBtn = document.querySelector(selector.sendBtn);
                if (sendBtn) {
                    sendBtn.disabled = false;
                    sendBtn.click();
                } else {
                    console.error("[Bridge] ❌ Send button not found");
                }
            }, 800);

        } else {
            console.error("[Bridge] ❌ Input box not found");
        }
    }
}

function simulateUserInput(element, text) {
    element.innerHTML = '';
    if (element.tagName === 'TEXTAREA' || element.tagName === 'INPUT') {
        element.value = text;
    } else {
        element.textContent = text;
    }
    element.dispatchEvent(new Event('input', {bubbles: true}));
    element.dispatchEvent(new Event('change', {bubbles: true}));
    element.dispatchEvent(new KeyboardEvent('keyup', {bubbles: true}));
}

let lastScrapedText = "";

async function scrapeAndReport() {
    const selector = SELECTORS[AGENT_ID];
    if (!selector) return;

    const msgElements = document.querySelectorAll(selector.lastMessage);
    if (msgElements.length === 0) return;

    const lastMsg = msgElements[msgElements.length - 1];
    const text = lastMsg.innerText;

    if (text && text !== lastScrapedText && text.length > 5) {
        console.log(`[Bridge] ⚡ New content (${text.length} chars). Payload prepared.`);
        lastScrapedText = text;

        try {
            await fetch(`${SERVER_URL}/capture`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    agent_id: AGENT_ID,
                    content: text
                })
            });
        } catch (err) {
            console.error("[Bridge] 💥 NETWORK ERROR in scrapeAndReport:", err);
        }
    }
}

// --- MESSAGING ---
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "START_BRIDGE") {
        startBridge(request.agentId);
        chrome.storage.local.set({agentId: request.agentId, isConnected: true});
    } else if (request.action === "STOP_BRIDGE") {
        stopBridge();
        chrome.storage.local.set({isConnected: false});
    }
});

// --- INITIALIZATION (Self-Correcting Auto-Resume) ---
function initAutoResume() {
    chrome.storage.local.get(['agentId', 'isConnected'], (result) => {
        // 1. Check if we SHOULD be connected
        if (result.isConnected && result.agentId) {
            const url = window.location.href;
            const id = result.agentId;

            // 2. Verify we are on the right site
            const isClaude = id === "Claude" && url.includes("claude");
            const isGemini = id === "Gemini" && url.includes("google.com");
            const isGPT = id === "ChatGPT" && url.includes("chatgpt.com");

            if (isClaude || isGemini || isGPT) {
                console.log(`[Bridge] 🔄 Refresh detected. Resuming ${id}...`);
                startBridge(id);
            } else {
                // 3. FAILSAFE: We are on the wrong site (or a blank tab).
                // We must disconnect to prevent the "Zombie UI" in the popup.
                console.log(`[Bridge] 🛑 State mismatch. Resetting connection state.`);
                chrome.storage.local.set({isConnected: false});
            }
        }
    });
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAutoResume);
} else {
    initAutoResume();
}