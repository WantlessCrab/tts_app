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
        lastMessage: ".model-response-text, .message-content:not(.user-query)",
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
    // 1. Check if already exists
    if (document.getElementById('llm-bridge-overlay')) return;

    // 2. Safety: If body is missing, wait for it
    if (!document.body) {
        console.warn("[Bridge] ⏳ Body not ready. Retrying injection...");
        setTimeout(injectOverlay, 500);
        return;
    }

    const div = document.createElement('div');
    div.id = 'llm-bridge-overlay';
    div.innerHTML = `
    <div class="bridge-agent-name">${AGENT_ID || "Bridge"}</div>
    <div class="bridge-status-container">
        <div id="bridge-status-dot" class="online"></div>
        <span id="llm-bridge-status-text">Connected</span>
    </div>`;

    // FIX: Attach to body so it sits on top of the page content
    document.body.appendChild(div);
    console.log("[Bridge] ✅ UI Injected into Body.");
}

function updateOverlayStatus(status) {
    const dot = document.getElementById('bridge-status-dot');
    const text = document.getElementById('llm-bridge-status-text');
    if (dot && text) {
        dot.className = status === 'online' ? 'online' : 'offline';
        text.innerText = status === 'online' ? 'Online' : 'Offline';
    }
}

// --- CORE LOOPS ---
function startBridge(agentId) {
    if (POLL_INTERVAL) clearInterval(POLL_INTERVAL);
    if (UI_WATCHDOG) clearInterval(UI_WATCHDOG);

    AGENT_ID = agentId;
    console.log(`[Bridge] Starting as agent: ${AGENT_ID}`);

    // NEW: Snapshot existing content to prevent "Past Message" routing
    const selector = SELECTORS[AGENT_ID];
    if (selector) {
        // We use querySelectorAll because Gemini/Claude often have multiple blocks
        const msgs = document.querySelectorAll(selector.lastMessage);
        if (msgs.length > 0) {
            // Set the baseline to the current text so we don't re-send it
            lastScrapedText = msgs[msgs.length - 1].innerText;
            console.log(`[Bridge] Baseline set. Ignoring ${lastScrapedText.length} chars.`);
        }
    }

    injectOverlay();
    POLL_INTERVAL = setInterval(pollServer, 2000);

    UI_WATCHDOG = setInterval(() => {
        if (!document.getElementById('llm-bridge-overlay')) injectOverlay();
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

    // FIX: Use Proxy instead of direct fetch
    chrome.runtime.sendMessage({
        action: "PROXY_STATUS",
        payload: {
            agent_id: AGENT_ID,
            state: "idle",
            active_url: window.location.href,
            tab_id: 0
        }
    }, (response) => {
        if (response && response.success) {
            handleInstruction(response.data);
            scrapeAndReport();
        } else {
            // updateOverlayStatus("offline");
        }
    });
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

            // FIX: Always run simulation for Gemini to unlock button
            if (!success || inputBox.innerText.trim() === "" || AGENT_ID === "Gemini") {
                simulateUserInput(inputBox, instr.content_payload);
            }

            console.log("[Bridge] ✅ Injection complete. Waiting for UI...");

            setTimeout(() => {
                const sendBtn = document.querySelector(selector.sendBtn);
                if (sendBtn) {
                    // Try to force enable
                    sendBtn.disabled = false;
                    sendBtn.setAttribute('aria-disabled', 'false');

                    if (!sendBtn.disabled) {
                        sendBtn.click();
                    } else {
                        console.log("[Bridge] ⚠️ Button disabled. Using Nuclear Enter.");
                        pressEnter(inputBox);
                    }
                } else {
                    console.log("[Bridge] ☢️ Button not found. Using Nuclear Enter.");
                    pressEnter(inputBox);
                }
            }, 1500);
        }
    }
}

// FIX: "Heavy Duty" Simulator for Google Frameworks
function simulateUserInput(element, text) {
    element.focus();
    element.innerHTML = '';

    if (element.tagName === 'TEXTAREA' || element.tagName === 'INPUT') {
        element.value = text;
    } else {
        element.textContent = text;
    }

    // 1. Standard Events
    element.dispatchEvent(new Event('input', {bubbles: true}));
    element.dispatchEvent(new Event('change', {bubbles: true}));

    // 2. Legacy/Framework Events (Critical for Gemini)
    const textInputEvent = document.createEvent('TextEvent');
    textInputEvent.initTextEvent('textInput', true, true, null, text, 9, "en-US");
    element.dispatchEvent(textInputEvent);
}

function pressEnter(element) {
    const event = new KeyboardEvent('keydown', {
        bubbles: true,
        cancelable: true,
        key: 'Enter',
        code: 'Enter',
        keyCode: 13
    });
    element.dispatchEvent(event);
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
        console.log(`[Bridge] ⚡ New content (${text.length} chars). Sending...`);
        lastScrapedText = text;

        // FIX: Use Proxy
        chrome.runtime.sendMessage({
            action: "PROXY_CAPTURE",
            payload: {agent_id: AGENT_ID, content: text}
        });
    }
}

// --- MESSAGING & STATE ---
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "START_BRIDGE") {
        startBridge(request.agentId);
        const key = `bridge_state_${request.agentId}`;
        chrome.storage.local.set({[key]: true});
    } else if (request.action === "STOP_BRIDGE") {
        stopBridge();
        const key = `bridge_state_${AGENT_ID}`;
        chrome.storage.local.set({[key]: false});
    } else if (request.action === "CHECK_HEARTBEAT") {
        const isAlive = (AGENT_ID === request.expectedAgent);
        sendResponse({isConnected: isAlive});
    }
});

// --- AUTO-RESUME ---
function initAutoResume() {
    const agents = ["Claude", "Gemini", "ChatGPT"];
    const keys = agents.map(id => `bridge_state_${id}`);

    chrome.storage.local.get(keys, (result) => {
        const url = window.location.href;
        let detectedId = null;

        if (url.includes("claude")) detectedId = "Claude";
        else if (url.includes("google")) detectedId = "Gemini";
        else if (url.includes("chatgpt")) detectedId = "ChatGPT";

        if (detectedId) {
            const stateKey = `bridge_state_${detectedId}`;
            if (result[stateKey] === true) {
                console.log(`[Bridge] 🔄 Auto-resuming ${detectedId}...`);
                startBridge(detectedId);
            }
        }
    });
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAutoResume);
} else {
    initAutoResume();
}