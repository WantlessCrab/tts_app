const SERVER_URL = "https://localhost:8005/api/v1";
let POLL_INTERVAL = null;
let UI_OBSERVER = null;
let AGENT_ID = null;
let userHasActed = false;
let lastScrapedText = "";

// IMMORTAL LISTENER: Detects typing even if React redraws the box
document.body.addEventListener('input', (e) => {
    if (!AGENT_ID) return;
    const selector = SELECTORS[AGENT_ID];
    if (!selector) return;

    // Check if event came from our target input
    if (e.target.matches(selector.input) || e.target.closest(selector.input)) {
        if (!userHasActed) {
            console.log("[Bridge] 👤 User Activity Detected. Arming capture trigger.");
            userHasActed = true;
        }
    }
}, true);

// NEW: Stability Config
const STABILITY_THRESHOLD = 2500; // Wait 2.5s of silence
let captureDebounceTimer = null;

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

    // New Code: Inject into <HTML> (The Root) to bypass React's "Body" control
    document.documentElement.appendChild(div);
    console.log("[Bridge] ✅ UI Injected into Root (React-Safe Zone).");
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
    // 🛑 STOP GAP 5: The Zombie Killer
    // We forcefully kill any existing process before starting a new one.
    // This prevents "double-binding" if the user clicks Connect rapidly.
    stopBridge();

    AGENT_ID = agentId;
    console.log(`[Bridge] Starting as agent: ${AGENT_ID}`);

    // Snapshot existing content
    const selector = SELECTORS[AGENT_ID];
    if (selector) {
        const msgs = document.querySelectorAll(selector.lastMessage);
        if (msgs.length > 0) {
            lastScrapedText = msgs[msgs.length - 1].innerText;
        }
    }

    injectOverlay();

    // Passive Observer (No Stutter)
    UI_OBSERVER = new MutationObserver((mutations) => {
        if (!document.getElementById('llm-bridge-overlay')) {
            console.log("[Bridge] Overlay lost. Re-injecting...");
            injectOverlay();
        }
    });

    UI_OBSERVER.observe(document.body, {childList: true, subtree: false});

    // Start Polling
    pollServer();
    POLL_INTERVAL = setInterval(pollServer, 2000);
}

function stopBridge() {
    if (POLL_INTERVAL) clearInterval(POLL_INTERVAL);
    if (UI_OBSERVER) {
        UI_OBSERVER.disconnect();
        UI_OBSERVER = null;
    }
    const overlay = document.getElementById('llm-bridge-overlay');
    if (overlay) overlay.remove();
    console.log("[Bridge] Stopped.");
}

/**
 * DETECT REAL AGENT STATE
 * Prevents "Rude Interruptions" by checking if the AI is busy or if the user is typing.
 */
function detectAgentState() {
    // 1. Check for "Stop Generating" indicators (Busy)
    // ChatGPT/Claude often replace the Send button with a Stop button
    // or disable the Send button during generation.
    const selector = SELECTORS[AGENT_ID];
    if (!selector) return "idle";

    const sendBtn = document.querySelector(selector.sendBtn);
    const stopBtn = document.querySelector('button[aria-label="Stop generating"]'); // Common pattern

    // A. Is the agent actively writing? (Busy)
    if (stopBtn || (sendBtn && sendBtn.disabled && !sendBtn.hasAttribute('disabled'))) {
        // Note: Some UIs disable send when empty, so this is a heuristic.
        // Better check: Is there a "result-streaming" class?
        if (document.querySelector('.result-streaming') || document.querySelector('.text-cursor')) {
            return "processing";
        }
    }

    // B. Is the User typing? (Awaiting Input)
    const inputBox = document.querySelector(selector.input);
    if (inputBox) {
        const hasText = (inputBox.value || inputBox.innerText || "").trim().length > 0;
        if (hasText && !userHasActed) {
            // Text exists, but WE didn't put it there. The user must be typing.
            return "awaiting_input";
        }
    }

    return "idle";
}

async function pollServer() {
    if (!AGENT_ID) return;

    // 1. Get Real State
    const realState = detectAgentState();

    try {
        chrome.runtime.sendMessage({
            action: "PROXY_STATUS",
            payload: {
                agent_id: AGENT_ID,
                state: realState, // <--- No longer hardcoded "idle"
                active_url: window.location.href,
                tab_id: 0
            }
        }, (response) => {
            if (chrome.runtime.lastError) {
                // ... (Error handling code remains the same) ...
                return;
            }

            // Only process instructions if we successfully reported our state
            if (response && response.success) {
                // The backend will now see "processing" and SHOULD return "noop"
                handleInstruction(response.data);
                scrapeAndReport();
            }
        });
    } catch (e) {
        console.warn("[Bridge] Poll skipped.");
    }
}

// --- INSTRUCTION HANDLER ---
// --- INSTRUCTION HANDLER ---
async function handleInstruction(instr) {
    updateOverlayStatus("online");

    if (instr.command === "inject_content" && instr.content_payload) {
        console.log("[Bridge] 💉 Job Received. Initiating human-like sequence...");

        const selector = SELECTORS[AGENT_ID];
        const inputBox = document.querySelector(selector.input);

        if (inputBox) {
            // 1. Human Delay (The Anti-Bot Fix)
            // Wait 1.5s before touching the box. Prevents "Instant Type" detection.
            setTimeout(() => {

                // 2. Execute the Safe Insertion Strategy
                // This function (defined below) chooses the correct method for the specific LLM.
                const success = safelyInsertText(inputBox, instr.content_payload);

                if (success) {
                    console.log("[Bridge] ✅ Text inserted safely.");
                    userHasActed = true; // Arm the capture trigger

                    // 3. Send Delay
                    // Wait another 1s "reaction time" before pressing Enter
                    setTimeout(() => {
                        triggerSend(selector, inputBox);
                    }, 1000);
                } else {
                    console.error("[Bridge] ❌ Insertion failed. Input is blocked.");
                }

            }, 1500);
        }
    }
}

/**
 * THE SURGICAL INSERTION ENGINE v3
 */
/**
 * THE SURGICAL INSERTION ENGINE v4 (Hardened)
 * * Strategy:
 * 1. "Typewriter" (execCommand) - Simulates human typing. Best for React/ProseMirror.
 * 2. "Native Setter" - Bypasses React state locking on TextAreas.
 * 3. "Range Injection" - Fallback for older ContentEditable divs.
 */
function safelyInsertText(element, text) {
    // 1. PREPARE: Focus is mandatory for the Typewriter to know "where" to type.
    try {
        element.focus();
    } catch (e) {
        console.warn("[Bridge] ⚠️ Could not focus element:", e);
    }

    // 2. PRIMARY STRATEGY: The Typewriter Protocol
    // This is the "Solution" we verified in Probe v3.
    // It automatically triggers 'input' and 'change' events trusted by React.
    const typeWriterSuccess = document.execCommand('insertText', false, text);

    if (typeWriterSuccess) {
        console.log("[Bridge] 🟢 Injection Strategy: Typewriter (Success)");
        return true;
    }

    // --- FALLBACKS (If browser blocks execCommand) ---
    console.warn("[Bridge] ⚠️ Typewriter failed. Attempting Fallbacks...");

    const tagName = element.tagName.toUpperCase();

    // FALLBACK A: React Textarea/Input (Native Prototype Setter)
    if (tagName === 'TEXTAREA' || tagName === 'INPUT') {
        try {
            const valueSetter = Object.getOwnPropertyDescriptor(element, 'value').set;
            const prototype = Object.getPrototypeOf(element);
            const prototypeValueSetter = Object.getOwnPropertyDescriptor(prototype, 'value').set;

            // Call the Setter from the Prototype (bypassing React's overwrite)
            if (prototypeValueSetter && valueSetter !== prototypeValueSetter) {
                prototypeValueSetter.call(element, text);
            } else {
                valueSetter.call(element, text);
            }

            // Manually fire the event chain
            element.dispatchEvent(new Event('input', {bubbles: true}));
            return true;

        } catch (e) {
            console.error("[Bridge] ❌ Fallback A failed:", e);
            return false;
        }
    }

    // FALLBACK B: Rich Text Divs (Range Injection)
    else {
        try {
            // 1. Create a Text Node
            const textNode = document.createTextNode(text);

            // 2. Wipe & Replace
            const selection = window.getSelection();
            selection.removeAllRanges();

            const range = document.createRange();
            range.selectNodeContents(element);
            range.deleteContents(); // Clear existing
            range.insertNode(textNode); // Insert new

            // 3. Move cursor to end
            range.collapse(false);
            selection.addRange(range);

            // 4. Fire Events
            element.dispatchEvent(new Event('input', {bubbles: true}));
            return true;
        } catch (e) {
            console.error("[Bridge] ❌ Fallback B failed:", e);
            return false;
        }
    }
}

function triggerSend(selector, inputBox) {
    if (AGENT_ID === "Gemini") {
        // Gemini: Enter Key (Google Framework Requirement)
        const event = new KeyboardEvent('keydown', {
            bubbles: true, cancelable: true, key: 'Enter', code: 'Enter', keyCode: 13
        });
        inputBox.dispatchEvent(event);
    } else {
        // ChatGPT & Claude: The "Double-Check" Click Strategy
        const getBtn = () => document.querySelector(selector.sendBtn);
        let sendBtn = getBtn();

        if (sendBtn && !sendBtn.disabled) {
            console.log("[Bridge] ⏳ Button looks ready... stabilizing (300ms).");

            // STABILITY WAIT: Ensure React hydration is complete
            setTimeout(() => {
                sendBtn = getBtn(); // Re-fetch to ensure fresh reference
                if (sendBtn && !sendBtn.disabled) {
                    console.log("[Bridge] 🖱️ Click confirmed.");
                    sendBtn.click();
                } else {
                    console.warn("[Bridge] ⚠️ Button disabled after wait. Aborting click.");
                }
            }, 300); // 300ms buffer
        } else {
            console.warn("[Bridge] Send button not ready.");
        }
    }
}

async function scrapeAndReport() {
    const selector = SELECTORS[AGENT_ID];
    if (!selector) return;

    const msgElements = document.querySelectorAll(selector.lastMessage);
    if (msgElements.length === 0) return;

    const lastMsgElement = msgElements[msgElements.length - 1];
    const text = lastMsgElement.innerText;

    // 🛑 STOP GAP 4: The "Thought Pause" Validator
    // Ensure we are capturing the AI, not the User.
    if (isUserMessage(lastMsgElement)) {
        // console.log("[Bridge] Ignored capture (User message detected)");
        return;
    }

    // LOGIC: Capture if text is new OR if we are waiting for a response to user
    const isNewText = text && text !== lastScrapedText && text.length > 5;
    const isResponseToUser = userHasActed && text.length > 5;

    if (isNewText || isResponseToUser) {
        if (captureDebounceTimer) clearTimeout(captureDebounceTimer);

        captureDebounceTimer = setTimeout(() => {
            console.log(`[Bridge] ⚡ Text Stable (${text.length} chars). Sending...`);
            lastScrapedText = text;

            if (userHasActed) {
                console.log("[Bridge] 🔄 Cycle complete. Resetting User Flag.");
                userHasActed = false;
            }

            chrome.runtime.sendMessage({
                action: "PROXY_CAPTURE",
                payload: {agent_id: AGENT_ID, content: text}
            });

            captureDebounceTimer = null;
        }, STABILITY_THRESHOLD);
    }
}

// HELPER: Detects if an element belongs to the User (Platform Specific)
function isUserMessage(element) {
    // 1. Generic Heuristic (Parents often hold the class)
    // We traverse up 3 levels to check containers
    let parent = element.parentElement;
    for (let i = 0; i < 3; i++) {
        if (!parent) break;
        const classStr = (parent.className || "").toString();

        // Claude User Message Class
        if (classStr.includes("font-user-message")) return true;

        // ChatGPT User Message (data-message-author-role="user")
        if (parent.getAttribute && parent.getAttribute("data-message-author-role") === "user") return true;

        // Gemini User Message (often wrapped in .user-query)
        if (classStr.includes("user-query")) return true;

        parent = parent.parentElement;
    }
    return false;
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

window.addEventListener("load", initAutoResume);