const SERVER_URL = "https://localhost:8005";

document.addEventListener('DOMContentLoaded', async () => {
    // --- 1. UI REFERENCES ---
    const statusSpan = document.getElementById('server-status');
    const connectBtn = document.getElementById('connect-btn');
    const disconnectBtn = document.getElementById('disconnect-btn');
    const agentSelect = document.getElementById('agent-select');
    const controlPanel = document.getElementById('control-panel');

    // Routing UI (Using target-grid, NOT target-select)
    const targetGrid = document.getElementById('target-grid');
    const toggleNext = document.getElementById('toggle-next');
    const toggleStream = document.getElementById('toggle-stream');
    const toggleLoop = document.getElementById('toggle-loop');

    // Wrapper UI
    const toggleWrapper = document.getElementById('toggle-wrapper');
    const personaInput = document.getElementById('persona-input');
    const wrapperStyle = document.getElementById('wrapper-style');

    const statusMsg = document.getElementById('status-msg');

    // --- 2. AUTO-DETECT AGENT FROM URL ---
    function detectCurrentAgent() {
        chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
            if (!tabs[0] || !tabs[0].url) {
                refreshUI();
                return;
            }
            const url = tabs[0].url;
            let detectedId = null;
            if (url.includes("claude.ai")) detectedId = "Claude";
            else if (url.includes("gemini.google.com")) detectedId = "Gemini";
            else if (url.includes("chatgpt.com")) detectedId = "ChatGPT";

            if (detectedId) {
                console.log(`[Popup] Detected ${detectedId}. Auto-selecting.`);
                agentSelect.value = detectedId;
            }
            refreshUI();
        });
    }

    // --- 3. DYNAMIC STATE LOADER (Fixed) ---
    function refreshUI() {
        const agentId = agentSelect.value;
        const stateKey = `bridge_state_${agentId}`;

        chrome.storage.local.get([stateKey], (result) => {
            const isConnected = result[stateKey] || false;
            toggleUI(isConnected);
            if (isConnected) loadRoutingState();
        });
    }

    // Start Sequence
    detectCurrentAgent();
    agentSelect.addEventListener('change', refreshUI);

    // --- 4. UI HELPERS ---
    function toggleUI(isConnected) {
        if (isConnected) {
            connectBtn.style.display = 'none';
            disconnectBtn.style.display = 'block';
            agentSelect.disabled = true;
            controlPanel.style.display = 'block';
            loadRoutingState();
        } else {
            connectBtn.style.display = 'block';
            disconnectBtn.style.display = 'none';
            agentSelect.disabled = false;
            controlPanel.style.display = 'none';
        }
    }

    // --- 5. CONNECT / DISCONNECT LOGIC ---
    connectBtn.addEventListener('click', () => {
        const agentId = agentSelect.value;
        const stateKey = `bridge_state_${agentId}`;

        chrome.storage.local.set({[stateKey]: true}, () => {
            chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
                if (tabs[0]) chrome.tabs.sendMessage(tabs[0].id, {action: "START_BRIDGE", agentId: agentId});
            });
            toggleUI(true);
        });
    });

    disconnectBtn.addEventListener('click', () => {
        const agentId = agentSelect.value;
        const stateKey = `bridge_state_${agentId}`;

        chrome.storage.local.set({[stateKey]: false}, () => {
            chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
                if (tabs[0]) chrome.tabs.sendMessage(tabs[0].id, {action: "STOP_BRIDGE"});
            });
            toggleUI(false);
        });
    });

    // --- 6. ROUTING & WRAPPER LOGIC (Phase 1 Complete) ---
    async function loadRoutingState() {
        const agentId = agentSelect.value;
        try {
            const res = await fetch(`${SERVER_URL}/api/v1/admin/state`);
            const data = await res.json();
            const myConfig = data.state.agent_configs[agentId] || {};

            // A. Generate Target Grid (Checkboxes)
            if (targetGrid) {
                targetGrid.innerHTML = '';
                const peers = Object.keys(data.state.connected_agents).filter(id => id !== agentId);

                if (peers.length === 0) {
                    targetGrid.innerHTML = '<span style="color:#aaa; font-size:11px;">No other agents connected.</span>';
                } else {
                    peers.forEach(peerId => {
                        const row = document.createElement('div');
                        row.className = 'target-item';

                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.value = peerId;
                        checkbox.className = 'target-checkbox';

                        if (myConfig.target_agents && myConfig.target_agents.includes(peerId)) {
                            checkbox.checked = true;
                        }

                        checkbox.addEventListener('change', pushConfig);

                        const label = document.createElement('span');
                        label.textContent = peerId;

                        row.appendChild(checkbox);
                        row.appendChild(label);
                        targetGrid.appendChild(row);
                    });
                }
            }

            // B. Set Toggles
            toggleNext.checked = myConfig.send_next_only || false;
            toggleStream.checked = myConfig.auto_stream || false;
            toggleLoop.checked = myConfig.recursive_loop || false;

            // C. Set Wrapper Inputs
            if (toggleWrapper) toggleWrapper.checked = myConfig.use_wrapper || false;
            if (personaInput && myConfig.persona_name) personaInput.value = myConfig.persona_name;
            if (wrapperStyle && myConfig.wrapper_style) wrapperStyle.value = myConfig.wrapper_style;

        } catch (e) {
            console.warn("Failed to load state", e);
        }
    }

    async function pushConfig() {
        const agentId = agentSelect.value;

        // Collect targets from Grid
        const checkedBoxes = document.querySelectorAll('.target-checkbox:checked');
        const selectedTargets = Array.from(checkedBoxes).map(cb => cb.value);

        const payload = {
            target_agents: selectedTargets,
            send_next_only: toggleNext.checked,
            auto_stream: toggleStream.checked,
            recursive_loop: toggleLoop.checked,

            use_wrapper: toggleWrapper ? toggleWrapper.checked : false,
            persona_name: personaInput ? personaInput.value : "AI",
            wrapper_style: wrapperStyle ? wrapperStyle.value : "Standard"
        };

        try {
            await fetch(`${SERVER_URL}/api/v1/admin/config/${agentId}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });

            statusMsg.textContent = "Saved ✓";
            statusMsg.style.color = "#00b894";
            setTimeout(() => statusMsg.textContent = "", 1000);
        } catch (e) {
            statusMsg.textContent = "Save Failed ✗";
            statusMsg.style.color = "#d63031";
        }
    }

    // Listeners
    toggleNext.addEventListener('change', pushConfig);
    toggleStream.addEventListener('change', pushConfig);
    toggleLoop.addEventListener('change', pushConfig);

    if (toggleWrapper) toggleWrapper.addEventListener('change', pushConfig);
    if (personaInput) personaInput.addEventListener('change', pushConfig);
    if (wrapperStyle) wrapperStyle.addEventListener('change', pushConfig);

    setInterval(() => {
        if (controlPanel.style.display !== 'none' && document.activeElement !== personaInput) {
            loadRoutingState();
        }
    }, 2000);
});