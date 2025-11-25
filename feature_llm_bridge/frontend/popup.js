const SERVER_URL = "https://localhost:8005";

document.addEventListener('DOMContentLoaded', async () => {
    // --- UI REFERENCES ---
    const statusSpan = document.getElementById('server-status');
    const connectBtn = document.getElementById('connect-btn');
    const disconnectBtn = document.getElementById('disconnect-btn');
    const agentSelect = document.getElementById('agent-select');
    const controlPanel = document.getElementById('control-panel');

    // Routing UI
    const targetSelect = document.getElementById('target-select');
    const toggleNext = document.getElementById('toggle-next');
    const toggleStream = document.getElementById('toggle-stream');
    const toggleLoop = document.getElementById('toggle-loop');
    const statusMsg = document.getElementById('status-msg');

    // --- 1. CHECK SERVER HEALTH ---
    try {
        const response = await fetch(`${SERVER_URL}/health`);
        if (response.ok) {
            statusSpan.textContent = "Online 🟢";
            statusSpan.className = "status-ok";
        }
    } catch (e) {
        statusSpan.textContent = "Offline 🔴";
    }

    // --- 2. DYNAMIC STATE LOADER (The Highlander Fix) ---
    function refreshUI() {
        const agentId = agentSelect.value;
        // Create a unique key for this specific agent (e.g., "bridge_state_Claude")
        const stateKey = `bridge_state_${agentId}`;

        chrome.storage.local.get([stateKey], (result) => {
            const isConnected = result[stateKey] || false;
            toggleUI(isConnected);
            if (isConnected) loadRoutingState();
        });
    }

    // --- 3. UI HELPERS ---
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

    // Initial Load
    refreshUI();
    // Update whenever user switches the dropdown
    agentSelect.addEventListener('change', refreshUI);

    // --- 4. CONNECT / DISCONNECT LOGIC ---
    connectBtn.addEventListener('click', () => {
        const agentId = agentSelect.value;
        const stateKey = `bridge_state_${agentId}`;

        // Save TRUE to the specific agent key
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

        // Save FALSE to the specific agent key
        chrome.storage.local.set({[stateKey]: false}, () => {
            chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
                if (tabs[0]) chrome.tabs.sendMessage(tabs[0].id, {action: "STOP_BRIDGE"});
            });
            toggleUI(false);
        });
    });

    // --- 5. ROUTING LOGIC ---
    async function loadRoutingState() {
        const agentId = agentSelect.value;
        try {
            const res = await fetch(`${SERVER_URL}/api/v1/admin/state`);
            const data = await res.json();

            targetSelect.innerHTML = '<option value="None">Select Target...</option>';
            Object.keys(data.state.connected_agents).forEach(otherId => {
                if (otherId !== agentId) {
                    const opt = document.createElement('option');
                    opt.value = otherId;
                    opt.textContent = otherId;
                    targetSelect.appendChild(opt);
                }
            });

            const myConfig = data.state.agent_configs[agentId] || {};
            if (myConfig.target_agent) targetSelect.value = myConfig.target_agent;
            toggleNext.checked = myConfig.send_next_only || false;
            toggleStream.checked = myConfig.auto_stream || false;
            toggleLoop.checked = myConfig.recursive_loop || false;

        } catch (e) {
            console.warn("Failed to load state", e);
        }
    }

    async function pushConfig() {
        const agentId = agentSelect.value;
        const payload = {
            target_agent: targetSelect.value,
            send_next_only: toggleNext.checked,
            auto_stream: toggleStream.checked,
            recursive_loop: toggleLoop.checked
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

    targetSelect.addEventListener('change', pushConfig);
    toggleNext.addEventListener('change', pushConfig);
    toggleStream.addEventListener('change', pushConfig);
    toggleLoop.addEventListener('change', pushConfig);

    setInterval(() => {
        if (controlPanel.style.display !== 'none') loadRoutingState();
    }, 2000);
});