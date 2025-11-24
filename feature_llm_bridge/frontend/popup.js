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

    // --- 2. LOAD SAVED STATE ---
    chrome.storage.local.get(['agentId', 'isConnected'], (result) => {
        if (result.agentId) agentSelect.value = result.agentId;
        toggleUI(result.isConnected);
        if (result.isConnected) loadRoutingState();
    });

    // --- 3. UI HELPERS ---
    function toggleUI(isConnected) {
        if (isConnected) {
            connectBtn.style.display = 'none';
            disconnectBtn.style.display = 'block';
            agentSelect.disabled = true;
            controlPanel.style.display = 'block'; // Show Routing Controls
            loadRoutingState(); // Fetch latest rules from server
        } else {
            connectBtn.style.display = 'block';
            disconnectBtn.style.display = 'none';
            agentSelect.disabled = false;
            controlPanel.style.display = 'none';
        }
    }

    // --- 4. CONNECT / DISCONNECT LOGIC ---
    connectBtn.addEventListener('click', () => {
        const agentId = agentSelect.value;
        chrome.storage.local.set({agentId: agentId, isConnected: true}, () => {
            chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
                chrome.tabs.sendMessage(tabs[0].id, {action: "START_BRIDGE", agentId: agentId});
            });
            toggleUI(true);
        });
    });

    disconnectBtn.addEventListener('click', () => {
        chrome.storage.local.set({isConnected: false}, () => {
            chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
                chrome.tabs.sendMessage(tabs[0].id, {action: "STOP_BRIDGE"});
            });
            toggleUI(false);
        });
    });

    // --- 5. ROUTING LOGIC (The New Stuff) ---

    // Fetch current rules from the Brain to update UI
    async function loadRoutingState() {
        const agentId = agentSelect.value;
        try {
            const res = await fetch(`${SERVER_URL}/api/v1/admin/state`);
            const data = await res.json();

            // A. Populate Dropdown (Everyone except me)
            targetSelect.innerHTML = '<option value="None">Select Target...</option>';
            Object.keys(data.state.connected_agents).forEach(otherId => {
                if (otherId !== agentId) {
                    const opt = document.createElement('option');
                    opt.value = otherId;
                    opt.textContent = otherId;
                    targetSelect.appendChild(opt);
                }
            });

            // B. Set Toggles based on backend config
            const myConfig = data.state.agent_configs[agentId] || {};
            if (myConfig.target_agent) targetSelect.value = myConfig.target_agent;
            toggleNext.checked = myConfig.send_next_only || false;
            toggleStream.checked = myConfig.auto_stream || false;
            toggleLoop.checked = myConfig.recursive_loop || false;

        } catch (e) {
            console.warn("Failed to load state", e);
        }
    }

    // Send changes to the Brain immediately
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

            // Visual Feedback
            statusMsg.textContent = "Saved ✓";
            statusMsg.style.color = "#00b894";
            setTimeout(() => statusMsg.textContent = "", 1000);

        } catch (e) {
            statusMsg.textContent = "Save Failed ✗";
            statusMsg.style.color = "#d63031";
        }
    }

    // Attach Listeners to Inputs
    targetSelect.addEventListener('change', pushConfig);
    toggleNext.addEventListener('change', pushConfig);
    toggleStream.addEventListener('change', pushConfig);
    toggleLoop.addEventListener('change', pushConfig);

    // Auto-refresh state while popup is open (to see if flags turn off)
    setInterval(() => {
        if (controlPanel.style.display !== 'none') loadRoutingState();
    }, 2000);
});