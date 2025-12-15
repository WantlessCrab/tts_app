from typing import Dict, Optional, List, Set
from models import ClientStatus, ClientInstruction, RelaySessionState, AgentRoutingConfig
import json
import os
import logging

# --- LOGGING SETUP ---
# Aligning with Principle 2: Observable Spine
logger = logging.getLogger("uvicorn")

# --- WRAPPER TEMPLATES ---
WRAPPERS = {
    "Standard": "--- Incoming Transmission: {persona} ({agent}) ---\n\n{content}\n\n--- End Transmission ---",
    "Minimal": "**{persona}:** {content}",
    "XML": "<response from='{agent}' role='{persona}'>\n{content}\n</response>"
}

STATE_FILE = "/app/relay_state.json"


class RelayService:
    def __init__(self):
        self.pending_payloads: Dict[str, str] = {}
        # 🛑 STOP GAP 6: Load State on Startup
        self.state = self.load_state()

    def load_state(self) -> RelaySessionState:
        """Loads state from JSON or returns fresh."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                    logger.info(f"💾 [System] State loaded from {STATE_FILE}")
                    return RelaySessionState(**data)
            except Exception as e:
                logger.error(f"⚠️ [System] Failed to load state: {e}. Starting fresh.")
        return RelaySessionState()

    def save_state(self):
        """Persists state to JSON."""
        try:
            with open(STATE_FILE, "w") as f:
                f.write(self.state.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"⚠️ [System] Failed to save state: {e}")

    def register_heartbeat(self, status: ClientStatus) -> ClientInstruction:
        """
        The 'Inbox' Check.
        Delivers messages ONLY if the agent is idle.
        """
        agent_id = status.agent_id
        self.state.connected_agents[agent_id] = status

        if agent_id not in self.state.agent_configs:
            self.state.agent_configs[agent_id] = AgentRoutingConfig()

        # 🛑 STOP GAP 1: "Lost Mail" Prevention
        # If the agent is busy or user is typing, DO NOT deliver mail.
        if status.state != "idle":
            # logger.debug(f"⏳ {agent_id} is busy ({status.state}). Holding payload.")
            return ClientInstruction(command="noop")

        # Delivery Logic
        if agent_id in self.pending_payloads:
            payload = self.pending_payloads.pop(agent_id)
            logger.info(f"🚚 [Service] Delivering payload to {agent_id}")
            return ClientInstruction(command="inject_content", content_payload=payload)

        return ClientInstruction(command="noop")

    def _apply_wrapper(self, content: str, agent_id: str, config: AgentRoutingConfig) -> str:
        """Wraps the content if enabled."""
        if not config.use_wrapper:
            return content

        template = WRAPPERS.get(config.wrapper_style, WRAPPERS["Standard"])
        return template.format(
            agent=agent_id,
            persona=config.persona_name,
            content=content
        )

    def submit_content(self, source_id: str, content: str):
        """
        ONE-SHOT ROUTER: Checks for permission to send ONCE, then resets.
        """
        # Safety Check
        if not self.state.is_active or source_id not in self.state.agent_configs:
            return {"status": "ignored"}

        # 1. Get Configuration
        config = self.state.agent_configs[source_id]
        targets = config.target_agents

        if not targets:
            logger.info(f"💾 [Service] Captured from {source_id} (No targets)")
            return {"status": "captured_only"}

        should_send = False

        # --- LOGIC GATES ---
        if config.send_next_only:
            should_send = True
            config.send_next_only = False

            # 🛑 STOP GAP 2: Persistence for Switch Flip
            # We must save immediately so a crash doesn't revert the switch to ON
            self.save_state()
            logger.info(f"🔫 [Service] One-Shot Triggered: {source_id} -> {targets}")

        # --- EXECUTION ---
        if should_send:
            wrapped_content = self._apply_wrapper(content, source_id, config)
            delivery_count = 0

            for target in targets:
                # 🛑 CRITICAL FIX: The Self-Immolation Circuit Breaker
                if target == source_id:
                    logger.warning(f"🛑 [Service] Loop Blocked: {source_id} targeted itself.")
                    continue

                self.pending_payloads[target] = wrapped_content
                delivery_count += 1
                logger.info(f"📨 [Service] Routed: {source_id} -> {target}")

            return {"status": "routed", "targets": targets, "count": delivery_count}

        return {"status": "captured_only", "reason": "switch_off"}

    def set_mode(self, mode: str, active: bool):
        """Updates global operation mode and PERSISTS it."""
        self.state.operation_mode = mode
        self.state.is_active = active
        self.save_state()  # <--- The critical fix
        return {"status": "updated", "mode": mode, "active": active}

    def update_config(self, agent_id: str, new_config: AgentRoutingConfig):
        self.state.agent_configs[agent_id] = new_config
        # 🛑 SAVE TRIGGER: Persist every time config changes
        self.save_state()
        return self.state.agent_configs[agent_id]


relay_service = RelayService()