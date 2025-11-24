from typing import Dict, Optional
from models import ClientStatus, ClientInstruction, RelaySessionState, AgentRoutingConfig


class RelayService:
    def __init__(self):
        self.state = RelaySessionState()
        self.pending_payloads: Dict[str, str] = {}

    def register_heartbeat(self, status: ClientStatus) -> ClientInstruction:
        agent_id = status.agent_id
        self.state.connected_agents[agent_id] = status

        # Ensure config exists for this agent
        if agent_id not in self.state.agent_configs:
            self.state.agent_configs[agent_id] = AgentRoutingConfig()

        # Check for work
        if agent_id in self.pending_payloads:
            payload = self.pending_payloads.pop(agent_id)
            return ClientInstruction(command="inject_content", content_payload=payload)

        return ClientInstruction(command="noop")

    def submit_content(self, source_id: str, content: str):
        if not self.state.is_active or source_id not in self.state.agent_configs:
            return {"status": "ignored"}

        config = self.state.agent_configs[source_id]
        target = config.target_agent

        if target == "None" or target not in self.state.connected_agents:
            return {"status": "no_target"}

        # --- LOGIC GATE ---

        should_send = False

        # 1. The "Toggle Next" (One-Shot)
        if config.send_next_only:
            should_send = True
            # Auto-disable after use (The "Toggle" effect)
            config.send_next_only = False
            print(f"[Brain] One-Shot fired: {source_id} -> {target}")

        # 2. The "One-Way Stream"
        elif config.auto_stream:
            should_send = True
            print(f"[Brain] Streaming: {source_id} -> {target}")

        # 3. The "Two-Way Loop"
        elif config.recursive_loop:
            if self.state.max_iterations > 0:
                should_send = True
                self.state.max_iterations -= 1
                print(f"[Brain] Loop Step: {source_id} -> {target}")
            else:
                # Safety Brake
                config.recursive_loop = False

                # Execute
        if should_send:
            self.pending_payloads[target] = content
            return {"status": "routed", "target": target}

        return {"status": "captured_only"}

    def update_config(self, agent_id: str, new_config: AgentRoutingConfig):
        self.state.agent_configs[agent_id] = new_config
        return self.state.agent_configs[agent_id]


relay_service = RelayService()