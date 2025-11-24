from typing import Dict, Optional
from models import ClientStatus, ClientInstruction, RelaySessionState, AgentRoutingConfig


class RelayService:
    def __init__(self):
        self.state = RelaySessionState()
        self.pending_payloads: Dict[str, str] = {}

    def register_heartbeat(self, status: ClientStatus) -> ClientInstruction:
        agent_id = status.agent_id
        self.state.connected_agents[agent_id] = status

        # Ensure config exists
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

        should_send = False

        # 1. One-Shot
        if config.send_next_only:
            should_send = True
            config.send_next_only = False
            print(f"[Brain] One-Shot fired: {source_id} -> {target}")

        # 2. Streaming
        elif config.auto_stream:
            should_send = True
            print(f"[Brain] Streaming: {source_id} -> {target}")

        # 3. Loop
        elif config.recursive_loop:
            if self.state.max_iterations > 0:
                should_send = True
                self.state.max_iterations -= 1
                print(
                    f"[Brain] Loop Step: {source_id} -> {target} (Turns left: {self.state.max_iterations})")
            else:
                config.recursive_loop = False
                print(f"[Brain] 🛑 Max turns reached. Stopping {source_id}.")

        if should_send:
            self.pending_payloads[target] = content
            return {"status": "routed", "target": target}

        return {"status": "captured_only"}

    def update_config(self, agent_id: str, new_config: AgentRoutingConfig):
        """
        Updates configuration and handles 'Linked State' logic.
        """
        # 1. Get the previous state to detect changes
        old_config = self.state.agent_configs.get(agent_id, AgentRoutingConfig())

        # 2. Apply the new state
        self.state.agent_configs[agent_id] = new_config

        # 3. THE SYNC LOGIC: Did we just turn OFF the loop?
        if old_config.recursive_loop and not new_config.recursive_loop:
            target_id = new_config.target_agent

            # Check if target exists
            if target_id in self.state.agent_configs:
                target_config = self.state.agent_configs[target_id]

                # Check for MUTUAL PAIRING (Are they linked?)
                # Condition 1: Target is looping
                # Condition 2: Target is pointing back at us
                if (target_config.recursive_loop and
                        target_config.target_agent == agent_id):
                    print(
                        f"[Brain] 🔗 Link broken by {agent_id}. Auto-stopping partner: {target_id}.")
                    target_config.recursive_loop = False

        return self.state.agent_configs[agent_id]


relay_service = RelayService()