from typing import Dict, Optional, List, Set
from models import ClientStatus, ClientInstruction, RelaySessionState, AgentRoutingConfig

# --- WRAPPER TEMPLATES ---
WRAPPERS = {
    "Standard": "--- Incoming Transmission: {persona} ({agent}) ---\n\n{content}\n\n--- End Transmission ---",
    "Minimal": "**{persona}:** {content}",
    "XML": "<response from='{agent}' role='{persona}'>\n{content}\n</response>"
}


class RelayService:
    def __init__(self):
        self.state = RelaySessionState()
        self.pending_payloads: Dict[str, str] = {}

    def register_heartbeat(self, status: ClientStatus) -> ClientInstruction:
        agent_id = status.agent_id
        self.state.connected_agents[agent_id] = status

        if agent_id not in self.state.agent_configs:
            self.state.agent_configs[agent_id] = AgentRoutingConfig()

        if agent_id in self.pending_payloads:
            payload = self.pending_payloads.pop(agent_id)
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
        Main Routing Logic with Assembly Queue.
        """
        if not self.state.is_active or source_id not in self.state.agent_configs:
            return {"status": "ignored"}

        # --- 1. CHECK IF THIS MESSAGE BELONGS TO AN ASSEMBLY QUEUE ---
        # Is anyone waiting for THIS source?
        # We iterate all assembly queues to see if 'source_id' is in a 'waiting_for' set
        queue_hit = False

        for recipient, queue_data in self.state.assembly_queues.items():
            if source_id in queue_data['waiting_for']:
                # Found a match! Add to bucket.
                print(f"[Brain] 📥 Assembly: Received part from {source_id} for {recipient}.")

                # Wrap before storing
                config = self.state.agent_configs[source_id]
                wrapped_msg = self._apply_wrapper(content, source_id, config)

                queue_data['parts'].append(wrapped_msg)
                queue_data['waiting_for'].remove(source_id)
                queue_hit = True

                # CHECK: Is the bucket full?
                if not queue_data['waiting_for']:
                    # Release the bundle!
                    combined_msg = "\n\n".join(queue_data['parts'])
                    print(f"[Brain] 📦 Assembly Complete for {recipient}. Delivering bundle.")
                    self.pending_payloads[recipient] = combined_msg
                    del self.state.assembly_queues[recipient]
                break

        if queue_hit:
            return {"status": "assembled", "waiting_for_others": True}

        # --- 2. STANDARD ROUTING (If not caught by assembly) ---
        config = self.state.agent_configs[source_id]
        targets = config.target_agents  # List[str]

        if not targets:
            return {"status": "no_targets"}

        should_send = False
        is_loop_trigger = False

        # Logic Gates
        if config.send_next_only:
            should_send = True
            config.send_next_only = False
            print(f"[Brain] One-Shot fired: {source_id} -> {targets}")

        elif config.auto_stream:
            should_send = True
            print(f"[Brain] Streaming: {source_id} -> {targets}")

        elif config.recursive_loop:
            if self.state.max_iterations > 0:
                should_send = True
                is_loop_trigger = True
                self.state.max_iterations -= 1
                print(
                    f"[Brain] Loop Step: {source_id} -> {targets} (Turns: {self.state.max_iterations})")
            else:
                config.recursive_loop = False  # Brake

        # Execution
        if should_send:
            wrapped_content = self._apply_wrapper(content, source_id, config)

            # Distribute to all targets
            for target in targets:
                if target in self.state.connected_agents:
                    self.pending_payloads[target] = wrapped_content

            # --- 3. CREATE ASSEMBLY EXPECTATION (If Looping) ---
            # If this was a Loop trigger sent to MULTIPLE people,
            # we (Source) expect replies from ALL of them before we move again.
            if is_loop_trigger and len(targets) > 1:
                print(f"[Brain] ⏳ {source_id} is now waiting for replies from {targets}")
                self.state.assembly_queues[source_id] = {
                    'waiting_for': set(targets),  # Uses Set for O(1) lookup
                    'parts': []
                }

            return {"status": "routed", "targets": targets}

        return {"status": "captured_only"}

    def update_config(self, agent_id: str, new_config: AgentRoutingConfig):
        """
        Updates configuration.
        Note: We simplified the 'Link Break' logic for now as multi-target
        makes 'partners' ambiguous. We rely on the global Max Iterations.
        """
        self.state.agent_configs[agent_id] = new_config
        return self.state.agent_configs[agent_id]


relay_service = RelayService()