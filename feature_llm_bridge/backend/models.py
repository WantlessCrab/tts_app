from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict


# --- API Contracts ---

class ClientStatus(BaseModel):
    agent_id: str
    state: Literal["idle", "processing", "awaiting_input"]
    active_url: str
    tab_id: int


class ClientInstruction(BaseModel):
    command: Literal["noop", "inject_content"]
    target_agent_id: Optional[str] = None
    content_payload: Optional[str] = None


# --- NEW: Per-Agent Routing Configuration ---

class AgentRoutingConfig(BaseModel):
    target_agent: str = "None"  # Who do I talk to?
    send_next_only: bool = False  # The "One-Shot" Toggle
    auto_stream: bool = False  # The "One-Way" Toggle
    recursive_loop: bool = False  # The "Two-Way" Toggle


class RelaySessionState(BaseModel):
    # We map "Claude" -> His specific Rules
    agent_configs: Dict[str, AgentRoutingConfig] = {}
    connected_agents: Dict[str, ClientStatus] = {}
    max_iterations: int = 10
    is_active: bool = True