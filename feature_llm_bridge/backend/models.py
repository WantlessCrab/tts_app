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


# --- NEW: Enhanced Routing & Wrappers ---

class AgentRoutingConfig(BaseModel):
    # Multi-Target: List of agent IDs (e.g. ["Gemini", "ChatGPT"])
    target_agents: List[str] = []

    # Routing Toggles
    send_next_only: bool = False
    auto_stream: bool = False
    recursive_loop: bool = False

    # Wrapper Settings
    use_wrapper: bool = False
    persona_name: str = "AI Assistant"  # e.g. "Lead Coder"
    wrapper_style: str = "Standard"  # "Standard", "Minimal", "XML"


class RelaySessionState(BaseModel):
    agent_configs: Dict[str, AgentRoutingConfig] = {}
    connected_agents: Dict[str, ClientStatus] = {}
    max_iterations: int = 10
    is_active: bool = True

    # The "Waiting Room" for Assembly
    # Key: Recipient (Who is waiting?), Value: Dict of parts
    assembly_queues: Dict[str, Dict] = {}