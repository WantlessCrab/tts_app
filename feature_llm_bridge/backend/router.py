# feature_llm_bridge/backend/router.py

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
# 🛑 FIX: Added AgentRoutingConfig to the import list below
from models import ClientStatus, ClientInstruction, CaptureRequest, AgentRoutingConfig
from service import relay_service
import logging

# 1. Force Standard Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

app = FastAPI(title="LLM Relay Service", version="1.0.0")


# --- SECURITY LAYER (The Fix) ---

@app.middleware("http")
async def add_private_network_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://claude.ai",
        "https://chatgpt.com",
        "https://gemini.google.com",
        "http://localhost:8005",
        "https://localhost:8005"
    ],
    allow_origin_regex="chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "online", "service": "llm-relay"}


# --- ENDPOINTS ---

@app.post("/api/v1/status", response_model=ClientInstruction)
async def heartbeat(status: ClientStatus):
    return relay_service.register_heartbeat(status)


@app.post("/api/v1/capture")
async def capture_content(payload: CaptureRequest):
    """Standardized Ingestion Endpoint."""
    logger.info("🔥 [Router] /capture endpoint HIT!")
    try:
        result = relay_service.submit_content(payload.agent_id, payload.content)
        logger.info(f"✅ [Router] Processed. Result: {result}")
        return result
    except Exception as e:
        logger.error(f"💥 [Router] Crash in /capture: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/admin/state")
async def get_state():
    return {
        "state": relay_service.state,
        "pending_queues": relay_service.pending_payloads
    }


@app.post("/api/v1/admin/config/{agent_id}")
async def update_agent_config(agent_id: str, config: AgentRoutingConfig):
    return relay_service.update_config(agent_id, config)


@app.post("/api/v1/admin/mode")
async def set_mode(mode: str = Body(..., embed=True), active: bool = Body(True, embed=True)):
    return relay_service.set_mode(mode, active)