# feature_llm_bridge/backend/router.py

from fastapi import FastAPI, HTTPException, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from models import ClientStatus, ClientInstruction, RelaySessionState
from service import relay_service
import logging
from models import AgentRoutingConfig

# 1. Force Standard Logging to Console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

app = FastAPI(title="LLM Relay Service", version="1.0.0")


# --- SECURITY FIX: Private Network Access & CORS ---
# We use a custom middleware to ensure EVERY response gets the PNA stamp.
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    # 1. Process the request
    response = await call_next(request)

    # 2. Stamp the "Permission Slip" for Chrome PNA
    response.headers["Access-Control-Allow-Private-Network"] = "true"

    # 3. Explicitly allow the Origin (vs '*' which can be flaky with PNA)
    origin = request.headers.get("origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"

    return response


# Standard CORS Middleware (Still needed for basic handling)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Our custom middleware above refines this
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
    # logger.info(f"❤️ Heartbeat from {status.agent_id}")
    return relay_service.register_heartbeat(status)


@app.post("/api/v1/capture")
async def capture_content(request: Request):
    """
    DEBUG MODE: Raw request inspection.
    """
    logger.info("🔥 [DEBUG] /capture endpoint HIT!")

    try:
        raw_body = await request.json()
        logger.info(f"📦 [DEBUG] Payload received: {str(raw_body)[:100]}...")

        agent_id = raw_body.get("agent_id")
        content = raw_body.get("content")

        if not agent_id or not content:
            logger.error("❌ [DEBUG] Missing agent_id or content")
            raise HTTPException(status_code=400, detail="Missing fields")

        result = relay_service.submit_content(agent_id, content)
        logger.info(f"✅ [DEBUG] Processed. Result: {result}")
        return result

    except Exception as e:
        logger.error(f"💥 [DEBUG] Crash in /capture: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/admin/state")
async def get_state():
    return {
        "state": relay_service.state,
        "pending_queues": relay_service.pending_payloads
    }


@app.post("/api/v1/admin/config/{agent_id}")
async def update_agent_config(agent_id: str, config: AgentRoutingConfig):
    """Updates the routing rules for a specific agent."""
    return relay_service.update_config(agent_id, config)

@app.post("/api/v1/admin/mode")
async def set_mode(mode: str = Body(..., embed=True), active: bool = Body(True, embed=True)):
    relay_service.state.operation_mode = mode
    relay_service.state.is_active = active
    return {"status": "updated", "mode": mode, "active": active}