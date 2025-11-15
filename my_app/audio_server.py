# ~/TTS/my_app/audio_server.py
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from pathlib import Path
import logging
import json
import re
import httpx
from fastapi import Query
from typing import Optional

# ========================================
# Configuration & Setup
# ========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AudioServer")

app = FastAPI(title="TTS Audio Server")

APP_DIR = Path(__file__).parent.resolve()
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
OUTPUT_DIR = APP_DIR.parent / "outputs"
OBSIDIAN_DIR = APP_DIR.parent / "obsidian_audio"
AUDIOBOOKS_DIR = OUTPUT_DIR / "audiobooks"
PDF_CACHE_DIR = Path("/workspace/pdf_cache")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

AUDIO_SOURCES = {
    "audiobooks": AUDIOBOOKS_DIR,        # /workspace/outputs/audiobooks
    "obsidian": OBSIDIAN_DIR,          # /workspace/obsidian_audio
    "standalone": OUTPUT_DIR           # /workspace/outputs (for non-audiobook files)
}
# Use lowercase default consistent with keys
DEFAULT_AUDIO_SOURCE_NAME = "audiobooks"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Add client for pdf-service ---
client = httpx.AsyncClient(timeout=30.0)
PDF_SERVICE_URL = "http://pdf-service:8001"

@app.on_event("startup")
async def startup_event():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OBSIDIAN_DIR.mkdir(parents=True, exist_ok=True) # Ensure Obsidian dir exists
    AUDIOBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        await client.get(f"{PDF_SERVICE_URL}/docs")
        logger.info(f"Successfully connected to PDF service at {PDF_SERVICE_URL}")
    except Exception as e:
        logger.error(f"Failed to connect to PDF service at {PDF_SERVICE_URL}: {e}")

    logger.info(f"Serving audio from defined sources: {list(AUDIO_SOURCES.keys())}") # Updated log message
    logger.info("Audio server started successfully.")

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("HTTP client closed.")

# ========================================
# Frontend Endpoints
# ========================================

@app.get("/", response_class=HTMLResponse)
async def get_player_interface():
    player_html_path = TEMPLATES_DIR / "player.html"
    if not player_html_path.exists():
        logger.error(f"FATAL: player.html not found at {player_html_path}")
        raise HTTPException(status_code=500, detail="Player interface file missing")
    return FileResponse(player_html_path)

# ========================================
# API Endpoints
# ========================================

# --- Step 1.4: Modified Endpoint to Handle Source Parameter ---
@app.get("/api/list_audio")
async def list_audio_files(source: Optional[str] = Query(DEFAULT_AUDIO_SOURCE_NAME)):
    """Lists audio files from the specified source directory."""

    # Validate the source name
    if source not in AUDIO_SOURCES:
        logger.warning(f"Invalid source requested in list_audio: {source}")
        raise HTTPException(status_code=400, detail=f"Invalid audio source specified. Valid sources: {list(AUDIO_SOURCES.keys())}")

    target_directory = AUDIO_SOURCES[source]
    logger.info(f"Listing audio files from source '{source}' at path: {target_directory}")

    files = []
    if target_directory.exists():
        try:
            # Only get WAV files directly within the target directory
            for f in target_directory.glob("*.wav"):
                if f.is_file():
                    files.append({
                        "name": f.name,
                        "path": str(f), # Internal path, might not be needed by frontend
                        "size_bytes": f.stat().st_size,
                        "type": source # Return the source type
                    })
        except Exception as e:
            logger.error(f"Error scanning directory {target_directory}: {e}")
            # Don't raise HTTPException here, just return empty list or partial results

    return {"files": files, "source": source} # Also return the source used

# --- Step 1.5: Modified Endpoint to Handle Source Parameter ---

@app.get("/api/audio/{filename}")
async def serve_audio_file(filename: str, source: Optional[str] = Query(DEFAULT_AUDIO_SOURCE_NAME)):
    """Serves a specific audio file from the specified source directory."""

    if ".." in filename or filename.startswith("/"):
        logger.warning(f"Blocked invalid filename request: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Validate the source name
    if source not in AUDIO_SOURCES:
        logger.warning(f"Invalid source requested in serve_audio: {source}")
        raise HTTPException(status_code=400, detail=f"Invalid audio source specified. Valid sources: {list(AUDIO_SOURCES.keys())}")

    target_directory = AUDIO_SOURCES[source]
    file_path = target_directory / filename

    logger.info(f"Request received for file '{filename}' from source '{source}' at path: {file_path}")

    if file_path.exists() and filename.lower().endswith('.wav'):
        logger.info(f"Serving file: {file_path}")
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes"}
        )

    logger.warning(f"File not found: {file_path}")
    raise HTTPException(status_code=404, detail="Audio file not found in the specified source")

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Placeholder for future live-streaming playback."""
    await websocket.accept()
    logger.info("WebSocket connection established (placeholder)")
    try:
        await websocket.send_json({"status": "ready", "note": "Streaming endpoint placeholder - not yet functional"})
        await asyncio.sleep(1)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed")


# --- Replaced direct import with API call ---
@app.get("/api/audiobook/{book_id}/citation")
async def get_citation_for_timestamp(book_id: str, timestamp: float = 0.0):
    """
    Get citation information by proxying the request to the pdf-service.
    """
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()

    try:
        # Forward the request to the pdf-service
        api_url = f"{PDF_SERVICE_URL}/api/v1/citation/{safe_book_id}"
        response = await client.get(api_url, params={"timestamp": timestamp})

        # Pass the response (success or error) back to the client
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"Error getting citation from pdf-service: {e.response.status_code}")
        # Pass the error detail from the downstream service
        detail = e.response.json().get("detail", "Failed to retrieve citation")
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        logger.error(f"Error getting citation: {e}")
        raise HTTPException(status_code=500, detail="Citation system not available")


@app.get("/api/audiobooks")
async def list_audiobooks():
    """Lists all available audiobooks with their processing status"""
    books = []
    if not AUDIOBOOKS_DIR.exists():
        return {"audiobooks": books}

    for book_dir in AUDIOBOOKS_DIR.iterdir():
        if book_dir.is_dir():
            manifest_path = book_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        total_chunks = manifest.get('total_chunks', 0)
                        ready_chunks = len(manifest.get('ready_chunks', []))
                        books.append({
                            "book_id": book_dir.name,
                            "title": manifest['metadata'].get('title', book_dir.name),
                            "author": manifest['metadata'].get('author', 'Unknown'),
                            "source_file": manifest['metadata'].get('source_filename', ''),
                            "total_chunks": total_chunks,
                            "ready_chunks": ready_chunks,
                            "is_complete": ready_chunks == total_chunks and total_chunks > 0
                        })
                except Exception as e:
                    logger.error(f"Failed to read manifest for {book_dir.name}: {e}")
    return {"audiobooks": books}


@app.get("/api/audiobook/{book_id}/status")
async def get_audiobook_status(book_id: str):
    """Get detailed status and chunk list for a specific audiobook"""
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()
    manifest_path = AUDIOBOOKS_DIR / safe_book_id / "manifest.json"

    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Audiobook '{book_id}' not found")

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        total_chunks = manifest.get('total_chunks', 1)
        ready_chunks_list = manifest.get('ready_chunks', [])
        progress = (len(ready_chunks_list) / total_chunks) * 100 if total_chunks > 0 else 0

        return {
            "book_id": safe_book_id,
            "metadata": manifest['metadata'],
            "total_chunks": manifest.get('total_chunks', 0),
            "ready_chunks": ready_chunks_list,
            "progress_percentage": round(progress, 1),
            "is_complete": len(ready_chunks_list) == total_chunks and total_chunks > 0
        }
    except Exception as e:
        logger.error(f"Error reading manifest for {book_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read audiobook data")


@app.get("/api/pdf/{pdf_filename}")
async def proxy_serve_pdf(pdf_filename: str):
    """
    Proxies request for source PDF document to the pdf-service.
    FIXED: Uses client.get() for small files to avoid stream closure bugs.
    """
    safe_filename = re.sub(r'[^\w\-\.]', '', pdf_filename).strip()

    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    try:
        api_url = f"{PDF_SERVICE_URL}/api/v1/document/{safe_filename}"
        logger.info(f"Proxying PDF request via GET: {api_url}")

        # --- THE FIX ---
        # 1. Use client.get() to download the entire file (it's small)
        response = await client.get(api_url, timeout=30.0)

        # Propagate error from pdf-service if it occurs
        response.raise_for_status()

        # 2. Return a standard Response with the full content
        return Response(
            content=response.content,  # Send all content at once
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/pdf"),
            headers={
                k: v for k, v in response.headers.items()
                if k.lower() in [
                    'content-disposition', 'content-length', 'etag',
                    'accept-ranges', 'last-modified', 'cache-control'
                ]
            }
        )
        # --- END FIX ---

    except httpx.HTTPStatusError as e:
        logger.error(f"Error proxying PDF from pdf-service: {e.response.status_code}")
        try:
            detail = e.response.json().get("detail", "Failed to retrieve PDF")
        except:
            detail = f"Failed to retrieve PDF (HTTP {e.response.status_code})"
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        logger.error(f"Error proxying PDF: {e}")
        raise HTTPException(status_code=500, detail="PDF service not available")

@app.get("/api/audiobook/{book_id}/play/{chunk_filename}")
async def serve_audiobook_chunk(book_id: str, chunk_filename: str):
    """Serve a specific audio chunk from an audiobook"""
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()
    safe_filename = re.sub(r'[^\w\-\.]', '', chunk_filename).strip()

    if not safe_filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    file_path = AUDIOBOOKS_DIR / safe_book_id / safe_filename

    if not file_path.exists():
        logger.warning(f"Audio file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Audio chunk not found")

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )


@app.get("/api/available_pdfs")
async def list_available_pdfs():
    """List PDFs available for processing"""
    pdf_input_dir = Path("/workspace/pdf_input")
    pdfs = []
    if pdf_input_dir.exists():
        for pdf_file in pdf_input_dir.glob("*.pdf"):
            pdfs.append({
                "filename": pdf_file.name,
                "size_bytes": pdf_file.stat().st_size,
                "size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2)
            })
    return {"available_pdfs": pdfs}


# --- Replaced subprocess with API call ---
@app.post("/api/process_pdf")
async def start_pdf_processing(filename: str):
    """
    Trigger PDF processing pipeline by proxying the request to pdf-service.
    """
    safe_filename = re.sub(r'[^\w\-\.]', '', filename).strip()
    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    try:
        # Forward the request to the pdf-service
        api_url = f"{PDF_SERVICE_URL}/api/v1/process/{safe_filename}"
        response = await client.post(api_url)

        # Pass the response (success or error) back to the client
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"Error triggering processing: {e.response.status_code}")
        detail = e.response.json().get("detail", "Failed to start processing")
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        logger.error(f"Failed to start processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")

# --- Step 1.2: New Endpoint to List Sources ---
@app.get("/api/audio_sources")
async def list_audio_sources():
    """Returns a list of available audio source names."""
    return {"sources": list(AUDIO_SOURCES.keys())}

