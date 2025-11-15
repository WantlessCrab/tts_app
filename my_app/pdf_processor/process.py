# ~/TTS/my_app/pdf_processor/process.py
from fastapi.responses import FileResponse
import fitz  # PyMuPDF
import sys
import json
from pathlib import Path
import logging
import re  # For sentence splitting
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks

# ========================================
# Configuration & Setup
# ========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PDFProcessorService")

# Define paths relative to the /workspace
BASE_DIR = Path("/workspace")
CACHE_DIR = BASE_DIR / "pdf_cache"
INPUT_DIR = BASE_DIR / "pdf_input"
OUTPUT_DIR = BASE_DIR / "outputs" / "audiobooks"
CACHE_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Service Setup ---
app = FastAPI(title="PDF Processing Service")

# Create a persistent HTTP client for communicating with tts-service
# The 'app' logic handles startup/shutdown events
client = httpx.AsyncClient(timeout=300.0)
TTS_SERVICE_URL = "http://tts-service:5002/api/tts"
# This is the model specified in your docker-compose.base.yml
TTS_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"


@app.on_event("startup")
async def startup_event():
    # Test connection to TTS service on startup
    try:
        response = await client.get("http://tts-service:5002/")
        response.raise_for_status() # Will raise error on 4xx/5xx
        logger.info(f"Successfully connected to TTS service at http://tts-service:5002")
        # Log snippet of response to confirm it's the demo page (optional)
        # logger.debug(f"TTS service root response snippet: {response.text[:100]}...")
    except Exception as e:
        logger.error(f"Failed to connect to TTS service at http://tts-service:5002: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("HTTP client closed.")


# ========================================
# API Endpoints
# ========================================

@app.post("/api/v1/process/{pdf_filename}")
async def start_pdf_processing(pdf_filename: str, background_tasks: BackgroundTasks):
    """
    Triggers the full PDF-to-Audio pipeline in the background.
    """
    safe_filename = re.sub(r'[^\w\-\.]', '', pdf_filename).strip()
    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    pdf_path = INPUT_DIR / safe_filename
    if not pdf_path.exists():
        logger.warning(f"Process request failed: File not found at {pdf_path}")
        raise HTTPException(status_code=404, detail="PDF not found in input directory")

    # Run the entire pipeline as a background task
    # This makes the endpoint return immediately
    background_tasks.add_task(run_full_pipeline, safe_filename)

    logger.info(f"Accepted job for {safe_filename}. Processing started in background.")
    return {"status": "processing_started", "filename": safe_filename}


@app.get("/api/v1/citation/{book_id}")
async def get_citation(book_id: str, timestamp: float = 0.0):
    """
    Gets citation information for a specific timestamp in an audiobook.
    """
    # CORRECTED: Don't reverse the sanitization
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()
    # Ensure underscores match the Stage 2 output
    safe_book_id_sanitized = safe_book_id.replace(' ', '_')
    citation_filename = safe_book_id_sanitized + '_citation_ready.json'
    citation_path = CACHE_DIR / citation_filename

    if not citation_path.exists():
        # Fallback: scan for partial match
        found_path = None
        for f in CACHE_DIR.glob(f"*{safe_book_id_sanitized}*citation_ready.json"):
            found_path = f
            break
        if not found_path:
            logger.warning(f"Citation file not found for book_id: {safe_book_id}")
            raise HTTPException(status_code=404, detail="Citation data not available")
        citation_path = found_path

    citation_data = get_citation_at_timestamp(citation_path, timestamp)
    if not citation_data:
        raise HTTPException(status_code=404, detail=f"No citation found for timestamp {timestamp}")
    return citation_data


@app.get("/api/v1/document/{pdf_filename}")
async def serve_pdf_document(pdf_filename: str):
    """
    Serves the original source PDF document from the input directory.
    """
    safe_filename = re.sub(r'[^\w\-\.]', '', pdf_filename).strip()

    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    file_path = INPUT_DIR / safe_filename

    # Verify file exists and is actually a file
    if not file_path.exists() or not file_path.is_file():
        logger.warning(f"Document request failed: File not found at {file_path}")
        raise HTTPException(status_code=404, detail="PDF document not found")

    # Optional: Check file size (prevent serving corrupted/huge files)
    file_size = file_path.stat().st_size
    if file_size > 100 * 1024 * 1024:  # 100MB
        logger.error(f"PDF exceeds size limit: {file_size} bytes")
        raise HTTPException(status_code=413, detail="PDF file too large")

    logger.info(f"Serving source PDF: {file_path} ({file_size} bytes)")
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename=\"{safe_filename}\"",
            "Cache-Control": "public, max-age=3600",
            "Accept-Ranges": "bytes"
        }
    )

# ========================================
# Core Processing Pipeline
# (These functions are now called by the API, not __main__)
# ========================================

# --- Phase 1, Step 2: Add Cache Check ---
async def run_full_pipeline(pdf_filename: str):
    """
    The full background processing chain with cache check.
    """
    logger.info(f"Pipeline started for: {pdf_filename}")

    # Derive expected citation path based on the logic in prepare_tts_chunks...
    # We need to read metadata first to get the title for sanitization.
    # This is slightly less efficient but necessary for consistent naming.
    pdf_path = INPUT_DIR / pdf_filename
    raw_cache_file_name = pdf_path.stem + "_raw.json" # Needed if we run Stage 1

    # Attempt to find citation path early - assumes Stage 1 might have run before
    potential_citation_path = None
    try:
        temp_title = pdf_path.stem # Use stem as fallback title
        if (CACHE_DIR / raw_cache_file_name).exists():
             with open(CACHE_DIR / raw_cache_file_name, 'r', encoding='utf-8') as f_raw:
                 raw_data = json.load(f_raw)
                 temp_title = raw_data['metadata'].get('title', pdf_path.stem)

        book_name_sanitized = re.sub(r'[^\w\s-]', '', temp_title).strip().replace(' ', '_')
        potential_citation_path = CACHE_DIR / (book_name_sanitized + '_citation_ready.json')

    except Exception as e:
        logger.warning(f"Could not determine potential citation path early: {e}")
        potential_citation_path = None # Ensure it's None on error

    # E2's Cache Validation Logic (Option A)
    citation_path = None
    if potential_citation_path and potential_citation_path.exists():
        logger.info(f"Citation cache found at {potential_citation_path}, skipping Stage 1 & 2.")
        citation_path = potential_citation_path
    else:
        # Run Stage 1: Process PDF
        logger.info("Citation cache not found or check failed, running Stage 1...")
        raw_cache_path = process_pdf(pdf_filename) # Uses raw_cache_file_name
        if not raw_cache_path:
            logger.error(f"Pipeline HALTED at Stage 1 for: {pdf_filename}")
            return # Stop processing

        # Run Stage 2: Prepare Chunks
        logger.info("Running Stage 2...")
        citation_path = prepare_tts_chunks_with_citations(raw_cache_path)
        if not citation_path:
            logger.error(f"Pipeline HALTED at Stage 2 for: {pdf_filename}")
            return # Stop processing

    # Always run Stage 3: Generate Audio (It skips existing files internally)
    if citation_path:
        logger.info("Proceeding to Stage 3 (Audio Generation)...")
        await generate_audio_streaming(citation_path)
        logger.info(f"Pipeline FINISHED for: {pdf_filename}")
    else:
        # Should not happen if logic is correct, but added as safeguard
        logger.error(f"Pipeline HALTED for {pdf_filename} before Stage 3 due to missing citation path.")


# ADD THIS NEW HELPER FUNCTION
# (This is from E2's proposal)

def extract_text_blocks_with_coords(page):
    """
    Extract text blocks with bounding box coordinates from a PDF page.

    Uses PyMuPDF's get_text("dict") for structured extraction.

    Args:
        page: fitz.Page object

    Returns:
        list: Text blocks with coordinates
    """
    blocks_data = []

    # Get structured text data
    text_dict = page.get_text("dict")

    for block in text_dict["blocks"]:
        # Skip non-text blocks (images, etc)
        if block["type"] != 0:
            continue

        # Process each line in the block
        for line in block["lines"]:
            # Process each span (text run with same formatting)
            for span in line["spans"]:
                # Extract bounding box
                bbox = span["bbox"]  # [x0, y0, x1, y1]

                blocks_data.append({
                    "text": span["text"],
                    "bbox": {
                        "x": round(bbox[0], 2),
                        "y": round(bbox[1], 2),
                        "width": round(bbox[2] - bbox[0], 2),
                        "height": round(bbox[3] - bbox[1], 2)
                    },
                    "font_size": round(span["size"], 2),
                    "font_name": span["font"],
                    "block_type": "span"  # Granularity level
                })

    return blocks_data

# --- Stage 1: Process PDF to Raw JSON ---
# (This function is unchanged from your original)
def process_pdf(pdf_filename: str):
    pdf_path = INPUT_DIR / pdf_filename
    if not pdf_path.exists():
        logger.error(f"Error: File not found: {pdf_path}")
        return None
    cache_file_name = pdf_path.stem + "_raw.json"
    cache_file_path = CACHE_DIR / cache_file_name
    logger.info(f"Stage 1: Processing '{pdf_path.name}'...")
    output_data = {"metadata": {}, "content": []}
    try:
        with fitz.open(pdf_path) as doc:
            meta = doc.metadata or {}
            output_data["metadata"] = {
                "title": meta.get("title", pdf_path.stem),
                "author": meta.get("author", "Unknown"),
                "source_filename": pdf_path.name,
                "total_pages": doc.page_count
            }
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)

                # --- OLD (Keep for backward compatibility) ---
                blocks = page.get_text("blocks", sort=True)
                page_text_chunks = []
                for b_index, b in enumerate(blocks):
                    block_text = b[4].replace("-\n", "").replace("\n", " ").strip()
                    if block_text:
                        page_text_chunks.append(block_text)

                # --- NEW (Add coordinate data) ---
                coordinate_blocks = extract_text_blocks_with_coords(page)

                # --- MODIFIED (Add new key to output) ---
                if page_text_chunks or coordinate_blocks:
                    output_data["content"].append({
                        "page_number": page_num + 1,
                        "text_blocks": page_text_chunks,  # <-- OLD list of strings
                        "coordinate_blocks": coordinate_blocks  # <-- NEW list of dicts
                    })

            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Stage 1 complete. Raw text cache saved to: {cache_file_path}")
        return cache_file_path
    except Exception as e:
        logger.error(f"Error in Stage 1 processing PDF {pdf_path.name}: {e}")
        return None


# --- Stage 2: Prepare Citation-Ready Chunks (Hybrid) ---
# (This function is unchanged from your original)
def prepare_tts_chunks_with_citations(cache_file_path: Path, max_chars=400):
    if not cache_file_path or not cache_file_path.exists():
        logger.error(f"Stage 2 Error: Invalid cache file path: {cache_file_path}")
        return None
    logger.info(f"Starting Stage 2 Enhanced: Preparing citation-ready TTS chunks...")
    with open(cache_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tts_chunks = []
    global_sentence_index = 0
    estimated_total_time = 0.0
    AVG_CHARS_PER_SECOND = 14
    try:
        for page_data in data['content']:
            page_num = page_data['page_number']
            for block_index, block_text in enumerate(page_data['text_blocks']):
                sentences = re.split(r'(?<=[.!?])\s+', block_text)
                current_chunk = []
                current_chunk_chars = 0
                chunk_sentences_data = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence: continue
                    sentence_chars = len(sentence) + 1
                    if current_chunk_chars + sentence_chars < max_chars:
                        current_chunk.append(sentence)
                        current_chunk_chars += sentence_chars
                        chunk_sentences_data.append({
                            'global_index': global_sentence_index,
                            'sentence_in_block': len(chunk_sentences_data),
                            'text': sentence
                        })
                        global_sentence_index += 1
                    else:
                        if current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            est_duration = len(chunk_text) / AVG_CHARS_PER_SECOND
                            tts_chunks.append({
                                'chunk_id': len(tts_chunks), 'text': chunk_text,
                                'page': page_num, 'block_index': block_index + 1,
                                'sentences': chunk_sentences_data,
                                'start_time': estimated_total_time,
                                'duration_seconds': est_duration,
                                'end_time': estimated_total_time + est_duration
                            })
                            estimated_total_time += est_duration
                        current_chunk = [sentence]
                        current_chunk_chars = sentence_chars
                        chunk_sentences_data = [{
                            'global_index': global_sentence_index,
                            'sentence_in_block': 0, 'text': sentence
                        }]
                        global_sentence_index += 1
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    est_duration = len(chunk_text) / AVG_CHARS_PER_SECOND
                    tts_chunks.append({
                        'chunk_id': len(tts_chunks), 'text': chunk_text,
                        'page': page_num, 'block_index': block_index + 1,
                        'sentences': chunk_sentences_data,
                        'start_time': estimated_total_time,
                        'duration_seconds': est_duration,
                        'end_time': estimated_total_time + est_duration
                    })
                    estimated_total_time += est_duration

        # --- REFACTOR ---
        # The book ID must be derived from the title for the citation API to find it
        book_title = data['metadata'].get('title', cache_file_path.stem.replace("_raw", ""))
        book_name_sanitized = re.sub(r'[^\w\s-]', '', book_title).strip().replace(' ', '_')
        # Use the sanitized name for the citation file
        citation_file_name = book_name_sanitized + '_citation_ready.json'
        citation_path = CACHE_DIR / citation_file_name

        output_data = {
            'metadata': data['metadata'],
            'book_id': book_name_sanitized,  # Store the ID we will use
            'processing': {
                'total_chunks': len(tts_chunks),
                'total_sentences': global_sentence_index,
                'total_estimated_duration_seconds': estimated_total_time,
            },
            'chunks': tts_chunks
        }
        with open(citation_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Stage 2 complete. Citation-ready chunks saved to: {citation_path}")
        return citation_path
    except Exception as e:
        logger.error(f"Error in Stage 2 preparing chunks: {e}", exc_info=True)
        return None


# --- Stage 3: Generate Audio (Refactored to use API) ---

async def generate_audio_streaming(citation_json_path: Path, limit=None):
    """
    Stage 3 Streaming: Generate audio chunks one at a time via API, updating a manifest.
    """
    if not citation_json_path.exists():
        logger.error(f"Stage 3 Error: Citation file not found: {citation_json_path}")
        return None

    with open(citation_json_path, 'r') as f:
        data = json.load(f)

    # Use the pre-sanitized book_id from the citation file
    book_name = data.get('book_id', citation_json_path.stem.replace('_citation_ready', ''))
    audio_dir = OUTPUT_DIR / book_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = audio_dir / "manifest.json"
    manifest = {
        "metadata": data['metadata'],
        "book_id": book_name,
        "total_chunks": len(data['chunks']),
        "ready_chunks": []
    }

    # Pre-populate ready_chunks from existing manifest if it exists
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                existing_manifest = json.load(f)
                # Safely get ready_chunks, defaulting to empty list if key missing
                manifest['ready_chunks'] = existing_manifest.get('ready_chunks', [])
        except Exception as e:
            # E2 Fix: Log error and HALT on read failure to prevent overwrite
            logger.error(f"CRITICAL: Failed to read existing manifest at {manifest_path}: {e}")
            logger.error(
                "Halting audio generation for this book to prevent data loss. Fix manifest file manually or delete it to restart.")
            return None  # Stop processing this book

    chunks_to_process = data['chunks'][:limit] if limit else data['chunks']
    logger.info(f"Stage 3: Generating audio for {len(chunks_to_process)} chunks...")

    for chunk in chunks_to_process:
        chunk_id = chunk['chunk_id']
        page = chunk['page']
        audio_filename = f"chunk_{chunk_id:04d}_p{page}.wav"
        audio_path = audio_dir / audio_filename

        if audio_path.exists() or any(c['chunk_id'] == chunk_id for c in manifest['ready_chunks']):
            logger.info(f"Skipping chunk {chunk_id} (already exists or in manifest).")
            # Ensure it IS in the manifest if the file exists but wasn't listed before
            if audio_path.exists() and not any(c['chunk_id'] == chunk_id for c in manifest['ready_chunks']):
                logger.warning(f"Chunk {chunk_id} file exists but was missing from manifest. Adding it now.")
                manifest['ready_chunks'].append({
                    "chunk_id": chunk_id,
                    "filename": audio_filename,
                    "page": page,
                    "text_snippet": chunk['text'][:50] + "...",
                    "start_time": chunk['start_time'],
                    "duration_seconds": chunk['duration_seconds']
                })
                manifest['ready_chunks'].sort(key=lambda c: c['chunk_id'])
                # Re-save manifest immediately to fix inconsistency
                try:
                    with open(manifest_path, 'w') as f_fix:
                        json.dump(manifest, f_fix, indent=2)
                except Exception as e_fix:
                    logger.error(f"Failed to update manifest for skipped chunk {chunk_id}: {e_fix}")
            continue  # Skip TTS generation

        # --- REFACTOR: Replaced subprocess with API call ---
        try:
            logger.info(f"Generating chunk {chunk_id}/{len(chunks_to_process)} via API...")

            # This is the standard API payload for the Coqui TTS server
            params = {
                "text": chunk['text'],
                "speaker_id": "",  # Use speaker_id as per Coqui server.py/demo JS
                "style_wav": "",  # Use style_wav as per Coqui server.py/demo JS
                "language_id": "",  # Use language_id as per Coqui server.py/demo JS
            }

            # Make the POST request, sending data in URL params, not JSON body
            response = await client.post(TTS_SERVICE_URL, params=params, timeout=300.0)
            # --- END CORRECTION ---

            response.raise_for_status()  # Will raise error if (4xx or 5xx)

            # Save the raw audio content
            with open(audio_path, 'wb') as f:
                f.write(response.content)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed chunk {chunk_id}: HTTP Error {e.response.status_code} from TTS service. {e.response.text}")
            continue  # Skip to next chunk
        except Exception as e:
            logger.error(f"Failed chunk {chunk_id}: {e}")
            continue  # Skip to next chunk
        # --- End Refactor ---

        # IMMEDIATELY update manifest when chunk is ready
        manifest['ready_chunks'].append({
            "chunk_id": chunk_id,
            "filename": audio_filename,
            "page": page,
            "text_snippet": chunk['text'][:50] + "...",
            "start_time": chunk['start_time'],
            "duration_seconds": chunk['duration_seconds']
        })
        manifest['ready_chunks'].sort(key=lambda c: c['chunk_id'])

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Chunk {chunk_id} marked as ready in manifest.")

    logger.info(f"Stage 3 complete. Audio generation finished. Files saved to: {audio_dir}")
    return audio_dir


# --- Citation Lookup Function (For API) ---
def get_citation_at_timestamp(citation_json_path: Path, timestamp_seconds: float):
    if not citation_json_path.exists():
        logger.error(f"Citation file not found: {citation_json_path}")
        return None
    with open(citation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for chunk in data['chunks']:
        if chunk['start_time'] <= timestamp_seconds < chunk['end_time']:
            time_into_chunk = timestamp_seconds - chunk['start_time']
            progress_ratio = time_into_chunk / chunk['duration_seconds'] if chunk['duration_seconds'] > 0 else 0
            sentence_index = int(progress_ratio * len(chunk['sentences']))
            sentence_index = min(sentence_index, len(chunk['sentences']) - 1)
            sentence = chunk['sentences'][sentence_index]
            metadata = data['metadata']
            return {
                'citation': (
                    f"{metadata.get('author', 'Unknown')} - {metadata.get('title', 'Unknown Title')}, "
                    f"p.{chunk['page']}, ¶{chunk['block_index']}, sent.{sentence['sentence_in_block'] + 1}"
                ),
                'timestamp': f"{int(timestamp_seconds // 60)}:{int(timestamp_seconds % 60):02d}",
                'page': chunk['page'],
                'block': chunk['block_index'],
                'sentence_in_block': sentence['sentence_in_block'] + 1,
                'sentence_text': sentence['text'][:100] + "..." if len(sentence['text']) > 100 else sentence['text'],
            }
    logger.warning(f"No chunk found for timestamp: {timestamp_seconds}")
    return None