from __future__ import annotations

import base64
import importlib.util
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .service_health import build_health_response

logger = logging.getLogger(__name__)

app = FastAPI(title="docTR OCR Service")

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from doctr.models import ocr_predictor
        _MODEL = ocr_predictor(pretrained=True)
    return _MODEL


class OCRRequest(BaseModel):
    image_b64: str
    page_width: float
    page_height: float
    page_num: int = 0


class OCRResponse(BaseModel):
    spans: List[Dict[str, Any]]


@app.get("/health")
def health():
    checks = {
        "doctr_dependency": "ok" if importlib.util.find_spec(
            "doctr") else "error: doctr is not installed",
        "model_loaded": "ok" if _MODEL is not None else "not_loaded",
    }
    return build_health_response(
        service="tts_doctr_service",
        role="ocr_api",
        checks=checks,
        status="ok" if checks["doctr_dependency"] == "ok" else "degraded",
    )


@app.post("/ocr", response_model=OCRResponse)
def run_ocr(req: OCRRequest):
    try:
        from doctr.io import DocumentFile

        image_bytes = base64.b64decode(req.image_b64)
        doc = DocumentFile.from_images([image_bytes])
        result = _get_model()(doc)
        spans = _doctr_to_spans(result, req.page_num, req.page_width, req.page_height)
        return OCRResponse(spans=spans)
    except Exception as e:
        logger.error("OCR failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def _row_key(y0: float, font_size: float) -> float:
    return round(y0 / max(1.0, font_size * 0.25), 0)


def _doctr_to_spans(
        result,
        page_num: int,
        page_width: float,
        page_height: float,
) -> List[Dict[str, Any]]:
    spans = []
    if not result.pages:
        return spans

    page = result.pages[0]

    for block_idx, block in enumerate(page.blocks):
        for line_idx, line in enumerate(block.lines):
            words = line.words
            if not words:
                continue

            line_text = " ".join(w.value for w in words)
            span_count = len(words)

            for word_idx, word in enumerate(words):
                text = word.value.strip()
                if not text:
                    continue

                (x0n, y0n), (x1n, y1n) = word.geometry
                x0 = x0n * page_width
                y0 = y0n * page_height
                x1 = x1n * page_width
                y1 = y1n * page_height
                bbox = (x0, y0, x1, y1)
                font_size = round(max(y1 - y0, 1.0) * 0.72, 2)

                spans.append({
                    "raw_text": text,
                    "cleaned_text": None,
                    "bbox": bbox,
                    "font_size": font_size,
                    "bbox_is_valid": True,
                    "bbox_invalid_reason": None,
                    "line_text": line_text,
                    "font": "",
                    "flags": 0,
                    "color": 0,
                    "origin": (x0, y1),
                    "baseline_y": y1,
                    "line_y_band": round(y0, 1),
                    "page_number": page_num + 1,
                    "row_key": _row_key(y0, font_size),
                    "is_line_end": (word_idx == span_count - 1),
                    "span_count_in_line": span_count,
                    "column_index": 0,
                    "paragraph_index": 0,
                    "is_paragraph_start": False,
                    "role": "body",
                    "char_offset": 0,
                    "block_id": block_idx,
                    "line_index": line_idx,
                    "line_id": f"{page_num}:{block_idx}:{line_idx}",
                    "_ocr_source": "doctr",
                })

    return spans