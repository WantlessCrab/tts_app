# my_app/convert/validation.py
"""
Convert Service — PDF quality gate.
"""

import logging
from pathlib import Path

logger = logging.getLogger("ConvertService")

# ─────────────────────────────────────────────
# Quality Gate Thresholds
# ─────────────────────────────────────────────

MIN_PDF_SIZE_BYTES = 1024  # < 1KB = empty/failed conversion
MIN_CHAR_COUNT = 50  # fewer chars = blank or near-blank
MAX_GARBLE_RATIO = 0.15  # >15% non-printable = encoding failure
MAX_GLYPH_RATIO = 0.10  # >10% replacement glyphs (□ / \ufffd)


def _quality_gate(pdf_bytes: bytes, source_name: str = "", trace_id: str = None) -> list:
    """
    Run quality checks on a converted PDF. Synchronous — called via asyncio.to_thread.
    Returns a list of issue strings — empty list = pass.

    PyMuPDF is not async-safe. Caller must wrap with asyncio.to_thread.

    Checks:
      1. File size threshold        — catches empty / failed conversions
      2. Page count > 0             — catches zero-page outputs
      3. Character count threshold  — catches blank conversions
      4. Garbled character ratio    — catches encoding failures
      5. Replacement glyph ratio    — catches font failures (□ / U+FFFD)
      6. Page count collapse        — catches reflow failures on multi-section docs
    """
    issues = []

    # Check 1 — File size
    if len(pdf_bytes) < MIN_PDF_SIZE_BYTES:
        issues.append(f"file too small: {len(pdf_bytes)} bytes")
        return issues

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count

        # Check 2 — Page count
        if page_count == 0:
            issues.append("zero pages")
            doc.close()
            return issues

        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        # Check 3 — Character count
        if len(full_text.strip()) < MIN_CHAR_COUNT:
            issues.append(f"insufficient text: {len(full_text.strip())} chars")

        # Check 4 — Garbled character ratio
        if full_text:
            non_printable = sum(
                1 for c in full_text
                if not c.isprintable() and c not in "\n\r\t "
            )
            garble_ratio = non_printable / max(len(full_text), 1)
            if garble_ratio > MAX_GARBLE_RATIO:
                issues.append(f"high garble ratio: {garble_ratio:.2%}")

        # Check 5 — Replacement glyph ratio (□ U+25A1, replacement char U+FFFD)
        if full_text:
            glyph_count = full_text.count("\u25a1") + full_text.count("\ufffd")
            glyph_ratio = glyph_count / max(len(full_text), 1)
            if glyph_ratio > MAX_GLYPH_RATIO:
                issues.append(
                    f"high replacement glyph ratio: {glyph_ratio:.2%} ({glyph_count} glyphs)"
                )

        # Check 6 — Page count collapse
        if page_count == 1 and source_name:
            ext = Path(source_name).suffix.lower()
            if ext in {".docx", ".odt", ".pptx", ".odp"}:
                if len(full_text.strip()) > 500:
                    issues.append(
                        f"possible reflow collapse: {page_count} page from {ext} "
                        f"with {len(full_text.strip())} chars"
                    )

    except ImportError:
        logger.warning(
            f"PyMuPDF not available for quality gate — skipping deep checks [trace_id={trace_id}]")
    except Exception as e:
        logger.warning(f"Quality gate error for {source_name}: {e} [trace_id={trace_id}]")

    if issues:
        logger.warning(f"Quality gate [{source_name}]: FAIL — {issues} [trace_id={trace_id}]")
    else:
        logger.info(f"Quality gate [{source_name}]: PASS [trace_id={trace_id}]")

    return issues