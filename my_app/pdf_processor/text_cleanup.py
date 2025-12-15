# ~/TTS/my_app/pdf_processor/text_cleanup.py

"""
Unified text filtering and cleanup for PDF-to-TTS pipeline.
All content filtering decisions live here.

Architecture Layers:
    Layer 1: Block-level filters (The Eyes & Stomach)
    Layer 2: Span-level filters (The Skin & Shield)
    Layer 3: Text reconstruction (The Fingers)
    Layer 4: Sentence segmentation (The Brain)
    Layer 5: TTS sanitization (Output Gate)

### NOTE (Lead/E2 Canonical Scope):
This module must:
- Preserve geometry fidelity (span order, positions)
- Preserve author-intended structure (no destructive filtering)
- Support academic legal / research / textbook content
- Maintain perfect highlight sync (span_map integrity)
- Only perform moderate cleanup (no rewriting)

The annotations below indicate where small changes are needed.
"""

import re
import unicodedata
import logging

# External dependencies
import fitz  # PyMuPDF - needed for geometry operations
import pysbd
import ftfy

logger = logging.getLogger("TextCleanup")

# ============================================
# TUNABLE CONSTANTS
# ============================================

# Gap Detection (The Fingers)
GAP_THRESHOLD_X = 0.2
GAP_THRESHOLD_Y = 0.5
HEADER_FONT_RATIO = 1.3
# NOTE: HEADER_FONT_RATIO will be revised. See FINDING 2.

# Header/Footer Detection
HEADER_FOOTER_THRESHOLD = 0.75
PAGE_EDGE_ZONE = 0.10

# Span Filtering (The Shield)
MIN_FONT_SIZE_FOOTNOTE = 7
MIN_TEXT_LENGTH = 1
VALID_SINGLE_CHARS = set('aAIi.!?,:;-—')

# Caption Patterns
CAPTION_PATTERNS = [
    r'^(Figure|Table|Fig\.)\s*\d+',
    r'^(Exhibit|Chart|Graph|Diagram)\s*\d+',
]

# Meta-content Patterns (Candidate for removal/refinement)
META_CONTENT_PATTERNS = [
    # Many of these are TOO broad.
    # SEE FINDING 4 — these should be removed or made doc-type-specific.
    r'sample document',
    r'figure captions',
    r'sidenotes are shown',
    r'css\d*\.pub',
    r'page-based',
    # r'for more examples',  # REMOVE for safety
    # r'see appendix',        # REMOVE for safety
]

FOOTNOTE_MARKER_PATTERN = r'^[\d\*†‡§¶]+$'


# ============================================
# LAYER 1: BLOCK-LEVEL FILTERS (The Eyes & Stomach)
# ============================================

def detect_header_footer_bands(doc, threshold: float = None) -> set:
    """
    ### NOTE (FINDING 7):
    - Approve as generally safe, but multi-column or repeated headings can create edge cases.
    - Future improvement: use text-similarity or detect repeated short phrases.

    Current implementation is acceptable for now.
    """

    if threshold is None:
        threshold = HEADER_FOOTER_THRESHOLD

    y_counts = {}
    total_pages = doc.page_count

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]
        seen_y = set()

        for block in blocks:
            if block["type"] != 0:
                continue

            block_y = block["bbox"][1]
            y_rounded = round(block_y, -1)

            # Heavier weighting near page edges
            if block_y < page_height * PAGE_EDGE_ZONE or block_y > page_height * (
                    1 - PAGE_EDGE_ZONE):
                y_rounded = round(block_y, -1)

            if y_rounded not in seen_y:
                seen_y.add(y_rounded)
                y_counts[y_rounded] = y_counts.get(y_rounded, 0) + 1

    banned = {y for y, count in y_counts.items() if count / total_pages >= threshold}

    if banned:
        logger.info(f"Y-Histogram: Filtering {len(banned)} bands on >{threshold * 100}% pages")
    return banned


def should_exclude_block(block: dict, exclusion_bands: set, page_tables: list) -> bool:
    """
    ### NOTE:
    Table exclusion logic is correct.
    Header/Footer filtering is correct (minor future refinements advised).
    """

    if exclusion_bands:
        block_y = round(block["bbox"][1], -1)
        if block_y in exclusion_bands:
            return True

    if page_tables:
        bbox = fitz.Rect(block["bbox"])
        for table_rect in page_tables:
            if bbox.intersects(table_rect):
                intersect_area = (bbox & table_rect).get_area()
                block_area = bbox.get_area()
                if block_area > 0 and (intersect_area / block_area) > 0.5:
                    return True

    return False


# ============================================
# LAYER 2: SPAN-LEVEL FILTERS (The Skin & Shield)
# ============================================

def clean_span_text(raw_text: str) -> str:
    """
    The Skin: Clean and normalize span text.

    Operations:
        1. Fix mojibake/encoding issues (ftfy)
        2. Unicode normalization (NFKC)
        3. Fix camelCase joins (missing spaces)
        4. Collapse multiple spaces (NEW - safe at span level)
        5. Fix space before punctuation within span (NEW)

    Args:
        raw_text: Raw text extracted from PDF span

    Returns:
        Cleaned text string (may be empty)

    Status: ✅ APPROVED
        - All operations are span-local (no cross-span effects)
        - Safe for span_map construction
    """
    if not raw_text:
        return ""

    # Fix mojibake and encoding issues
    clean = ftfy.fix_text(raw_text)

    # Unicode normalization
    clean = unicodedata.normalize('NFKC', clean)

    # Fix camelCase joins (e.g., "corpuscleSebaceous" → "corpuscle Sebaceous")
    clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)

    # NEW: Collapse multiple spaces within span
    clean = re.sub(r' +', ' ', clean)

    # NEW: Fix space before punctuation within span
    # e.g., "word ." → "word." (only if both are in same span)
    clean = re.sub(r'\s+([.,;:!?])', r'\1', clean)

    return clean


def is_valid_span(clean_text: str, span: dict, figure_rects: list = None) -> bool:
    """
    The Shield: Determine if a span should be included in output.

    Args:
        clean_text: Already-cleaned text from clean_span_text()
        span: PyMuPDF span dict with 'size', 'bbox' keys
        figure_rects: Optional list of figure regions for context-aware filtering

    Returns:
        True if span is valid content, False to filter
    """
    if not clean_text or not clean_text.strip():
        return False

    text = clean_text.strip()

    # --- Preserve full figure captions (E4 Proposal #2) ---
    # "Figure 5. The diagram shows..." = KEEP
    if re.match(r'^(Figure|Fig\.|Table)\s*\d+[\.:]\s*\S', text):
        return True

    # --- Filter bare caption references ---
    # "Figure 5" alone (no content after) = FILTER
    for pattern in CAPTION_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return False

    # --- Footnote markers ---
    font_size = span.get("size", 12)
    if font_size < MIN_FONT_SIZE_FOOTNOTE and re.match(FOOTNOTE_MARKER_PATTERN, text):
        return False

    # --- Single character noise ---
    if len(text) == MIN_TEXT_LENGTH and text not in VALID_SINGLE_CHARS:
        return False

    # --- Diagram labels (context-aware) ---
    if ENABLE_DIAGRAM_LABEL_FILTER:
        if _is_diagram_labels(text, span, figure_rects):
            return False

    return True


def _is_meta_content(text: str) -> bool:
    """
    ### NOTE:
    Many patterns should be removed per FINDING 4.
    """
    for pattern in META_CONTENT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ============================================
# LAYER 2.5: Diagram and Table Handling
# ============================================

# Add to TUNABLE CONSTANTS section:
DIAGRAM_LABEL_MIN_WORDS = 3  # Minimum words to even consider
DIAGRAM_LABEL_MIN_WORDS_NO_CONTEXT = 8  # Stricter when no figure context
DIAGRAM_LABEL_CAP_RATIO = 0.7  # >70% capitalized words
DIAGRAM_LABEL_MAX_FONT = 10  # Labels typically < 10pt


def _is_diagram_labels(text: str, span: dict = None, figure_rects: list = None) -> bool:
    """
    Detect diagram labels — text fragments labeling parts of figures.

    Strategy:
        1. If figure_rects available: Only filter text INSIDE/NEAR figures
        2. If no figure context: Use very conservative heuristics
        3. Never filter text with prose indicators or punctuation

    Args:
        text: Text content to analyze
        span: Span dict with 'bbox', 'size' keys
        figure_rects: List of fitz.Rect from detect_figure_regions()

    Returns:
        True if this is a diagram label (should be filtered)
        False if this is content (should be kept)

    Design Principle:
        False negatives (keeping some labels) are acceptable.
        False positives (removing real content) are NOT acceptable.
    """
    words = text.split()

    # --- Gate 1: Minimum length ---
    # Very short text can't be reliably classified
    if len(words) < DIAGRAM_LABEL_MIN_WORDS:
        return False

    # --- Gate 2: Punctuation = Prose ---
    # Diagram labels are bare words; sentences have punctuation
    if re.search(r'[.,;:!?]', text):
        return False

    # --- Gate 3: Prose indicators = Content ---
    prose_indicators = r'\b(is|are|was|were|has|have|the|a|an|and|or|but|in|on|at|to|for|of|with|that|this|which)\b'
    if re.search(prose_indicators, text.lower()):
        return False

    # --- Gate 4: Capitalization ratio ---
    cap_count = sum(1 for w in words if w and w[0].isupper())
    cap_ratio = cap_count / len(words)
    if cap_ratio < DIAGRAM_LABEL_CAP_RATIO:
        return False

    # --- Gate 5: Figure proximity (THE KEY GATE) ---
    if figure_rects and span and 'bbox' in span:
        span_rect = fitz.Rect(span['bbox'])
        near_figure = any(span_rect.intersects(fr) for fr in figure_rects)

        if not near_figure:
            # Text is NOT near any figure — this is body content, KEEP IT
            return False

        # Text IS near a figure — apply font size check
        font_size = span.get('size', 12)
        if font_size > DIAGRAM_LABEL_MAX_FONT:
            # Large text near figure = caption or heading, KEEP IT
            return False

        # Small text + near figure + all caps + no prose = LABEL
        return True

    # --- No figure context: Ultra-conservative mode ---
    # Without knowing where figures are, we risk false positives
    # Only filter VERY obvious cases
    if len(words) < DIAGRAM_LABEL_MIN_WORDS_NO_CONTEXT:
        return False

    if span:
        font_size = span.get('size', 12)
        if font_size >= 9:  # Only tiny text
            return False

    return True


def detect_figure_regions(page) -> list:
    """
    Detect regions of the page containing figures/diagrams.

    Scans for:
        1. Embedded images (photos, diagrams)
        2. Vector drawings (charts, flowcharts, anatomical illustrations)

    Returns:
        List of fitz.Rect objects representing figure zones
        (expanded slightly to catch adjacent labels)

    Usage:
        Called in process.py alongside table detection.
        Passed to is_valid_span for context-aware filtering.
    """
    figure_rects = []

    # --- Detection Method 1: Embedded Images ---
    try:
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
                for rect in rects:
                    if rect.is_empty or rect.is_infinite:
                        continue
                    # Expand by 15pt to catch labels positioned just outside
                    expanded = rect + (-15, -15, 15, 15)
                    figure_rects.append(expanded)
            except Exception:
                pass  # Some images don't have extractable rects
    except Exception:
        pass

    # --- Detection Method 2: Vector Drawings ---
    # Drawings include: lines, curves, rectangles (often used in diagrams)
    try:
        drawings = page.get_drawings()
        if drawings:
            # Group drawings by proximity to find diagram clusters
            drawing_rects = []
            for d in drawings:
                rect = fitz.Rect(d["rect"])
                # Filter tiny drawings (bullets, underlines)
                if rect.width > 30 and rect.height > 30:
                    drawing_rects.append(rect)

            # Merge nearby drawing rects into figure zones
            if drawing_rects:
                merged = _merge_nearby_rects(drawing_rects, gap_threshold=20)
                for rect in merged:
                    # Only keep significant figure regions
                    if rect.width > 80 and rect.height > 80:
                        expanded = rect + (-15, -15, 15, 15)
                        figure_rects.append(expanded)
    except Exception:
        pass

    return figure_rects


def _merge_nearby_rects(rects: list, gap_threshold: float = 20) -> list:
    """
    Merge rectangles that are within gap_threshold of each other.

    This groups scattered drawing elements (arrows, boxes, lines)
    into coherent figure regions.
    """
    if not rects:
        return []

    # Simple greedy merge
    merged = []
    used = set()

    for i, rect1 in enumerate(rects):
        if i in used:
            continue

        # Start with this rect
        current = rect1
        used.add(i)

        # Keep merging until no more overlaps
        changed = True
        while changed:
            changed = False
            for j, rect2 in enumerate(rects):
                if j in used:
                    continue
                # Expand current slightly to check for proximity
                expanded = current + (-gap_threshold, -gap_threshold,
                                      gap_threshold, gap_threshold)
                if expanded.intersects(rect2):
                    # Merge: create bounding box of both
                    current = current | rect2  # Union operator
                    used.add(j)
                    changed = True

        merged.append(current)

    return merged


# ============================================
# LAYER 3 & 4: TEXT RECONSTRUCTION & SEGMENTATION
# ============================================

def segment_page_text(coordinate_blocks: list) -> list:
    """
    ### CRITICAL NOTE (FINDING 8 — P0 FIX):
    `_clean_reconstructed_text()` runs AFTER building span_map, causing index drift.
    This is a true bug and must be fixed.

    Temporary mitigation (annotated below):
      - Remove aggressive cleaning
      - Ensure only spacing insertions occur BEFORE span_map

    ### NOTE (FINDING 3):
    pysbd mismatches can occur — log failures.
    """

    if not coordinate_blocks:
        return []

    full_text = ""
    span_map = []
    last_span = None

    for i, span in enumerate(coordinate_blocks):
        text = span['text']
        prefix = ""

        if last_span:
            prefix, full_text = _calculate_spacing_prefix(span, last_span, full_text)

        start_idx = len(full_text) + len(prefix)
        full_text += prefix + text
        end_idx = len(full_text)

        span_map.append({
            "start": start_idx,
            "end": end_idx,
            "span_index": i
        })

        last_span = span

    ### CRITICAL FIX:
    # REMOVE _clean_reconstructed_text(full_text) FOR NOW
    # Because it causes span_map drift.
    #
    # full_text = _clean_reconstructed_text(full_text)

    seg = pysbd.Segmenter(language="en", clean=False)
    sentences_text = seg.segment(full_text)

    page_sentences = []
    cursor = 0

    for sent in sentences_text:
        sent = sent.strip()
        if not sent:
            continue

        start_char = full_text.find(sent, cursor)

        if start_char == -1:
            # Fallback: strip double spaces / punctuation normalization
            alt = re.sub(r'\s+', ' ', sent)
            start_char = full_text.find(alt, cursor)

        if start_char == -1:
            # Fallback 2: try removing trailing punctuation
            alt2 = sent.rstrip(" .,:;!?")
            start_char = full_text.find(alt2, cursor)

        if start_char == -1:
            logger.warning(f"[TextCleanup] Fuzzy sentence alignment failed: '{sent[:35]}...'")
            continue

        end_char = start_char + len(sent)
        cursor = end_char

        first_span_idx = -1
        last_span_idx = -1

        for entry in span_map:
            if entry['end'] > start_char and entry['start'] < end_char:
                if first_span_idx == -1:
                    first_span_idx = entry['span_index']
                last_span_idx = entry['span_index']

        if first_span_idx != -1:
            page_sentences.append({
                "text": sent,
                "span_start_index": first_span_idx,
                "span_end_index": last_span_idx
            })

    return page_sentences


def _calculate_spacing_prefix(span: dict, last_span: dict, current_text: str) -> tuple:
    """
    The Fingers: Determine spacing between two consecutive spans.

    Analyzes:
        1. Vertical gaps (new lines)
        2. Horizontal gaps (word spacing)
        3. Trailing hyphens (join hyphenated words)
        4. NEW: Punctuation-start spans (no prefix needed)

    ... [rest of docstring unchanged] ...
    """
    bbox = span['bbox']
    last_bbox = last_span['bbox']
    prefix = ""

    # NEW: If current span starts with punctuation, don't add space
    # This handles: Span1="word " + Span2="." → "word." not "word ."
    current_text_stripped = span['text'].lstrip()
    if current_text_stripped and current_text_stripped[0] in '.,;:!?)]\'"':
        # Also strip trailing space from accumulated text
        current_text = current_text.rstrip()
        return "", current_text

    # Check 1: Vertical gap (new line)
    vertical_diff = abs(bbox['y'] - last_bbox['y'])
    line_height = last_span['font_size']

    if vertical_diff > (line_height * GAP_THRESHOLD_Y):
        # New line detected
        if current_text.endswith("-"):
            next_text = span['text']
            # Join only if next starts with a letter
            if re.match(r'^[A-Za-z]', next_text):
                current_text = current_text[:-1]
                prefix = ""
            else:
                prefix = " "
    else:
        # Check 2: Horizontal gap (same line)
        prev_right = last_bbox['x'] + last_bbox['width']
        curr_left = bbox['x']
        gap = curr_left - prev_right

        threshold = line_height * GAP_THRESHOLD_X
        if gap > threshold:
            prefix = " "

    return prefix, current_text


# ============================================
# LAYER 5: TTS SANITIZATION
# ============================================

def sanitize_for_tts(text: str) -> str:
    """
    ### NOTE (FINDING 5):
    Remove bracketed reference removal — breaks citations.

    ### NOTE:
    Zero-width character removal is safe here.
    """

    if not text:
        return ""

    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    text = text.replace('\u00ad', '')

    text = re.sub(r'[•◦▪▸►→←↑↓°†‡§¶]', '', text)

    text = text.replace('&', ' and ')
    text = text.replace('%', ' percent ')
    text = text.replace('+', ' plus ')
    text = text.replace('=', ' equals ')
    text = text.replace('@', ' at ')

    ### FIX (FINDING 5): REMOVE this
    # text = re.sub(r'\[\d+\]', '', text)

    text = re.sub(r'(?<=[.!?])\s*\d{1,2}(?=\s+[A-Z])', '', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()