# ~/TTS/my_app/pdf_processor/extraction_engine.py

"""
EXTRACTION ENGINE (Gold Standard v3 – V1 Final)
===============================================
The canonical engine for PDF text extraction, cleaning, and structure analysis.
Replaces the old 'text_cleanup.py' and 'process.py' extraction logic.

Architecture:
    - Stage 1: Geometry & Structure (Extraction, Column/Para Detection, Roles)
    - Stage 2: TTS Compilation (Segmentation, Sanitization, Chunking)

Principles:
    1. Dual-Stream: Preserves 'raw_text' (Fidelity) separate from 'tts_text' (Audio).
    2. Sacred Sequence: Columns -> Paragraphs -> Sort -> Filter -> Roles.
    3. Virtual Segmentation: Injects delimiters for pysbd without altering span data.
    4. Observable Spine: Returns structured data ready for observability.
"""
from __future__ import annotations
import re
import os
import unicodedata
import logging
import fitz  # PyMuPDF
import pysbd
import ftfy
import math
import sys
import copy
from typing import List, Dict, Tuple, Optional, Set
from typing_extensions import Literal
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, Field, model_validator
from collections import Counter
from collections import defaultdict
from difflib import SequenceMatcher

logger = logging.getLogger("ExtractionEngine")

if TYPE_CHECKING:
    from typing import Self


# ✦────────────────────✦────────────────────✦
#               ✿  CONSTANTS  ✿
# ✦────────────────────✦────────────────────✦

# ✦───────────── 1 ENUMS ─────────────✦

class TextRole(str, Enum):
    """
    Classification roles for extracted text spans.
    Inherits from str for backward compatibility with string comparisons.
    """
    # Content Roles
    BODY = "body"
    HEADING = "heading"
    SUBHEADING = "subheading"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    SIDEBAR = "sidebar"
    LIST_ITEM = "list_item"
    CODE = "code"
    HYPERLINK = "hyperlink"
    INLINE_EQUATION = "inline_equation"
    PAGE_NUMBER = "page_number"

    # Structural/Layout Roles
    HEADER_ARTIFACT = "header_artifact"
    FOOTER_ARTIFACT = "footer_artifact"
    EMPTY = "empty"
    INSIDE_FIGURE = "inside_figure"
    FIGURE_LABEL = "figure_label"
    DIAGRAM_LABEL = "diagram_label"
    TABLE_CELL = "table_cell"

    # Typographical Roles
    SUBSCRIPT = "subscript"
    SUPERSCRIPT = "superscript"
    FOOTNOTE_MARKER = "footnote_marker"


class TableSubRole(str, Enum):
    """Sub-classification for table cell content."""
    HEADER = "header"
    STUB = "stub"
    DATA = "data"


class PyMuPDFFlag(IntEnum):
    """
    PyMuPDF span flag bitmasks.
    Reference: https://pymupdf.readthedocs.io/en/latest/textpage.html
    """
    SUPERSCRIPT = 1
    ITALIC = 2
    MONOSPACE = 8


class InterruptionBehavior(str, Enum):
    SKIP = "skip"
    ANNOUNCE = "announce"
    BREAK = "break"


@dataclass
class InterruptionConfig:
    figure_behavior: InterruptionBehavior = InterruptionBehavior.SKIP
    table_behavior: InterruptionBehavior = InterruptionBehavior.BREAK
    sidebar_behavior: InterruptionBehavior = InterruptionBehavior.SKIP
    caption_format: str = "[Figure {index}: {text}]"


# ✦───────────── 2 PYDANTIC CONFIG MODELS ─────────────✦

class ChunkConfig(BaseModel):
    """Text chunking parameters for TTS processing."""
    model_config = ConfigDict(frozen=True)

    max_chars: int = Field(
        default=600,
        gt=0,
        description="Maximum characters per audio chunk"
    )
    min_viable_chars: int = Field(
        default=25,
        gt=0,
        description="Minimum characters for standalone chunk viability"
    )
    avg_chars_per_sec: float = Field(
        default=15.0,  # synced with _AVG_CHARS_PER_SEC
        gt=0,
        description="Average characters spoken per second (duration estimation)"
    )

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> Self:
        if self.max_chars <= self.min_viable_chars:
            raise ValueError("max_chars must exceed min_viable_chars")
        return self


class PageLayoutConfig(BaseModel):
    """Page geometry thresholds for layout analysis."""
    model_config = ConfigDict(frozen=True)

    edge_zone_ratio: float = Field(
        default=0.10,
        ge=0.0,
        le=0.5,
        description="Top/bottom edge zone ratio for header/footer detection"
    )
    header_threshold_y: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Normalized Y threshold below which text is candidate header"
    )
    footer_threshold_y: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Normalized Y threshold above which text is candidate footer"
    )
    gap_threshold_x_ratio: float = Field(
        default=0.2,
        gt=0.0,
        description="Horizontal gap threshold relative to font size"
    )
    gap_threshold_y_ratio: float = Field(
        default=0.7,
        gt=0.0,
        description="Vertical gap threshold relative to font size"
    )

    @model_validator(mode="after")
    def validate_header_footer_order(self) -> Self:
        if self.header_threshold_y >= self.footer_threshold_y:
            raise ValueError("header_threshold_y must be less than footer_threshold_y")
        return self


class ParagraphConfig(BaseModel):
    """Paragraph boundary detection thresholds."""
    model_config = ConfigDict(frozen=True)

    indent_threshold_ratio: float = Field(
        default=1.2,
        gt=0.0,
        description="Indent change ratio triggering new paragraph"
    )
    font_drop_ratio: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Font size drop ratio triggering new paragraph"
    )
    vertical_gap_ratio: float = Field(
        default=1.5,
        gt=0.0,
        description="Vertical gap ratio triggering new paragraph"
    )


class RoleClassificationConfig(BaseModel):
    """Thresholds for text role classification."""
    model_config = ConfigDict(frozen=True)

    footnote_size_ratio: float = Field(
        default=0.75,
        gt=0.0,
        le=1.0,
        description="Font size ratio below which text is footnote-like"
    )
    sidebar_margin_ratio: float = Field(
        default=0.18,
        gt=0.0,
        lt=0.5,
        description="Page margin ratio for sidebar detection (both sides)"
    )
    footnote_y_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Normalized Y threshold for footnote zone"
    )
    min_font_size_footnote: float = Field(
        default=7.0,
        gt=0.0,
        description="Minimum font size for footnote classification"
    )
    subscript_offset_ratio: float = Field(
        default=0.3,
        gt=0.0,
        description="Baseline offset ratio for subscript detection"
    )


class TTSViabilityConfig(BaseModel):
    """TTS output viability thresholds."""
    model_config = ConfigDict(frozen=True)

    min_viable_alpha_chars: int = Field(
        default=2,
        ge=0,
        description="Minimum alphabetic characters for TTS viability"
    )
    min_text_length: int = Field(
        default=1,
        ge=0,
        description="Absolute minimum text length"
    )


class FeatureFlags(BaseModel):
    """Runtime feature toggles."""
    model_config = ConfigDict(frozen=True)

    enable_diagram_label_filter: bool = Field(
        default=True,
        description="Filter out isolated diagram labels (axis markers, etc.)"
    )


class ExtractionConfig(BaseModel):
    """
    Root configuration container for extraction engine.
    Immutable after instantiation.
    """
    model_config = ConfigDict(frozen=True)

    chunking: ChunkConfig = Field(default_factory=ChunkConfig)
    layout: PageLayoutConfig = Field(default_factory=PageLayoutConfig)
    paragraph: ParagraphConfig = Field(default_factory=ParagraphConfig)
    roles: RoleClassificationConfig = Field(default_factory=RoleClassificationConfig)
    tts: TTSViabilityConfig = Field(default_factory=TTSViabilityConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)


# ✦───────────── 3 PROCESSING STATE ─────────────✦

@dataclass
class ProcessingState:
    """
    Mutable state container for document processing.
    Passed through pipeline methods to track cross-span context.

    RESERVED: Currently unused (role assignment uses local variables).
    Retained for future multi-pass or resumable pipeline patterns.
    """
    previous_role: TextRole | None = None
    previous_y: float | None = None
    previous_x: float | None = None
    previous_text: str = ""
    previous_font_size: float | None = None
    current_page_num: int = 0

    # Caption chain tracking
    caption_chain_start_y: float | None = None
    caption_chain_start_font_size: float | None = None
    caption_chain_length: int = 0

    def reset(self) -> None:
        """Reset state for new document processing."""
        self.previous_role = None
        self.previous_y = None
        self.previous_x = None
        self.previous_text = ""
        self.previous_font_size = None
        self.caption_chain_start_y = None
        self.caption_chain_start_font_size = None
        self.caption_chain_length = 0


# ✦───────────── 4 STATIC DATA SETS ─────────────✦


# --- Layout & Column Detection ---

# Default page dimensions (US Letter)
_LAYOUT_GAP_RATIO_STANDARD: float = 0.05  # 5% of page width (~30px)
_LAYOUT_GAP_RATIO_WITH_MARGINS: float = 0.04  # More permissive with sidebars
_LAYOUT_MIN_GAP_THRESHOLD: float = 20.0  # Minimum gap to consider (pixels)
_LAYOUT_MAX_COLUMNS: int = 4
# US Letter: 8.5" × 72 points/inch = 612 points
_LAYOUT_DEFAULT_PAGE_WIDTH: float = 612.0

# --- Column Index Reservations (Margin Isolation) ---
_COLUMN_INDEX_LEFT_MARGIN: int = -1  # Left sidebar/margin content
_COLUMN_INDEX_RIGHT_MARGIN: int = 100  # Right sidebar/margin content

# --- Configuration Constants ---
ENABLE_DIAGRAM_LABEL_FILTER: bool = True

# --- Ingestion Constants ---

# PyMuPDF block types
_PYMUPDF_TEXT_BLOCK_TYPE: int = 0

# Group 0 Constants — ADD
# NOTE: VALID_SHORT_WORDS is for token survival;
# PROTECTED_SHORT_WORDS is for semantic/TTS preservation.
VALID_SHORT_WORDS: frozenset[str] = frozenset({
    # 1-letter words
    'a', 'i',
    # 2-letter words (common)
    'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is',
    'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we',
    # 3-letter words (very common)
    'all', 'and', 'are', 'but', 'can', 'did', 'for', 'get', 'got', 'had',
    'has', 'her', 'him', 'his', 'how', 'its', 'let', 'may', 'new', 'nor',
    'not', 'now', 'old', 'one', 'our', 'out', 'own', 'per', 'put', 'run',
    'say', 'see', 'set', 'she', 'the', 'too', 'try', 'two', 'use', 'was',
    'way', 'who', 'why', 'yet', 'you',
})

# --- Exclusion Reason Strings ---

_REASON_INVALID_BBOX: str = "invalid_bbox"
_REASON_EMPTY: str = "empty"
_REASON_HEADER_BAND: str = "header_band"
_REASON_FOOTER_BAND: str = "footer_band"
_REASON_HEADER_ARTIFACT: str = "header_artifact"
_REASON_DIAGRAM_LABEL: str = "diagram_label"
_REASON_BARE_CAPTION: str = "bare_caption"
_REASON_NOISE_SINGLE_CHAR: str = "noise_single_char"
_REASON_NOISE_DIGIT_ONLY: str = "noise_digit_only"
_REASON_NOISE_PUNCTUATION: str = "noise_punctuation"
_REASON_NOISE_FRAGMENT: str = "noise_fragment"
_REASON_VISUAL_OVERLAP: str = "visual_overlap"

# A2 chain protection: noise reasons eligible for inline continuation rescue
_A2_NOISE_REASONS: frozenset[str] = frozenset({
    _REASON_NOISE_SINGLE_CHAR,
    _REASON_NOISE_DIGIT_ONLY,
    _REASON_NOISE_PUNCTUATION,
    _REASON_NOISE_FRAGMENT,
})

# STAGE 1 SOFT CLASSIFICATION CONSTANTS (Schema v2.0)
_SOFT_CLASSIFY_NEAR_THRESHOLD_RATIO: float = 0.02  # 2% of page dimension
_SOFT_CLASSIFY_OVERLAP_THRESHOLD: float = 0.50  # 50% overlap = "inside"

_BOUNDARY_BODY_FONT_TOLERANCE: float = 0.25
_BOUNDARY_BODY_MIN_TEXT_LENGTH: int = 3
_BOUNDARY_BODY_PERCENTILE: float = 0.90
_BOUNDARY_GAP_MULTIPLIER: float = 2.5
_BOUNDARY_FOOTER_CEILING: float = 0.70
_BOUNDARY_FOOTER_FLOOR: float = 0.92
_REASON_PUBLISHER_METADATA: str = "publisher_metadata"

# --- Filter Threshold Constants ---

# US Letter: 11.0" × 72 points/inch = 792 points
_FILTER_DEFAULT_PAGE_HEIGHT: float = 792.0
_FILTER_Y_BAND_PRECISION_RATIO: float = 0.01  # 1% of page height
_FILTER_Y_BAND_MIN_PIXELS: int = 5
_FILTER_HEADER_ARTIFACT_ZONE_RATIO: float = 0.07  # Top 7% of page
_FILTER_BARE_CAPTION_MAX_CHARS: int = 60
_FILTER_FRAGMENT_THRESHOLD: int = 8
_FILTER_VERY_SHORT_THRESHOLD: int = 3
_FILTER_SIGNIFICANT_DIGIT_LENGTH: int = 3
_FILTER_SHORT_FRAGMENT_THRESHOLD: int = 2

# --- Global Band Filtering Thresholds ---

_FILTER_BAND_TOLERANCE_RATIO: float = 0.01  # 1% of page height
_FILTER_BAND_MIN_TOLERANCE: int = 3  # Minimum 3 pixels
_FILTER_BAND_MAX_TOLERANCE: int = 15  # Maximum 15 pixels

# --- Text Similarity Thresholds for Header/Footer Matching ---

_FILTER_EXACT_MATCH_THRESHOLD: float = 1.0
_FILTER_FUZZY_MATCH_THRESHOLD: float = 0.75
_FILTER_SHORT_TEXT_THRESHOLD: float = 0.85  # Higher threshold for short text
_FILTER_SHORT_TEXT_LENGTH: int = 10
_FILTER_MIN_SAMPLE_LENGTH: int = 2

# Year prefix detection
_FILTER_YEAR_PREFIXES: tuple[str, ...] = ("19", "20")
_FILTER_YEAR_LENGTH: int = 4

# Vowels for artifact detection
_FILTER_VOWELS: frozenset[str] = frozenset("AEIOU")

# --- TTS Character Substitutions ---

# paste actual curly quotes pre test
TTS_SUBSTITUTIONS: dict[str, str] = {
    # Symbols to words
    "&": " and ",
    "%": " percent ",
    "+": " plus ",
    "=": " equals ",
    "@": " at ",
    "#": " number ",
    "/": " or ",

    # Dashes
    "–": "-",  # en-dash → hyphen
    "—": ", ",  # em-dash → comma pause
    "―": ", ",  # horizontal bar → comma pause

    # Curly quotes → straight
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    # Other typography
    "…": "...",
    "•": ", ",
    "·": " ",
    "°": " degrees ",
    "±": " plus or minus ",
    "×": " times ",
    "÷": " divided by ",
    "≈": " approximately ",
    "≠": " not equal to ",
    "≤": " less than or equal to ",
    "≥": " greater than or equal to ",
    "√": " square root of ",

    # Fractions
    "½": " one half ",
    "¼": " one quarter ",
    "¾": " three quarters ",
    "⅓": " one third ",
    "⅔": " two thirds ",
    "⅕": " one fifth ",
    "⅖": " two fifths ",
    "⅗": " three fifths ",
    "⅘": " four fifths ",
    "⅙": " one sixth ",
    "⅚": " five sixths ",
    "⅛": " one eighth ",
    "⅜": " three eighths ",
    "⅝": " five eighths ",
    "⅞": " seven eighths ",

    # Scientific subscripts
    "₀": " 0 ", "₁": " 1 ", "₂": " 2 ", "₃": " 3 ", "₄": " 4 ",
    "₅": " 5 ", "₆": " 6 ", "₇": " 7 ", "₈": " 8 ", "₉": " 9 ",

    # Scientific superscripts
    "⁰": " to the 0 ",
    "¹": " to the 1 ",
    "²": " squared ",
    "³": " cubed ",
    "⁴": " to the 4 ",
    "⁵": " to the 5 ",
    "⁶": " to the 6 ",
    "⁷": " to the 7 ",
    "⁸": " to the 8 ",
    "⁹": " to the 9 ",
    "ⁿ": " to the n ",
    "ˣ": " to the x ",

    # Comparison operators
    "<": " less than ",
    ">": " greater than ",
    "~": " approximately ",

    # Currency
    "$": " dollars ",
    "€": " euros ",
    "£": " pounds ",
    "¥": " yen ",
    "¢": " cents ",

    # Legal/trademark
    "©": " copyright ",
    "®": " registered ",
    "™": " trademark ",
    "℠": " service mark ",

    # Math symbols
    "∞": " infinity ",
    "∑": " sum of ",
    "∫": " integral of ",
    "π": " pi ",
    "θ": " theta ",
    "α": " alpha ",
    "β": " beta ",
    "γ": " gamma ",
    "δ": " delta ",
    "ε": " epsilon ",
    "λ": " lambda ",
    "μ": " mu ",
    "σ": " sigma ",
    "τ": " tau ",
    "φ": " phi ",
    "ω": " omega ",

    # Temperature
    "℃": " degrees Celsius ",
    "℉": " degrees Fahrenheit ",

    # Quotation normalization
    "‹": "'",
    "›": "'",

    # Dashes (normalize)
    "‐": "-",
    "‑": "-",
    "‒": "-",

    # Other punctuation
    "′": "'",
    "″": '"',
    "‚": ",",
    "ʼ": "'",
    "ˈ": "'",
    "⁃": ", ",

    "→": " to ",
    "←": " from ",
    "§": " section ",
    "¶": " paragraph ",
    "†": "",
    "‡": "",
    "∏": " product of ",
    "∂": " partial ",
    "Δ": " delta ",

}

# --- Subscript/Superscript Digit Maps ---

SUBSCRIPT_DIGITS: dict[str, str] = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")",
    "ₐ": "a", "ₑ": "e", "ₒ": "o", "ₓ": "x", "ₔ": "schwa",
    "ₕ": "h", "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n",
    "ₚ": "p", "ₛ": "s", "ₜ": "t",
}

SUPERSCRIPT_DIGITS: dict[str, str] = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")",
    "ⁿ": "n", "ⁱ": "i",
}

# PHASE 1.5: Continuity Override Configuration

# Roles that can be overridden by stream continuity (geometry-based roles only)
_CONTINUITY_OVERRIDE_CANDIDATES: frozenset[str] = frozenset({
    TextRole.INSIDE_FIGURE.value,
    TextRole.FIGURE_LABEL.value,
})

# Hard semantic veto patterns - spans matching these are NEVER promoted to body
_CONTINUITY_VETO_PATTERNS: tuple[str, ...] = (
    "figure", "fig.", "fig ", "table", "chart", "graph", "diagram",
    "source:", "note:", "©", "http://", "https://", "www.", "doi:",
    "page ", "p. ", "pp.", "vol.", "chapter", "section", "appendix",
)

# Maximum Y-gap (points) between spans for adjacency (prevents cross-block contagion)
_CONTINUITY_MAX_Y_GAP: float = 30.0

# Terminal punctuation that definitively ends a sentence
# NOTE: Excludes ':' (introduces clauses) and ';' (joins clauses) per lead review
_CONTINUITY_TERMINAL_CHARS: str = ".!?"

# Tail: Trailing context from previous page (2-3 paragraphs typically)
_WINDOW_TAIL_SPAN_COUNT: int = 10
# Head: Look-ahead for semantic resolution (covers most single-page layouts)
# Asymmetric because forward context is more valuable for RONC linking
_WINDOW_HEAD_SPAN_COUNT: int = 35

# --- TTS Sanitization Thresholds ---

_TTS_ACRONYM_LENGTH_THRESHOLD: int = 4
_TTS_LONG_TEXT_THRESHOLD: int = 20

# --- TTS Viability: Role Gate ---

_TTS_NON_VIABLE_ROLES: frozenset[str] = frozenset({
    TextRole.SIDEBAR.value,
    TextRole.FOOTNOTE.value,
    TextRole.INSIDE_FIGURE.value,
    TextRole.FIGURE_LABEL.value,
    TextRole.DIAGRAM_LABEL.value,
    TextRole.TABLE_CELL.value,
    TextRole.CODE.value,
    TextRole.HYPERLINK.value,
    TextRole.EMPTY.value,
    TextRole.HEADER_ARTIFACT.value,
    TextRole.FOOTER_ARTIFACT.value,
    TextRole.PAGE_NUMBER.value,
})

_TTS_ORDERABLE_ROLES: frozenset[str] = frozenset({
    TextRole.BODY.value,
    TextRole.HEADING.value,
    TextRole.SUBHEADING.value,
    TextRole.CAPTION.value,
})

# --- TTS Viability: Punctuation Noise ---

_TTS_PUNCT_CHARS: frozenset[str] = frozenset(
    '.,;:!?-–—―/()[]{}"\'""«»„‹›*†‡§¶ '
)

# --- TTS Viability: Valid Lowercase Starters ---

_TTS_VALID_LOWERCASE_STARTERS: frozenset[str] = frozenset({
    "iphone", "ph", "mrna", "e.g.", "i.e.", "etc."
})

# --- TTS Viability: Length Thresholds ---

_TTS_MIN_CHAR_COUNT: int = 5
_TTS_MIN_ALPHA_COUNT: int = 2
_TTS_MIN_SINGLE_WORD_CHARS: int = 3
_TTS_SUBSTANTIAL_WORD_COUNT: int = 5
_TTS_LONG_FRAGMENT_WORD_COUNT: int = 8
_TTS_TRUNCATION_MIN_WORDS: int = 10
_TTS_TRUNCATION_MIN_CHARS: int = 60

# --- TTS Viability: Garble Detection ---

_TTS_GARBLE_MIN_WORDS: int = 4
_TTS_GARBLE_CAPS_RATIO: float = 0.40
_TTS_GARBLE_CAPS_COUNT: int = 3
_TTS_GARBLE_LONG_WORD_COUNT: int = 8

# --- Chunking Configuration ---
_CHUNK_MIN_CHARS: int = 50
_CHUNK_MAX_CHARS: int = 450
# Hard ceiling = soft max + average sentence length (~50 chars)
# Allows one additional sentence for cross-page continuations
_CHUNK_ABSOLUTE_MAX_CHARS: int = 500
_AVG_CHARS_PER_SEC: float = 15.0

# --- Dialogue Mode Detection ---
_DIALOGUE_SHORT_RESPONSE_CHARS: int = 50
_ORCHESTRATOR_DIALOGUE_TRIGGERS: frozenset[str] = frozenset({":", "?", "said", "asked"})

# --- Sort Configuration ---
_SORT_TOLERANCE_PROSE: int = 8  # Loose: catches italic/superscript baseline drift (up to 7.9px)
_SORT_TOLERANCE_TABLE: int = 3  # Tight: preserves row structure integrity (typical table rows 8-12px)

# --- Text Cleaning Constants ---

# Characters that become regular spaces
_CLEAN_SPACE_CHARS: frozenset[str] = frozenset({
    "\xa0",  # non-breaking space
    "\u2000",  # en quad
    "\u2001",  # em quad
    "\u2002",  # en space
    "\u2003",  # em space
    "\u2004",  # three-per-em space
    "\u2005",  # four-per-em space
    "\u2006",  # six-per-em space
    "\u2007",  # figure space
    "\u2008",  # punctuation space
    "\u2009",  # thin space
    "\u200a",  # hair space
    "\u202f",  # narrow no-break space
    "\u205f",  # medium mathematical space
    "\u3000",  # ideographic space
    "\t",  # tab
    "\u2028",  # line separator
})

# Characters to remove entirely
_CLEAN_REMOVE_CHARS: frozenset[str] = frozenset({
    "\u00ad",  # soft hyphen
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # BOM / zero-width no-break space
    "\u2029",  # paragraph separator
})

# --- Diagram Label Detection Term Sets ---
# FORCED_LABEL_TERMS: High-confidence terms that FORCE label classification
# when 2+ appear together (regardless of other signals). Subset of _LABEL_TECHNICAL_TERMS.
# Used in: _is_diagram_label() kill-list path (line ~8672)
FORCED_LABEL_TERMS: frozenset[str] = frozenset({
    'force', 'length', 'velocity', 'feedback', 'error',
    'signal', 'control', 'gain', 'loop', 'input', 'output',
    'driving', 'afferents', 'efferents',
})
# NOTE: _LABEL_TECHNICAL_TERMS (line ~1250) is a superset used for
# density-based detection. FORCED_LABEL_TERMS is for hard override.

# --- Subscript Detection ---

_SUBSCRIPT_OFFSET_RATIO: float = 0.3  # Baseline drop > 30% of font_size => subscript

# --- Acronym Detection ---

_ACRONYM_MIN_LENGTH: int = 2
_ACRONYM_MAX_LENGTH: int = 4
_ACRONYM_MIXED_CASE_MAX_LENGTH: int = 5
_ACRONYM_STRIP_CHARS: str = ".,;:"

# --- Fragment Detection ---

_FRAGMENT_SHORT_MAX_LENGTH: int = 3
_FRAGMENT_BROKEN_WORD_MIN_LENGTH: int = 2

# Common English word-start bigrams (indicate broken word if isolated lowercase)
_FRAGMENT_MIDWORD_PREFIXES: frozenset[str] = frozenset({
    "th", "he", "in", "er", "an", "on", "or", "ed", "ng"
})

# Ragged-Edge Magnet
MAGNET_GAP_EM: float = 2.0

# Roles NOT eligible for a2-based magnet reclassification
_MAGNET_A2_NON_PROMOTABLE_ROLES: frozenset[str] = frozenset({
    TextRole.TABLE_CELL.value,
    TextRole.CAPTION.value,
    TextRole.FIGURE_LABEL.value,
    TextRole.FOOTNOTE.value,
    TextRole.PAGE_NUMBER.value,
    TextRole.HEADER_ARTIFACT.value,
    TextRole.FOOTER_ARTIFACT.value,
})

# --- Margin Detection (Histogram Analysis) ---

# Histogram bin configuration
_MARGIN_MIN_BINS: int = 8
_MARGIN_MAX_BINS: int = 20
_MARGIN_TARGET_BIN_WIDTH: float = 50.0  # points

# Density threshold parameters
_MARGIN_DENSITY_RATIO: float = 0.12  # 12% of max bin count
_MARGIN_MIN_DENSITY_COUNT: int = 2  # Absolute minimum
_MARGIN_AVG_WEIGHT_MULTIPLIER: float = 0.5

# Margin constraints (as ratio of page width)
_MARGIN_MIN_RATIO: float = 0.04  # Minimum 4% margin
_MARGIN_MAX_RATIO: float = 0.35  # Maximum 35% margin
_MARGIN_FALLBACK_RATIO: float = 0.10  # Default fallback (10%)

# Feature flags
_MARGIN_WEIGHT_BY_CHARS: bool = True

# Default excluded roles for margin detection
_MARGIN_EXCLUDE_ROLES: frozenset[TextRole] = frozenset({
    TextRole.INSIDE_FIGURE,
    TextRole.FIGURE_LABEL,
    TextRole.TABLE_CELL,
})

# ============================================================================
# STAGE 2 SEMANTIC ENGINE CONSTANTS
# ============================================================================

_SEMANTIC_NEVER_OVERRIDE_ROLES: frozenset[str] = frozenset({
    TextRole.PAGE_NUMBER.value,
    TextRole.HEADER_ARTIFACT.value,
    TextRole.FOOTER_ARTIFACT.value,
    TextRole.FOOTNOTE.value,
})

# Disposition vocabulary (E2 P0 alignment)
_SEM_DISP_INCLUDED: str = "included"
_SEM_DISP_EXCLUDED: str = "excluded"
_SEM_DISP_INTERRUPTION: str = "interruption"
_SEM_DISP_PENDING: str = "pending"

# --- Title Case Conversion ---

_TITLE_CASE_UPPERCASE_THRESHOLD: float = 0.70
_TITLE_CASE_SHORT_WORD_MAX_LENGTH: int = 4

# --- Chunk Boundary Roles ---

_CHUNK_BOUNDARY_ROLES: frozenset[str] = frozenset({
    TextRole.HEADING.value,
    TextRole.SUBHEADING.value,
    TextRole.CAPTION.value,
    TextRole.CODE.value,
    TextRole.SIDEBAR.value,
    TextRole.FOOTNOTE.value,
    TextRole.TABLE_CELL.value,
    TextRole.INSIDE_FIGURE.value,
})

# --- Cross-Page Stitching Configuration ---

_STITCH_MAX_PASSES: int = 3
_STITCH_MAX_LOOKAHEAD: int = 5
_STITCH_MAX_Y_GAP_SAME_PAGE: float = 50.0
_STITCH_MAX_NEGATIVE_Y_GAP: float = 25.0  # px, defensive threshold

# NEW: Maximum span index gap for stitching
# Prevents merging sentences from distant content blocks
_STITCH_MAX_SPAN_GAP: int = 8

# --- Roles that block/skip stitching ---

_STITCH_BLOCKING_ROLES: frozenset[str] = frozenset({
    TextRole.HEADING.value,
    TextRole.SUBHEADING.value,
    TextRole.CAPTION.value,
    TextRole.LIST_ITEM.value,
    TextRole.CODE.value,
})

_STITCH_SKIP_ROLES: frozenset[str] = frozenset({
    TextRole.HEADING.value,
    TextRole.SUBHEADING.value,
    TextRole.CAPTION.value,
    TextRole.SIDEBAR.value,
    TextRole.FOOTNOTE.value,
    TextRole.FIGURE_LABEL.value,
    TextRole.DIAGRAM_LABEL.value,
    TextRole.TABLE_CELL.value,
})

# Continuation words that signal incomplete previous sentence
_STITCH_CONTINUATION_WORDS: frozenset[str] = frozenset({
    "and", "or", "but", "nor", "yet", "so", "then", "thus", "hence",
    "therefore", "however", "moreover", "furthermore", "additionally",
    "consequently", "nevertheless", "nonetheless", "otherwise",
    "meanwhile", "subsequently", "accordingly", "specifically",
    "particularly", "especially", "including", "excluding",
    "whereas", "whereby", "wherein", "whereupon",
})

# Noise substrings to remove during cleaning
# Document metadata, URLs, and artifacts that contaminate content
_NOISE_SUBSTRINGS: tuple[str, ...] = (
    # Document metadata
    "css4.pub",
    "publishing, see",
    "see css4",

    # Source attribution (should be header/footer but sometimes escapes)
    "FROM WIKIBOOKS",
    "from wikibooks",

    # Common PDF artifacts
    "page break",
    "continued on next page",
    "continued from previous",
)

# NEW: Noise patterns (regex) for more complex matches
_NOISE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    # Publishing references
    re.compile(r'\bsee\s+\w+\.pub\b', re.IGNORECASE),
    re.compile(r'\bcss\d+\.\w+\b', re.IGNORECASE),

    # Wikibooks attribution (case-insensitive, with variations)
    re.compile(r'\bfrom\s+wikibooks?\b', re.IGNORECASE),

    # Page number artifacts
    re.compile(r'^page\s+\d+\s*$', re.IGNORECASE),
    re.compile(r'^\d+\s*of\s*\d+\s*$', re.IGNORECASE),
)

# Incomplete endings that override terminal punctuation
# These words almost NEVER end a sentence legitimately
_STITCH_INCOMPLETE_ENDINGS: frozenset[str] = frozenset({
    # Articles
    "a", "an", "the",
    # Prepositions
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "under", "over", "about", "against", "among", "around",
    # Conjunctions
    "and", "or", "but", "nor", "yet", "so",
    # Modals (Keep these - they demand continuation)
    "can", "could", "may", "might", "shall", "should", "will", "would",
    "must",
    # REMOVED: is, are, was, were, be, been, being (Too greedy)
    # REMOVED: has, have, had, do, does, did (Too greedy)
    # Determiners
    "this", "that", "these", "those", "some", "any", "no", "every",
    # Adjectives that require nouns
    "such", "other", "another", "each", "either", "neither",
    # Sentence starters that got cut off
    "if", "when", "while", "although", "because", "since", "unless",
    "whether", "whereas", "whereby",
    # Participles/gerunds that often signal breaks
    "adapting", "responding", "following", "including", "containing",
})

# Sentence starters that strongly indicate a new sentence boundary
# Used in RULE 3.5 (Capitalized Sentence Hard Stop)
_STITCH_COMMON_SENTENCE_STARTERS: frozenset[str] = frozenset({
    "The", "A", "An", "It", "This", "These", "Those",
    "He", "She", "They",
    "But", "However", "Therefore", "Furthermore",
    "In", "On",
    "Figure", "Table"
})

# Margin-stream inline connectives: single-word spans that signal
# continuation rather than standalone content (used by _resolve_semantic_continuity)
_MARGIN_INLINE_CONNECTIVES: frozenset[str] = frozenset({
    "or", "and", "but", "is", "are", "was", "were",
})

# --- Punctuation Classification ---

_STITCH_CONTINUING_PUNCT: frozenset[str] = frozenset({",", ";", ":", ")", "]", '"', "'"})
_STITCH_NEW_SENTENCE_PUNCT: frozenset[str] = frozenset({'"', "'", "(", "[", "—", "-"})

# --- Number Context Words ---

_STITCH_NUMBER_CONTEXT_WORDS: frozenset[str] = frozenset({
    "approximately", "about", "around", "nearly",
    "over", "under", "between", "from", "to",
    "was", "were", "is", "are", "of", "than",
})

# --- Text Reconstruction (Break Insertion) ---

_RECONSTRUCT_BREAK_BEFORE_ROLES: frozenset[str] = frozenset({
    TextRole.HEADING.value,
    TextRole.SUBHEADING.value,
    TextRole.CAPTION.value,
    TextRole.LIST_ITEM.value,
})

_RECONSTRUCT_BREAK_AFTER_ROLES: frozenset[str] = frozenset({
    TextRole.HEADING.value,
    TextRole.SUBHEADING.value,
    TextRole.CAPTION.value,
})

# --- Sentence Segmentation Alignment ---

_SEGMENT_MAX_PROBE_DISTANCE: int = 500
_SEGMENT_MIN_PROBE_DISTANCE: int = 50
_SEGMENT_PROBE_SLACK: int = 20
_SEGMENT_FUZZY_WORD_COUNT: int = 3

# --- Global Band Detection ---
_GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT: float = 0.15  # Widen search to top/bottom 15%
_GLOBAL_BAND_MERGE_TOLERANCE: int = 5
_GLOBAL_BAND_MIN_PAGE_FRACTION: float = 0.25
_GLOBAL_BAND_ROUNDING_PRECISION: int = -1  # round(y, -1) → nearest 10

# --- Region Detection ---

_REGION_FIGURE_EXPAND_PADDING: int = 15  # Pixels to expand figure rects
_REGION_MIN_DRAWING_DIMENSION: int = 30  # Minimum width/height for drawings
_REGION_DRAWING_MERGE_GAP: float = 20.0  # Gap for merging nearby drawings

# Figure prose validation thresholds
_REGION_FIGURE_MAX_PROSE_SPANS: int = 3
_REGION_FIGURE_MAX_TEXT_LENGTH: int = 200
_REGION_PROSE_MIN_WORD_COUNT: int = 8
_REGION_PROSE_MIN_TEXT_LENGTH: int = 40
_REGION_SENTENCE_PUNCTUATION: frozenset[str] = frozenset(".!?")

# Header/footer band detection
_REGION_Y_BAND_ROUNDING: int = 10  # Round Y to nearest 10
_REGION_MIN_BAND_COUNT: int = 3
_REGION_BAND_THRESHOLD_RATIO: float = 0.2

# DEPRECATED: Originally mirrored from PageLayoutConfig defaults.
# Replaced by _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT for global band detection.
# Retained for backward compatibility with PageLayoutConfig contract.
_LAYOUT_HEADER_THRESHOLD_Y: float = 0.20
_LAYOUT_FOOTER_THRESHOLD_Y: float = 0.80

# --- Paragraph Detection ---

_PARA_MAX_LINE_GAP: float = 50.0  # Maximum reasonable line gap
_PARA_DEFAULT_LINE_HEIGHT: float = 12.0
_PARA_GAP_MULTIPLIER: float = 1.25  # Gap > this * line_height = new para

# --- Code Detection ---

_CODE_MIN_PUNCT_COUNT: int = 3
_CODE_PUNCT_DENSITY_DIVISOR: int = 10

# --- Inline Equation Detection ---

_EQUATION_LONG_TEXT_THRESHOLD: int = 20
_EQUATION_MIN_SYMBOLS_FOR_LONG: int = 2

# Math font hints (distinct from code fonts)
_MATH_FONT_HINTS: tuple[str, ...] = ("math", "cmmi", "cmsy")

# --- Diagram Label Detection ---

_LABEL_MAX_CHARS: int = 60
_LABEL_MAX_WORDS: int = 6
_LABEL_FONT_RATIO: float = 0.80
_LABEL_FIGURE_SHRINK_RATIO: float = 0.08
_LABEL_FIGURE_SHRINK_MIN: int = 5
_LABEL_FIGURE_SHRINK_MAX: int = 25
_LABEL_MIN_SHRUNK_DIMENSION: int = 10
_LABEL_SHORT_WORD_MAX_LENGTH: int = 8

# --- Role Assignment: Page Zones ---

_ROLE_LINE_HEIGHT_MULTIPLIER: float = 1.4
_ROLE_HEADER_ZONE_RATIO: float = 0.08
_ROLE_FOOTER_ZONE_RATIO: float = 0.92

# --- Role Assignment: Heading/Subheading ---

_ROLE_HEADING_FONT_RATIO: float = 1.15
_ROLE_SUBHEADING_FONT_RATIO: float = 1.08
_ROLE_HEADING_MAX_WORDS: int = 12
_ROLE_HEADING_MAX_CHARS: int = 80
_ROLE_TERMINAL_PUNCTUATION: frozenset[str] = frozenset(".,;:?!")

# Content-based heading detection: section number pattern.
# Matches spans like "1 Introduction", "3.2 White Space", "3.10 Long Name".
# The uppercase letter after whitespace confirms title text follows the number.
# Used as an alternative to font-size/weight signals for heading classification.
_HEADING_SECTION_NUMBER_PATTERN: re.Pattern[str] = re.compile(
    r'^\d+(?:\.\d+)*\s+[A-Z]'
)

# Font weight hints for heading detection.
# Replaces inline ("bold", "heavy", "black") tuple at Priority 6.
# Extended with "semibold" and "demibold" to cover font families where
# PyMuPDF reports weight variants rather than plain "bold".
_HEADING_FONT_WEIGHT_HINTS: frozenset[str] = frozenset({
    "bold", "heavy", "black", "semibold", "demibold",
})

# Layout-separation multiplier for heading detection.
# Measured as a multiple of estimated line height.
# 1.5–2.0 typical; tuned conservatively to avoid false positives.
_HEADING_VERTICAL_ISOLATION_MULTIPLIER: float = 1.75

# Font weight hints that qualify as heading-weight ONLY when combined with
# a section number pattern match. "medium" weight fonts are used for headings
# in some academic publishers (e.g., Springer LNNS) but are too ambiguous
# as a standalone heading signal — many body fonts contain "medium" in name.
_HEADING_FONT_WEIGHT_CONDITIONAL_HINTS: frozenset[str] = frozenset({
    "medium",
})


# 2. Keyword prefixes (e.g., "Section 1", "Chapter 2")
_HEADING_KEYWORD_PREFIX_PATTERN: re.Pattern[str] = re.compile(
    r'^(?:Section|Chapter|Article|Part|Appendix)\s+\d+',
    re.IGNORECASE,
)

_HEADING_LEGAL_SYMBOL_PATTERN: re.Pattern[str] = re.compile(
    r'^§\s*\d'
)

# Letter-prefix section numbering (outline format).
# Matches: "A. Background", "B. Methods", "I. Introduction"
# Single uppercase letter followed by period, whitespace, and uppercase
# start of title text. Common in legal briefs, policy documents, and
# outline-format academic papers. This pattern directly resolves the
# conflict with caption pattern ^[A-Z]\.\s+ at Priority 4.
_HEADING_LETTER_PREFIX_PATTERN: re.Pattern[str] = re.compile(
    r'^[A-Z]\.\s+[A-Z]'
)

# Multi-character roman numeral prefix.
# Matches: "II. Scope", "III. Definitions", "IV. Liability", "XII. Termination"
# Two or more roman numeral characters followed by period, whitespace,
# and uppercase. These do NOT conflict with the caption pattern (which
# requires exactly one letter before the period) but need Path 2
# coverage for correct heading classification at Priority 6.
_HEADING_ROMAN_NUMERAL_PREFIX_PATTERN: re.Pattern[str] = re.compile(
    r'^[IVXLCDM]{2,}\.\s+[A-Z]'
)

_HEADING_STRUCTURAL_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    _HEADING_SECTION_NUMBER_PATTERN,
    _HEADING_KEYWORD_PREFIX_PATTERN,
    _HEADING_LEGAL_SYMBOL_PATTERN,
    _HEADING_LETTER_PREFIX_PATTERN,
    _HEADING_ROMAN_NUMERAL_PREFIX_PATTERN,
)

# --- Role Assignment: Footnotes ---

_ROLE_FOOTNOTE_FONT_RATIO: float = 0.85
_ROLE_FOOTNOTE_MARKER_FONT_RATIO: float = 0.70
_ROLE_FOOTNOTE_MARKER_SYMBOLS: frozenset[str] = frozenset({
    "*", "†", "‡", "§", "¶", "‖"
})

# --- Role Assignment: Figure Labels ---

_ROLE_FIGURE_PROXIMITY_BUFFER: float = 20.0
_ROLE_FIGURE_LABEL_MAX_WORDS: int = 4
_ROLE_FIGURE_LABEL_MAX_CHARS: int = 25
_ROLE_FIGURE_LABEL_FONT_RATIO: float = 0.95
_ROLE_FIGURE_LABEL_SHORT_WORD_COUNT: int = 3

# --- Role Assignment: Caption Chain ---

_ROLE_CAPTION_Y_THRESHOLD_MULTIPLIER: float = 1.5
_ROLE_CAPTION_X_THRESHOLD_RATIO: float = 0.08
_ROLE_CAPTION_FONT_TOLERANCE: float = 1.5
_ROLE_CAPTION_MAX_CHAIN_LINES: int = 8
_ROLE_CAPTION_NEW_SECTION_GAP_MULTIPLIER: float = 1.2
_ROLE_CAPTION_LARGE_GAP_MULTIPLIER: float = 2.0

# --- Role Assignment: Code Detection ---

_ROLE_CODE_MIN_CHAR_COUNT: int = 10
_ROLE_CODE_PUNCT_RATIO: float = 0.15

# --- Role Assignment: Sub/Superscript ---

_ROLE_VERY_SMALL_FONT_RATIO: float = 0.75

# --- Role Assignment: Header Artifacts ---

_ROLE_SHORT_FRAGMENT_CHAR_COUNT: int = 30
_ROLE_SHORT_FRAGMENT_WORD_COUNT: int = 2
_ROLE_VERY_SHORT_UPPERCASE_THRESHOLD: int = 10

# --- Role Assignment: Sidebar ---

_ROLE_SIDEBAR_MIN_CHARS: int = 3

# --- Table Subrole Detection ---

_TABLE_HEADER_THRESHOLD_PIXELS: float = 30.0
_TABLE_HEADER_THRESHOLD_RATIO: float = 0.15
_TABLE_STUB_THRESHOLD_PIXELS: float = 80.0
_TABLE_STUB_THRESHOLD_RATIO: float = 0.20

# --- Structural Continuity Detection ---

_CONTINUITY_X_ALIGNMENT_TOLERANCE: float = 15.0
_CONTINUITY_WIDTH_SIMILARITY_RATIO: float = 0.80
_CONTINUITY_TABLE_BOTTOM_MARGIN_RATIO: float = 0.90
_CONTINUITY_TABLE_TOP_MARGIN_RATIO: float = 0.15
_CONTINUITY_FIGURE_BOTTOM_MARGIN_RATIO: float = 0.92
_CONTINUITY_FIGURE_TOP_MARGIN_RATIO: float = 0.10

# --- Caption Association ---

_CAPTION_ASSOC_MAX_DISTANCE_RATIO: float = 0.15
_CAPTION_ASSOC_HORIZ_WEIGHT: float = 0.3
_CAPTION_ASSOC_ABOVE_BONUS: float = 0.8

# PHASE 1.3 HARDEN: Document-Adaptive Inline Detection
_HARDEN_GAP_TOLERANCE_MULTIPLIER: float = 1.5
_HARDEN_SINGLE_PEER_WIDTH_RATIO: float = 0.06
_HARDEN_MAX_INLINE_WIDTH_RATIO: float = 0.10

# --- Acronyms & Protected Words ---

PROTECTED_ACRONYMS: frozenset[str] = frozenset({
    # Biology/Medicine
    "DNA", "RNA", "ATP", "ADP", "AMP", "GTP", "GDP", "NAD", "NADH", "FAD",
    "HIV", "AIDS", "PCR", "CRISPR", "mRNA", "tRNA", "rRNA", "siRNA",
    "CNS", "PNS", "ANS", "ECG", "EKG", "EEG", "MRI", "CT", "PET",
    # Chemistry
    "pH", "pKa", "NMR", "IR", "UV", "MS", "HPLC", "GC",
    # Physics/Engineering
    "AC", "DC", "RF", "EM", "LED", "LCD", "CPU", "GPU", "RAM", "ROM",
    # General
    "USA", "UK", "EU", "UN", "NATO", "NASA", "CEO", "CFO", "CTO",
    "PDF", "HTML", "CSS", "API", "URL", "USB", "HDMI",
    "ID", "vs", "etc", "ie", "eg",
})

# NOTE: VALID_SHORT_WORDS is for token survival;
# PROTECTED_SHORT_WORDS is for semantic/TTS preservation.
PROTECTED_SHORT_WORDS: frozenset[str] = frozenset({
    # 1-2 letter meaningful words
    "I", "A", "a",
    "OK", "US", "UK", "EU", "UN", "AI", "IT", "TV", "PC", "ID",
    "pH", "vs", "or", "if", "is", "as", "at", "by", "in", "on", "to",
    "an", "be", "do", "go", "he", "it", "me", "my", "no", "of", "so", "up", "we",
})

VALID_SINGLE_CHARS: frozenset[str] = frozenset({
    # =========================================================================
    # LATIN LETTERS (Drop caps, list markers, variables)
    # =========================================================================
    *"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",

    # =========================================================================
    # GREEK LETTERS (CRITICAL — Scientific terminology)
    # =========================================================================
    # Lowercase Greek (most common in scientific text)
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ",
    "ν", "ξ", "ο", "π", "ρ", "σ", "ς", "τ", "υ", "φ", "χ", "ψ", "ω",
    # Uppercase Greek
    "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ", "Λ", "Μ",
    "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω",

    # =========================================================================
    # MATHEMATICAL SYMBOLS
    # =========================================================================
    "∞", "∑", "∫", "∂", "∆", "∇", "√", "≈", "≠", "≤", "≥",
    "±", "∓", "×", "÷", "·", "∝", "∈", "∉", "⊂", "⊃", "∪", "∩",

    # =========================================================================
    # COMMON SYMBOLS
    # =========================================================================
    "&", "@", "#", "$", "%", "+", "=", "*", "°",

    # =========================================================================
    # CURRENCY
    # =========================================================================
    "€", "£", "¥", "¢", "₹", "₽",

    # =========================================================================
    # BULLETS AND MARKERS
    # =========================================================================
    "•", "◦", "▪", "▸", "►", "○", "●", "■", "□", "★", "☆",

    # =========================================================================
    # SUPERSCRIPT/SUBSCRIPT DIGITS (Common in scientific notation)
    # =========================================================================
    "⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹",
    "₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉",

    # =========================================================================
    # FRACTIONS
    # =========================================================================
    "½", "⅓", "¼", "⅕", "⅙", "⅛", "⅔", "¾", "⅖", "⅗",
})

_HEALING_CUT_INDICATORS: frozenset[str] = frozenset({
    # Auxiliary verbs
    "can", "will", "may", "shall", "should", "could", "would", "must",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    # Articles
    "the", "a", "an",
    # Prepositions
    "of", "to", "for", "in", "on", "at", "by", "with", "from",
    # Conjunctions
    "and", "or", "but", "nor",
    # Demonstratives
    "this", "that", "these", "those",
})

_HEAL_SHORT_FRAGMENT_MAX_WORDS: int = 4

# Minimum character length for the last word to qualify as a "real word"
# rather than an abbreviation suffix. Words like "al" (et al.), "vs",
# "cf", "eq" are typically abbreviation tails, not fragment endings.
_HEAL_SHORT_FRAGMENT_MIN_LAST_WORD_LEN: int = 3

NOISE_PUNCTUATION: frozenset[str] = frozenset({
    ".", ",", ";", ":", "-", "–", "—", "/",
    "(", ")", "[", "]", "{", "}", "<", ">",
    '"', "'", "`", "\\", "|", "^", "~",
})

VALID_SHORT_SENTENCES: frozenset[str] = frozenset({
    # Affirmatives/Negatives
    "yes.", "no.", "yeah.", "nah.", "yep.", "nope.", "okay.", "ok.",
    "sure.", "fine.", "right.", "wrong.", "true.", "false.",
    # Questions
    "why?", "how?", "what?", "when?", "where?", "who?", "which?",
    "really?", "seriously?", "honestly?",
    # Commands/Exclamations
    "go.", "stop.", "wait.", "run.", "help!", "look!", "listen!",
    "come.", "stay.", "leave.", "move.", "watch.",
    # Responses
    "i do.", "i am.", "i can.", "i will.", "i did.", "i know.",
    "he did.", "she did.", "we did.", "they did.",
    "it is.", "it was.", "it does.", "that's right.", "that's true.",
    # Emotional
    "oh.", "ah.", "wow.", "ouch.", "damn.", "god.",
    # Discourse markers
    "indeed.", "exactly.", "absolutely.", "certainly.", "definitely.",
    "perhaps.", "maybe.", "possibly.", "probably.",
    "however.", "therefore.", "moreover.", "furthermore.",
    "anyway.", "besides.",
    # Frequency adverbs (from DISCOURSE_OK integration)
    "often.", "always.", "never.", "sometimes.",
    # Clarity markers (from DISCOURSE_OK integration)
    "clearly.", "obviously.", "thus.",
    # Phrase markers (from DISCOURSE_OK integration)
    "in contrast.", "for example.", "for instance.",
})

CODE_FONT_HINTS: tuple[str, ...] = (
    "courier", "mono", "consolas", "menlo", "monospace"
)

# ✦───────────── 5 COMPILED REGEX PATTERNS ─────────────✦

# --- Text Classification ---

_PROSE_WORDS_PATTERN: re.Pattern[str] = re.compile(
    r"\b("
    r"is|are|was|were|be|been|being|"
    r"the|a|an|this|that|these|those|"
    r"in|on|at|by|for|with|from|to|of|"
    r"and|or|but|so|yet|nor|"
    r"have|has|had|having|"
    r"do|does|did|"
    r"can|could|will|would|may|might|must|shall|should|"
    r"not|no|"
    r"if|then|when|where|which|who|whom|whose|what|how|why|"
    r"as|than|"
    r"it|its|they|their|them|we|our|us|you|your|he|his|him|she|her|"
    r"also|only|just|even|still|already|never|always|often|"
    r"very|more|most|less|least|much|many|few|some|any|all|"
    r"such|same|other|another|each|every|both|either|neither|"
    r"about|between|during|through|within|without|under|over|"
    r"called|known|named|termed|used|found|shown|described|"
    r"receptors?|neurons?|cells?|tissue|muscle|skin|nerve|fibers?"
    r")\b",
    re.IGNORECASE,
)

_MATH_SYMBOLS_PATTERN: re.Pattern[str] = re.compile(
    r"[=±×÷∑∫∏√∞≈≠≡≤≥∈∉⊂⊃∪∩∂∇"
    r"αβγδεζηθλμπσφωΔΣΩ"
    r"²³⁴⁵⁶⁷⁸⁹⁰₀₁₂₃₄₅₆₇₈₉]"
)
_LABEL_TECHNICAL_TERMS: frozenset[str] = frozenset({
    # Control systems terminology
    'signal', 'control', 'feedback', 'force', 'length', 'velocity',
    'error', 'driving', 'input', 'output', 'gain', 'loop',
    # Neuroscience terminology
    'afferents', 'efferents', 'neuron', 'neurons', 'receptor',
    'receptors', 'muscle', 'spindle', 'tendon', 'organ', 'interneurons',
    # Greek letter designators (common in scientific diagrams)
    'primary', 'secondary', 'alpha', 'beta', 'gamma', 'delta',
})

_LABEL_PROSE_INDICATORS: frozenset[str] = frozenset({
    # Articles (indicate prose, not labels)
    'the', 'a', 'an',
    # Demonstratives (indicate prose context)
    'this', 'that', 'these', 'those',
    # Pronouns (indicate prose flow)
    'it', 'they', 'we', 'he', 'she', 'there', 'here',
    # Verbs (labels don't contain verbs)
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had',
    'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must',
    'do', 'does', 'did',
    'shows', 'displays', 'indicates', 'represents', 'contains',
    'demonstrates', 'illustrates', 'provides', 'includes', 'consists',
})

GREEK_WORD_FORMS: frozenset[str] = frozenset({
    "alpha", "beta", "gamma", "delta", "epsilon",
    "theta", "lambda", "mu", "sigma", "omega", "pi",
})

_PARA_ABBREVIATIONS: frozenset[str] = frozenset({
    "dr", "mr", "ms", "mrs", "prof", "fig", "eq",
    "st", "vs", "etc", "e.g", "i.e", "cf", "al",
})

# --- Layout Detection ---

_LABEL_PATTERN: re.Pattern[str] = re.compile(
    r"^("
    r"[A-Z]|"
    r"[xyz]|"
    r"\d+\.?\d*|"
    r"\d+%|"
    r"[A-Z]\d*|"
    r"\([a-z]\)|"
    r"[ivxIVX]+|"
    r"[xyz][\s\-]?axis|"
    r"n\s*=\s*\d+|"
    r"p\s*[<>=]\s*[\d.]+|"
    r"r\s*=\s*[\d.]+|"
    r"\*+|"
    r"ns|NS"
    r")$",
    re.IGNORECASE,
)

# --- Statistical Label Pattern (p-values, n-values, correlations) ---

_STATISTICAL_LABEL_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:"
    r"n\s*=\s*\d+|"
    r"p\s*[<>=]\s*[\d.]+|"
    r"r\s*=\s*[\d.]+|"
    r"t\s*=\s*[\d.]+|"
    r"F\s*=\s*[\d.]+|"
    r"df\s*=\s*\d+|"
    r"(?:min|max|avg|mean|std|sd|se)\.?"
    r")$",
    re.IGNORECASE
)

# --- Role Code Punctuation Pattern ---

_ROLE_CODE_PUNCT_PATTERN: re.Pattern[str] = re.compile(r"[{}[\]()<>;=+\-*/&|^~]")

# --- Cross-Page Stitching Patterns ---

_STITCH_TERMINAL_PUNCT_PATTERN: re.Pattern[str] = re.compile(
    r"[.!?][\"')\]]*$"
)

# --- Text Normalization (for header/footer comparison) ---

_NORMALIZE_DIGIT_PATTERN: re.Pattern[str] = re.compile(r"\d+")
_NORMALIZE_PUNCT_PATTERN: re.Pattern[str] = re.compile(r"[^\w\s]")

# --- Superscript Ordinal Patterns ---

_TTS_ORDINAL_1ST_PATTERN: re.Pattern[str] = re.compile(r"(\d+)ˢᵗ")
_TTS_ORDINAL_2ND_PATTERN: re.Pattern[str] = re.compile(r"(\d+)ⁿᵈ")
_TTS_ORDINAL_3RD_PATTERN: re.Pattern[str] = re.compile(r"(\d+)ʳᵈ")
_TTS_ORDINAL_NTH_PATTERN: re.Pattern[str] = re.compile(r"(\d+)ᵗʰ")

# --- Mathematical Superscript Patterns ---

_TTS_SQUARED_PATTERN: re.Pattern[str] = re.compile(r"(\w+)²")
_TTS_CUBED_PATTERN: re.Pattern[str] = re.compile(r"(\w+)³")
_TTS_UNIT_SLASH_PATTERN: re.Pattern[str] = re.compile(r'(\d+(?:\.\d+)?)\s*/\s*([a-zA-Z²³µμ]+\d*)')

# --- List/Item Detection ---

_LIST_ITEM_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:"
    r"[\u2022\u2023\u25E6\u2043\u2219•◦▪▸►-]\s"
    r"|\d+[\.\)]\s"
    r"|[IVXLCDM]+\.\s"
    r"|\([a-z]\)\s"
    r"|\([A-Z]\)\s"
    r"|\([0-9]+\)\s"
    r")"
)

# ===========================================================================
# BIBLIOGRAPHY / REFERENCE BLOCK DETECTION (cluster-based, no magic headings)
# Used by _refine_roles_via_content_flow() to demote confirmed reference blocks
# into TextRole.FOOTNOTE (already non-viable for TTS).
# ===========================================================================

_BIBLIO_HEADING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"^\s*references\s*$",
        r"^\s*bibliography\s*$",
    )
)

_BIBLIO_ENTRY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in (
        r"^\d{1,3}\.\s+[A-Z]",  # "1. Author..."
        r"^\[\d{1,3}\]\s*[A-Z]",  # "[1] Author..."
        r"^[A-Z][a-z]{1,15},\s+[A-Z]\.",  # "Smith, J."
    )
)

_BIBLIO_CONTINUATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"\(\d{4}[a-z]?\)",  # (2022) or (2022a)
        r"\bpp?\.\s*\d",  # pp. 123 / p. 45
        r"\bvol\.\s*\d",  # vol. 4
        r"\bno\.\s*\d",  # no. 2
        r"\bdoi\s*:\s*",  # doi:
        r"\b10\.\d{4,9}/",  # DOI prefix
        r"https?://",  # URLs
        r"\bwww\.",  # URLs
    )
)

_BIBLIO_JOURNAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in (
        r"\b[A-Z][a-z]{1,8}\.\s+[A-Z]",  # "Proc. ACM", "J. Comp."
        r"\bTrans\.\s+[A-Z]",  # "Trans. Vis."
        r"\bInt\.\s+J\.",  # "Int. J."
    )
)

_BIBLIO_MIN_CLUSTER_SIZE: int = 3
_BIBLIO_CLUSTER_SCORE_THRESHOLD: float = 0.4
_BIBLIO_CLUSTER_RATIO_THRESHOLD: float = 0.55
_BIBLIO_PUNCT_DENSITY_THRESHOLD: float = 0.08
_BIBLIO_SMALL_FONT_RATIO: float = 0.94

# --- Cleaning Patterns ---

_EMPTY_BRACKETS_PATTERN: re.Pattern[str] = re.compile(r"[\(\[\{]\s*[\)\]\}]")
_PUNCT_ONLY_BRACKETS_PATTERN: re.Pattern[str] = re.compile(r"[\(\[]\s*[,;:.]\s*[\)\]]")
_WHITESPACE_PATTERN: re.Pattern[str] = re.compile(r"\s+")

# Collapses whitespace immediately preceding sentence-terminal punctuation.
_PRE_PUNCT_WHITESPACE_PATTERN: re.Pattern[str] = re.compile(r'\s+([,.;:])')

_CHEMICAL_SUBSCRIPT_PATTERN: re.Pattern[str] = re.compile(r"([A-Za-z])([₀-₉]+)")

# --- Cleaning Patterns ---

# Safe Control Chars (Keep \n \t \r, remove others)
_CLEAN_CONTROL_CHAR_PATTERN: re.Pattern[str] = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

# Quote/Dash Normalization Table
# Maps fancy quotes/dashes to standard ASCII for better TTS compatibility.
_CLEAN_TRANS_TABLE: dict[int, str] = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": ",", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2013": "-", "\u2014": " - ", "\u2026": "..."
})

_CLEAN_EMPTY_BRACKETS_PATTERN: re.Pattern[str] = re.compile(
    r"[\(\[\{<]\s*[\)\]\}>]"
)

_CLEAN_ORPHAN_PUNCT_PATTERN: re.Pattern[str] = re.compile(
    r"\(\s*[,;.]\s*\)"
)

_CLEAN_LEADING_ORPHAN_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*[,;]\s*"
)

_CLEAN_TRAILING_ORPHAN_PATTERN: re.Pattern[str] = re.compile(
    r"\s*[,;]\s*$"
)

# --- Code Detection Patterns ---

_CODE_PUNCT_PATTERN: re.Pattern[str] = re.compile(r"[{}[\]();<>/=]")

# --- Label Detection Patterns ---

_LABEL_SENTENCE_END_PATTERN: re.Pattern[str] = re.compile(r"[.!?]$")

# --- Caption Detection (Compiled) ---

_COMPILED_CAPTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in [
        r"^Fig(?:ure)?\.?\s*\d+",
        r"^Table\.?\s*\d+",
        r"^Chart\.?\s*\d+",
        r"^Graph\.?\s*\d+",
        r"^Exhibit\.?\s*\d+",
        r"^Plate\.?\s*\d+",
        r"^Diagram\.?\s*\d+",
        r"^Photo(?:graph)?\.?\s*\d+",
        r"^Image\.?\s*\d+",
        r"^Map\.?\s*\d+",
        r"^Scheme\.?\s*\d+",
        r"^Panel\.?\s*[A-Z]",
        r"^Source\s*:",
        r"^Note\s*:",
        r"^Notes?\s*:",
        r"^\(?[a-z]\)\s*$",
        r"^[A-Z]\.\s+",
    ]
)

# Prose continuation check (for bare caption filtering)
_CAPTION_PROSE_CONTINUATION_PATTERN: re.Pattern[str] = re.compile(
    r"[:\.]\s+\S"
)

_RAWDICT_WORD_GAP_RATIO = 0.10

# Fallback font size when span metadata is missing or zero.
_RAWDICT_FALLBACK_FONT_SIZE = 10.0

# ✦───────────── 6 DERIVED CONSTANTS ─────────────✦

# Glyph-level word boundary detection (rawdict mode)
_SUBSCRIPT_TABLE: dict[int, str] = str.maketrans(SUBSCRIPT_DIGITS)
_SUPERSCRIPT_TABLE: dict[int, str] = str.maketrans(SUPERSCRIPT_DIGITS)

# ✦───────────── 7 MODULE INITIALIZATION ─────────────✦

# Lazy-loaded sentence segmenter (initialized on first use)
_SENTENCE_SEGMENTER: Optional[pysbd.Segmenter] = None

# Connector words that cannot terminate a sentence meaningfully
_STITCH_CONNECTOR_WORDS: frozenset[str] = frozenset({
    "and", "or", "but", "nor", "so", "yet", "for",
    "to", "of", "in", "with", "by", "at", "from",
})

# Words that, when prev ends with them, allow capitalized next to merge
_STITCH_PREV_CONNECTOR_WORDS: frozenset[str] = frozenset({
    "and", "or", "of", "the", "a", "an", "in", "on",
    "at", "to", "for", "with", "by"
})

# ═══════════════════════════════════════════════════════════════════════════
# RONC v1.0 — Reading Order Normalization Contract
# ═══════════════════════════════════════════════════════════════════════════

# Boundary profiling thresholds
_RONC_V2_NEEDS_PREDECESSOR_SCORE: float = 0.30
_RONC_V2_NEEDS_SUCCESSOR_SCORE: float = 0.25

# Signal weights (start boundary)
_RONC_V2_WEIGHT_LOWERCASE: float = 0.35
_RONC_V2_WEIGHT_DEPENDENT_TOKEN: float = 0.30
_RONC_V2_WEIGHT_CONTINUATION_PUNCT: float = 0.25
_RONC_V2_WEIGHT_A2_FROM: float = 0.40

# Signal weights (end boundary)
_RONC_V2_WEIGHT_SENTENCE_TERMINAL: float = -0.40  # Negative: reduces truncation score
_RONC_V2_WEIGHT_TRAILING_COMMA: float = 0.30
_RONC_V2_WEIGHT_TRAILING_SEMICOLON: float = 0.25
_RONC_V2_WEIGHT_TRAILING_COLON: float = 0.20
_RONC_V2_WEIGHT_TRAILING_DASH: float = 0.35
_RONC_V2_WEIGHT_UNBALANCED_OPEN: float = 0.40
_RONC_V2_WEIGHT_A2_TO: float = 0.40

# Dependent tokens that suggest continuation from previous context
_RONC_V2_DEPENDENT_TOKENS: frozenset[str] = frozenset({
    "and", "or", "but", "nor", "yet", "so",
    "which", "that", "who", "whom", "whose", "where", "when",
    "however", "therefore", "thus", "hence",
    "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "nonetheless", "although", "though",
    "whereas", "while", "since", "because", "unless", "until",
    "after", "before", "if", "as",
})

# Roles excluded from candidate pools (low semantic value)
_RONC_V2_EXCLUDED_ROLES: frozenset[str] = frozenset({
    TextRole.PAGE_NUMBER.value,
    TextRole.HEADER_ARTIFACT.value,
    TextRole.FOOTER_ARTIFACT.value,
    TextRole.FOOTNOTE.value,
})

_NUMBER_DECOMPOUND_PATTERN: re.Pattern[str] = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(one|two|three|four|five|six|seven|eight|nine)\b",
    re.IGNORECASE,
)
# Roles that CAN be candidates but with reduced priority (Phase 3 scoring)
_RONC_V2_DEPRIORITIZED_ROLES: frozenset[str] = frozenset({
    TextRole.TABLE_CELL.value,
    TextRole.DIAGRAM_LABEL.value,
    TextRole.FIGURE_LABEL.value,
    TextRole.CAPTION.value,
})

# Maximum window distance for candidate consideration
# Spans further apart than this are not considered candidates
_RONC_V2_MAX_CANDIDATE_DISTANCE: int = 15

# Window positions eligible for cross-page linking
_RONC_V2_CROSS_PAGE_POSITIONS: frozenset[str] = frozenset({
    "prev_tail",  # Previous page spans can link TO current
    "current",  # Current page spans (primary)
    "next_head",  # Next page spans can link FROM current
})

# Minimum score required to create a link
_RONC_V2_MIN_LINK_CONFIDENCE: float = 0.50

# Score threshold for "strong" links (skip mutual check)
_RONC_V2_STRONG_LINK_CONFIDENCE: float = 0.80

# Confidence boost when link is mutual (bidirectional)
_RONC_V2_MUTUAL_BOOST: float = 0.10

# Maximum confidence after boost (cap at 1.0)
_RONC_V2_MAX_CONFIDENCE: float = 1.0

# Minimum link confidence to trigger protection
_RONC_V2_PROTECTION_THRESHOLD: float = 0.60

_REASON_DOCUMENT_METADATA: str = "document_metadata"

# ════════════════════════════════════════════════════════════════
# STRUCTURAL AUTHORITY CONSTANTS
# Span-level exclusion domains that override rescue and contract
# logic. Enforces the authority hierarchy:
#   Structural > Contractual (RONC) > Rescue > Default
# ════════════════════════════════════════════════════════════════

_STRUCTURAL_EXCLUSION_REASONS: frozenset[str] = frozenset({
    _REASON_DOCUMENT_METADATA,
    _REASON_PUBLISHER_METADATA,
    _REASON_HEADER_BAND,
    _REASON_FOOTER_BAND,
    _REASON_HEADER_ARTIFACT,
})

_STRUCTURAL_EXCLUSION_OUTLIER_REASONS: frozenset[str] = frozenset({
    "bibliography_heading",
    "bibliography_heading_spurious",
    "bibliography_cluster",
})

# RONC veto reasons now alias structural exclusion reasons
# (identical set; centralized source of truth)
_RONC_V2_PROTECTION_VETO_REASONS = _STRUCTURAL_EXCLUSION_REASONS

# Protection reasons (for audit trail)
_RONC_V2_PROTECTION_ANCHOR: str = "anchor" # Has outgoing link
_RONC_V2_PROTECTION_CONTINUATION: str = "continuation"  # Has incoming link
_RONC_V2_PROTECTION_MUTUAL: str = "mutual_link"  # Bidirectional link

# Minimum linked unit size to assign unit ID
# Single-span "units" don't need atomic grouping
_RONC_V2_MIN_UNIT_SIZE = 2

# Legacy role values (for backward compatibility)
_RONC_V2_LEGACY_ROLE_ANCHOR: str = "anchor"
_RONC_V2_LEGACY_ROLE_MEMBER: str = "member"
_RONC_V2_LEGACY_ROLE_TAIL: str = "tail"

# Add to constants block (if not present)
_RONC_V2_TOP_K_CANDIDATES: int = 3

# Scoring weights (per spec)
_RONC_V2_WEIGHT_SEMANTIC_FLOW: float = 0.65
_RONC_V2_WEIGHT_PROXIMITY: float = 0.35

# TTS NARRATION ADMISSION GATE — CONSTANTS


_TTS_HARD_GATE_PROMO_KEYWORDS: tuple[str, ...] = (
    "css4.pub",
    "wikibook",
    "sample document",
    "showcase",
    "html and css",
    "paper-based publishing",
    "page-based formatting",
)

_TTS_HARD_GATE_SHORT_WORD_MAX_CHARS: int = 4

_TTS_HARD_GATE_MARGIN_STREAM_PREFIX: str = "margin_"

_TTS_HARD_GATE_SHORT_WORD_ALLOWED_ROLES: tuple[str, ...] = (
    TextRole.HEADING.value,
    TextRole.SUBHEADING.value,
    TextRole.BODY.value,
)

# ═══════════════════════════════════════════════════════════════════════════════
# RONC v2.1: Evidence-Gated Expansion Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Expanded search distance when semantic evidence suggests continuation
_RONC_V2_EXPANDED_CANDIDATE_DISTANCE: int = 30  # 2x base

# Boundary score thresholds to trigger expansion
_RONC_V2_EXPANSION_THRESHOLD_START: float = 0.30 # needs_predecessor trigger
_RONC_V2_EXPANSION_THRESHOLD_END: float = 0.25  # needs_successor trigger

# Pool size thresholds
_RONC_V2_MIN_POOL_STRICT: int = 1  # Below this triggers expansion consideration
_RONC_V2_MIN_POOL_EXPANDED: int = 3  # Fallback: stop expanding once we have this many
_RONC_V2_EXPANSION_QUALITY_DISTANCE: int = 3  # Stop early if we find candidate this close

# Effective distance shaping factors
_RONC_V2_BLOCK_ADJACENT_EFFECTIVE_DISTANCE: int = 1  # Same block, adjacent line → near-adjacent
_RONC_V2_CROSS_COLUMN_DISTANCE_REDUCTION: int = 2 # Cross-column wrap softening
_RONC_V2_BLOCK_EDGE_DISTANCE_REDUCTION: int = 3  # Block-edge pair bonus

# Phase 3 scoring adjustments (refinement layer)
_RONC_V2_EXPANDED_CANDIDATE_PENALTY: float = 0.85  # Expanded must outscore strict
_RONC_V2_CROSS_COLUMN_PENALTY: float = 0.80  # Soft penalty, not rejection
_RONC_V2_BLOCK_EDGE_BONUS: float = 0.10  # Structural continuation hint

# Role ordering priority for span sorting (lower = higher priority)
# Read-only lookup table — do not mutate at runtime
_RONC_ROLE_PRIORITY: dict[str | None, int] = {
    "anchor": 0,
    "member": 1,
    "tail": 2,
    None: 3,
}

# ═══════════════════════════════════════════════════════════════════════════════
# RONC v2.0 Boundary Profile Labels & Thresholds
# ═══════════════════════════════════════════════════════════════════════════════

# Start boundary labels (distinct from _RONC_V2_PROTECTION_CONTINUATION)
_RONC_V2_START_LABEL_CLEAN: str = "clean"
_RONC_V2_START_LABEL_CONTINUATION: str = "continuation"
_RONC_V2_START_LABEL_FRAGMENT: str = "fragment"
_RONC_V2_START_LABEL_UNKNOWN: str = "unknown"

# End boundary labels
_RONC_V2_END_LABEL_COMPLETE: str = "complete"
_RONC_V2_END_LABEL_TRUNCATED: str = "truncated"
_RONC_V2_END_LABEL_MID_SENTENCE: str = "mid_sentence"
_RONC_V2_END_LABEL_UNKNOWN: str = "unknown"

# Label assignment thresholds
_RONC_V2_START_CONTINUATION_THRESHOLD: float = 0.60
_RONC_V2_START_FRAGMENT_THRESHOLD: float = 0.30
_RONC_V2_END_TRUNCATED_THRESHOLD: float = 0.50
_RONC_V2_END_MID_SENTENCE_THRESHOLD: float = 0.25

# Contract version
_RONC_V2_CONTRACT_VERSION: str = "2.0"

# RONC v2.0 Semantic Flow Scoring Weights

# Boundary component weights (sum to 1.0)
_RONC_V2_SCORE_END_WEIGHT: float = 0.50
_RONC_V2_SCORE_START_WEIGHT: float = 0.50

# Label compatibility bonuses/penalties
_RONC_V2_SCORE_MATCH_IDEAL_BONUS: float = 0.20
_RONC_V2_SCORE_MATCH_GOOD_BONUS: float = 0.10
_RONC_V2_SCORE_MATCH_CONFLICT_PENALTY: float = -0.10

# A2 signal agreement bonuses
_RONC_V2_SCORE_A2_ALIGNED_BONUS: float = 0.15
_RONC_V2_SCORE_A2_PARTIAL_BONUS: float = 0.05

# Score clamping
_RONC_V2_SCORE_SEMANTIC_MAX: float = 0.90
_RONC_V2_SCORE_PROXIMITY_MAX: float = 1.00

# ═══════════════════════════════════════════════════════════════════════════════
# RONC v2.0 Effective Distance & Proximity Hints
# ═══════════════════════════════════════════════════════════════════════════════

# Stream prefix for body column detection
_RONC_V2_BODY_COLUMN_PREFIX: str = "body_col_"

# Block start detection threshold (line_index <= this = start of block)
_RONC_V2_BLOCK_START_LINE_THRESHOLD: int = 1

# Minimum effective distances
_RONC_V2_MIN_EFFECTIVE_DISTANCE: int = 1
_RONC_V2_MIN_CROSS_COLUMN_DISTANCE: int = 2

# Distance bias adjustments
_RONC_V2_BIAS_ADJACENT_LINE: float = 0.08
_RONC_V2_BIAS_SAME_BLOCK: float = 0.05
_RONC_V2_BIAS_CROSS_COLUMN: float = -0.05
_RONC_V2_BIAS_BLOCK_EDGE: float = 0.10
# NOTE: Block edge bias uses existing _RONC_V2_BLOCK_EDGE_BONUS (0.10)

# Bias clamping range
_RONC_V2_BIAS_MIN: float = -0.10
_RONC_V2_BIAS_MAX: float = 0.20

# ═══════════════════════════════════════════════════════════════════════════════
# RONC v2.0 Proximity Hint Labels
# ═══════════════════════════════════════════════════════════════════════════════

_RONC_V2_HINT_SAME_PAGE: str = "same_page"
_RONC_V2_HINT_CROSS_PAGE: str = "cross_page"
_RONC_V2_HINT_SAME_BLOCK: str = "same_block"
_RONC_V2_HINT_ADJACENT_LINE: str = "adjacent_line"
_RONC_V2_HINT_CROSS_COLUMN: str = "cross_column"
_RONC_V2_HINT_BLOCK_EDGE: str = "block_edge"
_RONC_V2_HINT_EXPANSION_TRIGGER: str = "expansion_trigger"
_RONC_V2_WINDOW_POS_PREV_TAIL: str = "prev_tail"
_RONC_V2_WINDOW_POS_NEXT_HEAD: str = "next_head"
# Candidate pool mode labels
_RONC_V2_MODE_STRICT: str = "strict"
_RONC_V2_MODE_EXPANDED: str = "expanded"


# After _RONC_V2_SCORE_A2_PARTIAL_BONUS (or _RONC_V2_MAX_CONFIDENCE):
_RONC_V2_HIGH_CONFIDENCE_THRESHOLD: float = 0.70

# Proximity multipliers block:
_RONC_V2_PROX_PENALTY_DEPRIORITIZED: float = 0.70
_RONC_V2_PROX_BONUS_SAME_PAGE: float = 1.10
_RONC_V2_PROX_BONUS_SAME_BLOCK: float = 1.15
_RONC_V2_PROX_BONUS_SAME_LINE: float = 1.10
_RONC_V2_PROX_BONUS_ADJACENT_LINE: float = 1.05

# Authority levels (Phase 7)
_RONC_V2_AUTHORITY_STRONG: str = "strong"
_RONC_V2_AUTHORITY_WEAK: str = "weak"
_RONC_V2_AUTHORITY_NONE: str = "none"

# Authority ranking
_RONC_V2_AUTHORITY_RANK: dict = {
    _RONC_V2_AUTHORITY_STRONG: 3,
    _RONC_V2_AUTHORITY_WEAK: 2,
    _RONC_V2_AUTHORITY_NONE: 1,
}

# Authority numeric mapping
_RONC_V2_AUTHORITY_NUMERIC: dict = {
    _RONC_V2_AUTHORITY_STRONG: 2,
    _RONC_V2_AUTHORITY_WEAK: 1,
    _RONC_V2_AUTHORITY_NONE: 0,
}

# Structural disqualifiers
_RONC_V2_DISQ_CROSS_STREAM: str = "cross_stream"
_RONC_V2_DISQ_MISSING_STREAM: str = "missing_layout_stream"

# =============================================================================
# PROSODIC CLAUSE SPLITTING CONSTANTS (v1.5.0 - TTS Decoder Safety)
# =============================================================================
# Thresholds and patterns for splitting clause-dense sentences into
# decoder-safe segments for audio-level TTS generation.
#
# ARCHITECTURAL NOTE:
#   Clause metadata is computed in _sanitize_for_tts() and stored in change_tracker.
#   Actual audio splitting + concatenation happens in process.py at TTS generation time.
#   Return type of _sanitize_for_tts() remains str (no breaking changes).

_TTS_PROSODIC_SPLIT_CHAR_THRESHOLD: int = 120  # Match existing decoder risk trigger
_TTS_PROSODIC_SPLIT_COMMA_THRESHOLD: int = 1  # Match existing decoder risk trigger
_TTS_PROSODIC_MIN_CLAUSE_WORDS: int = 4  # Safety: no tiny fragments
_TTS_PROSODIC_MAX_CLAUSES: int = 4  # Safety: cap clause count

_TTS_MONOTONE_SPLIT_CONJUNCTIONS: frozenset = frozenset({
    'and', 'but', 'or', 'nor', 'yet', 'so',
    'because', 'since', 'although', 'though',
    'while', 'whereas', 'unless',
    'however', 'therefore', 'nevertheless',
})
_TTS_PROACTIVE_SPLIT_CHARS: int = 130

# Clause boundary patterns (ordered by prosodic break strength)
# These patterns split AT the boundary, keeping the connector with the following clause
_TTS_PROSODIC_CLAUSE_PATTERNS: Tuple[re.Pattern, ...] = (
    # Priority 1: Relative/subordinate clauses (strongest break)
    re.compile(r",\s+(which|that|who|whom|where)\s+", re.IGNORECASE),
    # Priority 1.5: Subordinating conjunctions (causal/concessive/conditional)
    re.compile(
        r",\s+(because|since|although|unless|whereas|while|if)\s+",
        re.IGNORECASE
    ),    # Priority 2: Discourse markers
    re.compile(r",\s+(however|therefore|moreover|nevertheless|in contrast|by contrast)\s+",
               re.IGNORECASE),
    # Priority 3: Participial/state verb phrases (common in academic text)
    re.compile(r",\s+(arising from|consisting of|including|is also|can be|are also)\s+",
               re.IGNORECASE),
    # Priority 4: Conjunctions (weakest - use only as fallback)
    re.compile(r",\s+(and|or|but)\s+", re.IGNORECASE),
)

# ✦────────────────────✦────────────────────✦
#                ✿   METHODS  ✿
# ✦────────────────────✦────────────────────✦

def _contains_greek(text: str) -> bool:
    """
    Return True if text contains at least one Unicode Greek character.
    Greek Unicode block: U+0370–U+03FF
    """
    if not text:
        return False
    return any('\u0370' <= ch <= '\u03FF' for ch in text)

# ✦                  ✦                  ✦                  ✦
# ✦──────── 1 RONC v2.0 — Contract-Maker Architecture ───────✦
# ✦                  ✦                  ✦                  ✦

def _ronc_v2_profile_start_boundary(
        span: Dict,
        prev_span: Optional[Dict] = None,
        trace_id: Optional[str] = None,
) -> Dict:
    """
    Profile how a span BEGINS to determine if it needs a predecessor.

    RONC v2.0 Phase 1: Semantic Boundary Profiling (Node Analysis)

    Analyzes lexical and structural signals at span start to determine
    whether this span is likely a continuation of prior content.

    Args:
        span: Span dict with cleaned_text or text field
        prev_span: Previous span in window (optional, for future use)
        trace_id: Trace ID for logging

    Returns:
        Dict with keys:
            label: "clean" | "continuation" | "fragment" | "unknown"
            score: float 0.0-1.0 (higher = more likely continuation)
            signals: list of signal names that fired
    """
    # NOTE: prev_span and trace_id are intentionally unused in v2.0.
    # They are reserved for future boundary heuristics and diagnostics.
    _ = prev_span, trace_id

    text = (span.get("cleaned_text") or span.get("text") or "").strip()
    signals = []
    score = 0.0

    # Empty span: unknown boundary
    if not text:
        return {"label": _RONC_V2_START_LABEL_UNKNOWN, "score": 0.0, "signals": []}

    first_char = text[0]
    words = text.split()
    first_word = words[0].lower().rstrip(",.;:") if words else ""

    # ─────────────────────────────────────────────────────────────────
    # Signal: Lowercase start
    # A span starting with lowercase letter suggests mid-sentence entry
    # ─────────────────────────────────────────────────────────────────
    if first_char.isalpha() and first_char.islower():
        signals.append("lowercase")
        score += _RONC_V2_WEIGHT_LOWERCASE

    # ─────────────────────────────────────────────────────────────────
    # Signal: Dependent token
    # Conjunctions, relative pronouns, and transitional words suggest
    # this span continues a thought from a previous span
    # ─────────────────────────────────────────────────────────────────
    if first_word in _RONC_V2_DEPENDENT_TOKENS:
        signals.append("dependent_token")
        score += _RONC_V2_WEIGHT_DEPENDENT_TOKEN

    # ─────────────────────────────────────────────────────────────────
    # Signal: Continuation punctuation
    # Starting with comma, semicolon, closing bracket, etc.
    # ─────────────────────────────────────────────────────────────────
    if first_char in ",;:—–)]\"}'>":
        signals.append("continuation_punct")
        score += _RONC_V2_WEIGHT_CONTINUATION_PUNCT

    # ─────────────────────────────────────────────────────────────────
    # Signal: A2 continuation flag (from existing pipeline)
    # Highest confidence signal when present
    # ─────────────────────────────────────────────────────────────────
    if span.get("a2_continues_from_previous"):
        signals.append("a2_continues_from")
        score += _RONC_V2_WEIGHT_A2_FROM

    # Cap score at 1.0
    score = min(score, 1.0)

    # ─────────────────────────────────────────────────────────────────
    # Label assignment based on score thresholds
    # ─────────────────────────────────────────────────────────────────
    if score >= _RONC_V2_START_CONTINUATION_THRESHOLD:
        label = _RONC_V2_START_LABEL_CONTINUATION
    elif score >= _RONC_V2_START_FRAGMENT_THRESHOLD:
        label = _RONC_V2_START_LABEL_FRAGMENT
    elif not signals:
        label = _RONC_V2_START_LABEL_CLEAN
    else:
        label = _RONC_V2_START_LABEL_UNKNOWN

    return {"label": label, "score": round(score, 3), "signals": signals}


def _ronc_v2_profile_end_boundary(
        span: Dict,
        next_span: Optional[Dict] = None,
        trace_id: Optional[str] = None,
) -> Dict:
    """
    Profile how a span ENDS to determine if it needs a successor.

    RONC v2.0 Phase 1: Semantic Boundary Profiling (Node Analysis)

    Analyzes lexical and structural signals at span end to determine
    whether this span is likely truncated or continues to next content.

    Args:
        span: Span dict with cleaned_text or text field
        next_span: Next span in window (optional, for future use)
        trace_id: Trace ID for logging

    Returns:
        Dict with keys:
            label: "complete" | "truncated" | "mid_sentence" | "unknown"
            score: float 0.0-1.0 (higher = more likely truncated)
            signals: list of signal names that fired
    """
    # NOTE: next_span and trace_id are intentionally unused in v2.0.
    # They are reserved for future boundary heuristics and diagnostics.
    _ = next_span, trace_id
    text = (span.get("cleaned_text") or span.get("text") or "").strip()
    signals = []
    score = 0.0

    # Empty span: unknown boundary
    if not text:
        return {"label": _RONC_V2_END_LABEL_UNKNOWN, "score": 0.0, "signals": []}

    last_char = text[-1]

    # ─────────────────────────────────────────────────────────────────
    # Signal: Sentence terminal (NEGATIVE weight)
    # Period, exclamation, question mark suggest completion
    # ─────────────────────────────────────────────────────────────────
    if last_char in ".!?":
        signals.append("sentence_terminal")
        score += _RONC_V2_WEIGHT_SENTENCE_TERMINAL  # Negative value

    # ─────────────────────────────────────────────────────────────────
    # Signal: Trailing comma
    # Strong indicator of incomplete clause
    # ─────────────────────────────────────────────────────────────────
    if last_char == ",":
        signals.append("trailing_comma")
        score += _RONC_V2_WEIGHT_TRAILING_COMMA

    # ─────────────────────────────────────────────────────────────────
    # Signal: Trailing semicolon
    # Suggests clause boundary but continuation expected
    # ─────────────────────────────────────────────────────────────────
    if last_char == ";":
        signals.append("trailing_semicolon")
        score += _RONC_V2_WEIGHT_TRAILING_SEMICOLON

    # ─────────────────────────────────────────────────────────────────
    # Signal: Trailing colon
    # Often precedes list or elaboration
    # ─────────────────────────────────────────────────────────────────
    if last_char == ":":
        signals.append("trailing_colon")
        score += _RONC_V2_WEIGHT_TRAILING_COLON

    # ─────────────────────────────────────────────────────────────────
    # Signal: Trailing dash
    # Interrupted thought or em-dash continuation
    # ─────────────────────────────────────────────────────────────────
    if last_char in "—–-":
        signals.append("trailing_dash")
        score += _RONC_V2_WEIGHT_TRAILING_DASH

    # ─────────────────────────────────────────────────────────────────
    # Signal: Unbalanced parentheses or quotes
    # Open paren/quote without close suggests truncation
    # ─────────────────────────────────────────────────────────────────
    open_parens = text.count("(") - text.count(")")
    open_quotes = text.count('"') % 2  # Odd count = unbalanced

    if open_parens > 0 or open_quotes:
        signals.append("unbalanced_open")
        score += _RONC_V2_WEIGHT_UNBALANCED_OPEN

    # ─────────────────────────────────────────────────────────────────
    # Signal: A2 continuation flag (from existing pipeline)
    # Highest confidence signal when present
    # ─────────────────────────────────────────────────────────────────
    if span.get("a2_continues_to_next"):
        signals.append("a2_continues_to")
        score += _RONC_V2_WEIGHT_A2_TO

    # Clamp to valid range
    score = max(0.0, min(score, 1.0))

    # ─────────────────────────────────────────────────────────────────
    # Label assignment based on score thresholds
    # ─────────────────────────────────────────────────────────────────
    if score >= _RONC_V2_END_TRUNCATED_THRESHOLD:
        label = _RONC_V2_END_LABEL_TRUNCATED
    elif score >= _RONC_V2_END_MID_SENTENCE_THRESHOLD:
        label = _RONC_V2_END_LABEL_MID_SENTENCE
    elif "sentence_terminal" in signals:
        label = _RONC_V2_END_LABEL_COMPLETE
    else:
        label = _RONC_V2_END_LABEL_UNKNOWN

    return {"label": label, "score": round(score, 3), "signals": signals}


def _ronc_v2_init_contract(
        span: Dict,
        prev_span: Optional[Dict] = None,
        next_span: Optional[Dict] = None,
        trace_id: Optional[str] = None,
) -> Dict:
    """
    Initialize a RONC v2.0 contract for a span with boundary profiling.

    RONC v2.0 Phase 1: Contract Initialization

    Creates the full contract structure with boundary analysis populated.
    Affinity, links, and protection sections are initialized to defaults
    and will be populated by subsequent phases.

    Args:
        span: Span dict to analyze
        prev_span: Previous span in window order
        next_span: Next span in window order
        trace_id: Trace ID for logging

    Returns:
        Complete contract dict ready for attachment to span
    """
    # Run boundary profiling
    start_profile = _ronc_v2_profile_start_boundary(span, prev_span, trace_id)
    end_profile = _ronc_v2_profile_end_boundary(span, next_span, trace_id)

    # Derive need flags from scores
    needs_predecessor = start_profile["score"] >= _RONC_V2_NEEDS_PREDECESSOR_SCORE
    needs_successor = end_profile["score"] >= _RONC_V2_NEEDS_SUCCESSOR_SCORE

    contract = {
        # ═══════════════════════════════════════════════════════════════
        # SECTION A: Semantic Boundary Profiling (Phase 1)
        # ═══════════════════════════════════════════════════════════════
        "boundary": {
            "start": start_profile,
            "end": end_profile,
            "needs_predecessor": needs_predecessor,
            "needs_successor": needs_successor,
        },

        # ═══════════════════════════════════════════════════════════════
        # SECTION B: Affinity Ranking (Phase 2-3, initialized empty)
        # ═══════════════════════════════════════════════════════════════
        "affinity": {
            "prev_candidates": [],
            "next_candidates": [],
        },

        # ═══════════════════════════════════════════════════════════════
        # SECTION C: Resolved Instructions (Phase 4, initialized empty)
        # ═══════════════════════════════════════════════════════════════
        "links": {
            "prev": {
                "cid": None,
                "confidence": 0.0,
                "reason": None,
                "mutual": False,
            },
            "next": {
                "cid": None,
                "confidence": 0.0,
                "reason": None,
                "mutual": False,
            },
        },

        # ═══════════════════════════════════════════════════════════════
        # SECTION D: Protection Semantics (Phase 5, initialized empty)
        # ═══════════════════════════════════════════════════════════════
        "protection": {
            "must_include": False,
            "reason": None,
            "linked_unit_cids": [],
        },

        # ═══════════════════════════════════════════════════════════════
        # SECTION E: Metadata
        # ═══════════════════════════════════════════════════════════════
        "meta": {
            "contract_version": _RONC_V2_CONTRACT_VERSION,
            "trace_id": trace_id,
        },
    }

    return contract


def _ronc_v2_empty_contract(trace_id: Optional[str] = None) -> Dict:
    """
    Return an empty/safe contract for error recovery.

    Used when contract generation fails for a span to ensure
    pipeline continues without interruption.
    """
    return {
        "boundary": {
            "start": {"label": _RONC_V2_START_LABEL_UNKNOWN, "score": 0.0, "signals": []},
            "end": {"label": _RONC_V2_END_LABEL_UNKNOWN, "score": 0.0, "signals": []},
            "needs_predecessor": False,
            "needs_successor": False,
        },
        "affinity": {
            "prev_candidates": [],
            "next_candidates": [],
        },
        "links": {
            "prev": {"cid": None, "confidence": 0.0, "reason": None, "mutual": False},
            "next": {"cid": None, "confidence": 0.0, "reason": None, "mutual": False},
        },
        "protection": {
            "must_include": False,
            "reason": None,
            "linked_unit_cids": [],
        },
        "meta": {
            "contract_version": "2.0",
            "trace_id": trace_id,
            "error": True,
        },
    }


def _ronc_v2_extract_column_from_stream(layout_stream: str) -> Optional[int]:
    """
    Extract column index from layout_stream string.

    Examples:
        "body_col_0" → 0
        "body_col_1" → 1
        "margin_right" → None
    """
    if not layout_stream or not isinstance(layout_stream, str):
        return None

    if "_col_" in layout_stream:
        try:
            parts = layout_stream.split("_col_")
            if len(parts) >= 2:
                col_part = parts[1].split("_")[0]
                return int(col_part)
        except (ValueError, IndexError):
            pass

    return None



def _ronc_v2_is_candidate_eligible(
        span: Dict,
        exclude_excluded: bool = False,  # CHANGED: Safe default for MVP
        trace_id: Optional[str] = None,
) -> bool:
    """
    Determine if a span is eligible to be a link candidate.

    RONC v2.0 Phase 2: Candidate Eligibility Filter

    A span is eligible if:
    - It has meaningful text content
    - Its role is not in the excluded set (page numbers, headers, etc.)
    - It is not already marked for TTS exclusion (optional, disabled by default)

    Args:
        span: Span dict to evaluate
        exclude_excluded: If True, spans with _tts_excluded=True are ineligible.
                         Default False because RONC runs before exclusion translation.
        trace_id: Trace ID for logging

    Returns:
        True if span can be a candidate, False otherwise
    """
    # NOTE: trace_id is intentionally unused in v2.0.
    # Reserved for future eligibility diagnostics and audit logging.
    _ = trace_id
    # Must have text content
    text = (span.get("cleaned_text") or span.get("text") or "").strip()
    if not text:
        return False

    # Exclude certain roles entirely
    role = span.get("role")
    if role in _RONC_V2_EXCLUDED_ROLES:
        return False

    # Optional: exclude spans already marked for TTS exclusion
    # NOTE: Disabled by default because RONC runs BEFORE exclusion translation.
    # Enable in Phase 5+ when exclusion state is reliable.
    if exclude_excluded and span.get("_tts_excluded"):
        return False

    # Must have valid window position
    window_pos = span.get("_window_position")
    if window_pos is None:
        # Defensive: missing window metadata means we cannot safely link.
        return False
    if window_pos not in _RONC_V2_CROSS_PAGE_POSITIONS:
        return False

    return True



def _ronc_v2_is_block_edge_pair(
        earlier_span: Dict,
        later_span: Dict,
) -> bool:
    """
    Detect if two spans form a block-edge pair (end of one block → start of next).

    Strong structural hint of continuation across column break.
    """
    # Must be on same page
    if earlier_span.get("_source_page_idx") != later_span.get("_source_page_idx"):
        return False

    # Must be different blocks
    earlier_block = earlier_span.get("block_id")
    later_block = later_span.get("block_id")

    if earlier_block is None or later_block is None or earlier_block == later_block:
        return False

    # Later span should be at start of its block (line 0 or 1)
    later_line = later_span.get("line_index")
    if later_line is not None and later_line <= _RONC_V2_BLOCK_START_LINE_THRESHOLD:
        return True

    return False


def _ronc_v2_compute_effective_distance(
        current_span: Dict,
        candidate_span: Dict,
        raw_distance: int,
        *,
        current_idx: int,
        candidate_idx: int,
) -> Tuple[int, float, List[str]]:
    """
    Compute structure-aware effective distance for candidate ranking.

    Lower effective distance → higher proximity score in Phase 3.
    This is where "weighted variable logic" lives upstream.

    Returns:
        Tuple of (effective_distance: int, distance_bias: float, hints: List[str])

        distance_bias is an additive proximity score adjustment (0.0-0.2 range)
        applied in Phase 3 for fine-grained control without integer cliffs.
    """
    eff = raw_distance
    hints = []

    # ─────────────────────────────────────────────────────────────────
    # Factor 1: Same page baseline
    # ─────────────────────────────────────────────────────────────────
    same_page = (current_span.get("_source_page_idx") == candidate_span.get("_source_page_idx"))

    if not same_page:
        hints.append(_RONC_V2_HINT_CROSS_PAGE)
        return max(_RONC_V2_MIN_EFFECTIVE_DISTANCE, eff), 0.0, hints
    hints.append(_RONC_V2_HINT_SAME_PAGE)

    # ─────────────────────────────────────────────────────────────────
    # Factor 2: Same block + line adjacency → near-adjacent
    # ─────────────────────────────────────────────────────────────────
    curr_block = current_span.get("block_id")
    cand_block = candidate_span.get("block_id")

    if curr_block is not None and cand_block is not None:
        if curr_block == cand_block:
            hints.append(_RONC_V2_HINT_SAME_BLOCK)
            curr_line = current_span.get("line_index")
            cand_line = candidate_span.get("line_index")
            if isinstance(curr_line, int) and isinstance(cand_line, int):
                line_dist = abs(curr_line - cand_line)
                if line_dist <= _RONC_V2_BLOCK_START_LINE_THRESHOLD:
                    eff = _RONC_V2_BLOCK_ADJACENT_EFFECTIVE_DISTANCE
                    hints.append(_RONC_V2_HINT_ADJACENT_LINE)

    # ─────────────────────────────────────────────────────────────────
    # Factor 3: Cross-column body streams → soften distance
    # Column wraps often insert noise spans; reduce effective distance
    # ─────────────────────────────────────────────────────────────────
    curr_stream = current_span.get("layout_stream") or ""
    cand_stream = candidate_span.get("layout_stream") or ""

    if (isinstance(curr_stream, str) and isinstance(cand_stream, str) and
            curr_stream.startswith(_RONC_V2_BODY_COLUMN_PREFIX) and cand_stream.startswith(
                _RONC_V2_BODY_COLUMN_PREFIX)):

        curr_col = _ronc_v2_extract_column_from_stream(curr_stream)
        cand_col = _ronc_v2_extract_column_from_stream(cand_stream)

        if curr_col is not None and cand_col is not None and curr_col != cand_col:
            # Cross-column: soften distance
            eff = max(_RONC_V2_MIN_CROSS_COLUMN_DISTANCE, eff - _RONC_V2_CROSS_COLUMN_DISTANCE_REDUCTION)
            hints.append(_RONC_V2_HINT_CROSS_COLUMN)

    # ─────────────────────────────────────────────────────────────────
    # Factor 4: Block-edge pair → strong continuation signal
    # Direction-aware: only fires when reading order is correct
    # ─────────────────────────────────────────────────────────────────
    if curr_block != cand_block:
        # Determine reading order based on raw_distance
        # raw_distance > 0 means candidate is BEFORE current (prev direction)
        # raw_distance < 0 means candidate is AFTER current (next direction)
        # But raw_distance is always positive (abs), so we infer from context:
        # In _make_entry, we're always evaluating candidate relative to current
        # candidate_span is the potential predecessor/successor

        if candidate_idx < current_idx:
            # Candidate is earlier → check candidate.end → current.start
            if _ronc_v2_is_block_edge_pair(candidate_span, current_span):
                eff = max(_RONC_V2_MIN_EFFECTIVE_DISTANCE,
                          eff - _RONC_V2_BLOCK_EDGE_DISTANCE_REDUCTION)
                hints.append(_RONC_V2_HINT_BLOCK_EDGE)
        else:
            # Candidate is later → check current.end → candidate.start
            if _ronc_v2_is_block_edge_pair(current_span, candidate_span):
                eff = max(_RONC_V2_MIN_EFFECTIVE_DISTANCE, eff - _RONC_V2_BLOCK_EDGE_DISTANCE_REDUCTION)
                hints.append(_RONC_V2_HINT_BLOCK_EDGE)
    # ─────────────────────────────────────────────────────────────────
    # Compute distance_bias: fine-grained additive score adjustment
    # This avoids integer cliff effects from distance reductions
    # ─────────────────────────────────────────────────────────────────
    bias = 0.0
    if _RONC_V2_HINT_ADJACENT_LINE in hints:
        bias += _RONC_V2_BIAS_ADJACENT_LINE
    if _RONC_V2_HINT_BLOCK_EDGE in hints:
        bias += _RONC_V2_BLOCK_EDGE_BONUS
    if _RONC_V2_HINT_CROSS_COLUMN in hints:
        bias += _RONC_V2_BIAS_CROSS_COLUMN
    if _RONC_V2_HINT_SAME_BLOCK in hints:
        bias += _RONC_V2_BIAS_SAME_BLOCK
    # Clamp bias to reasonable range
    bias = max(_RONC_V2_BIAS_MIN, min(_RONC_V2_BIAS_MAX, bias))

    return max(_RONC_V2_MIN_EFFECTIVE_DISTANCE, eff), round(bias, 3), hints


def _ronc_v2_build_candidate_pools(
        window_spans: List[Dict],
        current_idx: int,
        trace_id: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build predecessor and successor candidate pools for a span.

    RONC v2.1 Phase 2: Evidence-Gated Candidate Pool Construction

    Two-pass strategy:
      Pass 1: Strict search (original behavior, bounded by BASE_MAX)
      Pass 2: Conditional expansion (if boundary signals indicate need)

    Each candidate entry contains:
      - idx: window index
      - cid: canonical span ID
      - distance_raw: absolute window distance
      - distance: effective distance (structure-aware, used by Phase 3)
      - mode: "strict" | "expanded"
      - hints: list of structural factors applied

    Returns:
        Tuple of (prev_pool, next_pool) as lists of candidate dicts
    """
    if current_idx < 0 or current_idx >= len(window_spans):
        return [], []

    current_span = window_spans[current_idx]
    current_page = current_span.get("_source_page_idx")

    # ─────────────────────────────────────────────────────────────────
    # Extract boundary signals from Phase 1 contract
    # ─────────────────────────────────────────────────────────────────
    contract = current_span.get("_ronc_contract") or {}
    boundary = contract.get("boundary") or {}

    needs_pred = bool(boundary.get("needs_predecessor"))
    needs_succ = bool(boundary.get("needs_successor"))

    start_profile = boundary.get("start") or {}
    end_profile = boundary.get("end") or {}

    start_score = float(start_profile.get("score", 0.0) or 0.0)
    end_score = float(end_profile.get("score", 0.0) or 0.0)

    start_label = start_profile.get("label", _RONC_V2_START_LABEL_UNKNOWN)
    end_label = end_profile.get("label", _RONC_V2_END_LABEL_UNKNOWN)

    # ─────────────────────────────────────────────────────────────────
    # Helper: Create candidate entry with effective distance
    # ─────────────────────────────────────────────────────────────────
    def _make_entry(idx: int, mode: str) -> Optional[Dict]:
        # Ensure window index is available for direction-aware computations
        # NOTE: do NOT mutate spans with window-local indices
        cid = candidate.get("_canonical_span_id")
        if not cid:
            return None

        raw_distance = abs(current_idx - idx)
        eff_distance, distance_bias, hints = _ronc_v2_compute_effective_distance(
            current_span,
            candidate,
            raw_distance,
            current_idx=current_idx,
            candidate_idx=idx,
        )

        return {
            "idx": idx,
            "cid": cid,
            "distance_raw": raw_distance,
            "distance": eff_distance,
            "distance_bias": distance_bias,  # Fine-grained proximity adjustment
            "mode": mode,
            "hints": hints,
            # Phase 3 will populate these:
            "score": 0.0,
            "reason": None,
            "components": {},
        }

    prev_pool: List[Dict] = []
    next_pool: List[Dict] = []

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 1: STRICT PREV POOL
    # ═══════════════════════════════════════════════════════════════════════
    for i in range(current_idx - 1, -1, -1):
        raw = current_idx - i
        if raw > _RONC_V2_MAX_CANDIDATE_DISTANCE:
            break

        candidate = window_spans[i]

        if not _ronc_v2_is_candidate_eligible(candidate, trace_id=trace_id):
            continue

        cand_page = candidate.get("_source_page_idx")
        cand_pos = candidate.get("_window_position")

        # Cross-page rule: only prev_tail allowed
        if cand_page != current_page:
            if cand_pos != _RONC_V2_WINDOW_POS_PREV_TAIL:
                continue

        entry = _make_entry(i, mode=_RONC_V2_MODE_STRICT)
        if entry:
            prev_pool.append(entry)

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 1: STRICT NEXT POOL
    # ═══════════════════════════════════════════════════════════════════════
    for i in range(current_idx + 1, len(window_spans)):
        raw = i - current_idx
        if raw > _RONC_V2_MAX_CANDIDATE_DISTANCE:
            break

        candidate = window_spans[i]

        if not _ronc_v2_is_candidate_eligible(candidate, trace_id=trace_id):
            continue

        cand_page = candidate.get("_source_page_idx")
        cand_pos = candidate.get("_window_position")

        # Cross-page rule: only next_head allowed
        if cand_page != current_page:
            if cand_pos != _RONC_V2_WINDOW_POS_NEXT_HEAD:
                continue

        entry = _make_entry(i, mode=_RONC_V2_MODE_STRICT)
        if entry:
            next_pool.append(entry)

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 2: EVIDENCE-GATED PREV EXPANSION
    #
    # Triggers when:
    #   - needs_predecessor is True, AND
    #   - strict pool is weak (< MIN_POOL_STRICT), AND
    #   - start_score >= threshold OR start_label indicates continuation
    # ═══════════════════════════════════════════════════════════════════════
    expand_prev = (
            needs_pred and
            len(prev_pool) < _RONC_V2_MIN_POOL_STRICT and
            (start_score >= _RONC_V2_EXPANSION_THRESHOLD_START or
             start_label in (_RONC_V2_START_LABEL_CONTINUATION, _RONC_V2_START_LABEL_FRAGMENT))
    )

    if expand_prev:
        for i in range(current_idx - 1, -1, -1):
            raw = current_idx - i

            # Skip strict range (already processed)
            if raw <= _RONC_V2_MAX_CANDIDATE_DISTANCE:
                continue

            # Expanded limit
            if raw > _RONC_V2_EXPANDED_CANDIDATE_DISTANCE:
                break

            candidate = window_spans[i]

            if not _ronc_v2_is_candidate_eligible(candidate, trace_id=trace_id):
                continue

            cand_page = candidate.get("_source_page_idx")
            cand_pos = candidate.get("_window_position")

            # Relaxed rules for expansion:
            # - Same page: always allowed (cross-column OK)
            # - Different page: still requires prev_tail
            if cand_page != current_page:
                if cand_pos != _RONC_V2_WINDOW_POS_PREV_TAIL:
                    continue

            entry = _make_entry(i, mode=_RONC_V2_MODE_EXPANDED)
            if entry:
                entry["hints"].append(_RONC_V2_HINT_EXPANSION_TRIGGER)
                prev_pool.append(entry)

            # Stop once we have enough
            # OR we've accumulated enough to ensure scoring has options
            if any(e.get("distance", 999) <= _RONC_V2_EXPANSION_QUALITY_DISTANCE for e in
                   prev_pool):
                break  # Quality threshold met
            if len(prev_pool) >= _RONC_V2_MIN_POOL_EXPANDED:
                break  # Fallback: pool size limit

        if trace_id and any(e.get("mode") == _RONC_V2_MODE_EXPANDED for e in prev_pool):
            logger.debug(
                "[%s] RONC v2.1 Phase 2: PREV expansion for %s (score=%.2f, label=%s)",
                trace_id, current_span.get("_canonical_span_id"), start_score, start_label
            )

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 2: EVIDENCE-GATED NEXT EXPANSION
    # ═══════════════════════════════════════════════════════════════════════
    expand_next = (
            needs_succ and
            len(next_pool) < _RONC_V2_MIN_POOL_STRICT and
            (end_score >= _RONC_V2_EXPANSION_THRESHOLD_END or
             end_label in (_RONC_V2_END_LABEL_TRUNCATED, _RONC_V2_END_LABEL_MID_SENTENCE))
    )

    if expand_next:
        for i in range(current_idx + 1, len(window_spans)):
            raw = i - current_idx

            if raw <= _RONC_V2_MAX_CANDIDATE_DISTANCE:
                continue

            if raw > _RONC_V2_EXPANDED_CANDIDATE_DISTANCE:
                break

            candidate = window_spans[i]

            if not _ronc_v2_is_candidate_eligible(candidate, trace_id=trace_id):
                continue

            cand_page = candidate.get("_source_page_idx")
            cand_pos = candidate.get("_window_position")

            if cand_page != current_page:
                if cand_pos != _RONC_V2_WINDOW_POS_NEXT_HEAD:
                    continue

            entry = _make_entry(i, mode=_RONC_V2_MODE_EXPANDED)
            if entry:
                entry["hints"].append(_RONC_V2_HINT_EXPANSION_TRIGGER)
                next_pool.append(entry)

            # Stop once we have a quality candidate (effective distance ≤ 3)
            # OR we've accumulated enough to ensure scoring has options
            if any(e.get("distance", 999) <= _RONC_V2_EXPANSION_QUALITY_DISTANCE for e in
                   next_pool):
                break # Quality threshold met
            if len(next_pool) >= _RONC_V2_MIN_POOL_EXPANDED:
                break  # Fallback: pool size limit

        if trace_id and any(e.get("mode") == _RONC_V2_MODE_EXPANDED for e in next_pool):
            logger.debug(
                "[%s] RONC v2.1 Phase 2: NEXT expansion for %s (score=%.2f, label=%s)",
                trace_id, current_span.get("_canonical_span_id"), end_score, end_label
            )

    return prev_pool, next_pool


def _ronc_v2_populate_candidate_pools(
        window_spans: List[Dict],
        trace_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Populate candidate pools for all spans in window.

    RONC v2.0 Phase 2: Batch Candidate Pool Population

    Iterates through all window spans and populates the
    affinity.prev_candidates and affinity.next_candidates
    fields with raw candidate lists (unscored).

    Candidates are stored as lightweight references:
    - cid: canonical span ID
    - idx: window index (for Phase 3 scoring)
    - distance: window distance from current span

    Args:
        window_spans: Full window span list with contracts attached
        trace_id: Trace ID for logging

    Returns:
        Audit dict with counts:
        - total_spans: spans processed
        - spans_with_prev: spans with at least one prev candidate
        - spans_with_next: spans with at least one next candidate
        - total_prev_candidates: sum of all prev candidates
        - total_next_candidates: sum of all next candidates
    """
    audit = {
        "total_spans": len(window_spans),
        "spans_with_prev": 0,
        "spans_with_next": 0,
        "total_prev_candidates": 0,
        "total_next_candidates": 0,
    }

    for i, span in enumerate(window_spans):
        contract = span.get("_ronc_contract")
        if not contract:
            # No contract = Phase 1 didn't run or failed
            continue

        # Build pools
        prev_pool, next_pool = _ronc_v2_build_candidate_pools(
            window_spans, i, trace_id
        )

        # ─────────────────────────────────────────────────────────────
        # Populate prev_candidates (unscored)
        # ─────────────────────────────────────────────────────────────
        prev_candidates = []
        for entry in prev_pool:
            # Entry is already a rich dict from _ronc_v2_build_candidate_pools
            if not entry.get("cid"):
                continue
            prev_candidates.append(entry)

        contract["affinity"]["prev_candidates"] = prev_candidates

        if prev_candidates:
            audit["spans_with_prev"] += 1
            audit["total_prev_candidates"] += len(prev_candidates)

        # ─────────────────────────────────────────────────────────────
        # Populate next_candidates (unscored)
        # ─────────────────────────────────────────────────────────────
        next_candidates = []
        for entry in next_pool:
            if not entry.get("cid"):
                continue
            next_candidates.append(entry)

        contract["affinity"]["next_candidates"] = next_candidates

        if next_candidates:
            audit["spans_with_next"] += 1
            audit["total_next_candidates"] += len(next_candidates)

    # ─────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────
    if trace_id:
        logger.debug(
            "[%s] RONC v2.0 Phase 2: pools built for %d spans "
            "(prev: %d spans / %d candidates, next: %d spans / %d candidates)",
            trace_id,
            audit["total_spans"],
            audit["spans_with_prev"],
            audit["total_prev_candidates"],
            audit["spans_with_next"],
            audit["total_next_candidates"],
        )

    return audit


def _ronc_v2_select_link(
        candidates: List[Dict],
        direction: str,
        needs_link: bool,
        trace_id: Optional[str] = None,
) -> Dict:
    """
    Select best link from scored candidates.

    RONC v2.0 Phase 4: Single Link Selection

    Selects the top candidate if:
    - Candidate list is non-empty
    - Top candidate score >= MIN_LINK_CONFIDENCE
    - Span actually needs this link (needs_predecessor/needs_successor)

    Args:
        candidates: Scored and ranked candidate list (best first)
        direction: "prev" or "next"
        needs_link: Whether span's boundary profile indicates need
        trace_id: Trace ID for logging

    Returns:
        Link dict with:
            cid: str | None (canonical span ID of selected link)
            confidence: float (score, possibly boosted later)
            reason: str | None (selection reason)
            mutual: bool (will be set in second pass)
    """
    # Default: no link
    empty_link = {
        "cid": None,
        "confidence": 0.0,
        "reason": None,
        "mutual": False,
    }

    # No candidates available
    if not candidates:
        return empty_link

    # Get top candidate
    top = candidates[0]
    score = top.get("score", 0.0)
    cid = top.get("cid")

    # Must have valid CID
    if not cid:
        return empty_link

    # ─────────────────────────────────────────────────────────────────
    # Decision Logic
    # ─────────────────────────────────────────────────────────────────
    #
    # Case 1: Strong score (>= 0.80) - link regardless of need
    #   Rationale: High confidence overrides boundary uncertainty
    #
    # Case 2: Moderate score (>= 0.50) AND needs_link - create link
    #   Rationale: Boundary profile confirms semantic need
    #
    # Case 3: Moderate score but no need - no link
    #   Rationale: Score alone is insufficient without semantic evidence
    #
    # Case 4: Weak score (< 0.50) - no link
    #   Rationale: Below threshold, likely false positive
    # ─────────────────────────────────────────────────────────────────

    # Case 1: Strong score overrides need check
    if score >= _RONC_V2_STRONG_LINK_CONFIDENCE:
        return {
            "cid": cid,
            "confidence": score,
            "reason": f"strong_score({score:.3f})",
            "mutual": False,
        }

    # Case 2: Moderate score with semantic need
    if score >= _RONC_V2_MIN_LINK_CONFIDENCE and needs_link:
        return {
            "cid": cid,
            "confidence": score,
            "reason": f"needs_{direction}({score:.3f})",
            "mutual": False,
        }

    # Case 3 & 4: No link
    return empty_link


def _ronc_v2_build_linked_unit_cids(
        span: Dict,
        trace_id: Optional[str] = None,
) -> List[str]:
    """
    Build list of CIDs in this span's linked unit.

    RONC v2.0 Phase 5: Linked Unit Construction (MVP)

    For MVP, linked unit contains DIRECT links only:
    - Self CID
    - Prev link CID (if exists)
    - Next link CID (if exists)

    Transitive closure (A→B→C means A,B,C are all linked) is deferred
    to future phase.

    Args:
        span: Span dict with _ronc_contract and _canonical_span_id
        trace_id: Trace ID for logging

    Returns:
        Sorted list of canonical span IDs in linked unit
    """
    unit_cids = set()

    # Add self
    self_cid = span.get("_canonical_span_id")
    if self_cid:
        unit_cids.add(self_cid)

    # Get contract
    contract = span.get("_ronc_contract", {})
    links = contract.get("links", {})

    # Add prev link target
    prev_cid = links.get("prev", {}).get("cid")
    if prev_cid:
        unit_cids.add(prev_cid)

    # Add next link target
    next_cid = links.get("next", {}).get("cid")
    if next_cid:
        unit_cids.add(next_cid)

    return sorted(unit_cids)


def _ronc_v2_assign_protection(
        window_spans: List[Dict],
        trace_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Assign protection flags based on link graph.

    RONC v2.0 Phase 5: Protection Assignment

    A span receives protection (must_include=True) if:

    1. ANCHOR: Has strong outgoing NEXT link
       - Something depends on this span
       - Excluding it would orphan the continuation

    2. CONTINUATION: Has strong incoming PREV link
       - This span depends on something
       - Excluding it would waste the anchor's inclusion

    3. MUTUAL: Has bidirectional link (both directions)
       - Strongest evidence of semantic unity
       - Both spans in pair are critical

    Protection threshold is intentionally conservative (0.60) to avoid
    over-protecting low-confidence links.

    Args:
        window_spans: Window spans with reconciled links
        trace_id: Trace ID for logging

    Returns:
        Audit dict with counts:
        - total_spans: spans processed
        - anchors_protected: spans protected as anchors
        - continuations_protected: spans protected as continuations
        - mutual_protected: spans protected via mutual links
        - total_protected: unique spans protected
    """
    audit = {
        "total_spans": len(window_spans),
        "anchors_protected": 0,
        "continuations_protected": 0,
        "mutual_protected": 0,
        "safety_net_protected": 0,
        "total_protected": 0,
    }

    # Precompute unit membership counts across the FULL window
    unit_counts: Dict[int, int] = {}
    for sp in window_spans:
        uid = sp.get("_ronc_atomic_unit_id")
        if uid is None:
            continue
        unit_counts[uid] = unit_counts.get(uid, 0) + 1

    for span in window_spans:
        contract = span.get("_ronc_contract")
        if not contract:
            continue

        links = contract.get("links", {})
        protection = contract.setdefault("protection", {})

        # Reset protection — Phase 5 is authoritative for must_include
        protection["must_include"] = False
        protection["reason"] = None

        # Get link details
        prev_link = links.get("prev", {})
        next_link = links.get("next", {})

        prev_cid = prev_link.get("cid")
        prev_conf = prev_link.get("confidence", 0.0)
        prev_mutual = prev_link.get("mutual", False)

        next_cid = next_link.get("cid")
        next_conf = next_link.get("confidence", 0.0)
        next_mutual = next_link.get("mutual", False)

        # ─────────────────────────────────────────────────────────────
        # Structural Boilerplate Veto (v4.0)
        # Spans flagged as publisher metadata (copyright, DOI, proceedings
        # citations) must never receive continuity protection — they are
        # not prose and should not anchor or continue reading chains.
        #
        # IMPORTANT: Do NOT early-continue; linked_unit_cids must still
        # be built for audit/debug even when protection is vetoed.
        # ─────────────────────────────────────────────────────────────
        candidate_reasons = set(span.get("_exclusion_candidate_reasons") or [])
        veto_protection = bool(candidate_reasons & _RONC_V2_PROTECTION_VETO_REASONS)

        if veto_protection and trace_id:
            logger.debug(
                "[%s] RONC v2.0 Phase 5: Veto must_include for '%s...' (reasons=%s)",
                trace_id,
                (span.get("cleaned_text") or span.get("text") or span.get("raw_text") or "")[:30],
                sorted(candidate_reasons & _RONC_V2_PROTECTION_VETO_REASONS),
            )

        # ─────────────────────────────────────────────────────────────
        # Protection Logic (Priority Order)
        # ─────────────────────────────────────────────────────────────

        protection_applied = False
        reasons = []

        # Check 1: Mutual link (highest priority)
        if next_cid and next_mutual and next_conf >= _RONC_V2_PROTECTION_THRESHOLD:
            protection_applied = True
            reasons.append(f"{_RONC_V2_PROTECTION_MUTUAL}:{next_cid}")
            audit["mutual_protected"] += 1

        elif prev_cid and prev_mutual and prev_conf >= _RONC_V2_PROTECTION_THRESHOLD:
            protection_applied = True
            reasons.append(f"{_RONC_V2_PROTECTION_MUTUAL}:{prev_cid}")
            audit["mutual_protected"] += 1

        # Check 2: Anchor (has outgoing NEXT link)
        elif next_cid and next_conf >= _RONC_V2_PROTECTION_THRESHOLD:
            protection_applied = True
            reasons.append(f"{_RONC_V2_PROTECTION_ANCHOR}:{next_cid}")
            audit["anchors_protected"] += 1

        # Check 3: Continuation (has incoming PREV link)
        elif prev_cid and prev_conf >= _RONC_V2_PROTECTION_THRESHOLD:
            protection_applied = True
            reasons.append(f"{_RONC_V2_PROTECTION_CONTINUATION}:{prev_cid}")
            audit["continuations_protected"] += 1

        # Apply protection (unless vetoed by structural boilerplate check)
        if protection_applied and not veto_protection:
            protection["must_include"] = True
            protection["reason"] = "+".join(reasons)
            audit["total_protected"] += 1

        # ─────────────────────────────────────────────────────────────
        # Build linked unit CIDs (always, for audit)
        # ─────────────────────────────────────────────────────────────
        protection["linked_unit_cids"] = _ronc_v2_build_linked_unit_cids(
            span, trace_id
        )

    # ─────────────────────────────────────────────────────────────────
    # Safety Net: Unit-integrity protection for current-page anchors
    # (Migrated from legacy orphan-prevention intent)
    #
    # Applies ONLY if not already protected by link thresholds.
    # Considers FULL window membership (prev_tail/current/next_head)
    # to preserve cross-page atomic unit integrity.
    # ─────────────────────────────────────────────────────────────────
    for span in window_spans:
        # Only current-page spans should be granted canonical protection here
        if span.get("_window_position") != "current":
            continue

        contract = span.get("_ronc_contract")
        if not contract:
            continue

        protection = contract.setdefault("protection", {})

        # Skip if already protected by threshold-based logic
        if protection.get("must_include"):
            continue

        # Only anchors can orphan continuations
        if span.get("_ronc_atomic_role") != _RONC_V2_LEGACY_ROLE_ANCHOR:
            continue

        unit_id = span.get("_ronc_atomic_unit_id")
        if unit_id is None:
            continue

        # If unit has no other members in this window, nothing to protect
        if unit_counts.get(unit_id, 0) < 2:
            continue

        # Structural boilerplate veto (same reasons as main loop)
        candidate_reasons = set(span.get("_exclusion_candidate_reasons") or [])
        if candidate_reasons & _RONC_V2_PROTECTION_VETO_REASONS:
            continue

        # Veto check: some anchors must never be protected
        span_text = (
                span.get("cleaned_text")
                or span.get("text")
                or span.get("raw_text")
                or ""
        )
        if _matches_continuity_veto_pattern(span_text):
            continue

        # Apply safety net protection
        protection["must_include"] = True

        # Preserve/extend reason instead of overwriting
        prior_reason = protection.get("reason")
        add_reason = f"unit_integrity:{unit_id}"
        protection["reason"] = f"{prior_reason}+{add_reason}" if prior_reason else add_reason

        # Preserve existing downstream expectations (audit/logging hooks)
        span["_ronc_rescue_applied"] = True

        audit["safety_net_protected"] += 1
        audit["anchors_protected"] += 1
        audit["total_protected"] += 1

    # ─────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────
    if trace_id:
        logger.info(
            "[%s] RONC v2.0 Phase 5: %d spans protected "
            "(anchors=%d, continuations=%d, mutual=%d, safety_net=%d)",
            trace_id,
            audit["total_protected"],
            audit["anchors_protected"],
            audit["continuations_protected"],
            audit["mutual_protected"],
            audit["safety_net_protected"],
        )

    return audit


def _ronc_v2_build_unit_groups(
        window_spans: List[Dict],
        trace_id: Optional[str] = None,
) -> List[List[int]]:
    """
    Build atomic unit groups from link graph using connected components.

    RONC v2.0 Phase 6: Unit Group Construction (REVISED)

    Treats the link graph as UNDIRECTED and finds connected components:
    - Each span is a node
    - Each prev/next link is an undirected edge
    - Connected components become atomic units
    - Each span belongs to EXACTLY ONE unit

    Within each component, spans are ordered by window index
    for consistent role assignment (first=anchor, last=tail).

    Args:
        window_spans: Window spans with reconciled links
        trace_id: Trace ID for logging

    Returns:
        List of unit groups, where each group is a list of span indices
        ordered by window position (for role assignment)
    """
    # Build CID → index map
    cid_to_idx = {}
    for i, sp in enumerate(window_spans):
        cid = sp.get("_canonical_span_id")
        if cid:
            cid_to_idx[cid] = i

    # ─────────────────────────────────────────────────────────────────
    # RONC v2.2: Reverse link index for truly undirected traversal
    #
    # Each span stores only one prev/next pointer, but many spans may
    # link to the same target (fan-in). Outbound traversal alone cannot
    # discover inbound-only neighbors, which breaks the "UNDIRECTED DFS"
    # contract and causes unitless-but-linked spans.
    #
    # Map: target_idx -> list[from_idx]
    # NOTE: We store indices only (not streams) to avoid stale capture.
    # Stream gating is evaluated at traversal time from window_spans.
    # ─────────────────────────────────────────────────────────────────
    inbound_edges = {}  # target_idx -> list[from_idx]

    for from_idx, sp in enumerate(window_spans):
        contract = sp.get("_ronc_contract", {})
        links = contract.get("links", {})

        for direction in ("prev", "next"):
            neighbor_cid = links.get(direction, {}).get("cid")
            if not neighbor_cid:
                continue

            target_idx = cid_to_idx.get(neighbor_cid)
            if target_idx is None:
                continue

            inbound_edges.setdefault(target_idx, []).append(from_idx)

    # Track visited spans
    visited = set()
    unit_groups = []

    # ─────────────────────────────────────────────────────────────────
    # Find connected components via DFS
    # ─────────────────────────────────────────────────────────────────
    for start_idx in range(len(window_spans)):
        if start_idx in visited:
            continue

        sp = window_spans[start_idx]
        contract = sp.get("_ronc_contract", {})
        links = contract.get("links", {})

        prev_cid = links.get("prev", {}).get("cid")
        next_cid = links.get("next", {}).get("cid")

        # No edges: treat as standalone node.
        # Mark visited to ensure stability across overlapping windows.
        if not prev_cid and not next_cid:
            visited.add(start_idx)
            continue

        # DFS to collect connected component
        stack = [start_idx]
        component = []

        while stack:
            idx = stack.pop()

            if idx in visited:
                continue

            visited.add(idx)
            component.append(idx)

            # Get this span's links
            sp = window_spans[idx]
            contract = sp.get("_ronc_contract", {})
            links = contract.get("links", {})

            # Add neighbors (treat links as undirected edges)
            # RONC v2.1: STRICT stream gate — atomic units may not cross layout_stream
            curr_stream = sp.get("layout_stream") or ""

            # ─────────────────────────────────────────────────────────────
            # Outbound edges: spans this span links TO (prev/next)
            # ─────────────────────────────────────────────────────────────
            for direction in ("prev", "next"):
                neighbor_cid = links.get(direction, {}).get("cid")
                if not neighbor_cid:
                    continue

                neighbor_idx = cid_to_idx.get(neighbor_cid)
                if neighbor_idx is None or neighbor_idx in visited:
                    continue

                neighbor_sp = window_spans[neighbor_idx]
                neighbor_stream = neighbor_sp.get("layout_stream") or ""

                # STRICT POLICY:
                # - both streams must be non-empty
                # - must be exactly equal
                if curr_stream and neighbor_stream and curr_stream == neighbor_stream:
                    stack.append(neighbor_idx)

            # ─────────────────────────────────────────────────────────────
            # RONC v2.2: Inbound edges — spans that link TO this span
            # Required to satisfy the "UNDIRECTED" connected-component contract.
            # Handles fan-in topologies (many spans pointing to one target).
            # Stream gating is evaluated from current window spans (no stale state).
            # ─────────────────────────────────────────────────────────────
            for from_idx in inbound_edges.get(idx, []):
                if from_idx in visited:
                    continue

                from_sp = window_spans[from_idx]
                from_stream = from_sp.get("layout_stream") or ""

                if curr_stream and from_stream and curr_stream == from_stream:
                    stack.append(from_idx)

        # ─────────────────────────────────────────────────────────────
        # Only keep components meeting minimum size
        # Sort by window index for consistent role assignment
        # ─────────────────────────────────────────────────────────────
        if len(component) >= _RONC_V2_MIN_UNIT_SIZE:
            unit_groups.append(sorted(component))

    return unit_groups


def _ronc_v2_derive_legacy_fields(
        window_spans: List[Dict],
        trace_id: Optional[str] = None,
        unit_id_offset: int = 0,
) -> Dict[str, int]:
    """
    Derive legacy RONC fields from contract link graph.

    RONC v2.0 Phase 6: Legacy Field Derivation

    Populates for backward compatibility:
    - _ronc_atomic_unit_id: Numeric group identifier
    - _ronc_atomic_role: "anchor" | "member" | "tail" | None
    - _ronc_break_after: True if TTS should pause after this span

    Role assignment:
    - First span in chain: "anchor"
    - Last span in chain: "tail"
    - Middle spans: "member"
    - Unlinked spans: None

    Break logic:
    - Tail spans: break_after = True (end of semantic unit)
    - Anchor/member spans: break_after = False (unit continues)
    - Unlinked spans: break_after = True (standalone)

    Args:
        window_spans: Window spans with reconciled links and protection
        trace_id: Trace ID for logging

    Returns:
        Audit dict with counts:
        - total_spans: spans processed
        - units_created: number of atomic units
        - anchors: spans assigned anchor role
        - members: spans assigned member role
        - tails: spans assigned tail role
        - unlinked: spans with no role
    """
    audit = {
        "total_spans": len(window_spans),
        "units_created": 0,
        "anchors": 0,
        "members": 0,
        "tails": 0,
        "unlinked": 0,
    }

    # ─────────────────────────────────────────────────────────────────
    # Step 1: Initialize all spans with default (no unit)
    # ─────────────────────────────────────────────────────────────────
    for sp in window_spans:
        sp["_ronc_atomic_unit_id"] = None
        sp["_ronc_atomic_role"] = None
        sp["_ronc_break_after"] = True  # Default: break after standalone

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Build unit groups from link graph
    # ─────────────────────────────────────────────────────────────────
    unit_groups = _ronc_v2_build_unit_groups(window_spans, trace_id)

    audit["units_created"] = len(unit_groups)

    # ─────────────────────────────────────────────────────────────────
    # Step 3: Assign unit IDs and roles
    # ─────────────────────────────────────────────────────────────────
    for unit_id, group in enumerate(unit_groups):
        group_size = len(group)

        for position, span_idx in enumerate(group):
            sp = window_spans[span_idx]

            # Assign unit ID
            sp["_ronc_atomic_unit_id"] = unit_id + unit_id_offset

            # Assign role based on position
            if position == 0:
                # First in chain: anchor
                sp["_ronc_atomic_role"] = _RONC_V2_LEGACY_ROLE_ANCHOR
                sp["_ronc_break_after"] = False
                audit["anchors"] += 1

            elif position == group_size - 1:
                # Last in chain: tail
                sp["_ronc_atomic_role"] = _RONC_V2_LEGACY_ROLE_TAIL
                sp["_ronc_break_after"] = True  # Break after unit ends
                audit["tails"] += 1

            else:
                # Middle: member
                sp["_ronc_atomic_role"] = _RONC_V2_LEGACY_ROLE_MEMBER
                sp["_ronc_break_after"] = False
                audit["members"] += 1

    # ─────────────────────────────────────────────────────────────────
    # Step 4: Count unlinked spans
    # ─────────────────────────────────────────────────────────────────
    for sp in window_spans:
        if sp.get("_ronc_atomic_unit_id") is None:
            audit["unlinked"] += 1

    # ─────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────
    if trace_id:
        logger.info(
            "[%s] RONC v2.0 Phase 6: %d units created "
            "(anchors=%d, members=%d, tails=%d, unlinked=%d)",
            trace_id,
            audit["units_created"],
            audit["anchors"],
            audit["members"],
            audit["tails"],
            audit["unlinked"],
        )

    audit["units_created"] = len(unit_groups)
    return audit


def _build_ronc_contract_v2(
        window_spans: List[Dict],
        trace_id: Optional[str] = None,
        unit_id_offset: int = 0,
) -> Dict[str, Any]:
    """
    RONC v2.0 Main Orchestrator: Build semantic continuity contracts.

    Replaces _build_ronc_contract (v1.0) with contract-driven architecture.

    Pipeline:
        Phase 1: Initialize contracts with boundary profiling
        Phase 2: Build candidate pools
        Phase 3: Score and rank candidates by semantic affinity
        Phase 4: Reconcile links with thresholds and mutual agreement
        Phase 5: Assign protection flags
        Phase 6: Derive legacy fields for backward compatibility

    Contract guarantees:
        - Every span receives a _ronc_contract dict
        - Every span receives legacy fields (_ronc_atomic_unit_id, etc.)
        - Protection decisions are auditable via contract.protection
        - Links are explicit and scored

    Args:
        window_spans: Window spans from _build_sliding_window_spans
                     Must have _canonical_span_id, _window_position, _source_page_idx
        trace_id: Trace ID for logging

    Returns:
        Audit dict summarizing pipeline execution:
        - phase_1: boundary profiling stats
        - phase_2: candidate pool stats
        - phase_3: scoring stats
        - phase_4: link stats
        - phase_5: protection stats
        - phase_6: legacy derivation stats
        - total_spans: window size
        - execution_time_ms: total processing time
    """
    import time
    start_time = time.perf_counter()

    audit = {
        "total_spans": len(window_spans),
        "phase_1": {},
        "phase_2": {},
        "phase_3": {},
        "phase_4": {},
        "phase_5": {},
        "phase_6": {},
    }

    if not window_spans:
        if trace_id:
            logger.debug("[%s] RONC v2.0: Empty window, skipping", trace_id)
        return audit

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Initialize contracts with boundary profiling
    # ═══════════════════════════════════════════════════════════════════════
    phase_1_errors = 0
    for i, sp in enumerate(window_spans):
        prev_sp = window_spans[i - 1] if i > 0 else None
        next_sp = window_spans[i + 1] if i < len(window_spans) - 1 else None

        try:
            sp["_ronc_contract"] = _ronc_v2_init_contract(
                sp, prev_sp, next_sp, trace_id
            )
        except Exception as e:
            # Graceful degradation: assign empty contract
            if trace_id:
                logger.warning(
                    "[%s] RONC v2.0 Phase 1: Contract init failed for span %d: %s",
                    trace_id, i, e
                )
            sp["_ronc_contract"] = _ronc_v2_empty_contract(trace_id)
            phase_1_errors += 1

    # Phase 1 audit
    needs_pred = sum(
        1 for sp in window_spans
        if sp.get("_ronc_contract", {}).get("boundary", {}).get("needs_predecessor")
    )
    needs_succ = sum(
        1 for sp in window_spans
        if sp.get("_ronc_contract", {}).get("boundary", {}).get("needs_successor")
    )
    audit["phase_1"] = {
        "needs_predecessor": needs_pred,
        "needs_successor": needs_succ,
        "errors": phase_1_errors,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Build candidate pools
    # ═══════════════════════════════════════════════════════════════════════
    try:
        audit["phase_2"] = _ronc_v2_populate_candidate_pools(window_spans, trace_id)
    except Exception as e:
        if trace_id:
            logger.error("[%s] RONC v2.0 Phase 2 failed: %s", trace_id, e)
        audit["phase_2"] = {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: Score and rank candidates
    # ═══════════════════════════════════════════════════════════════════════
    try:
        audit["phase_3"] = _ronc_v2_score_and_rank_candidates(window_spans, trace_id)
    except Exception as e:
        if trace_id:
            logger.error("[%s] RONC v2.0 Phase 3 failed: %s", trace_id, e)
        audit["phase_3"] = {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: Reconcile links
    # ═══════════════════════════════════════════════════════════════════════
    try:
        audit["phase_4"] = _ronc_v2_reconcile_links(window_spans, trace_id)
    except Exception as e:
        if trace_id:
            logger.error("[%s] RONC v2.0 Phase 4 failed: %s", trace_id, e)
        audit["phase_4"] = {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 5: Assign protection
    # ═══════════════════════════════════════════════════════════════════════
    try:
        audit["phase_5"] = _ronc_v2_assign_protection(window_spans, trace_id)
    except Exception as e:
        if trace_id:
            logger.error("[%s] RONC v2.0 Phase 5 failed: %s", trace_id, e)
        audit["phase_5"] = {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 6: Derive legacy fields
    # ═══════════════════════════════════════════════════════════════════════
    try:
        audit["phase_6"] = _ronc_v2_derive_legacy_fields(
            window_spans,
            trace_id,
            unit_id_offset,
        )
    except Exception as e:
        if trace_id:
            logger.error("[%s] RONC v2.0 Phase 6 failed: %s", trace_id, e)
        audit["phase_6"] = {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 7: A2 Edge Qualification + Evidence Integration + Authority (v4.0)
    # Computes A2 edge qualification HERE (layout_stream is now available).
    # Enriches contracts with A2 evidence and assigns graded authority.
    # INVARIANT: Does NOT mint _ronc_atomic_unit_id.
    # ═══════════════════════════════════════════════════════════════════════
    phase_7_qualified = 0
    phase_7_cross_stream = 0
    phase_7_weak = 0
    phase_7_strong = 0
    phase_7_none = 0

    for i, sp in enumerate(window_spans):
        prev_sp = window_spans[i - 1] if i > 0 else None
        contract = sp.get("_ronc_contract", {})

        # -------------------------------------------------------------
        # A2 EDGE QUALIFICATION (bidirectional + stream-aware)
        # -------------------------------------------------------------
        if i == 0:
            sp["_a2_edge_exists"] = False
            sp["_a2_cross_stream"] = False
            sp["_a2_qualified"] = False
            sp["_a2_edge_prev_id"] = None
        else:
            edge_exists = (
                    prev_sp.get("a2_continues_to_next", False)
                    and sp.get("a2_continues_from_previous", False)
            )
            sp["_a2_edge_exists"] = edge_exists

            if not edge_exists:
                sp["_a2_cross_stream"] = False
                sp["_a2_qualified"] = False
                sp["_a2_edge_prev_id"] = None
            else:
                prev_stream = prev_sp.get("layout_stream", "")
                curr_stream = sp.get("layout_stream", "")
                cross_stream = (prev_stream != curr_stream)

                sp["_a2_cross_stream"] = cross_stream
                sp["_a2_qualified"] = not cross_stream

                # REVISION: Always record prev_id when edge exists (audit-safe)
                sp["_a2_edge_prev_id"] = prev_sp.get("_canonical_span_id")

                if cross_stream:
                    phase_7_cross_stream += 1
                else:
                    phase_7_qualified += 1

        # -------------------------------------------------------------
        # A2 EDGE EVIDENCE (structural facts)
        # -------------------------------------------------------------
        a2_edge = {
            "exists": sp.get("_a2_edge_exists", False),
            "qualified": sp.get("_a2_qualified", False),
            "prev_span_id": sp.get("_a2_edge_prev_id"),
            "curr_span_id": sp.get("_canonical_span_id"),
            "prev_stream": prev_sp.get("layout_stream") if prev_sp else None,
            "curr_stream": sp.get("layout_stream"),
            "prev_role": prev_sp.get("role") if prev_sp else None,
            "curr_role": sp.get("role"),
            "a2_mode": sp.get("a2_continuation_from_mode"),
            "a2_reason": sp.get("a2_continuation_from_reason"),
        }

        # -------------------------------------------------------------
        # STRUCTURAL DISQUALIFIERS (block weak authority)
        # -------------------------------------------------------------
        structural = []
        if sp.get("_a2_cross_stream"):
            structural.append(_RONC_V2_DISQ_CROSS_STREAM)
        if sp.get("_a2_edge_exists") and not sp.get("layout_stream"):
            structural.append(_RONC_V2_DISQ_MISSING_STREAM)

        # -------------------------------------------------------------
        # CONTEXTUAL ANNOTATIONS (informational only)
        # -------------------------------------------------------------
        contextual = []

        if prev_sp and prev_sp.get("role") != sp.get("role"):
            contextual.append("role_transition")

        if prev_sp and prev_sp.get("block_id") != sp.get("block_id"):
            contextual.append("block_boundary")

        if sp.get("_spatial_context", {}).get("inside_figure"):
            contextual.append("inside_figure")

        if sp.get("role") == "table_cell":
            contextual.append("table_content")

        # -------------------------------------------------------------
        # CONTRACT ENRICHMENT
        # -------------------------------------------------------------
        contract["continuation_evidence"] = {
            "a2_edge": a2_edge,
            "structural_disqualifiers": structural,
            "contextual_annotations": contextual,
        }

        # -------------------------------------------------------------
        # AUTHORITY ASSIGNMENT (graded, non-destructive)
        # -------------------------------------------------------------
        has_atomic_unit = sp.get("_ronc_atomic_unit_id") is not None
        existing_authority = contract.get("authority")

        if has_atomic_unit:
            contract["authority"] = _RONC_V2_AUTHORITY_STRONG
            phase_7_strong += 1
        elif sp.get("_a2_qualified") and not structural:
            if existing_authority in (None, _RONC_V2_AUTHORITY_NONE):
                contract["authority"] = _RONC_V2_AUTHORITY_WEAK
                phase_7_weak += 1

        else:
            if existing_authority is None:
                contract["authority"] = _RONC_V2_AUTHORITY_NONE
                phase_7_none += 1

        sp["_ronc_contract"] = contract

    audit["phase_7"] = {
        "a2_qualified": phase_7_qualified,
        "a2_cross_stream": phase_7_cross_stream,
        "authority_strong": phase_7_strong,
        "authority_weak": phase_7_weak,
        "authority_none": phase_7_none,
    }

    if trace_id:
        logger.info(
            "[%s] RONC v2.0 Phase 7: A2 qualified=%d cross-stream=%d | authority strong=%d weak=%d none=%d",
            trace_id,
            phase_7_qualified,
            phase_7_cross_stream,
            phase_7_strong,
            phase_7_weak,
            phase_7_none,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Execution summary
    # ═══════════════════════════════════════════════════════════════════════
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    audit["execution_time_ms"] = round(elapsed_ms, 2)

    if trace_id:
        protected = audit.get("phase_5", {}).get("total_protected", 0)
        units = audit.get("phase_6", {}).get("units_created", 0)
        logger.info(
            "[%s] RONC v2.0: Complete in %.1fms — %d spans, %d protected, %d units",
            trace_id, elapsed_ms, len(window_spans), protected, units
        )

    return audit


def _ronc_v2_check_mutual_agreement(
        window_spans: List[Dict],
        trace_id: Optional[str] = None,
) -> int:
    """
    Check for mutual agreement and boost confidence.

    RONC v2.0 Phase 4: Mutual Agreement Detection

    A mutual link exists when:
    - Span A links to Span B as NEXT
    - Span B links to Span A as PREV

    Mutual links receive a confidence boost because bidirectional
    evidence is stronger than unidirectional.

    Args:
        window_spans: Window spans with links.prev and links.next populated
        trace_id: Trace ID for logging

    Returns:
        Count of mutual agreements detected
    """
    # Build CID → span index map for fast lookup
    cid_to_idx = {}
    for i, sp in enumerate(window_spans):
        cid = sp.get("_canonical_span_id")
        if cid:
            cid_to_idx[cid] = i

    mutual_count = 0

    for i, span in enumerate(window_spans):
        contract = span.get("_ronc_contract")
        if not contract:
            continue

        my_cid = span.get("_canonical_span_id")
        if not my_cid:
            continue

        links = contract.get("links", {})

        # ─────────────────────────────────────────────────────────────
        # Check NEXT link for mutual agreement
        # ─────────────────────────────────────────────────────────────
        next_link = links.get("next", {})
        next_cid = next_link.get("cid")

        if next_cid:
            target_idx = cid_to_idx.get(next_cid)

            if target_idx is None and trace_id:
                logger.debug("[%s] RONC v2.0 mutual check: target CID not in window: %s", trace_id,
                             next_cid)

            if target_idx is not None:
                target_span = window_spans[target_idx]
                target_contract = target_span.get("_ronc_contract", {})
                target_prev = target_contract.get("links", {}).get("prev", {})

                # Mutual: my NEXT target has me as their PREV
                if target_prev.get("cid") == my_cid:
                    # Guard: avoid double-boosting if this function is re-run
                    # or if window overlap syncing re-invokes mutual checks.
                    if not next_link.get("mutual"):
                        next_link["mutual"] = True

                        # Boost confidence
                        boosted = min(
                            _RONC_V2_MAX_CONFIDENCE,
                            next_link.get("confidence", 0.0) + _RONC_V2_MUTUAL_BOOST
                        )
                        next_link["confidence"] = round(boosted, 3)

                        base_reason = next_link.get("reason") or ""
                        next_link["reason"] = (base_reason + "+mutual") if base_reason else "mutual"

                        mutual_count += 1
                    else:
                        # Already mutual; ensure reason is tagged for audit consistency
                        base_reason = next_link.get("reason") or ""
                        if "+mutual" not in base_reason and base_reason != "mutual":
                            next_link["reason"] = base_reason + "+mutual"

        # ─────────────────────────────────────────────────────────────
        # Check PREV link for mutual agreement
        # (Note: we only count once per pair, so only increment for NEXT)
        # ─────────────────────────────────────────────────────────────
        prev_link = links.get("prev", {})
        prev_cid = prev_link.get("cid")

        if prev_cid:
            target_idx = cid_to_idx.get(prev_cid)

            if target_idx is not None:
                target_span = window_spans[target_idx]
                target_contract = target_span.get("_ronc_contract", {})
                target_next = target_contract.get("links", {}).get("next", {})

                # Mutual: my PREV target has me as their NEXT
                if target_next.get("cid") == my_cid:
                    # Guard: avoid double-boosting if re-run
                    if not prev_link.get("mutual"):
                        prev_link["mutual"] = True

                        # Boost confidence
                        boosted = min(
                            _RONC_V2_MAX_CONFIDENCE,
                            prev_link.get("confidence", 0.0) + _RONC_V2_MUTUAL_BOOST
                        )
                        prev_link["confidence"] = round(boosted, 3)

                        base_reason = prev_link.get("reason") or ""
                        prev_link["reason"] = (base_reason + "+mutual") if base_reason else "mutual"
                    else:
                        base_reason = prev_link.get("reason") or ""
                        if "+mutual" not in base_reason and base_reason != "mutual":
                            prev_link["reason"] = base_reason + "+mutual"

    return mutual_count


def _ronc_v2_reconcile_links(
        window_spans: List[Dict],
        trace_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Reconcile scored candidates into final link selections.

    RONC v2.0 Phase 4: Batch Link Reconciliation

    Two-pass process:
    1. Select best candidate for each span's prev/next links
    2. Detect mutual agreements and boost confidence

    Args:
        window_spans: Window spans with scored candidates
        trace_id: Trace ID for logging

    Returns:
        Audit dict with counts:
        - total_spans: spans processed
        - prev_links_created: count of prev links
        - next_links_created: count of next links
        - mutual_agreements: count of bidirectional links
        - strong_links: links created via strong score override
        - need_based_links: links created via boundary need
    """
    audit = {
        "total_spans": len(window_spans),
        "prev_links_created": 0,
        "next_links_created": 0,
        "mutual_agreements": 0,
        "strong_links": 0,
        "need_based_links": 0,
    }

    # ─────────────────────────────────────────────────────────────────
    # PASS 1: Select best candidates
    # ─────────────────────────────────────────────────────────────────
    for span in window_spans:
        contract = span.get("_ronc_contract")
        if not contract:
            continue

        boundary = contract.get("boundary", {})
        affinity = contract.get("affinity", {})

        # ─────────────────────────────────────────────────────────────
        # Select PREV link
        # ─────────────────────────────────────────────────────────────
        prev_candidates = affinity.get("prev_candidates", [])
        needs_pred = boundary.get("needs_predecessor", False)

        prev_link = _ronc_v2_select_link(
            prev_candidates,
            direction="prev",
            needs_link=needs_pred,
            trace_id=trace_id,
        )

        contract["links"]["prev"] = prev_link

        if prev_link["cid"]:
            audit["prev_links_created"] += 1
            if "strong_score" in (prev_link.get("reason") or ""):
                audit["strong_links"] += 1
            else:
                audit["need_based_links"] += 1

        # ─────────────────────────────────────────────────────────────
        # Select NEXT link
        # ─────────────────────────────────────────────────────────────
        next_candidates = affinity.get("next_candidates", [])
        needs_succ = boundary.get("needs_successor", False)

        next_link = _ronc_v2_select_link(
            next_candidates,
            direction="next",
            needs_link=needs_succ,
            trace_id=trace_id,
        )

        contract["links"]["next"] = next_link

        if next_link["cid"]:
            audit["next_links_created"] += 1
            if "strong_score" in (next_link.get("reason") or ""):
                audit["strong_links"] += 1
            else:
                audit["need_based_links"] += 1

    # ─────────────────────────────────────────────────────────────────
    # PASS 2: Detect mutual agreements and boost
    # ─────────────────────────────────────────────────────────────────
    audit["mutual_agreements"] = _ronc_v2_check_mutual_agreement(
        window_spans, trace_id
    )

    # ─────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────
    if trace_id:
        logger.info(
            "[%s] RONC v2.0 Phase 4: %d prev links, %d next links, "
            "%d mutual agreements (strong=%d, need-based=%d)",
            trace_id,
            audit["prev_links_created"],
            audit["next_links_created"],
            audit["mutual_agreements"],
            audit["strong_links"],
            audit["need_based_links"],
        )

    return audit


def _ronc_v2_score_semantic_flow(
        current_span: Dict,
        candidate_span: Dict,
        direction: str,
        trace_id: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Score semantic boundary compatibility between two spans.

    RONC v2.0 Phase 3: Semantic Flow Scoring

    Evaluates how well boundaries "fit" together:
    - For PREV candidate: Does candidate's END flow into current's START?
    - For NEXT candidate: Does current's END flow into candidate's START?

    High scores indicate strong semantic continuity evidence.

    Args:
        current_span: Span being analyzed (has _ronc_contract)
        candidate_span: Potential link target (has _ronc_contract)
        direction: "prev" or "next"
        trace_id: Trace ID for logging

    Returns:
        Tuple of (score: float 0.0-1.0, reason: str)
    """
    # ─────────────────────────────────────────────────────────────────
    # Extract contracts
    # ─────────────────────────────────────────────────────────────────
    current_contract = current_span.get("_ronc_contract", {})
    candidate_contract = candidate_span.get("_ronc_contract", {})

    current_boundary = current_contract.get("boundary", {})
    candidate_boundary = candidate_contract.get("boundary", {})

    # ─────────────────────────────────────────────────────────────────
    # Determine which boundaries to compare based on direction
    #
    # PREV candidate: candidate comes BEFORE current
    #   → Compare: candidate.end → current.start
    #   → Good fit: candidate ends truncated, current starts as continuation
    #
    # NEXT candidate: candidate comes AFTER current
    #   → Compare: current.end → candidate.start
    #   → Good fit: current ends truncated, candidate starts as continuation
    # ─────────────────────────────────────────────────────────────────
    if direction == "prev":
        # Candidate is predecessor: does candidate's END flow into my START?
        end_profile = candidate_boundary.get("end", {})
        start_profile = current_boundary.get("start", {})
    elif direction == "next":
        # Candidate is successor: does my END flow into candidate's START?
        end_profile = current_boundary.get("end", {})
        start_profile = candidate_boundary.get("start", {})
    else:
        # Defensive: invalid direction (should never occur)
        return 0.0, "invalid_direction"

    # ─────────────────────────────────────────────────────────────────
    # Extract boundary scores and labels
    # ─────────────────────────────────────────────────────────────────
    end_score = end_profile.get("score", 0.0)
    end_label = end_profile.get("label", _RONC_V2_END_LABEL_UNKNOWN)

    start_score = start_profile.get("score", 0.0)
    start_label = start_profile.get("label", _RONC_V2_START_LABEL_UNKNOWN)

    # ─────────────────────────────────────────────────────────────────
    # Scoring Logic: Boundary Compatibility Matrix
    #
    # Best case: truncated END → continuation START
    #   Both boundaries indicate incomplete semantic unit
    #
    # Good case: mid_sentence END → fragment/continuation START
    #   Partial evidence on both sides
    #
    # Neutral case: complete END → clean START
    #   No continuation evidence (but not incompatible)
    #
    # Poor case: complete END → continuation START
    #   Conflicting signals (complete shouldn't need continuation)
    # ─────────────────────────────────────────────────────────────────

    score = 0.0
    reasons = []

    # ─────────────────────────────────────────────────────────────────
    # Component 1: End boundary truncation (0.0 - 0.5)
    # Higher end_score = more likely truncated = more likely needs successor
    # ─────────────────────────────────────────────────────────────────
    end_component = end_score * _RONC_V2_SCORE_END_WEIGHT
    score += end_component

    if end_label == _RONC_V2_END_LABEL_TRUNCATED:
        reasons.append("end:truncated")
    elif end_label == _RONC_V2_END_LABEL_MID_SENTENCE:
        reasons.append("end:mid")

    # ─────────────────────────────────────────────────────────────────
    # Component 2: Start boundary continuation (0.0 - 0.5)
    # Higher start_score = more likely continuation = more likely needs predecessor
    # ─────────────────────────────────────────────────────────────────
    start_component = start_score * _RONC_V2_SCORE_START_WEIGHT
    score += start_component

    if start_label == _RONC_V2_START_LABEL_CONTINUATION:
        reasons.append("start:continuation")
    elif start_label == _RONC_V2_START_LABEL_FRAGMENT:
        reasons.append("start:fragment")

    # ─────────────────────────────────────────────────────────────────
    # Component 3: Label compatibility bonus/penalty
    # ─────────────────────────────────────────────────────────────────

    # Best match: truncated → continuation
    if end_label == _RONC_V2_END_LABEL_TRUNCATED and start_label == _RONC_V2_START_LABEL_CONTINUATION:
        score += _RONC_V2_SCORE_MATCH_IDEAL_BONUS
        reasons.append("match:ideal")

    # Good match: mid_sentence → continuation/fragment
    elif end_label == _RONC_V2_END_LABEL_MID_SENTENCE and start_label in (
            _RONC_V2_START_LABEL_CONTINUATION, _RONC_V2_START_LABEL_FRAGMENT):
        score += _RONC_V2_SCORE_MATCH_GOOD_BONUS
        reasons.append("match:good")

    # Good match: truncated → fragment
    elif end_label == _RONC_V2_END_LABEL_TRUNCATED and start_label == _RONC_V2_START_LABEL_FRAGMENT:
        score += _RONC_V2_SCORE_MATCH_GOOD_BONUS
        reasons.append("match:good")

    # Conflict: complete → continuation (suspicious)
    elif end_label == _RONC_V2_END_LABEL_COMPLETE and start_label == _RONC_V2_START_LABEL_CONTINUATION:
        score += _RONC_V2_SCORE_MATCH_CONFLICT_PENALTY
        reasons.append("match:conflict")

    # ─────────────────────────────────────────────────────────────────
    # Component 4: A2 signal agreement bonus
    # If both spans have A2 continuation flags aligned, strong evidence
    # ─────────────────────────────────────────────────────────────────
    if direction == "prev":
        # For prev: candidate should have a2_continues_to_next
        #           current should have a2_continues_from_previous
        candidate_continues_to = candidate_span.get("a2_continues_to_next", False)
        current_continues_from = current_span.get("a2_continues_from_previous", False)

        if candidate_continues_to and current_continues_from:
            score += _RONC_V2_SCORE_A2_ALIGNED_BONUS
            reasons.append("a2:aligned")
        elif candidate_continues_to or current_continues_from:
            score += _RONC_V2_SCORE_A2_PARTIAL_BONUS
            reasons.append("a2:partial")

    else:  # direction == "next"
        # For next: current should have a2_continues_to_next
        #           candidate should have a2_continues_from_previous
        current_continues_to = current_span.get("a2_continues_to_next", False)
        candidate_continues_from = candidate_span.get("a2_continues_from_previous", False)

        if current_continues_to and candidate_continues_from:
            score += _RONC_V2_SCORE_A2_ALIGNED_BONUS
            reasons.append("a2:aligned")
        elif current_continues_to or candidate_continues_from:
            score += _RONC_V2_SCORE_A2_PARTIAL_BONUS
            reasons.append("a2:partial")

    # ─────────────────────────────────────────────────────────────────
    # Clamp to valid range
    # ─────────────────────────────────────────────────────────────────
    score = max(0.0, min(_RONC_V2_SCORE_PROXIMITY_MAX, score))

    reason = "+".join(reasons) if reasons else "neutral"

    return round(score, 3), reason


def _ronc_v2_score_proximity(
        current_span: Dict,
        candidate_span: Dict,
        distance: int,
        candidate_entry: Optional[Dict] = None,  # NEW
        trace_id: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Score candidate based on window distance and structural factors.

    RONC v2.0 Phase 3: Proximity Scoring

    Closer spans are more likely to be semantically related.
    Distance-based decay with adjustments for structural factors.

    Args:
        current_span: Span being analyzed
        candidate_span: Potential link target
        distance: Absolute window index distance (always positive)
        trace_id: Trace ID for logging

    Returns:
        Tuple of (score: float 0.0-1.0, reason: str)
    """
    reasons = []

    # ─────────────────────────────────────────────────────────────────
    # Component 1: Distance decay (primary factor)
    #
    # Score decreases as distance increases:
    #   distance=1 → 1.0 (adjacent, ideal)
    #   distance=2 → 0.85
    #   distance=3 → 0.70
    #   ...
    #   distance=15 → 0.0 (at max candidate distance)
    #
    # Formula: 1.0 - ((distance - 1) / (MAX_DISTANCE - 1))
    # This gives linear decay from 1.0 at distance=1 to 0.0 at MAX_DISTANCE
    # ─────────────────────────────────────────────────────────────────
    if distance <= 0:
        # Invalid distance (shouldn't happen)
        return 0.0, "invalid_distance"

    if distance == 1:
        distance_score = 1.0
        reasons.append("adjacent")
    elif distance > _RONC_V2_MAX_CANDIDATE_DISTANCE:
        distance_score = 0.0
        reasons.append("too_far")
    else:
        # Linear decay
        max_dist = _RONC_V2_MAX_CANDIDATE_DISTANCE
        distance_score = 1.0 - ((distance - 1) / (max_dist - 1))
        reasons.append(f"dist:{distance}")

    score = distance_score

    # ─────────────────────────────────────────────────────────────────
    # Component 2: Role-based adjustment
    #
    # Deprioritized roles (table cells, captions, etc.) receive penalty.
    # These are valid candidates but less likely to be primary content flow.
    # ─────────────────────────────────────────────────────────────────
    candidate_role = candidate_span.get("role")

    if candidate_role in _RONC_V2_DEPRIORITIZED_ROLES:
        score *= _RONC_V2_PROX_PENALTY_DEPRIORITIZED
        reasons.append(f"deprioritized:{candidate_role}")

    # ─────────────────────────────────────────────────────────────────
    # Component 3: Same-page bonus
    #
    # Spans on the same page are more likely to be related than
    # cross-page candidates (which require window boundary crossing).
    # ─────────────────────────────────────────────────────────────────
    current_page = current_span.get("_source_page_idx")
    candidate_page = candidate_span.get("_source_page_idx")

    if current_page is not None and candidate_page is not None:
        if current_page == candidate_page:
            score *= _RONC_V2_PROX_BONUS_SAME_PAGE
            score = min(score, 1.0)
            reasons.append(_RONC_V2_HINT_SAME_PAGE)
        else:
            # Cross-page: no penalty, but note it
            reasons.append(_RONC_V2_HINT_CROSS_PAGE)

    # ─────────────────────────────────────────────────────────────────
    # Component 4: Same-block bonus
    #
    # Spans from the same PDF block are highly likely to be related.
    # This is strong structural evidence of semantic unity.
    # ─────────────────────────────────────────────────────────────────
    current_block = current_span.get("block_id")
    candidate_block = candidate_span.get("block_id")

    if current_block is not None and candidate_block is not None:
        if current_block == candidate_block:
            score *= _RONC_V2_PROX_BONUS_SAME_BLOCK
            score = min(score, 1.0)
            reasons.append(_RONC_V2_HINT_SAME_BLOCK)

    # ─────────────────────────────────────────────────────────────────
    # Component 5: Line adjacency bonus
    #
    # If spans are on adjacent lines within same block, very strong signal.
    # ─────────────────────────────────────────────────────────────────
    current_line = current_span.get("line_index")
    candidate_line = candidate_span.get("line_index")

    if (current_block is not None and candidate_block is not None and
            current_block == candidate_block and
            current_line is not None and candidate_line is not None):

        line_distance = abs(current_line - candidate_line)

        if line_distance == 0:
            # Same line (different spans on same line)
            score *= _RONC_V2_PROX_BONUS_SAME_LINE
            score = min(score, 1.0)
            reasons.append("same_line")
        elif line_distance == 1:
            # Adjacent lines
            score *= _RONC_V2_PROX_BONUS_ADJACENT_LINE
            score = min(score, 1.0)
            reasons.append(_RONC_V2_HINT_ADJACENT_LINE)

    # ─────────────────────────────────────────────────────────────────
    # Component 6: Expanded Candidate Penalty (v2.1)
    # Candidates found via expansion must earn their place
    # ─────────────────────────────────────────────────────────────────
    candidate_mode = candidate_entry.get("mode") if isinstance(candidate_entry, dict) else None
    if candidate_mode == _RONC_V2_MODE_EXPANDED:
        score *= _RONC_V2_EXPANDED_CANDIDATE_PENALTY
        reasons.append("expanded")

    # ─────────────────────────────────────────────────────────────────
    # Component 7: Cross-Column Penalty (v2.1)
    # Soft penalty for column transitions (already softened in effective distance)
    # ─────────────────────────────────────────────────────────────────
    candidate_hints = candidate_entry.get("hints", []) if isinstance(candidate_entry,
                                                                     dict) else []
    if _RONC_V2_HINT_CROSS_COLUMN in candidate_hints:
        score *= _RONC_V2_CROSS_COLUMN_PENALTY
        reasons.append("cross_column_penalty")

    # ─────────────────────────────────────────────────────────────────
    # Component 8: Block-Edge Bonus (v2.1)
    # Structural continuation signal
    # ─────────────────────────────────────────────────────────────────
    if _RONC_V2_HINT_BLOCK_EDGE in candidate_hints:
        score += _RONC_V2_BLOCK_EDGE_BONUS
        reasons.append("block_edge_bonus")

    # ─────────────────────────────────────────────────────────────────
    # Component 9: Distance Bias (v2.1 fine-grained adjustment)
    # Avoids integer cliff effects from effective distance shaping
    # ─────────────────────────────────────────────────────────────────
    distance_bias = candidate_entry.get("distance_bias", 0.0) if isinstance(candidate_entry,
                                                                            dict) else 0.0
    if distance_bias != 0.0:
        score += distance_bias
        if distance_bias > 0:
            reasons.append(f"bias_boost:{distance_bias:.3f}")
        else:
            reasons.append(f"bias_penalty:{distance_bias:.3f}")

    # ─────────────────────────────────────────────────────────────────
    # Clamp to valid range
    # ─────────────────────────────────────────────────────────────────
    score = max(0.0, min(_RONC_V2_SCORE_SEMANTIC_MAX, score))

    reason = "+".join(reasons) if reasons else "proximity"

    return round(score, 3), reason


def _ronc_v2_score_and_rank_candidates(
        window_spans: List[Dict],
        trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score and rank all candidates for all spans in window.

    RONC v2.0 Phase 3: Batch Affinity Scoring & Ranking

    For each span's candidate pools (prev/next):
    1. Score each candidate on semantic flow (boundary compatibility)
    2. Score each candidate on proximity (distance + structure)
    3. Combine with weighted formula
    4. Sort by score descending
    5. Truncate to top-K candidates

    Weighted formula:
        final_score = (semantic_flow * 0.65) + (proximity * 0.35)

    Args:
        window_spans: Window spans with candidate pools populated (Phase 2)
        trace_id: Trace ID for logging

    Returns:
        Audit dict with:
        - total_spans: spans processed
        - candidates_scored: total candidates scored
        - prev_candidates_scored: prev direction count
        - next_candidates_scored: next direction count
        - avg_top_score: average score of top candidates
        - high_confidence_candidates: count with score >= 0.70
    """
    audit = {
        "total_spans": len(window_spans),
        "candidates_scored": 0,
        "prev_candidates_scored": 0,
        "next_candidates_scored": 0,
        "avg_top_score": 0.0,
        "high_confidence_candidates": 0,
    }

    # Build index lookup for candidate spans
    # (candidates store idx, we need the actual span for scoring)
    top_scores = []

    for current_idx, current_span in enumerate(window_spans):
        contract = current_span.get("_ronc_contract")
        if not contract:
            continue

        affinity = contract.get("affinity", {})

        # ─────────────────────────────────────────────────────────────
        # Score PREV candidates
        # ─────────────────────────────────────────────────────────────
        prev_candidates = affinity.get("prev_candidates", [])

        for candidate_entry in prev_candidates:
            candidate_idx = candidate_entry.get("idx")
            distance = candidate_entry.get("distance", 1)

            # Validate candidate index
            if candidate_idx is None or candidate_idx < 0 or candidate_idx >= len(window_spans):
                candidate_entry["score"] = 0.0
                candidate_entry["reason"] = "invalid_idx"
                candidate_entry["components"] = {}
                continue

            candidate_span = window_spans[candidate_idx]

            # Score semantic flow
            semantic_score, semantic_reason = _ronc_v2_score_semantic_flow(
                current_span=current_span,
                candidate_span=candidate_span,
                direction="prev",
                trace_id=trace_id,
            )

            # Score proximity
            proximity_score, proximity_reason = _ronc_v2_score_proximity(
                current_span=current_span,
                candidate_span=candidate_span,
                distance=distance,
                candidate_entry=candidate_entry,  # NEW: pass entry for mode/hints
                trace_id=trace_id,
            )


            # Combine with weights
            final_score = (
                    (semantic_score * _RONC_V2_WEIGHT_SEMANTIC_FLOW) +
                    (proximity_score * _RONC_V2_WEIGHT_PROXIMITY)
            )
            final_score = round(max(0.0, min(1.0, final_score)), 3)

            # Populate candidate entry
            candidate_entry["score"] = final_score
            candidate_entry["reason"] = f"sem:{semantic_reason}|prox:{proximity_reason}"
            candidate_entry["components"] = {
                "semantic_flow": semantic_score,
                "semantic_reason": semantic_reason,
                "proximity": proximity_score,
                "proximity_reason": proximity_reason,
                "weights": {
                    "semantic": _RONC_V2_WEIGHT_SEMANTIC_FLOW,
                    "proximity": _RONC_V2_WEIGHT_PROXIMITY,
                },
            }

            audit["candidates_scored"] += 1
            audit["prev_candidates_scored"] += 1

            if final_score >= _RONC_V2_HIGH_CONFIDENCE_THRESHOLD:
                audit["high_confidence_candidates"] += 1

        # Sort prev candidates by score (descending)
        prev_candidates.sort(
            key=lambda c: (
                -c.get("score", 0.0),
                c.get("distance", 9999),
                c.get("idx", 9999),
            )
        )

        # Truncate to top-K
        if len(prev_candidates) > _RONC_V2_TOP_K_CANDIDATES:
            prev_candidates = prev_candidates[:_RONC_V2_TOP_K_CANDIDATES]
            affinity["prev_candidates"] = prev_candidates

        # Track top score for audit
        if prev_candidates:
            top_scores.append(prev_candidates[0].get("score", 0.0))

        # ─────────────────────────────────────────────────────────────
        # Score NEXT candidates
        # ─────────────────────────────────────────────────────────────
        next_candidates = affinity.get("next_candidates", [])

        for candidate_entry in next_candidates:
            candidate_idx = candidate_entry.get("idx")
            distance = candidate_entry.get("distance", 1)

            # Validate candidate index
            if candidate_idx is None or candidate_idx < 0 or candidate_idx >= len(window_spans):
                candidate_entry["score"] = 0.0
                candidate_entry["reason"] = "invalid_idx"
                candidate_entry["components"] = {}
                continue

            candidate_span = window_spans[candidate_idx]

            # Score semantic flow
            semantic_score, semantic_reason = _ronc_v2_score_semantic_flow(
                current_span=current_span,
                candidate_span=candidate_span,
                direction="next",
                trace_id=trace_id,
            )

            # Score proximity
            proximity_score, proximity_reason = _ronc_v2_score_proximity(
                current_span=current_span,
                candidate_span=candidate_span,
                distance=distance,
                candidate_entry=candidate_entry,  # REQUIRED for bias + hints
                trace_id=trace_id,
            )

            # Combine with weights
            final_score = (
                    (semantic_score * _RONC_V2_WEIGHT_SEMANTIC_FLOW) +
                    (proximity_score * _RONC_V2_WEIGHT_PROXIMITY)
            )
            final_score = round(max(0.0, min(1.0, final_score)), 3)

            # Populate candidate entry
            candidate_entry["score"] = final_score
            candidate_entry["reason"] = f"sem:{semantic_reason}|prox:{proximity_reason}"
            candidate_entry["components"] = {
                "semantic_flow": semantic_score,
                "semantic_reason": semantic_reason,
                "proximity": proximity_score,
                "proximity_reason": proximity_reason,
                "weights": {
                    "semantic": _RONC_V2_WEIGHT_SEMANTIC_FLOW,
                    "proximity": _RONC_V2_WEIGHT_PROXIMITY,
                },
            }

            audit["candidates_scored"] += 1
            audit["next_candidates_scored"] += 1

            if final_score >= _RONC_V2_HIGH_CONFIDENCE_THRESHOLD:
                audit["high_confidence_candidates"] += 1

        # Sort next candidates by score (descending)
        next_candidates.sort(
            key=lambda c: (
                -c.get("score", 0.0),
                c.get("distance", 9999),
                c.get("idx", 9999),
            )
        )

        # Truncate to top-K
        if len(next_candidates) > _RONC_V2_TOP_K_CANDIDATES:
            next_candidates = next_candidates[:_RONC_V2_TOP_K_CANDIDATES]
            affinity["next_candidates"] = next_candidates

        # Track top score for audit
        if next_candidates:
            top_scores.append(next_candidates[0].get("score", 0.0))

    # ─────────────────────────────────────────────────────────────────
    # Compute average top score
    # ─────────────────────────────────────────────────────────────────
    if top_scores:
        audit["avg_top_score"] = round(sum(top_scores) / len(top_scores), 3)

    # ─────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────
    if trace_id:
        logger.debug(
            "[%s] RONC v2.0 Phase 3: scored %d candidates "
            "(prev=%d, next=%d, high_conf=%d, avg_top=%.3f)",
            trace_id,
            audit["candidates_scored"],
            audit["prev_candidates_scored"],
            audit["next_candidates_scored"],
            audit["high_confidence_candidates"],
            audit["avg_top_score"],
        )

    return audit


# ✦                  ✦                  ✦                  ✦
# ✦───────────── 1 Geometry & Spatial Primitives ─────────────✦
# ✦                  ✦                  ✦                  ✦

# ✦────── a. Bounding Box Primitives ──────✦

def _to_bbox_tuple(bbox_input: BboxInput) -> Optional[BboxTuple]:
    """
    Convert various bbox representations to standard tuple format.

    Accepts:
        - fitz.Rect object (has x0, y0, x1, y1 attributes)
        - (x0, y0, x1, y1) tuple or list
        - {"x0", "y0", "x1", "y1"} dict (PDF tool standard)
        - {"x", "y", "width", "height"} dict (web/JS style)
        - None

    Returns:
        (x0, y0, x1, y1) tuple with float coordinates, or None if invalid.
    """
    if bbox_input is None:
        return None

    x0: float
    y0: float
    x1: float
    y1: float

    # fitz.Rect object (check for characteristic attribute)
    if hasattr(bbox_input, "x0"):
        x0, y0, x1, y1 = bbox_input.x0, bbox_input.y0, bbox_input.x1, bbox_input.y1

    # Tuple or list format
    elif isinstance(bbox_input, (list, tuple)) and len(bbox_input) >= 4:
        x0, y0, x1, y1 = bbox_input[0], bbox_input[1], bbox_input[2], bbox_input[3]

    # Dict formats
    elif isinstance(bbox_input, dict):
        # Case A: Coordinate box style (x0, y0, x1, y1) — common in PDF tools
        if all(k in bbox_input for k in ("x0", "y0", "x1", "y1")):
            x0 = bbox_input["x0"]
            y0 = bbox_input["y0"]
            x1 = bbox_input["x1"]
            y1 = bbox_input["y1"]

        # Case B: Dimension style (x, y, width, height) — web/JS convention
        elif all(k in bbox_input for k in ("x", "y", "width", "height")):
            x0 = bbox_input["x"]
            y0 = bbox_input["y"]
            x1 = x0 + bbox_input["width"]
            y1 = y0 + bbox_input["height"]

        else:
            # Unknown dict schema
            return None

    else:
        # Unknown format
        return None

    # Validate dimensions (width and height must be positive)
    if x1 <= x0 or y1 <= y0:
        return None

    return float(x0), float(y0), float(x1), float(y1)


def _span_to_rect(span: Dict) -> Optional[BboxTuple]:
    """
    Extract (x0, y0, x1, y1) tuple from span's bbox field.

    Handles multiple bbox formats via delegation to _to_bbox_tuple.

    Args:
        span: Dictionary containing "bbox" key with coordinate data.

    Returns:
        BboxTuple or None if bbox missing/invalid.
    """
    bbox_data = span.get("bbox")
    if bbox_data is None:
        return None

    return _to_bbox_tuple(bbox_data)


def _rects_intersect(r1: BboxTuple, r2: BboxTuple) -> bool:
    """
    Check if two (x0, y0, x1, y1) rectangles intersect.

    Returns True if any overlap exists, including shared edges.
    Uses standard AABB (Axis-Aligned Bounding Box) intersection test.

    Args:
        r1: First rectangle as (x0, y0, x1, y1).
        r2: Second rectangle as (x0, y0, x1, y1).

    Returns:
        True if rectangles overlap or touch, False otherwise.
    """
    # No overlap if: r1 is entirely left/right/above/below r2
    return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])


def _rect_area(rect: Tuple[float, float, float, float]) -> float:
    """
    Compute area of a rectangle defined as (x0, y0, x1, y1).

    Returns 0.0 if rect is invalid or degenerate.
    """
    if not rect or len(rect) < 4:
        return 0.0

    x0, y0, x1, y1 = rect

    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)

    return width * height


def _get_intersection_rect(r1: BboxTuple, r2: BboxTuple) -> Optional[BboxTuple]:
    x0 = max(r1[0], r2[0])
    y0 = max(r1[1], r2[1])
    x1 = min(r1[2], r2[2])
    y1 = min(r1[3], r2[3])
    if x0 < x1 and y0 < y1:
        try:
            x0 = float(x0)
            y0 = float(y0)
            x1 = float(x1)
            y1 = float(y1)
        except (TypeError, ValueError):
            return None

            # Reject NaN / Inf
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            return None

            # Normalize inverted boxes
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

            # Reject degenerate boxes
        if x1 == x0 or y1 == y0:
            return None

        return x0, y0, x1, y1
    return None


def _merge_rects(rects: List[BboxTuple], gap: float) -> List[BboxTuple]:
    """
    Greedy merge of overlapping or nearby rectangles.

    Args:
        rects: List of (x0, y0, x1, y1) coordinate tuples.
        gap: Expansion distance for proximity detection (pixels).

    Returns:
        List of merged (x0, y0, x1, y1) tuples.

    Algorithm:
        O(n²) greedy clustering. For each unmerged rect, expands by gap
        and absorbs all intersecting rects until no more intersections.

    Note:
        Uses fitz.Rect internally for efficient rectangle arithmetic,
        but accepts and returns pure tuples to maintain boundary isolation.
    """
    if not rects:
        return []

    import fitz  # Local import to isolate dependency

    # Convert input tuples to fitz.Rect for efficient operations
    fitz_rects: List[fitz.Rect] = [fitz.Rect(r) for r in rects]

    merged: List[fitz.Rect] = []
    used: set[int] = set()

    for i, r1 in enumerate(fitz_rects):
        if i in used:
            continue

        curr = r1
        used.add(i)
        changed = True

        while changed:
            changed = False
            for j, r2 in enumerate(fitz_rects):
                if j in used:
                    continue
                # Expand current rect by gap and check intersection
                expanded = curr + (-gap, -gap, gap, gap)
                if expanded.intersects(r2):
                    curr = curr | r2  # Union operation
                    used.add(j)
                    changed = True

        merged.append(curr)

    # Convert back to tuples at boundary
    return [(r.x0, r.y0, r.x1, r.y1) for r in merged]


def _get_span_sort_tolerance(span: Dict) -> int:
    """
    Returns appropriate row tolerance based on span context.

    Table content uses tight tolerance to preserve row integrity.
    Prose uses loose tolerance to handle italic/superscript baseline drift.

    This enables "Context-Aware Sorting" where the same document can have
    different sorting behavior for different content types, eliminating
    the need for a single "magic number" tolerance that compromises on both.

    Args:
        span: Span dictionary with optional '_is_table_content' flag
              (set by Step 4.5 of extract_page).

    Returns:
        Pixel tolerance for row bucketing:
        - 3px for table content (tight)
        - 8px for prose content (loose)

    Example:
        >>> span_prose = {"_is_table_content": False}
        >>> _get_span_sort_tolerance(span_prose)
        8
        >>> span_table = {"_is_table_content": True}
        >>> _get_span_sort_tolerance(span_table)
        3
    """
    return _SORT_TOLERANCE_TABLE if span.get("_is_table_content") else _SORT_TOLERANCE_PROSE


# ═══════════════════════════════════════════════════════════════════════════
# RONC v2.0 PHASE 5 — Protection Semantics
# ═══════════════════════════════════════════════════════════════════════════

# ✦────── b. Spatial Queries ──────✦

def _find_bbox_for_text(
        page: "fitz.Page",
        target_text: str,
        constraint_bbox: BboxInput = None
) -> Optional[BboxTuple]:
    """
    Fallback method for locating a bounding box when PyMuPDF doesn't provide one.

    Used for table cells with missing bboxes. Searches page text blocks
    for target text and returns the first matching bbox.

    Args:
        page: PyMuPDF page object.
        target_text: Text string to locate.
        constraint_bbox: Optional region constraint (any bbox format).
                        Only returns matches whose CENTER falls within this region.

    Returns:
        BboxTuple (x0, y0, x1, y1) or None if not found.

    Note:
        Uses center-point containment (not intersection) for constraint matching.
        This ensures text is "substantially owned" by the target region,
        avoiding false positives from spans that slightly bleed over boundaries.
    """
    if not target_text or not target_text.strip():
        return None

    # Search for text instances on page
    text_instances = page.search_for(target_text)
    if not text_instances:
        return None

    # Normalize constraint to tuple if provided
    constraint_tuple: Optional[BboxTuple] = None
    if constraint_bbox is not None:
        constraint_tuple = _to_bbox_tuple(constraint_bbox)

    # If we have a valid constraint, find instance with center inside it
    if constraint_tuple is not None:
        c_x0, c_y0, c_x1, c_y1 = constraint_tuple

        for rect in text_instances:
            # Check if rect center is within constraint
            rect_cx = (rect.x0 + rect.x1) / 2
            rect_cy = (rect.y0 + rect.y1) / 2

            if (c_x0 <= rect_cx <= c_x1) and (c_y0 <= rect_cy <= c_y1):
                return float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)

        # No match found within constraint
        return None

    # No constraint — return first instance
    rect = text_instances[0]
    return float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)


def _span_inside_visual_region(
        span: Dict,
        figure_tuples: Optional[List[BboxTuple]] = None,
        table_regions: Optional[List[Dict]] = None,
) -> bool:
    """
    Check if a span lies within any figure or table region.

    Args:
        span: Text span dictionary containing "bbox" field.
        figure_tuples: Pre-normalized figure regions as (x0, y0, x1, y1) tuples.
        table_regions: Table region dicts (normalized internally via _to_bbox_tuple).

    Returns:
        True if span intersects any visual region, False otherwise.

    Note:
        Uses intersection (any overlap) for containment check.
        Figure regions expected as tuples for performance (pre-normalized).
        Table regions accepted as dicts for caller convenience (normalized on-the-fly).
    """
    span_rect = _span_to_rect(span)
    if span_rect is None:
        return False

    # Check figures (already normalized to tuples)
    if figure_tuples:
        for fig_rect in figure_tuples:
            intersection = _get_intersection_rect(span_rect, fig_rect)
            if intersection:
                intersection_area = (intersection[2] - intersection[0]) * (
                        intersection[3] - intersection[1])
                span_area = (span_rect[2] - span_rect[0]) * (span_rect[3] - span_rect[1])
                if span_area > 0 and (intersection_area / span_area) > 0.5:
                    return True

    # Check tables (normalize dict bbox on-the-fly)
    if table_regions:
        for table in table_regions:
            table_bbox = table.get("bbox")
            if table_bbox is None:
                continue

            table_rect = _to_bbox_tuple(table_bbox)
            if table_rect is None:
                continue

            # Hardened: require significant overlap (same rationale as figures)
            span_area = _rect_area(span_rect)
            if span_area > 0:
                inter = _get_intersection_rect(span_rect, table_rect)
                if inter is not None:
                    inter_area = _rect_area(inter)
                    if (inter_area / span_area) >= 0.50:
                        return True

    return False


# ===========================================================================
# STAGE 1 SOFT CLASSIFICATION HELPERS (Schema v2.0)
# ===========================================================================

def _translate_exclusion_flags(
        spans: List[Dict],
        *,
        trace_id: str = None,
) -> Dict[str, int]:
    """
    Unified Flag Translator (Post-Semantic) v1.2.1:
      - Converts Stage 1 soft flags into authoritative exclusion decisions
      - Writes BOTH _tts_excluded AND _semantic_disposition for reconstruction
      - Respects EXPLICIT semantic decisions (not default inclusions)
      - Honors Pass 1 rescued spans (HARDEN normalization already applied)
    """
    audit = {"included": 0, "excluded_by": {}}

    def _inc(reason: str):
        audit["excluded_by"][reason] = audit["excluded_by"].get(reason, 0) + 1

    def _exclude(span: Dict, excl_reason: str):
        span["_tts_excluded"] = True
        span["_tts_exclude_reason"] = excl_reason
        span["_semantic_disposition"] = _SEM_DISP_EXCLUDED
        existing_reasons = list(span.get("_semantic_reasons") or [])
        existing_reasons.append(f"flag_translated:{excl_reason}")
        span["_semantic_reasons"] = existing_reasons
        _inc(excl_reason)

        if trace_id:
            logger.debug(
                "[%s] TTS exclusion applied: reason=%s, text='%s...'",
                trace_id,
                excl_reason,
                (span.get("cleaned_text") or "")[:40]
            )

    for sp in spans:
        if not isinstance(sp, dict):
            continue

        disp = sp.get("_semantic_disposition")
        reasons = sp.get("_semantic_reasons") or []
        confidence = sp.get("_semantic_confidence", 0.0)

        # ------------------------------------------------------------
        # Priority 0: Semantic explicit exclusion
        # ------------------------------------------------------------
        if disp == _SEM_DISP_EXCLUDED:
            sp["_tts_excluded"] = True
            sp["_tts_exclude_reason"] = sp.get("_tts_exclude_reason") or "semantic_excluded"
            _inc(sp["_tts_exclude_reason"])
            continue

        # ════════════════════════════════════════════════════════════════
        # STRUCTURAL AUTHORITY GATE (after semantic, before rescue)
        # Structural exclusion overrides rescue/protection logic.
        # Authority: Structural > Contractual (RONC) > Rescue > Default
        # ════════════════════════════════════════════════════════════════
        candidate_reasons_set = set(sp.get("_exclusion_candidate_reasons") or [])
        outlier_reasons_set = set(sp.get("_outlier_reasons") or [])
        if (candidate_reasons_set & _STRUCTURAL_EXCLUSION_REASONS
                or outlier_reasons_set & _STRUCTURAL_EXCLUSION_OUTLIER_REASONS):
            if not sp.get("_exclusion_protected", False):
                _exclude(sp, "structural_exclusion")
                continue

        # ------------------------------------------------------------
        # Priority 1: Semantic interruption (treat as excluded for TTS)
        # ------------------------------------------------------------
        if disp == _SEM_DISP_INTERRUPTION:
            # ─────────────────────────────────────────────────────────
            # P9 FIX: RONC protection override for interruptions
            # Spatial analysis can misclassify body text near figures
            # as "non-prose" (visual_overlap:non_prose). If RONC has
            # determined this span must be included (e.g., it anchors
            # a reading chain), trust semantic linking over spatial.
            # ─────────────────────────────────────────────────────────
            contract = sp.get("_ronc_contract") or {}
            protection = contract.get("protection") or {}

            if protection.get("must_include", False):
                sp["_tts_excluded"] = False
                sp["_tts_exclude_reason"] = None
                sp["_tts_include_reason"] = "ronc_override:interruption_rescued"
                sp["_tts_rescued"] = True  # Gate bypass for non-viable roles
                audit["included"] += 1
                continue

            if trace_id:
                logger.debug(
                    "[%s] Excluding interruption '%s...' (no RONC protection)",
                    trace_id, (sp.get("cleaned_text") or "")[:20]
                )
            _exclude(sp, "semantic_interruption")
            continue

        # ------------------------------------------------------------
        # Priority 2: Explicit semantic include
        # CRITICAL: default inclusion ≠ explicit include
        # ------------------------------------------------------------
        is_explicit_include = (
                disp == _SEM_DISP_INCLUDED
                and "default" not in reasons
                and confidence >= 0.6
        )
        if is_explicit_include:
            sp["_tts_excluded"] = False
            sp["_tts_exclude_reason"] = None
            sp["_tts_include_reason"] = "semantic_explicit_include"
            sp["_has_semantic_authority"] = True
            audit["included"] += 1
            continue

        # ════════════════════════════════════════════════════════════
        # Priority 2.3: Diagram Label Signal (P8 HARDENED)
        # ════════════════════════════════════════════════════════════
        # CRITICAL: Must run BEFORE RONC protection (Priority 2.4)
        # RONC can chain diagram labels (Force→control→signal) and
        # protect them as "atomic reading units", bypassing later checks.
        # ════════════════════════════════════════════════════════════
        signals = sp.get("_signals") or {}
        if (signals.get("diagram_label") or {}).get("is_label", False):
            # P5 FIX: Never exclude headings/subheadings as diagram_label
            # Short headings with technical terms trigger the detector,
            # but section structure must be preserved for TTS flow.
            if sp.get("role") in ("heading", "subheading"):
                sp["_tts_excluded"] = False
                sp["_tts_exclude_reason"] = None
                sp["_tts_include_reason"] = "heading_protected_from_diagram_label"
                audit["included"] += 1
                continue

            # P8 FIX: Update role from body to diagram_label for correct data model
            # This enables reconstruction systems to identify diagram labels
            if sp.get("role") == "body":
                sp["_original_role"] = "body"
                sp["role"] = TextRole.DIAGRAM_LABEL.value
                sp["_role_origin"] = "signal:diagram_label"

            _exclude(sp, "diagram_label")
            continue

        # ════════════════════════════════════════════════════════════
        # Priority 2.4: RONC v2.0 Contract-Based Protection
        # ════════════════════════════════════════════════════════════
        # Check contract protection FIRST (v2.0 takes precedence)
        # Falls through to v1.0 logic if no contract or no protection
        contract = sp.get("_ronc_contract")
        if contract:
            protection = contract.get("protection", {})
            if protection.get("must_include"):
                current_role = sp.get("role", "")

                # ──────────────────────────────────────────────────
                # GUARD (v10.0): Non-viable roles require physical
                # continuation evidence to override exclusion.
                #
                # RONC mutual links can form across role boundaries
                # (e.g., inside_figure ↔ body) via proximity scoring
                # alone. Without A2 edges or whitelist protection,
                # the span has no physical evidence of text-flow
                # membership. Author metadata and figure labels with
                # RONC links must not bypass role-based exclusion.
                #
                # Spans WITH evidence (e.g., somatosensory P0:2/P0:3
                # with _a2_qualified + greek_symbol whitelist) pass
                # this guard and receive protection as before.
                # ──────────────────────────────────────────────────
                allow_override = True
                if current_role in _TTS_NON_VIABLE_ROLES:
                    has_continuation_evidence = (
                            sp.get("_a2_qualified", False)
                            or sp.get("_a2_edge_exists", False)
                            or sp.get("_whitelist_protected") is not None
                    )
                    if not has_continuation_evidence:
                        allow_override = False
                        if trace_id:
                            logger.info(
                                "[%s] RONC v2.0: must_include blocked for "
                                "non-viable role '%s' without continuation "
                                "evidence (no A2/whitelist). text=%r",
                                trace_id, current_role,
                                (sp.get("cleaned_text") or "")[:40]
                            )

                if allow_override:
                    # Contract says this span must be included
                    sp["_tts_excluded"] = False
                    sp["_tts_exclude_reason"] = None
                    sp["_tts_include_reason"] = f"ronc_v2_contract:{protection.get('reason')}"
                    audit["included"] += 1

                    if trace_id:
                        span_text = (
                                            sp.get("cleaned_text")
                                            or sp.get("text")
                                            or sp.get("raw_text")
                                            or ""
                                    )[:40]
                        logger.debug(
                            "[%s] RONC v2.0: Contract protection applied: '%s...' reason=%s",
                            trace_id, span_text, protection.get('reason')
                        )

                    continue  # Skip to next span
                    # else: fall through to Priority 3+ chain

        # ------------------------------------------------------------
        # Priority 3.5: inside_figure rescue (PATCH 7B authoritative)
        # ------------------------------------------------------------
        if sp.get("_inside_figure_rescued", False):
            sp["_tts_excluded"] = False
            sp["_tts_exclude_reason"] = None
            sp["_tts_include_reason"] = "inside_figure_rescued"
            audit["included"] += 1
            continue

        # ------------------------------------------------------------
        # Priority 4+: Rule-based exclusion
        # ------------------------------------------------------------
        role = sp.get("role", "")

        # Priority 4: Non-viable role OR spatial inside-figure
        is_role_nv = role in _TTS_NON_VIABLE_ROLES
        is_spatial_fig = (sp.get("_spatial_context") or {}).get("inside_figure", False)

        if is_role_nv or is_spatial_fig:
            # Inside-figure rescue opportunity (role-based or spatial-based)
            can_attempt_rescue = (
                    (role == "inside_figure")
                    or (is_spatial_fig and not is_role_nv and "visual_overlap" not in (
                    sp.get("_exclusion_candidate_reasons") or []))
            )

            if can_attempt_rescue and _is_rescuable_inside_figure(sp):
                sp["_tts_excluded"] = False
                sp["_tts_exclude_reason"] = None
                sp["_tts_include_reason"] = "inside_figure_rescued"
                sp["_tts_rescued"] = True
                sp["_inside_figure_rescued"] = True
                sp["_semantic_disposition"] = _SEM_DISP_INCLUDED
                existing_reasons = list(sp.get("_semantic_reasons") or [])
                existing_reasons.append("flag_translated:inside_figure_rescued")
                sp["_semantic_reasons"] = existing_reasons
                audit["included"] += 1
                continue

            # Determine exclusion reason
            if is_role_nv:
                _exclude(sp, f"role:{role}" if role != "inside_figure" else "non_viable_role")
            else:
                _exclude(sp, "inside_figure")
            continue

        if sp.get("_is_table_content", False) or (sp.get("_spatial_context") or {}).get(
                "inside_table", False):
            _exclude(sp, "inside_table")
            continue

        if sp.get("_exclusion_candidate", False):
            if sp.get("_requires_semantic_review", False):
                _exclude(sp, "exclusion_candidate_unreviewed")
            else:
                _exclude(sp, "exclusion_candidate")
            continue

        # Catch-all for noise roles not handled by specific logic above
        if sp.get("role") in {"caption", "diagram_label", "margin", "inside_figure"}:
            _exclude(sp, f"final_guard_{sp.get('role')}")
            continue

        # ------------------------------------------------------------
        # Default include
        # ------------------------------------------------------------
        sp["_tts_excluded"] = False
        sp["_tts_exclude_reason"] = None
        sp["_tts_include_reason"] = "no_exclusion_rule_matched"
        audit["included"] += 1

    if trace_id:
        logger.info(
            "[%s] Flag translation: %d included, %d excluded %s",
            trace_id,
            audit["included"],
            sum(audit["excluded_by"].values()),
            dict(audit["excluded_by"]),
        )

    if trace_id:
        rescue_count = sum(
            1 for sp in spans
            if sp.get("_ronc_rescue_applied", False)
        )

        logger.info(
            "[%s] RONC enforcement: %d anchors rescued",
            trace_id, rescue_count
        )

    return audit


def _apply_line_coherent_streams(
        window_spans: List[Dict],
        *,
        trace_id: str = None,
) -> int:
    """
    Semantic-gated stream inheritance on WINDOW spans.

    Promotes margin_* spans to body_col_* ONLY when:
      1) Anchor (first span on line) is body_col_*
      2) Target span is semantically included
      3) Continuation gate passes
    """
    # PHASE 4 CONTRACT:
    # - This method MAY adjust layout_stream and structural ordering
    # - This method MUST NOT finalize semantic inclusion or exclusion
    # - This method MUST NOT set or remove _has_semantic_authority
    # - All exclusion signals here are PROVISIONAL and resolved in Phase 5

    from collections import defaultdict

    by_line: Dict[str, List[Dict]] = defaultdict(list)
    for sp in window_spans:
        if not isinstance(sp, dict):
            continue
        lid = sp.get("line_id")
        if lid:
            by_line[lid].append(sp)

    changed = 0

    for lid, line_spans in by_line.items():
        if len(line_spans) <= 1:
            continue

        line_spans.sort(key=lambda s: s.get("span_index_in_line", 0))

        # ─────────────────────────────────────────────────────────────────
        # RONC v2.1: Stream-aware anchor selection
        # Lines may contain spans from multiple streams (visual line collision).
        # Only proceed if there is exactly ONE body stream represented.
        # ─────────────────────────────────────────────────────────────────
        body_streams_on_line = set()
        for sp in line_spans:
            stream = sp.get("layout_stream") or ""
            if stream.startswith(_RONC_V2_BODY_COLUMN_PREFIX):
                body_streams_on_line.add(stream)

        # Gate 0a: Must have exactly one body stream (no ambiguity)
        if len(body_streams_on_line) != 1:
            continue

        anchor_stream = list(body_streams_on_line)[0]

        # Find first span in that body stream as anchor
        anchor = None
        for sp in line_spans:
            if sp.get("layout_stream") == anchor_stream:
                anchor = sp
                break

        if anchor is None:
            continue

        for sp in line_spans[1:]:
            current_stream = sp.get("layout_stream") or ""

            # Gate 1: target must be margin stream
            if not current_stream.startswith("margin_"):
                continue

            # IMPORTANT:
            # Semantic inclusion MUST be resolved before layout inheritance.
            # This gate must never be removed or weakened.
            # Gate 2: target must be semantically included
            disp = sp.get("_semantic_disposition")
            if disp != _SEM_DISP_INCLUDED:
                continue

            # RONC v2: do not structurally mutate protected spans
            contract = sp.get("_ronc_contract")
            if contract and contract.get("protection", {}).get("must_include"):
                continue

            # Gate 3: continuation gate
            if not _passes_inline_continuation_gate(anchor, sp):
                continue

            # Apply inheritance (COMPLETE)
            sp["_original_layout_stream"] = current_stream
            sp["layout_stream"] = anchor_stream

            # Inherit ALL structure to force correct sort order
            sp["_original_column_index"] = sp.get("column_index")
            sp["column_index"] = anchor.get("column_index")

            sp["_original_block_id"] = sp.get("block_id")
            sp["block_id"] = anchor.get("block_id")

            sp["_original_line_index"] = sp.get("line_index")
            sp["line_index"] = anchor.get("line_index")

            sp["_stream_inherited_from_line"] = True
            sp["_stream_inheritance_reason"] = "same_line_body_anchor"
            sp["_stream_inheritance_confidence"] = "high"
            changed += 1

    if trace_id and changed:
        logger.info("[%s] Line-coherent streams: corrected %d spans", trace_id, changed)

    # =========================================================================
    # PATCH 9B: Rescue sidebar spans that share line_id with body spans
    # Some italic text gets misclassified as sidebar due to positional offset.
    # Constraint: Only rescue short inline fragments (≤5 words) on body lines.
    # Restricted to body_col* streams only (prevents footnote/margin leakage).
    # =========================================================================
    sidebar_rescue_count = 0

    # ─────────────────────────────────────────────────────────────────────
    # RONC v2.1: Build stream-qualified body line index
    # Track line_id -> set of body streams to prevent cross-stream rescue
    # ─────────────────────────────────────────────────────────────────────
    body_line_streams = {}  # line_id -> set of body streams on that line
    for sp in window_spans:
        if sp.get("role") == "body":
            line_id = sp.get("line_id")
            stream = sp.get("layout_stream") or ""
            if line_id and stream.startswith(_RONC_V2_BODY_COLUMN_PREFIX):
                if line_id not in body_line_streams:
                    body_line_streams[line_id] = set()
                body_line_streams[line_id].add(stream)

    # Rescue sidebar spans on body lines
    for sp in window_spans:
        role = sp.get("role", "")
        line_id = sp.get("line_id")
        layout_stream = sp.get("layout_stream", "")

        if role == "sidebar" and line_id and line_id in body_line_streams:
            # ─────────────────────────────────────────────────────────────
            # RONC v2.1: Stream-aware rescue validation
            # Sidebar must be in a body stream AND that stream must match
            # one of the body streams present on this line.
            # ─────────────────────────────────────────────────────────────
            is_inline_context = layout_stream.startswith(_RONC_V2_BODY_COLUMN_PREFIX)
            streams_on_line = body_line_streams.get(line_id, set())
            stream_matches = layout_stream in streams_on_line

            if is_inline_context and stream_matches:
                text = (sp.get("cleaned_text") or "").strip()
                # Only rescue short inline fragments (≤5 words, likely italic terms)
                if text and len(text.split()) <= 5:
                    # --- Audit trail ---
                    if "_original_role" not in sp:
                        sp["_original_role"] = sp.get("role")

                    # Do NOT mutate role in Phase 4; defer semantic reclass to Phase 5
                    sp["_proposed_role"] = "body"
                    sp["_rescue_reason"] = "sidebar_rescued:same_line_body_sibling"
                    sp["_semantic_rescue_reason"] = "same_line_body_sibling"
                    sp["_stream_inherited_from_line"] = True
                    sp["_stream_inheritance_reason"] = "sidebar_same_line_body_sibling"
                    sp["_stream_inheritance_confidence"] = "medium"
                    sp["_layout_correction_applied"] = [
                        "sidebar_rescue_same_line_body",
                        "line_coherent_stream_inheritance"
                    ]
                    sidebar_rescue_count += 1
                    changed += 1

    if trace_id and sidebar_rescue_count > 0:
        logger.info(
            "[%s] Patch 9B: Rescued %d sidebar spans via same-line body sibling",
            trace_id, sidebar_rescue_count
        )

    # =========================================================================
    # PHASE 4 INVARIANT (DEBUG ONLY)
    # Phase 4 must not create, remove, or modify semantic authority.
    # =========================================================================
    if trace_id and __debug__:
        for sp in window_spans:
            if "_has_semantic_authority" in sp:
                # value must be unchanged from entry
                pass  # presence is allowed, mutation is not detectable here

    return changed


def _sync_window_spans_back_to_cache(
        window_spans: List[Dict],
        page_span_cache: Dict[int, List[Dict]],
        *,
        trace_id: str = None,
) -> int:
    """
    Sync mutations from window_spans back into page_span_cache.

    Handles ALL window spans (prev_tail + current + next_head) so
    overlapping windows converge.

    Confidence-aware: higher confidence decisions are not overwritten.
    """
    SYNC_FIELDS = (
        "_ronc_contract",
        "_semantic_disposition",
        "_semantic_reasons",
        "_semantic_confidence",
        "_tts_excluded",
        "_tts_exclude_reason",
        "_tts_include_reason",
        "_tts_rescued",

        # Stream inheritance
        "layout_stream",
        "_original_layout_stream",
        "_stream_inherited_from_line",

        # STRUCTURAL KEYS (CRITICAL)
        "column_index",
        "block_id",
        "line_index",

        # ─────────────────────────────────────────────────────
        # A2 Edge Qualification (v2.0)
        # Computed in Phase 1.8, consumed by RONC Phase 7
        # ─────────────────────────────────────────────────────
        "_a2_edge_exists",
        "_a2_cross_stream",
        "_a2_qualified",
        "_a2_edge_prev_id",

        # ─────────────────────────────────────────────────────
        # RONC — Reading Order Normalization Contract (Phase 5)
        # Metadata only. NO enforcement in sync.
        # ─────────────────────────────────────────────────────
        "_ronc_atomic_unit_id",
        "_ronc_atomic_role",
        "_ronc_break_after",
        "_ronc_rescue_applied",

        # Optional provenance (safe to keep)
        "_original_column_index",
        "_original_block_id",
        "_original_line_index",
        "_original_role",
        "_rescue_reason",
        "_inside_figure_rescued",
    )

    changed = 0

    for ws in window_spans:
        if not isinstance(ws, dict):
            continue

        page_idx = ws.get("_source_page_idx")
        local_idx = ws.get("_page_local_idx")

        # ─────────────────────────────────────────────────────────────────
        # P10 FIX: Fallback sync by canonical ID when source indices missing
        # Source indices may be absent in processed_spans_collector path.
        # Parse _canonical_span_id (e.g., "P3:45") to derive page/local idx.
        # ─────────────────────────────────────────────────────────────────
        if page_idx is None or local_idx is None:
            cid = ws.get("_canonical_span_id")
            if cid and ":" in cid:
                try:
                    page_part, idx_part = cid.split(":", 1)
                    if page_part.startswith("P"):
                        page_idx = int(page_part[1:])
                        local_idx = int(idx_part)
                        if trace_id:
                            logger.debug("[%s] P10 fallback: %s → page=%d, idx=%d",
                                         trace_id, cid, page_idx, local_idx)
                except (ValueError, IndexError):
                    continue
            else:
                continue

        if page_idx is None or local_idx is None:
            continue

        page_list = page_span_cache.get(page_idx)
        if not page_list:
            continue

        # ─────────────────────────────────────────────────────────────────
        # FIX v9.5: CID-based target selection
        # INVARIANT: Never trust list position after any sort/filter.
        # ─────────────────────────────────────────────────────────────────
        cid = ws.get("_canonical_span_id")
        if not cid:
            continue

        target = next(
            (s for s in page_list if s.get("_canonical_span_id") == cid),
            None
        )
        if target is None:
            continue
        if not isinstance(target, dict):
            continue

        ws_conf = ws.get("_semantic_confidence")
        tgt_conf = target.get("_semantic_confidence")

        INHERITANCE_GATED_FIELDS = {
            "layout_stream",
            "column_index",
            "block_id",
            "line_index",
        }

        for key in SYNC_FIELDS:
            if key not in ws:
                continue
            # Always sync semantic confidence itself (authority signal)
            if key == "_semantic_confidence":
                if target.get(key) != ws.get(key):
                    target[key] = ws[key]
                    changed += 1
                continue

            # Only sync structural fields if inheritance actually occurred
            if key in INHERITANCE_GATED_FIELDS:
                if not ws.get("_stream_inherited_from_line", False):
                    continue

            # ─────────────────────────────────────────────────────────────────
            # Confidence-aware sync: semantic disposition + RONC legacy fields
            # ─────────────────────────────────────────────────────────────────
            if key in {
                "_semantic_disposition",
                "_ronc_atomic_unit_id",
                "_ronc_atomic_role",
                "_ronc_break_after",
            }:
                if ws_conf is not None and tgt_conf is not None and ws_conf < tgt_conf:
                    continue

            # ─────────────────────────────────────────────────────────────────
            # Authority-aware sync for RONC contracts
            # Priority:
            #   1. Higher semantic confidence wins
            #   2. If confidence equal/unknown, protected beats unprotected
            #   3. Otherwise, last writer wins
            # ─────────────────────────────────────────────────────────────────
            if key == "_ronc_contract":
                ws_contract = ws.get("_ronc_contract", {})
                tgt_contract = target.get("_ronc_contract", {})

                if ws_conf is not None and tgt_conf is not None:
                    if ws_conf < tgt_conf:
                        continue

                # Priority 2: Protection-based authority (among equal/unknown confidence)
                ws_protected = ws_contract.get("protection", {}).get("must_include", False)
                tgt_protected = tgt_contract.get("protection", {}).get("must_include", False)

                if tgt_protected and not ws_protected:
                    continue

            if target.get(key) != ws.get(key):
                target[key] = ws[key]
                changed += 1


def _init_stage1_soft_fields(span: Dict) -> None:
    """
    Initialize soft classification fields on a span (idempotent).

    Stage 1 Contract: These fields enable deferred exclusion decisions.
    Never removes existing fields.
    """
    span.setdefault("_spatial_context", {
        "inside_figure": False,
        "figure_index": None,
        "overlap_ratio": 0.0,
        "near_figure": False,
        "figure_distance_ratio": 0.0,
        "inside_table": False,
        "table_index": None,
    })
    span.setdefault("_exclusion_candidate", False)
    span.setdefault("_exclusion_candidate_reasons", [])
    span.setdefault("_requires_semantic_review", False)


def _is_rescuable_inside_figure(sp: Dict) -> bool:
    """Single source of truth for inside_figure rescue eligibility.

    Consolidates identical logic from Priority 4a (role-based) and
    Priority 4b (spatial-based) in _translate_exclusion_flags.

    NOT used by:
      - Pass 2 rescue (_resolve_semantic_continuity): different criteria
        (a2 body anchor, role promotion, confidence scoring)
      - Narration gate P11 (_passes_narration_gate): different criteria
        (character-length threshold, role whitelist)
    """
    text = (sp.get("cleaned_text") or sp.get("raw_text") or "").strip()
    if not text:
        return False
    word_count = len(text.split())
    has_sentence_punct = text[-1] in ".!?,"
    is_short_continuation = (
            2 <= word_count <= 5
            and (text[0].islower() or text[0] in ",;:)")
    )
    is_body_like = word_count >= 5 or has_sentence_punct or is_short_continuation
    layout_stream = sp.get("layout_stream", "")
    is_body_stream = isinstance(layout_stream, str) and layout_stream.startswith("body_col")
    text_lower = text.lower()
    is_boilerplate = (
            "html" in text_lower or "css" in text_lower
            or "examples of how to" in text_lower
            or text_lower.startswith("figure ")
            or text_lower.startswith("table ")
    )
    return is_body_like and is_body_stream and not is_boilerplate


def _flag_candidate(
        span: Dict,
        reason: str,
        *,
        requires_review: bool = False
) -> None:
    """
    Flag a span as an exclusion candidate with reason tracking.

    Stage 1 Contract: Flags only, never removes spans.

    Args:
        span: Span to flag.
        reason: Exclusion reason constant.
        requires_review: If True, span MUST reach semantic window.
    """
    span["_exclusion_candidate"] = True
    reasons = span.setdefault("_exclusion_candidate_reasons", [])
    if reason not in reasons:
        reasons.append(reason)
    if requires_review:
        span["_requires_semantic_review"] = True


def _span_visual_context(
        span: Dict,
        figure_tuples: Optional[List[BboxTuple]] = None,
        table_regions: Optional[List[Dict]] = None,
        *,
        page_width: float,
        page_height: float,
        near_threshold_ratio: float = _SOFT_CLASSIFY_NEAR_THRESHOLD_RATIO,
) -> Dict:
    """
    Compute spatial context for a span relative to figures and tables.

    Stage 1 Contract: Reports context only, never decides viability.

    Args:
        span: Span with bbox.
        figure_tuples: Figure regions as (x0, y0, x1, y1) tuples.
        table_regions: Table region dicts with 'bbox' key.
        page_width: Page width for ratio calculations.
        page_height: Page height for ratio calculations.
        near_threshold_ratio: Proximity threshold as ratio of page dimension.

    Returns:
        Spatial context dictionary (never None).
    """
    ctx = {
        "inside_figure": False,
        "figure_index": None,
        "overlap_ratio": 0.0,
        "near_figure": False,
        "figure_distance_ratio": 0.0,
        "inside_table": False,
        "table_index": None,
    }

    span_rect = _span_to_rect(span)
    if span_rect is None:
        return ctx

    # Normalize to float for cross-source consistency (PDF / DOM)
    span_rect = tuple(float(v) for v in span_rect)

    page_dimension = max(page_width, page_height, 1.0)
    near_pad = min(page_width, page_height) * near_threshold_ratio

    # === FIGURE ANALYSIS ===
    best_overlap_ratio = 0.0
    best_figure_idx = None
    min_figure_distance = float('inf')

    if figure_tuples:
        for idx, fig_rect in enumerate(figure_tuples):
            intersection = _get_intersection_rect(span_rect, fig_rect)

            if intersection is not None:
                inter_area = _rect_area(intersection)
                span_area = max(1e-6, _rect_area(span_rect))
                ratio = inter_area / span_area

                if ratio > best_overlap_ratio:
                    best_overlap_ratio = ratio
                    best_figure_idx = idx
            else:
                # Compute edge distance for near-figure detection
                dx = max(fig_rect[0] - span_rect[2], span_rect[0] - fig_rect[2], 0)
                dy = max(fig_rect[1] - span_rect[3], span_rect[1] - fig_rect[3], 0)
                dist = (dx ** 2 + dy ** 2) ** 0.5
                min_figure_distance = min(min_figure_distance, dist)

                # Near-figure by center point within padded bounds
                cx = (span_rect[0] + span_rect[2]) / 2
                cy = (span_rect[1] + span_rect[3]) / 2
                if (fig_rect[0] - near_pad <= cx <= fig_rect[2] + near_pad and
                        fig_rect[1] - near_pad <= cy <= fig_rect[3] + near_pad):
                    ctx["near_figure"] = True

    ctx["overlap_ratio"] = best_overlap_ratio
    if best_figure_idx is not None and best_overlap_ratio >= _SOFT_CLASSIFY_OVERLAP_THRESHOLD:
        ctx["inside_figure"] = True
        ctx["figure_index"] = best_figure_idx

    # Figure distance ratio (0.0 if inside, otherwise normalized distance)
    if not ctx["inside_figure"] and min_figure_distance < float('inf'):
        ctx["figure_distance_ratio"] = min_figure_distance / page_dimension

    # === TABLE ANALYSIS ===
    if table_regions:
        for ti, table in enumerate(table_regions):
            table_bbox = table.get("bbox")
            if table_bbox is None:
                continue

            table_rect = _to_bbox_tuple(table_bbox)
            if table_rect is None:
                continue

            intersection = _get_intersection_rect(span_rect, table_rect)
            if intersection is not None:
                inter_area = _rect_area(intersection)
                span_area = max(1e-6, _rect_area(span_rect))
                if (inter_area / span_area) >= _SOFT_CLASSIFY_OVERLAP_THRESHOLD:
                    ctx["inside_table"] = True
                    ctx["table_index"] = ti
                    break

    return ctx


def _diagram_label_signal(
        span: Dict,
        figure_tuples: List[BboxTuple],
        spatial_context: Dict,
        baseline_font_size: float = 12.0,
        trace_id: str = None,
) -> Dict:
    """
    Compute diagram label detection signal.

    Stage 1 Contract: Returns signal dict, never decides viability.

    Args:
        span: Span to evaluate.
        figure_tuples: Figure regions.
        spatial_context: Pre-computed spatial context (avoids recomputation).
        baseline_font_size: Document baseline font size.
        trace_id: Optional trace ID.

    Returns:
        Signal dict with is_label, confidence, detection_path, reasons.
    """
    try:
        is_label = _is_diagram_label(span, figure_tuples, baseline_font_size, trace_id)
    except Exception:
        is_label = False

    # Determine detection path based on spatial context
    detection_path = None
    if is_label:
        if spatial_context.get("inside_figure", False):
            detection_path = "geometric"
        else:
            detection_path = "orphan"

    return {
        "is_label": is_label,
        "confidence": 0.9 if is_label else 0.0,
        "detection_path": detection_path,
        "reasons": ["diagram_label_detector"] if is_label else [],
    }


def _derive_legacy_content(classified_spans: List[Dict]) -> List[Dict]:
    """
    Derive legacy 'content' list from classified spans for backward compatibility.

    Stage 1 Contract: classified_spans is never mutated.

    Returns:
        List of spans that would have passed the old filter logic.
    """
    return [
        s for s in classified_spans
        if not s.get("_exclusion_candidate", False)
           or s.get("_exclusion_protected", False)
    ]


# ✦────── c. Layout & Margins ──────✦

def _get_page_margins(
        page_width: float,
        config: ExtractionConfig = None
) -> Tuple[float, float]:
    """
    Compute left and right sidebar zone boundaries for role classification.

    Args:
        page_width: Page width in points.
        config: Optional extraction config (uses default if not provided).

    Returns:
        (left_boundary, right_boundary): X-coordinates defining sidebar zones.
        Content with X < left_boundary or X > right_boundary is sidebar/margin.
    """
    if config is None:
        config = ExtractionConfig()

    margin_ratio = config.roles.sidebar_margin_ratio
    left_margin = page_width * margin_ratio
    right_margin = page_width * (1.0 - margin_ratio)

    return left_margin, right_margin


def _detect_columns(
        spans: List[Dict],
        figure_tuples: List[BboxTuple],
        tables: List[Dict],
        page_width: float,
        trace_id: str = None
) -> int:
    """
    Adaptive column detection with smart thresholds and margin isolation.

    HARDENED v2.1 (Margin Isolation Update):
        1. REMOVED: "Imbalance Correction" that forced 1-column layout on sidebars.
        2. ENHANCED: Variable gap threshold based on page density.
        3. NEW: Margin content gets reserved column indices to ensure structural
           separation from body text during downstream processing.

    Column Index Assignment:
        - Left margin content: _COLUMN_INDEX_LEFT_MARGIN (-1)
        - Right margin content: _COLUMN_INDEX_RIGHT_MARGIN (100)
        - Body content: 0 to (num_columns - 1)

    Args:
        spans: List of span dictionaries to analyze and assign columns.
        figure_tuples: Figure bounding boxes to exclude from analysis.
        tables: Table dictionaries with bbox to exclude from analysis.
        page_width: Page width in points for threshold calculations.
        trace_id: Optional trace ID for observability logging.

    Returns:
        Number of body columns detected (excludes margin columns).
    """
    if not spans:
        return 1

    if not page_width or page_width <= 0:
        page_width = _LAYOUT_DEFAULT_PAGE_WIDTH

    # =========================================================================
    # Step 1: Exclusion Zones (Figures, Tables)
    # =========================================================================
    exclusion_rects = [r for r in figure_tuples if r and len(r) >= 4]

    for table in tables:
        if table.get("bbox"):
            t_rect = _to_bbox_tuple(table["bbox"])
            if t_rect and len(t_rect) >= 4:
                exclusion_rects.append(t_rect)

    # =========================================================================
    # Step 2: Detect Margin Boundaries
    # =========================================================================
    body_left, body_right = _detect_margin_boundaries(spans, page_width, trace_id=trace_id)
    # HARDEN: Ensure sane body boundaries
    if body_left is None or body_right is None:
        body_left, body_right = 0.0, page_width

    if body_right <= body_left:
        body_left, body_right = 0.0, page_width

    # =========================================================================
    # Step 3: Calculate X-Centroids and Tag Margin Content
    # =========================================================================
    all_y = [s.get("bbox", [0, 0, 0, 0])[1] for s in spans if s.get("bbox")]
    all_y += [s.get("bbox", [0, 0, 0, 0])[3] for s in spans if s.get("bbox")]

    if all_y:
        page_y_min, page_y_max = min(all_y), max(all_y)
        page_h_est = max(page_y_max - page_y_min, 1.0)
    else:
        page_y_min, page_h_est = 0.0, 1000.0

    x_centroids = []
    centroid_map = {}
    span_rect_map = {}

    for span in spans:
        span_rect = _span_to_rect(span)
        if span_rect is None:
            span["column_index"] = 0
            continue

        x0, y0, x1, y1 = span_rect
        cx = (x0 + x1) / 2
        centroid_map[id(span)] = cx
        span_rect_map[id(span)] = span_rect

        # NOTE: If this ever misclassifies body content as margin,
        # the bug is in _detect_margin_boundaries, not here.
        is_margin = (x0 < body_left or x0 > body_right)
        span["is_margin_content"] = is_margin

        # Only body content (non-margin, non-excluded) contributes to grid detection
        in_exclusion = any(_rects_intersect(span_rect, ex) for ex in exclusion_rects)
        span_width = x1 - x0
        is_structural_banner = span_width > (page_width * 0.80)

        # REVISION (Vertical Safety Zone / "Gutter Pollution" Guard):
        # _detect_columns() does not receive page_height, so infer vertical bounds
        # from observed span rects. This prevents header/footer tokens (often centered)
        # from contributing centroids that "plug" the true gutter.
        #
        # NOTE: This guard ONLY affects x_centroids sampling (grid detection),
        # not downstream column assignment.
        y0 = span_rect[1]
        y1 = span_rect[3]

        # Infer page vertical extent from already-seen spans (cheap + stable).
        # Update inferred page vertical extent (cheap + stable)

        y_center = (y0 + y1) / 2.0
        norm_y = (y_center - page_y_min) / page_h_est

        # Ignore top/bottom 8% for centroid sampling (headers/footers/live page numbers).
        is_vertical_extreme = (norm_y < 0.08) or (norm_y > 0.92)

        # =====================================================================
        # [P2 HARDENING] Figure-annotation exclusion to prevent "ghost columns"
        #
        # Even when figure/table bboxes are imperfect, short label-like spans
        # (e.g., "Force", "Signal", axis ticks) can pollute the centroid
        # distribution and create phantom boundaries (Page 4 failure mode).
        # =====================================================================
        span_text = (span.get("raw_text", "") or span.get("cleaned_text", "") or "").strip()
        span_role = span.get("role", "body")
        is_figure_annotation = (
                span_role in (TextRole.FIGURE_LABEL.value, TextRole.CAPTION.value,
                              TextRole.INSIDE_FIGURE.value, TextRole.TABLE_CELL.value) or
                (0 < len(span_text) < 5 and in_exclusion)
        )

        if (
                not in_exclusion
                and not is_margin
                and not is_vertical_extreme
                and not is_structural_banner
                and not is_figure_annotation
        ):
            x_centroids.append(cx)

    if not x_centroids:
        return 1

    # =========================================================================
    # Step 4: Gap Analysis
    # =========================================================================
    x_centroids.sort()
    gaps = []
    for i in range(1, len(x_centroids)):
        gap_size = x_centroids[i] - x_centroids[i - 1]
        gaps.append((gap_size, x_centroids[i - 1], x_centroids[i]))

    # =========================================================================
    # Step 5: Threshold Calculation (The "Typographic Anchor")
    # =========================================================================
    # REPLACED: base_threshold = max(page_width * 0.05, 20.0)
    #
    # FIX: Decouple from page width (avoids High-DPI "Resolution Trap").
    # Anchor to text size: A column gutter is typographically distinct if
    # it is wider than ~2.0 characters (2.0em).

    font_sizes = [s.get("font_size", 10.0) for s in spans if s.get("font_size", 0) > 0]

    if font_sizes:
        # Quick median calculation (surgical inline sort)
        font_sizes.sort()
        median_fs = font_sizes[len(font_sizes) // 2]
    else:
        median_fs = 10.0

    # THRESHOLD: 2.5x Font Size or 15.0pt floor.
    # Academic gutters are often 1pc (12pt); previous 30pt floor was too aggressive.

    base_threshold = max(median_fs * 2.5, 15.0)

    significant_gaps = [g for g in gaps if g[0] > base_threshold]
    significant_gaps.sort(key=lambda x: x[0], reverse=True)

    # =========================================================================
    # Step 6: Determine Column Boundaries
    # =========================================================================
    num_columns = 1
    boundaries = []

    if significant_gaps:
        valid_boundaries = []
        for size, left, right in significant_gaps:
            midpoint = (left + right) / 2
            # Only accept boundaries in center 50% of page
            if (page_width * 0.20) < midpoint < (page_width * 0.80):

                # REVISION A: Block-aware suppression
                # If the same physical block_id exists on both sides of the proposed boundary,
                # the boundary is likely slicing a single true text block (phantom column wall).
                left_blocks = set()
                right_blocks = set()

                for s in spans:
                    if s.get("is_margin_content"):
                        continue

                    rect = span_rect_map.get(id(s))
                    if not rect:
                        continue

                    # Exclude figure/table regions from this guard (same as centroid grid input)
                    if any(_rects_intersect(rect, ex) for ex in exclusion_rects):
                        continue

                    cx = centroid_map.get(id(s))
                    if cx is None:
                        cx = (rect[0] + rect[2]) / 2

                    bid = s.get("block_id")
                    if bid is None:
                        continue

                    if cx < midpoint:
                        left_blocks.add(bid)
                    elif cx > midpoint:
                        right_blocks.add(bid)

                # FUZZY VETO (P0 FIX):
                # Allow a boundary even if a small number of blocks cross it (e.g., Titles),
                # provided the vast majority of body spans respect the split.
                overlapping_blocks = left_blocks & right_blocks

                if overlapping_blocks:
                    total_body_spans = 0
                    violating_spans = 0

                    # Dynamic Y-Bounds for Gating
                    all_y0 = [s["bbox"][1] for s in spans if s.get("bbox")]
                    all_y1 = [s["bbox"][3] for s in spans if s.get("bbox")]

                    if all_y0 and all_y1:
                        page_y_min = min(all_y0)
                        page_y_max = max(all_y1)
                        page_height_est = max(1.0, page_y_max - page_y_min)
                    else:
                        page_y_min, page_height_est = 0.0, 1000.0  # Fallback

                    for s in spans:
                        if s.get("is_margin_content"):
                            continue

                        rect = span_rect_map.get(id(s))
                        if not rect:
                            continue

                        # Ignore excluded regions (figures/tables)
                        if any(_rects_intersect(rect, ex) for ex in exclusion_rects):
                            continue

                        # Ignore vertical extremes (headers / footers)
                        y0, y1 = rect[1], rect[3]
                        y_center = (y0 + y1) / 2.0
                        norm_y = (y_center - page_y_min) / page_height_est

                        # SKIP Header/Footer Zones (Top/Bottom 8%)
                        if norm_y < 0.08 or norm_y > 0.92:
                            continue

                        total_body_spans += 1

                        bid = s.get("block_id")
                        if bid in overlapping_blocks:
                            violating_spans += 1

                    # HARD RULE: allow boundary if ≥90% of body spans respect it
                    if total_body_spans > 0:
                        violation_ratio = violating_spans / total_body_spans
                        if violation_ratio > 0.10:
                            continue

                valid_boundaries.append(midpoint)
                if len(valid_boundaries) >= (_LAYOUT_MAX_COLUMNS - 1):
                    break

        if valid_boundaries:
            boundaries = sorted(valid_boundaries)
            num_columns = len(boundaries) + 1

    # =========================================================================
    # Step 7: Assign Column Indices (MODIFIED v2.1 — Margin Isolation)
    # =========================================================================
    # Margin content gets reserved column indices to ensure structural
    # separation from body content. This guarantees downstream methods
    # (_reconstruct_text_for_segmentation, _stitch_helper_columns_match)
    # treat margin and body as separate streams.
    #
    # Assignment:
    #   - Left margin (cx < body_left): _COLUMN_INDEX_LEFT_MARGIN (-1)
    #   - Right margin (cx > body_right): _COLUMN_INDEX_RIGHT_MARGIN (100)
    #   - Body content: 0 to (num_columns - 1) based on boundaries

    margin_left_count = 0
    margin_right_count = 0

    # HARDEN: Ensure all spans receive layout_stream (prevents silent corruption)
    for span in spans:
        span.setdefault("layout_stream", "body_col_0")

    for span in spans:
        cx = centroid_map.get(id(span))
        span_rect = span_rect_map.get(id(span))

        if cx is None:
            span["column_index"] = 0
            span["layout_stream"] = "body_col_0"
            continue

        # MARGIN ISOLATION: Reserved column indices for margin content
        if not span_rect:
            span["column_index"] = 0
            span["layout_stream"] = "body_col_0"
            continue

        span_width = span_rect[2] - span_rect[0]

        # Full-width banners should never be treated as margin
        if span_width > (page_width * 0.85):
            span["column_index"] = 0
            span["layout_stream"] = "full_width_banner"
            span["is_margin_content"] = False
            continue

        # MARGIN ISOLATION: Reserved column indices for margin content
        if span.get("is_margin_content"):
            if cx < body_left:
                span["column_index"] = _COLUMN_INDEX_LEFT_MARGIN
                span["layout_stream"] = "margin_left"
                margin_left_count += 1
            else:
                span["column_index"] = _COLUMN_INDEX_RIGHT_MARGIN
                span["layout_stream"] = "margin_right"
                margin_right_count += 1
            continue

        if num_columns == 1:
            span["column_index"] = 0
            span["layout_stream"] = "body_col_0"
        else:
            effective_center = cx
            if span_rect:
                effective_center = (span_rect[0] + span_rect[2]) / 2

            col_idx = 0
            for boundary in boundaries:
                if effective_center >= boundary:
                    col_idx += 1
                else:
                    break
            span["column_index"] = col_idx
            span["layout_stream"] = f"body_col_{col_idx}"

    # =========================================================================
    # PHASE 2.1: Ragged-Edge Magnet Rule
    # Corrects margin misclassification for spans on BODY-majority visual lines.
    # VLG = (line_id, row_key)
    # Safe Harbor: visual distance check prevents sidebar pollution.
    # Contract:
    #   May mutate: is_margin_content, column_index, layout_stream
    #   Must not mutate: raw_text, bbox, line_bbox, block_bbox
    # =========================================================================

    vlg_groups = defaultdict(list)
    for span in spans:
        lid = span.get("line_id")
        rk = span.get("row_key")
        if lid is not None and rk is not None:
            vlg_groups[(lid, rk)].append(span)

    magnet_reclassified = 0

    for _, group in vlg_groups.items():
        body_spans = [s for s in group if not s.get("is_margin_content")]
        margin_spans = [s for s in group if s.get("is_margin_content")]

        # Majority vote — BODY must dominate
        if len(body_spans) <= len(margin_spans):
            continue

        if not body_spans or not margin_spans:
            continue

        for m in margin_spans:
            m_rect = span_rect_map.get(id(m))
            if not m_rect:
                continue

            mx0, mx1 = float(m_rect[0]), float(m_rect[2])
            fs = float(m.get("font_size", 10.0) or 10.0)

            min_gap = float("inf")
            nearest_col = 0

            for b in body_spans:
                b_rect = span_rect_map.get(id(b))
                if not b_rect:
                    continue

                bx0, bx1 = float(b_rect[0]), float(b_rect[2])

                gap = min(
                    abs(mx0 - bx1),  # margin right of body
                    abs(bx0 - mx1),  # margin left of body
                )

                if gap < min_gap:
                    min_gap = gap
                    nearest_col = int(b.get("column_index", 0) or 0)

            # Safe-harbor threshold (existing behavior)
            if min_gap <= (MAGNET_GAP_EM * fs):
                m["is_margin_content"] = False
                m["column_index"] = nearest_col
                m["layout_stream"] = f"body_col_{nearest_col}"
                magnet_reclassified += 1

            # FM2 ADDITION (from e2, approved):
            # Within-VLG a2 fallback when gap fails but body-majority already holds
            elif (
                    m.get("role") not in _MAGNET_A2_NON_PROMOTABLE_ROLES
                    and (m.get("a2_continues_from_previous")
                         or m.get("a2_continues_to_next"))
            ):
                m["is_margin_content"] = False
                m["column_index"] = nearest_col
                m["layout_stream"] = f"body_col_{nearest_col}"
                m["_a2_magnet_reclassified"] = True
                magnet_reclassified += 1

    # ─────────────────────────────────────────────────────────────────────
    # FM2 (your design): Cross-VLG A2 Magnet Fallback
    # Handles split line_id cases (Diagnostic A)
    # NO body-majority requirement — a2 + body anchor replaces it
    # ─────────────────────────────────────────────────────────────────────
    row_key_groups = defaultdict(list)
    for span in spans:
        rk = span.get("row_key")
        if rk is not None:
            row_key_groups[rk].append(span)

    for rk, group in row_key_groups.items():
        body_on_row = [
            s for s in group
            if not s.get("is_margin_content")
               and str(s.get("layout_stream", "")).startswith("body_col")
        ]
        if not body_on_row:
            continue

        for m in group:
            if not m.get("is_margin_content"):
                continue
            if m.get("role") in _MAGNET_A2_NON_PROMOTABLE_ROLES:
                continue
            if not (m.get("a2_continues_from_previous")
                    or m.get("a2_continues_to_next")):
                continue

            m_rect = span_rect_map.get(id(m))
            if not m_rect:
                continue

            mx0, mx1 = float(m_rect[0]), float(m_rect[2])
            nearest_col = 0
            min_gap = float("inf")

            for b in body_on_row:
                b_rect = span_rect_map.get(id(b))
                if not b_rect:
                    continue
                bx0, bx1 = float(b_rect[0]), float(b_rect[2])
                gap = min(abs(mx0 - bx1), abs(bx0 - mx1))
                if gap < min_gap:
                    min_gap = gap
                    nearest_col = int(b.get("column_index", 0) or 0)

            m["is_margin_content"] = False
            m["column_index"] = nearest_col
            m["layout_stream"] = f"body_col_{nearest_col}"
            m["_a2_magnet_reclassified"] = True
            magnet_reclassified += 1

    # FM2 cross-VLG observability
    if trace_id:
        a2_cross_vlg = sum(1 for s in spans if s.get("_a2_magnet_reclassified"))
        if a2_cross_vlg:
            logger.info(
                "[%s] FM2 cross-VLG: %d spans reclassified via a2 chain",
                trace_id, a2_cross_vlg
            )

    # =========================================================================
    # PHASE 2.2: Line-End Punctuation Guard (REQUIRED)
    # Flags legitimate punctuation so it survives short-fragment pruning.
    # =========================================================================
    LINE_END_PUNCTUATION = {
        ".", ")", ").", ":", ";", "?", "\"", "]", "].", "!", "'",
    }
    LINE_START_PUNCTUATION = {
        "\"", "'", "(", "[", "¿", "¡", "«",
    }

    punctuation_protected = 0

    for _, group in vlg_groups.items():
        body_count = sum(1 for s in group if not s.get("is_margin_content"))
        margin_count = sum(1 for s in group if s.get("is_margin_content"))

        if body_count <= margin_count:
            continue

        for s in group:
            text = (s.get("raw_text") or "").strip()
            if not text:
                continue

            is_line_end = s.get("is_line_end", False)
            is_line_start = s.get("span_index_in_line", -1) == 0

            # Line-end protection (allows short punctuation)
            if is_line_end:
                if text in LINE_END_PUNCTUATION or (len(text) <= 3 and not text.isalnum()):
                    s["filter_protected"] = True
                    punctuation_protected += 1
                    continue

            # Line-start protection (STRICT punctuation only)
            if is_line_start:
                if text in LINE_START_PUNCTUATION:
                    s["filter_protected"] = True
                    punctuation_protected += 1

    if trace_id and (magnet_reclassified > 0 or punctuation_protected > 0):
        logger.info(
            "[%s] Phase 2: Magnet reclassified %d spans; protected %d punctuation",
            trace_id, magnet_reclassified, punctuation_protected
        )

    if trace_id:
        margin_left_count = sum(1 for s in spans if s.get("layout_stream") == "margin_left")
        margin_right_count = sum(1 for s in spans if s.get("layout_stream") == "margin_right")

    # =========================================================================
    # Logging
    # =========================================================================
    if trace_id:
        logger.info(
            "[%s] Column Detection: %d cols (boundaries: %s), margins: L=%d R=%d",
            trace_id, num_columns, [round(b, 1) for b in boundaries],
            margin_left_count, margin_right_count
        )

    return num_columns


def _extract_column_number(stream_name: str) -> int:
    """Extract numeric column index from layout_stream for deterministic ordering."""
    try:
        return int(stream_name.split("_")[-1])
    except (ValueError, IndexError, AttributeError, TypeError):
        return 10 ** 9


def _detect_margin_boundaries(
        spans: List[Dict],
        page_width: float,
        exclude_roles: frozenset[TextRole] = None,
        trace_id: str = None
) -> Tuple[float, float]:
    """
    Adaptive margin detection via weighted X-histogram analysis.

    Strategy:
        1. Build weighted histogram of span X-positions (adaptive bin count)
        2. Exclude specified roles from analysis
        3. Find leftmost and rightmost "dense" bins using adaptive threshold
        4. Return body boundaries with validation

    Args:
        spans: List of span dicts with bbox.
        page_width: Page width in points.
        page_height: Page height in points (reserved for future use).
        exclude_roles: Roles to exclude from histogram (defaults to figure/table roles).
        trace_id: Optional trace ID for logging.

    Returns:
        (body_left, body_right): X-coordinates defining body region.
        Spans with X < body_left or X > body_right are margin content.
    """
    # =========================================================================
    # VALIDATION
    # =========================================================================
    if not spans or page_width <= 0:
        default_width = page_width if page_width > 0 else _LAYOUT_DEFAULT_PAGE_WIDTH
        return 0.0, default_width

    if exclude_roles is None:
        exclude_roles = _MARGIN_EXCLUDE_ROLES

    # =========================================================================
    # CONFIGURATION: DPI/Resolution normalization (Reference-Width Histogram)
    # =========================================================================
    REF_WIDTH = 600.0
    norm_scale = (REF_WIDTH / page_width) if page_width > REF_WIDTH else 1.0
    norm_width = page_width * norm_scale  # equals REF_WIDTH when page is wider than REF

    # =====================================================================
    # [P0 HARDENING] High-resolution histogram
    # Rationale:
    # - With coarse bins (~30px), legitimate 2-col gutters and sidebar gaps
    # can quantize into similar "empty-bin" counts.
    # - Force ~12px/bin at REF_WIDTH≈600 so:
    # gutter (12–20px) -> 1–2 bins
    # sidebar gap (40–60px) -> 3–5 bins
    # This enables density-aware region classification below.
    # =====================================================================
    HIGH_RES_BIN_COUNT = 50  # ~12px per bin at 600px reference width
    num_bins = max(
        HIGH_RES_BIN_COUNT,
        min(_MARGIN_MAX_BINS, int(norm_width / 12.0))
    )
    bin_width = norm_width / num_bins

    # =========================================================================
    # BUILD WEIGHTED HISTOGRAM
    # =========================================================================
    bins: List[float] = [0.0] * num_bins
    bin_y_ranges: List[List[float]] = [[float('inf'), float('-inf')] for _ in range(num_bins)]
    span_count = 0

    for span in spans:
        # Rationale: Table spans can cover full width and corrupt margin/body bands
        if span.get("_is_table_content", False):
            continue

        # Skip excluded roles
        role_str = span.get("role", "body")

        # Check against TextRole enum values
        try:
            role = TextRole(role_str)
            if role in exclude_roles:
                continue
        except ValueError:
            # Unknown role string — include in analysis
            pass

        span_rect = _span_to_rect(span)
        if span_rect is None:
            continue

        left_x = span_rect[0]
        left_x_norm = left_x * norm_scale

        # Clamp to valid bin range (normalized coordinate system)
        bin_idx = max(0, min(int(left_x_norm / bin_width), num_bins - 1))

        # Weight by character count (optional)
        if _MARGIN_WEIGHT_BY_CHARS:
            span_text = span.get("cleaned_text") or span.get("raw_text") or ""
            weight = max(1, len(span_text))
        else:
            weight = 1

        bins[bin_idx] += weight
        span_count += 1

        # Track vertical extent for this bin
        y0, y1 = span_rect[1], span_rect[3]
        if y0 < bin_y_ranges[bin_idx][0]:
            bin_y_ranges[bin_idx][0] = y0
        if y1 > bin_y_ranges[bin_idx][1]:
            bin_y_ranges[bin_idx][1] = y1

    # =========================================================================
    # HANDLE EDGE CASES
    # =========================================================================
    if span_count == 0:
        if trace_id:
            logger.debug("[%s] No valid spans for margin detection, using defaults", trace_id)
        return (
            page_width * _MARGIN_MIN_RATIO,
            page_width * (1 - _MARGIN_MIN_RATIO)
        )

    max_count = max(bins) if bins else 1
    if max_count == 0:
        return (
            page_width * _MARGIN_MIN_RATIO,
            page_width * (1 - _MARGIN_MIN_RATIO)
        )

    # =========================================================================
    # APPLY VERTICAL COVERAGE BONUS
    # =========================================================================
    all_y_min = min((r[0] for r in bin_y_ranges if r[0] != float('inf')), default=0)
    all_y_max = max((r[1] for r in bin_y_ranges if r[1] != float('-inf')), default=1)
    page_h_est = max(all_y_max - all_y_min, 1.0)

    for i in range(num_bins):
        y_min, y_max = bin_y_ranges[i]
        if y_min != float('inf') and bins[i] > 0:
            vertical_coverage = (y_max - y_min) / page_h_est
            # Bonus: 1.0x at 0% coverage → 1.5x at 100% coverage
            coverage_bonus = 1.0 + (vertical_coverage * 0.5)
            bins[i] *= coverage_bonus

    # =========================================================================
    # CALCULATE DENSITY THRESHOLD
    # =========================================================================
    active_bins = sorted(b for b in bins if b > 0)
    if not active_bins:
        return (
            page_width * _MARGIN_MIN_RATIO,
            page_width * (1 - _MARGIN_MIN_RATIO)
        )

    median_val = active_bins[len(active_bins) // 2]
    density_threshold = max(
        _MARGIN_MIN_DENSITY_COUNT,
        median_val * 0.35
    )

    # =========================================================================
    # FIND BODY BOUNDARIES (WIDTH-DOMINANT / LCR)
    # Strategy:
    #   1. Identify all contiguous dense regions
    #   2. Merge across small gaps (figures, indents)
    #   3. Select the WIDEST region as Body
    #   4. Use centrality only as a tiebreaker
    # =========================================================================

    max_gap = 2

    regions = []
    region_start = None
    region_weight = 0
    gap_count = 0

    for i, val in enumerate(bins):
        if val >= density_threshold:
            if region_start is None:
                region_start = i
            region_weight += val
            gap_count = 0
        else:
            gap_count += 1
            if gap_count > max_gap and region_start is not None:
                region_end = i - gap_count
                regions.append((region_start, region_end, region_weight))
                region_start = None
                region_weight = 0

    if region_start is not None:
        # Find actual last dense bin (exclude trailing sparse bins)
        actual_end = region_start
        for j in range(num_bins - 1, region_start - 1, -1):
            if bins[j] >= density_threshold:
                actual_end = j
                break
        regions.append((region_start, actual_end, region_weight))

    if not regions:
        body_left_bin = int(num_bins * 0.1)
        body_right_bin = int(num_bins * 0.9)
    else:
        # =====================================================================
        # [P0 CRITICAL] Density-aware region classification
        #
        # Problem:
        #   - "Pick the heaviest region" can exclude a legitimate 2nd column
        #     (IEEE/ACM) OR include a sidebar/margin block as body.
        #
        # Solution:
        #   - Compare the top two regions by weight.
        #       ratio = secondary_weight / primary_weight
        #      * ratio > 0.50  => likely true multi-column -> MERGE
        #      * ratio < 0.30  => likely sidebar/margin    -> EXCLUDE secondary
        #      * else          => ambiguous -> choose leftmost (western bias)
        # =====================================================================
        sorted_regions = sorted(regions, key=lambda r: r[2], reverse=True)
        primary = sorted_regions[0]

        if len(sorted_regions) == 1:
            body_left_bin, body_right_bin = primary[0], primary[1]
        else:
            secondary = sorted_regions[1]
            primary_weight = primary[2]
            secondary_weight = secondary[2]
            ratio = (secondary_weight / primary_weight) if primary_weight > 0 else 0.0

            if ratio > 0.50:
                # MERGE: treat as true multi-column body (preserve both)
                body_left_bin = min(primary[0], secondary[0])
                body_right_bin = max(primary[1], secondary[1])
                if trace_id:
                    logger.debug(
                        "[%s] Density-aware: MERGE (ratio=%.2f) -> bins %d-%d",
                        trace_id, ratio, body_left_bin, body_right_bin
                    )
            elif ratio < 0.30:
                # EXCLUDE: treat secondary as sidebar/margin (preserve primary only)
                body_left_bin, body_right_bin = primary[0], primary[1]
                if trace_id:
                    logger.debug(
                        "[%s] Density-aware: EXCLUDE secondary (ratio=%.2f) -> bins %d-%d",
                        trace_id, ratio, body_left_bin, body_right_bin
                    )

            else:
                # AMBIGUOUS: prefer PRIMARY (heavier) region
                # Western (leftmost) bias fails for right-body layouts
                # (e.g., figure captions on left, body on right)
                body_left_bin, body_right_bin = primary[0], primary[1]
                if trace_id:
                    logger.debug(
                        "[%s] Density-aware: AMBIGUOUS -> PRIMARY (ratio=%.2f, w=%.0f vs %.0f) -> bins %d-%d",
                        trace_id, ratio, primary_weight, secondary_weight, body_left_bin,
                        body_right_bin
                    )

    # =========================================================================
    # CONVERT TO COORDINATES
    # =========================================================================
    # Convert from normalized coordinates back to original page coordinates
    body_left_norm = body_left_bin * bin_width
    body_right_norm = (body_right_bin + 1) * bin_width

    inv = (1.0 / norm_scale) if norm_scale != 0 else 1.0
    body_left = body_left_norm * inv
    body_right = body_right_norm * inv

    # =========================================================================
    # APPLY MARGIN CONSTRAINTS
    # =========================================================================
    min_margin = page_width * _MARGIN_MIN_RATIO  # ~5%

    # Ensure body does not touch extreme edges
    if body_left < min_margin:
        body_left = min_margin

    if body_right > page_width - min_margin:
        body_right = page_width - min_margin

    # =========================================================================
    # VALIDATION: Ensure body_left < body_right
    # =========================================================================
    if body_left >= body_right:
        if trace_id:
            logger.warning(
                "[%s] Invalid margin boundaries (left=%.1f >= right=%.1f), using defaults",
                trace_id, body_left, body_right
            )
        body_left = page_width * _MARGIN_FALLBACK_RATIO
        body_right = page_width * (1 - _MARGIN_FALLBACK_RATIO)

    # =========================================================================
    # LOGGING
    # =========================================================================
    if trace_id:
        bin_viz = "".join(
            "█" if b >= density_threshold else ("▄" if b > 0 else "░")
            for b in bins
        )
        region_info = ", ".join(
            f"({r[0]}-{r[1]}:w={r[2]:.0f})" for r in regions
        ) if regions else "none"

        logger.debug(
            "[%s] Margin detection: left=%.0f, right=%.0f, bins=[%s], "
            "threshold=%.1f, spans=%d, regions=[%s]",
            trace_id, body_left, body_right, bin_viz,
            density_threshold, span_count, region_info
        )

    return body_left, body_right


# ✦                  ✦                  ✦                  ✦
# ✦───────────── 2 Raw Extraction & Classification ─────────────✦
# ✦                  ✦                  ✦                  ✦


# ✦────── a. Ingestion & Cleaning ──────✦

def _row_key(span_obj):
    """
    Normalize y-position into a stable visual row bucket.
    Font-scaled tolerance absorbs scan jitter and baseline drift.
    """
    bbox = span_obj.get("bbox") or [0, 0, 0, 0]
    y0 = float(bbox[1])
    fs = float(span_obj.get("size", 0) or 10.0)

    return round(y0 / max(1.0, fs * 0.25), 0)


def _flatten_to_raw_spans(
        text_page: dict,
        page_num: int,
        trace_id: str = None
) -> List[Dict]:
    """
    Flatten PyMuPDF block/line/span hierarchy into a linear list of span dicts.
    Phase 1.5:  Deterministic order stabilization (row_key → x1 → x0 → index)
    Phase 1.75: Horizontal adjacency signaling (metadata-only, lossless)
    A2-V: Vertical continuation detection within block
    """

    def _reconstruct_text_from_chars(span_dict: Dict) -> str:
        """
        Reconstruct span text from rawdict character array, inserting spaces
        at detected word boundaries based on inter-glyph gaps.

        Falls back to span["text"] if no chars array is present (dict mode compat).
        """
        chars = span_dict.get("chars")
        if not chars:
            return span_dict.get("text", "")

        fs = float(span_dict.get("size") or _RAWDICT_FALLBACK_FONT_SIZE)
        gap_threshold = fs * _RAWDICT_WORD_GAP_RATIO

        parts = []
        for i, ch in enumerate(chars):
            c = ch.get("c", "")
            if i > 0 and c != " " and parts and parts[-1] != " ":
                prev_bbox = chars[i - 1].get("bbox")
                curr_bbox = ch.get("bbox")
                if prev_bbox and curr_bbox:
                    gap = curr_bbox[0] - prev_bbox[2]
                    if gap >= gap_threshold:
                        parts.append(" ")
            parts.append(c)

        return "".join(parts)

    raw_spans: List[Dict] = []
    skipped_count = 0

    # REVISION A2-M/V: Continuation link counters
    horizontal_link_count = 0
    vertical_link_count = 0

    blocks = text_page.get("blocks", [])

    # HARDENED: Enforce top-to-bottom, left-to-right block order
    blocks = sorted(
        blocks,
        key=lambda b: (
            b.get("bbox", [0, 0, 0, 0])[1],  # y0
            b.get("bbox", [0, 0, 0, 0])[0],  # x0
        )
    )

    for block_idx, block in enumerate(blocks):
        if block.get("type") != _PYMUPDF_TEXT_BLOCK_TYPE:
            continue

        prev_span = None
        vertical_continuation_pending = None

        lines = block.get("lines")
        if not lines:
            continue

        for line_idx, line in enumerate(lines):
            # REVISION A2-V: Capture prior line tail before reset
            # Enables vertical continuation detection without cross-column risk
            carry_tail = prev_span

            # CRITICAL (retained): Horizontal micro-stitching remains line-local
            prev_span = None

            spans = line.get("spans")
            if not spans:
                continue

            # =========================================================
            # PHASE 1.5: Order Stabilization (Diagnostic-Verified)
            # PyMuPDF x0 is unreliable for spans split by inline formatting.
            # x1 (end position) is always accurate from glyph placement.
            # Sort by: y0 (row) → x1 (end) → x0 → original index
            # =========================================================
            spans_indexed = list(enumerate(spans))

            spans_indexed.sort(key=lambda s: (
                _row_key(s[1]),  # 1) visual row (stable)
                float(s[1].get("bbox", [0, 0, 0, 0])[2] or 0.0),  # 2) x1 (glyph end, reliable)
                float(s[1].get("bbox", [0, 0, 0, 0])[0] or 0.0),  # 3) x0 (tie-break)
                s[0],  # 4) original index (determinism)
            ))

            spans = [s[1] for s in spans_indexed]

            # Cache row_key per span for consistency and micro-perf
            for s in spans:
                s["_row_key"] = _row_key(s)

            # Cache reconstructed text per span (rawdict chars → text is deterministic)
            # Hoisted: avoids redundant char-array iteration across adjacency + assignment sites
            for s in spans:
                s["_text"] = _reconstruct_text_from_chars(s)

            line_text_canonical = "".join(s["_text"] for s in spans)

            # =========================================================
            # PHASE 1.75: Horizontal Adjacency Signaling (Lossless)
            # Captured separately to avoid mutating raw PyMuPDF spans
            # =========================================================
            horizontal_links = {}

            for i in range(len(spans) - 1):
                curr = spans[i]
                nxt = spans[i + 1]

                if curr["_row_key"] != nxt["_row_key"]:
                    continue

                curr_text = curr["_text"].rstrip()
                curr_text_stripped = curr_text.rstrip(" '\"')]}")
                next_text = nxt["_text"].lstrip()

                if not curr_text or not next_text:
                    continue

                if curr_text_stripped.endswith((".", "?", "!")):
                    continue

                if curr_text.endswith("-"):
                    horizontal_links[i] = ("hyphen_wrap", "a2_horizontal_hyphen")
                else:
                    horizontal_links[i] = ("space_join", "a2_horizontal_same_row")

                horizontal_link_count += 1

            # =========================================================
            # PHASE 1.76: Build reverse lookup for horizontal continuation
            # Maps successor index → predecessor's continuation info
            # =========================================================
            horizontal_links_reverse = {}
            for pred_idx, (mode, reason) in horizontal_links.items():
                succ_idx = pred_idx + 1
                horizontal_links_reverse[succ_idx] = (mode, reason)

            # =========================================================
            # PHASE 1.5B (DIAGNOSTIC): Flag suspicious bbox geometry
            # =========================================================
            if trace_id:
                for _s in spans:
                    _bbox = _s.get("bbox") or []
                    _txt = _s.get("text", "") or ""
                    if len(_bbox) >= 4 and _txt:
                        _w = float(_bbox[2]) - float(_bbox[0])
                        _fs = float(_s.get("size", 0) or 0)
                        if _fs > 0 and len(_txt) < 60 and _w > (_fs * 20):
                            logger.debug(
                                "[%s] Phase1.5B: suspicious span bbox w=%.1f fs=%.1f chars=%d x0=%.1f x1=%.1f text=%r",
                                trace_id, _w, _fs, len(_txt),
                                float(_bbox[0]), float(_bbox[2]), _txt
                            )

            for span_idx, span in enumerate(spans):
                try:
                    # PHASE 0: Lossless bbox validation (never drop text)
                    bbox_raw = span.get("bbox")
                    bbox_is_valid = bool(bbox_raw and len(bbox_raw) >= 4)

                    bbox = None
                    bbox_invalid_reason = None
                    x0, y0, x1, y1 = 0.0, 0.0, 0.0, 0.0  # Safe defaults for invalid bbox

                    # Phase 1 geometry provenance

                    if not bbox_is_valid:
                        bbox_invalid_reason = "missing_or_short_bbox"
                    else:
                        x0, y0, x1, y1 = bbox_raw[:4]
                        if x1 <= x0 or y1 <= y0:
                            bbox_invalid_reason = "degenerate_bbox"
                        else:
                            bbox = (float(x0), float(y0), float(x1), float(y1))

                    if bbox_invalid_reason:
                        skipped_count += 1  # keep metric, but DO NOT continue

                    # Compute origin/baseline using validated coordinates (or safe defaults)
                    origin = span.get("origin", (x0, y1))
                    baseline_y = float(origin[1])
                    line_y_band = round(baseline_y, 2)
                    raw_text = span["_text"]

                    # =========================================================
                    # REVISION A2-V: Vertical Continuation Detection
                    # First span of line only — links to prior line tail if safe
                    # =========================================================
                    # NOTE: Phase-1 continuation fields are SIGNALS ONLY.
                    # They MUST NOT be consumed before Phase 4/5.
                    # They MUST NOT cause sentence boundaries.

                    if span_idx == 0 and carry_tail is not None and raw_text:
                        prev_text = (carry_tail.get("raw_text") or "").rstrip()
                        curr_text = raw_text.lstrip()

                        # Guard 1: Must share native block identity
                        carry_block_id = carry_tail.get("block_id")
                        curr_block_id = block.get("number", block_idx)
                        same_block = (
                                carry_block_id is not None and
                                carry_block_id == curr_block_id
                        )

                        # ═══════════════════════════════════════════════════════════════════
                        # Guard 2: AUTHORITATIVE — Sentence boundary detection
                        # This is the definitive signal. If no terminal punctuation,
                        # the sentence is unfinished BY DEFINITION.
                        # ═══════════════════════════════════════════════════════════════════
                        prev_tail_stripped = prev_text.rstrip(" '\"')]}")
                        prev_hard_end = prev_tail_stripped.endswith((".", "?", "!"))

                        # Derived: If no terminal punctuation, sentence is mid-flight
                        sentence_is_unfinished = not prev_hard_end

                        # Guard 3: ADVISORY — Uppercase heuristic
                        # Uppercase can ONLY signal a new sentence if the previous sentence ended.
                        curr_starts_upper = len(curr_text) > 0 and curr_text[0].isupper()

                        curr_looks_new_sentence = (
                                curr_starts_upper and prev_hard_end
                        )

                        # ═══════════════════════════════════════════════════════════════════
                        # Guard 4: CONDITIONAL — Geometry sanity
                        # Strict when sentence ended; lenient when mid-sentence.
                        # PDF bbox jitter should not break legitimate continuations.
                        # ═══════════════════════════════════════════════════════════════════
                        allow_geom = False
                        prev_bbox = carry_tail.get("bbox")

                        if bbox is not None and prev_bbox is not None and len(prev_bbox) >= 4:
                            v_gap = bbox[1] - prev_bbox[3]  # curr_y0 - prev_y1
                            indent_dx = abs(bbox[0] - prev_bbox[0])  # x-offset
                            fs = float(carry_tail.get("font_size") or 10.0)

                            max_v_gap = max(2.0, fs * 1.8)
                            max_indent = max(12.0, fs * 3.5)

                            geom_within_tolerance = (v_gap <= max_v_gap) and (
                                        indent_dx <= max_indent)

                            if sentence_is_unfinished:
                                lenient_v_gap = max_v_gap * 2.0
                                lenient_indent = max_indent * 2.0
                                geom_lenient = (v_gap <= lenient_v_gap) and (
                                            indent_dx <= lenient_indent)
                                allow_geom = geom_within_tolerance or geom_lenient
                            else:
                                allow_geom = geom_within_tolerance
                        else:
                            # Geometry unavailable — do not block mid-sentence continuation
                            allow_geom = sentence_is_unfinished

                        # ═══════════════════════════════════════════════════════════════════
                        # FINAL DECISION: Establish vertical A2 edge
                        # Guard 1 (same_block) + Guard 2 (not ended) + Guard 4 (geometry)
                        # Guard 3 only blocks when sentence actually ended
                        # ═══════════════════════════════════════════════════════════════════
                        vertical_continuation_applied = False

                        if same_block and allow_geom and not prev_hard_end and not curr_looks_new_sentence:
                            carry_tail["a2_continues_to_next"] = True
                            vertical_continuation_applied = True

                            if prev_tail_stripped.endswith(("-", "\u00AD")):
                                carry_tail["a2_continuation_mode"] = "hyphen_wrap"
                                carry_tail["a2_continuation_reason"] = "a2_vertical_hyphen"
                                vertical_continuation_pending = ("hyphen_wrap",
                                                                 "a2_vertical_hyphen")
                            else:
                                carry_tail["a2_continuation_mode"] = "space_join"
                                carry_tail["a2_continuation_reason"] = "a2_vertical_same_block"
                                vertical_continuation_pending = ("space_join",
                                                                 "a2_vertical_same_block")

                            vertical_link_count += 1
                        else:
                            vertical_continuation_pending = None

                        # ═══════════════════════════════════════════════════════════════
                        # DIAGNOSTIC: Record why vertical continuation was not applied
                        # (No behavioral change; enables future audit/debugging)
                        # ═══════════════════════════════════════════════════════════════
                        if not vertical_continuation_applied and sentence_is_unfinished:
                            # Sentence was mid-flight but continuation blocked — worth tracking
                            carry_tail["_a2_vertical_attempted"] = True
                            carry_tail["_a2_vertical_blocked_by"] = (
                                "block_mismatch" if not same_block else
                                "geometry" if not allow_geom else
                                "unknown"
                            )

                        # CRITICAL: Clear carry_tail after processing to prevent unbounded chaining
                        # Each vertical link applies to exactly one line transition
                        carry_tail = None

                    # 2. Append new span if not merged
                    new_span_entry = {
                        "raw_text": raw_text,
                        "cleaned_text": None,
                        "bbox": bbox,
                        # Geometry provenance: True if bbox was missing/degenerate and we fell back to defaults
                        "font_size": float(span.get("size", 0)),

                        "bbox_is_valid": (bbox is not None),
                        # True when bbox was missing/degenerate

                        "bbox_invalid_reason": bbox_invalid_reason,
                        "line_text": line_text_canonical,
                        "font": span.get("font", ""),
                        "flags": span.get("flags", 0),
                        "color": span.get("color", 0),
                        "origin": origin,
                        "baseline_y": baseline_y,
                        "line_y_band": line_y_band,

                        "page_number": page_num + 1,
                        "row_key": _row_key(span),
                        "is_line_end": (span_idx == len(spans) - 1),
                        "span_count_in_line": len(spans),
                        "column_index": 0,
                        "paragraph_index": 0,
                        "is_paragraph_start": False,
                        "role": TextRole.BODY.value,
                        "char_offset": 0,
                        "block_id": block.get("number", block_idx),

                        # PHASE 0: Preserve native line identity & bbox (enables Phase 1/2/4 determinism)
                        "line_index": line_idx,
                        "line_id": f"{page_num + 1}:{block.get('number', block_idx)}:{line_idx}",
                        "line_bbox": tuple(line.get("bbox", [0, 0, 0, 0])),
                        "block_bbox": tuple(block.get("bbox", [0, 0, 0, 0])),
                        "span_index_in_line": span_idx,

                        # PHASE 0: per-span provenance (audit / debug)
                        "source_block_type": block.get("type"),
                        "source_block_number": block.get("number", block_idx),

                        # REVISION A2-M: Continuation metadata (Phase B consumes)
                        "a2_continues_to_next": span_idx in horizontal_links,
                        "a2_continuation_mode": horizontal_links.get(span_idx, (None, None))[0],
                        "a2_continuation_reason": horizontal_links.get(span_idx, (None, None))[1],
                        "a2_continues_from_previous": (
                                (span_idx == 0 and vertical_continuation_pending is not None) or
                                span_idx in horizontal_links_reverse
                        ),
                        "a2_continuation_from_mode": (
                            vertical_continuation_pending[
                                0] if span_idx == 0 and vertical_continuation_pending
                            else horizontal_links_reverse.get(span_idx, (None, None))[0]
                        ),
                        "a2_continuation_from_reason": (
                            vertical_continuation_pending[
                                1] if span_idx == 0 and vertical_continuation_pending
                            else horizontal_links_reverse.get(span_idx, (None, None))[1]
                        ),
                        "is_subscript": False,
                        "figure_index": None,
                    }

                    raw_spans.append(new_span_entry)

                    # Phase 1 A2 completeness: propagate backward signal
                    if prev_span and prev_span.get("a2_continues_to_next"):
                        new_span_entry["a2_continues_from_previous"] = True

                    prev_span = new_span_entry

                except (KeyError, TypeError, IndexError):
                    skipped_count += 1
                    # HARDENED: Fallback for malformed spans
                    raw_spans.append({
                        "raw_text": _reconstruct_text_from_chars(span),
                        "cleaned_text": None,
                        "bbox": None,
                        "flags": ["span_exception"],
                        "page_number": page_num + 1,
                        "role": "body",  # Use string literal to avoid import dependency issues
                    })
                    prev_span = None
                    continue

    if trace_id and (horizontal_link_count > 0 or vertical_link_count > 0):
        logger.info(
            "[%s] Ingestion: emitted %d horizontal + %d vertical continuation links "
            "(bidirectional a2_continues_to/from)",
            trace_id, horizontal_link_count, vertical_link_count
        )
    # =========================================================================
    # PHASE 1 INVARIANT (DEBUG ONLY)
    # Phase 1 must not introduce sentence, semantic, or exclusion authority.
    # =========================================================================
    if trace_id and __debug__:
        for span in raw_spans:
            assert "_sentence_id" not in span, "Phase 1 leaked sentence state"
            assert "_semantic_disposition" not in span, "Phase 1 leaked semantic state"
            assert "_exclusion_candidate" not in span, "Phase 1 leaked exclusion logic"
            assert "_has_semantic_authority" not in span, "Phase 1 leaked authority"
            assert not span.get("_tts_excluded"), "Phase 1 leaked exclusion state"

    # === PHASE 1 EXIT ===
    return raw_spans


def _build_lines_from_spans(
        spans: List[Dict],
        trace_id: str = None
) -> Dict[str, Dict]:
    """
    PHASE 3: Build Line abstractions from flat span list.

    Groups spans by line_id and creates canonical line representations.
    Enables line-coherent semantic operations without altering geometry.

    Returns:
        Dict mapping line_id -> Line dict with:
            - line_id, page_number, block_id, line_index
            - line_bbox, y_band
            - spans: List[Dict] ordered by span_index_in_line (then x0 fallback)
            - text_raw: str (lossless concatenation)
            - font_sizes: List[float]
            - fonts: List[str]
            - span_count: int
    """
    # PHASE 3 CONTRACT:
    # - This method groups spans into line abstractions only
    # - This method MUST NOT introduce sentence boundaries
    # - This method MUST NOT assign semantic inclusion/exclusion
    # - This method MUST NOT set or modify _has_semantic_authority

    lines: Dict[str, Dict] = {}

    for span in spans:
        line_id = span.get("line_id")
        if not line_id:
            continue

        if line_id not in lines:
            lines[line_id] = {
                "line_id": line_id,
                "page_number": span.get("page_number"),
                "block_id": span.get("block_id"),
                "line_index": span.get("line_index"),
                "line_bbox": span.get("line_bbox"),
                "y_band": span.get("line_y_band"),
                "spans": [],
                "font_sizes": set(),
                "fonts": set(),
            }

        line = lines[line_id]
        line["spans"].append(span)

        if span.get("font_size"):
            line["font_sizes"].add(span["font_size"])
        if span.get("font"):
            line["fonts"].add(span["font"])

    # Order spans within each line and build lossless text
    for line in lines.values():
        line["spans"].sort(key=lambda s: (
            s.get("span_index_in_line", 0),
            (s.get("bbox") or [0, 0, 0, 0])[0],  # x0 fallback only
            (s.get("bbox") or [0, 0, 0, 0])[1],  # y0 tie-breaker
        ))

        line["text_raw"] = "".join(s.get("raw_text", "") for s in line["spans"])

        line["span_count"] = len(line["spans"])
        line["font_sizes"] = sorted(line["font_sizes"])
        line["fonts"] = sorted(line["fonts"])

    # =========================================================================
    # PHASE 3 DIAGNOSTIC: PyMuPDF duplicate detection
    # =========================================================================
    if trace_id and __debug__:
        for lid, line_data in lines.items():
            line_spans = line_data.get("spans", [])
            if len(line_spans) > 1:
                texts = [
                    ((s.get("cleaned_text") or s.get("raw_text") or "").strip(),
                     s.get("layout_stream", "unknown"))
                    for s in line_spans
                ]
                for i, (t1, s1) in enumerate(texts):
                    for j, (t2, s2) in enumerate(texts):
                        if i < j and t1 == t2 and s1 != s2:
                            logger.warning(
                                "[%s] Duplicate span detected: line_id=%s, text='%s...', streams=%s/%s",
                                trace_id, lid, t1[:30], s1, s2
                            )
    # =========================================================================
    # PHASE 3: Tag duplicate rank/count for spans with overlapping text
    # =========================================================================
    # NOTE:
    # - _line_duplicate_rank == 0 indicates the most complete span for this line
    # - Higher ranks indicate partial or mirrored duplicates (e.g., PyMuPDF artifacts)
    # - These tags MUST NOT be used to drop spans; they are advisory only

    for lid, line_data in lines.items():
        line_spans = line_data.get("spans", [])
        if len(line_spans) > 1:
            texts = [
                (s.get("cleaned_text") or s.get("raw_text") or "")
                for s in line_spans
            ]

            # Only tag duplicates if overlap exists (substring or equality)
            has_overlap = any(
                i != j and (t1 in t2 or t2 in t1)
                for i, t1 in enumerate(texts)
                for j, t2 in enumerate(texts)
            )

            if not has_overlap:
                continue

            # Sort by text length descending (longest = most complete)
            sorted_spans = sorted(
                line_spans,
                key=lambda s: len(s.get("cleaned_text") or s.get("raw_text", "")),
                reverse=True
            )
            for rank, sp in enumerate(sorted_spans):
                sp["_line_duplicate_rank"] = rank
                sp["_line_duplicate_count"] = len(line_spans)
    if trace_id and __debug__:
        for line in lines.values():
            for s in line.get("spans", []):
                assert "_sentence_id" not in s, "Phase 3 leaked sentence state"
                assert "_semantic_disposition" not in s, "Phase 3 leaked semantic state"
                assert "_has_semantic_authority" not in s, "Phase 3 leaked authority"

    if trace_id:
        logger.debug(
            "[%s] Built %d line abstractions from %d spans",
            trace_id, len(lines), len(spans)
        )

    return lines


# ===========================================================================
# STAGE 1 SOFT CLASSIFICATION (Schema v2.0 - Lossless)
# ===========================================================================

def _soft_classify_spans(
        spans: List[Dict],
        regions: Dict,
        *,
        page_width: float,
        page_height: float,
        trace_id: str = None,
) -> List[Dict]:
    """
    STAGE 1 SOFT CLASSIFICATION: Annotate spans without exclusion.

    This function is the SOLE AUTHORITY for Stage 1 classification.
    It replaces _filter_spans for the new lossless pipeline.

    Contract:
        - ALL input spans are returned (no removal)
        - Spans are annotated with soft classification fields
        - Final exclusion decisions deferred to Stage 2 semantic window

    Args:
        spans: Raw spans from extraction (will be deep-copied if needed).
        regions: Page regions dict with figures, tables, header_bands, etc.
        page_width: Page width for ratio calculations.
        page_height: Page height for ratio calculations.
        trace_id: Optional trace ID for logging.

    Returns:
        List of all spans with soft classification fields attached.
    """
    if not spans:
        return []

    # --- SETUP: Extract region data ---
    header_bands: set = set(regions.get("header_bands", []))
    footer_bands: set = set(regions.get("footer_bands", []))
    figure_tuples: List[BboxTuple] = regions.get("figure_tuples", regions.get("figures", []))
    table_regions: List[Dict] = regions.get("tables", [])

    # --- CONFIGURATION ---
    effective_page_height = page_height if page_height and page_height > 0 else _FILTER_DEFAULT_PAGE_HEIGHT
    effective_page_width = page_width if page_width and page_width > 0 else 612.0
    y_band_precision = _REGION_Y_BAND_ROUNDING
    header_artifact_zone_y = effective_page_height * _FILTER_HEADER_ARTIFACT_ZONE_RATIO
    drop_cap_font_threshold = _PARA_DEFAULT_LINE_HEIGHT * 1.5

    classified: List[Dict] = []
    for span_idx, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
        # RONC authority override: protected spans may not be exclusion candidates
        if span.get("_ronc_contract", {}).get("protection", {}).get("must_include"):
            classified.append(span)
            continue

        # Initialize soft classification fields
        _init_stage1_soft_fields(span)

        # --- VALIDATION: bbox ---
        bbox = span.get("bbox")
        if bbox is None or not isinstance(bbox, tuple) or len(bbox) < 4:
            _flag_candidate(span, _REASON_INVALID_BBOX, requires_review=True)
            classified.append(span)
            continue

        span_y = bbox[1]
        text = (span.get("cleaned_text") or span.get("raw_text") or "").strip()

        if not text:
            _flag_candidate(span, _REASON_EMPTY, requires_review=True)
            classified.append(span)
            continue

        # --- Y-BAND calculation ---
        y_band = round(span_y / y_band_precision) * y_band_precision

        # === GREEK WHITELIST BYPASS (STRICT) ===
        # Only protect actual Greek symbols, not ASCII letters that happen to be
        # in VALID_SINGLE_CHARS (those will be handled by normal classification)
        text_lower = text.lower()

        is_greek_symbol = (
                _contains_greek(text) or
                text_lower in GREEK_WORD_FORMS
        )

        if is_greek_symbol:
            if y_band not in header_bands and y_band not in footer_bands:
                span["_whitelist_protected"] = "greek_symbol"
                classified.append(span)
                continue

        # === HEADER/FOOTER BAND FLAGS ===
        in_header_band = y_band in header_bands
        in_footer_band = y_band in footer_bands

        if in_header_band:
            # ─────────────────────────────────────────────────────────
            # Band continuation: A span starting with sentence-boundary
            # punctuation AND having A2 continuation from the previous
            # span is body text crossing a band coordinate, not a band
            # artifact. The A2 requirement prevents false rescue of
            # actual header/footer content (which never has A2 chains
            # from body text on the same line).
            # ─────────────────────────────────────────────────────────
            is_band_continuation = text and (
                    text[0].islower()
                    or (text[0] in '.,;:)]}?!' and span.get("a2_continues_from_previous", False))
            )
            if is_band_continuation:
                span["_exclusion_protected"] = True
                span.setdefault("_exclusion_protection_reasons", []).append(
                    "lowercase_continuation")
            else:
                _flag_candidate(span, _REASON_HEADER_BAND, requires_review=False)
        elif in_footer_band:
            is_band_continuation = text and (
                    text[0].islower()
                    or (text[0] in '.,;:)]}?!' and span.get("a2_continues_from_previous", False))
            )
            if is_band_continuation:
                span["_exclusion_protected"] = True
                span.setdefault("_exclusion_protection_reasons", []).append(
                    "lowercase_continuation")
            else:
                _flag_candidate(span, _REASON_FOOTER_BAND, requires_review=False)

        # === PUBLISHER / CITATION METADATA FLAG (SEMANTIC BACKSTOP) ===
        # Flags spans that look like publisher metadata even if they do not
        # repeat across pages or fall outside footer bands.
        #
        # Contract:
        #   - NEVER excludes directly
        #   - ONLY flags for downstream semantic confirmation
        #   - Geometry-agnostic (handles first-page-only metadata)

        text_lower = text.lower()

        matches_publisher_metadata = False

        # Core publisher / license markers
        if any(token in text_lower for token in (
                "©",
                "copyright",
                "the author(s)",
                "exclusive license",
                "springer",
                "springer nature",
                "elsevier",
                "ieee",
                "acm",
                "taylor & francis",
        )):
            matches_publisher_metadata = True

        # Citation / proceedings markers
        elif any(token in text_lower for token in (
                "pp.",
                "vol.",
                "eds.",
                "doi.org",
                "doi:",
                "10.1007/",
                "10.1109/",
                "10.1145/",
                "lnns",
                "lncs",
                "proceedings",
                "arxiv:",
                "arxiv.org",
                "issn",
                "isbn",
        )):
            matches_publisher_metadata = True

        if matches_publisher_metadata:
            # Lowercase continuation protection still applies
            if text and text[0].islower():
                span["_exclusion_protected"] = True
                span.setdefault("_exclusion_protection_reasons", []).append(
                    "publisher_metadata_continuation")
            else:
                _flag_candidate(
                    span,
                    _REASON_PUBLISHER_METADATA,
                    requires_review=True
                )

        # === DOCUMENT METADATA FLAG (v10.0) ===
        # Flags author blocks, affiliations, and email addresses on
        # page 1 that should not narrate. Follows the publisher_metadata
        # pattern: flag only, no exclusion. Downstream flag translation,
        # RONC veto (Point 3), and narration gate (Point 6) handle
        # actual exclusion decisions.
        #
        # Scoping constraints (all required to enter detection):
        #   1. Page 1 only (page_number == 1)
        #   2. Above 35% of page height (title/author region)
        #
        # Detection paths (either triggers):
        #   Path A — Standalone email line (always metadata on page 1)
        #   Path B — Author name list shape (short, high cap ratio,
        #            contains list indicators: commas, "and", or "@")
        page_number = span.get("page_number", 0)
        if page_number == 1 and span_y < effective_page_height * 0.35:
            text_words = text.split()
            word_count = len(text_words)

            # Path A: Email lines are always metadata on page 1
            # (e.g., "100428305@alumnos.uc3m.es")
            is_email_line = "@" in text and word_count <= 5

            # Path B: Author name list shape
            # High capitalization ratio + list indicators + short length
            # Guards against body text: word_count <= 12 prevents
            # matching sentences; cap ratio >= 0.5 prevents matching
            # lowercase prose that mentions names in passing.
            capitalized_ratio = (
                    sum(1 for w in text_words if w[:1].isupper())
                    / max(word_count, 1)
            )
            has_list_indicators = (
                    "," in text
                    or " and " in f" {text_lower} "
                    or "@" in text
            )
            is_author_block = (
                    word_count <= 12
                    and capitalized_ratio >= 0.5
                    and has_list_indicators
            )

            if is_email_line or is_author_block:
                # Lowercase continuation protection still applies
                # (matches publisher_metadata pattern)
                if text and text[0].islower():
                    span["_exclusion_protected"] = True
                    span.setdefault("_exclusion_protection_reasons", []).append(
                        "document_metadata_continuation")
                else:
                    _flag_candidate(
                        span,
                        _REASON_DOCUMENT_METADATA,
                        requires_review=True
                    )

        # === HEADER ARTIFACT ZONE ===
        if span_y < header_artifact_zone_y:
            span.setdefault("processing_tags", []).append("suspected_header_artifact_zone")
            text_clean = text.strip()

            if len(text_clean) == 1 and text_clean.isalpha() and text_clean.isupper():
                is_large_font = span.get("font_size", 0) >= drop_cap_font_threshold
                if is_large_font:
                    # Drop cap protection
                    span["_exclusion_protected"] = True
                    span.setdefault("_exclusion_protection_reasons", []).append("drop_cap")
                else:
                    _flag_candidate(span, _REASON_HEADER_ARTIFACT, requires_review=False)

        # === SPATIAL CONTEXT (figures/tables) ===
        ctx = _span_visual_context(
            span,
            figure_tuples=figure_tuples,
            table_regions=table_regions,
            page_width=effective_page_width,
            page_height=effective_page_height,
        )
        span["_spatial_context"] = ctx

        # === VISUAL OVERLAP FLAG ===
        if ctx["inside_figure"]:
            try:
                is_caption = _is_caption_candidate(span, figure_tuples)
            except Exception:
                is_caption = False

            if not is_caption:
                # P0 AMENDMENT 1: visual_overlap MUST require semantic review
                _flag_candidate(span, _REASON_VISUAL_OVERLAP, requires_review=True)

        # === DIAGRAM LABEL SIGNAL ===
        if ENABLE_DIAGRAM_LABEL_FILTER:
            label_sig = _diagram_label_signal(
                span,
                figure_tuples,
                spatial_context=ctx,
                baseline_font_size=12.0,
                trace_id=trace_id,
            )
            if label_sig["is_label"]:
                _flag_candidate(span, _REASON_DIAGRAM_LABEL, requires_review=False)
                span.setdefault("_signals", {})["diagram_label"] = label_sig

        # === BARE CAPTION FLAG ===
        for pattern in _COMPILED_CAPTION_PATTERNS:
            if pattern.match(text):
                if len(text) < _FILTER_BARE_CAPTION_MAX_CHARS:
                    if not _CAPTION_PROSE_CONTINUATION_PATTERN.search(text):
                        _flag_candidate(span, _REASON_BARE_CAPTION, requires_review=True)
                        break

        # === NOISE FLAGS (with protections) ===
        text_stripped = text.strip()

        # Single char noise
        if len(text_stripped) == 1 and not text_stripped.isdigit():
            if text_stripped not in VALID_SINGLE_CHARS:
                # Check for drop cap protection (large font single uppercase)
                if text_stripped.isalpha() and text_stripped.isupper():
                    is_large_font = span.get("font_size", 0) >= drop_cap_font_threshold
                    if is_large_font:
                        span["_exclusion_protected"] = True
                        span.setdefault("_exclusion_protection_reasons", []).append("drop_cap")
                    else:
                        _flag_candidate(span, _REASON_NOISE_SINGLE_CHAR, requires_review=False)
                else:
                    _flag_candidate(span, _REASON_NOISE_SINGLE_CHAR, requires_review=False)

        # Digit-only noise (with structural context / year / significant digit protection)
        # ─────────────────────────────────────────────────────────────
        # INVARIANT: Noise classification must respect structural units.
        # A span that shares its line with other spans is structurally
        # contextual and must not be classified as noise based on content
        # alone. Noise is defined by isolation, not by content.
        # ─────────────────────────────────────────────────────────────
        if text.isdigit():
            is_line_contextual = span.get("span_count_in_line", 1) > 1
            is_year = len(text) == 4 and text.startswith(("19", "20"))
            is_significant = len(text) >= 3
            if is_line_contextual:
                span["_exclusion_protected"] = True
                span.setdefault("_exclusion_protection_reasons", []).append(
                    "line_context_digit")
            elif is_year:
                span["_exclusion_protected"] = True
                span.setdefault("_exclusion_protection_reasons", []).append("year")
            elif is_significant:
                span["_exclusion_protected"] = True
                span.setdefault("_exclusion_protection_reasons", []).append(
                    "significant_digits")
            else:
                _flag_candidate(span, _REASON_NOISE_DIGIT_ONLY, requires_review=False)

        # Punctuation noise
        # Guard: preserve inline punctuation between alphanumeric spans
        if text in NOISE_PUNCTUATION:
            is_inline_punct = False
            if text in {",", ";", ":", "."}:
                prev_span = spans[span_idx - 1] if span_idx > 0 else None
                next_span = spans[span_idx + 1] if (span_idx + 1) < len(spans) else None
                if isinstance(prev_span, dict) and isinstance(next_span, dict):
                    prev_text = (prev_span.get("cleaned_text") or prev_span.get(
                        "raw_text") or "").strip()
                    next_text = (next_span.get("cleaned_text") or next_span.get(
                        "raw_text") or "").strip()
                    if prev_text and next_text and prev_text[-1].isalnum() and next_text[
                        0].isalnum():
                        is_inline_punct = True
            if is_inline_punct:
                span["_exclusion_protected"] = True
                span.setdefault("_exclusion_protection_reasons", []).append(
                    "inline_punctuation")
            else:
                _flag_candidate(span, _REASON_NOISE_PUNCTUATION, requires_review=False)

        # Fragment noise
        if len(text) <= _FILTER_SHORT_FRAGMENT_THRESHOLD and not span.get("filter_protected"):
            text_norm = text.strip(".,;:")
            is_letter = text_norm.isalpha()
            is_protected_word = text_norm in PROTECTED_SHORT_WORDS
            is_connector = text_norm.lower() in {"and", "or", "to", "of", "but", "nor", "for", "so",
                                                 "yet"}

            if not (is_letter or is_protected_word or is_connector):
                _flag_candidate(span, _REASON_NOISE_FRAGMENT, requires_review=False)

        # ─────────────────────────────────────────────────────────────
        # FM1: A2 CHAIN INLINE NOISE PROTECTION
        # Protect noise-flagged spans ONLY when:
        #  - a2 chain exists
        #  - AND a body_col span exists on the SAME line_id
        # span_index_in_line is line-local; do NOT compare across lines.
        # ─────────────────────────────────────────────────────────────
        if (span.get("_exclusion_candidate")
                and not span.get("_exclusion_protected")):
            reasons = set(span.get("_exclusion_candidate_reasons") or [])
            if (reasons & _A2_NOISE_REASONS
                    and (span.get("a2_continues_from_previous")
                         or span.get("a2_continues_to_next"))):
                lid = span.get("line_id")
                if lid and any(
                        s is not span
                        and s.get("line_id") == lid
                        and str(s.get("layout_stream", "")).startswith("body_col")
                        for s in spans
                ):
                    span["_exclusion_protected"] = True
                    span.setdefault("_exclusion_protection_reasons", []).append(
                        "a2_inline_continuation")

        classified.append(span)

    if trace_id:
        candidate_count = sum(1 for s in classified if s.get("_exclusion_candidate"))
        review_count = sum(1 for s in classified if s.get("_requires_semantic_review"))
        protected_count = sum(1 for s in classified if s.get("_exclusion_protected"))
        logger.debug(
            "[%s] Soft classification: %d spans, %d candidates, %d require review, %d protected",
            trace_id, len(classified), candidate_count, review_count, protected_count
        )

    return classified


def _filter_spans(
        spans: List[Dict],
        regions: Dict,
        page_height: float = None,
        trace_id: str = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split spans into valid vs excluded with standardized thresholds.

    MODIFIED v3.3:
        1. Greek/scientific symbol whitelist bypasses all filters.
        2. Standardized Y-band rounding matches Global Detector.
        3. FIX: Corrected _FILTER_SHORT_TEXT_THRESHOLD → _FILTER_SHORT_TEXT_LENGTH.
        4. FIX: Unified large-font protection across Filter 2 and Filter 5:
           - Font threshold: _PARA_DEFAULT_LINE_HEIGHT * 1.5 (18pt for 12pt body)
           - Filter 2: Large font added to is_protected (prevents artifact deletion)
           - Filter 5: Large font OR position qualifies as Drop Cap
        5. FIX: Filter 5 excludes digits for proper segregation to Filter 6.

    Filter categories:
        1. Header/footer bands (from global detection)
        2. Header zone artifacts (orphaned fragments at top of page)
        3. Diagram labels (inside figures)
        4. Bare captions (caption patterns without prose continuation)
        5. Noise — Single characters (excludes digits)
        6. Noise — Digit-only spans
        7. Noise — Isolated punctuation
        8. Noise — Very short non-word fragments

    Args:
        spans: List of span dictionaries to filter.
        regions: Dictionary containing header_bands, footer_bands, figure_tuples.
        page_height: Page height for adaptive threshold calculations.
        trace_id: Optional trace ID for logging.

    Returns:
        Tuple of (valid_spans, excluded_spans).

    Mutates:
        Excluded spans receive 'exclusion_reason' key.
    """

    if not spans:
        return [], []

    # HARD STOP: legacy filter must not run after RONC v2
    if any("_ronc_contract" in s for s in spans):
        raise RuntimeError("filter_spans must not run after RONC v2 semantic window")

    valid: List[Dict] = []
    excluded: List[Dict] = []

    # =========================================================================
    # SETUP: Extract region data
    # =========================================================================
    header_bands: set = set(regions.get("header_bands", []))
    footer_bands: set = set(regions.get("footer_bands", []))
    figure_tuples: List[BboxTuple] = regions.get("figure_tuples", [])

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    effective_page_height = (
        page_height if page_height and page_height > 0
        else _FILTER_DEFAULT_PAGE_HEIGHT
    )

    # Must match _detect_page_regions and _compute_global_header_footer_bands
    y_band_precision = _REGION_Y_BAND_ROUNDING

    # Header artifact zone (top 7% of page)
    header_artifact_zone_y = effective_page_height * _FILTER_HEADER_ARTIFACT_ZONE_RATIO

    # Drop Cap thresholds (pre-calculated, loop-invariant)
    drop_cap_font_threshold = _PARA_DEFAULT_LINE_HEIGHT * 1.5

    # =========================================================================
    # MAIN FILTER LOOP
    # =========================================================================
    for span in spans:
        if not isinstance(span, dict):
            continue

        # ---------------------------------------------------------------------
        # VALIDATION: Extract bbox (tuple format)
        # ---------------------------------------------------------------------
        bbox = span.get("bbox")
        if bbox is None:
            span["exclusion_reason"] = _REASON_INVALID_BBOX
            excluded.append(span)
            continue

        # Bbox is tuple: (x0, y0, x1, y1)
        if not isinstance(bbox, tuple) or len(bbox) < 4:
            span["exclusion_reason"] = _REASON_INVALID_BBOX
            excluded.append(span)
            continue

        span_y = bbox[1]  # y0

        # Get text
        text = (span.get("cleaned_text") or span.get("raw_text") or "").strip()
        text_norm = text.strip(".,;:")

        if not text:
            span["exclusion_reason"] = _REASON_EMPTY
            excluded.append(span)
            continue

        # Standardized Y-band rounding
        y_band = round(span_y / y_band_precision) * y_band_precision

        # =====================================================================
        # WHITELIST CHECK — Greek/scientific symbols bypass ALL filters
        # Must be checked BEFORE any exclusion logic
        # STRICT: Only actual Greek characters, not ASCII letters
        # =====================================================================
        text_lower = text.lower()

        is_greek_symbol = (
                _contains_greek(text) or
                text_lower in GREEK_WORD_FORMS
        )

        if is_greek_symbol:
            if y_band not in header_bands and y_band not in footer_bands:
                valid.append(span)
                continue

        keep = True
        reason = ""

        # ---------------------------------------------------------------------
        # FILTER 1: Header/Footer Bands (Global Detection)
        # ---------------------------------------------------------------------
        if y_band in header_bands:
            # HARDENED: Preserve sentence continuations (lowercase start)
            if text and text[0].islower():
                keep = True
            else:
                text_stripped = text.strip()
                is_drop_cap = (
                        len(text_stripped) == 1 and
                        text_stripped.isalpha() and
                        text_stripped.isupper()
                )

                if not is_drop_cap:
                    keep = False
                    reason = _REASON_HEADER_BAND

        elif y_band in footer_bands:
            # HARDENED: Preserve sentence continuations (lowercase start)
            if text and text[0].islower():
                keep = True
            else:
                text_stripped = text.strip()
                is_drop_cap = (
                        len(text_stripped) == 1 and
                        text_stripped.isalpha() and
                        text_stripped.isupper()
                )

                if not is_drop_cap:
                    keep = False
                    reason = _REASON_FOOTER_BAND


        # ---------------------------------------------------------------------
        # FILTER 2: Header Zone Artifacts
        # FIX v3.3: Corrected threshold + unified large-font protection
        # ---------------------------------------------------------------------
        elif span_y < header_artifact_zone_y:
            # HARDENED: No deletion here. Mark as suspected artifact only.
            # IMPORTANT: Do NOT touch PyMuPDF font flags (int).
            span.setdefault("processing_tags", []).append(
                "suspected_header_artifact_zone"
            )

            # NOTE:
            # 'suspected_header_artifact_zone' is a NON-DESTRUCTIVE diagnostic flag.
            # It must NEVER be used as a sole deletion criterion.
            # It may only be considered in conjunction with:
            #   - global header/footer band membership, OR
            #   - high text similarity to confirmed header samples.

        # ---------------------------------------------------------------------
        # EARLY GUARD: Header-Zone Single-Letter Kill Switch
        # ---------------------------------------------------------------------
        if keep:
            text_clean = text.strip()

            # Single uppercase letters in the header artifact zone
            # are ONLY valid if they are Drop Caps (large font).
            if (
                    len(text_clean) == 1 and
                    text_clean.isalpha() and
                    text_clean.isupper() and
                    span_y < header_artifact_zone_y
            ):
                is_large_font = span.get("font_size", 0) >= drop_cap_font_threshold

                if not is_large_font:
                    keep = False
                    reason = _REASON_HEADER_ARTIFACT

        # ---------------------------------------------------------------------
        # FILTER 3a: Geometric Visual Region Guard (Figures ONLY)
        # Caption Safe-Harbor + Table Sanctuary enforced
        # ---------------------------------------------------------------------
        if figure_tuples:
            # Pure geometric intersection with FIGURES ONLY
            if _span_inside_visual_region(span, figure_tuples=figure_tuples):
                # Caption Safe-Harbor: preserve potential captions
                try:
                    is_caption_candidate = _is_caption_candidate(span, figure_tuples)
                except Exception:
                    is_caption_candidate = False

                if not is_caption_candidate:
                    keep = False
                    reason = _REASON_VISUAL_OVERLAP

        # ---------------------------------------------------------------------
        # FILTER 3: Diagram Labels
        # ---------------------------------------------------------------------
        # HARDENED: Run label check even if figures exist (catch labels outside the box)
        if keep and ENABLE_DIAGRAM_LABEL_FILTER:
            if _is_diagram_label(span, figure_tuples):
                keep = False
                reason = _REASON_DIAGRAM_LABEL

        # ---------------------------------------------------------------------
        # FILTER 4: Bare Captions
        # ---------------------------------------------------------------------
        if keep:
            for pattern in _COMPILED_CAPTION_PATTERNS:
                if pattern.match(text):
                    if len(text) < _FILTER_BARE_CAPTION_MAX_CHARS:
                        if not _CAPTION_PROSE_CONTINUATION_PATTERN.search(text):
                            keep = False
                            reason = _REASON_BARE_CAPTION
                            break

        # ---------------------------------------------------------------------
        # FILTER 5: Noise — Single Characters (excludes digits → Filter 6)
        # FIX v3.3: Expanded Drop Cap protection (position OR font size)
        # ---------------------------------------------------------------------
        text_clean = text.strip()
        if keep and len(text_clean) == 1 and not text_clean.isdigit():
            if text_clean not in VALID_SINGLE_CHARS:
                is_uppercase_letter = text_clean.isalpha() and text_clean.isupper()

                if is_uppercase_letter:
                    # CONTEXTUAL PROTECTION (Lead-approved):
                    # Reject header-zone glyph junk unless it is a Drop Cap
                    is_in_artifact_zone = span_y < header_artifact_zone_y
                    is_large_font = span.get("font_size", 0) >= drop_cap_font_threshold

                    if is_in_artifact_zone and not is_large_font:
                        keep = False
                        reason = _REASON_HEADER_ARTIFACT
                    else:
                        # Preserve valid body single-letter words ("I", "A")
                        # and legitimate Drop Caps
                        keep = True
                else:
                    keep = False
                    reason = _REASON_NOISE_SINGLE_CHAR

        # ---------------------------------------------------------------------
        # FILTER 6: Noise — Digit-Only Spans
        # ---------------------------------------------------------------------
        if keep and text.isdigit():
            is_year = (
                    len(text) == _FILTER_YEAR_LENGTH and
                    text.startswith(_FILTER_YEAR_PREFIXES)
            )
            is_significant = len(text) >= _FILTER_SIGNIFICANT_DIGIT_LENGTH

            if not (is_year or is_significant):
                keep = False
                reason = _REASON_NOISE_DIGIT_ONLY

        # ---------------------------------------------------------------------
        # FILTER 7: Noise — Isolated Punctuation
        # ---------------------------------------------------------------------
        if keep and text in NOISE_PUNCTUATION:
            keep = False
            reason = _REASON_NOISE_PUNCTUATION

        # ---------------------------------------------------------------------
        # FILTER 8: Noise — Very Short Non-Word Fragments
        # ---------------------------------------------------------------------
        if keep and len(text) <= _FILTER_SHORT_FRAGMENT_THRESHOLD and not span.get(
                "filter_protected"):

            is_letter = text_norm.isalpha()
            is_protected = text_norm in PROTECTED_SHORT_WORDS

            # Preserve linguistic connectors even with punctuation
            is_connector = text_norm.lower() in {
                "and", "or", "to", "of", "but", "nor", "for", "so", "yet"
            }

            if not (is_letter or is_protected or is_connector):
                keep = False
                reason = _REASON_NOISE_FRAGMENT

        # ---------------------------------------------------------------------
        # ASSIGNMENT
        # ---------------------------------------------------------------------
        if keep:
            valid.append(span)
        else:
            span["exclusion_reason"] = reason
            excluded.append(span)
            if trace_id:
                logger.debug(
                    "[%s] Filtered span: '%s' (reason=%s, y=%.1f)",
                    trace_id, text[:30], reason, span_y
                )

    # =========================================================================
    # LOGGING: Summary statistics
    # =========================================================================
    if trace_id:
        reason_counts: Dict[str, int] = {}
        for span in excluded:
            r = span.get("exclusion_reason", "unknown")
            reason_counts[r] = reason_counts.get(r, 0) + 1

        logger.debug(
            "[%s] Filter results: %d valid, %d excluded — %s",
            trace_id, len(valid), len(excluded), reason_counts
        )

    return valid, excluded


def _clean_spans(spans: List[Dict], trace_id: str = None) -> None:
    """
    In-place structural text normalization.

    ARCHITECTURAL GUARDRAIL:
    This method is the SOLE owner of structural normalization
    (Unicode normalization, control/zero-width character removal,
    and whitespace normalization).
    It must NOT perform:
      - Semantic deletion (e.g. removing "noise" words)
      - Sentence-level repair or healing
      - Pronunciation or TTS-specific adjustments
    Operations performed:
        1. Unicode NFKC normalization
        2. Control / zero-width character removal
        3. Whitespace normalization (collapsing to single spaces)

    An empty cleaned_text ("") indicates a STRUCTURAL artifact,
    not semantic deletion. Downstream processes MUST NOT treat it as noise.
    """
    if not spans:
        return

    cleaned_count = 0
    erased_count = 0

    for span in spans:
        if not isinstance(span, dict):
            continue

        original_text = span.get("raw_text", "")
        text = original_text

        if not isinstance(text, str):
            span["cleaned_text"] = ""
            continue

        # original_text is the raw span content, used only for comparison/logging

        if not text:
            span["cleaned_text"] = ""
            continue

        # =====================================================================
        # STEP 1: NFKC Normalization
        # =====================================================================
        text = unicodedata.normalize("NFKC", text)

        # =====================================================================
        # STEP 2: Remove Control Characters
        # =====================================================================
        text = _CLEAN_CONTROL_CHAR_PATTERN.sub("", text)

        # =====================================================================
        # STEP 3: Remove Zero-Width Characters
        # =====================================================================
        for char in _CLEAN_REMOVE_CHARS:
            text = text.replace(char, "")

        # =====================================================================
        # STEP 4: Normalize Space Characters
        # =====================================================================
        for char in _CLEAN_SPACE_CHARS:
            text = text.replace(char, " ")

        # =====================================================================
        # STEP 5: Normalize Quotes/Dashes
        # =====================================================================
        text = text.translate(_CLEAN_TRANS_TABLE)

        # =====================================================================
        # STEP 6: Whitespace Normalization (Final)
        # NOTE: .strip() removes leading indentation and trailing whitespace.
        # =====================================================================
        text = _WHITESPACE_PATTERN.sub(" ", text).strip()

        # =====================================================================
        # STORE RESULT
        # =====================================================================
        if not text and original_text.strip():
            # GUARDRAIL: Never erase meaningful content at this stage.
            # If structural cleaning would erase all text, revert to the
            # minimally normalized original (strip-only) instead.
            span["cleaned_text"] = original_text.strip()
            erased_count += 1
            if trace_id:
                logger.warning(
                    "[%s] Cleaner attempted full erasure; reverting to original text: '%s'",
                    trace_id, original_text
                )
        else:
            span["cleaned_text"] = text
            if text != original_text:
                cleaned_count += 1

    if trace_id and (cleaned_count > 0 or erased_count > 0):
        logger.debug(
            "[%s] Cleaned %d/%d spans (%d reverted to original)",
            trace_id, cleaned_count, len(spans), erased_count
        )


# ✦────── b. Text Heuristics ──────✦

def _is_protected_acronym(text: str) -> bool:
    """
    Check if text is a protected acronym that should not be filtered.

    Protects:
        - Known scientific/technical abbreviations (from PROTECTED_SHORT_WORDS)
        - All-caps 2-4 letter words (likely acronyms: DNA, NASA, IEEE)
        - Mixed case technical terms (pH, mRNA, kDa)

    Args:
        text: Text to check.

    Returns:
        True if text should be protected from filtering.
    """
    if not text:
        return False

    cleaned = text.strip().rstrip(_ACRONYM_STRIP_CHARS)

    # Already in protected set
    if cleaned in PROTECTED_SHORT_WORDS:
        return True

    # All-caps within acronym length range = likely acronym
    if (
            cleaned.isupper() and
            _ACRONYM_MIN_LENGTH <= len(cleaned) <= _ACRONYM_MAX_LENGTH and
            cleaned.isalpha()
    ):
        return True

    # Mixed case technical terms (pH, mRNA, kDa)
    if (
            len(cleaned) <= _ACRONYM_MIXED_CASE_MAX_LENGTH and
            any(c.isupper() for c in cleaned) and
            any(c.islower() for c in cleaned)
    ):
        return True

    return False


def _is_orphan_fragment(text: str) -> bool:
    """
    Detect orphaned text fragments that are likely OCR/extraction artifacts.

    HARDENED v1.9 (The "Alpha" Protection):
        1. Respects VALID_SINGLE_CHARS allowlist.
        2. Protects valid single letters (including Greek α, β, etc.).
    """
    clean = text.strip()

    if not clean:
        return False

    # GUARD 1: The Allowlist (Protects 'a', 'I', '1', etc.)
    if clean in VALID_SINGLE_CHARS:
        return False

    # GUARD 2: Protect any single character that is a valid Letter
    # This specifically saves Greek 'α' which isn't in ASCII.
    if len(clean) == 1 and clean.isalpha():
        return False

    # HEURISTIC 2: Very short all-caps with no vowels
    if (
            len(clean) <= _FRAGMENT_SHORT_MAX_LENGTH and
            clean.isalpha() and
            clean.isupper()
    ):
        # HARDENED: Treat 'Y' as a vowel for short all-caps words to avoid
        # false positives like MY/BY/WHY/TRY/SHY.
        vowels = set(_FILTER_VOWELS) | {"Y"}
        if not any(c in vowels for c in clean.upper()):
            return True

    # HEURISTIC 3: Isolated lowercase starting with mid-word pattern
    if (
            len(clean) > _FRAGMENT_BROKEN_WORD_MIN_LENGTH and
            clean[0].islower() and
            clean.isalpha()
    ):
        if clean[:2] in _FRAGMENT_MIDWORD_PREFIXES:
            return True

    return False


def _detect_subscripts(spans: List[Dict], trace_id: str = None) -> None:
    """
    Detect subscript spans by comparing baseline positions.

    Strategy:
        Group spans by line band, compute median baseline per line,
        flag spans whose baseline drops below median by threshold ratio.

    Args:
        spans: List of span dictionaries with baseline_y and font_size.
        trace_id: Optional trace ID for logging.

    Mutates:
        Each span receives 'is_subscript' key (bool).
    """
    if not spans:
        return

    # Group spans by approximate line band
    line_bands: Dict[float, List[Dict]] = {}

    for span in spans:
        band = span.get("line_y_band")
        if band is not None:
            line_bands.setdefault(band, []).append(span)

    for band, band_spans in line_bands.items():
        # Collect valid baselines
        baselines = []
        if band_spans:
            # Anchor to the first span in the visual line
            anchor = band_spans[0]
            anchor_y0, anchor_y1 = anchor["bbox"][1], anchor["bbox"][3]

            for s in band_spans:
                if s.get("baseline_y") is None or "bbox" not in s:
                    continue

                # Check vertical overlap with anchor to prevent cross-line pollution
                s_y0, s_y1 = s["bbox"][1], s["bbox"][3]

                inter_y0 = max(s_y0, anchor_y0)
                inter_y1 = min(s_y1, anchor_y1)
                overlap = max(0.0, inter_y1 - inter_y0)
                shortest_h = min(s_y1 - s_y0, anchor_y1 - anchor_y0)

                # Require >50% vertical overlap to contribute to the baseline pool
                if shortest_h > 0 and (overlap / shortest_h) > 0.50:
                    baselines.append(s["baseline_y"])

        if not baselines:
            continue

        # Compute dominant baseline (Weighted by frequency to ignore outliers)
        baseline_counts = {}
        for b in baselines:
            b_rounded = round(b, 1)
            baseline_counts[b_rounded] = baseline_counts.get(b_rounded, 0) + 1

        # The most frequent baseline is the "anchor"
        dominant_rounded = max(baseline_counts, key=baseline_counts.get)
        # Find the exact value closest to this rounded anchor
        median_baseline = next(b for b in baselines if round(b, 1) == dominant_rounded)

        for span in band_spans:
            baseline = span.get("baseline_y")
            if baseline is None:
                span["is_subscript"] = False
                continue

            font_size = span.get("font_size", 0)
            if font_size <= 0:
                span["is_subscript"] = False
                continue

            # Positive delta = below the median baseline => candidate subscript
            delta = baseline - median_baseline
            threshold = font_size * _SUBSCRIPT_OFFSET_RATIO

            if delta > threshold:
                # NOTE: is_subscript is a glyph-position marker only.
                # It must NOT be interpreted as a sentence or block boundary.
                span["is_subscript"] = True
                if trace_id:
                    logger.debug(
                        "[%s] Subscript detected: '%s' (delta=%.1f, threshold=%.1f)",
                        trace_id,
                        span.get("raw_text", "")[:20],
                        delta,
                        threshold
                    )
            else:
                span["is_subscript"] = False


# ✦────── c. Structure & Segmentation ──────✦

def _format_cell(
        cell,
        row: int,
        col: int,
        is_header: bool,
        page: "fitz.Page" = None,
        table_bbox: BboxTuple = None,
        trace_id: str = None
) -> Dict:
    """
    Extract text, bbox, and semantic role from PyMuPDF table cell.
    """
    if __debug__:
        assert cell is not None, "Phase 1 guard: _format_cell must only be used during span creation"

    text: str = ""  # Default for None cell or extraction failure
    cell_bbox: Optional[BboxTuple] = None

    # =========================================================================
    # TEXT EXTRACTION
    # =========================================================================
    if cell is not None:  # Changed: only try extraction if cell exists
        try:
            if hasattr(cell, "text"):
                text = cell.text.strip() if cell.text else ""
            else:
                text = str(cell).strip() if cell else ""
        except Exception as e:
            # text remains "" from initialization
            if trace_id:
                logger.debug("[%s] Cell text extraction failed: %s", trace_id, e)

    # =========================================================================
    # PRIMARY BBOX EXTRACTION (tuple format)
    # =========================================================================
    if cell is not None:
        try:
            if hasattr(cell, "bbox") and cell.bbox is not None:
                # HARDENING: Normalize via _to_bbox_tuple to ensure validity (x1 > x0)
                cell_bbox = _to_bbox_tuple(cell.bbox)

        except Exception as e:
            cell_bbox = None
            if trace_id:
                logger.debug("[%s] Cell bbox extraction failed: %s", trace_id, e)

    # =========================================================================
    # FALLBACK BBOX LOOKUP (if primary failed)
    # =========================================================================
    if cell_bbox is None and text and page is not None:
        cell_bbox = _find_bbox_for_text(page, text, table_bbox)

    # =========================================================================
    # DETERMINE SUBROLE
    # =========================================================================
    if is_header:
        subrole = TableSubRole.HEADER.value
    elif col == 0:
        subrole = TableSubRole.STUB.value
    else:
        subrole = TableSubRole.DATA.value

    return {
        "row": row,
        "col": col,
        "is_header": is_header,
        "subrole": subrole,
        "text": text,
        "bbox": cell_bbox,
    }


def _detect_page_regions(
        page: "fitz.Page",
        raw_spans: List[Dict],
        *,
        global_median_font_size: float = None,
        trace_id: str = None
) -> Dict:
    """
    Detect figures, tables, links, and header/footer bands on a page.

    MODIFIED v3.0:
        Step 1d: Synthetic Figure Detection (Text Clouds)
        Step 4: Permissive header/footer candidate collection.

    Strategy:
        1. Collect figure candidates from images and drawings
        1d. NEW: Detect synthetic figures from label clusters
        2. Validate figures don't contain prose text (false positive filter)
        3. Extract tables with cell structure
        4. Collect hyperlinks
        5. Detect header/footer bands via Y-histogram (permissive in zones)

    Args:
        page: PyMuPDF page object.
        raw_spans: List of span dicts with tuple bboxes.
        trace_id: Optional trace ID for logging.

    Returns:
        Dictionary with keys: figures, tables, links, header_bands, footer_bands.
        All bboxes are tuples (x0, y0, x1, y1).
    """
    regions: Dict = {
        "figures": [],
        "tables": [],
        "links": [],
        "header_bands": [],
        "footer_bands": [],
    }

    page_height = page.rect.height

    # =========================================================================
    # 1. FIGURE DETECTION (Images + Drawings)
    # =========================================================================
    candidate_figures: List[BboxTuple] = []

    # 1a. Image regions
    try:
        for img in page.get_images(full=True):
            rects = page.get_image_rects(img[0])
            for r in rects:
                if not r.is_infinite and not r.is_empty:
                    expanded = r + (
                        -_REGION_FIGURE_EXPAND_PADDING,
                        -_REGION_FIGURE_EXPAND_PADDING,
                        _REGION_FIGURE_EXPAND_PADDING,
                        _REGION_FIGURE_EXPAND_PADDING
                    )
                    x0 = max(0.0, expanded.x0)
                    y0 = max(0.0, expanded.y0)
                    x1 = min(page.rect.width, expanded.x1)
                    y1 = min(page.rect.height, expanded.y1)

                    if x1 > x0 and y1 > y0:
                        candidate_figures.append((x0, y0, x1, y1))
    except Exception as e:
        if trace_id:
            logger.debug("[%s] Image detection failed: %s", trace_id, e)

    # 1b. Drawing regions
    try:
        drawings = page.get_drawings()
        draw_rects: List[BboxTuple] = []
        for d in drawings:
            rect = d.get("rect")
            if rect is None:
                continue
            if rect.width > _REGION_MIN_DRAWING_DIMENSION and \
                    rect.height > _REGION_MIN_DRAWING_DIMENSION:
                draw_rects.append((rect.x0, rect.y0, rect.x1, rect.y1))
        merged_drawings = _merge_rects(draw_rects, gap=_REGION_DRAWING_MERGE_GAP)
        candidate_figures.extend(merged_drawings)
    except Exception as e:
        if trace_id:
            logger.debug("[%s] Drawing detection failed: %s", trace_id, e)

    # =========================================================================
    # 1d. SYNTHETIC FIGURE DETECTION (Text Clouds) — NEW v3.0
    # Detects diagram regions by clustering label-like text spans.
    # Catches diagrams without vector borders (e.g., Figure 3 labels).
    # =========================================================================
    try:
        synthetic_figs = _detect_synthetic_figures(
            raw_spans, page.rect.width, page.rect.height, trace_id
        )
        candidate_figures.extend(synthetic_figs)
    except Exception as e:
        if trace_id:
            logger.debug("[%s] Synthetic figure detection failed: %s", trace_id, e)

    # =========================================================================
    # 1c. VALIDATE FIGURES (Filter False Positives with Prose)
    # =========================================================================
    for fig_rect in candidate_figures:
        prose_span_count = 0
        total_text_length = 0
        fig_x0, fig_y0, fig_x1, fig_y1 = fig_rect

        # FIX: Initialize containers for density check
        contained_y_coords = []
        contained_font_sizes = []

        for span in raw_spans:
            span_bbox = span.get("bbox")
            if span_bbox is None or len(span_bbox) < 4:
                continue
            span_x0, span_y0, span_x1, span_y1 = span_bbox

            # Intersection Logic
            inter_x0 = max(span_x0, fig_x0)
            inter_y0 = max(span_y0, fig_y0)
            inter_x1 = min(span_x1, fig_x1)
            inter_y1 = min(span_y1, fig_y1)

            if inter_x1 > inter_x0 and inter_y1 > inter_y0:
                inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                span_area = max(1e-6, (span_x1 - span_x0) * (span_y1 - span_y0))
                overlap_ratio = inter_area / span_area

                # Require meaningful containment (>50%), not a 1px graze
                if overlap_ratio >= 0.50:
                    # FIX: Collect data for density check
                    contained_y_coords.append(span_y0)
                    fs = span.get("font_size", 0)
                    if fs > 0:
                        contained_font_sizes.append(fs)

                    span_text = span.get("raw_text", "")
                    total_text_length += len(span_text)
                    word_count = len(span_text.split())

                    has_sentence_punct = any(
                        c in span_text for c in _REGION_SENTENCE_PUNCTUATION
                    )

                    if (word_count >= _REGION_PROSE_MIN_WORD_COUNT or
                            (
                                    len(span_text) > _REGION_PROSE_MIN_TEXT_LENGTH and has_sentence_punct)):
                        prose_span_count += 1

        # HARDENED v4.0: Density Check
        # Distinguish between "Text Box" (reject) and "Dense Diagram" (keep).
        should_reject = False

        # 1. Trigger rejection on raw counts (existing logic)
        if (prose_span_count >= _REGION_FIGURE_MAX_PROSE_SPANS or
                total_text_length > _REGION_FIGURE_MAX_TEXT_LENGTH):
            should_reject = True

            # 2. OVERRIDE: If it's a dense diagram, keep it!
            if len(contained_y_coords) > 5:
                # FIX: Deduplicate Ys to handle horizontal labels (prevent div-by-zero or low gap)
                unique_y = sorted(list(set(contained_y_coords)))

                if len(unique_y) > 1:
                    gaps = [unique_y[i] - unique_y[i - 1] for i in range(1, len(unique_y))]
                    avg_gap = sum(gaps) / len(gaps)

                    # FIX: Use actual font size of content
                    if contained_font_sizes:
                        contained_font_sizes.sort()
                        est_fs = contained_font_sizes[len(contained_font_sizes) // 2]
                    else:
                        est_fs = 10.0

                    # If average gap is loose (> 2.0x font size), it's a diagram.
                    if avg_gap > (est_fs * 2.0):
                        should_reject = False

        region_w = fig_rect[2] - fig_rect[0]
        region_h = fig_rect[3] - fig_rect[1]
        page_w = page.rect.width
        page_h = page.rect.height
        page_area = page_w * page_h

        # HARDENED: Background Layer Protection (with full-page-diagram escape hatch)
        # Reject near-full-page regions ONLY if they also closely touch all margins,
        # which is a strong watermark/background signature.
        if (region_w * region_h) > (page_area * 0.90):
            touches_left = fig_rect[0] <= (page_w * 0.02)
            touches_right = fig_rect[2] >= (page_w * 0.98)
            touches_top = fig_rect[1] <= (page_h * 0.02)
            touches_bottom = fig_rect[3] >= (page_h * 0.98)
            if touches_left and touches_right and touches_top and touches_bottom:
                continue

        if should_reject:
            if trace_id:
                logger.debug(
                    "[%s] Rejecting false-positive figure at (%.0f,%.0f): "
                    "%d prose spans, %d chars",
                    trace_id, fig_x0, fig_y0, prose_span_count, total_text_length
                )
        else:
            if fig_rect not in regions["figures"]:
                regions["figures"].append(fig_rect)

    # =========================================================================
    # 2. TABLE DETECTION
    # =========================================================================
    try:
        tables = page.find_tables()
        for tab in tables:
            table_bbox: BboxTuple = (
                float(tab.bbox[0]),
                float(tab.bbox[1]),
                float(tab.bbox[2]),
                float(tab.bbox[3])
            )
            tab_dict: Dict = {
                "bbox": table_bbox,
                "rows": tab.row_count,
                "cols": tab.col_count,
                "cells": [],
            }
            try:
                if getattr(tab, "header", None) and tab.header.cells:
                    for c_idx, cell in enumerate(tab.header.cells):
                        tab_dict["cells"].append(
                            _format_cell(
                                cell,
                                row=0,
                                col=c_idx,
                                is_header=True,
                                page=page,
                                table_bbox=table_bbox
                            )
                        )
            except Exception as e:
                if trace_id:
                    logger.debug("[%s] Table header extraction failed: %s", trace_id, e)
            for r_idx, row in enumerate(tab.rows):
                for c_idx, cell in enumerate(row.cells):
                    tab_dict["cells"].append(
                        _format_cell(
                            cell,
                            row=r_idx + 1,
                            col=c_idx,
                            is_header=False,
                            page=page,
                            table_bbox=table_bbox
                        )
                    )
            regions["tables"].append(tab_dict)
    except Exception as e:
        if trace_id:
            logger.warning("[%s] Table detection error: %s", trace_id, e)

    # =========================================================================
    # 3. LINK DETECTION
    # =========================================================================
    try:
        import fitz
        for link in page.get_links():
            if link.get("kind") == fitz.LINK_URI:
                link_rect = link.get("from")
                if link_rect is not None:
                    link_bbox: BboxTuple = (
                        float(link_rect.x0),
                        float(link_rect.y0),
                        float(link_rect.x1),
                        float(link_rect.y1)
                    )
                    regions["links"].append({
                        "uri": link.get("uri", ""),
                        "bbox": link_bbox,
                    })
    except Exception as e:
        if trace_id:
            logger.debug("[%s] Link detection failed: %s", trace_id, e)

    # =========================================================================
    # 3.5 DERIVE ADAPTIVE FOOTER BOUNDARY (GEOMETRY-ONLY, PRE-ROLE)
    # =========================================================================
    # Identify body-like spans using geometry proxy (no roles available yet)
    body_like_y_bottoms = []

    for span in raw_spans:
        bbox = span.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        text = (span.get("raw_text") or "").strip()
        if len(text) < 3:
            continue
        if not any(c.isalpha() for c in text):
            continue

        fs = span.get("font_size", 0)
        if fs <= 0:
            continue

        # Font size must be within ±25% of document median
        if global_median_font_size:
            if not (0.75 * global_median_font_size <= fs <= 1.25 * global_median_font_size):
                continue

        # Exclude spans inside detected figures or tables
        span_x0, span_y0, span_x1, span_y1 = bbox
        inside_structure = False
        for fx0, fy0, fx1, fy1 in regions["figures"]:
            if span_x0 >= fx0 and span_x1 <= fx1 and span_y0 >= fy0 and span_y1 <= fy1:
                inside_structure = True
                break
        if inside_structure:
            continue

        body_like_y_bottoms.append(span_y1)

    footer_zone_min_y = None
    if body_like_y_bottoms:
        # Use high percentile to avoid single-line outliers
        body_like_y_bottoms.sort()
        idx = int(len(body_like_y_bottoms) * 0.90)
        idx = min(idx, len(body_like_y_bottoms) - 1)
        footer_zone_min_y = body_like_y_bottoms[idx]

        # Apply safety clamps (geometry-only)
        max_ceiling = page_height * 0.70
        min_floor = page_height * 0.92
        footer_zone_min_y = max(footer_zone_min_y, max_ceiling)
        footer_zone_min_y = min(footer_zone_min_y, min_floor)
    else:
        # Fallback to legacy behavior if no body-like spans found
        footer_zone_min_y = page_height * (1.0 - _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT)

    # =========================================================================
    # 4. HEADER/FOOTER BAND DETECTION (MODIFIED v3.3 — Permissive Candidates)
    # =========================================================================
    y_bands: Dict[int, int] = {}

    for span in raw_spans:
        span_bbox = span.get("bbox")
        if span_bbox is None or len(span_bbox) < 4:
            continue
        span_y = span_bbox[1]

        # FIXED v3.3: Strict INT casting to match global pipeline
        y_rounded = int(round(span_y / _REGION_Y_BAND_ROUNDING) * _REGION_Y_BAND_ROUNDING)
        y_bands[y_rounded] = y_bands.get(y_rounded, 0) + 1

    for y, count in y_bands.items():
        norm_y = y / page_height if page_height > 0 else 0

        is_header_zone = norm_y < _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT
        is_footer_zone = y >= footer_zone_min_y

        # HARDEN: still permissive by zone, but ignore ultra-rare bands (noise)
        if count >= _REGION_MIN_BAND_COUNT:
            if is_header_zone:
                regions["header_bands"].append(y)
            elif is_footer_zone:
                regions["footer_bands"].append(y)

        # Body bands: Reserved for future use
        # elif count >= band_threshold:
        #     pass

    if trace_id:
        logger.debug(
            "[%s] Region detection: %d figures, %d tables, %d links, "
            "%d header bands, %d footer bands",
            trace_id,
            len(regions["figures"]),
            len(regions["tables"]),
            len(regions["links"]),
            len(regions["header_bands"]),
            len(regions["footer_bands"])
        )

    regions["header_bands"] = sorted(set(regions["header_bands"]))
    regions["footer_bands"] = sorted(set(regions["footer_bands"]))

    # Persist adaptive footer boundary for downstream global scan
    regions["footer_zone_min_y"] = footer_zone_min_y

    return regions


def _assign_block_ids(spans: List[Dict]) -> None:
    """
    V1.1: Assigns a stable block_id for each contiguous paragraph-flow group.

    For V1.1, block_id == paragraph_index.
    Future V1.2 can refine to merge multi-paragraph blocks (e.g., heading + body).
    """
    for s in spans:
        # REVISION 2: Non-destructive block assignment
        # Preserve native block_id unless paragraph logic has explicitly run.
        p_idx = s.get("paragraph_index", 0)
        if p_idx > 0:
            s["block_id"] = p_idx


def _detect_paragraphs(
        spans: List[Dict],
        trace_id: str = None,
        global_median_line_height: float = None
) -> None:
    """
    Assign paragraph_index and paragraph provenance to spans.
    Phase A Hardened: Uses "Mixed Layout Protection" and global context.
    """
    # PHASE 3 CONTRACT:
    # - This method MAY assign paragraph_index and paragraph provenance only
    # - This method MUST NOT set sentence boundaries
    # - This method MUST NOT set semantic inclusion/exclusion
    # - This method MUST NOT set or modify _has_semantic_authority

    # 1. Sort by reading order (Page, Y, X)
    spans.sort(
        key=lambda s: (
            s.get("page_number", 0),
            s.get("bbox", [0, 0, 0, 0])[1],
            s.get("bbox", [0, 0, 0, 0])[0],
            s.get("span_index", 0),  # Phase 2.3 optional: stable ordering tiebreaker
        )
    )

    if not spans:
        return

    # 2. Group by Column
    columns: Dict[int, List[Dict]] = {}
    for span in spans:
        col_idx = span.get("column_index", 0)
        if col_idx not in columns:
            columns[col_idx] = []
        columns[col_idx].append(span)

    # 3. Calculate Local Median Line Height
    line_heights: List[float] = []

    for col_spans in columns.values():
        for i in range(1, len(col_spans)):
            prev_bbox = col_spans[i - 1].get("bbox")
            curr_bbox = col_spans[i].get("bbox")
            if prev_bbox is None or curr_bbox is None:
                continue
            prev_bottom = prev_bbox[3]
            curr_top = curr_bbox[1]
            gap = curr_top - prev_bottom
            if 0 < gap < _PARA_MAX_LINE_GAP:
                line_heights.append(gap)

    local_median = _PARA_DEFAULT_LINE_HEIGHT
    if line_heights:
        sorted_heights = sorted(line_heights)
        local_median = sorted_heights[len(sorted_heights) // 2]

    # 4. Determine Threshold (Global vs Local)
    if global_median_line_height and global_median_line_height > 0:
        median_line_height = max(
            global_median_line_height * 0.75,
            min(local_median, global_median_line_height * 1.5)
        )
    else:
        median_line_height = local_median

    para_gap_threshold = median_line_height * _PARA_GAP_MULTIPLIER

    # 5. Assign Indices (The "Decision Loop")
    global_para_idx = 0

    col_keys = list(columns.keys())
    body_cols = sorted([k for k in col_keys if k not in (-1, 100)])
    margin_cols = [k for k in (-1, 100) if k in col_keys]

    for col_idx in (body_cols + margin_cols):
        col_spans = columns[col_idx]
        if not col_spans:
            continue
        # Phase 3 provenance: default no column transition
        for _s in col_spans:
            _s["_column_transition"] = False

        # Tag first span of each column after the first as column transition
        if col_idx != (body_cols + margin_cols)[0]:
            col_spans[0]["_column_transition"] = True
            col_spans[0].setdefault("_paragraph_break_reason", "column_transition")

        col_spans[0]["paragraph_index"] = global_para_idx
        prev_bottom = col_spans[0].get("bbox", [0, 0, 0, 0])[3]

        for i in range(1, len(col_spans)):
            curr_span = col_spans[i]
            curr_bbox = curr_span.get("bbox")

            if curr_bbox is None:
                curr_span["paragraph_index"] = global_para_idx
                continue

            prev_span = col_spans[i - 1]
            prev_bbox = prev_span.get("bbox")

            curr_top = curr_bbox[1]
            gap = curr_top - prev_bottom
            start_new_paragraph = False

            # ------------------------------------------------------------------
            # TEXT EXTRACTION (SPAN-SAFE)
            # ------------------------------------------------------------------
            prev_text = (prev_span.get("raw_text") or prev_span.get("text") or "").strip()
            curr_text = (curr_span.get("raw_text") or curr_span.get("text") or "").strip()

            prev_tail = prev_text.rstrip()
            curr_head = curr_text.lstrip()

            # Guard against empty text
            if not prev_text or not curr_text:
                curr_span["paragraph_index"] = global_para_idx
                prev_bottom = curr_bbox[3]
                continue

            # ------------------------------------------------------------------
            # SEMANTIC SIGNALS
            # ------------------------------------------------------------------
            has_lowercase_start = curr_head[:1].islower()
            prev_ends_hard = prev_tail.endswith((".", "!", "?"))
            curr_starts_cap = curr_head[:1].isupper()

            # Abbreviation detection (robust)
            prev_last_token = prev_tail.lower().rstrip(".!?")
            prev_last_token = prev_last_token.split()[-1] if prev_last_token.split() else ""
            is_abbreviation = prev_last_token in _PARA_ABBREVIATIONS

            # ------------------------------------------------------------------
            # A. VERTICAL GAP CHECK (OUTER GATE — REQUIRED)
            # Paragraph logic must only engage when the vertical gap exceeds
            # normal line spacing.
            # ------------------------------------------------------------------

            if gap > para_gap_threshold:

                # VISUAL GAP ZONES
                is_visual_break = gap > (median_line_height * 1.5)
                is_borderline = para_gap_threshold < gap <= (para_gap_threshold * 1.25)

                # SEMANTIC HARD STOP (GUARDED BY GAP)
                semantic_break = (
                        prev_ends_hard and
                        curr_starts_cap and
                        not is_abbreviation
                )

                if is_visual_break or semantic_break:
                    start_new_paragraph = True


                elif is_borderline:
                    # Borderline zone (1.25x to 1.5x median line height):
                    # Use semantic signals to decide if sentence continues.
                    continues_sentence = (
                            not prev_ends_hard and
                            (
                                    has_lowercase_start or
                                    curr_head[:1] in {",", ";", ":", ")", "]"} or
                                    prev_tail.endswith((",", ";", ":", "-", "—", "(", "["))
                            )
                    )

                    if not continues_sentence:
                        start_new_paragraph = True
                    # else: Keep same paragraph (honor continuation signals)

            # ------------------------------------------------------------------
            # PHASE 2.3: Same-line detection (used by multiple guards)
            # line_id is primary (discrete), row_key is fallback (bucketed float)
            # ------------------------------------------------------------------
            same_line = (
                    (prev_span.get("line_id") is not None and prev_span.get(
                        "line_id") == curr_span.get("line_id")) or
                    (prev_span.get("row_key") is not None and prev_span.get(
                        "row_key") == curr_span.get("row_key"))
            )

            # ------------------------------------------------------------------
            # HORIZONTAL COHESION CHECK (MIXED LAYOUT PROTECTION)
            # PHASE 2.3: Skip for same visual line (inline spans have zero overlap)
            # ------------------------------------------------------------------
            if not start_new_paragraph and prev_bbox:
                if not same_line:
                    overlap_x = min(prev_bbox[2], curr_bbox[2]) - max(prev_bbox[0], curr_bbox[0])
                    min_width = min(
                        prev_bbox[2] - prev_bbox[0],
                        curr_bbox[2] - curr_bbox[0]
                    )

                    horizontal_cohesion = (
                            min_width > 0 and
                            (overlap_x / min_width) >= 0.65
                    )

                    if not horizontal_cohesion:
                        start_new_paragraph = True
                        # Paragraph break provenance (do not overwrite if already set)
                        curr_span.setdefault("_paragraph_break_reason", "horizontal_break")

            # ------------------------------------------------------------------
            # FINALIZE PARAGRAPH INDEX
            # ------------------------------------------------------------------

            if start_new_paragraph:
                global_para_idx += 1

                # Paragraph break provenance (Phase 3 contract)
                if gap > para_gap_threshold:
                    curr_span.setdefault("_paragraph_break_reason", "visual_gap")
                elif prev_ends_hard and curr_starts_cap and not is_abbreviation:
                    curr_span["_paragraph_break_reason"] = "semantic_gap"
                else:
                    # Fallback: paragraph split without dominant visual or semantic signal
                    curr_span["_paragraph_break_reason"] = "layout_transition"

            curr_span["paragraph_index"] = global_para_idx

            # PHASE 2.3 HARDEN: Track max bottom across same-line spans.
            # Prevents artificially large gaps when inline spans vary in height.
            if start_new_paragraph or not same_line:
                prev_bottom = curr_bbox[3]
            else:
                prev_bottom = max(prev_bottom, curr_bbox[3])

        # Advance paragraph index after each column
        # Tag column transition on first span of next column (handled in P3.4)
        global_para_idx += 1
        # Note: column boundary paragraph increment
        # First span of next column will carry _column_transition = True
        col_spans[0]["_paragraph_break_reason"] = "column_transition"

    if trace_id and __debug__:
        for s in spans:
            assert "_sentence_id" not in s, "Phase 3 leaked sentence state"
            assert "_semantic_disposition" not in s, "Phase 3 leaked semantic state"
            assert "_has_semantic_authority" not in s, "Phase 3 leaked authority"

    # ─────────────────────────────────────────────────────────────────
    # FIX v6.5e: Punctuation Paragraph Inheritance
    # Punctuation-only spans should inherit paragraph_index from body neighbors
    # to avoid being sorted into wrong position during reconstruction.
    # Prefer body-role neighbors over sidebar/margin neighbors.
    # ─────────────────────────────────────────────────────────────────
    for i, span in enumerate(spans):
        text = (span.get("cleaned_text") or "").strip()
        # Only process punctuation-only spans (1-3 chars, no alphanumeric)
        if len(text) <= 3 and text and not any(c.isalnum() for c in text):
            curr_para = span.get("paragraph_index")

            # Get neighbors
            prev_span = spans[i - 1] if i > 0 else None
            next_span = spans[i + 1] if i < len(spans) - 1 else None

            # Prefer body-role neighbors for paragraph inheritance
            prev_para = prev_span.get("paragraph_index") if prev_span else None
            next_para = next_span.get("paragraph_index") if next_span else None
            prev_is_body = prev_span.get("role") == "body" if prev_span else False
            next_is_body = next_span.get("role") == "body" if next_span else False

            # Inheritance priority: body neighbor > any neighbor
            if next_is_body and next_para is not None:
                span["paragraph_index"] = next_para
                span["_paragraph_inherited_from"] = "next_body"
            elif prev_is_body and prev_para is not None:
                span["paragraph_index"] = prev_para
                span["_paragraph_inherited_from"] = "prev_body"
            elif prev_para is not None and prev_para != curr_para:
                span["paragraph_index"] = prev_para
                span["_paragraph_inherited_from"] = "prev"
            elif next_para is not None and next_para != curr_para:
                span["paragraph_index"] = next_para
                span["_paragraph_inherited_from"] = "next"
    # Summary logging (unconditional)
    if trace_id:
        logger.debug(
            "[%s] Paragraph detection: %d paragraphs across %d columns",
            trace_id, global_para_idx, len(columns)
        )


# ✦────── d. Role Detectors ──────✦

def _span_is_code(span: Dict) -> bool:
    """
    Detect code spans using font and punctuation heuristics.

    Criteria (any match = code):
        1. Font name contains monospace hint (courier, mono, etc.)
        2. High density of code punctuation characters

    Args:
        span: Span dictionary with font and text fields.

    Returns:
        True if span appears to be code.
    """
    # Check font hints
    font_name = span.get("font", "").lower()
    if any(hint in font_name for hint in CODE_FONT_HINTS):
        return True

    # FIX: Define text before using it
    text = (span.get("cleaned_text") or "").strip()

    # HARDENING: Avoid math-heavy false positives (equations are not code)
    if any(c in text for c in "∑∫√≈≤≥"):
        return False

    punct_chars = _CODE_PUNCT_PATTERN.findall(text)
    punct_threshold = max(_CODE_MIN_PUNCT_COUNT, len(text) // _CODE_PUNCT_DENSITY_DIVISOR)

    if len(punct_chars) >= punct_threshold:
        return True

    return False


def _is_inline_equation(text: str, span: Dict) -> bool:
    """
    Detect inline equations using 3-gate validation.

    Gates:
        1. Must contain non-ASCII math/Greek symbols
        2. If long text, must have multiple symbols
        3. Math font hint boosts confidence (auto-pass)

    Args:
        text: Text content to analyze.
        span: Span dictionary with font metadata.

    Returns:
        True if text appears to be an inline equation.
    """
    if not text:
        return False

    # Gate 1: Must contain math symbols
    symbols = _MATH_SYMBOLS_PATTERN.findall(text)
    if not symbols:
        return False

    # Gate 2: Long text needs multiple symbols
    if len(text) > _EQUATION_LONG_TEXT_THRESHOLD and len(symbols) < _EQUATION_MIN_SYMBOLS_FOR_LONG:
        return False

    # Gate 3: Math font = high confidence
    font_name = span.get("font", "").lower()
    if any(hint in font_name for hint in _MATH_FONT_HINTS):
        return True

    # Passed gates 1 & 2
    return True


def _is_diagram_label(
        span: Dict,
        figure_tuples: List[BboxTuple],
        baseline_font_size: float = 12.0,
        trace_id: str = None
) -> bool:
    """
    Conservative diagram label detection with adaptive thresholds.

    MODIFIED v3.3 MINIMAL:
        1. PROSE PROTECTION: Italic body text bypasses all detection.
        2. PROSE INDICATORS: Combined articles/starters/verbs guard.
        3. ALL thresholds derived from existing constants.
        4. Only 2 new frozensets added (term data, not thresholds).

    Threshold Derivations (from existing constants):
        - Body range: _LABEL_FONT_RATIO+0.1 to 1/_LABEL_FONT_RATIO
        - Short: _LABEL_MAX_WORDS
        - Forced max: _LABEL_MAX_WORDS - 1
        - Title max: _LABEL_MAX_WORDS // 2
        - Header ceiling: 2 / _LABEL_FONT_RATIO
        - Technical ceiling: 1.5 / _LABEL_FONT_RATIO

    Args:
        span: Span dictionary to evaluate.
        figure_tuples: List of (x0, y0, x1, y1) figure bounding boxes.
        baseline_font_size: Document's baseline font size.
        trace_id: Optional trace ID for logging.

    Returns:
        True only if span is definitively a diagram label.
    """
    span_text = span.get("raw_text", "").strip()
    if not span_text:
        return False

    # =========================================================================
    # SETUP
    # =========================================================================
    font_name = span.get("font", "").lower()
    font_size = span.get("font_size", baseline_font_size)
    is_italic = "italic" in font_name or "oblique" in font_name

    # Body size range for prose protection
    body_size_lower = baseline_font_size * (_LABEL_FONT_RATIO + 0.1)
    body_size_upper = baseline_font_size / _LABEL_FONT_RATIO

    words = span_text.split()
    word_count = len(words)
    words_lower = [w.lower().rstrip('.,;:()') for w in words]

    # FIXED v3.5: Stabilized thresholds (Decoupled from constant changes)
    forced_max_words = max(5, _LABEL_MAX_WORDS - 1)
    title_case_max = max(3, _LABEL_MAX_WORDS // 2)
    prose_check_min = 4  # Standard English threshold for "phrase vs sentence"

    header_ceiling = baseline_font_size * (2 / _LABEL_FONT_RATIO)
    technical_ceiling = baseline_font_size * (1.5 / _LABEL_FONT_RATIO)
    standard_ceiling = baseline_font_size / _LABEL_FONT_RATIO

    # =========================================================================
    # GUARD 1: Character length ceiling (existing constant)
    # =========================================================================
    if len(span_text) > _LABEL_MAX_CHARS:
        return False

    # =========================================================================
    # GUARD 2: Caption detection
    # =========================================================================
    if any(pattern.match(span_text) for pattern in _COMPILED_CAPTION_PATTERNS):
        if trace_id:
            logger.debug(
                "[%s] _is_diagram_label: '%s' rejected (caption)",
                trace_id, span_text[:25]
            )
        return False

    # =========================================================================
    # GUARD 2.5: Heading protection (P5 FIX)
    # Headings should never be classified as diagram labels, even if short
    # and containing technical terms. Section structure must be preserved.
    # =========================================================================
    span_role = span.get("role", "")
    if span_role in ("heading", "subheading"):
        if trace_id:
            logger.debug(
                "[%s] _is_diagram_label: '%s' rejected (heading/subheading role)",
                trace_id, span_text[:25]
            )
        return False

    # =========================================================================
    # PATH A: GEOMETRIC CONTAINMENT
    # =========================================================================
    span_rect = _span_to_rect(span)
    inside_any_figure = False

    # FIX: Add padding to catch labels floating just outside the crop box
    pad = _REGION_FIGURE_EXPAND_PADDING  # Uses constant (15px)

    if span_rect is not None and figure_tuples:
        span_center_x = (span_rect[0] + span_rect[2]) / 2
        span_center_y = (span_rect[1] + span_rect[3]) / 2

        for fig_rect in figure_tuples:
            fig_x0, fig_y0, fig_x1, fig_y1 = fig_rect
            fig_width = fig_x1 - fig_x0
            fig_height = fig_y1 - fig_y0

            if fig_width <= 0 or fig_height <= 0:
                continue

            # EXPANDED CHECK: Apply padding to figure bounds
            if (fig_x0 - pad <= span_center_x <= fig_x1 + pad and
                    fig_y0 - pad <= span_center_y <= fig_y1 + pad):
                inside_any_figure = True
                break

    # =========================================================================
    # PATH B: CONTENT-BASED DETECTION (Orphan Labels)
    # =========================================================================
    is_orphan_label = False

    if not inside_any_figure:
        # MOVED v3.5: Prose Protection (Italic Body Text)
        # We only apply this check if the text is NOT inside a figure.
        # This allows italic labels (e.g., biological names) inside diagrams
        # while protecting emphasis text in the margins.
        if is_italic and body_size_lower <= font_size <= body_size_upper:
            if trace_id:
                logger.debug(
                    "[%s] _is_diagram_label: '%s' PROTECTED (italic body text)",
                    trace_id, span_text[:30]
                )
            return False
        # =====================================================================
        # FORCED LABEL DETECTION (uses existing FORCED_LABEL_TERMS)
        # =====================================================================
        forced_term_count = sum(1 for w in words_lower if w in FORCED_LABEL_TERMS)

        if forced_term_count >= 2 and word_count <= forced_max_words:
            if font_size >= header_ceiling:
                pass  # Too large, likely a header
            else:
                # FIXED v3.5: Allow bold labels.
                # Textbooks frequently use bold for emphasis in diagrams.
                if trace_id:
                    logger.debug(
                        "[%s] _is_diagram_label: '%s' FORCED (kill list)",
                        trace_id, span_text[:30]
                    )
                return True

        # =====================================================================
        # TECHNICAL TERMS ANALYSIS
        # =====================================================================
        technical_count = sum(1 for w in words_lower if w in _LABEL_TECHNICAL_TERMS)
        technical_ratio = technical_count / word_count if word_count > 0 else 0

        # =====================================================================
        # FONT SIZE CHECK
        # Uses derived thresholds from SETUP section
        # =====================================================================
        is_high_density = technical_ratio >= 0.5 and technical_count >= 2

        if is_high_density and font_size < technical_ceiling:
            font_is_acceptable = True
            if trace_id:
                logger.debug(
                    "[%s] _is_diagram_label: '%s' density override (%.0f%%)",
                    trace_id, span_text[:30], technical_ratio * 100
                )
        else:
            font_is_acceptable = font_size <= standard_ceiling

        if not font_is_acceptable:
            if trace_id:
                logger.debug(
                    "[%s] _is_diagram_label: '%s' rejected (font %.1f)",
                    trace_id, span_text[:25], font_size
                )
        else:
            # =================================================================
            # ORPHAN LABEL INDICATORS
            # Uses combined _LABEL_PROSE_INDICATORS set
            # Short threshold: _LABEL_MAX_WORDS (existing)
            # =================================================================
            has_prose_indicator = any(w in _LABEL_PROSE_INDICATORS for w in words_lower)
            is_short = word_count <= _LABEL_MAX_WORDS
            has_technical = technical_count >= 1
            has_parenthetical = False
            if '(' in span_text and ')' in span_text:
                inside_parens = span_text[span_text.find('(') + 1: span_text.rfind(')')]

                # HARDENED: Exclude citations (years) and eq refs from being labels
                is_citation = re.search(r'\b(19|20)\d{2}\b', inside_parens)
                is_eq_ref = inside_parens.lower().startswith("eq")

                if not is_citation and not is_eq_ref:
                    if len(span_text) > 0 and (len(inside_parens) / len(span_text) > 0.60):
                        has_parenthetical = True

            # Title case detection
            is_title_case = span_text[0].isupper() and not span_text.isupper()
            ends_with_period = span_text.rstrip().endswith('.')

            # =================================================================
            # ORPHAN LABEL DECISION
            # =================================================================
            if is_short and not has_prose_indicator:
                if has_technical:
                    is_orphan_label = True
                    if trace_id:
                        logger.debug(
                            "[%s] _is_diagram_label: '%s' orphan (technical)",
                            trace_id, span_text[:30]
                        )
                elif has_parenthetical:
                    is_orphan_label = True
                    if trace_id:
                        logger.debug(
                            "[%s] _is_diagram_label: '%s' orphan (parenthetical)",
                            trace_id, span_text[:30]
                        )
                elif word_count <= title_case_max and is_title_case and ends_with_period:
                    is_orphan_label = True
                    if trace_id:
                        logger.debug("[%s] _is_diagram_label: '%s' orphan (title-case)", trace_id,
                                     span_text[:30])

                    # FIXED v3.5: Extended Patterns (Engineering/Anatomy/Science)
                elif span_text.isupper() and word_count <= title_case_max:
                    is_orphan_label = True  # ALL-CAPS LABELS
                elif re.match(r'^(\d+|[a-zA-Z])\.$', span_text):
                    is_orphan_label = True  # Enumerators "1.", "A."
                elif any(c in span_text for c in "→←↑↓"):
                    is_orphan_label = True  # Directional Arrows
                elif re.match(r'^\d+\s*[µnmk]?m$', span_text):
                    is_orphan_label = True  # Measurements (10 mm, 5um)
                elif re.match(r'^\(?(?:[a-zA-Z]|[ivxIVX]{2,4})\)?\.?$', span_text.strip()):
                    is_orphan_label = True

    # =========================================================================
    # GUARD 3: Must pass either path
    # =========================================================================
    if not inside_any_figure and not is_orphan_label:
        return False

    # =========================================================================
    # GUARD 4: Strong prose indicators
    # =========================================================================
    has_sentence_end = _LABEL_SENTENCE_END_PATTERN.search(span_text)

    # HARDENED: Reject sentence-like structures even if short (e.g. "Data flows up.")
    if has_sentence_end and word_count >= 6:
        return False

    if span_text.count('.') > 1:
        return False

    if has_sentence_end and word_count > _LABEL_MAX_WORDS:
        return False

    # =========================================================================
    # GUARD 5: Prose word patterns
    # =========================================================================
    if word_count > prose_check_min and _PROSE_WORDS_PATTERN.search(span_text):
        return False

    # =========================================================================
    # GUARD 6: Word count ceiling (existing constant)
    # =========================================================================
    if word_count > _LABEL_MAX_WORDS:
        return False

    # =========================================================================
    # GUARD 7: Sentence Continuation
    # =========================================================================
    # Labels may be lowercase ("flow rate"), but not long sentence tails ("flow rate increases.")
    if span_text and span_text[0].islower() and word_count >= 4:
        return False

    # =========================================================================
    # If reached via orphan path, already accepted
    # =========================================================================
    if is_orphan_label:
        return True

    # =========================================================================
    # GEOMETRIC PATH: Accept all text inside figures that passed guards
    # FIXED v3.6: Text inside figures that isn't a caption/sentence is a label.
    # =========================================================================
    if inside_any_figure:
        if trace_id:
            logger.debug(
                "[%s] _is_diagram_label: '%s' accepted (inside figure)",
                trace_id, span_text[:30]
            )
        return True

    return False


def _is_caption_candidate(
        span: Dict,
        figure_rects: List[BboxTuple]
) -> bool:
    """
    Detect figure/table captions using pattern matching and proximity.

    A span is a caption candidate if:
        1. Text matches explicit caption patterns (Figure 1, Table 2, etc.), OR
        2. Span geometrically intersects a figure region

    Args:
        span: Span dictionary with cleaned_text and bbox.
        figure_rects: List of (x0, y0, x1, y1) figure bounding boxes.

    Returns:
        True if span appears to be a caption.
    """
    text = (span.get("cleaned_text") or "").strip()
    if not text:
        return False

    # Check explicit caption patterns
    if any(pattern.match(text) for pattern in _COMPILED_CAPTION_PATTERNS):
        return True

    # Check proximity to figures
    if figure_rects:
        span_rect = _span_to_rect(span)
        if span_rect is not None:
            buffer = _ROLE_FIGURE_PROXIMITY_BUFFER

            # HARDENING: Avoid long prose misclassified as caption
            # Captions are rarely longer than the bare caption threshold
            if len(text) > _FILTER_BARE_CAPTION_MAX_CHARS:
                return False

            # GUARD: Proximity captions (no pattern) rarely end in a period if they are long
            # This prevents "The data is shown below." from being a caption.
            if text.endswith(".") and len(text.split()) > 6:
                return False

            for fig_rect in figure_rects:
                # Create an expanded detection box
                expanded_rect = (
                    fig_rect[0] - buffer,  # x0
                    fig_rect[1] - buffer,  # y0
                    fig_rect[2] + buffer,  # x1
                    fig_rect[3] + buffer  # y1
                )

                # Check intersection with the expanded zone
                if _rects_intersect(span_rect, expanded_rect):
                    return True

    return False


# ✦────── e. Classification ──────✦

def _assign_roles(
        spans: List[Dict],
        regions: Dict,
        page_width: float = None,
        page_height: float = None,
        trace_id: str = None,
        global_baseline_font_size: float = None
) -> None:
    """
    Comprehensive role classification with adaptive thresholds.

    Priority Cascade (highest to lowest):
        1. Table Cell — text inside table bounding box
        2. Inside Figure — text inside figure bounding box
        3. Figure Label — short text near/inside figure (axis labels, annotations)
        4. Caption — explicit caption patterns or continuation (with guards)
        5. Sidebar — margin content (adaptive detection)
        6. Heading — large/bold, short, title case, no terminal punct
        7. Subheading — medium size, short, title case
        8. Footnote — small font at page bottom
        9. List Item — bullet/number prefix patterns
        10. Code — monospace font
        11. Hyperlink — text matches link URI
        12. Inline Equation — math symbols present
        13. Subscript/Superscript — small font with baseline offset
        14. Header Artifact — orphaned fragments in header zone
        15. Body — default

    Args:
        spans: List of span dictionaries with tuple bboxes.
        regions: Dictionary with figures, tables, links (from _detect_page_regions).
        page_width: Page width in points.
        page_height: Page height in points.
        trace_id: Optional trace ID for logging.
        global_baseline_font_size: Document-wide baseline font size for heading
            detection. If provided, prevents per-page skewing from figure
            annotations. If None, falls back to per-page median.

    Mutates:
        Each span receives 'role' key.
    """
    # AUTHORITY BOUNDARY:
    # Phase 4 MUST NOT finalize inclusion or exclusion.
    # All exclusion signals here are provisional and MUST be resolved in Phase 5.
    if not spans:
        return

    # =========================================================================
    # SETUP: Unpack regions (already tuples from _detect_page_regions)
    # =========================================================================
    figure_rects: List[BboxTuple] = regions.get("figures", [])
    tables: List[Dict] = regions.get("tables", [])
    links: List[Dict] = regions.get("links", [])

    # Table bboxes are already tuples from Subgroup 2.3
    table_rects: List[BboxTuple] = [
        t["bbox"] for t in tables if t.get("bbox") is not None
    ]

    # Expanded figure rects for "near figure" detection
    figure_rects_expanded: List[BboxTuple] = [
        (
            r[0] - _ROLE_FIGURE_PROXIMITY_BUFFER,
            r[1] - _ROLE_FIGURE_PROXIMITY_BUFFER,
            r[2] + _ROLE_FIGURE_PROXIMITY_BUFFER,
            r[3] + _ROLE_FIGURE_PROXIMITY_BUFFER
        )
        for r in figure_rects
    ]

    link_uris: set = {lnk.get("uri", "") for lnk in links}

    # =========================================================================
    # METRICS: Calculate font statistics
    # =========================================================================
    # FIX v6.2: Use document-wide baseline if provided.
    # This prevents figure-heavy pages from skewing the baseline downward,
    # which would cause normal body text to be misclassified as headings.
    # Typography principle: documents use consistent body fonts across pages.
    # =========================================================================
    if global_baseline_font_size is not None and global_baseline_font_size > 0:
        baseline_font_size = global_baseline_font_size
    else:
        # Fallback to per-page median if document-wide not available
        font_sizes = [s.get("font_size", 0) for s in spans if s.get("font_size", 0) > 0]
        if font_sizes:
            font_sizes_sorted = sorted(font_sizes)
            median_font_size = font_sizes_sorted[len(font_sizes_sorted) // 2]
            baseline_font_size = median_font_size
        else:
            baseline_font_size = 12.0

    # =========================================================================
    # PAGE DIMENSIONS: Adaptive thresholds
    # =========================================================================
    effective_page_height = page_height if page_height else _FILTER_DEFAULT_PAGE_HEIGHT
    effective_page_width = page_width if page_width else _LAYOUT_DEFAULT_PAGE_WIDTH

    # Estimated line height
    estimated_line_height = baseline_font_size * _ROLE_LINE_HEIGHT_MULTIPLIER

    # Zone thresholds
    header_zone_y = effective_page_height * _ROLE_HEADER_ZONE_RATIO
    footer_zone_y = effective_page_height * _ROLE_FOOTER_ZONE_RATIO

    # Caption chain thresholds
    caption_y_threshold = estimated_line_height * _ROLE_CAPTION_Y_THRESHOLD_MULTIPLIER
    caption_x_threshold = effective_page_width * _ROLE_CAPTION_X_THRESHOLD_RATIO
    max_caption_chain_distance = estimated_line_height * _ROLE_CAPTION_MAX_CHAIN_LINES

    # =========================================================================
    # STATE INITIALIZATION (for continuity detection)
    # =========================================================================
    previous_role: str = ""
    previous_y: float = 0.0
    previous_x: float = 0.0
    previous_text: str = ""
    # Path 3 (Layout Separation): Track Y-position of last body-classified
    # span. Updated ONLY when a span receives the body role. Isolates the
    prev_body_y: Optional[float] = None

    # Caption chain tracking
    caption_chain_start_y: Optional[float] = None
    caption_chain_start_font_size: Optional[float] = None
    caption_chain_length: int = 0

    # =========================================================================
    # A2-CHAIN HEADING DETECTION (Strategy 2)
    #
    # Walks a2 continuation chains, joins fragment text, tests against
    # _HEADING_STRUCTURAL_PREFIX_PATTERNS. Marks chain members directly
    # on span dicts because _canonical_span_id is not yet assigned
    # (Step 11.1 runs after Step 9 — CID timing constraint).
    # Transient _a2_chain_is_heading marker cleaned up after main loop.
    # =========================================================================
    _a2_chains: List[List[Dict]] = []
    _current_chain: List[Dict] = []
    for _s in spans:
        if _s.get("a2_continues_from_previous") and _current_chain:
            _current_chain.append(_s)
            # Bidirectional guard: verify forward/backward a2 agreement
            if not _current_chain[-2].get("a2_continues_to_next"):
                _a2_chains.append(_current_chain[:-1])
                _current_chain = [_s]
        else:
            if _current_chain:
                _a2_chains.append(_current_chain)
            _current_chain = [_s]
    if _current_chain:
        _a2_chains.append(_current_chain)

    for _chain in _a2_chains:
        if len(_chain) < 2:
            continue
        _chain_sorted = sorted(
            _chain, key=lambda s: s.get("span_index_in_line", 0)
        )
        _joined = " ".join(
            (s.get("cleaned_text") or s.get("raw_text") or "").strip()
            for s in _chain_sorted
        ).strip()
        if _joined and any(
                p.match(_joined) for p in _HEADING_STRUCTURAL_PREFIX_PATTERNS
        ):
            for _s in _chain:
                _s["_a2_chain_is_heading"] = True

    # =========================================================================
    # MAIN CLASSIFICATION LOOP
    # =========================================================================
    for span in spans:
        role = TextRole.BODY.value
        role_origin = "default"

        # HARDENED: Enforce schema contract
        if not isinstance(span.get("flags", 0), int):
            span["flags"] = 0

        # Extract span properties (tuple bbox)
        bbox = span.get("bbox")
        if bbox is None or len(bbox) < 4:
            # PHASE 0: Don't mark as EMPTY just because bbox is invalid
            # Assign BODY default and let downstream handle based on text content
            if not (span.get("raw_text") or "").strip():
                span["role"] = TextRole.EMPTY.value
                previous_role = TextRole.EMPTY.value
                continue
            # Has text but no bbox — assign default role, skip geometry-dependent checks
            span["role"] = TextRole.BODY.value
            span["_role_origin"] = "geometry"
            span["_bbox_invalid_in_role_assignment"] = True
            previous_role = TextRole.BODY.value
            continue

        span_x0, span_y0, span_x1, span_y1 = bbox
        span_x = span_x0
        span_y = span_y0

        font_size = span.get("font_size", baseline_font_size)
        font_name = span.get("font", "").lower()
        span_text = (span.get("cleaned_text") or span.get("raw_text") or "").strip()

        span_rect = _span_to_rect(span)
        if span_rect is None:
            span["role"] = TextRole.EMPTY.value
            span["_role_origin"] = "default"
            previous_role = TextRole.EMPTY.value
            continue

        word_count = len(span_text.split()) if span_text else 0
        char_count = len(span_text)

        # ─────────────────────────────────────────────────────────────
        # HEADING EVIDENCE SIGNALS (computed once, referenced by
        # Priorities 2, 3, and 6)
        # ─────────────────────────────────────────────────────────────

        # A2-chain heading signal (Strategy 2, set by pre-loop scan).
        # Allows a2 continuation fragments to inherit heading candidacy
        # from their chain when joined text matches structural patterns.
        _line_is_heading = span.get("_a2_chain_is_heading", False)

        # Bold font weight (computed here for Priority 2/3 escape guards).
        # Priority 6 recomputes independently as is_bold within its block.
        has_bold_weight = any(
            hint in font_name for hint in _HEADING_FONT_WEIGHT_HINTS
        )

        # Path 2: Structural prefix match (section numbers, keywords, symbols).
        has_structural_prefix = any(
            p.match(span_text) for p in _HEADING_STRUCTURAL_PREFIX_PATTERNS
        ) if span_text else False

        # Path 3: Significant vertical gap from last body-classified span.
        # Only meaningful after at least one body span has been seen.
        has_vertical_isolation = (
                prev_body_y is not None and
                abs(span_y - prev_body_y) >
                estimated_line_height * _HEADING_VERTICAL_ISOLATION_MULTIPLIER
        )

        # Skip empty spans
        if not span_text:
            span["role"] = TextRole.EMPTY.value
            span["_role_origin"] = "default"
            previous_role = TextRole.EMPTY.value
            continue

        # =====================================================================
        # PRIORITY 1: Table Cell
        # =====================================================================
        for t_rect in table_rects:
            if _rects_intersect(span_rect, t_rect):
                role = TextRole.TABLE_CELL.value
                role_origin = "geometry"
                break

        # =====================================================================
        # PRIORITY 2: Inside Figure
        # =====================================================================
        if role == TextRole.BODY.value:
            for f_rect in figure_rects:
                if _rects_intersect(span_rect, f_rect):
                    # ═══════════════════════════════════════════════════════════
                    # FIX v6.0: Heading Rescue Guard (Prevention)
                    # Don't classify as inside_figure if span has strong heading
                    # signals. This prevents headings trapped in imprecise figure
                    # bboxes (from images, drawings, or synthetic detection) from
                    # being lost. The span will fall through to PRIORITY 6.
                    # ═══════════════════════════════════════════════════════════
                    is_heading_candidate = (
                            (font_size >= baseline_font_size * _ROLE_HEADING_FONT_RATIO
                             or has_structural_prefix
                             or has_vertical_isolation
                             or has_bold_weight
                             or _line_is_heading) and
                            span_text and (span_text[
                                               0].isupper() or has_structural_prefix or _line_is_heading) and
                            span_text[-1] not in _ROLE_TERMINAL_PUNCTUATION and
                            span.get("span_index_in_line", 0) == 0 and
                            (not span.get("a2_continues_from_previous", False) or _line_is_heading)
                    )

                    if is_heading_candidate:
                        # Skip inside_figure; let PRIORITY 6 (Heading) handle it
                        span["_exclusion_protected"] = True
                        span.setdefault("_exclusion_protection_reasons", []).append(
                            "assign_roles:heading_escape_from_figure"
                        )
                        break

                    role = TextRole.INSIDE_FIGURE.value
                    role_origin = "geometry"
                    break

        # =====================================================================
        # PRIORITY 3: Figure Label
        # =====================================================================
        if role == TextRole.BODY.value:
            is_near_figure = any(
                _rects_intersect(span_rect, f_exp)
                for f_exp in figure_rects_expanded
            )
            if is_near_figure:
                is_short = (
                        word_count <= _ROLE_FIGURE_LABEL_MAX_WORDS and
                        char_count <= _ROLE_FIGURE_LABEL_MAX_CHARS
                )
                is_small_font = font_size <= baseline_font_size * _ROLE_FIGURE_LABEL_FONT_RATIO
                # Check label patterns
                matches_label = _LABEL_PATTERN.match(span_text) is not None
                matches_stats = _STATISTICAL_LABEL_PATTERN.match(span_text) is not None

                # ═══════════════════════════════════════════════════════════════
                # FIX v6.0b: Heading Guard for Figure Label
                # Don't classify as figure_label if span has heading signals.
                # Headings near figures should not be demoted to labels.
                # ═══════════════════════════════════════════════════════════════
                is_heading_candidate = (
                        (font_size >= baseline_font_size * _ROLE_HEADING_FONT_RATIO
                         or has_structural_prefix
                         or has_bold_weight
                         or _line_is_heading) and
                        span_text and (span_text[
                                           0].isupper() or has_structural_prefix or _line_is_heading) and                        span_text[-1] not in _ROLE_TERMINAL_PUNCTUATION and
                        span.get("span_index_in_line", 0) == 0 and
                        (not span.get("a2_continues_from_previous", False) or _line_is_heading)
                )

                if is_heading_candidate:
                    span["_exclusion_protected"] = True
                    span.setdefault("_exclusion_protection_reasons", []).append(
                        "assign_roles:heading_escape_from_figure_label"
                    )

                if matches_label or matches_stats:
                    if not is_heading_candidate:
                        role = TextRole.FIGURE_LABEL.value
                        role_origin = "content_pattern"
                elif is_short and word_count <= _ROLE_FIGURE_LABEL_SHORT_WORD_COUNT:
                    if not is_heading_candidate:  # FIX v6.0b guard
                        role = TextRole.FIGURE_LABEL.value
                        role_origin = "content_pattern"
                elif is_short and is_small_font:
                    if not is_heading_candidate:  # FIX v6.0b guard
                        role = TextRole.FIGURE_LABEL.value
                        role_origin = "content_pattern"

        # =====================================================================
        # PRIORITY 4: Caption (with guards)
        # =====================================================================
        # AUTHORITY BOUNDARY:
        # Phase 4 MUST NOT finalize inclusion or exclusion.
        # All exclusion signals here are provisional and MUST be resolved in Phase 5.

        if role == TextRole.BODY.value:
            # Explicit caption pattern match — starts new chain
            is_explicit_caption = any(
                pattern.match(span_text)
                for pattern in _COMPILED_CAPTION_PATTERNS
            )

            if is_explicit_caption:
                # ═════════════════════════════════════════════════════════
                # Heading Rescue Guard for Caption (mirrors P2/P3 guards)
                #
                # Letter-prefixed headings ("A. Background", "B. Methods",
                # "I. Introduction") match caption pattern ^[A-Z]\.\s+ but
                # are structural section markers. Don't classify as caption
                # if heading evidence is present — let Priority 6 handle.
                #
                # Guard uses has_structural_prefix ONLY (not vertical
                # isolation). Caption patterns are content-based; a span
                # matching a caption pattern AND having a vertical gap is
                # more likely a figure caption after whitespace than a
                # heading. Structural prefix is the stronger override.
                # ═════════════════════════════════════════════════════════
                is_heading_candidate = (
                        has_structural_prefix and
                        word_count <= _ROLE_HEADING_MAX_WORDS and
                        char_count <= _ROLE_HEADING_MAX_CHARS and
                        span_text[-1] not in _ROLE_TERMINAL_PUNCTUATION and
                        span.get("span_index_in_line", 0) == 0 and
                        not span.get("a2_continues_from_previous", False)
                )
                if not is_heading_candidate:
                    role = TextRole.CAPTION.value
                    role_origin = "content_pattern"
                    caption_chain_start_y = span_y
                    caption_chain_start_font_size = font_size
                    caption_chain_length = 1

            # Caption continuation check — guarded
            elif previous_role == TextRole.CAPTION.value:
                y_diff = abs(span_y - previous_y)
                x_diff = abs(span_x - previous_x)

                # Chain metrics
                total_chain_distance = (
                    span_y - caption_chain_start_y
                    if caption_chain_start_y is not None else 0
                )
                font_diff = (
                    abs(font_size - caption_chain_start_font_size)
                    if caption_chain_start_font_size is not None else 0
                )

                # Guards
                guard_y_close = y_diff < caption_y_threshold
                guard_x_aligned = x_diff < caption_x_threshold  # NEW: Prevents column jumping
                guard_font_match = font_diff < _ROLE_CAPTION_FONT_TOLERANCE
                guard_chain_length = caption_chain_length < _ROLE_CAPTION_MAX_CHAIN_LINES
                guard_chain_distance = total_chain_distance < max_caption_chain_distance
                guard_not_new_section = not (
                        (
                                previous_text.rstrip().endswith(".") and
                                span_text[0].isupper() and
                                y_diff > estimated_line_height * _ROLE_CAPTION_NEW_SECTION_GAP_MULTIPLIER
                        ) or
                        y_diff > estimated_line_height * _ROLE_CAPTION_LARGE_GAP_MULTIPLIER
                )

                if all([
                    guard_y_close,
                    guard_x_aligned,  # NEW
                    guard_font_match,
                    guard_chain_length,
                    guard_chain_distance,
                    guard_not_new_section
                ]):
                    role = TextRole.CAPTION.value
                    role_origin = "continuity"
                    caption_chain_length += 1
                else:
                    # Chain broken
                    caption_chain_start_y = None
                    caption_chain_start_font_size = None
                    caption_chain_length = 0

                    if trace_id:
                        logger.debug(
                            "[%s] Caption chain BROKEN at '%s' "
                            "(y_diff=%.1f, font_diff=%.1f, chain_len=%d)",
                            trace_id, span_text[:25], y_diff, font_diff, caption_chain_length
                        )
            else:
                # Reset chain tracking if not continuing
                caption_chain_start_y = None
                caption_chain_start_font_size = None
                caption_chain_length = 0

        # =====================================================================
        # PRIORITY 5: Sidebar
        # =====================================================================
        if role == TextRole.BODY.value:
            if span.get("is_margin_content", False):
                if char_count > _ROLE_SIDEBAR_MIN_CHARS:
                    role = TextRole.SIDEBAR.value
                    role_origin = "geometry"

        # =====================================================================
        # PRIORITY 6: Heading — Structural Evidence Framework
        #
        # A span is classified as heading only when it satisfies textual
        # heading constraints AND exhibits at least one structural evidence
        # source. Evidence sources are boolean gates, not scores.
        #
        # Evidence sources (enumerated, finite):
        #   Typographic emphasis — font size or weight above baseline
        #   Structural prefix   — section numbers, keywords, legal symbols
        #   Layout separation   — vertical gap from prior body text
        #                         (subordinate: requires is_title_case)
        #
        # Three classification paths, each with full guard stacks.
        # Paths are independent — any single path firing classifies the
        # span as heading. Separate paths preserve relaxed_length for
        # visual headings and role_origin differentiation for tracing.
        # =====================================================================
        if role == TextRole.BODY.value:

            # ─────────────────────────────────────────────────────────
            # SHARED TEXTUAL GUARDS
            # Every classification path requires these properties.
            # ─────────────────────────────────────────────────────────
            is_large = font_size >= baseline_font_size * _ROLE_HEADING_FONT_RATIO
            is_bold = any(
                hint in font_name
                for hint in _HEADING_FONT_WEIGHT_HINTS
            )
            # Extended font weight: "medium" qualifies ONLY with
            # structural prefix evidence. Prevents body fonts with
            # "medium" in the name from triggering heading detection.
            is_conditional_weight = (
                    has_structural_prefix and any(
                hint in font_name
                for hint in _HEADING_FONT_WEIGHT_CONDITIONAL_HINTS
            )
            )
            is_short = (
                    word_count <= _ROLE_HEADING_MAX_WORDS and
                    char_count <= _ROLE_HEADING_MAX_CHARS
            )
            is_title_case = span_text and (
                    span_text[0].isupper() or span_text.isupper()
            )
            no_terminal_punct = span_text and (
                    span_text[-1] not in _ROLE_TERMINAL_PUNCTUATION
            )

            # ─────────────────────────────────────────────────────────
            # EVIDENCE: Typographic emphasis
            # ─────────────────────────────────────────────────────────
            has_typographic_emphasis = (
                    is_large or is_bold or is_conditional_weight
            )
            relaxed_length = word_count <= (_ROLE_HEADING_MAX_WORDS * 2.0)

            # GUARD: Lowercase start disqualifies unless all caps
            starts_like_sentence = span_text and span_text[0].islower()
            if starts_like_sentence and not span_text.isupper():
                has_typographic_emphasis = False
            elif span_text.isupper() and word_count > _ROLE_HEADING_MAX_WORDS:
                has_typographic_emphasis = False

            # ─────────────────────────────────────────────────────────
            # EVIDENCE: Structural independence
            # Required for structural prefix and layout separation
            # paths. Heading must be at line start and not a
            # continuation fragment.
            # ─────────────────────────────────────────────────────────
            is_structurally_independent = (
                    span.get("span_index_in_line", 0) == 0 and
                    (not span.get("a2_continues_from_previous", False) or _line_is_heading)
            )

            # ─────────────────────────────────────────────────────────
            # PATH 0: A2-chain structural heading inheritance
            #
            # If the a2-chain pre-scan identified this span's chain
            # as matching a structural heading pattern (joined text
            # like "3.3 Font Size"), promote to heading. Rescues bare
            # numeric anchors that individually fail is_title_case and
            # has_structural_prefix but are validated by chain context.
            # ─────────────────────────────────────────────────────────
            if (_line_is_heading and is_short and no_terminal_punct
                    and is_structurally_independent
                    and any("heading_escape_from_figure" in r
                            for r in span.get("_exclusion_protection_reasons", []))):
                role = TextRole.HEADING.value
                role_origin = "content_pattern:a2_chain_structural"

            # ─────────────────────────────────────────────────────────
            # PATH 1: Typographic emphasis (Original + Prong B)
            #
            # Font-based heading detection. Allows relaxed length
            # (up to 2× max words) when typographic signal is strong.
            # ─────────────────────────────────────────────────────────
            if (is_short or (
                    has_typographic_emphasis and relaxed_length
            )) and is_title_case and no_terminal_punct:
                if has_typographic_emphasis:
                    role = TextRole.HEADING.value
                    role_origin = "content_pattern"

            # ─────────────────────────────────────────────────────────
            # PATH 2: Structural prefix detection
            #
            # Section numbers ("1 Introduction", "3.2 White Space"),
            # keyword prefixes ("Section 1", "Article 2", "Chapter 3"),
            # or legal symbols ("§ 1.1") classify as heading WITHOUT
            # requiring typographic signals.
            #
            # Bypasses: has_typographic_emphasis, is_title_case
            #   (pattern confirms structural prefix — digit-leading
            #   spans fail the standard isupper() check).
            # Requires: is_short, no_terminal_punct, structural
            #   independence.
            # ─────────────────────────────────────────────────────────
            if role == TextRole.BODY.value:
                if (has_structural_prefix and is_short
                        and no_terminal_punct
                        and is_structurally_independent):
                    role = TextRole.HEADING.value
                    role_origin = "content_pattern:structural_prefix"

            # ─────────────────────────────────────────────────────────
            # PATH 3: Layout separation detection
            #
            # Significant vertical whitespace above the last body-
            # classified span, combined with heading-like textual
            # properties, classifies as heading WITHOUT requiring
            # typographic signals or structural prefixes.
            #
            # SUBORDINATION CONSTRAINT: This path requires is_title_case
            # in addition to all other textual guards. Structural prefix
            # (Path 2) does NOT require is_title_case because the
            # pattern itself confirms structural intent. Layout
            # separation is a weaker signal — title case is the
            # additional evidence that prevents false promotion of
            # body fragments after paragraph-level vertical gaps.
            #
            # Gap measured from prev_body_y (last body-classified
            # span's Y), not previous_y, to avoid corruption from
            # intervening figures, captions, or empty spans.
            #
            # Covers: "References", "Acknowledgements", "Appendix",
            # "Terms and Conditions", legal section breaks without
            # numbering, policy documents with spacing-only headers.
            # ─────────────────────────────────────────────────────────
            if role == TextRole.BODY.value:
                if (has_vertical_isolation and is_short
                        and is_title_case and no_terminal_punct
                        and is_structurally_independent):
                    role = TextRole.HEADING.value
                    role_origin = "content_pattern:layout_separation"

        # =====================================================================
        # PRIORITY 7: Subheading
        # =====================================================================
        if role == TextRole.BODY.value:
            is_medium = font_size >= baseline_font_size * _ROLE_SUBHEADING_FONT_RATIO
            is_short = (
                    word_count <= _ROLE_HEADING_MAX_WORDS and
                    char_count <= _ROLE_HEADING_MAX_CHARS
            )
            is_title_case = span_text and span_text[0].isupper()
            is_italic = any(hint in font_name for hint in ("italic", "oblique"))

            # =================================================================
            # PHASE 3.0: Inline span protection
            # Subheadings must be structurally independent, not inline emphasis.
            # A span is inline if it appears mid-line (span_index_in_line > 0).
            # This prevents italic technical terms like "Meissner corpuscles"
            # from being promoted to subheading.
            # =================================================================
            is_inline_span = (
                    span.get("span_index_in_line", 0) > 0 or
                    span.get("a2_continues_to_next", False)
            )
            is_structurally_independent = not is_inline_span

            if is_medium and is_short and is_title_case and is_structurally_independent:
                role = TextRole.SUBHEADING.value
                role_origin = "content_pattern"
            elif is_italic and is_short and is_title_case and word_count >= 2:
                if is_structurally_independent:
                    role = TextRole.SUBHEADING.value
                    role_origin = "content_pattern"
        # =====================================================================
        # PRIORITY 8: Footnote
        # =====================================================================
        if role == TextRole.BODY.value:
            is_small = font_size < baseline_font_size * _ROLE_FOOTNOTE_FONT_RATIO
            is_at_bottom = span_y > footer_zone_y

            if is_small and is_at_bottom:
                role = TextRole.FOOTNOTE.value
                role_origin = "geometry"
            elif char_count <= 2 and font_size < baseline_font_size * _ROLE_FOOTNOTE_MARKER_FONT_RATIO:
                if span_text.isdigit() or span_text in _ROLE_FOOTNOTE_MARKER_SYMBOLS:
                    role = TextRole.FOOTNOTE_MARKER.value
                    role_origin = "geometry"

        # PAGE NUMBER detection (numeric-only, centered, header/footer zones)
        if role == TextRole.BODY.value:
            if (
                    span_text.isdigit() and
                    char_count <= 4 and
                    (
                            span_y < header_zone_y or
                            span_y > footer_zone_y
                    )
            ):
                role = TextRole.PAGE_NUMBER.value
                role_origin = "geometry"

        # =====================================================================
        # PRIORITY 9: List Item
        # =====================================================================
        if role == TextRole.BODY.value:
            if _LIST_ITEM_PATTERN.match(span_text):
                role = TextRole.LIST_ITEM.value
                role_origin = "content_pattern"

        # =====================================================================
        # PRIORITY 10: Code
        # =====================================================================
        if role == TextRole.BODY.value:
            is_monospace = any(hint in font_name for hint in CODE_FONT_HINTS)

            if is_monospace and "symbol" not in font_name and "wingdings" not in font_name:
                role = TextRole.CODE.value
                role_origin = "content_pattern"
            elif char_count > _ROLE_CODE_MIN_CHAR_COUNT:
                code_chars = _ROLE_CODE_PUNCT_PATTERN.findall(span_text)
                if len(code_chars) / char_count > _ROLE_CODE_PUNCT_RATIO:
                    role = TextRole.CODE.value
                    role_origin = "content_pattern"

        # PHASE 0: CODE veto for punctuation-only spans
        # Prevents body punctuation fragments from being dropped as non-viable CODE.
        if role == TextRole.CODE.value:
            alpha_count = sum(1 for c in span_text if c.isalpha())
            # Veto CODE if span has words (>3 letters) - clearly prose, not code
            if alpha_count >= 3:
                role = TextRole.BODY.value

        # =====================================================================
        # PRIORITY 11: Hyperlink
        # =====================================================================
        if role == TextRole.BODY.value:
            if span_text in link_uris or span_text.startswith(("http", "www.")):
                role = TextRole.HYPERLINK.value
                role_origin = "content_pattern"

        # =====================================================================
        # PRIORITY 12: Inline Equation
        # =====================================================================
        if role == TextRole.BODY.value:
            if _is_inline_equation(span_text, span):
                role = TextRole.INLINE_EQUATION.value
                role_origin = "content_pattern"

        # =====================================================================
        # PRIORITY 13: Subscript / Superscript
        # =====================================================================
        if role == TextRole.BODY.value:
            is_very_small = font_size < baseline_font_size * _ROLE_VERY_SMALL_FONT_RATIO

            if is_very_small:
                if span.get("is_subscript"):
                    role = TextRole.SUBSCRIPT.value
                    role_origin = "content_pattern"
                    span["merge_with_adjacent"] = True
                elif span.get("flags", 0) & PyMuPDFFlag.SUPERSCRIPT:
                    role = TextRole.SUPERSCRIPT.value
                    role_origin = "content_pattern"
                    span["merge_with_adjacent"] = True

        # =====================================================================
        # PRIORITY 14: Header Artifact Detection
        # =====================================================================
        if role == TextRole.BODY.value:
            is_in_header_zone = span_y < header_zone_y
            is_short_fragment = (
                    char_count < _ROLE_SHORT_FRAGMENT_CHAR_COUNT and
                    word_count <= _ROLE_SHORT_FRAGMENT_WORD_COUNT
            )
            is_uppercase = span_text.isupper()

            if is_in_header_zone and is_short_fragment:
                if _is_orphan_fragment(span_text):
                    role = TextRole.HEADER_ARTIFACT.value
                    role_origin = "geometry"
                elif is_uppercase and not _is_protected_acronym(span_text):
                    if char_count < _ROLE_VERY_SHORT_UPPERCASE_THRESHOLD:
                        if not any(c in span_text for c in ".,;:"):
                            role = TextRole.HEADER_ARTIFACT.value
                            role_origin = "geometry"

        # =====================================================================
        # ASSIGN ROLE AND UPDATE STATE
        # =====================================================================
        if role in {
            TextRole.FIGURE_LABEL.value,
            TextRole.INLINE_EQUATION.value,
            TextRole.SUBSCRIPT.value,
            TextRole.SUPERSCRIPT.value,
            TextRole.CAPTION.value,  # Added
            TextRole.LIST_ITEM.value,  # Added
        }:
            span["sentence_continuation_ok"] = True

        span["role"] = role
        span["_role_origin"] = role_origin

        previous_role = role
        previous_y = span_y
        previous_x = span_x
        previous_text = span_text

        # Path 3 state: track last body span's Y for isolation measurement
        if role == TextRole.BODY.value:
            prev_body_y = span_y

        if trace_id and role not in (TextRole.BODY.value, TextRole.EMPTY.value):
            logger.debug(
                "[%s] Role: '%s' -> %s (font=%.1f, baseline=%.1f)",
                trace_id, span_text[:25], role, font_size, baseline_font_size
            )
    # =========================================================================
    # POST-LOOP: Promote orphaned section numbers to heading
    #
    # Catches digit-only body spans (e.g., "3.5", "3.6", "1") that chain
    # forward via a2_continues_to_next to a heading-classified successor
    # but were missed by the pre-loop A2 chain scan due to unidirectional
    # A2 link (successor lacks a2_continues_from_previous).
    #
    # Runs AFTER main loop so successor roles are finalized.
    # Guards: digit-only text, forward A2 link, successor=heading, same stream,
    # and must be start-of-line (prevents inline numeric citations).
    # =========================================================================
    for i, span in enumerate(spans):
        if span.get("role") != TextRole.BODY.value:
            continue

        text = (span.get("cleaned_text") or "").strip()
        if not text or not text.replace(".", "").isdigit():
            continue

        if span.get("span_index_in_line", 0) != 0:
            continue

        if not span.get("a2_continues_to_next", False):
            continue

        if i + 1 >= len(spans):
            continue

        next_sp = spans[i + 1]
        if next_sp.get("role") != TextRole.HEADING.value:
            continue

        if span.get("layout_stream") != next_sp.get("layout_stream"):
            continue

        span["role"] = TextRole.HEADING.value
        span["_role_origin"] = "content_pattern:a2_chain_structural"

        # Cleanup: remove transient a2-chain heading markers (prevent artifact leakage)
    for span in spans:
        span.pop("_a2_chain_is_heading", None)


def _apply_continuity_role_resolution(
        spans: List[Dict],
        trace_id: str = None
) -> int:
    """
    PHASE 1.5: Apply continuity-aware role resolution.

    Fixes sentence fragmentation at figure boundaries by preserving
    spans that are part of continuous body text flow, even when
    geometric classification marks them as inside_figure.

    PRINCIPLE: Stream continuity trumps geometric classification.

    This is Phase 1 of the stream-first architecture migration:
    - Introduces continuity context concept
    - Documents the "stream integrity" principle
    - Creates foundation for full stream-first implementation

    Safety guards (per lead review):
    - Requires lowercase continuation (no short-span shortcut)
    - Enforces physical adjacency (block_id or Y-gap)
    - Hard veto for known label/metadata patterns
    - Explicit reading-order sort before processing

    Args:
        spans: List of span dicts (mutated in place)
        trace_id: Optional trace ID for logging

    Returns:
        Count of spans with role overrides applied.

    Mutations:
        For overridden spans:
        - role: Changed to "body"
        - _continuity_override: True
        - _continuity_override_reason: Detailed explanation
        - _original_geometry_role: Original role before override
    """
    if not spans or len(spans) < 2:
        return 0

    # GUARD: Ensure reading order before continuity analysis
    # Reading-order invariant:
    # Continuity must operate on the same linear order as reconstruction
    spans.sort(key=lambda s: (
        s.get("page_number", 0),
        s.get("block_id", 0),
        s.get("line_index", 0),
        s.get("span_index_in_line", 0),
    ))

    override_count = 0

    for i, span in enumerate(spans):
        current_role = span.get("role", TextRole.BODY.value)

        # Fast path: Only process override candidates
        if current_role not in _CONTINUITY_OVERRIDE_CANDIDATES:
            continue

        # Fast path: Only override if span is in a body column
        layout_stream = span.get("layout_stream", "")
        if not layout_stream.startswith("body_col"):
            continue

        # GUARD: Hard semantic veto for known non-body patterns
        span_text = (span.get("cleaned_text") or span.get("raw_text", "")).strip()
        if _matches_continuity_veto_pattern(span_text):
            if trace_id:
                logger.debug(
                    "[%s] Phase 1.5: Veto override for pattern match: '%s...'",
                    trace_id, span_text[:30]
                )
            continue

        # Build continuity context with adjacency checks
        previous_span = spans[i - 1] if i > 0 else None
        next_span = spans[i + 1] if i < len(spans) - 1 else None

        continues_from = _check_continuity_from_previous(previous_span, span)
        continues_to = _check_continuity_to_next(span, next_span)

        # Additional semantic continuity signal (post-window, post-RONC)
        # This field only exists after Stage 2.8; safe to check with .get()
        semantic_included = span.get("_semantic_disposition") == "included"

        # Decision: Override if stream continuity is established
        # Continuity sources (ordered by strength):
        # 1. Local physical adjacency (available at Stage 2.4)
        # 2. Semantic inclusion (available after Stage 2.8 second pass)
        #
        # GUARD (v10.0): Non-viable roles (inside_figure, figure_label,
        # etc.) require physical continuity evidence for promotion.
        # semantic_included alone is insufficient — it can originate
        # from RONC proximity scoring without confirming text-flow
        # membership. This prevents author metadata or isolated figure
        # labels from being promoted to body via RONC mutual links.
        #
        # Spans with physical continuity (lowercase start, non-terminal
        # predecessor, adjacency) still promote correctly — this only
        # restricts the semantic_included-only path for non-viable roles.
        if current_role in _TTS_NON_VIABLE_ROLES:
            in_active_flow = continues_from or continues_to
        else:
            in_active_flow = continues_from or continues_to or semantic_included

        if in_active_flow:
            # CONTINUITY WINS: Preserve stream integrity
            span["_continuity_override"] = True
            # NEW: Audit which pass performed the override
            span["_continuity_override_stage"] = (
                "post_semantic" if span.get("_semantic_disposition") == "included"
                else "pre_semantic"
            )
            span["_continuity_override_reason"] = (
                f"stream={layout_stream}, "
                f"from_prev={continues_from}, "
                f"to_next={continues_to}"
            )
            span["_original_geometry_role"] = current_role
            span["role"] = TextRole.BODY.value
            override_count += 1

            if trace_id:
                logger.debug(
                    "[%s] Phase 1.5: Continuity override %s → body: '%s...'",
                    trace_id, current_role, span_text[:40]
                )

    if trace_id and override_count > 0:
        logger.info(
            "[%s] Phase 1.5: Continuity role resolution overrode %d spans",
            trace_id, override_count
        )

    return override_count


def _matches_continuity_veto_pattern(text: str) -> bool:
    """
    Check if text matches known non-body patterns that should never be
    promoted to body role regardless of continuity signals.

    Prevents label/metadata leakage into TTS output.

    Args:
        text: Span text to check

    Returns:
        True if text matches a veto pattern (should NOT be overridden).
    """
    if not text:
        return False

    text_lower = text.lower().strip()

    # Check against veto patterns
    for pattern in _CONTINUITY_VETO_PATTERNS:
        if text_lower.startswith(pattern):
            return True

    # Additional check: Pure numeric/reference patterns (e.g., "[1]", "(a)", "1.")
    if len(text_lower) <= 5:
        # Very short + starts with bracket/paren/digit = likely reference
        if text_lower[0] in "[(0123456789":
            return True

    return False


def _check_continuity_from_previous(
        previous: Optional[Dict],
        current: Dict
) -> bool:
    """
    Check if current span continues from previous span.

    Continuation requires ALL of:
    1. Previous span exists and is body-like
    2. Previous text doesn't end with terminal punctuation (.!?)
    3. Current text starts lowercase (REQUIRED - no short-span shortcut)
    4. Both spans in body column family
    5. Physical adjacency (same block OR Y-gap within threshold)

    Returns:
        True if continuation is confirmed.
    """
    if not previous:
        return False

    # Get text
    prev_text = (previous.get("cleaned_text") or previous.get("raw_text", "")).strip()
    curr_text = (current.get("cleaned_text") or current.get("raw_text", "")).strip()

    if not prev_text or not curr_text:
        return False

    # REQUIREMENT 1: Previous role must be body-like
    prev_role = previous.get("role", TextRole.BODY.value)
    prev_is_body_like = prev_role in (
        TextRole.BODY.value,
        TextRole.HEADING.value,
        TextRole.INSIDE_FIGURE.value,  # May also be misclassified
        TextRole.FIGURE_LABEL.value,
    )
    if not prev_is_body_like:
        return False

    # REQUIREMENT 2: Both in body column family
    prev_stream = previous.get("layout_stream", "")
    curr_stream = current.get("layout_stream", "")
    both_body_streams = (
            prev_stream.startswith("body_col") and
            curr_stream.startswith("body_col")
    )
    if not both_body_streams:
        return False

    # REQUIREMENT 3: Physical adjacency (same block OR close Y-gap)
    # Prevents "role contagion" across disconnected figure blocks
    same_block = previous.get("block_id") == current.get("block_id")

    if not same_block:
        # Check Y-gap as fallback for cross-block but adjacent content
        prev_bbox = previous.get("bbox", [0, 0, 0, 0])
        curr_bbox = current.get("bbox", [0, 0, 0, 0])
        y_gap = curr_bbox[1] - prev_bbox[3]  # Current top - previous bottom

        if y_gap > _CONTINUITY_MAX_Y_GAP:
            return False

    # REQUIREMENT 4: Previous doesn't end with terminal punctuation
    # NOTE: Only .!? are terminal; : and ; are NOT (per lead review)
    ends_terminal = prev_text[-1] in _CONTINUITY_TERMINAL_CHARS
    if ends_terminal:
        return False

    # REQUIREMENT 5: Current starts lowercase (STRICT - no is_short shortcut)
    # This prevents promoting uppercase labels like "Figure 3:" even when short
    starts_lower = curr_text[0].islower()
    if not starts_lower:
        return False

    return True


def _check_continuity_to_next(
        current: Dict,
        next_span: Optional[Dict]
) -> bool:
    """
    Check if current span continues to next span.

    Continuation requires ALL of:
    1. Next span exists
    2. Current text doesn't end with terminal punctuation (.!?)
    3. Next text starts lowercase (REQUIRED)
    4. Both spans in body column family
    5. Physical adjacency (same block OR Y-gap within threshold)

    Returns:
        True if continuation is confirmed.
    """
    if not next_span:
        return False

    # Get text
    curr_text = (current.get("cleaned_text") or current.get("raw_text", "")).strip()
    next_text = (next_span.get("cleaned_text") or next_span.get("raw_text", "")).strip()

    if not curr_text or not next_text:
        return False

    # REQUIREMENT 1: Both in body column family
    curr_stream = current.get("layout_stream", "")
    next_stream = next_span.get("layout_stream", "")
    both_body_streams = (
            curr_stream.startswith("body_col") and
            next_stream.startswith("body_col")
    )
    if not both_body_streams:
        return False

    # REQUIREMENT 2: Physical adjacency (same block OR close Y-gap)
    same_block = current.get("block_id") == next_span.get("block_id")

    if not same_block:
        curr_bbox = current.get("bbox", [0, 0, 0, 0])
        next_bbox = next_span.get("bbox", [0, 0, 0, 0])
        y_gap = next_bbox[1] - curr_bbox[3]  # Next top - current bottom

        if y_gap > _CONTINUITY_MAX_Y_GAP:
            return False

    # REQUIREMENT 3: Current doesn't end with terminal punctuation
    ends_terminal = curr_text[-1] in _CONTINUITY_TERMINAL_CHARS
    if ends_terminal:
        return False

    # REQUIREMENT 4: Next starts lowercase (STRICT)
    starts_lower = next_text[0].islower()
    if not starts_lower:
        return False

    return True


# PHASE 2.0: Sliding Window Functions (Two-Pass Architecture)
def _build_sliding_window_spans(
        page_span_cache: Dict[int, List[Dict]],
        current_page_idx: int,
        trace_id: str = None
) -> Tuple[List[Dict], Tuple[int, int]]:
    """
    Build a sliding window of spans for cross-page aware segmentation.
        - Uses PRE-PASS cached spans to ensure all spans have identical viability
    processing
        - Window is LOSSLESS. Sidebar/margin spans are  tagged with _stage1_nonviable_hint=True and 
    included.
    Window structure:
    - Last N spans from previous page (deep copied, tagged)
    - ALL spans from current page (deep copied, tagged with _page_local_idx)
    - First N spans from next page (deep copied, tagged)
    - All spans tagged with _source_page_idx for attribution
    - Current page spans tagged with _page_local_idx for remapping
    Args:
        page_span_cache: Dict mapping page_idx → processed spans_for_text
        current_page_idx: Index of current page being processed
        trace_id: Optional trace ID for logging
    Returns: Tuple of:
        - window_spans: Combined span list for the window (deep copies)
        - page_span_range: (start_idx, end_idx) marking current page in window
    """

    def _copy_span_for_window(source_span: Dict, source_page_idx: int,
                              source_local_idx: int) -> Dict:
        copied = copy.deepcopy(source_span)

        copied["_requires_semantic_review"] = bool(
            copied.get("_requires_semantic_review", False)
        )

        # Prefer upstream CID if present; otherwise generate stable page-local CID
        # Canonical format MUST match the rest of pipeline: P{page}:{idx}
        if "_canonical_span_id" not in copied:
            copied["_canonical_span_id"] = f"P{source_page_idx}:{source_local_idx}"

        return copied

    def _get_same_line_body_anchor_stream(span: Dict, all_spans: List[Dict]) -> Optional[str]:
        """
        Check if margin span shares a line with a body_col span.

        Mid-line italic text can have high x0 coordinates causing misclassification
        as margin_right. If the span shares a line_id with a body_col span,
        return that body stream for promotion instead of marking nonviable.

        Returns None if zero or multiple distinct body_col streams exist
        on the line, to avoid ambiguous promotion in multi-column layouts.
        Returns: body_col stream name if exactly one body stream on line, None otherwise.
        """
        line_id = span.get("line_id")
        if not line_id:
            return None

        body_streams_on_line = set()
        for other in all_spans:
            if other is span:
                continue
            if other.get("line_id") != line_id:
                continue
            other_stream = other.get("layout_stream") or ""
            if other_stream.startswith("body_col"):
                body_streams_on_line.add(other_stream)

        # Gate: exactly one body stream prevents ambiguity in multi-column layouts
        if len(body_streams_on_line) == 1:
            return list(body_streams_on_line)[0]

        return None

    window_spans: List[Dict] = []
    current_spans = page_span_cache.get(current_page_idx, [])

    # Get page number from first span (or 0 if empty)
    current_page_num = 0
    if current_spans:
        current_page_num = current_spans[0].get("page_number", current_page_idx + 1)

    # =========================================================================
    # PREVIOUS PAGE TAIL (deep copy, tagged)
    # =========================================================================
    prev_tail_count = 0
    nonviable_prev_count = 0
    if current_page_idx > 0:
        prev_spans = page_span_cache.get(current_page_idx - 1, [])
        if prev_spans:
            tail_spans = prev_spans[-_WINDOW_TAIL_SPAN_COUNT:]
            tail_start_idx = len(prev_spans) - len(tail_spans)
            for idx, span in enumerate(tail_spans):
                role = span.get("role", "")
                layout_stream = span.get("layout_stream", "")

                # Never skip, tag instead
                is_sidebar_or_margin = (
                        role == TextRole.SIDEBAR.value
                        or (isinstance(layout_stream, str) and layout_stream.startswith("margin"))
                )

                prev_local_idx = tail_start_idx + idx
                span_copy = _copy_span_for_window(span, current_page_idx - 1, prev_local_idx)

                # FIX v9.0 PHASE 1: Always preserve original layout provenance
                if "_original_layout_stream" not in span_copy:
                    span_copy["_original_layout_stream"] = span.get("layout_stream")
                if "_original_role" not in span_copy:
                    span_copy["_original_role"] = span.get("role")

                # Handle both unpromoted margins AND "Zombie" spans
                is_zombie = (role == TextRole.SIDEBAR.value and str(layout_stream).startswith(
                    "body_col"))
                already_promoted = span.get("_tts_promoted_to_body_stream", False)

                if is_sidebar_or_margin and (not already_promoted or is_zombie):
                    # Check for same-line body anchor before marking nonviable
                    body_anchor_stream = _get_same_line_body_anchor_stream(span, prev_spans)
                    if body_anchor_stream:
                        span_copy["_original_layout_stream"] = span_copy.get("layout_stream")
                        span_copy["layout_stream"] = body_anchor_stream
                        # Force role to body so it merges with neighbors
                        # SAFEGUARD: Only convert 'sidebar' (ambiguous); preserve captions/headers.
                        if span_copy.get("role") == TextRole.SIDEBAR.value:
                            span_copy["_original_role"] = span_copy.get("role")
                            span_copy["role"] = "body"
                        span_copy["_same_line_promoted"] = True
                    elif is_zombie:
                        # Zombie already has body_col stream; fix role only
                        if span_copy.get("role") == TextRole.SIDEBAR.value:
                            span_copy["_original_role"] = span_copy.get("role")
                            span_copy["role"] = "body"
                        span_copy["_zombie_role_fixed"] = True
                    else:
                        span_copy["_stage1_nonviable_hint"] = True
                        nonviable_prev_count += 1
                span_copy["_window_position"] = "prev_tail"
                span_copy["_source_page_idx"] = current_page_idx - 1
                window_spans.append(span_copy)
                prev_tail_count += 1

    # =========================================================================
    # CURRENT PAGE (deep copy, tagged with _page_local_idx)
    # =========================================================================
    page_start_idx = len(window_spans)
    nonviable_current_count = 0

    for page_local_idx, span in enumerate(current_spans):
        role = span.get("role", "")
        layout_stream = span.get("layout_stream", "")

        # ─────────────────────────────────────────────────────────────────────
        # SCHEMA v2.0: Lossless window — never skip, tag instead
        # ─────────────────────────────────────────────────────────────────────
        is_sidebar_or_margin = (
                role == TextRole.SIDEBAR.value
                or (isinstance(layout_stream, str) and layout_stream.startswith("margin"))
        )

        span_copy = _copy_span_for_window(span, current_page_idx, page_local_idx)

        # FIX v9.0 PHASE 1: Always preserve original layout provenance
        # This ensures downstream semantic gating can detect margin-origin spans
        # even when they were not explicitly promoted.
        if "_original_layout_stream" not in span_copy:
            span_copy["_original_layout_stream"] = span.get("layout_stream")
        if "_original_role" not in span_copy:
            span_copy["_original_role"] = span.get("role")

        # P6 INTEGRATED FIX: Handle both unpromoted margins AND "Zombie" spans
        # Zombie = Patch 9B promoted stream but not role (sidebar + body_col)
        is_zombie = (role == TextRole.SIDEBAR.value and str(layout_stream).startswith("body_col"))
        already_promoted = span.get("_tts_promoted_to_body_stream", False)

        if is_sidebar_or_margin and (not already_promoted or is_zombie):
            # Check for same-line body anchor before marking nonviable
            body_anchor_stream = _get_same_line_body_anchor_stream(span, current_spans)
            if body_anchor_stream:
                span_copy["_original_layout_stream"] = span_copy.get("layout_stream")
                span_copy["layout_stream"] = body_anchor_stream
                # Force role to body so it merges with neighbors
                # SAFEGUARD: Only convert 'sidebar' (ambiguous); preserve captions/headers.
                if span_copy.get("role") == TextRole.SIDEBAR.value:
                    span_copy["_original_role"] = span_copy.get("role")
                    span_copy["role"] = "body"
                span_copy["_same_line_promoted"] = True
            elif is_zombie:
                # HARDENING: Zombie already has body_col stream; fix role only
                if span_copy.get("role") == TextRole.SIDEBAR.value:
                    span_copy["_original_role"] = span_copy.get("role")
                    span_copy["role"] = "body"
                span_copy["_zombie_role_fixed"] = True
            else:
                span_copy["_stage1_nonviable_hint"] = True
                nonviable_current_count += 1
        span_copy["_window_position"] = "current"
        span_copy["_source_page_idx"] = current_page_idx
        span_copy["_page_local_idx"] = page_local_idx
        window_spans.append(span_copy)

    page_end_idx = len(window_spans)

    # =========================================================================
    # NEXT PAGE HEAD (deep copy, tagged)
    # =========================================================================
    next_head_count = 0
    nonviable_next_count = 0
    if current_page_idx + 1 in page_span_cache:
        next_spans = page_span_cache.get(current_page_idx + 1, [])
        if next_spans:
            head_spans = next_spans[:_WINDOW_HEAD_SPAN_COUNT]
            for idx, span in enumerate(head_spans):
                role = span.get("role", "")
                layout_stream = span.get("layout_stream", "")

                # ─────────────────────────────────────────────────────────────
                # SCHEMA v2.0: Lossless window — never skip, tag instead
                # ─────────────────────────────────────────────────────────────
                is_sidebar_or_margin = (
                        role == TextRole.SIDEBAR.value
                        or (isinstance(layout_stream, str) and layout_stream.startswith("margin"))
                )

                span_copy = _copy_span_for_window(span, current_page_idx + 1, idx)

                # FIX v9.0 PHASE 1: Always preserve original layout provenance
                if "_original_layout_stream" not in span_copy:
                    span_copy["_original_layout_stream"] = span.get("layout_stream")
                if "_original_role" not in span_copy:
                    span_copy["_original_role"] = span.get("role")

                # P6 INTEGRATED FIX: Handle both unpromoted margins AND "Zombie" spans
                # Zombie = Patch 9B promoted stream but not role (sidebar + body_col)
                is_zombie = (role == TextRole.SIDEBAR.value and str(layout_stream).startswith(
                    "body_col"))
                already_promoted = span.get("_tts_promoted_to_body_stream", False)

                if is_sidebar_or_margin and (not already_promoted or is_zombie):
                    # Check for same-line body anchor before marking nonviable
                    body_anchor_stream = _get_same_line_body_anchor_stream(span, next_spans)
                    if body_anchor_stream:
                        span_copy["_original_layout_stream"] = span_copy.get("layout_stream")
                        span_copy["layout_stream"] = body_anchor_stream
                        # Force role to body so it merges with neighbors
                        # SAFEGUARD: Only convert 'sidebar' (ambiguous); preserve captions/headers.
                        if span_copy.get("role") == TextRole.SIDEBAR.value:
                            span_copy["_original_role"] = span_copy.get("role")
                            span_copy["role"] = "body"
                        span_copy["_same_line_promoted"] = True
                    elif is_zombie:
                        # HARDENING: Zombie already has body_col stream; fix role only
                        if span_copy.get("role") == TextRole.SIDEBAR.value:
                            span_copy["_original_role"] = span_copy.get("role")
                            span_copy["role"] = "body"
                        span_copy["_zombie_role_fixed"] = True
                    else:
                        span_copy["_stage1_nonviable_hint"] = True
                        nonviable_next_count += 1
                span_copy["_window_position"] = "next_head"
                span_copy["_source_page_idx"] = current_page_idx + 1
                window_spans.append(span_copy)
                next_head_count += 1

    # Count P6 promotions for logging
    promoted_count = sum(1 for sp in window_spans if sp.get("_same_line_promoted"))

    if trace_id:
        total_nonviable = nonviable_prev_count + nonviable_current_count + nonviable_next_count
        logger.debug(
            "[%s] Phase 2.0: Window for page %d: prev_tail=%d, current=%d, next_head=%d, "
            "total=%d, nonviable_hints=%d, same_line_promoted=%d",
            trace_id,
            current_page_num,
            prev_tail_count,
            page_end_idx - page_start_idx,
            next_head_count,
            len(window_spans),
            total_nonviable,
            promoted_count
        )

    return window_spans, (page_start_idx, page_end_idx)


def _filter_sentences_to_page(
        window_sentences: List[Dict],
        window_spans: List[Dict],
        page_span_range: Tuple[int, int],
        current_page_idx: int,
        page_num: int,
        trace_id: str = None
) -> List[Dict]:
    """
    PHASE 2.0: Filter window sentences to those belonging to current page.

    Attribution rule: A sentence belongs to the page where it STARTS.
    This prevents duplication when adjacent windows overlap.

    Remaps window span indices to page-local indices using the _page_local_idx
    tag set during window construction. This is robust against any future
    changes to window construction logic.

    Args:
        window_sentences: All sentences from window segmentation
        window_spans: The full window span list (with tags)
        page_span_range: (start_idx, end_idx) of current page within window
        current_page_idx: Index of current page (for validation)
        page_num: Current page number (for sentence attribution)
        trace_id: Optional trace ID for logging

    Returns:
        List of sentences belonging to this page, with page-local indices.

    Invariants:
        - Only sentences STARTING in current page are returned
        - span_start_index and span_end_index remain WINDOW-space indices (segmentation authority)
        - page-local span indices (debug/analytics only) are stored in:
          _page_local_span_start / _page_local_span_end
        - page_number is correctly set
    """
    page_start, page_end = page_span_range
    page_sentences: List[Dict] = []
    cross_page_context_count = 0

    for sent in window_sentences:
        # Get the starting span index in window coordinates
        window_start_idx = sent.get("span_start_index", -1)

        if window_start_idx < 0 or window_start_idx >= len(window_spans):
            # Invalid index, skip
            if trace_id:
                logger.warning(
                    "[%s] Phase 2.0: Invalid window_start_idx %d, skipping sentence",
                    trace_id, window_start_idx
                )
            continue

        # Get the source span to check attribution
        start_span = window_spans[window_start_idx]
        source_page_idx = start_span.get("_source_page_idx")

        # START-PAGE RULE: Only emit if sentence starts in current page
        if source_page_idx != current_page_idx:
            # Sentence starts in different page - skip (will be captured by that page's window)
            continue

        # Get page-local index from tag (robust against window changes)
        page_local_start = start_span.get("_page_local_idx")
        if page_local_start is None:
            if trace_id:
                logger.warning(
                    "[%s] Phase 2.0: Missing _page_local_idx on span, using fallback",
                    trace_id
                )
            # Fallback to arithmetic (less safe but better than dropping)
            page_local_start = window_start_idx - page_start

        # Handle end index similarly
        window_end_idx = sent.get("span_end_index", window_start_idx)
        if 0 <= window_end_idx < len(window_spans):
            end_span = window_spans[window_end_idx]
            end_source_page = end_span.get("_source_page_idx")

            if end_source_page == current_page_idx:
                # End is also in current page - use its local index
                page_local_end = end_span.get("_page_local_idx", window_end_idx - page_start)
            else:
                # Sentence extends into next page
                # Use last span of current page as end
                page_local_end = page_end - page_start - 1
                sent["_crosses_to_next_page"] = True
                cross_page_context_count += 1
        else:
            page_local_end = page_local_start

        # Create sentence copy with remapped indices
        page_sent = dict(sent)

        # KEEP window-space indices as authoritative
        page_sent["page_number"] = page_num

        # OPTIONAL: retain page-local indices for debugging or analytics only
        page_sent["_page_local_span_start"] = page_local_start
        page_sent["_page_local_span_end"] = page_local_end

        # Track if sentence benefited from window context
        if sent.get("_crosses_to_next_page") or window_start_idx < page_start:
            page_sent["_used_window_context"] = True

        page_sentences.append(page_sent)

    if trace_id and cross_page_context_count > 0:
        logger.info(
            "[%s] Phase 2.0: Page %d: %d sentences used cross-page context",
            trace_id, page_num, cross_page_context_count
        )

    return page_sentences


def _refine_roles_via_content_flow(spans: List[Dict], trace_id: str = None) -> None:
    """
    PHASE X: Content-Flow Outlier Refinement ("Island Logic")
    Detects and demotes orphan artifacts (attribution lines, drifted headers/footers,
    margin notes) using multi-signal combination scoring.

    IMPORTANT PIPELINE CONTRACT:
      - Mutates spans in place by setting role to *non-TTS-viable* roles only.
      - Must run AFTER initial role assignment + block_id assignment
        and BEFORE spans_for_text filtering / sentence segmentation.

    Signals (binary):
      1) boundary: near top/bottom of content extent
      2) small_font: font size below median
      3) gap: block has large vertical gap before it (relative + absolute)
      4) pattern: attribution/meta boilerplate match
      5) orphan: small block (<=2 spans)
    """
    if not spans:
        return

    # ---- A) Collect only bbox-valid spans for geometry stats ----
    geo_spans = [s for s in spans if s.get("bbox") and len(s["bbox"]) == 4]
    if not geo_spans:
        return

    # Content vertical bounds (use y0/y1 to avoid center-only bias)
    y0s = [s["bbox"][1] for s in geo_spans]
    y1s = [s["bbox"][3] for s in geo_spans]
    content_min_y, content_max_y = min(y0s), max(y1s)
    content_height = max(1.0, content_max_y - content_min_y)

    # Median font size (robust default)
    fonts = sorted([float(s.get("font_size", 11.0)) for s in geo_spans])
    median_font = fonts[len(fonts) // 2] if fonts else 11.0

    # Build block stats: y-range + count
    blocks: Dict[int, Dict[str, float]] = {}
    for s in geo_spans:
        bid = int(s.get("block_id", 0))
        b = blocks.get(bid)
        if b is None:
            blocks[bid] = {"y0": s["bbox"][1], "y1": s["bbox"][3], "count": 1}
        else:
            b["y0"] = min(b["y0"], s["bbox"][1])
            b["y1"] = max(b["y1"], s["bbox"][3])
            b["count"] += 1

    # Block gaps: gap before each block, based on sorted block y0
    sorted_bids = sorted(blocks.keys(), key=lambda blk: blocks[blk]["y0"])
    block_gaps: Dict[int, float] = {}
    prev_bid = None
    for bid in sorted_bids:
        if prev_bid is None:
            block_gaps[bid] = 0.0
        else:
            gap = blocks[bid]["y0"] - blocks[prev_bid]["y1"]
            block_gaps[bid] = max(0.0, float(gap))
        prev_bid = bid

    # Median positive gap for relative comparisons
    positive_gaps = sorted([g for g in block_gaps.values() if g > 0.0])
    median_gap = positive_gaps[len(positive_gaps) // 2] if positive_gaps else 20.0

    # Pattern set: keep narrowly attribution/meta (safe for legal/web too)
    # (Extend list, do NOT add new signal types.)
    meta_patterns = (
        "based on", "adapted from", "source:", "courtesy of", "lecture notes",
        "from wikibooks", "licensed under", "all rights reserved", "copyright",
        "retrieved", "doi:", "http://", "https://", "www."
    )

    # Roles that should never be altered by this pass (structural truth sources)
    protected_roles = {
        TextRole.TABLE_CELL.value,
        TextRole.INSIDE_FIGURE.value,
        TextRole.FIGURE_LABEL.value,
        TextRole.CODE.value,
    }

    # Heading-aware bibliography demotion:
    # After a "References"/"Bibliography" heading, treat subsequent content as non-narrative.
    _biblio_heading_roles_to_skip = {
        TextRole.PAGE_NUMBER.value,
        TextRole.HEADER_ARTIFACT.value,
        TextRole.FOOTER_ARTIFACT.value,
        TextRole.CAPTION.value,
    }

    def _sort_key(sp):
        pn = sp.get("page_number", 0)
        cid = str(sp.get("_canonical_span_id") or "")
        try:
            idx = int(cid.split(":")[1])
        except:
            idx = 0
        return (pn, idx)

    in_biblio_section = False
    for span in sorted(spans, key=_sort_key):
        text_raw = (span.get("cleaned_text") or span.get("raw_text") or "").strip()
        if not text_raw:
            continue

        role = span.get("role", TextRole.BODY.value)

        # Heading handling: start/stop bibliography section
        if role == TextRole.HEADING.value:
            if any(p.match(text_raw) for p in _BIBLIO_HEADING_PATTERNS):
                in_biblio_section = True
            elif in_biblio_section:
                # ── Validate before resetting: spurious headings within
                # bibliography (e.g. "Springer, Cham (2022)") must not
                # kill the section flag. Uses module-level patterns only
                # (_biblio_span_score is defined later, not yet callable). ──
                has_biblio_signals = any(
                    p.search(text_raw) for p in _BIBLIO_CONTINUATION_PATTERNS
                )
                text_raw_lower = text_raw.lower()
                has_publisher = any(
                    pub in text_raw_lower
                    for pub in ("springer", "ieee", "acm", "elsevier")
                )
                has_year_pattern = bool(re.search(r'\(\d{4}\)', text_raw))
                if has_biblio_signals or (has_publisher and has_year_pattern):
                    # Spurious heading — demote to footnote, keep flag alive
                    span["role"] = TextRole.FOOTNOTE.value
                    span["_outlier_score"] = 0.90
                    span["_outlier_reasons"] = ["bibliography_heading_spurious"]
                    if trace_id:
                        logger.debug(
                            "[%s] Bibliography: spurious heading demoted, "
                            "section flag preserved. text=%r",
                            trace_id, text_raw[:60]
                        )
                else:
                    in_biblio_section = False
            else:
                in_biblio_section = False
            continue

        if not in_biblio_section:
            continue

        # Skip structural artifacts (page numbers, headers/footers, captions)
        if role in _biblio_heading_roles_to_skip:
            continue

        span["role"] = TextRole.FOOTNOTE.value
        span["_outlier_score"] = 0.90
        span["_outlier_reasons"] = ["bibliography_heading"]

    # ---- A.2) Bibliography-like scoring helper (inline) ----
    def _biblio_span_score(text_raw: str, font_ratio: float) -> float:
        """
        Score bibliography-likelihood for a single span.
        Returns score in [0, 1]. Uses only structural/text-shape signals.
        """
        if not text_raw:
            return 0.0

        text = text_raw.strip()
        if not text:
            return 0.0

        score = 0.0

        # Entry start patterns
        for p in _BIBLIO_ENTRY_PATTERNS:
            if p.match(text):
                score += 0.30
                break

        # Continuation markers (counted, diminishing returns)
        cont_matches = 0
        for p in _BIBLIO_CONTINUATION_PATTERNS:
            if p.search(text):
                cont_matches += 1
        if cont_matches >= 1:
            score += min(0.25, cont_matches * 0.10)

        # Journal abbreviations (high specificity)
        for p in _BIBLIO_JOURNAL_PATTERNS:
            if p.search(text):
                score += 0.20
                break

        # Punctuation density
        punct_count = sum(1 for c in text if c in ".,;:()[]")
        punct_density = punct_count / max(len(text), 1)
        if punct_density > _BIBLIO_PUNCT_DENSITY_THRESHOLD:
            score += 0.15

        # Small font (relative to median body font)
        if font_ratio < _BIBLIO_SMALL_FONT_RATIO:
            score += 0.10

        # Clamp
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0

        return score

    # ---- A.3) Bibliography cluster detection (layout_stream + block_id) ----
    # We only demote spans when there is block-level structural evidence.
    biblio_cluster_keys: set[tuple[str, int, int]] = set()

    groups: Dict[tuple[str, int, int], List[Dict]] = {}
    for s in geo_spans:
        stream = str(s.get("layout_stream") or "")
        bid = int(s.get("block_id", 0))
        page_num = s.get("page_number", 0)  # ← FIXED: use `s`
        key = (stream, bid, page_num)
        groups.setdefault(key, []).append(s)

    for key, group in groups.items():
        if len(group) < _BIBLIO_MIN_CLUSTER_SIZE:
            continue

        high_score = 0
        for s in group:
            t_raw = (s.get("cleaned_text") or s.get("raw_text") or "").strip()
            fsz = float(s.get("font_size", median_font))
            fr = fsz / max(1e-6, median_font)
            if _biblio_span_score(t_raw, fr) >= _BIBLIO_CLUSTER_SCORE_THRESHOLD:
                high_score += 1

        ratio = high_score / max(len(group), 1)
        if high_score >= _BIBLIO_MIN_CLUSTER_SIZE and ratio >= _BIBLIO_CLUSTER_RATIO_THRESHOLD:
            biblio_cluster_keys.add(key)

            if trace_id:
                logger.debug(
                    "[%s] Bibliography cluster detected: layout_stream=%s block_id=%d high_score=%d/%d (%.0f%%)",
                    trace_id, key[0], key[1], high_score, len(group), ratio * 100.0
                )

    # ---- B) Single-pass scoring ----
    for span in spans:
        bbox = span.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        # NEW: Bibliography cluster membership (highest confidence demotion)
        if span.get("_outlier_reasons") == ["bibliography_heading"]:
            continue

        stream = str(span.get("layout_stream") or "")
        bid = int(span.get("block_id", 0))
        page_num = span.get("page_number", 0)
        if (stream, bid, page_num) in biblio_cluster_keys:
            span["role"] = TextRole.FOOTNOTE.value
            span["_outlier_score"] = 0.90
            span["_outlier_reasons"] = ["bibliography_cluster"]
            continue

        # Respect structural roles (don’t “override ground truth”)
        role = span.get("role", TextRole.BODY.value)
        if role in protected_roles:
            continue

        # Respect explicit protection flags if present in your pipeline
        if span.get("filter_protected") is True:
            continue

        # Text for semantic patterns
        text_raw = (span.get("cleaned_text") or span.get("raw_text") or "").strip()
        text = text_raw.lower() if text_raw else ""
        if not text:
            continue

        # Signal 1: boundary (tight thresholds reduce false positives)
        y_center = (bbox[1] + bbox[3]) / 2.0
        rel_pos = (y_center - content_min_y) / content_height
        boundary = (rel_pos <= 0.05) or (rel_pos >= 0.95)

        # Signal 2: small font
        font_size = float(span.get("font_size", median_font))
        font_ratio = font_size / max(1e-6, median_font)
        small_font = font_ratio < 0.92

        # Signal 3/5: gap + orphan block
        bid = int(span.get("block_id", 0))
        gap_before = float(block_gaps.get(bid, 0.0))
        orphan = int(blocks.get(bid, {}).get("count", 0)) <= 2

        # Gap must be both relative AND absolute (prevents “figure gap” false hits)
        gap = (gap_before > (median_gap * 2.0)) and (gap_before > 40.0)

        # Signal 4: pattern match (attribution/meta only)
        pattern = any(p in text for p in meta_patterns)

        # ---- Combination scoring (compact, but conservative) ----
        score = 0.0
        reasons = []

        if pattern:
            score = 0.50
            reasons.append("pattern")
            if boundary:
                score = 0.70
                reasons.append("boundary")
            if small_font:
                score = 0.90
                reasons.append("small_font")

        elif boundary and small_font:
            score = 0.50
            reasons.extend(["boundary", "small_font"])
            if gap or orphan:
                score = 0.70
                if gap: reasons.append("gap")
                if orphan: reasons.append("orphan")

        elif boundary and gap and orphan:
            score = 0.50
            reasons.extend(["boundary", "gap", "orphan"])

        # Verdict threshold: only demote on high confidence
        if score >= 0.70:
            # CRITICAL: demote ONLY into roles already guaranteed non-viable for TTS.
            # Bottom artifacts behave like footnotes; top artifacts like header artifacts.
            if rel_pos >= 0.50:
                span["role"] = TextRole.FOOTNOTE.value
            else:
                span["role"] = TextRole.HEADER_ARTIFACT.value

            # Debug tags (cheap, optional)
            span["_outlier_score"] = round(score, 2)
            span["_outlier_reasons"] = reasons

    # Summary logging
    if trace_id:
        demoted_count = sum(1 for s in spans if s.get("_outlier_score", 0) >= 0.70)
        if demoted_count > 0:
            logger.debug(
                "[%s] Content-flow refinement: demoted %d outlier spans",
                trace_id, demoted_count
            )


def _associate_captions_to_figures(
        spans: List[Dict],
        figure_rects: List[BboxTuple],
        page_height: float = None,
        trace_id: str = None
) -> None:
    """
    Assign figure_index to spans marked as caption.

    Strategy:
        1. Find nearest figure ABOVE the caption (standard layout)
        2. If none above, find nearest figure BELOW
        3. Apply distance threshold to prevent incorrect binding
        4. Orphan captions get figure_index = -1

    Args:
        spans: List of span dictionaries.
        figure_rects: List of (x0, y0, x1, y1) figure bounding boxes.
        page_height: Page height for distance threshold.
        trace_id: Optional trace ID for logging.

    Mutates:
        Caption spans receive 'figure_index' key.
    """
    if not spans:
        return

    effective_page_height = page_height if page_height else _FILTER_DEFAULT_PAGE_HEIGHT
    max_caption_distance = effective_page_height * _CAPTION_ASSOC_MAX_DISTANCE_RATIO

    # Handle case where there are no figures
    if not figure_rects:
        for span in spans:
            if span.get("role") == TextRole.CAPTION.value:
                span["figure_index"] = -1
                if trace_id:
                    caption_preview = (span.get("cleaned_text") or "")[:30]
                    logger.debug(
                        "[%s] Caption orphaned (no figures): '%s...'",
                        trace_id, caption_preview
                    )
        return

    # Process each caption span
    for span in spans:
        if span.get("role") != TextRole.CAPTION.value:
            continue

        span_rect = _span_to_rect(span)
        if span_rect is None:
            span["figure_index"] = -1
            continue

        span_center_y = (span_rect[1] + span_rect[3]) / 2
        span_center_x = (span_rect[0] + span_rect[2]) / 2

        best_idx: Optional[int] = None
        best_dist: float = float("inf")
        best_direction: str = ""

        for idx, f_rect in enumerate(figure_rects):
            f_center_y = (f_rect[1] + f_rect[3]) / 2
            f_center_x = (f_rect[0] + f_rect[2]) / 2

            # Vertical distance
            vert_dist = abs(f_center_y - span_center_y)

            # Horizontal overlap check
            horiz_overlap = not (span_rect[2] < f_rect[0] or span_rect[0] > f_rect[2])
            horiz_dist = 0.0 if horiz_overlap else min(
                abs(span_rect[0] - f_rect[2]),
                abs(span_rect[2] - f_rect[0])
            )

            # Combined distance (vertical weighted more)
            combined_dist = vert_dist + (horiz_dist * _CAPTION_ASSOC_HORIZ_WEIGHT)

            if f_rect[3] <= span_rect[1]:
                direction = "above"
            elif f_rect[1] >= span_rect[3]:
                direction = "below"
            elif not horiz_overlap:
                # HARDENED: Any non-overlapping horizontal adjacency is a "Side" caption
                # This catches margin notes that aren't perfectly vertically aligned
                direction = "side"
            else:
                direction = "overlap"

            # Prefer figures above caption (standard layout)
            if direction == "above":
                # Standard Top-Bottom layout
                adjusted_dist = combined_dist * _CAPTION_ASSOC_ABOVE_BONUS
            elif direction == "side":
                # Side-by-Side / Margin Note layout
                # Treat side captions with same priority bonus as top captions
                adjusted_dist = combined_dist * _CAPTION_ASSOC_ABOVE_BONUS
            else:
                adjusted_dist = combined_dist

            # HARDENED: Deterministic tie-breaker
            # If distances are effectively equal, prefer horizontal alignment
            if adjusted_dist < best_dist:
                best_dist = adjusted_dist
                best_idx = idx
                best_direction = direction
            elif abs(adjusted_dist - best_dist) < 1e-5:
                # Break ties by choosing the figure closer horizontally
                curr_offset = abs(span_center_x - f_center_x)
                best_fig_rect = figure_rects[best_idx]
                best_center_x = (best_fig_rect[0] + best_fig_rect[2]) / 2
                best_offset = abs(span_center_x - best_center_x)

                if curr_offset < best_offset:
                    best_dist = adjusted_dist
                    best_idx = idx
                    best_direction = direction

        # Apply distance threshold
        blocked = False
        if best_idx is not None:
            target_rect = figure_rects[best_idx]
            target_y = (target_rect[1] + target_rect[3]) / 2

            # Check all other figures
            for other_idx, other_rect in enumerate(figure_rects):
                if other_idx == best_idx: continue
                other_y = (other_rect[1] + other_rect[3]) / 2

                # If other figure is strictly between target and caption vertically
                if min(target_y, span_center_y) < other_y < max(target_y, span_center_y):
                    # And roughly aligned horizontally (within 50px)
                    other_cx = (other_rect[0] + other_rect[2]) / 2
                    if abs(other_cx - span_center_x) < 50:
                        blocked = True
                        break

        if best_idx is not None and best_dist <= max_caption_distance and not blocked:
            span["figure_index"] = best_idx
            if trace_id:
                caption_preview = (span.get("cleaned_text") or "")[:30]
                logger.debug(
                    "[%s] Caption linked: '%s...' → figure %d (%s, dist=%.1f)",
                    trace_id, caption_preview, best_idx, best_direction, best_dist
                )
        else:
            span["figure_index"] = -1
            if trace_id:
                caption_preview = (span.get("cleaned_text") or "")[:30]
                logger.debug(
                    "[%s] Caption orphaned: '%s...' (nearest dist=%.1f > threshold %.1f)",
                    trace_id, caption_preview,
                    best_dist if best_dist != float("inf") else -1,
                    max_caption_distance
                )

# ✦                  ✦                  ✦                  ✦
# ✦───────────── 3 Table & Structure Logic ─────────────✦
# ✦                  ✦                  ✦                  ✦

def _determine_table_subrole(span: Dict, table: Dict) -> str:
    """
    Determine the semantic subrole of a span within a table.

    Strategy:
        1. Check if span Y is in the header region (top portion of table)
        2. Check if span X is in the stub column (left portion of table)
        3. Otherwise, it's a data cell

    Args:
        span: Span dictionary with tuple bbox.
        table: Table dictionary with tuple bbox.

    Returns:
        Subrole string: 'header', 'stub', or 'data'.
    """
    if not table:
        return TableSubRole.DATA.value

    table_bbox = table.get("bbox")
    if table_bbox is None or len(table_bbox) < 4:
        return TableSubRole.DATA.value

    span_bbox = span.get("bbox")
    if span_bbox is None or len(span_bbox) < 4:
        return TableSubRole.DATA.value

    # Extract coordinates from tuples
    # Table bbox: (x0, y0, x1, y1)
    table_x0, table_y0, table_x1, table_y1 = table_bbox
    table_width = table_x1 - table_x0
    table_height = table_y1 - table_y0

    # Span position: (x0, y0, x1, y1)
    span_x0, span_y0 = span_bbox[0], span_bbox[1]

    # Header region: top portion of table
    # Use larger of: fixed pixels OR percentage of table height
    header_threshold_y = table_y0 + max(
        _TABLE_HEADER_THRESHOLD_PIXELS,
        table_height * _TABLE_HEADER_THRESHOLD_RATIO
    )

    # Stub column: left portion of table
    # Use smaller of: fixed pixels OR percentage of table width
    stub_threshold_x = table_x0 + min(
        _TABLE_STUB_THRESHOLD_PIXELS,
        table_width * _TABLE_STUB_THRESHOLD_RATIO
    )

    if span_y0 < header_threshold_y:
        return TableSubRole.HEADER.value
    elif span_x0 < stub_threshold_x:
        return TableSubRole.STUB.value
    else:
        return TableSubRole.DATA.value


def _detect_structural_continuity(
        prev_page: Dict,
        curr_page: Dict,
        trace_id: str = None
) -> Dict[str, bool]:
    """
    Detect structural elements that continue across page boundaries.

    Detection Strategy:
        - Table continuity: Last table on prev_page aligns horizontally with
          first table on curr_page (X-position within tolerance)
        - Figure continuity: Figure on prev_page extends into bottom margin zone
          AND figure on curr_page starts in top margin zone

    Args:
        prev_page: Stage 1 output for previous page.
        curr_page: Stage 1 output for current page.
        trace_id: Optional trace ID for logging.

    Returns:
        Dictionary with continuity flags and bbox tuples for UI linking.
    """
    # Signals produced here MUST be consumed in Phase 4.
    # This method MUST NOT decide sentence or semantic outcomes.

    continuity: Dict = {
        "table_continues": False,
        "figure_continues": False,
        "prev_table_bbox": None,
        "curr_table_bbox": None,
    }

    if not prev_page or not curr_page:
        return continuity

    prev_structure = prev_page.get("structure", {})
    curr_structure = curr_page.get("structure", {})
    prev_metadata = prev_page.get("metadata", {})
    curr_metadata = curr_page.get("metadata", {})

    prev_height = prev_metadata.get("height", _FILTER_DEFAULT_PAGE_HEIGHT)
    curr_height = curr_metadata.get("height", _FILTER_DEFAULT_PAGE_HEIGHT)

    # =========================================================================
    # TABLE CONTINUITY DETECTION
    # =========================================================================
    prev_tables = prev_structure.get("tables", [])
    curr_tables = curr_structure.get("tables", [])

    if prev_tables and curr_tables:
        # Get last table on previous page
        prev_last_table = prev_tables[-1]
        prev_tb: Optional[BboxTuple] = prev_last_table.get("bbox")

        # Get first table on current page
        curr_first_table = curr_tables[0]
        curr_tb: Optional[BboxTuple] = curr_first_table.get("bbox")

        if prev_tb is not None and curr_tb is not None:
            # Extract coordinates from tuples: (x0, y0, x1, y1)
            prev_x0, prev_y0, prev_x1, prev_y1 = prev_tb
            curr_x0, curr_y0, curr_x1, curr_y1 = curr_tb

            prev_width = prev_x1 - prev_x0
            curr_width = curr_x1 - curr_x0

            # Horizontal alignment check
            align_tolerance = max(
                _CONTINUITY_X_ALIGNMENT_TOLERANCE,
                prev_width * 0.10,
                curr_width * 0.10
            )
            x_aligned = abs(prev_x0 - curr_x0) < align_tolerance

            # Width similarity check
            max_width = max(prev_width, curr_width, 1)
            min_width = min(prev_width, curr_width)
            width_ratio = min_width / max_width
            width_similar = width_ratio > _CONTINUITY_WIDTH_SIMILARITY_RATIO

            # Previous table extends to bottom margin
            prev_bottom = prev_y1
            prev_at_bottom = prev_bottom > (prev_height * _CONTINUITY_TABLE_BOTTOM_MARGIN_RATIO)

            # Current table starts at top margin
            curr_top = curr_y0
            curr_at_top = curr_top < (curr_height * _CONTINUITY_TABLE_TOP_MARGIN_RATIO)

            if x_aligned and width_similar and prev_at_bottom and curr_at_top:
                continuity["table_continues"] = True
                continuity["prev_table_bbox"] = prev_tb
                continuity["curr_table_bbox"] = curr_tb

                if trace_id:
                    logger.debug(
                        "[%s] Table continuity detected: page %s → %s",
                        trace_id,
                        prev_metadata.get("page_number", "?"),
                        curr_metadata.get("page_number", "?")
                    )

    # =========================================================================
    # FIGURE CONTINUITY DETECTION
    # =========================================================================
    prev_figures: List[BboxTuple] = prev_structure.get("figure_tuples", [])
    curr_figures: List[BboxTuple] = curr_structure.get("figure_tuples", [])

    if prev_figures and curr_figures:
        for prev_fig in prev_figures:
            # prev_fig is (x0, y0, x1, y1)
            prev_fig_x0, prev_fig_y0, prev_fig_x1, prev_fig_y1 = prev_fig
            prev_fig_bottom = prev_fig_y1

            # Check if figure extends to bottom margin
            prev_at_bottom = prev_fig_bottom > (
                    prev_height * _CONTINUITY_FIGURE_BOTTOM_MARGIN_RATIO)

            if prev_at_bottom:
                for curr_fig in curr_figures:
                    curr_fig_x0, curr_fig_y0, curr_fig_x1, curr_fig_y1 = curr_fig
                    curr_fig_top = curr_fig_y0

                    # Check if figure starts at top margin
                    curr_at_top = curr_fig_top < (curr_height * _CONTINUITY_FIGURE_TOP_MARGIN_RATIO)

                    # Horizontal overlap check (Hardened: Require 20% overlap)
                    inter_x0 = max(prev_fig_x0, curr_fig_x0)
                    inter_x1 = min(prev_fig_x1, curr_fig_x1)
                    overlap_w = max(0.0, inter_x1 - inter_x0)

                    prev_w = max(1.0, prev_fig_x1 - prev_fig_x0)
                    curr_w = max(1.0, curr_fig_x1 - curr_fig_x0)

                    x_overlap = (overlap_w / min(prev_w, curr_w)) >= 0.20

                    if curr_at_top and x_overlap:
                        continuity["figure_continues"] = True

                        if trace_id:
                            logger.debug(
                                "[%s] Figure continuity detected: page %s → %s",
                                trace_id,
                                prev_metadata.get("page_number", "?"),
                                curr_metadata.get("page_number", "?")
                            )
                        break

                if continuity["figure_continues"]:
                    break

    return continuity


# ✦                  ✦                  ✦                  ✦
# ✦───────────── 4 Sentence Stitching & Segmentation ─────────────✦
# ✦                  ✦                  ✦                  ✦

# ✦────── a. Stitching Utilities ──────✦
def _text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two strings.

    Uses SequenceMatcher for robust fuzzy matching.

    Args:
        text1: First string to compare.
        text2: Second string to compare.

    Returns:
        Similarity ratio between 0.0 and 1.0.
    """
    if not text1 or not text2:
        return 0.0

    # HARDENED: Local import prevents NameError if global imports drift.
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()


def _smart_title_case(text: str) -> str:
    """
    Convert ALL CAPS text to Title Case for TTS safety.

    Prevents TTS screaming or letter-spelling behavior on uppercase text.

    Strategy:
        1. Skip if text is not majority uppercase (< 70%)
        2. Preserve known acronyms (DNA, NASA, etc.)
        3. Preserve short all-caps words (≤ 4 chars, likely acronyms)
        4. Preserve words with internal lowercase (iPhone, PhD)
        5. Convert long all-caps words to Title Case

    Examples:
        "CHAPTER 1: INTRODUCTION" → "Chapter 1: Introduction"
        "THE NASA REPORT" → "The NASA Report"
        "DNA SEQUENCING METHODS" → "DNA Sequencing Methods"

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text safe for TTS.
    """
    if not text:
        return text

    # Count uppercase vs alphabetic characters
    upper_chars = sum(1 for c in text if c.isupper())
    alpha_chars = sum(1 for c in text if c.isalpha())

    if alpha_chars == 0:
        return text

    # Only process if majority uppercase
    if upper_chars / alpha_chars < _TITLE_CASE_UPPERCASE_THRESHOLD:
        return text

    words = text.split()
    result: List[str] = []

    for word in words:
        # Strip punctuation for analysis, preserve for output
        stripped = word.strip(".,;:!?()[]\"'")

        # Rule 1: Known acronym — preserve
        if stripped.upper() in PROTECTED_ACRONYMS:
            result.append(word)
            continue

        # Rule 2: Short all-caps word — likely acronym, preserve
        if stripped.isupper() and len(stripped) <= _TITLE_CASE_SHORT_WORD_MAX_LENGTH:
            result.append(word)
            continue

        # Rule 3: Has internal lowercase (e.g., "iPhone") — preserve
        if any(c.islower() for c in stripped):
            result.append(word)
            continue

        # Rule 4: Long all-caps word — convert to title case
        if stripped.isupper() and len(stripped) > _TITLE_CASE_SHORT_WORD_MAX_LENGTH:
            # Extract leading punctuation
            prefix = ""
            suffix = ""
            temp = word

            while temp and not temp[0].isalnum():
                prefix += temp[0]
                temp = temp[1:]

            # Extract trailing punctuation
            while temp and not temp[-1].isalnum():
                suffix = temp[-1] + suffix
                temp = temp[:-1]

            # Apply title case to the core word
            titled = temp.title() if temp else ""
            result.append(prefix + titled + suffix)
        else:
            result.append(word)

    return " ".join(result)


def _is_role_boundary(prev_sent: Dict, curr_sent: Dict) -> bool:
    """
    Determine if a role transition should trigger a chunk boundary.

    Boundaries occur when transitioning:
        - INTO or OUT OF headings/subheadings (section changes)
        - INTO or OUT OF captions
        - INTO or OUT OF code blocks
        - INTO or OUT OF sidebars
        - INTO or OUT OF footnotes
        - INTO or OUT OF tables
        - INTO or OUT OF figures

    Args:
        prev_sent: Previous sentence dictionary with 'role' key.
        curr_sent: Current sentence dictionary with 'role' key.

    Returns:
        True if chunk should be finalized before curr_sent.
    """
    if not prev_sent or not curr_sent:
        return False

    prev_role = prev_sent.get("role", TextRole.BODY.value)
    curr_role = curr_sent.get("role", TextRole.BODY.value)

    # Same role = no boundary
    if prev_role == curr_role:
        return False

    # Boundary if either role is in the boundary set
    if prev_role in _CHUNK_BOUNDARY_ROLES or curr_role in _CHUNK_BOUNDARY_ROLES:
        return True

    # List items: boundary when entering or leaving list
    if curr_role == TextRole.LIST_ITEM.value and prev_role != TextRole.LIST_ITEM.value:
        return True

    if prev_role == TextRole.LIST_ITEM.value and curr_role != TextRole.LIST_ITEM.value:
        return True

    return False


# ✦────── b. Cross-Page Stitching ──────✦

def _stitch_helper_find_next(
        sentences: List[Dict],
        start_idx: int,
        curr_sent: Dict,
        trace_id: Optional[str] = None,
) -> Optional[int]:
    """
    Find next sentence with body-like role, preferring RONC-linked candidates.

    Limited to _STITCH_MAX_LOOKAHEAD to prevent distant false matches.

    HARDENED v2.0 (RONC-aware):
        1. Explicit bounds checking on input.
        2. Safe .get() access for role.
        3. Null sentence handling.
        4. NEW: RONC-preferred — shared atomic units get priority.

    Selection order:
        1. RONC-linked candidate (shared _ronc_atomic_units with curr_sent)
        2. First geometric body-like candidate (fallback)

    Note:
        Candidate discovery is intentionally permissive.
        Stitch eligibility (unmapped, contamination) enforced downstream.

    Args:
        sentences: List of sentence dictionaries.
        start_idx: Index to start searching from.
        curr_sent: Current sentence for RONC linkage check (required).

    Returns:
        Index of next body-like sentence, or None if not found.
    """
    # Safety: Input validity
    if not sentences or start_idx < 0 or start_idx >= len(sentences):
        return None

    # Enforce lookahead limit
    end_idx = min(start_idx + _STITCH_MAX_LOOKAHEAD, len(sentences))

    # ─────────────────────────────────────────────────────────────────────────
    # RONC v2.0: Extract curr's atomic units for linkage check
    # curr_sent is REQUIRED — semantic continuity depends on it
    # ─────────────────────────────────────────────────────────────────────────
    curr_units = set(curr_sent.get("_ronc_atomic_units") or [])

    # Two-pass approach: RONC-linked first, geometric fallback second
    geometric_fallback = None

    for j in range(start_idx, end_idx):
        sent = sentences[j]
        if not sent:
            continue

        role = sent.get("role", TextRole.BODY.value)

        # Check against skip set (O(1) lookup)
        if role in _STITCH_SKIP_ROLES:
            continue

        # ─────────────────────────────────────────────────────────────────────
        # RONC PRIORITY: Shared atomic units → immediate return
        # Semantic linkage trumps geometric proximity
        # ─────────────────────────────────────────────────────────────────────
        sent_units = set(sent.get("_ronc_atomic_units") or [])
        if sent_units and curr_units.intersection(sent_units):
            return j

        # Track first geometric candidate as fallback
        if geometric_fallback is None:
            geometric_fallback = j

    return geometric_fallback


def _stitch_helper_columns_match(prev: Dict, next_s: Dict) -> bool:
    """
    Check if column indices match between sentences.

    Exceptions (column mismatch allowed):
        1. Same RONC atomic unit (semantic continuity)
        2. Cross-page (layout may shift)
        3. Missing column metadata (unknown ≠ mismatch)
    """
    # ─────────────────────────────────────────────────────────────────────────
    # RONC OVERRIDE: Same atomic unit allows column mismatch
    # Semantic continuity trumps geometric column check
    # ─────────────────────────────────────────────────────────────────────────
    prev_units = set(prev.get("_ronc_atomic_units") or [])
    next_units = set(next_s.get("_ronc_atomic_units") or [])
    if prev_units and next_units and prev_units.intersection(next_units):
        return True

    prev_page = prev.get("page_number")
    next_page = next_s.get("page_number")

    # Cross-page: ignore column index (layout shift is common)
    if prev_page != next_page:
        return True

    prev_col = prev.get("column_index")
    next_col = next_s.get("column_index")

    # Missing column metadata should not block stitching
    if prev_col is None or next_col is None:
        return True

    return prev_col == next_col


def _is_cross_page_continuation(
        prev_sent: Dict,
        next_sent: Dict,
        prev_spans: List[Dict] = None,
        next_spans: List[Dict] = None
) -> bool:
    """
    Determine if two sentences should be chunked together across a page break.

    PHASE 2.8 CONTRACT:
        Input sentences should be TTS-viable (from pre-filtered spans).
        Guard B is defense-in-depth for non-viable roles that slip through.

    Decision hierarchy:
        0. RONC authority (atomic units, break signals)
        0.5. Contamination guard (uncertain roles → block cross-page)
        1. Linguistic signals (via _stitch_helper_should_merge when no spans)
        2. Structural guards (margin, role, stage1 hints)

    Args:
        prev_sent: Previous sentence dictionary.
        next_sent: Next sentence dictionary.
        prev_spans: Optional span list (auto-extracted from sentence if missing).
        next_spans: Optional span list (auto-extracted from sentence if missing).

    Returns:
        True if sentences should be allowed to merge across page boundary.
    """
    if not prev_sent or not next_sent:
        return False

    # =========================================================================
    # Gate 0: RONC v2.0 Authority (Cross-Page Specific)
    # =========================================================================
    # Extract boundary spans (if available)
    # Spans enable RONC authority checks; without them, fall back to heuristics
    # ─────────────────────────────────────────────────────────────────────────
    # RONC v2.1 FIX: Auto-extract spans from sentences when not passed explicitly
    # Prevents silent degradation to heuristics-only mode
    # ─────────────────────────────────────────────────────────────────────────
    if prev_spans is None:
        prev_spans = prev_sent.get("_source_spans") or prev_sent.get("source_spans") or []
    if next_spans is None:
        next_spans = next_sent.get("_source_spans") or next_sent.get("source_spans") or []

    has_spans = bool(prev_spans) and bool(next_spans)
    prev_boundary = prev_spans[-1] if has_spans else None
    next_boundary = next_spans[0] if has_spans else None

    prev_unit = prev_boundary.get("_ronc_atomic_unit_id") if prev_boundary else None
    next_unit = next_boundary.get("_ronc_atomic_unit_id") if next_boundary else None

    # Same atomic unit across page break → Allow (RONC says they belong together)
    if has_spans and prev_unit is not None and next_unit is not None and prev_unit == next_unit:
        return True

    # Explicit break → Block
    if has_spans and prev_boundary.get("_ronc_break_after") is True:
        return False

    # Different atomic units → Block
    if has_spans and prev_unit is not None and next_unit is not None and prev_unit != next_unit:
        return False

    # =========================================================================
    # Gate 0.5: Contamination guard (cross-page risk amplification)
    # Contaminated sentences have uncertain role classification;
    # cross-page stitching compounds this uncertainty.
    # =========================================================================
    if prev_sent.get("_contaminated") or next_sent.get("_contaminated"):
        return False

    # =========================================================================
    # Gate 1: Sentence-level linguistic check
    #
    # NOTE:
    # - Stitching callers already ran _stitch_helper_should_merge
    # - Chunking callers do NOT pass spans and rely on this gate
    # =========================================================================
    if not has_spans:
        # Sentence-level RONC fallback: block if both have units but no overlap
        prev_units = set(prev_sent.get("_ronc_atomic_units") or [])
        next_units = set(next_sent.get("_ronc_atomic_units") or [])
        if prev_units and next_units and not prev_units.intersection(next_units):
            return False

        should_merge, _ = _stitch_helper_should_merge(prev_sent, next_sent)
        if not should_merge:
            return False

    # =========================================================================
    # Gate 2: Span-level structural guards (cross-page specific)
    # Only applies when boundary spans are available
    # =========================================================================
    if not has_spans:
        return True

    # -------------------------------------------------------------
    # Guard A: Layout stream — no margin involvement cross-page
    # -------------------------------------------------------------
    prev_stream = prev_boundary.get("layout_stream", "") if prev_boundary else ""
    next_stream = next_boundary.get("layout_stream", "") if next_boundary else ""

    if (
            (isinstance(prev_stream, str) and prev_stream.startswith("margin"))
            or (isinstance(next_stream, str) and next_stream.startswith("margin"))
            or bool(prev_boundary and prev_boundary.get("is_margin_content"))
            or bool(next_boundary and next_boundary.get("is_margin_content"))
    ):
        return False

    # -------------------------------------------------------------
    # Guard B: Role compatibility (defense-in-depth after Phase 2.8 pre-filter)
    # -------------------------------------------------------------
    if prev_boundary and next_boundary and (
            prev_boundary.get("role") in _TTS_NON_VIABLE_ROLES
            or next_boundary.get("role") in _TTS_NON_VIABLE_ROLES
    ):
        return False

    # -------------------------------------------------------------
    # Guard C: Stage 1 non-viable hint
    # -------------------------------------------------------------
    if next_boundary and next_boundary.get("_stage1_nonviable_hint", False):
        return False

    return True


def _stitch_cross_page_sentences(
        all_sentences: List[Dict],
        trace_id: str = None
) -> List[Dict]:
    """
    Merge sentence fragments split by page or paragraph boundaries.

    ARCHITECTURAL v1.8:
        Renamed conceptually to "Sentence Healer" — handles both cross-page
        and same-page stitching. Function name preserved for API compatibility.

    Uses multi-pass approach with lookahead to find continuation candidates.
    Each pass may enable new stitches (e.g., A+B merge enables A+B+C merge).

    Ordering Note:
        Non-body sentences (headers, captions) between merged fragments
        are preserved but may be reordered relative to the merged result.
        This maintains body text flow continuity.

    Args:
        all_sentences: List of sentence dictionaries to process.
        trace_id: Optional trace ID for logging.

    Returns:
        List of sentences with fragments merged. Returns input unchanged
        if None, empty, or single-element list.
    """
    # =========================================================================
    # INPUT VALIDATION (Hardened)
    # =========================================================================
    if not all_sentences:
        return all_sentences or []

    if len(all_sentences) < 2:
        return all_sentences

    # =========================================================================
    # DIAGNOSTIC: Source Spans Population Check (RONC v2.1)
    #
    # _source_spans enables RONC authority gate in _stitch_helper_should_merge().
    # Without it, stitch decisions degrade to heuristics-only (silent failure).
    # This diagnostic surfaces the integration gap for rapid triage.
    # =========================================================================
    if trace_id:
        missing_source_spans = 0
        missing_source_span_ids = 0
        contaminated_sentences = 0
        total_body_sentences = 0

        for sent in all_sentences:
            if not isinstance(sent, dict):
                continue

            # Skip non-body sentences (stitch does not operate on these)
            role = sent.get("role", TextRole.BODY.value)
            if role in _STITCH_SKIP_ROLES:
                continue

            # Skip unmapped sentences (no spans by definition; not a wiring failure)
            if sent.get("unmapped") or sent.get("span_start_index") == -1:
                continue

            total_body_sentences += 1

            # Data availability check (truthy): detects missing OR empty runtime spans
            if not sent.get("_source_spans"):
                missing_source_spans += 1

            # Wiring/provenance check (key presence): avoids false positives on empty/None-heavy lists
            if "_source_span_ids" not in sent:
                missing_source_span_ids += 1
            if sent.get("_contaminated"):
                contaminated_sentences += 1

        if total_body_sentences > 0:
            if missing_source_spans > 0:
                logger.warning(
                    "[%s] RONC DEGRADED: %d/%d body sentences missing _source_spans — "
                    "stitch authority gate will fall back to heuristics",
                    trace_id, missing_source_spans, total_body_sentences
                )

            if missing_source_span_ids > 0:
                logger.warning(
                    "[%s] PROVENANCE GAP: %d/%d body sentences missing _source_span_ids — "
                    "reconstruction reversibility compromised",
                    trace_id, missing_source_span_ids, total_body_sentences
                )

            if contaminated_sentences > 0:
                logger.warning(
                    "[%s] CONTAMINATION: %d/%d body sentences marked contaminated — "
                    "cross-page stitching guarded",
                    trace_id, contaminated_sentences, total_body_sentences
                )
            if missing_source_spans == 0 and missing_source_span_ids == 0:
                logger.debug(
                    "[%s] Source span linkage verified: %d body sentences fully populated",
                    trace_id, total_body_sentences
                )

    # =========================================================================
    # MULTI-PASS STITCHING
    # =========================================================================
    for pass_num in range(_STITCH_MAX_PASSES):
        result: List[Dict] = []
        i = 0
        stitches_this_pass = 0
        same_page_stitches = 0  # NEW: Track same-page vs cross-page
        cross_page_stitches = 0
        rejections_this_pass: Dict[str, int] = {}

        while i < len(all_sentences):
            curr = all_sentences[i]

            # Defensive: Skip malformed entries
            if not curr or not isinstance(curr, dict):
                i += 1
                continue

            curr_role = curr.get("role", TextRole.BODY.value)

            # Skip non-body sentences (just pass through)
            if curr_role in _STITCH_SKIP_ROLES:
                result.append(curr)
                i += 1
                continue

            # Skip unmapped / spanless sentences (do not attempt healing)
            # Aligns with diagnostic contract: not a wiring failure, not stitchable.
            if curr.get("unmapped") or curr.get("span_start_index") == -1:
                result.append(curr)
                i += 1
                continue

            # Find next body-like sentence
            next_idx = _stitch_helper_find_next(all_sentences, i + 1, curr, trace_id)

            if next_idx is not None:
                next_sent = all_sentences[next_idx]

                # Defensive: Validate next_sent
                if not next_sent or not isinstance(next_sent, dict):
                    result.append(curr)
                    i += 1
                    continue

                # ---------------------------------------------------------------------
                # CONTAMINATION GUARD (CROSS-PAGE ONLY)
                # Avoid semantic jumps when contamination is present.
                # Same-page stitching may still be valid and is allowed.
                # ---------------------------------------------------------------------
                if (
                        curr.get("_contaminated") or next_sent.get("_contaminated")
                ) and curr.get("page_number") != next_sent.get("page_number"):
                    if trace_id:
                        logger.debug(
                            "[%s] Skipping CROSS-PAGE stitch due to contamination: curr=%s next=%s",
                            trace_id,
                            curr.get("_contaminated"),
                            next_sent.get("_contaminated")
                        )
                    result.append(curr)
                    i += 1
                    continue

                # ---------------------------------------------------------------------
                # ECHO GUARD (Overlap Trimmer) - ROBUST VERSION
                # Detects if 'next' (Page N+1) starts with text already present at
                # the end of 'curr' (Page N). Handles merged ghosts + punctuation diffs.
                # ---------------------------------------------------------------------
                if curr.get("page_number") != next_sent.get("page_number"):
                    c_text = (curr.get("text") or "").strip()
                    n_text = (next_sent.get("text") or "").strip()

                    # Normalize for comparison: Strip trailing punctuation from c_text
                    # because the "Ghost" version in n_text might be merged and missing it.
                    # v3.2: Expanded strip list to catch brackets, quotes, spaces (robust echo)
                    c_text_clean = c_text.rstrip(".,;:!?[]()\"' ")

                    overlap_len = 0
                    max_check = min(len(c_text_clean), len(n_text), 300)

                    for length in range(max_check, 15, -1):
                        candidate = n_text[:length]
                        # Clean the candidate too, just in case
                        # v3.2: Match expanded strip list
                        candidate_clean = candidate.rstrip(".,;:!?[]()\"' ")

                        if c_text_clean.endswith(candidate_clean):
                            # FOUND IT. Now we need the *real* length in n_text to trim.
                            overlap_len = length
                            break

                    if overlap_len > 0:
                        trimmed_text = n_text[overlap_len:].strip()

                        if trace_id:
                            logger.info(
                                "[%s] Echo Guard: Trimming %d char overlap ('%s...') from page %s",
                                trace_id, overlap_len, n_text[:20], next_sent.get("page_number")
                            )

                        if not trimmed_text:
                            # CASE 1: Next sentence is purely an echo -> DROP IT
                            result.append(curr)
                            # Preserve intermediates
                            for k in range(i + 1, next_idx):
                                if all_sentences[k]: result.append(all_sentences[k])
                            i = next_idx + 1
                            continue
                        else:
                            # CASE 2: Next sentence is Echo + New Text -> TRIM IT
                            next_sent["text"] = trimmed_text
                            # Fall through to standard processing

                # ---------------------------------------------------------------------
                # CONNECTOR RULE (INLINE, NO HELPER)
                # If curr ends with a connector word + punctuation (e.g. "or."),
                # treat punctuation as artifact and FORCE merge with next sentence.
                # ---------------------------------------------------------------------
                curr_text = (curr.get("text") or "").rstrip()
                next_text = (next_sent.get("text") or "").lstrip()

                # Extract last token safely
                curr_words = curr_text.split()
                if curr_words:
                    last_token = curr_words[-1]
                    # Strip trailing punctuation from last token
                    stripped = last_token.rstrip(".,;:!?")
                    trailing_punct = last_token[len(stripped):]

                    # Check for hanging connector with punctuation
                    if (
                            stripped.lower() in _STITCH_CONNECTOR_WORDS and
                            trailing_punct
                    ):
                        # ─────────────────────────────────────────────────────
                        # RONC GUARD: Connector rule must still respect RONC
                        # FIX v2.8: Use _ronc_atomic_units (list), not scalar
                        # ─────────────────────────────────────────────────────
                        prev_units = set(curr.get("_ronc_atomic_units") or [])
                        next_units = set(next_sent.get("_ronc_atomic_units") or [])

                        # Block if RONC says don't merge (span-level authority)
                        ronc_blocks = False

                        # Check break_after on boundary SPAN, not sentence
                        # Prefer runtime spans, fall back to legacy for backward compatibility
                        curr_spans = curr.get("_source_spans") or curr.get("source_spans") or []
                        next_spans_rt = next_sent.get("_source_spans") or next_sent.get(
                            "source_spans") or []

                        curr_boundary_span = curr_spans[-1] if curr_spans else {}
                        next_boundary_span = next_spans_rt[0] if next_spans_rt else {}

                        # 1) Explicit RONC break always blocks
                        if curr_boundary_span.get("_ronc_break_after") is True:
                            ronc_blocks = True
                        else:
                            # 2) Span-level unit boundary blocks if both are known and differ
                            curr_unit = curr_boundary_span.get("_ronc_atomic_unit_id")
                            next_unit = next_boundary_span.get("_ronc_atomic_unit_id")
                            if curr_unit is not None and next_unit is not None and curr_unit != next_unit:
                                ronc_blocks = True
                            # 3) Sentence-level fallback: block if both have units but no overlap
                            elif (prev_units and next_units and not prev_units.intersection(
                                    next_units)):
                                ronc_blocks = True

                        if ronc_blocks:
                            # RONC authority overrides connector rule
                            result.append(curr)
                            i += 1
                            continue

                        # ─────────────────────────────────────────────────────
                        # SMART CONNECTOR CHECK (v1.4.0):
                        # Even if word looks like a connector, if we have a hard
                        # period and next text starts Uppercase, it's a new sentence.
                        # Prevents: "deformation." + "They" → "deformation They"
                        # ─────────────────────────────────────────────────────
                        if trailing_punct == "." and next_text and next_text[0].isupper():
                            # Hard period + Uppercase = likely sentence boundary
                            result.append(curr)
                            i += 1
                            continue

                        # Remove ONLY the trailing punctuation from curr text
                        fixed_curr_text = curr_text[:-len(trailing_punct)].rstrip()

                        # Force merge: connector implies continuation by definition
                        merged = _stitch_helper_merge(
                            {**curr, "text": fixed_curr_text},
                            next_sent
                        )

                        # Enforce exactly one space between fragments
                        merged["text"] = fixed_curr_text + " " + next_text

                        # Metrics
                        is_same_page = curr.get("page_number") == next_sent.get("page_number")
                        if is_same_page:
                            same_page_stitches += 1
                        else:
                            cross_page_stitches += 1

                        stitches_this_pass += 1

                        if trace_id:
                            stitch_type = "same-page" if is_same_page else "cross-page"
                            logger.info(
                                "[%s] Stitched (%s): page %s → %s (reason=connector_force_merge, text='%s...')",
                                trace_id,
                                stitch_type,
                                curr.get("page_number"),
                                next_sent.get("page_number"),
                                merged.get("text", "")[:50]
                            )

                        result.append(merged)

                        # Preserve skipped non-body sentences
                        for k in range(i + 1, next_idx):
                            skipped = all_sentences[k]
                            if skipped:
                                result.append(skipped)

                        i = next_idx + 1
                        continue

                should, reason = _stitch_helper_should_merge(curr, next_sent)

                # Cross-page merges require additional structural approval
                if should and curr.get("page_number") != next_sent.get("page_number"):
                    # Backward-compatible span access (runtime only)
                    prev_spans = curr.get("_source_spans") or curr.get("source_spans") or []
                    next_spans = next_sent.get("_source_spans") or next_sent.get(
                        "source_spans") or []
                    if not _is_cross_page_continuation(curr, next_sent, prev_spans, next_spans):
                        should = False
                        reason = "cross_page_structural_guard"

                if should:
                    # Merge sentences
                    merged = _stitch_helper_merge(curr, next_sent)

                    # Track same-page vs cross-page for metrics
                    is_same_page = curr.get("page_number") == next_sent.get("page_number")
                    if is_same_page:
                        same_page_stitches += 1
                    else:
                        cross_page_stitches += 1

                    if trace_id:
                        stitch_type = "same-page" if is_same_page else "cross-page"
                        logger.info(
                            "[%s] Stitched (%s): page %s → %s (reason=%s, text='%s...')",
                            trace_id,
                            stitch_type,
                            curr.get("page_number"),
                            next_sent.get("page_number"),
                            reason,
                            merged.get("text", "")[:50]
                        )

                    result.append(merged)
                    stitches_this_pass += 1

                    # Preserve any skipped non-body sentences between curr and next
                    # Note: These are added AFTER merged, which maintains body flow
                    for k in range(i + 1, next_idx):
                        skipped = all_sentences[k]
                        if skipped:
                            result.append(skipped)

                    i = next_idx + 1  # Skip past the merged sentence
                    continue
                else:
                    # Track rejection reasons for debugging
                    rejections_this_pass[reason] = rejections_this_pass.get(reason, 0) + 1

            result.append(curr)
            i += 1

        all_sentences = result

        if trace_id:
            logger.debug(
                "[%s] Stitch pass %d: %d total (%d same-page, %d cross-page), rejections=%s",
                trace_id,
                pass_num + 1,
                stitches_this_pass,
                same_page_stitches,
                cross_page_stitches,
                rejections_this_pass
            )

        if stitches_this_pass == 0:
            break  # Convergence — no more stitches possible

    # =========================================================================
    # FINAL LOGGING
    # =========================================================================
    if trace_id:
        total_stitched = sum(1 for s in all_sentences if s.get("is_stitched"))
        logger.info(
            "[%s] Sentence healing complete: %d stitched sentences",
            trace_id, total_stitched
        )

    return all_sentences


def _stitch_helper_merge(curr: Dict, next_sent: Dict) -> Dict:
    """
    Merge two sentences, preserving all relevant metadata.

    HARDENED v1.8:
        1. Cross-Page BBox Safety: Prevents merging coordinates from different pages.
        2. Null Safety: Robust handling of missing text fields.

    Args:
        curr: Current sentence dictionary.
        next_sent: Next sentence dictionary to merge.

    Returns:
        Merged sentence dictionary with combined text and metadata.
    """
    # Safe text retrieval with stripping
    t1 = (curr.get("text") or "").rstrip()
    t2 = (next_sent.get("text") or "").lstrip()

    # Join with single space (safest default)
    merged_text = f"{t1} {t2}".strip()

    merged = curr.copy()
    merged["text"] = merged_text
    merged["is_stitched"] = True

    # Propagate contamination audit flags (defense-in-depth for later passes)
    if curr.get("_contaminated") or next_sent.get("_contaminated"):
        merged["_contaminated"] = True
        roles = set(curr.get("_contaminated_roles") or []) | set(
            next_sent.get("_contaminated_roles") or [])
        if roles:
            merged["_contaminated_roles"] = sorted(roles)



    # ─────────────────────────────────────────────────────────────────────────
    # RONC v2.0: Preserve lineage for stitched sentences
    # ─────────────────────────────────────────────────────────────────────────

    curr_sources = curr.get("_source_spans") or curr.get("source_spans") or []
    next_sources = next_sent.get("_source_spans") or next_sent.get("source_spans") or []

    # Collect canonical span provenance IDs (v2.1: _source_span_ids)
    # Must preserve order and keep None sentinels (provenance gaps) without dropping.
    # --- source_cids merge (contribution truth, Pass 1 addition) ---
    prev_source_cids = curr.get("source_cids") or []
    next_source_cids = next_sent.get("source_cids") or []

    combined_cids = list(prev_source_cids) + list(next_source_cids)

    # Dedupe exact CID duplicates only, preserve order
    seen_cids = set()
    deduped_cids = []
    for cid in combined_cids:
        if cid is not None and cid not in seen_cids:
            seen_cids.add(cid)
            deduped_cids.append(cid)

    if deduped_cids:
        merged["source_cids"] = deduped_cids

    # ─────────────────────────────────────────────────────────────────────────
    # RONC v2.1 FIX: Use _ronc_atomic_units (list), not _ronc_atomic_unit_id
    # Sentences store unit membership as a list; spans store single ID.
    # ─────────────────────────────────────────────────────────────────────────
    prev_units = set(curr.get("_ronc_atomic_units") or [])
    next_units = set(next_sent.get("_ronc_atomic_units") or [])

    combined_units = prev_units | next_units
    if combined_units:
        merged["_ronc_atomic_units"] = sorted(combined_units)

        # Convenience scalar: only set if wholly within one unit
        if len(combined_units) == 1:
            merged["_ronc_atomic_unit_id"] = list(combined_units)[0]

        # Audit signal: merge crossed unit boundaries (no overlap)
        if prev_units and next_units and not prev_units.intersection(next_units):
            merged["_ronc_cross_unit_merge"] = True

    # ─────────────────────────────────────────────────────────────────────────
    # RONC v2.1: Project flow identity from authoritative span sources
    # Flow identity is DERIVED from _source_spans, never stored on sentences.
    # This projection is for merge audit only; manifest derives fresh.
    # ─────────────────────────────────────────────────────────────────────────
    prev_flows = set()
    next_flows = set()

    for sp in curr_sources:
        flow = sp.get("layout_stream")
        if flow:
            prev_flows.add(flow)

    for sp in next_sources:
        flow = sp.get("layout_stream")
        if flow:
            next_flows.add(flow)

    # Audit signal only: cross-flow merge detected (should rarely happen after gate)
    if prev_flows and next_flows and not prev_flows.intersection(next_flows):
        merged["_ronc_cross_flow_merge"] = True

    # Update span indices (defensive: preserve full provenance range)
    curr_start = curr.get("span_start_index")
    next_start = next_sent.get("span_start_index")
    if curr_start is not None or next_start is not None:
        merged["span_start_index"] = min(
            x for x in (curr_start, next_start) if x is not None
        )

    merged["span_end_index"] = max(
        curr.get("span_end_index", -1),
        next_sent.get("span_end_index", -1)
    )

    # Update character indices (critical for timing alignment)
    # v2.1: segmentation emits char_start / char_end
    if "char_end" in next_sent:
        merged["char_end"] = next_sent["char_end"]
    if "char_start" not in curr and "char_start" in next_sent:
        # Defensive: if curr was missing but next has it, keep something consistent
        merged["char_start"] = next_sent["char_start"]

    # Merge runtime span views if present (v2.1), fallback to legacy
    if curr_sources or next_sources:
        merged["_source_spans"] = list(curr_sources) + list(next_sources)

    # NOTE: _source_span_ids already merged + deduped above (keep single authority)

    # =========================================================================
    # BBOX MERGE (HARDENED — Cross-Page Safety)
    # =========================================================================
    curr_bbox = curr.get("bbox")
    next_bbox = next_sent.get("bbox")

    # CRITICAL: Check if same page before merging coordinates
    same_page = curr.get("page_number") == next_sent.get("page_number")

    if same_page and curr_bbox and next_bbox:
        # Standard Union: valid because coordinate systems match
        if len(curr_bbox) >= 4 and len(next_bbox) >= 4:
            merged["bbox"] = (
                min(curr_bbox[0], next_bbox[0]),  # min x0
                min(curr_bbox[1], next_bbox[1]),  # min y0
                max(curr_bbox[2], next_bbox[2]),  # max x1
                max(curr_bbox[3], next_bbox[3]),  # max y1
            )
    elif not same_page:
        # Cross-Page: Keep STARTING bbox only
        # Merging Page 1 + Page 2 coords creates invalid geometry
        merged["bbox"] = curr_bbox
    elif curr_bbox:
        merged["bbox"] = curr_bbox
    elif next_bbox:
        merged["bbox"] = next_bbox
    # else: no bbox in merged

    # Track pages involved
    pages: set = set()
    if "page_number" in curr:
        pages.add(curr["page_number"])
    if "page_number" in next_sent:
        pages.add(next_sent["page_number"])
    if "stitched_pages" in curr:
        pages.update(curr["stitched_pages"])
    merged["stitched_pages"] = sorted(pages)

    # Update end position
    if "end_y" in next_sent:
        merged["end_y"] = next_sent["end_y"]

    # ------------------------------------------------------------------
    # Preserve multi-page geometry during iterative sentence merges
    # ------------------------------------------------------------------

    # Initialize page_bboxes if missing
    if "page_bboxes" not in merged or not isinstance(merged.get("page_bboxes"), dict):
        merged["page_bboxes"] = {}

    # Seed with current sentence bbox
    curr_page = curr.get("page_number")
    curr_bbox = curr.get("bbox")
    if curr_page is not None and curr_bbox:
        merged["page_bboxes"][curr_page] = curr_bbox

    # Add next sentence bbox
    next_page = next_sent.get("page_number")
    next_bbox = next_sent.get("bbox")
    if next_page is not None and next_bbox:
        merged["page_bboxes"][next_page] = next_bbox

    curr_risks = curr.get("boundary_risks") or []
    next_risks = next_sent.get("boundary_risks") or []
    if curr_risks or next_risks:
        merged["boundary_risks"] = list(dict.fromkeys(curr_risks + next_risks))

    if curr.get("alignment_risk") or next_sent.get("alignment_risk"):
        merged["alignment_risk"] = True

    # Preserve structural crossing flags (also detect new cross-page from merge)
    if curr.get("crosses_pages") or next_sent.get("crosses_pages") or len(pages) > 1:
        merged["crosses_pages"] = True
    if curr.get("crosses_columns") or next_sent.get("crosses_columns"):
        merged["crosses_columns"] = True

    return merged


def _stitch_helper_should_merge(
        prev: Dict,
        next_s: Dict
) -> Tuple[bool, str]:
    """
    Determine if two sentences should be merged.

    ARCHITECTURAL CHANGES:
        v2.0: Y-gap guard, stricter same-page role check
        v2.1: PRE-COMPUTE moved up, dynamic span gap, linguistic overrides
        v2.8: RONC authority gate, sentence-level fallback

    Processing Order:
        0. RONC Authority Gate (span-level, then sentence-level fallback)
        1. A2 Continuation signals
        2. PRE-COMPUTE: Text analysis
        3. RULE 0: Empty text check
        4. RULE 1: Span proximity
        5. RULE 1.5: Geometric Y-gap check
        6. RULE 2: Column alignment
        7. RULE 3: Role consistency
        8. RULE 3.25: Parenthetical/enumeration hard stop
        9. RULE 3.5: Capitalized sentence hard stop
        10. RULE 4: Terminal punctuation (with overrides)
        11. RULE 6: Drop cap reassembly
        12. Continuation signal detection (A-E)

    Args:
        prev: Previous sentence dictionary (should contain _source_spans for RONC).
        next_s: Next sentence dictionary (should contain _source_spans for RONC).

    Returns:
        Tuple of (should_merge: bool, reason: str).
    """
    prev_text = (prev.get("text") or "").rstrip()
    next_text = (next_s.get("text") or "").lstrip()

    # =========================================================================
    # TTS PHYSICAL CONSTRAINT GUARD
    # =========================================================================
    # This is NOT a semantic judgment — RONC authority is not being overridden.
    # Sentences may belong together semantically but cannot coexist in a single
    # TTS chunk due to service limitations. Content and metadata are preserved;
    # only the audio will have a natural pause at the sentence boundary.
    # =========================================================================
    if len(prev_text) + 1 + len(next_text) > _CHUNK_ABSOLUTE_MAX_CHARS:
        return False, "exceeds_tts_limit"

    # =========================================================================
    # RONC v2.0 AUTHORITY GATE (Phases 4-6)
    # Contract-based signals take precedence over linguistic heuristics
    #
    # Authority Stack:
    #   1. RONC explicit links → FORCE merge
    #   2. RONC same atomic unit → FORCE merge
    #   3. RONC break_after → BLOCK merge
    #   4. RONC different atomic units → BLOCK merge
    #   5. No RONC opinion → Continue to linguistic rules
    #
    # NOTE: must_include (protection) is NOT a merge blocker.
    # Protection prevents EXCLUSION from TTS, not COMBINATION of spans.
    # =========================================================================

    # ─────────────────────────────────────────────────────────────────────────
    # Extract boundary spans for RONC field access
    # RONC fields live on spans, not sentences. For merge decisions:
    #   - prev: check LAST span (boundary exiting this sentence)
    #   - next: check FIRST span (boundary entering next sentence)
    # ─────────────────────────────────────────────────────────────────────────
    # Runtime span access: prefer _source_spans (v2.1), fallback to legacy source_spans
    prev_spans = prev.get("_source_spans") or prev.get("source_spans", [])
    next_spans = next_s.get("_source_spans") or next_s.get("source_spans", [])

    prev_boundary_span = prev_spans[-1] if prev_spans else {}
    next_boundary_span = next_spans[0] if next_spans else {}
    # ─────────────────────────────────────────────────────────────────────────
    # HEADING ISOLATION GUARD (v5.2)
    # Never merge sentences where either boundary span has role=heading/subheading.
    # Headings must remain standalone for proper TTS pacing.
    # This runs BEFORE all other checks to ensure heading isolation is absolute.
    # ─────────────────────────────────────────────────────────────────────────
    prev_role = prev_boundary_span.get("role", "")
    next_role = next_boundary_span.get("role", "")
    if prev_role in ("heading", "subheading"):
        return False, "heading_isolation_guard:prev_is_heading"
    if next_role in ("heading", "subheading"):
        return False, "heading_isolation_guard:next_is_heading"

    prev_unit = prev_boundary_span.get("_ronc_atomic_unit_id")
    next_unit = next_boundary_span.get("_ronc_atomic_unit_id")

    # ─────────────────────────────────────────────────────────────────────────
    # SACRED PERIOD GUARD (v3.2 Integrated)
    # Grammar trumps structure: Period/Question/Exclamation + Uppercase = stop.
    # This prevents run-on monsters that overwhelm TTS decoders.
    # Exception: Cut indicators (the., of., and.) may be pysbd artifacts.
    # ─────────────────────────────────────────────────────────────────────────
    if prev_text.endswith(('.', '?', '!')) and next_text and next_text[0].isupper():
        prev_base = prev_text.rstrip('.?!')
        prev_words = prev_base.split()
        if prev_words:
            # Check last word against cut indicators (abbreviations/conjunctions)
            # We define a local safety list in case the global one isn't in scope
            _safe_cut_indicators = {
                'the', 'and', 'of', 'for', 'is', 'was', 'dr', 'mr', 'mrs',
                'vs', 'etc', 'fig', 'eq', 'al', 'i', 'e', 'g', 'inc', 'ltd'
            }
            last_word = prev_words[-1].lower().rstrip(".,;:!?\"')")

            # Use global if available, else local backup
            allowed_cuts = globals().get('_HEALING_CUT_INDICATORS', _safe_cut_indicators)

            if last_word not in allowed_cuts:
                return False, "sacred_period_guard:grammar_boundary"

    # ─────────────────────────────────────────────────────────────────────────
    # FORCE MERGE: Same atomic unit (strongest positive signal)
    # Spans in same unit are semantically continuous by RONC Phase 6 analysis
    # ─────────────────────────────────────────────────────────────────────────
    if prev_unit is not None and next_unit is not None and prev_unit == next_unit:
        return True, "ronc:same_atomic_unit"

    # ─────────────────────────────────────────────────────────────────────────
    # FORCE MERGE: Explicit RONC link from prev to next
    # Phase 4 established this link with confidence threshold
    # ─────────────────────────────────────────────────────────────────────────
    prev_contract = prev_boundary_span.get("_ronc_contract", {})
    prev_links = prev_contract.get("links", {})
    prev_next_link = prev_links.get("next", {})

    next_cid = next_boundary_span.get("_canonical_span_id")
    linked_cid = prev_next_link.get("cid")

    if linked_cid and next_cid and linked_cid == next_cid:
        confidence = prev_next_link.get("confidence", 0.0)
        mutual = prev_next_link.get("mutual", False)
        if mutual:
            return True, f"ronc:mutual_link:{confidence:.2f}"
        elif confidence >= 0.60:
            return True, f"ronc:explicit_link:{confidence:.2f}"

    # ─────────────────────────────────────────────────────────────────────────
    # NOTE: Contamination guard for cross-page lives in _stitch_cross_page_sentences()
    # Same-page stitching of contaminated pairs is allowed (linguistic signals strong)
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK MERGE: Explicit break signal (Phase 6 semantic delimiter)
    # ─────────────────────────────────────────────────────────────────────────
    if prev_boundary_span.get("_ronc_break_after") is True:
        return False, "ronc:break_after"

    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK MERGE: Different atomic units (semantic boundary)
    # ─────────────────────────────────────────────────────────────────────────
    if prev_unit is not None and next_unit is not None and prev_unit != next_unit:
        return False, "ronc:atomic_unit_boundary"

    # ─────────────────────────────────────────────────────────────────────────
    # RONC v2.1: BLOCK MERGE — Flow identity mismatch (cross-stream)
    # Different layout_streams represent distinct content flows that should
    # not be stitched together, even if linguistically compatible.
    # Flow identity is read from SPANS (authoritative), not sentences.
    # ─────────────────────────────────────────────────────────────────────────
    prev_flow = prev_boundary_span.get("layout_stream", "")
    next_flow = next_boundary_span.get("layout_stream", "")

    if prev_flow and next_flow and prev_flow != next_flow:
        return False, f"ronc:flow_identity_mismatch({prev_flow}→{next_flow})"

    # ─────────────────────────────────────────────────────────────────────────
    # FALLBACK: Sentence-level unit check when boundary spans unavailable
    # Only triggers when span-level checks couldn't run (missing _source_spans)
    # ─────────────────────────────────────────────────────────────────────────
    if not prev_spans or not next_spans:
        prev_sent_units = set(prev.get("_ronc_atomic_units") or [])
        next_sent_units = set(next_s.get("_ronc_atomic_units") or [])

        # Block only if BOTH have units AND no overlap
        # Empty = unknown = no opinion (fall through to A2/linguistic)
        if (prev_sent_units and next_sent_units
                and not prev_sent_units.intersection(next_sent_units)):
            return False, "ronc:sentence_unit_boundary_fallback"

    # ─────────────────────────────────────────────────────────────────────────
    # NO OPINION: Continue to A2 and linguistic rules
    # ─────────────────────────────────────────────────────────────────────────

    # Deterministic continuation from A2 (with adjacency verification)
    # Adjacency check: same page, reasonable vertical proximity
    if prev.get("a2_continues_to_next"):
        # Verify adjacency to guard against orphaned flags
        same_page = prev.get("page_number") == next_s.get("page_number")

        # Geometric adjacency: next span should be reasonably close
        prev_bbox = prev.get("bbox")
        next_bbox = next_s.get("bbox")
        geom_adjacent = False
        if prev_bbox and next_bbox and len(prev_bbox) >= 4 and len(next_bbox) >= 4:
            v_gap = next_bbox[1] - prev_bbox[3]  # next_y0 - prev_y1
            fs = float(prev.get("font_size") or 10.0)  # sentence may not carry font_size; default ok
            # Allow gap up to ~3–4 lines (generous to account for header/footer removal)
            geom_adjacent = v_gap <= (fs * 4.0)

        if same_page and geom_adjacent:
            mode = prev.get("a2_continuation_mode", "unknown")
            reason = prev.get("a2_continuation_reason", "a2_signal")
            return True, f"{reason}:{mode}"

    # =========================================================================
    # PRE-COMPUTE: Text Analysis (MOVED UP v2.1)
    # We need these signals available for the Guard Rules below.
    # =========================================================================

    first_char = next_text[0] if next_text else ""
    words_prev = prev_text.split()
    words_next = next_text.split()

    # Extract last word, stripping punctuation
    last_word_raw = words_prev[-1] if words_prev else ""
    last_word = last_word_raw.lower().rstrip(".,;:!?\"')")
    first_word = words_next[0].lower().lstrip("(\"'") if words_next else ""

    # Signals
    has_incomplete_ending = last_word in _STITCH_INCOMPLETE_ENDINGS  # e.g., "the", "of", "and"
    has_lowercase_start = first_char.islower()
    # Strong same-page continuation requires BOTH signals
    strong_linguistic_continuation = has_incomplete_ending and has_lowercase_start

    # =========================================================================
    # RULE 0: Empty Text Check
    # =========================================================================
    if not prev_text or not next_text:
        return False, "empty_text"

    # Track page context
    prev_page = prev.get("page_number")
    next_page = next_s.get("page_number")
    is_same_page = (prev_page == next_page)

    # Initialize span_gap explicitly (assigned only for same-page logic)
    span_gap = None

    columns_match = _stitch_helper_columns_match(prev, next_s)

    # =========================================================================
    # RULE 1: Span Proximity Check (SAME-PAGE ONLY)
    # Prevents merging sentences from distant content blocks.
    #
    # CRITICAL: span_start_index/span_end_index are PAGE-LOCAL indices.
    # They are computed relative to each page's spans_for_text list.
    # Cross-page index comparison is mathematically meaningless:
    #   Page 1 end: span_end_index=17 (of 18 spans)
    #   Page 2 start: span_start_index=0 (of 15 spans)
    #   span_gap = 0 - 17 = -17 (meaningless!)
    #
    # For cross-page pairs, skip this rule entirely.
    # RULE 1.5 (geometric Y-gap) provides backup validation.
    # =========================================================================
    if is_same_page:
        prev_span_end = prev.get("span_end_index", 0)
        next_span_start = next_s.get("span_start_index", 0)
        span_gap = next_span_start - prev_span_end

        allowed_gap = _STITCH_MAX_SPAN_GAP

        # Inflate gap ONLY when structurally adjacent
        if has_incomplete_ending and columns_match and span_gap >= 0:
            allowed_gap = _STITCH_MAX_SPAN_GAP * 2

        # Enforce span proximity
        if span_gap > allowed_gap:
            return False, f"span_gap_too_large:{span_gap}"

    # =========================================================================
    # RULE 1.5: Geometric Proximity Check (NEW v2.0)
    # Prevents merging sentences that are far apart vertically on same page.
    # This catches cross-region bleeding even when column detection fails.
    # =========================================================================
    if is_same_page:
        # Get Y coordinates: end of previous, start of next
        prev_y_end = prev.get("end_y")
        if prev_y_end is None:
            prev_bbox = prev.get("bbox")
            prev_y_end = prev_bbox[3] if prev_bbox and len(prev_bbox) >= 4 else 0

        next_bbox = next_s.get("bbox")
        next_y_start = next_bbox[1] if next_bbox and len(next_bbox) >= 4 else 0

        y_gap = next_y_start - prev_y_end

        # Positive gap: next is below prev
        if y_gap > _STITCH_MAX_Y_GAP_SAME_PAGE:
            return False, f"y_gap_too_large:{y_gap:.1f}"

        # Negative gap: next is ABOVE prev → likely reorder or cross-block
        if y_gap < -_STITCH_MAX_NEGATIVE_Y_GAP:
            return False, f"y_gap_negative:{y_gap:.1f}"

    # =========================================================================
    # RULE 2: Column Alignment
    # =========================================================================
    if not columns_match:
        return False, "column_mismatch"

    # =========================================================================
    # RULE 3: Role Consistency (MODIFIED v2.0)
    # Same-page: Stricter check — only matching roles allowed
    # Cross-page: Looser check — body+non-body allowed (layout shifts common)
    # =========================================================================
    prev_role = prev.get("role", TextRole.BODY.value)
    next_role = next_s.get("role", TextRole.BODY.value)

    if prev_role != next_role:
        if is_same_page:
            # Require STRONG linguistic continuation on same page
            if not strong_linguistic_continuation:
                return False, "same_page_role_mismatch"
        else:
            # Cross-page: Only block non-body + non-body
            # Rationale: Layout shifts between pages are common
            if prev_role != TextRole.BODY.value and next_role != TextRole.BODY.value:
                return False, "role_mismatch"

    # -------------------------------------------------------------------------
    # HEAD NORMALIZATION (for hard-stop checks)
    # Strip leading tokens that commonly mask capitalization in PDFs.
    # -------------------------------------------------------------------------
    curr_head = next_text.lstrip().lstrip("'\"“”‘’([{—–•")
    head_first_char = curr_head[:1] if curr_head else ""
    head_first_word = (curr_head.split()[0] if curr_head.split() else "")

    effective_first_char = head_first_char or first_char

    # -------------------------------------------------------------------------
    # Connector normalization (local, non-invasive)
    # Avoid punctuation-attached tokens breaking connector detection.
    # -------------------------------------------------------------------------
    _last_word_norm = (last_word or "").strip().strip(",;:")

    # =========================================================================
    # RULE 3.25: Parenthetical / Enumeration Completion Hard Stop [P1 FIX]
    # Prevents: "(nociceptors)" + "The receptors..." from merging
    # =========================================================================
    prev_stripped = prev_text.rstrip()

    prev_ends_paren_pattern = (
            prev_stripped.endswith(")") or
            prev_stripped.endswith(").") or
            prev_stripped.endswith("),")
    )

    comma_count = prev_text.count(",")
    prev_is_listish = (comma_count >= 2 and len(prev_stripped) <= 120)

    if (prev_ends_paren_pattern or prev_is_listish) and head_first_char.isupper():
        return False, "hard_stop:parenthetical_or_enumeration_complete"

    # =========================================================================
    # RULE 3.5: Capitalized Sentence Hard Stop [P1 FIX]
    # =========================================================================
    if head_first_char.isupper():
        prev_ends_connector = (
                _last_word_norm in _STITCH_PREV_CONNECTOR_WORDS or
                prev_text.rstrip().endswith((",", ";", ":"))
        )

        if not prev_ends_connector:

            if head_first_word in _STITCH_COMMON_SENTENCE_STARTERS:
                return False, "hard_stop:capitalized_sentence_starter"

            return False, "hard_stop:capitalized_no_connector"

    # =========================================================================
    # RULE 4: Terminal Punctuation Check
    # Includes override for incomplete endings
    # =========================================================================
    has_terminal_punct = bool(_STITCH_TERMINAL_PUNCT_PATTERN.search(prev_text))

    if has_terminal_punct:
        # OVERRIDE 1: Incomplete ending + lowercase start
        # Example: "Mechanoreceptors can." + "be free receptors..."
        if has_incomplete_ending and has_lowercase_start:
            reason = "same_page:incomplete_override" if is_same_page else "incomplete_override"
            return True, reason

        # OVERRIDE 3 (v2.1): Grammatical patterns (Possessives/Gerunds)
        # CRITICAL SAFETY: Only implies continuation if next word is lowercase.
        if has_lowercase_start:
            if (last_word.endswith("'s") or  # Possessive (Merkel's)
                    last_word.endswith("ing")):  # Gerund (responding)
                reason = "same_page:grammatical_fragment_override" if is_same_page else "grammatical_fragment_override"
                return True, reason

        # OVERRIDE 4 (v3.4): Lowercase Terminal Override
        # Handles false sentence breaks: "is called." + "first pain..."
        # Guards: Same page + tight vertical proximity
        if has_lowercase_start and is_same_page:
            prev_y_end = prev.get("end_y")
            if prev_y_end is None:
                prev_bbox = prev.get("bbox")
                prev_y_end = prev_bbox[3] if prev_bbox and len(prev_bbox) >= 4 else 0

            next_bbox = next_s.get("bbox")
            next_y_start = next_bbox[1] if next_bbox and len(next_bbox) >= 4 else 0

            y_gap = next_y_start - prev_y_end

            # Tight threshold: half of max gap (defensive against cross-paragraph merge)
            if y_gap < _STITCH_MAX_Y_GAP_SAME_PAGE * 0.75:
                return True, "same_page:lowercase_terminal_override"

        # No override conditions met — block merge
        return False, "has_terminal_punct"

    # RULE 5 DISABLED: span index domain mismatch (global vs local)

    # =========================================================================
    # RULE 6: Drop Cap / Initial Reassembly (SURGICAL FIX v3.5)
    # Handles: "F" + "ROM" -> "FROM"
    # =========================================================================
    if (
            is_same_page and
            span_gap == 1 and
            len(prev_text) == 1 and
            prev_text.isupper() and
            prev_text.isalpha()
    ):
        prev_y_end = prev.get("end_y") or (prev.get("bbox", [0, 0, 0, 0])[3])
        next_y_start = next_s.get("bbox", [0, 0, 0, 0])[1]
        y_diff = abs(next_y_start - prev_y_end)

        # Allow slight vertical misalignment (drop caps often offset)
        # but require horizontal flow
        if y_diff < 20.0:
            return True, "same_page:drop_cap_reassembly"

    # =========================================================================
    # CONTINUATION SIGNAL DETECTION (Standard — no terminal punct)
    # GUARDED: Only trigger if structurally adjacent
    # =========================================================================

    # Compute structural adjacency
    is_structurally_adjacent = (
            is_same_page and
            span_gap is not None and
            columns_match and
            span_gap <= _STITCH_MAX_SPAN_GAP
    )

    # Signal A: Lowercase start — REQUIRE structural adjacency (no bypass)
    if has_lowercase_start and is_structurally_adjacent:
        return True, "same_page:lowercase_start"

    # Signal B: Continuation word — require adjacency
    if first_word in _STITCH_CONTINUATION_WORDS:
        if is_structurally_adjacent:
            reason = "same_page:continuation_word" if is_same_page else "continuation_word"
            return True, reason

    # Signal C: Incomplete ending — require adjacency
    if has_incomplete_ending:
        if is_structurally_adjacent:
            reason = "same_page:incomplete_ending" if is_same_page else "incomplete_ending"
            return True, reason

    # Signal D: Continuing punctuation — require adjacency
    if effective_first_char in _STITCH_CONTINUING_PUNCT and first_char not in _STITCH_NEW_SENTENCE_PUNCT:
        if is_structurally_adjacent:
            reason = "same_page:continuing_punct" if is_same_page else "continuing_punct"
            return True, reason

    # Signal E: Number continuation — require adjacency
    if first_char.isdigit() and last_word in _STITCH_NUMBER_CONTEXT_WORDS:
        if is_structurally_adjacent:
            reason = "same_page:number_continuation" if is_same_page else "number_continuation"
            return True, reason

    # No signals found
    return False, "same_page_no_signal" if is_same_page else "no_signal"


# ✦────── c. Sentence Segmentation ──────✦

def _resolve_semantic_continuity(
        window_spans: List[Dict],
        page_span_range: Tuple[int, int],
        *,
        interruption_config: InterruptionConfig = None,
        trace_id: str = None,
) -> None:
    """
    Stage 2 semantic engine (v1):

      - assigns _semantic_disposition in {included, excluded, interruption}
      - never mutates legacy role
      - uses Stage 1 flags as hints (not gates)
      - lossless by default
    """
    if not window_spans:
        return

    # Reserved for future semantic tuning (v2)
    if interruption_config is None:
        interruption_config = InterruptionConfig()
    _ = interruption_config  # Explicitly unused in v1

    start_idx, end_idx = page_span_range

    # ------------------------------------------------------------------
    # Helpers (renamed to avoid shadowing outer variables)
    # ------------------------------------------------------------------
    def _set(target: Dict, disp: str, reason_list: List[str], conf: float = 1.0) -> None:
        target["_semantic_disposition"] = disp
        target["_semantic_reasons"] = reason_list
        target["_semantic_confidence"] = round(float(conf), 2)

    def _text(src: Dict) -> str:
        return (src.get("cleaned_text") or src.get("raw_text") or "").strip()

    def _is_body_like(s: str) -> bool:
        if not s:
            return False
        # Substantial text is always body-like
        if len(s) > 60 or len(s.split()) > 6:
            return True
        # FIX: lowercase heuristic requires substance (prevents single-word labels)
        if s[0].islower() and len(s.split()) >= 3:
            return True
        # Punctuated prose with sufficient length
        if any(ch in s for ch in [".", ",", ";", ":"]) and len(s.split()) >= 4:
            return True
        return False

    # ------------------------------------------------------------------
    # Pass 1: defaults + hard excludes
    # ------------------------------------------------------------------
    for i, span in enumerate(window_spans):
        if not isinstance(span, dict):
            continue

        role = span.get("role", "")
        if role in _SEMANTIC_NEVER_OVERRIDE_ROLES:
            _set(span, _SEM_DISP_EXCLUDED, [f"never_override_role:{role}"], 1.0)
            continue

        # ─────────────────────────────────────────────────────────────────────
        # RONC PROTECTION GATE (Phase 5 Authority)
        # If RONC says must_include, semantic continuity MUST respect it.
        # This overrides all downstream exclusion logic.
        # ─────────────────────────────────────────────────────────────────────
        ronc_contract = span.get("_ronc_contract", {})
        ronc_protection = ronc_contract.get("protection", {})
        ronc_must_include = ronc_protection.get("must_include", False)

        if ronc_must_include:
            # ─────────────────────────────────────────────────────────────────
            # FIX v9.0 PHASE 2: MARGIN PROVENANCE GATE WITHIN RONC PROTECTION
            # RONC protection must NOT unconditionally override layout origin.
            # Margin-origin spans require explicit inline proof before inclusion.
            # ─────────────────────────────────────────────────────────────────
            original_stream = span.get("_original_layout_stream") or ""
            original_role = span.get("_original_role") or ""

            is_margin_origin = (
                    (isinstance(original_stream, str) and original_stream.startswith("margin")) or
                    original_role in ("sidebar", "margin")
            )

            if is_margin_origin:
                text = _text(span)
                word_count = len(text.split()) if text else 0

                # INLINE WHITELIST (strict):
                is_punct_only = len(text) <= 3 and text and not any(c.isalnum() for c in text)
                is_connective = word_count == 1 and text.lower().strip(
                    ",.;:") in _MARGIN_INLINE_CONNECTIVES
                if is_punct_only or is_connective:
                    _set(span, _SEM_DISP_INCLUDED,
                         ["ronc:must_include_protected", "margin_origin:inline_whitelisted"], 1.0)
                    span["_has_semantic_authority"] = True
                    continue
                else:
                    # Margin prose → INTERRUPTION (RONC does not override provenance)
                    _set(span, _SEM_DISP_INTERRUPTION,
                         ["margin_origin:ronc_overridden_by_provenance"], 0.7)
                    continue
            else:
                # Non-margin origin: RONC protection applies normally
                _set(span, _SEM_DISP_INCLUDED, ["ronc:must_include_protected"], 1.0)
                span["_has_semantic_authority"] = True
                continue

        # ─────────────────────────────────────────────────────────────────────
        # FIX v9.1 PHASE 2B: NON-RONC MARGIN PROVENANCE GATE
        #
        # Semantic continuity runs BEFORE RONC contract construction.
        # Therefore, margin-origin spans that bypass RONC and Stage 1 hints
        # MUST be gated here to prevent silent default inclusion.
        # ─────────────────────────────────────────────────────────────────────
        original_stream = span.get("_original_layout_stream") or ""
        original_role = span.get("_original_role") or ""

        is_margin_origin = (
                (isinstance(original_stream, str) and original_stream.startswith(
                    "margin")) or
                original_role in ("sidebar", "margin")
        )

        if is_margin_origin:
            text = _text(span)
            word_count = len(text.split()) if text else 0

            # STRICT INLINE WHITELIST (grammar glue only)
            is_punct_only = (
                    len(text) <= 3 and
                    text and
                    not any(c.isalnum() for c in text)
            )

            is_connective = (
                    word_count == 1 and
                    text.lower().strip(",.;:") in _MARGIN_INLINE_CONNECTIVES
            )

            if is_punct_only or is_connective:
                _set(
                    span,
                    _SEM_DISP_INCLUDED,
                    ["default", "margin_origin:inline_whitelisted"],
                    0.6
                )
                continue

            # All other margin-origin spans FAIL CLOSED here
            _set(
                span,
                _SEM_DISP_INTERRUPTION,
                ["margin_origin:provenance_fail_closed"],
                0.7
            )
            continue

        # Default inclusion applies ONLY to non-margin-origin spans
        _set(span, _SEM_DISP_INCLUDED, ["default"], 0.5)

        candidate_reasons = set(span.get("_exclusion_candidate_reasons") or [])

        if span.get("_exclusion_candidate") and not span.get("_exclusion_protected"):
            # ─────────────────────────────────────────────────────────────────
            # Defense-in-depth: Re-check RONC protection wasn't missed
            # (Should never fire if Patch A ran, but guards edge cases)
            # ─────────────────────────────────────────────────────────────────
            ronc_protected = span.get("_ronc_contract", {}).get("protection", {}).get(
                "must_include", False)
            if ronc_protected:
                _set(span, _SEM_DISP_INCLUDED, ["ronc:protection_override_exclusion_candidate"],
                     0.95)
            elif any(
                    r.startswith("global_header_match_") or r.startswith("global_footer_match_")
                    for r in candidate_reasons
            ):
                _set(span, _SEM_DISP_EXCLUDED, ["global_band_match"], 0.95)
            elif {
                     "noise_punctuation",
                     "noise_digit_only",
                     "noise_single_char",
                 } & candidate_reasons:
                # ─────────────────────────────────────────────────────────
                # PHASE C: Structural section number rescue
                # Digit-only spans that chain (a2) to a heading-role span
                # are section numbers (e.g., "3" → "Web Readability").
                # Rescue instead of excluding as noise.
                # ─────────────────────────────────────────────────────────
                _text_val = (span.get("cleaned_text") or "").strip()
                is_heading_number = False
                if (
                        "noise_digit_only" in candidate_reasons
                        and _text_val.isdigit()
                        and span.get("a2_continues_to_next", False)
                        and span.get("span_index_in_line", -1) == 0
                ):
                    if i + 1 < len(window_spans):
                        next_sp = window_spans[i + 1]
                        if (isinstance(next_sp, dict)
                                and next_sp.get("role") == "heading"
                                and next_sp.get("a2_continues_from_previous", False)):
                            is_heading_number = True

                if is_heading_number:
                    _set(span, _SEM_DISP_INCLUDED,
                         ["stage1_noise_override:heading_section_number"], 0.9)
                    span["_has_semantic_authority"] = True
                else:
                    _set(span, _SEM_DISP_EXCLUDED, ["stage1_noise"], 0.9)

        # Sidebar/margin hint - check for inline continuation potential
        if span.get("_stage1_nonviable_hint"):
            text = _text(span)

            # Short tokens with continuation characteristics should NOT be interruption
            # These are likely inline italics (e.g., "or", "and", terms)
            is_likely_inline = False

            if text:
                first_word = text.split()[0].lower() if text.split() else ""
                # Connectives are almost always inline continuation
                if first_word in {"or", "and", "but", "the", "a", "an", "of", "in", "to",
                                  "is", "are"}:
                    is_likely_inline = True
                # (removed – Phase 2B now handles margin-origin gating)

            if is_likely_inline:
                # Keep as candidate for line-coherent promotion
                _set(
                    span,
                    _SEM_DISP_INCLUDED,
                    ["nonviable_hint_inline_candidate"],
                    0.65,
                )
            else:
                _set(
                    span,
                    _SEM_DISP_INTERRUPTION,
                    ["nonviable_hint_sidebar_or_margin"],
                    0.6,
                )

    # ------------------------------------------------------------------
    # Pass 2: required semantic review overrides
    # ------------------------------------------------------------------
    review_count = 0

    for i, cur_span in enumerate(window_spans):
        if not isinstance(cur_span, dict):
            continue
        if not cur_span.get("_requires_semantic_review"):
            continue

        review_count += 1

        text = _text(cur_span)
        candidate_reasons = set(cur_span.get("_exclusion_candidate_reasons") or [])

        prev_span = window_spans[i - 1] if i > 0 else None
        next_span = window_spans[i + 1] if i + 1 < len(window_spans) else None

        prev_text = _text(prev_span) if isinstance(prev_span, dict) else ""
        next_text = _text(next_span) if isinstance(next_span, dict) else ""

        prev_incomplete = (
                bool(prev_text)
                and prev_text.split()[-1].lower().rstrip(".,;:!?\"')") in _STITCH_INCOMPLETE_ENDINGS
        )

        this_lower = bool(text) and text[0].islower()
        next_lower = bool(next_text) and next_text[0].islower()
        body_like = _is_body_like(text)

        # Visual overlap resolution
        if "visual_overlap" in candidate_reasons:
            # ── Structural exclusion precedence: spans carrying any
            # structural exclusion reason skip rescue entirely. Stage 1
            # correctly identified these spans as non-content (metadata,
            # header/footer bands, header artifacts).
            # Do NOT force-exclude here — the span retains its Pass 1
            # disposition (included/default/0.5) which falls through
            # Priority 2 in _translate_exclusion_flags (requires non-default
            # reasons + confidence >= 0.6), reaching the structural
            # authority gate or Priority 4+ where _exclusion_candidate
            # is respected.
            #
            # NOTE: continue skips bare_caption and invalid_bbox checks
            # for this iteration. Verified harmless — structural exclusion
            # spans are geometrically disjoint from caption patterns and
            # have valid bboxes by prerequisite. If future Pass 2 checks
            # are added that should apply to structural spans, revisit. ──
            if candidate_reasons & _STRUCTURAL_EXCLUSION_REASONS:
                if trace_id:
                    logger.debug(
                        "[%s] Pass 2: visual_overlap rescue skipped for "
                        "structural exclusion span. reasons=%s text=%r cid=%s",
                        trace_id,
                        sorted(candidate_reasons & _STRUCTURAL_EXCLUSION_REASONS),
                        text[:50],
                        cur_span.get("_canonical_span_id")
                    )
                continue
            # =================================================================
            # PATCH 7B-E: Rescue body-like / continuation prose BEFORE diagram_label hard exclude
            #
            # Problem: Stage-1 can over-attach "diagram_label" to real prose near figures.
            # If we hard-exclude here, _translate_exclusion_flags Priority 0 will "continue"
            # and later rescue logic never runs.
            #
            # Guardrails:
            #   - Only rescue in body_col* streams (prevents margin leakage)
            #   - Block boilerplate / documentation artifacts
            #   - Allow short continuation tokens (e.g., "are called", "gamma neurons")
            #   - Keep figure_label hard-excluded
            # =================================================================
            layout_stream = cur_span.get("layout_stream", "")
            is_body_stream = isinstance(layout_stream, str) and layout_stream.startswith("body_col")

            text_lower = text.lower() if text else ""
            is_boilerplate = (
                    "html" in text_lower or
                    "css" in text_lower or
                    "examples of how to" in text_lower or
                    text_lower.startswith("figure ") or
                    text_lower.startswith("table ")
            )

            # Short continuation heuristic (keeps stitch continuity alive)
            # --- PATCH 7B-F: Word count (required for all subsequent checks) ---
            word_count = len(text.split()) if text else 0

            # --- PATCH 7B-F: Block-level structural context (page-scoped) ---
            cur_page = cur_span.get("page_number")
            block_id = cur_span.get("block_id")

            # Guard: Only compute block context if both identifiers are present
            is_diagram_heavy_block = False
            if block_id is not None and cur_page is not None:
                block_spans = [
                    s for s in window_spans
                    if isinstance(s, dict)
                       and s.get("block_id") == block_id
                       and s.get("page_number") == cur_page
                ]

                diagram_label_count = sum(
                    1 for s in block_spans
                    if "diagram_label" in (s.get("_exclusion_candidate_reasons") or [])
                )

                diagram_ratio = diagram_label_count / max(len(block_spans), 1)

                # CRITICAL: strictly greater than (block at exactly 50% is NOT diagram-heavy)
                is_diagram_heavy_block = diagram_ratio > 0.5

            # --- PATCH 7B-F: Short continuation heuristic (>= 2 words ONLY) ---
            is_short_continuation = (
                    2 <= word_count <= 5 and
                    text and (
                            text[0].islower() or text[0] in ",;:.)"
                    )
            )

            # True diagram-label-ish tokens: short + uppercase-ish (prevents label leakage)
            is_true_label_like = (
                    bool(text) and
                    word_count <= 3 and
                    text.isupper()
            )

            # --- PATCH 7B-F: Neighbor-aware single-word glue preservation ---
            CONTINUATION_GLUE_WORDS = {
                "called", "are", "is", "was", "were", "and", "or", "but"
            }

            is_glue_word_rescue = False
            if word_count == 1 and text.lower() in CONTINUATION_GLUE_WORDS:
                if prev_incomplete:
                    is_glue_word_rescue = True
                elif next_text and _is_body_like(next_text):
                    is_glue_word_rescue = True

            # FM3: A2 BODY ANCHOR (STRUCTURAL, PASS-2)
            has_a2_body_anchor = False
            if cur_span.get("a2_continues_to_next") or cur_span.get("a2_continues_from_previous"):
                cur_lid = cur_span.get("line_id")
                cur_sil = cur_span.get("span_index_in_line", -1)
                if cur_lid is not None:
                    for ws in window_spans:
                        if ws is cur_span:
                            continue
                        if ws.get("line_id") != cur_lid:
                            continue
                        ws_sil = ws.get("span_index_in_line", -1)
                        in_chain_dir = (
                                (cur_span.get("a2_continues_to_next") and ws_sil > cur_sil) or
                                (cur_span.get("a2_continues_from_previous") and ws_sil < cur_sil)
                        )
                        if not in_chain_dir:
                            continue
                        ws_survives = (
                                not ws.get("_exclusion_candidate")
                                or ws.get("_exclusion_protected")
                        )
                        if ws_survives:
                            has_a2_body_anchor = True
                            break
            # Cross-line fallback: if same-line_id search found nothing,
            # follow the a2 chain to the immediate neighbor in reading order.
            # The a2 flag itself proves glyph-level adjacency — line_id is
            # redundant when the extraction engine has already asserted the link.
            if not has_a2_body_anchor:
                cur_idx_in_window = None
                for wi, ws in enumerate(window_spans):
                    if ws is cur_span:
                        cur_idx_in_window = wi
                        break
                if cur_idx_in_window is not None:
                    if cur_span.get(
                            "a2_continues_from_previous") and cur_idx_in_window > 0:
                        prev_ws = window_spans[cur_idx_in_window - 1]
                        if (isinstance(prev_ws, dict)
                                and prev_ws.get("a2_continues_to_next")):
                            if (not prev_ws.get("_exclusion_candidate")
                                    or prev_ws.get("_exclusion_protected")):
                                has_a2_body_anchor = True
                    if (not has_a2_body_anchor
                            and cur_span.get("a2_continues_to_next")
                            and cur_idx_in_window + 1 < len(window_spans)):
                        next_ws = window_spans[cur_idx_in_window + 1]
                        if (isinstance(next_ws, dict)
                                and next_ws.get("a2_continues_from_previous")):
                            if (not next_ws.get("_exclusion_candidate")
                                    or next_ws.get("_exclusion_protected")):
                                has_a2_body_anchor = True
            rescue_as_body = (
                is_body_stream and
                not is_boilerplate and
                (
                    (has_a2_body_anchor and not is_true_label_like) or
                    (
                        not is_true_label_like and
                        not (is_diagram_heavy_block and word_count <= 3
                             and not is_glue_word_rescue) and
                        (
                            body_like or
                            is_short_continuation or
                            is_glue_word_rescue or
                            (prev_incomplete and this_lower) or
                            (this_lower and next_lower)
                        )
                    )
                )
            )

            if rescue_as_body:
                cur_span["_inside_figure_rescued"] = True
                # ─────────────────────────────────────────────────────────
                # ROLE PROMOTION: Stage 3 gates on role, not disposition.
                # Mirrors _apply_continuity_role_resolution's mutation for
                # spans that Phase 1.5 could not reach (e.g. exclusion
                # candidates rescued here via a2 anchor or body heuristic).
                # ─────────────────────────────────────────────────────────
                if cur_span.get("role") == "inside_figure":
                    cur_span["_original_geometry_role"] = cur_span["role"]
                    cur_span["role"] = "body"
                    cur_span["_continuity_override"] = True
                    cur_span["_continuity_override_reason"] = (
                        "semantic_rescue:a2_body_anchor" if has_a2_body_anchor
                        else "semantic_rescue:body_like"
                    )
                if has_a2_body_anchor:
                    _set(cur_span, _SEM_DISP_INCLUDED,
                         ["visual_overlap:a2_body_anchor_rescue"], 0.90)
                else:
                    _set(cur_span, _SEM_DISP_INCLUDED,
                         ["visual_overlap:body_like_rescue"], 0.85)
                # ─────────────────────────────────────────────────────────────────
                # P5 FIX: Protect headings from diagram_label exclusion
                # Headings may be short and contain technical terms (triggering
                # diagram_label heuristic), but they are structural content that
                # must be preserved for document navigation.
                # ─────────────────────────────────────────────────────────────────
            elif cur_span.get("role") in ("heading",
                                              "subheading"):
                _set(cur_span, _SEM_DISP_INCLUDED, ["heading_protection:structural_content"], 0.95)

            # HARD EXCLUSION: keep figure_label absolute, but allow diagram_label only if not rescued
            elif "figure_label" in candidate_reasons:
                _set(cur_span, _SEM_DISP_EXCLUDED, ["figure_label:hard_exclude"], 0.95)

            elif "diagram_label" in candidate_reasons:
                _set(cur_span, _SEM_DISP_EXCLUDED, ["diagram_label:hard_exclude"], 0.95)

            elif body_like or (prev_incomplete and this_lower) or (this_lower and next_lower):
                _set(cur_span, _SEM_DISP_INCLUDED, ["visual_overlap_override:include"], 0.85)

            else:
                _set(cur_span, _SEM_DISP_INTERRUPTION, ["visual_overlap:non_prose"], 0.7)

        # Bare caption resolution
        if "bare_caption" in candidate_reasons:
            if body_like:
                _set(cur_span, _SEM_DISP_INCLUDED, ["bare_caption_override:include"], 0.8)
            else:
                _set(cur_span, _SEM_DISP_INTERRUPTION, ["bare_caption:interrupt"], 0.65)

        # Invalid bbox / empty text
        if "invalid_bbox" in candidate_reasons or "empty" in candidate_reasons:
            if body_like:
                _set(cur_span, _SEM_DISP_INCLUDED, ["bbox_or_empty:text_only_include"], 0.75)

    if trace_id:
        logger.debug(
            "[%s] Semantic continuity resolved: window=%d current_range=(%d,%d) reviewed=%d",
            trace_id, len(window_spans), start_idx, end_idx, review_count
        )


def _reconstruct_text_for_segmentation(
        spans: List[Dict],
        trace_id: str = None
) -> tuple[str, List[Dict], List[int]]:
    """
    Reconstruct a stable text stream for sentence segmentation.
    ARCHITECTURAL GUARDRAIL:
    This method is the SOLE authority for stream shaping.
    It may introduce structural whitespace separators only.
    This method must NOT:
      - Perform semantic deletion
      - Perform sentence repair or healing
      - Perform role-based or margin-based hard breaks
      - Modify text content beyond structural spacing
    Stream rules:
      - Column change → '\\n\\n'
      - Paragraph change → '\\n'
      - Page change → ' '
      - Default join → ' '
    """
    if not spans:
        return "", [], []

    # =========================================================================
    # PHASE 2.8: Pre-filter non-TTS viable spans BEFORE ordering
    # CRITICAL: Non-viable roles must not contaminate semantic chain ordering.
    # Filtering after ordering allows sidebars/footnotes to influence body text position.
    # =========================================================================
    tts_viable_spans = [sp for sp in spans if _is_tts_viable_span(sp, trace_id)]

    if trace_id:
        excluded_count = len(spans) - len(tts_viable_spans)
        if excluded_count > 0:
            logger.debug(
                "[%s] Phase 2.8 pre-filter: %d/%d spans excluded before ordering",
                trace_id, excluded_count, len(spans)
            )

    if not tts_viable_spans:
        if trace_id:
            logger.warning("[%s] Pre-filter excluded all spans — no TTS content", trace_id)
        return "", [], []

    # =========================================================================
    # PHASE 2.5: Enforce reading order before reconstruction
    # CRITICAL: Use integer fields to avoid string sort bugs ("1:3:10" < "1:3:2")
    # =========================================================================

    # -------------------------------------------------------------------------
    # STEP 1 (RONC v2.1): Apply semantic chain ordering BEFORE index attachment
    # This respects atomic units and chain links established by RONC Phase 6.
    # Ordering hierarchy: atomic_unit → chain_role → A2_edges → confidence → geometry
    # NOTE: Input is pre-filtered to TTS-viable spans only (Phase 2.8)
    # -------------------------------------------------------------------------
    chain_ordered_spans = _order_spans_by_semantic_chains(tts_viable_spans, trace_id=trace_id)

    # STEP 2: Attach original index from full spans list using canonical identity
    # NOTE: This preserves auditability even after Phase 2.8 pre-filtering
    # Canonical IDs are stable across phases and required for auditability
    span_to_orig_idx = {
        sp.get("_canonical_span_id"): i
        for i, sp in enumerate(spans)
        if sp.get("_canonical_span_id") is not None
    }

    indexed_spans = []
    for i, sp in enumerate(chain_ordered_spans):
        cid = sp.get("_canonical_span_id")
        orig_idx = span_to_orig_idx.get(cid, i)
        indexed_spans.append((orig_idx, sp))

    if trace_id:
        missing_cids = sum(
            1 for _, sp in indexed_spans
            if sp.get("_canonical_span_id") is None
        )
        if missing_cids:
            logger.warning(
                "[%s] %d spans missing canonical IDs during reconstruction; "
                "ordering stability reduced",
                trace_id, missing_cids
            )

    # =========================================================================
    # PHASE 2.8 DIAGNOSTIC: Detect cross-page chain adjacency before page sort
    # Chain ordering may suggest A(p1)→B(p2)→C(p1) but page sort forces A,C,B.
    # This diagnostic surfaces when chain intent conflicts with page grouping.
    # =========================================================================
    if trace_id and indexed_spans:
        chain_order_pages = [sp.get('page_number', 0) for _, sp in indexed_spans]

        # Detect ONLY true adjacency violations (semantic chain wants to cross pages)
        cross_page_breaks = 0
        for i in range(1, len(chain_order_pages)):
            if chain_order_pages[i] < chain_order_pages[i - 1]:
                cross_page_breaks += 1

        if cross_page_breaks > 0:
            logger.debug(
                "[%s] Chain ordering suggests %d cross-page adjacencies that page sort may defeat. "
                "Chain page sequence (first 20): %s",
                trace_id, cross_page_breaks, chain_order_pages[:20]
            )

    # STEP 3: Secondary sort by page only (chain order preserved within page)
    # This ensures cross-page rendering progresses correctly while maintaining
    # intra-page semantic ordering from chain analysis.
    sorted_spans = sorted(indexed_spans, key=lambda item: (
        item[1].get('page_number', 0),
    ))

    full_text = ""
    span_map: List[Dict] = []
    char_to_span: List[int] = []

    # Track previous span for boundary detection
    prev_column: Optional[int] = None
    prev_para: Optional[int] = None
    prev_page: Optional[int] = None
    prev_block: Optional[int] = None

    # REVISION B: Initialize reference tracker for A2 signals
    prev_span_ref: Optional[Dict] = None

    a2_signal_joins = 0
    ronc_atomic_welds = 0
    ronc_boundary_breaks = 0
    chain_protection_overrides = 0  # RONC v2.1: spans rescued from _tts_excluded
    boundary_contract_welds = 0  # RONC v2.1: truncated→continuation joins

    
    for orig_idx, span in sorted_spans:
        # X-RAY DEBUGGER
        cid = span.get("_canonical_span_id")
        if cid in ("P2:54", "P2:55", "P4:14", "P5:31"):
            # Get the previous span's role to check if the Split Logic should fire
            prev_cid = prev_span_ref.get("_canonical_span_id") if prev_span_ref else "None"
            prev_role = prev_span_ref.get("role") if prev_span_ref else "None"
            print(
                f"[X-RAY] Processing {cid} | Role: {span.get('role')} | PrevRef: {prev_cid} ({prev_role})")
        if not isinstance(span, dict):
            continue

        # =====================================================================
        # DEFENSE-IN-DEPTH: Should never fire after Phase 2.8 pre-filter
        # =====================================================================
        if span.get("_tts_excluded", False):
            contract = span.get("_ronc_contract") or {}
            authority = contract.get("authority")
            is_chain_protected = (
                    span.get("_a2_qualified", False) or
                    authority in (_RONC_V2_AUTHORITY_STRONG, _RONC_V2_AUTHORITY_WEAK) or
                    span.get("_ronc_atomic_unit_id") is not None
            )
            if is_chain_protected:
                chain_protection_overrides += 1
            else:
                if trace_id:
                    logger.error(
                        "[%s] Pre-filter gap: _tts_excluded span in loop. role=%s text=%r",
                        trace_id, span.get("role"), (span.get("cleaned_text") or "")[:40]
                    )
                continue

        # =====================================================================
        # CONTRACT CHECK: Disposition should be set by flag translator
        # (Fail-open for backward compatibility, but make it loud.)
        # =====================================================================
        disp = span.get("_semantic_disposition")
        if disp is None:
            if trace_id:
                logger.warning(
                    "[%s] Span with unset _semantic_disposition reached reconstruction. "
                    "Defaulting to INCLUDED (checkpoint bypass). role=%s text=%r",
                    trace_id,
                    span.get("role"),
                    (span.get("cleaned_text") or span.get("raw_text") or "")[:40],
                )
            disp = _SEM_DISP_INCLUDED  # backward-compatible default

        role = span.get("role", "")

        # Never-override roles are always excluded
        if role in _SEMANTIC_NEVER_OVERRIDE_ROLES:
            continue

        # Excluded spans never enter reconstruction
        if disp == _SEM_DISP_EXCLUDED:
            continue

        # Interruption spans: allow downstream config behaviors
        if disp == _SEM_DISP_INTERRUPTION:
            # ─────────────────────────────────────────────────────────────────
            # P11 FIX: Respect rescue authority (Resolves Type B - Ghost Spans)
            # P9 explicitly rescued this span via RONC must_include protection.
            # Skipping rescued interruptions violates the rescue contract.
            # Also respect stream authority for body_col content.
            # ─────────────────────────────────────────────────────────────────
            text = (span.get("cleaned_text") or "").strip()
            stream = span.get("layout_stream", "")
            is_stream_protected = len(text) > 25 and str(stream).startswith("body_col")

            if not (span.get("_tts_rescued") or is_stream_protected):
                # v1 conservative: skip unprotected interruption spans
                continue

        # =====================================================================
        # PHASE 2.7 GUARD 0: Inline-Role Override (Chameleon Logic)
        # Rescues valid body text mislabeled as Sidebar/Subheading/Heading.
        # CRITICAL: Do NOT rescue actual margin content.
        # =====================================================================
        effective_role = span.get("role", "")
        if effective_role in _TTS_NON_VIABLE_ROLES:
            is_actual_margin = span.get("is_margin_content", False)
            was_line_rescued = span.get("_tts_rescued", False)

            # Phase 1.3 rescue overrides margin classification
            if not is_actual_margin or was_line_rescued:

                is_inline = span.get("span_index_in_line", 0) > 0
                is_a2_source = span.get("a2_continues_to_next", False)
                is_a2_target = (
                        prev_span_ref is not None and
                        prev_span_ref.get("a2_continues_to_next", False)
                )
                if is_inline or is_a2_source or is_a2_target:
                    effective_role = TextRole.BODY.value

        # =====================================================================
        # PHASE 2.7 GUARD 1: Non-viable role quarantine
        # Uses effective_role (post-chameleon) to spare rescued inline spans.
        # These roles must not influence reconstruction boundaries or text stream.
        # =====================================================================
        if effective_role in _TTS_NON_VIABLE_ROLES:
            continue

        text = (span.get("cleaned_text") or "")
        if not text:
            continue

        # ─────────────────────────────────────────────────────────────────
        # FIX v5.2: Heading Sentence Isolation (String Literals)
        # Force headings to be standalone sentences with terminal punctuation.
        # Uses original 'role' to avoid chameleon demotion issues.
        # ─────────────────────────────────────────────────────────────────
        if role in ("heading", "subheading"):
            text_clean = text.rstrip()
            if text_clean and text_clean[-1] not in ".!?:":
                text = text_clean + "."

        # =================================================================
        # RONC PHASE 6: Extract atomic unit metadata for boundary decisions
        # =================================================================
        prev_ronc_unit = (
            prev_span_ref.get("_ronc_atomic_unit_id")
            if isinstance(prev_span_ref, dict) else None
        )
        curr_ronc_unit = span.get("_ronc_atomic_unit_id")
        same_atomic_unit = (
                prev_ronc_unit is not None
                and curr_ronc_unit is not None
                and prev_ronc_unit == curr_ronc_unit
        )
        prev_break_after = (
            prev_span_ref.get("_ronc_break_after")
            if isinstance(prev_span_ref, dict) else None
        )

        curr_column = span.get("column_index", 0) or 0
        curr_para = span.get("paragraph_index", 0) or 0
        curr_page = span.get("page_number", 0)
        curr_block = span.get("block_id")

        # Determine prefix based on boundaries
        prefix = ""  # Default for first span (no predecessor)
        if full_text:
            prefix = None  # Reset for priority resolution

        # ─────────────────────────────────────────────────────────────────
        # FIX v6.5d: Structural Punctuation Glue (Priority -1)
        # Prevents isolation of punctuation spans (e.g., P2:60 comma) by layout breaks.
        # Punctuation should adhere to preceding word, not form orphaned lines
        # that get rejected by sentence-level viability (too_short).
        # ─────────────────────────────────────────────────────────────────
        if prefix is None:
            _punct_text = (span.get("cleaned_text") or "").strip()
            # Single-char trailing punctuation glues to previous text
            if len(_punct_text) == 1 and _punct_text in ",;:.!?)]}":
                prefix = ""
            # Multi-char punctuation (e.g., "..." or closing quotes) also glues
            elif len(_punct_text) <= 3 and _punct_text and not any(
                    c.isalnum() for c in _punct_text):
                prefix = ""

        # PRIORITY 0: A2 override
        if prev_span_ref and prev_span_ref.get("a2_continues_to_next", False):
            prefix = " "
            a2_signal_joins += 1
            if same_atomic_unit:
                ronc_atomic_welds += 1

        # PRIORITY 0.5: Boundary contract (hyphenated weld)
        # Changed 'elif' to 'if' because of the inserted block above.
        if prefix is None and prev_span_ref:
            prev_boundary = (prev_span_ref.get("_ronc_contract") or {}).get(
                "boundary") or {}
            curr_boundary = (span.get("_ronc_contract") or {}).get("boundary") or {}
            if (
                    prev_boundary.get("end", {}).get(
                        "label") == _RONC_V2_END_LABEL_TRUNCATED
                    and curr_boundary.get("start", {}).get("label") in (
                    _RONC_V2_START_LABEL_CONTINUATION,
                    _RONC_V2_START_LABEL_FRAGMENT
            )
            ):
                prefix = ""
                boundary_contract_welds += 1


        # PRIORITY 1: Column break
        if prefix is None and prev_column is not None and curr_column != prev_column and curr_page == prev_page:
            prefix = " " if same_atomic_unit else "\n\n"
            if same_atomic_unit:
                ronc_atomic_welds += 1

        # PRIORITY 2: Page break
        if prefix is None and prev_page is not None and curr_page != prev_page:
            prefix = " "

        # PRIORITY 2.5: Block break (same page, different structural block)
        if prefix is None and prev_block is not None and curr_block is not None and curr_block != prev_block and curr_page == prev_page:
            prefix = " " if same_atomic_unit else "\n"
            if same_atomic_unit:
                ronc_atomic_welds += 1

        # PRIORITY 3: Paragraph break
        if prefix is None and prev_para is not None and curr_para != prev_para:
            prefix = " " if same_atomic_unit else "\n"
            if same_atomic_unit:
                ronc_atomic_welds += 1

        # PRIORITY 3.5: RONC tail break
        if prefix is None and prev_break_after and not same_atomic_unit:
            prefix = "\n"
            ronc_boundary_breaks += 1

        # PRIORITY 4: Default
        if prefix is None:
            prefix = " "

        # FINAL OVERRIDE: Heading Barrier
        # Must run AFTER all structural prefix logic so it cannot be overridden.
        if prev_span_ref:
            prev_role = prev_span_ref.get("role", "")
            curr_is_heading = role in ("heading", "title", "subheading")
            prev_is_heading = prev_role in ("heading", "title", "subheading")

            if curr_is_heading != prev_is_heading:
                if full_text and full_text.rstrip()[-1:] not in ".!?:":
                    prefix = ".\n\n"
                else:
                    prefix = "\n\n"

        # GUARDRAIL: prefix must be structural whitespace only
        # NOTE: ".\n\n" is allowed for Phase E heading boundary termination
        if prefix not in ("", " ", "\n", "\n\n", ".\n\n"):
            prefix = " "

        # Add prefix characters to source map (map to previous span or -1)
        prev_span_idx = span_map[-1]["span_index"] if span_map else -1
        prefix_map_idx = prev_span_idx if prefix == " " else -1
        for _ in prefix:
            char_to_span.append(prefix_map_idx)

        # Build span map entry
        start_idx = len(full_text) + len(prefix)
        full_text += prefix + text
        end_idx = len(full_text)

        # Add text characters to source map (map to current span)
        for _ in text:
            char_to_span.append(orig_idx)

            # NOTE: span_index now correctly refers to index in input list
        span_map.append({
            "start": start_idx,
            "end": end_idx,
            "span_index": orig_idx,
            "column_index": curr_column,
            "paragraph_index": curr_para,
            "_canonical_span_id": span.get("_canonical_span_id"),
            "role": role,
        })

        # Update tracking
        prev_column = curr_column
        prev_para = curr_para
        prev_page = curr_page
        prev_block = curr_block
        prev_span_ref = span

    if trace_id:
        logger.debug(
            "[%s] Reconstructed text: %d chars, %d spans, source map size: %d, "
            "a2_signal_joins: %d, ronc_atomic_welds: %d, ronc_boundary_breaks: %d, "
            "chain_protection_overrides: %d, boundary_contract_welds: %d",
            trace_id, len(full_text), len(span_map), len(char_to_span),
            a2_signal_joins, ronc_atomic_welds, ronc_boundary_breaks,
            chain_protection_overrides, boundary_contract_welds
        )

    # =========================================================================
    # PHASE 2.7 GUARD 3: Safety assertion (non-destructive)
    # Detect structural anomalies without altering output.
    # =========================================================================
    open_parens = full_text.count("(")
    close_parens = full_text.count(")")
    if open_parens != close_parens:
        if trace_id:
            logger.warning(
                "[%s] Phase 2.7: Unbalanced parentheses in stream (open=%d, close=%d)",
                trace_id, open_parens, close_parens
            )

    open_brackets = full_text.count("[")
    close_brackets = full_text.count("]")
    if open_brackets != close_brackets:
        if trace_id:
            logger.warning(
                "[%s] Phase 2.7: Unbalanced square brackets in stream "
                "(open=%d, close=%d, likely orphaned by citation exclusion)",
                trace_id, open_brackets, close_brackets
            )

    return full_text, span_map, char_to_span


def _should_join_without_space(prev_text: str, curr_text: str) -> bool:
    """
    Determine if two spans should be joined WITHOUT a space between them.

    PROFESSIONAL v3.1 — Conservative De-Kerning

    Core Principle:
        Only join when there's STRONG evidence of a mid-word kerning split.
        False negatives (keeping space) are far less harmful than
        false positives (merging separate words).

    Args:
        prev_text: Previous span's text.
        curr_text: Current span's text.

    Returns:
        True if spans should join without space (kerning detected).
        False if space should be preserved (default safe behavior).
    """

    # This method MUST remain pairwise and stateless.
    # It MUST NOT inspect layout, role, page, or semantic context.

    if not prev_text or not curr_text:
        return False

    prev_stripped = prev_text.rstrip()
    curr_stripped = curr_text.lstrip()

    if not prev_stripped or not curr_stripped:
        return False

    prev_lower = prev_stripped.lower()
    curr_lower = curr_stripped.lower()

    # =========================================================================
    # GLOBAL PROTECTION: Title Case Valid Words
    # =========================================================================
    # If curr is title case AND is a valid English word, it's starting a
    # new sentence. NEVER join. Blocks: "IKIBOOKS" + "Our" → "IKIBOOKS Our"
    # =========================================================================
    if len(curr_stripped) >= 2:
        is_title_case = curr_stripped[0].isupper() and curr_stripped[1:].islower()
        if is_title_case and curr_lower in VALID_SHORT_WORDS:
            return False

    # =========================================================================
    # PATTERN 1: Single uppercase letter + uppercase continuation
    # "F" + "ROM" → "FROM"
    # Exception: "I" is a valid word.
    # =========================================================================
    if len(prev_stripped) == 1 and prev_stripped.isupper():
        if curr_stripped[0].isupper():
            if prev_stripped != 'I':
                return True

    # =========================================================================
    # PATTERN 2: Short uppercase (2-3 chars) + uppercase continuation
    # "FR" + "OM" → "FROM"
    # Protection: Skip if prev is a valid short word (uppercase form).
    # =========================================================================
    if 2 <= len(prev_stripped) <= 3 and prev_stripped.isupper():
        if curr_stripped[0].isupper():
            if prev_lower not in VALID_SHORT_WORDS:
                valid_uc_endings = {'ED', 'ER', 'LY', 'AL', 'ON', 'AN', 'IN', 'OR'}
                prev_ending = prev_stripped[-2:] if len(prev_stripped) >= 2 else ""
                if prev_ending not in valid_uc_endings:
                    return True

    # =========================================================================
    # PATTERN 3: Extended uppercase kerning (4+ chars)
    # "FROMW" + "IKIBOOKS" → join (consonant → vowel, both all-caps)
    # STRICT: curr must also be ALL UPPERCASE (not title case).
    # =========================================================================
    if len(prev_stripped) >= 4 and prev_stripped.isupper():
        # Require curr to be all uppercase OR at least first 2 chars uppercase
        curr_is_all_caps = curr_stripped.isupper()
        curr_starts_all_caps = len(curr_stripped) >= 2 and curr_stripped[:2].isupper()

        if curr_is_all_caps or curr_starts_all_caps:
            vowels = set('AEIOU')
            prev_ends_consonant = prev_stripped[-1] not in vowels
            curr_starts_vowel = curr_stripped[0] in vowels

            if prev_ends_consonant and curr_starts_vowel:
                return True

    # =========================================================================
    # PATTERN 4: Lowercase suffix joining
    # "textu" + "re" → "texture"
    # PROTECTED: prev must NOT be a valid word.
    # =========================================================================
    common_suffixes = frozenset({
        're', 'er', 'ar', 'ed', 'es', 'ing', 'tion', 'sion',
        'ment', 'ness', 'less', 'able', 'ible', 'ous', 'ious', 'ly', 'ty',
        'ry', 'cy', 'al', 'el', 'le', 'ure', 'ture', 'sure', 'ive', 'ise',
        'ize', 'ful', 'ish', 'ase', 'ose', 'ide', 'ate', 'ine',
    })

    if 2 <= len(prev_stripped) <= 8:
        if prev_stripped[-1].islower() and curr_stripped[0].islower():
            if len(curr_stripped) <= 5 and curr_lower in common_suffixes:
                # Block if prev is a valid English word
                if prev_lower in VALID_SHORT_WORDS:
                    return False
                # Additional fragment validation: check for vowels
                prev_vowels = sum(1 for c in prev_lower if c in 'aeiou')
                if prev_vowels == 0:
                    return True  # No vowels = definite fragment
                # Has vowels: check ending pattern
                valid_endings = {'ed', 'er', 'ly', 'al', 'le', 're', 'an', 'en', 'in', 'on', 'ar',
                                 'or'}
                if prev_lower[-2:] not in valid_endings:
                    return True

    # =========================================================================
    # PATTERN 5: Scientific compound prefixes
    # "mechano" + "receptors" → "mechanoreceptors"
    # =========================================================================
    compound_prefixes = frozenset({
        'mechano', 'thermo', 'photo', 'electro', 'neuro', 'cardio', 'osteo',
        'cyto', 'hemo', 'haemo', 'immuno', 'bio', 'micro', 'macro', 'poly',
        'mono', 'anti', 'auto', 'endo', 'exo', 'hyper', 'hypo', 'intra',
        'inter', 'meta', 'para', 'peri', 'post', 'pre', 'proto', 'pseudo',
        'semi', 'sub', 'super', 'trans', 'ultra', 'extra', 'infra',
        'extrafusal', 'intrafusal', 'noci', 'proprio', 'chemo', 'baro',
        'osmo', 'soma', 'somato', 'viscero',
        # HARDENED: Added high-frequency scientific prefixes
        'non', 'multi', 'bi', 'tri', 'uni',
    })

    if prev_lower in compound_prefixes:
        if curr_stripped[0].islower():
            return True

    # =========================================================================
    # PATTERN 6: Title case fragment + suffix
    # "Someth" + "ing" → "Something"
    # PROTECTED: prev must NOT be a valid word.
    # =========================================================================
    if len(prev_stripped) >= 4:
        is_title_case = prev_stripped[0].isupper() and prev_stripped[1:].islower()
        if is_title_case:
            if len(curr_stripped) <= 5 and curr_lower in common_suffixes:
                if prev_lower not in VALID_SHORT_WORDS:
                    valid_endings = {'ed', 'er', 'ly', 'al', 'le', 're', 'an', 'en', 'in', 'on',
                                     'ar', 'or'}
                    if prev_lower[-2:] not in valid_endings:
                        return True

    # =========================================================================
    # DEFAULT: Preserve space (safe behavior)
    # =========================================================================
    return False


def _detect_synthetic_figures(
        spans: List[Dict],
        page_width: float,
        page_height: float,
        trace_id: str = None
) -> List[BboxTuple]:
    """
    Detect diagram regions by identifying clusters of label-like text.

    PROFESSIONAL v3.0 (Corrected):
        - Added horizontal spread check to distinguish diagrams from lists
        - Lists: vertically stacked (low X variance)
        - Diagrams: scattered in X and Y (high X variance)

    Args:
        spans: List of span dictionaries with bbox and text.
        page_width: Page width in points.
        page_height: Page height in points.
        trace_id: Optional trace ID for logging.

    Returns:
        List of (x0, y0, x1, y1) bounding boxes for synthetic figures.
    """
    # =========================================================================
    # Step 1: Identify Label-Like Spans
    # =========================================================================
    label_candidates: List[Dict] = []

    for span in spans:
        text = span.get("raw_text", "").strip()
        bbox = span.get("bbox")

        if not text or not bbox or len(bbox) < 4:
            continue

        word_count = len(text.split())
        char_count = len(text)
        tail = text.rstrip()
        has_sentence_punct = (tail[-1] in '.!?:;') if tail else False

        # Prose indicators
        prose_starters = {'the', 'a', 'an', 'this', 'that', 'these', 'those',
                          'it', 'they', 'we', 'he', 'she', 'there', 'here'}
        first_word = text.split()[0].lower() if text.split() else ""
        starts_with_prose = first_word in prose_starters

        # Label criteria
        is_label_like = (
                1 <= word_count <= 6 and
                char_count <= 60 and
                not starts_with_prose and
                (not has_sentence_punct or word_count <= 3)
        )

        if is_label_like:
            label_candidates.append({
                "text": text,
                "text_font_size": span.get("font_size", 0),
                "bbox": bbox,
                "center_x": (bbox[0] + bbox[2]) / 2,
                "center_y": (bbox[1] + bbox[3]) / 2,
            })

    if len(label_candidates) < 2:
        return []

    # =========================================================================
    # Step 2: Spatial Clustering (Simple Proximity Grouping)
    # =========================================================================
    # HARDENED: Scale distance by font size to handle High-DPI scans
    font_sizes = [
        s["text_font_size"] for s in label_candidates
        if "text_font_size" in s and s["text_font_size"] > 0
    ]
    median_fs = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0

    # Cluster distance = ~8 lines of text / characters. Scales with resolution.
    CLUSTER_DISTANCE = max(median_fs * 6.0, 70.0)

    clusters: List[List[Dict]] = []
    used: Set[int] = set()

    for i, label in enumerate(label_candidates):
        if i in used:
            continue

        cluster = [label]
        used.add(i)

        # Iteratively expand cluster
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(label_candidates):
                if j in used:
                    continue

                for member in cluster:
                    dx = abs(other["center_x"] - member["center_x"])
                    dy = abs(other["center_y"] - member["center_y"])

                    # REVISION 3: Removed Column-Aware Clustering
                    # Diagrams often span multiple columns (e.g. wide figures).
                    # We must allow large horizontal gaps for labels like "Force" (left) and "Length" (right).
                    # The previous guard prevented detecting wide diagrams, causing "Phantom Columns".

                    # Euclidean distance for local proximity
                    distance = (dx ** 2 + dy ** 2) ** 0.5

                    # HARDEN (anti-bridging):
                    # Prevent transitive "chain" merges across unrelated label groups.
                    # Even in wide diagrams, a single hop should not span most of the page.
                    MAX_HOP_X = page_width * 0.45
                    MAX_HOP_Y = page_height * 0.35
                    if dx > MAX_HOP_X or dy > MAX_HOP_Y:
                        continue

                    if distance <= CLUSTER_DISTANCE:
                        cluster.append(other)
                        used.add(j)
                        changed = True
                        break

        if len(cluster) >= 2:
            clusters.append(cluster)

    # =========================================================================
    # Step 3: Validate Clusters (NEW: Horizontal Spread Check)
    # =========================================================================
    MIN_DIMENSION_SPREAD = max(median_fs * 5.0, 50.0)

    valid_clusters: List[List[Dict]] = []

    for cluster in clusters:
        x_coords = [l["center_x"] for l in cluster]
        y_coords = [l["center_y"] for l in cluster]

        x_spread = max(x_coords) - min(x_coords)
        y_spread = max(y_coords) - min(y_coords)

        # FIXED: Accept if EITHER dimension shows scatter
        if x_spread >= MIN_DIMENSION_SPREAD or y_spread >= MIN_DIMENSION_SPREAD:
            valid_clusters.append(cluster)
        elif trace_id:
            logger.debug(
                "[%s] Rejected synthetic figure: %d labels, spread (X=%.1f, Y=%.1f) < %d (too compact)",
                trace_id, len(cluster), x_spread, y_spread, MIN_DIMENSION_SPREAD
            )

    # =========================================================================
    # Step 4: Create Synthetic Figure Bounding Boxes
    # =========================================================================
    PADDING = 20

    synthetic_figures: List[BboxTuple] = []

    for cluster in valid_clusters:
        x_coords = [l["bbox"][0] for l in cluster] + [l["bbox"][2] for l in cluster]
        y_coords = [l["bbox"][1] for l in cluster] + [l["bbox"][3] for l in cluster]

        bbox = (
            max(0, min(x_coords) - PADDING),
            max(0, min(y_coords) - PADDING),
            min(page_width, max(x_coords) + PADDING),
            min(page_height, max(y_coords) + PADDING),
        )

        synthetic_figures.append(bbox)

        if trace_id:
            logger.info(
                "[%s] Synthetic figure detected: %d labels, X-spread=%.1f, region (%.0f,%.0f)-(%.0f,%.0f)",
                trace_id, len(cluster), max(x_coords) - min(x_coords),
                bbox[0], bbox[1], bbox[2], bbox[3]
            )

    return synthetic_figures


def _heal_truncated_sentences(
        sentences: List[Dict],
        trace_id: str = None
) -> List[Dict]:
    """
    Post-segmentation healing for sentences cut at auxiliary verbs,
    articles, or prepositions.

    PROFESSIONAL v3.0 (Corrected):
        - Only heals sentences ending with PERIOD (not ?, !, :)
        - Truncation typically occurs when pysbd mistakes abbreviations

    Args:
        sentences: List of sentence dictionaries.
        trace_id: Optional trace ID for logging.

    Returns:
        List of healed sentence dictionaries.
    """

    # ---------------------------------------------------------------------
    # Phase 7 Hygiene: Clear transient consumption flags
    # Prevent cross-run state leakage during testing / retries
    # ---------------------------------------------------------------------
    for s in sentences:
        if isinstance(s, dict):
            s.pop("_consumed", None)

    healed: List[Dict] = []
    i = 0
    heal_count = 0
    ronc_driven_heals = 0
    heuristic_driven_heals = 0

    while i < len(sentences):
        curr = sentences[i]
        # Skip if consumed by a previous lookahead merge
        if curr.get("_consumed"):
            i += 1
            continue
        text = curr.get("text", "").strip()

        # =====================================================================
        # PHASE 7 PRIORITY 0: RONC Intersection Trigger (SUPREME AUTHORITY)
        #
        # If current sentence shares ANY atomic unit with the next sentence,
        # they were illegally split. MERGE IMMEDIATELY without heuristics.
        #
        # CRITICAL: This runs BEFORE legacy guards (period check, cut indicators)
        # to ensure RONC-driven healing is never blocked by punctuation state.
        # =====================================================================
        curr_page = curr.get("page_number")
        curr_ronc_units = set(curr.get("_ronc_atomic_units") or [])
        ronc_triggered = False

        curr_boundary_risks = curr.get("boundary_risks") or []
        is_short_fragment = "short_no_terminal_punct" in curr_boundary_risks

        if curr_ronc_units:
            MAX_RONC_LOOKAHEAD = 5
            for j in range(i + 1, min(i + 1 + MAX_RONC_LOOKAHEAD, len(sentences))):
                next_sent = sentences[j]

                # Stop if we hit a new page (same-page healer only)
                if next_sent.get("page_number") != curr_page:
                    break

                if next_sent.get("_consumed"):
                    continue

                next_text = (next_sent.get("text") or "").strip()
                next_ronc_units = set(next_sent.get("_ronc_atomic_units") or [])

                # ─────────────────────────────────────────────────────────────
                # HEADING ISOLATION GUARD
                # Never merge into a heading/subheading/title sentence.
                # Headings must remain standalone for TTS pacing.
                # Mirrors the guard in _stitch_helper_should_merge.
                # ─────────────────────────────────────────────────────────────
                if next_sent.get("role", "") in ("heading", "subheading", "title"):
                    break

                # ─────────────────────────────────────────────────────────────
                # RONC BARRIER FOR SHORT NON-TERMINAL FRAGMENTS
                # Short non-terminal sentences must not drive a RONC merge
                # that skips over an intervening complete sentence.
                # ─────────────────────────────────────────────────────────────
                if is_short_fragment:
                    next_boundary_risks = next_sent.get("boundary_risks") or []
                    next_is_short = "short_no_terminal_punct" in next_boundary_risks

                    # Treat sentence as complete if it is not short OR it ends with terminal punctuation
                    next_complete = (
                            not next_is_short or
                            (next_text and next_text[-1] in ".!?")
                    )

                    # Intervening complete sentence not sharing RONC units blocks lookahead
                    if next_complete and not (curr_ronc_units & next_ronc_units):
                        break

                # Existing RONC intersection logic (unchanged)
                if curr_ronc_units & next_ronc_units:
                    # ─────────────────────────────────────────────────────────
                    # CRITICAL FIX v3.1: Sacred Period Guard
                    # Do not force-merge across valid Period+Uppercase boundaries
                    # even if RONC units intersect (e.g. sentences in same paragraph).
                    # ─────────────────────────────────────────────────────────
                    if text.endswith('.') and next_text and next_text[0].isupper():
                        # Check for abbreviations (reuse legacy cut indicators)
                        _temp_base = text.rstrip('.')
                        _temp_words = _temp_base.split()
                        _temp_last = _temp_words[-1].lower().rstrip(
                            ".,;:!?\"')") if _temp_words else ""

                        # If not an abbreviation, respect the period.
                        if _temp_last not in _HEALING_CUT_INDICATORS:
                            # Stop looking for RONC merges. Fall through to legacy logic
                            # (which handles complete sentences correctly).
                            break

                    # Handle both period and non-period cases
                    text_clean = text.rstrip('.') if text.endswith('.') else text

                    # Perform RONC-driven merge
                    merged_text = text_clean + " " + next_text
                    merged = curr.copy()
                    merged["text"] = merged_text
                    merged["tts_text"] = merged_text
                    merged["span_end_index"] = next_sent.get("span_end_index",
                                                             curr.get("span_end_index"))
                    merged["char_end"] = next_sent.get("char_end", curr.get("char_end"))
                    merged["healed_truncation"] = True
                    merged["_ronc_heal_triggered"] = True

                    # Inherit semantic authority
                    if next_sent.get("_has_semantic_authority") and not merged.get(
                            "_has_semantic_authority"):
                        merged["_has_semantic_authority"] = True
                        merged["_authority_inherited_from"] = "ronc_truncation_heal"

                    # RONC METADATA PROPAGATION (union)
                    combined = curr_ronc_units | next_ronc_units
                    merged["_ronc_atomic_units"] = sorted(combined)

                    # source_cids propagation (contribution truth)
                    prev_source_cids = curr.get("source_cids") or []
                    next_source_cids = next_sent.get("source_cids") or []
                    combined_cids = list(prev_source_cids) + list(next_source_cids)
                    seen_cids = set()
                    deduped_cids = []
                    for cid in combined_cids:
                        if cid is not None and cid not in seen_cids:
                            seen_cids.add(cid)
                            deduped_cids.append(cid)
                    if deduped_cids:
                        merged["source_cids"] = deduped_cids

                    # _source_spans propagation (runtime authority)
                    curr_src_spans = curr.get("_source_spans") or []
                    next_src_spans = next_sent.get("_source_spans") or []
                    if curr_src_spans or next_src_spans:
                        merged["_source_spans"] = list(curr_src_spans) + list(next_src_spans)

                    healed.append(merged)
                    heal_count += 1
                    ronc_driven_heals += 1
                    ronc_triggered = True
                    sentences[j]["_consumed"] = True

                    if trace_id:
                        # ─────────────────────────────────────────────────────
                        # RONC AUDIT TRAIL (v2.1): Log contract fields that drove merge
                        # ─────────────────────────────────────────────────────
                        intersecting_units = sorted(curr_ronc_units & next_ronc_units)

                        # Extract authority ratings from source spans if available
                        curr_spans = curr.get("_source_spans", [])
                        next_spans = next_sent.get("_source_spans", [])

                        curr_authorities = set()
                        next_authorities = set()
                        for sp in curr_spans:
                            auth = (sp.get("_ronc_contract") or {}).get("authority")
                            if auth:
                                curr_authorities.add(auth)
                        for sp in next_spans:
                            auth = (sp.get("_ronc_contract") or {}).get("authority")
                            if auth:
                                next_authorities.add(auth)

                        # Extract chain exit/entry for lineage tracking
                        curr_chain_exit = None
                        next_chain_entry = None
                        # NOTE: _source_spans order is assumed to reflect reading order.
                        # If this invariant changes, boundary span selection must be revisited.
                        if curr_spans:
                            last_span = curr_spans[-1]
                            curr_chain_exit = last_span.get("_a2_edge_next_id")
                        if next_spans:
                            first_span = next_spans[0]
                            next_chain_entry = first_span.get("_a2_edge_prev_id")

                        # Determine if explicit chain link exists
                        has_explicit_link = (
                                curr_chain_exit is not None and
                                next_chain_entry is not None and
                                curr_chain_exit == next_chain_entry
                        )

                        logger.debug(
                            "[%s] RONC-driven truncation heal (gap=%d): "
                            "units=%s, curr_auth=%s, next_auth=%s, "
                            "chain_link=%s, text='%s...'",
                            trace_id,
                            j - i,
                            intersecting_units,
                            sorted(curr_authorities) if curr_authorities else "(none)",
                            sorted(next_authorities) if next_authorities else "(none)",
                            "explicit" if has_explicit_link else "unit_only",
                            merged.get("text", "")[:40]
                        )
                    break

        # RONC handled this sentence — skip legacy guards entirely
        if ronc_triggered:
            i += 1
            continue

            # ─────────────────────────────────────────────────────────────────
            # HEADING ISOLATION: Heading/title sentences bypass legacy healing.
            # Currently safe by coincidence (heading last-words are not in
            # _HEALING_CUT_INDICATORS and short-fragment checks reject them),
            # but made explicit to prevent regression if those sets change.
            # ─────────────────────────────────────────────────────────────────
        if curr.get("role", "") in ("heading", "subheading", "title"):
            healed.append(curr)
            i += 1
            continue

        # =====================================================================
        # PRIORITY 1 (LEGACY): Only process sentences ending with PERIOD
        # =====================================================================
        ends_with_period = text.endswith('.')
        if not ends_with_period:
            healed.append(curr)
            i += 1
            continue

        # Get last word (strip only the period)
        text_no_period = text.rstrip('.')
        words = text_no_period.split()
        last_word = words[-1].lower().rstrip(".,;:!?\"')") if words else ""

        # =====================================================================
        # GUARD: Only attempt healing if last word is a cut indicator
        # =====================================================================
        if last_word not in _HEALING_CUT_INDICATORS:
            # =================================================================
            # SHORT-FRAGMENT OVERRIDE
            #
            # Very short period-ending sentences may be pysbd mis-segments
            # even when the last word is not a traditional cut indicator.
            #   Defect 5.1: "You most likely." + "won't see..." → merge
            #
            # Conditions (ALL required to fall through to legacy lookahead):
            #   1. Word count <= _HEAL_SHORT_FRAGMENT_MAX_WORDS
            #   2. Not a known valid short sentence ("Indeed.", "I know.")
            #   3. Last word has no internal periods (not "U.S.", "e.g.")
            #   4. Last word is not an acronym (not "NATO.", "WHO.")
            #   5. Last word length >= _HEAL_SHORT_FRAGMENT_MIN_LAST_WORD_LEN
            #      (blocks abbreviation suffixes: "al", "vs", "cf")
            #
            # The existing legacy lookahead below enforces lowercase-start
            # on the next sentence — that check is NOT duplicated here.
            # =================================================================
            last_word_raw = words[-1] if words else ""
            has_internal_period = (
                '.' in last_word_raw[:-1] if len(last_word_raw) > 1 else False
            )
            is_acronym = (
                    last_word_raw.rstrip(".,;:").isupper()
                    and len(last_word_raw.rstrip(".,;:")) > 1
            )
            is_short_fragment_candidate = (
                    len(words) <= _HEAL_SHORT_FRAGMENT_MAX_WORDS
                    and text.lower() not in VALID_SHORT_SENTENCES
                    and not has_internal_period
                    and not is_acronym
                    and len(last_word) >= _HEAL_SHORT_FRAGMENT_MIN_LAST_WORD_LEN
            )
            if not is_short_fragment_candidate:
                healed.append(curr)
                i += 1
                continue
            # Fall through to legacy lookahead (checks lowercase continuation)

        # =====================================================================
        # PRIORITY 1 LEGACY LOOKAHEAD: Search for continuation (heuristic)
        # Only reached if RONC did not trigger above.
        # =====================================================================
        found_continuation = False
        MAX_LOOKAHEAD = 5

        for j in range(i + 1, min(i + 1 + MAX_LOOKAHEAD, len(sentences))):
            next_sent = sentences[j]

            # Stop if we hit a new page (same-page healer only)
            if next_sent.get("page_number") != curr_page:
                break

            # Stop if we hit a new page
            if next_sent.get("page_number") != curr_page:
                break

            # Skip if already consumed by a previous merge
            if next_sent.get("_consumed"):
                continue

            next_text = next_sent.get("text", "").strip()
            if not next_text:
                continue

            # HEADING ISOLATION GUARD (defense-in-depth, mirrors RONC path)
            if next_sent.get("role", "") in ("heading", "subheading", "title"):
                break

            # CONTINUATION SIGNAL: Lowercase start = strong match
            first_char = next_text[0]
            second_char = next_text[1] if len(next_text) > 1 else ""

            is_continuation = (
                    first_char.islower() or
                    (
                            first_char in {",", ";", ":", "—", "–"} and
                            second_char.islower()
                    ) or
                    (
                            first_char in {"(", "[", "{"} and
                            second_char.islower()
                    )
            )

            if is_continuation:
                # Perform Merge
                merged_text = text_no_period + " " + next_text

                merged = curr.copy()
                merged["text"] = merged_text
                merged["tts_text"] = merged_text
                merged["span_end_index"] = next_sent.get("span_end_index",
                                                         curr.get("span_end_index"))
                merged["char_end"] = next_sent.get("char_end", curr.get("char_end"))
                merged["healed_truncation"] = True

                # PATCH: Inherit semantic authority from continuation sentence
                if next_sent.get("_has_semantic_authority") and not merged.get(
                        "_has_semantic_authority"):
                    merged["_has_semantic_authority"] = True
                    merged["_authority_inherited_from"] = "truncation_heal_continuation"

                # RONC METADATA PROPAGATION (Phase 4 — Passive, NO decisions)
                left_units = set(curr.get("_ronc_atomic_units") or [])
                right_units = set(next_sent.get("_ronc_atomic_units") or [])
                combined = left_units | right_units
                if combined:
                    merged["_ronc_atomic_units"] = sorted(combined)

                # source_cids propagation (contribution truth)
                prev_source_cids = curr.get("source_cids") or []
                next_source_cids = next_sent.get("source_cids") or []
                combined_cids = list(prev_source_cids) + list(next_source_cids)
                seen_cids = set()
                deduped_cids = []
                for cid in combined_cids:
                    if cid is not None and cid not in seen_cids:
                        seen_cids.add(cid)
                        deduped_cids.append(cid)
                if deduped_cids:
                    merged["source_cids"] = deduped_cids

                # _source_spans propagation (runtime authority)
                curr_src_spans = curr.get("_source_spans") or []
                next_src_spans = next_sent.get("_source_spans") or []
                if curr_src_spans or next_src_spans:
                    merged["_source_spans"] = list(curr_src_spans) + list(next_src_spans)

                healed.append(merged)
                heal_count += 1
                heuristic_driven_heals += 1
                found_continuation = True

                # Mark the future sentence as consumed so the main loop skips it
                sentences[j]["_consumed"] = True

                if trace_id:
                    # ─────────────────────────────────────────────────────
                    # HEURISTIC AUDIT TRAIL: Log context for non-RONC heals
                    # ─────────────────────────────────────────────────────
                    # RONC availability inferred from runtime source spans (canonical)
                    had_ronc_data = bool(curr.get("_source_spans"))

                    logger.debug(
                        "[%s] Heuristic truncation heal (gap=%d): "
                        "cut_word='%s', ronc_available=%s, text='%s...'",
                        trace_id,
                        j - i,
                        last_word,
                        had_ronc_data,
                        merged.get("text", "")[:40]
                    )
                break

        if found_continuation:
            i += 1
        else:
            healed.append(curr)
            i += 1

    # =========================================================================
    # PHASE 7 INVARIANT (DEBUG ONLY)
    # Healing must not lose sentences unexpectedly
    # =========================================================================
    if trace_id and __debug__:
        consumed_count = sum(
            1 for s in sentences
            if isinstance(s, dict) and s.get("_consumed")
        )
        expected_min = len(sentences) - consumed_count
        if len(healed) < expected_min:
            logger.error(
                "[%s] HEALING LOSS: expected >= %d sentences, got %d",
                trace_id, expected_min, len(healed)
            )
    if trace_id and heal_count:
        logger.info(
            "[%s] Truncation healing: %d sentences repaired "
            "(ronc_driven=%d, heuristic=%d)",
            trace_id, heal_count, ronc_driven_heals, heuristic_driven_heals
        )

    return healed


def _heal_cross_page_truncations(
        all_sentences: List[Dict],
        trace_id: str = None
) -> List[Dict]:
    """
    Post-aggregation healing for sentences truncated at page/column boundaries.
    MODIFIED Phase 10 FINAL:
        1. LOOKAHEAD to skip garbage between valid sentences
        2. EXPANDED CUT_INDICATORS with relative pronouns

    PHASE 7 NOTE:
        RONC intersection is NOT used as merge trigger here.
        Atomic unit IDs are page-scoped; cross-page collision possible.
        Phase 8 will implement tuple-scoped identity for safe cross-page RONC.
    Args:
        all_sentences: List of all sentence dictionaries (all pages combined).
        trace_id: Optional trace ID for logging.

    Returns:
        List of healed sentence dictionaries.
    """

    # Phase 10: Maximum lookahead distance
    MAX_LOOKAHEAD = 5

    if not all_sentences:
        return all_sentences

    # ---------------------------------------------------------------------
    # Phase 7 Hygiene: Clear transient consumption flags
    # Prevent cross-run state leakage during testing / retries
    # ---------------------------------------------------------------------
    for s in all_sentences:
        if isinstance(s, dict):
            s.pop("_consumed", None)

    healed: List[Dict] = []
    consumed_indices: set = set()
    heal_count = 0

    for i, curr in enumerate(all_sentences):
        if i in consumed_indices:
            continue

        text = curr.get("text", "").strip()

        # HEADING ISOLATION: Heading/title sentences are never cross-page
        # heal candidates. Currently safe by coincidence (heading last-words
        # fail cut-indicator check and heading text starts uppercase/digit),
        # but made explicit for regression safety.
        if curr.get("role", "") in ("heading", "subheading", "title"):
            healed.append(curr)
            continue

        if not text.endswith('.'):
            healed.append(curr)
            continue

        text_no_period = text.rstrip('.')
        words = text_no_period.split()
        last_word = words[-1].lower() if words else ""

        if last_word not in _HEALING_CUT_INDICATORS:
            healed.append(curr)
            continue

        # =====================================================================
        # LOOKAHEAD LOOP (Phase 10)
        # =====================================================================
        found_continuation = False
        merge_target_idx = -1

        for lookahead in range(1, MAX_LOOKAHEAD + 1):
            peek_idx = i + lookahead

            if peek_idx >= len(all_sentences):
                break

            peek_sent = all_sentences[peek_idx]
            peek_text = peek_sent.get("text", "").strip()

            # Skip garbage patterns
            if not peek_text:
                continue
            if peek_text.isdigit():
                continue
            if len(peek_text) <= 2 and not peek_text.isalpha():
                continue

            first_char = peek_text[0]
            second_char = peek_text[1] if len(peek_text) > 1 else ""

            # HEADING ISOLATION GUARD (defense-in-depth, mirrors same-page healer)
            if peek_sent.get("role", "") in ("heading", "subheading", "title"):
                break

            # CONTINUATION SIGNALS (Phase 10, tightened)
            # NOTE: Short sentences (≤3 tokens) starting uppercase are
            # intentionally NOT merged to prevent false matches with
            # headings like "Figure 2" or "Table 3".
            is_continuation = (
                    first_char.islower() or
                    (
                            first_char in {",", ";", ":", "—", "–"} and
                            second_char.islower()
                    )
            )

            if is_continuation:
                found_continuation = True
                merge_target_idx = peek_idx
                break
            else:
                break

        if not found_continuation:
            healed.append(curr)
            continue

        # =====================================================================
        # PERFORM MERGE
        # =====================================================================
        next_sent = all_sentences[merge_target_idx]
        next_text = next_sent.get("text", "").strip()

        merged_text = text_no_period + " " + next_text.lstrip()

        merged = curr.copy()
        merged["text"] = merged_text
        merged["tts_text"] = merged_text
        merged["span_end_index"] = next_sent.get("span_end_index", curr.get("span_end_index"))
        merged["char_end"] = next_sent.get("char_end", curr.get("char_end"))
        merged["healed_truncation"] = True
        merged["lookahead_distance"] = merge_target_idx - i

        # PATCH: Inherit semantic authority from continuation sentence
        if next_sent.get("_has_semantic_authority") and not merged.get("_has_semantic_authority"):
            merged["_has_semantic_authority"] = True
            merged["_authority_inherited_from"] = "cross_page_heal_continuation"

        # RONC METADATA PROPAGATION (Phase 4 — Passive, NO decisions)
        left_units = set(curr.get("_ronc_atomic_units") or [])
        right_units = set(next_sent.get("_ronc_atomic_units") or [])
        combined = left_units | right_units
        if combined:
            merged["_ronc_atomic_units"] = sorted(combined)

        # source_cids propagation (contribution truth)
        prev_source_cids = curr.get("source_cids") or []
        next_source_cids = next_sent.get("source_cids") or []
        combined_cids = list(prev_source_cids) + list(next_source_cids)
        seen_cids = set()
        deduped_cids = []
        for cid in combined_cids:
            if cid is not None and cid not in seen_cids:
                seen_cids.add(cid)
                deduped_cids.append(cid)
        if deduped_cids:
            merged["source_cids"] = deduped_cids

        # _source_spans propagation (runtime authority)
        curr_src_spans = curr.get("_source_spans") or []
        next_src_spans = next_sent.get("_source_spans") or []
        if curr_src_spans or next_src_spans:
            merged["_source_spans"] = list(curr_src_spans) + list(next_src_spans)

        # =====================================================================
        # PHASE 8 PLACEHOLDER: Cross-page RONC intersection detection
        #
        # RONC intersection is NOT used as merge trigger here because
        # atomic unit IDs are page-scoped. Cross-page ID collision is possible.
        #
        # Phase 8 will implement tuple-scoped identity: (page_number, unit_id)
        # =====================================================================
        ronc_intersection = left_units & right_units
        if ronc_intersection and trace_id:
            logger.debug(
                "[%s] Phase 8 note: Cross-page heal found RONC intersection=%s "
                "(may indicate collision or true cross-page unit)",
                trace_id, sorted(ronc_intersection)
            )

        curr_page = curr.get("page_number", 1)
        next_page = next_sent.get("page_number", 1)
        if curr_page != next_page:
            merged["crosses_pages"] = True
            merged["page_range"] = [curr_page, next_page]

        healed.append(merged)
        heal_count += 1
        # Mark ALL intervening sentences as consumed to prevent loss or duplication
        for k in range(i + 1, merge_target_idx + 1):
            consumed_indices.add(k)

        if trace_id:
            logger.debug(
                "[%s] Truncation healed (lookahead=%d): '...%s' + '%s...' (pages %d→%d)",
                trace_id, merge_target_idx - i, text[-20:], next_text[:20], curr_page, next_page
            )

    if trace_id and heal_count:
        logger.info(
            "[%s] Cross-page truncation healing: %d sentences repaired",
            trace_id, heal_count
        )

    return healed


def _segment_sentences(
        full_text: str,
        char_to_span: List[int],
        all_spans: List[Dict],
        trace_id: str = None
) -> List[Dict]:
    """
    Segment a reconstructed text stream into sentence units.

    ARCHITECTURAL GUARDRAIL:
    This method is the SOLE authority for sentence segmentation (cutting).

    Responsibilities:
      - Cut full_text into sentences
      - Map sentences back to source spans via char_to_span
      - Populate sentence-level structural metadata

    Explicitly excluded:
      - Fragment healing or merging
      - Truncation repair
      - Punctuation normalization
      - Text mutation
      - TTS preparation
    """

    if not full_text or not char_to_span or not all_spans:
        if trace_id:
            logger.warning("[%s] Empty input to _segment_sentences", trace_id)
        return []

    # =========================================================================
    # SENTENCE SEGMENTATION
    # =========================================================================
    seg = _get_segmenter()
    raw_sentences = seg.segment(full_text)

    if not raw_sentences:
        return []

    # =========================================================================
    # PROCESS SENTENCES
    # =========================================================================
    processed_sentences: List[Dict] = []
    cursor = 0
    alignment_failures = 0
    empty_skipped = 0

    for sent_idx, sent_text in enumerate(raw_sentences):
        sent = sent_text.strip()
        if not sent:
            empty_skipped += 1
            continue

        # =====================================================================
        # STEP 1: Locate sentence in full_text
        # =====================================================================
        start_char = -1
        actual_end_char = -1
        alignment_method = "unknown"

        # Method 1: Direct find (fastest)
        direct_pos = full_text.find(sent, cursor)
        if direct_pos != -1:
            start_char = direct_pos
            actual_end_char = direct_pos + len(sent)
            alignment_method = "direct"

        # Method 2: Normalized match (handles whitespace variations)
        if start_char == -1:
            normalized_sent = _WHITESPACE_PATTERN.sub(" ", sent)
            probe_distance = max(
                _SEGMENT_MIN_PROBE_DISTANCE,
                min(_SEGMENT_MAX_PROBE_DISTANCE, len(sent) * 2)
            )
            for probe_start in range(cursor, min(cursor + probe_distance, len(full_text))):
                probe_end = probe_start + len(sent) + _SEGMENT_PROBE_SLACK
                if probe_end > len(full_text):
                    probe_end = len(full_text)
                probe_text = full_text[probe_start:probe_end]
                normalized_probe = _WHITESPACE_PATTERN.sub(" ", probe_text)
                if normalized_probe.startswith(normalized_sent):
                    start_char = probe_start
                    actual_end_char = _find_actual_end_position(
                        full_text, probe_start, normalized_sent
                    )
                    alignment_method = "normalized"
                    break

        # Method 3: Fuzzy word-boundary match
        if start_char == -1:
            words = sent.split()[:_SEGMENT_FUZZY_WORD_COUNT]
            if words:
                first_words = " ".join(words)
                fuzzy_pos = full_text.find(first_words, cursor)
                if fuzzy_pos != -1 and fuzzy_pos < cursor + _SEGMENT_MAX_PROBE_DISTANCE:
                    start_char = fuzzy_pos
                    actual_end_char = min(fuzzy_pos + len(sent), len(full_text))
                    alignment_method = "fuzzy"

        # Method 4: Fallback to cursor (never drop)
        if start_char == -1:
            alignment_failures += 1
            start_char = cursor
            actual_end_char = min(cursor + len(sent), len(full_text))
            alignment_method = "fallback"
            if trace_id:
                logger.warning(
                    "[%s] Sentence #%d alignment fallback to cursor %d: '%s...'",
                    trace_id, sent_idx, cursor, sent[:40]
                )

        # Ensure forward progress
        if start_char < cursor:
            start_char = cursor
            actual_end_char = min(start_char + len(sent), len(full_text))

        if actual_end_char == -1 or actual_end_char <= start_char:
            actual_end_char = min(start_char + len(sent), len(full_text))

        end_char = actual_end_char
        cursor = end_char

        # =====================================================================
        # STEP 2: Map to source spans via char_to_span
        # =====================================================================
        span_indices: set[int] = set()
        columns: set = set()
        pages: set = set()
        roles: set = set()

        for char_idx in range(start_char, min(end_char, len(char_to_span))):
            span_idx = char_to_span[char_idx]
            if 0 <= span_idx < len(all_spans):
                span_indices.add(span_idx)
                span = all_spans[span_idx]
                columns.add(span.get("column_index", 0))
                pages.add(span.get("page_number", 0))
                role = span.get("role")
                if role is not None:
                    roles.add(role)

        # Handle unmapped sentences
        if not span_indices:
            if trace_id:
                logger.warning(
                    "[%s] Sentence #%d has no mapped spans: '%s...'",
                    trace_id, sent_idx, sent[:40]
                )
            processed_sentences.append({
                "text": sent,
                "tts_text": None,
                "span_start_index": -1,
                "span_end_index": -1,
                "paragraph_index": -1,
                "column_index": -1,
                "page_number": -1,
                "role": TextRole.BODY.value,
                "char_start": start_char,
                "char_end": end_char,
                "alignment_method": alignment_method,
                "unmapped": True,
            })
            continue

        # =====================================================================
        # STEP 3: Resolve span metadata
        # =====================================================================
        sorted_spans = sorted(span_indices)
        start_span_idx = sorted_spans[0]
        end_span_idx = sorted_spans[-1]
        first_span = all_spans[start_span_idx]
        primary_page = first_span.get("page_number", 1)

        # Determine primary column
        if len(columns) > 1:
            column_counts: Dict[int, int] = {}
            for idx in span_indices:
                col = all_spans[idx].get("column_index", 0)
                column_counts[col] = column_counts.get(col, 0) + 1
            primary_column = max(column_counts, key=column_counts.get)
        else:
            primary_column = first_span.get("column_index", 0)

        # Deterministic role resolution
        if len(roles) > 1:
            role_priority = [
                TextRole.BODY.value,
                TextRole.HEADING.value if hasattr(TextRole, 'HEADING') else "heading",
                TextRole.SUBHEADING.value if hasattr(TextRole, 'SUBHEADING') else "subheading",
            ]
            # Find highest priority role present in the sentence
            primary_role = None
            for priority_role in role_priority:
                if priority_role in roles:
                    primary_role = priority_role
                    break
            # Fallback if no priority role found
            if primary_role is None:
                primary_role = first_span.get("role", TextRole.BODY.value)
        else:
            # Single role (or empty): use first span's role
            primary_role = first_span.get("role", TextRole.BODY.value)

        # =====================================================================
        # STEP 3.5: Compute sentence bbox
        # =====================================================================
        sentence_bbox = _compute_sentence_bbox(all_spans, sorted_spans)

        # =====================================================================
        # STEP 4: Build sentence record
        # =====================================================================
        sentence_record: Dict = {
            "text": sent,
            "tts_text": None,
            "span_start_index": start_span_idx,
            "span_end_index": end_span_idx,
            "paragraph_index": first_span.get("paragraph_index", 0),
            "column_index": primary_column,
            "page_number": primary_page,
            "role": primary_role,
            "char_start": start_char,
            "char_end": end_char,
            "alignment_method": alignment_method,
            "boundary_risks": [],
        }

        # Explicitly consume primary_role for downstream logic
        sentence_record["is_body"] = (primary_role == TextRole.BODY.value)
        sentence_record["is_heading"] = primary_role in {
            TextRole.HEADING.value if hasattr(TextRole, "HEADING") else "heading",
            TextRole.SUBHEADING.value if hasattr(TextRole, "SUBHEADING") else "subheading",
        }

        # Structural risk flag: non-body sentence crossing multiple spans
        if primary_role != TextRole.BODY.value and len(span_indices) > 1:
            sentence_record["structural_risk"] = "non_body_multi_span"

        # Alignment confidence flag: fallback/fuzzy methods have higher error rate
        if alignment_method in ("fallback", "fuzzy"):
            sentence_record["alignment_risk"] = True

        if sentence_bbox:
            sentence_record["bbox"] = sentence_bbox

        if len(pages) > 1:
            sentence_record["crosses_pages"] = True
            sentence_record["page_range"] = sorted(pages)

        if len(columns) > 1:
            sentence_record["crosses_columns"] = True
            sentence_record["column_range"] = sorted(columns)

        # Compound boundary risk: cross-page + cross-column
        if sentence_record.get("crosses_pages") and sentence_record.get("crosses_columns"):
            sentence_record["boundary_risks"].append("cross_page_cross_column")

        sentence_record["source_span_count"] = len(span_indices)

        # ==========================================================
        # PHASE 2.8: Contamination audit (defense-in-depth)
        # Should rarely fire after pre-filter. If it does, flag for triage.
        # ==========================================================
        contaminated_roles = roles.intersection(_TTS_NON_VIABLE_ROLES)
        if contaminated_roles:
            sentence_record["_contaminated"] = True
            sentence_record["_contaminated_roles"] = sorted(contaminated_roles)

            if trace_id:
                # Triage breadcrumbs: first few canonical span ids (if available)
                span_ids_for_log = []
                try:
                    for idx in sorted(span_indices)[:3]:
                        if 0 <= idx < len(all_spans):
                            span_ids_for_log.append(all_spans[idx].get("_canonical_span_id"))
                except Exception:
                    span_ids_for_log = []

                logger.warning(
                    "[%s] Sentence #%d contains non-viable roles after pre-filter: %s. "
                    "source_spans=%s text='%s...'",
                    trace_id, sent_idx, sorted(contaminated_roles),
                    span_ids_for_log, sent[:40]
                )

        # ==========================================================
        # PHASE 6.1: Source Span Linkage (RONC v2.1 + Pass 1 Semantic Split)
        #
        # ARCHITECTURAL CONTRACT (Pass 1 Revision):
        # Two distinct provenance concepts, intentionally separated:
        #
        # 1. _source_span_ids: WINDOW-ORDERED COVERAGE
        #    - Contiguous range [start_span_idx..end_span_idx]
        #    - May include non-contributing spans (within window bounds)
        #    - Preserves None sentinels for provenance gaps
        #    - Used for: page-turn markers, ordered traversal, reconstruction
        #
        # 2. source_cids: CONTRIBUTION TRUTH
        #    - Derived from sorted_spans (char_to_span membership)
        #    - Only spans whose text actually contributed to sentence
        #    - Excludes: None, _tts_excluded spans
        #    - Used for: highlighting, provenance audit, frontend geometry
        #
        # 3. _source_spans: RUNTIME AUTHORITY
        #    - Resolved span objects for stitch decisions
        #    - MUST be stripped before serialization
        #
        # INVARIANTS:
        # - source_cids ⊆ set(_source_span_ids) - {None}
        # - Every CID in source_cids contributed text to this sentence
        # - _source_spans MUST NOT be mutated by downstream consumers
        # ==========================================================

        # PART A: Build _source_span_ids (window coverage, contiguous)
        source_span_ids = []
        source_span_objects = []
        missing_canonical_ids = 0
        tts_excluded_in_window = 0

        for idx in range(start_span_idx, end_span_idx + 1):
            if not (0 <= idx < len(all_spans)):
                continue

            span = all_spans[idx]
            cid = span.get("_canonical_span_id")

            if span.get("_tts_excluded", False):
                tts_excluded_in_window += 1

            # Preserve exact window shape (no filtering)
            if cid is not None:
                source_span_ids.append(cid)
            else:
                source_span_ids.append(None)  # Sentinel: provenance gap
                missing_canonical_ids += 1

            source_span_objects.append(span)

        # Deduplicate while preserving order
        seen_ids = set()
        deduped_ids = []
        for cid in source_span_ids:
            if cid is None or cid not in seen_ids:
                deduped_ids.append(cid)
                if cid is not None:
                    seen_ids.add(cid)

        sentence_record["_source_span_ids"] = deduped_ids

        # PART B: Build source_cids (contribution truth from char_to_span)
        #
        # CRITICAL: Derive from sorted_spans, NOT from contiguous window.
        # sorted_spans contains exactly the span indices whose characters
        # appear in char_to_span for this sentence's [start_char..end_char].
        # This is the authoritative "which spans contributed text" answer.
        contributing_cids = []
        for idx in sorted_spans:
            if not (0 <= idx < len(all_spans)):
                continue

            span = all_spans[idx]
            cid = span.get("_canonical_span_id")

            # Exclude: missing CID, TTS-excluded spans
            if cid is None:
                continue
            if span.get("_tts_excluded", False):
                continue

            contributing_cids.append(cid)

        # Deduplicate while preserving order (edge case: duplicate CIDs)
        seen_contrib = set()
        deduped_contrib = []
        for cid in contributing_cids:
            if cid not in seen_contrib:
                seen_contrib.add(cid)
                deduped_contrib.append(cid)

        sentence_record["source_cids"] = deduped_contrib

        # PART C: Runtime span objects (for stitch authority)
        sentence_record["_source_spans"] = source_span_objects

        # PART D: Provenance integrity checks
        if missing_canonical_ids > 0 and trace_id:
            logger.warning(
                "[%s] Sentence #%d has %d spans without _canonical_span_id "
                "(provenance gap). window=[%d..%d], tts_excluded_in_window=%d",
                trace_id, sent_idx, missing_canonical_ids,
                start_span_idx, end_span_idx, tts_excluded_in_window
            )

        # G2 partial validation: source_cids should not be empty for non-empty text
        if not sentence_record["source_cids"] and sent.strip() and trace_id:
            logger.error(
                "[%s] Sentence #%d has EMPTY source_cids despite non-empty text. "
                "window=[%d..%d], sorted_spans=%s, tts_excluded_in_window=%d. "
                "Text='%s...'",
                trace_id, sent_idx, start_span_idx, end_span_idx,
                sorted_spans[:5], tts_excluded_in_window, sent[:40]
            )

        # Diagnostic: Log if contribution set differs significantly from window
        contribution_count = len(deduped_contrib)
        window_count = len([c for c in deduped_ids if c is not None])
        if window_count > 0 and contribution_count < window_count and trace_id:
            excluded_from_contribution = window_count - contribution_count
            if excluded_from_contribution > 2:  # Only log if significant gap
                logger.debug(
                    "[%s] Sentence #%d: %d CIDs in window, %d contributing "
                    "(gap=%d, likely non-body spans in range)",
                    trace_id, sent_idx, window_count, contribution_count,
                    excluded_from_contribution
                )

        # ==========================================================
        # PHASE 6: Boundary Risk Analysis (NON-DESTRUCTIVE)
        # ==========================================================
        tokens = sent.split()
        token_count = len(tokens)

        # Risk 1: Leading conjunction
        if sent.lstrip().lower().startswith(("and ", "or ", "but ")):
            sentence_record["boundary_risks"].append("leading_conjunction")

        # Risk 2: Very low token count
        if token_count < 3:
            sentence_record["boundary_risks"].append("low_token_count")

        # Risk 3: Short sentence without terminal punctuation
        if token_count < 5 and not sent.rstrip().endswith((".", "!", "?")):
            sentence_record["boundary_risks"].append("short_no_terminal_punct")

        # Risk 4: Single-span fragment
        if len(span_indices) == 1:
            sentence_record["boundary_risks"].append("single_span_fragment")

        # ==========================================================
        # AUTHORITY PROPAGATION (Phase 6)
        # ==========================================================
        if any(all_spans[idx].get("_has_semantic_authority") for idx in span_indices):
            sentence_record["_has_semantic_authority"] = True

        # ==========================================================
        # RONC METADATA PROPAGATION (Phase 4 — Passive, NO decisions)
        # ==========================================================
        ronc_units = set()
        for idx in span_indices:
            if 0 <= idx < len(all_spans):
                unit_id = all_spans[idx].get("_ronc_atomic_unit_id")
                if unit_id is not None:
                    ronc_units.add(unit_id)

        if ronc_units:
            sentence_record["_ronc_atomic_units"] = sorted(ronc_units)

        processed_sentences.append(sentence_record)

    # =========================================================================
    # LOGGING
    # =========================================================================
    if trace_id:
        from collections import Counter
        methods = Counter(s.get("alignment_method", "unknown") for s in processed_sentences)
        logger.debug(
            "[%s] Segmentation: %d sentences from %d raw, %d empty skipped, "
            "alignment methods=%s, failures=%d",
            trace_id, len(processed_sentences), len(raw_sentences),
            empty_skipped, methods, alignment_failures
        )

    return processed_sentences


def _prepare_sentence_for_serialization(sentence: Dict) -> Dict:
    """
    Strip runtime-only fields before manifest emission.

    ARCHITECTURAL CONTRACT:

    This is a serialization boundary that strips runtime-only fields that MUST NOT
    appear in serialized sentence payloads. Downstream (process.py) may further
    transform/project fields during manifest emission.

    Runtime-only fields typically use underscore prefix; some underscore fields
    may be intentionally preserved for downstream projection.
    """
    # Create shallow copy to avoid mutating working data
    output = dict(sentence)

    # Strip runtime-only fields (underscore prefix = internal)
    
    # [FIX v8.0] DECOUPLE DISPLAY vs SPOKEN TEXT
    # 1. Capture Human-Readable Text (Visual)
    #    Preserve original punctuation (;) and restore scientific notation (mm²)
    display_text = output.get("text", "") or ""

    # Normalize common scientific units for display only
    display_text = re.sub(r"\bmm2\b", "mm²", display_text)
    display_text = re.sub(r"\bmm3\b", "mm³", display_text)

    output["display_text"] = display_text

    # 2. Prepare Decoder-Safe Text (Spoken)
    #    Use sanitized tts_text if available, otherwise fallback to raw.
    tts_text = output.get("tts_text", output.get("text", ""))
    output["text"] = tts_text

    # ═══════════════════════════════════════════════════════════════════
    # V1.7: Explicit Prosodic Clause Serialization
    #
    # Prosodic metadata is AUDIO-ONLY but MUST be serialized so
    # Stage 3 TTS generation can act on it.
    #
    # This method is the ONLY safe boundary to promote runtime
    # prosodic signals into manifest-visible fields.
    # ═══════════════════════════════════════════════════════════════════
    output["needs_clause_splitting"] = bool(
        sentence.get("needs_clause_splitting")
        or sentence.get("_tts_change_tracker", {}).get("needs_clause_splitting")
    )

    clauses = (
            sentence.get("prosodic_clauses")
            or sentence.get("_tts_change_tracker", {}).get("prosodic_clauses")
    )
    output["prosodic_clauses"] = clauses if clauses else None

    # 3. Cleanup Internal Runtime Fields
    # Underscore-prefixed fields are runtime-only by contract.
    output.pop("_source_spans", None)
    output.pop("tts_text", None)
    output.pop("_tts_change_tracker", None)

    # =========================================================================
    # DEBUG: Prosodic Serialization Contract
    #
    # If a sentence declares clause splitting, serialized output MUST
    # contain prosodic_clauses. Violations indicate upstream loss.
    # =========================================================================
    if __debug__:
        if output.get("needs_clause_splitting") and not output.get("prosodic_clauses"):
            logger.error(
                "SERIALIZATION CONTRACT VIOLATION: "
                "needs_clause_splitting=True but prosodic_clauses missing"
            )

        # Provenance integrity (shape-only at serialization boundary)
        source_ids = output.get("_source_span_ids") or []
        if isinstance(source_ids, list) and any(sid is None for sid in source_ids):
            logger.warning(
                "PROVENANCE GAP: sentence global_index=%s has None in _source_span_ids",
                output.get("global_index")
            )

    return output


# ============================================================================
# SENTENCE SEMANTIC PROJECTION (v1.0)
# Read-only carry-forward of span-level semantic conclusions
# ============================================================================

def _inject_sentence_semantic_projection(sentence: Dict, spans: List[Dict]) -> None:
    """
    Attach read-only semantic projection to sentence object.

    Invariant:
    - Makes NO semantic decisions
    - Only carries forward conclusions already made at span level
    """
    if not spans:
        sentence["_semantic_projection"] = None
        return

    unit_ids = set()
    has_chain = False
    confidences = []
    authorities = []
    source_span_ids = []

    for sp in spans:
        cid = sp.get("_canonical_span_id")
        if cid:
            source_span_ids.append(cid)

        unit_id = sp.get("_ronc_atomic_unit_id")
        if unit_id is not None:
            unit_ids.add(unit_id)

        if sp.get("_a2_qualified"):
            has_chain = True

        conf = sp.get("_semantic_confidence")
        if isinstance(conf, (int, float)):
            confidences.append(conf)

        auth = (sp.get("_ronc_contract") or {}).get("authority")
        if auth:
            authorities.append(auth)
    _AUTHORITY_RANK = _RONC_V2_AUTHORITY_RANK
    min_authority = (
        min(authorities, key=lambda a: _AUTHORITY_RANK.get(a, 0))
        if authorities else None
    )

    avg_confidence = (
        sum(confidences) / len(confidences)
        if confidences else None
    )

    sentence["_semantic_projection"] = {
        # Provenance
        "source_span_ids": source_span_ids,
        "atomic_unit_ids": sorted(unit_ids),

        # Chain connectivity
        "has_chain_membership": has_chain,
        "chain_entry_id": spans[0].get("_a2_edge_prev_id"),
        "chain_exit_id": spans[-1].get("_a2_edge_next_id"),

        # Quality signals
        "avg_confidence": round(avg_confidence, 3) if avg_confidence else None,
        "min_authority": min_authority,
        "authorities": sorted(set(authorities)),

        # Derived flags (diagnostic only)
        "single_unit": len(unit_ids) == 1,
        "chain_intact": (
                len(unit_ids) == 1
                and has_chain
                and min_authority in (_RONC_V2_AUTHORITY_STRONG, _RONC_V2_AUTHORITY_WEAK)
        ),
    }


def _normalize_punctuation_spacing(text: str) -> str:
    """
    Clean up spacing artifacts around punctuation marks.

    NEW v3.0: Punctuation Normalization
    Strategy:
        1. Remove space inside brackets: "( text )" -> "(text)"
        2. Remove space before punctuation: "word ." -> "word."
        3. Ensure space after comma: "word,word" -> "word, word"
        4. Collapse multiple spaces
    """

    # ORDERING INVARIANT:
    # This method MUST run strictly BEFORE Phase 6 (sentence segmentation).
    # It MUST NOT be invoked after sentence boundaries are inferred.

    if not text:
        return ""

    # Remove space after opening brackets
    text = re.sub(r'([(\[{])\s+', r'\1', text)

    # Remove space before closing brackets
    text = re.sub(r'\s+([)\]}])', r'\1', text)

    # Remove space before comma/period/colon/semicolon
    text = _PRE_PUNCT_WHITESPACE_PATTERN.sub(r'\1', text)
    text = re.sub(r'([.,:;])\1+', r'\1', text)

    # Ensure single space after comma (not before closing bracket or digit)
    # Lookahead ensures we don't break "1,000" or "(a,b)" logic if needed,
    # but strictly "text,text" -> "text, text"
    text = re.sub(r',(?=[^\s\d)\]}])', ', ', text)

    # Normalize multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


def _find_actual_end_position(full_text: str, start: int, normalized_target: str) -> int:
    """
    Find the actual end position in full_text for a normalized match.

    HARDENED MAPPING RULE:
    - Whitespace in normalized_target may correspond to ANY run of whitespace in full_text.
    - For non-whitespace characters:
        * Always advance through full_text.
        * Only advance target_idx when the current non-space character matches (case-insensitive).
        * If full_text contains ignorable characters (digits/punct) that may have been removed
          during normalization, skip them WITHOUT consuming target characters.

    This makes the mapper safe even if normalized_target was produced with
    lowercase + digit/punctuation stripping, not strictly whitespace-only collapsing.

    Args:
        full_text: Original text with original whitespace.
        start: Starting position in full_text.
        normalized_target: Normalized target string (expected lowercase).

    Returns:
        End position in full_text.
    """

    def _is_ignorable(ch: str) -> bool:
        # Conservative: treat common "normalization-removed" characters as ignorable.
        # (Digits + most punctuation). Keep letters/whitespace.
        return ch.isdigit() or (not ch.isalnum() and not ch.isspace())

    target_idx = 0
    text_idx = start

    while target_idx < len(normalized_target) and text_idx < len(full_text):
        t_ch = normalized_target[target_idx]

        # 1) Whitespace in target: consume >=1 whitespace run in full_text
        if t_ch.isspace():
            # Advance target over any whitespace run
            while target_idx < len(normalized_target) and normalized_target[target_idx].isspace():
                target_idx += 1
            # Advance text over any whitespace run
            while text_idx < len(full_text) and full_text[text_idx].isspace():
                text_idx += 1
            continue

        # 2) Non-whitespace in target: walk forward in full_text until we match t_ch,
        # skipping whitespace differences and ignorable full_text chars.
        #
        # HARDENED: Bounded forward scan to prevent runaway resync
        max_scan = max(len(normalized_target) * 4, 32)
        scan_count = 0
        while text_idx < len(full_text) and scan_count < max_scan:
            scan_count += 1
            f_ch = full_text[text_idx]

            # Skip whitespace in full_text without consuming target chars
            if f_ch.isspace():
                text_idx += 1
                continue

            # If full_text char is ignorable under normalization rules, skip it
            if _is_ignorable(f_ch):
                text_idx += 1
                continue

            # Case-insensitive match (t_ch is already lowercase from _normalize_text)
            if f_ch.lower() == t_ch:
                text_idx += 1
                target_idx += 1
                break

            # Non-match, non-ignorable: advance full_text (do NOT consume target)
            text_idx += 1

        # Guard: if we failed to match within bounded scan, stop to prevent drift
        # TODO (Batch B): Return bounded_resync flag to caller for alignment_risk tagging
        # Currently returns text_idx without indicating resync occurred.
        # Phase 6 requirement: sentence should receive alignment_risk="bounded_resync"
        if scan_count >= max_scan:
            break

    return text_idx


def _compute_sentence_page_bboxes(
        all_spans: List[Dict],
        span_indices: List[int]
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Compute per-page bbox unions for a sentence's source spans.

    Returns:
        {page_number: (x0,y0,x1,y1), ...} for all pages represented in span_indices.
        Empty dict if nothing valid.
    """
    page_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}

    for idx in span_indices or []:
        if idx < 0 or idx >= len(all_spans):
            continue
        span = all_spans[idx]
        bbox = span.get("bbox")
        page = span.get("page_number")
        if not bbox or page is None or len(bbox) < 4:
            continue
        page_boxes.setdefault(int(page), []).append(bbox)

    out: Dict[int, Tuple[float, float, float, float]] = {}
    for p, bxs in page_boxes.items():
        out[p] = (
            min(b[0] for b in bxs),
            min(b[1] for b in bxs),
            max(b[2] for b in bxs),
            max(b[3] for b in bxs),
        )
    return out


def _compute_sentence_bbox(
        all_spans: List[Dict],
        span_indices: List[int]
) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute bounding box union for a sentence's source spans.

    Only computes bbox for same-page spans to avoid invalid geometry.

    Args:
        all_spans: List of all span dictionaries.
        span_indices: Sorted list of span indices belonging to this sentence.

    Returns:
        Bounding box tuple (x0, y0, x1, y1) or None if spans lack bbox data.
    """
    if not span_indices:
        return None

    # Collect bboxes, checking page consistency
    bboxes: List[Tuple[float, float, float, float]] = []
    first_page = None

    for idx in span_indices:
        if idx < 0 or idx >= len(all_spans):
            continue

        span = all_spans[idx]
        bbox = span.get("bbox")
        page = span.get("page_number")

        if first_page is None:
            first_page = page

        # Only include same-page bboxes
        if bbox and page == first_page and len(bbox) >= 4:
            bboxes.append(bbox)

    if not bboxes:
        return None

    # Compute union
    return (
        min(b[0] for b in bboxes),  # min x0
        min(b[1] for b in bboxes),  # min y0
        max(b[2] for b in bboxes),  # max x1
        max(b[3] for b in bboxes),  # max y1
    )


def _get_segmenter() -> pysbd.Segmenter:
    """
    Get or create the sentence segmenter singleton.

    Note: Not thread-safe. For multi-threaded usage, consider
    using threading.Lock or creating per-thread instances.

    Returns:
        Initialized pysbd Segmenter instance.
    """
    global _SENTENCE_SEGMENTER
    if _SENTENCE_SEGMENTER is None:
        _SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)
    return _SENTENCE_SEGMENTER


# ✦                  ✦                  ✦                  ✦
# ✦───────────── 5 Global Document Analysis ─────────────✦
# ✦                  ✦                  ✦                  ✦

# ✦────── a. Analysis Utilities ──────✦

def _normalize_text(input_text: str) -> str:
    """
    Normalize text for header/footer comparison.

    Normalization steps:
        1. Lowercase conversion
        2. Digit removal (so "Page 1" matches "Page 2")
        3. Punctuation removal
        4. Whitespace normalization

    Args:
        input_text: Text to normalize.

    Returns:
        Normalized text string.

    """
    # This normalization is for header/footer band comparison ONLY.
    # It MUST NOT be used for sentence text, joining, or TTS.

    if not input_text:
        return ""
    result = input_text.lower()
    result = _NORMALIZE_DIGIT_PATTERN.sub("", result)
    result = _NORMALIZE_PUNCT_PATTERN.sub("", result)
    result = _WHITESPACE_PATTERN.sub(" ", result)
    return result.strip()


# ✦────── b. Global Band Detection ──────✦

def _compute_global_header_footer_bands(
        page_outputs: List[Dict],
        trace_id: str = None,
        global_median_font_size: float = None  # NEW ARGUMENT
) -> Dict[str, List[int]]:
    """
    Compute global header/footer Y-bands across all pages.

    MODIFIED v3.3:
        1. ACTIVE SCAN: Directly examines span positions in header/footer zones
        2. Explicit 10px binning for robust Y-grouping
        3. Band merging within tolerance before frequency check
        4. Lowered frequency threshold (25%) for smaller documents

    Args:
        page_outputs: List of page extraction results.
        trace_id: Optional trace ID for logging.

    Returns:
        Dict with "header_bands" and "footer_bands" as sorted lists of Y-positions.
    """
    from typing import Tuple

    total_pages = len(page_outputs)
    if total_pages == 0:
        return {"header_bands": [], "footer_bands": []}

    # =========================================================================
    # Step 1: ACTIVE SCAN - Detect header/footer zones from span positions
    # FIXED v3.3: Directly scan spans rather than relying on pre-populated bands
    # =========================================================================
    header_band_pages: Dict[int, Set[int]] = {}  # rounded_y -> set of page indices
    footer_band_pages: Dict[int, Set[int]] = {}
    header_band_texts: Dict[int, List[str]] = {}  # rounded_y -> list of texts
    footer_band_texts: Dict[int, List[str]] = {}

    for page_idx, page_data in enumerate(page_outputs):
        # Get page height (required for zone calculation)
        page_height = page_data.get("page_height", _FILTER_DEFAULT_PAGE_HEIGHT)

        # Define header/footer zones using configurable threshold
        header_zone_max_y = page_height * _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT

        # Prefer adaptive footer boundary if available
        footer_zone_min_y = page_data.get("structure", {}).get(
            "footer_zone_min_y",
            page_height * (1.0 - _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT)
        )

        # Get spans from either structure
        spans = page_data.get("spans", page_data.get("content", []))

        for span in spans:
            bbox = span.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            span_y_top = bbox[1]  # y0 (top of span)
            text = span.get("raw_text", span.get("text", "")).strip()

            # Skip empty or very short text
            if not text or len(text) < 2:
                continue

                # HARDENED: Continuation shield
                # If a fragment starts lowercase, it is likely a cross-page continuation.
            if text[0].islower():
                continue

            # SAFETY FIX: Explicit Binning (Round to nearest 10px bucket)
            # Uses _REGION_Y_BAND_ROUNDING (10) for unambiguous grouping
            base_fs = span.get("font_size", 10.0)
            if global_median_font_size and global_median_font_size > 0:
                base_fs = global_median_font_size

            bucket_size = max(6.0, base_fs * 0.6)
            rounded_y = int(round(span_y_top / bucket_size) * bucket_size)

            # Check if span is in header zone
            if span_y_top <= header_zone_max_y:
                if rounded_y not in header_band_pages:
                    header_band_pages[rounded_y] = set()
                    header_band_texts[rounded_y] = []
                header_band_pages[rounded_y].add(page_idx)
                header_band_texts[rounded_y].append(text)

            # Check if span is in footer zone
            elif span_y_top >= footer_zone_min_y:
                if rounded_y not in footer_band_pages:
                    footer_band_pages[rounded_y] = set()
                    footer_band_texts[rounded_y] = []
                footer_band_pages[rounded_y].add(page_idx)
                footer_band_texts[rounded_y].append(text)

    # =========================================================================
    # Step 2: Merge nearby bands and apply frequency threshold
    # FIXED v3.3: Merge bands within tolerance before frequency check
    # =========================================================================

    def merge_nearby_bands(
            band_pages: Dict[int, Set[int]],
            band_texts: Dict[int, List[str]],
            tolerance: int
    ) -> Tuple[Dict[int, Set[int]], Dict[int, List[str]]]:
        """Merge bands within tolerance pixels of each other."""
        if not band_pages:
            return band_pages, band_texts

        sorted_bands = sorted(band_pages.keys())
        merged_pages: Dict[int, Set[int]] = {}
        merged_texts: Dict[int, List[str]] = {}

        current_anchor = sorted_bands[0]
        merged_pages[current_anchor] = set(band_pages[current_anchor])
        merged_texts[current_anchor] = list(band_texts.get(current_anchor, []))

        for band in sorted_bands[1:]:
            if band - current_anchor <= tolerance:
                # Merge into current anchor
                merged_pages[current_anchor].update(band_pages[band])
                merged_texts[current_anchor].extend(band_texts.get(band, []))
            else:
                # Start new anchor
                current_anchor = band
                merged_pages[current_anchor] = set(band_pages[current_anchor])
                merged_texts[current_anchor] = list(band_texts.get(current_anchor, []))

        return merged_pages, merged_texts

    # HARDENED: Merge adjacent bands caused by scan drift (NEW SPECIALIST TWEAK)
    # Use the calculated bucket_size (or global equivalent) as the tolerance.
    merge_tolerance = max(6.0, (global_median_font_size or 10.0) * 0.6)

    header_band_pages, header_band_texts = merge_nearby_bands(
        header_band_pages, header_band_texts, merge_tolerance
    )
    footer_band_pages, footer_band_texts = merge_nearby_bands(
        footer_band_pages, footer_band_texts, merge_tolerance
    )
    footer_band_pages, footer_band_texts = merge_nearby_bands(
        footer_band_pages, footer_band_texts, _GLOBAL_BAND_MERGE_TOLERANCE
    )

    # Apply frequency threshold
    # FIX: Use min_floor to prevent single-page artifacts from polluting global bands
    min_floor = 2 if total_pages > 1 else 1
    frequency_threshold = max(min_floor, int(total_pages * _GLOBAL_BAND_MIN_PAGE_FRACTION))

    global_headers: Set[int] = set()
    global_footers: Set[int] = set()

    for y_band, pages in header_band_pages.items():
        if len(pages) >= frequency_threshold:
            global_headers.add(y_band)

    for y_band, pages in footer_band_pages.items():
        if len(pages) >= frequency_threshold:
            global_footers.add(y_band)

    # =========================================================================
    # Step 3: Return with diagnostic logging
    # =========================================================================
    if trace_id:
        logger.info(
            "[%s] Global bands detected: %d header, %d footer (scanned %d pages, threshold=%d)",
            trace_id, len(global_headers), len(global_footers), total_pages, frequency_threshold
        )
        if global_headers:
            logger.debug("[%s] Header Y-bands: %s", trace_id, sorted(list(global_headers)))
        if global_footers:
            logger.debug("[%s] Footer Y-bands: %s", trace_id, sorted(list(global_footers)))

    return {
        "header_bands": sorted(list(global_headers)),
        "footer_bands": sorted(list(global_footers))
    }


def _apply_global_header_footer_roles(
        all_spans: List[Dict],
        global_bands: Dict[str, Any],
        page_heights: Dict[int, float],
        trace_id: str = None
) -> int:
    """
    Tag spans in global header/footer regions with appropriate roles.

    NEW v2.4: Applies computed global bands and text patterns to tag spans.

    This function must be called AFTER _compute_global_header_footer_bands()
    and BEFORE sentence segmentation to ensure header/footer content is
    properly excluded from TTS output.

    Matching Strategy:
        - Y-band match: Span's rounded Y is in global header/footer bands

    Args:
        all_spans: List of all span dictionaries to process.
        global_bands: Output from _compute_global_header_footer_bands().
        page_heights: Mapping of page_number → page_height in points.
        trace_id: Optional trace ID for logging.

    Returns:
        Count of spans tagged as header/footer.
    """
    header_bands = set(global_bands.get("header_bands", []))
    footer_bands = set(global_bands.get("footer_bands", []))

    tagged_header = 0
    tagged_footer = 0

    # Roles eligible for global header/footer reclassification.
    # CAPTION included: running headers frequently misclassified as
    # captions (short text adjacent to page numbers on same Y-band).
    # True mid-page captions are protected by the Y-band + zone guard
    # below — only spans physically in the header/footer strip match.
    # HEADING, TABLE, and other structural roles remain protected.
    _GLOBAL_HF_ELIGIBLE_ROLES = frozenset({
        TextRole.BODY.value,
        TextRole.CAPTION.value,
    })

    for span in all_spans:
        # Only reclassify eligible roles in header/footer bands.
        # Prevents clobbering validated HEADINGS, TABLES, or CODE.
        current_role = span.get("role", TextRole.BODY.value)
        if current_role not in _GLOBAL_HF_ELIGIBLE_ROLES:
            continue

        bbox = span.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        page_num = span.get("page_number", 1)
        page_height = page_heights.get(page_num, 800)

        # Calculate zones for safety check
        header_zone_max_y = page_height * _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT
        footer_zone_min_y = page_height * (1.0 - _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT)

        # =====================================================================
        # Y-Band Matching
        # =====================================================================
        span_y = bbox[1]
        y_rounded = int(round(span_y / _REGION_Y_BAND_ROUNDING) * _REGION_Y_BAND_ROUNDING)

        in_header_band = y_rounded in header_bands
        in_footer_band = y_rounded in footer_bands

        # =====================================================================
        # Apply Tags (OR logic: either match triggers)
        # =====================================================================
        # Guard: Even if in band, span must be physically in the zone
        if in_header_band and span_y > header_zone_max_y:
            in_header_band = False
        if in_footer_band and span_y < footer_zone_min_y:
            in_footer_band = False

        if in_header_band:
            span["role"] = TextRole.HEADER_ARTIFACT.value
            span["is_global_header"] = True
            span["header_match_reason"] = "band"
            tagged_header += 1
        elif in_footer_band:
            span["role"] = TextRole.FOOTER_ARTIFACT.value
            span["is_global_footer"] = True
            span["footer_match_reason"] = "band"
            tagged_footer += 1

    if trace_id:
        logger.info(
            "[%s] Global header/footer tagging: %d headers, %d footers tagged "
            "(bands: H=%d F=%d)",
            trace_id, tagged_header, tagged_footer,
            len(header_bands), len(footer_bands)
        )

    return tagged_header + tagged_footer


def _summarize_document_bands(
        raw_data_list: List[Dict],
        global_header_bands: List[int],
        global_footer_bands: List[int],
) -> Dict[str, List[Dict]]:
    """
    Build human-readable summaries for global header/footer bands.

    For each global band Y, captures a representative text snippet
    from any span (content or excluded) whose rounded Y matches.

    Args:
        raw_data_list: List of page data dictionaries.
        global_header_bands: List of global header Y-values.
        global_footer_bands: List of global footer Y-values.

    Returns:
        Dictionary with structure:
            {
                "headers": [{"y": 54, "sample_text": "CHAPTER 3 - METHODS"}, ...],
                "footers": [{"y": 760, "sample_text": "© 2024 Publisher"}, ...],
            }
    """
    header_set: set = set(global_header_bands)
    footer_set: set = set(global_footer_bands)

    # HARDENED: Collect set of samples to capture variable headers
    band_to_samples_header: Dict[int, Set[str]] = {y: set() for y in header_set}
    band_to_samples_footer: Dict[int, Set[str]] = {y: set() for y in footer_set}

    for page_data in raw_data_list:
        # Consider both kept and excluded spans
        content_spans = page_data.get("content", [])
        excluded_spans = page_data.get("excluded", [])
        all_spans = list(content_spans) + list(excluded_spans)

        for span in all_spans:
            # Extract Y from tuple bbox: (x0, y0, x1, y1)
            bbox = span.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue

            y = bbox[1]  # y0

            # Round to nearest 10 (consistent with band detection)
            y_band = int(round(y / _REGION_Y_BAND_ROUNDING) * _REGION_Y_BAND_ROUNDING)

            # Get sample text
            text = (span.get("cleaned_text") or span.get("raw_text") or "").strip()
            if not text:
                continue

            # Collect unique variants
            if y_band in header_set:
                band_to_samples_header[y_band].add(text)

            if y_band in footer_set:
                band_to_samples_footer[y_band].add(text)

    # Build output summaries (return lists)
    headers_summary: List[Dict] = [
        {"y": y, "samples": sorted(list(band_to_samples_header[y]))}
        for y in sorted(header_set)
    ]

    footers_summary: List[Dict] = [
        {"y": y, "samples": sorted(list(band_to_samples_footer[y]))}
        for y in sorted(footer_set)
    ]

    return {
        "headers": headers_summary,
        "footers": footers_summary,
    }


# ✦────── c. Application & Filtering ──────✦

def _filter_spans_by_global_bands(
        page_data: Dict,
        global_header_bands: List[int],
        global_footer_bands: List[int],
        header_samples: List[str] = None,
        footer_samples: List[str] = None,
        page_height: float = None,
        trace_id: str = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Semantic-aware header/footer filtering.

    A span is excluded only if BOTH conditions are met:
        1. Its Y-position is within tolerance of a global band
        2. Its text matches (or is similar to) known header/footer samples
           OR no samples are available (band-only mode)

    Args:
        page_data: Page dictionary with 'content' key containing spans.
        global_header_bands: List of Y-positions identified as headers.
        global_footer_bands: List of Y-positions identified as footers.
        header_samples: Sample text strings from detected headers.
        footer_samples: Sample text strings from detected footers.
        page_height: Page height for adaptive tolerance.
        trace_id: Optional trace ID for logging.

    Returns:
        Tuple of (filtered_spans, excluded_spans).
    """
    # =========================================================================
    # EXTRACT SPANS
    # =========================================================================
    spans = page_data.get("content", [])
    if not spans:
        return [], []

    # If no bands defined, nothing to filter
    if not global_header_bands and not global_footer_bands:
        return spans, []

    # =========================================================================
    # CONFIGURATION: Adaptive tolerance
    # =========================================================================
    effective_page_height = (
        page_height if page_height and page_height > 0
        else _FILTER_DEFAULT_PAGE_HEIGHT
    )

    tolerance = max(
        _FILTER_BAND_MIN_TOLERANCE,
        min(
            _FILTER_BAND_MAX_TOLERANCE,
            int(effective_page_height * _FILTER_BAND_TOLERANCE_RATIO)
        )
    )

    # Prepare sample sets (normalized for comparison)
    normalized_header_samples: List[str] = []
    normalized_footer_samples: List[str] = []

    if header_samples:
        normalized_header_samples = [
            _normalize_text(s) for s in header_samples
            if s and len(_normalize_text(s)) >= _FILTER_MIN_SAMPLE_LENGTH
        ]

    if footer_samples:
        normalized_footer_samples = [
            _normalize_text(s) for s in footer_samples
            if s and len(_normalize_text(s)) >= _FILTER_MIN_SAMPLE_LENGTH
        ]

    # Convert band lists to sets for O(1) lookup
    header_band_set: set = set(global_header_bands)
    footer_band_set: set = set(global_footer_bands)

    # =========================================================================
    # FILTERING LOOP
    # =========================================================================
    filtered_spans: List[Dict] = []
    excluded_spans: List[Dict] = []

    for span in spans:

        # ------------------------------------------------------------------
        # ROLE SHIELD — Semantic intent overrides band-based filtering
        # Protect validated content roles from global header/footer exclusion
        # ------------------------------------------------------------------
        role = span.get("role")
        if role in {
            TextRole.CAPTION.value,
            TextRole.HEADING.value,
            TextRole.SUBHEADING.value,
        }:
            filtered_spans.append(span)
            continue

        # Extract Y from tuple bbox: (x0, y0, x1, y1)
        bbox = span.get("bbox")
        if bbox is None or len(bbox) < 4:
            # No valid bbox — keep span (don't exclude based on uncertainty)
            filtered_spans.append(span)
            continue

        span_y = bbox[1]  # y0

        # Round to match band detection granularity
        span_y_rounded = round(span_y, _GLOBAL_BAND_ROUNDING_PRECISION)

        # Check if span Y matches any header band (within tolerance)
        matches_header_band = any(
            abs(span_y_rounded - band) <= tolerance
            for band in header_band_set
        )

        # Check if span Y matches any footer band (within tolerance)
        matches_footer_band = any(
            abs(span_y_rounded - band) <= tolerance
            for band in footer_band_set
        )

        # If no band match, keep the span
        if not matches_header_band and not matches_footer_band:
            filtered_spans.append(span)
            continue

        # =====================================================================
        # BAND MATCHED — Check text similarity
        # =====================================================================
        span_text = (span.get("cleaned_text") or span.get("raw_text") or "").strip()
        normalized_span_text = _normalize_text(span_text)
        should_exclude = False
        exclusion_reason = ""

        # Shared "Body Text" Shield Logic (Symmetrical)
        # 1. Lowercase start = likely continuation
        # 2. Dense text = likely displaced body
        is_continuation = bool(span_text) and span_text[0].islower()
        is_dense_body = len(span_text) > 60 or len(span_text.split()) > 6
        is_body_like = is_continuation or is_dense_body

        # =========================================================
        # HEADER BAND LOGIC (Semantic-First, Non-Destructive)
        # =========================================================
        if matches_header_band:
            if is_body_like:
                should_exclude = False

            elif not normalized_header_samples:
                # HARDENED: Never delete based on position alone.
                should_exclude = False

            else:
                similarity_threshold = (
                    _FILTER_SHORT_TEXT_THRESHOLD
                    if len(normalized_span_text) < _FILTER_SHORT_TEXT_LENGTH
                    else _FILTER_FUZZY_MATCH_THRESHOLD
                )

                # Semantic confirmation required
                for sample in normalized_header_samples:
                    similarity = _text_similarity(normalized_span_text, sample)
                    if similarity >= similarity_threshold:
                        should_exclude = True
                        exclusion_reason = f"header_match_{similarity:.2f}"
                        break

        # =========================================================
        # FOOTER BAND LOGIC (Runs only if header did not exclude)
        # =========================================================
        if not should_exclude and matches_footer_band:
            if is_body_like:
                should_exclude = False

            elif not normalized_footer_samples:
                # HARDENED: Never delete based on position alone.
                should_exclude = False

            else:
                similarity_threshold = (
                    _FILTER_SHORT_TEXT_THRESHOLD
                    if len(normalized_span_text) < _FILTER_SHORT_TEXT_LENGTH
                    else _FILTER_FUZZY_MATCH_THRESHOLD
                )

                # Semantic confirmation required
                for sample in normalized_footer_samples:
                    similarity = _text_similarity(normalized_span_text, sample)
                    if similarity >= similarity_threshold:
                        should_exclude = True
                        exclusion_reason = f"footer_match_{similarity:.2f}"
                        break

        # =====================================================================
        # ASSIGNMENT
        # =====================================================================
        if should_exclude:
            span["exclusion_reason"] = exclusion_reason
            excluded_spans.append(span)

            if trace_id:
                logger.debug(
                    "[%s] Excluded span: '%s' (reason=%s, y=%.0f)",
                    trace_id, span_text[:30], exclusion_reason, span_y
                )
        else:
            filtered_spans.append(span)

    # =========================================================================
    # LOGGING
    # =========================================================================
    if trace_id:
        logger.debug(
            "[%s] Band filtering: %d kept, %d excluded",
            trace_id, len(filtered_spans), len(excluded_spans)
        )

    return filtered_spans, excluded_spans


def _apply_global_band_signals(
        classified_spans: List[Dict],
        global_header_bands: List[int],
        global_footer_bands: List[int],
        *,
        header_samples: List[str] = None,
        footer_samples: List[str] = None,
        trace_id: str = None,
) -> None:
    """
    Schema v2.0: Apply global header/footer signals WITHOUT excluding.
    Mutates spans in place by adding candidate flags (via _flag_candidate()).

    Ported from _filter_spans_by_global_bands:
      - same band match logic
      - same similarity thresholding
      - same "body-like shield" (lowercase start / dense text)
    """

    if not classified_spans:
        return

    header_band_set: set = set(global_header_bands or [])
    footer_band_set: set = set(global_footer_bands or [])
    if not header_band_set and not footer_band_set:
        return

    tolerance = _GLOBAL_BAND_MERGE_TOLERANCE
    normalized_header_samples = [_normalize_text(s) for s in (header_samples or []) if s]
    normalized_footer_samples = [_normalize_text(s) for s in (footer_samples or []) if s]

    kept_band_matches = 0
    flagged_band_matches = 0

    for span in classified_spans:
        if not isinstance(span, dict):
            continue

        # Role shield (same as legacy): don't band-flag semantic roles
        role = span.get("role")
        if role in {
            TextRole.CAPTION.value,
            TextRole.HEADING.value,
            TextRole.SUBHEADING.value,
        }:
            continue

        bbox = span.get("bbox")
        if bbox is None or len(bbox) < 4:
            # Uncertain geometry: do not band-flag
            continue

        span_y = bbox[1]
        span_y_rounded = round(span_y, _GLOBAL_BAND_ROUNDING_PRECISION)

        matches_header_band = any(
            abs(span_y_rounded - band) <= tolerance for band in header_band_set)
        matches_footer_band = any(
            abs(span_y_rounded - band) <= tolerance for band in footer_band_set)

        if not matches_header_band and not matches_footer_band:
            continue

        span_text = (span.get("cleaned_text") or span.get("raw_text") or "").strip()
        normalized_span_text = _normalize_text(span_text)

        # Body-like shield logic (symmetrical, per legacy)
        is_continuation = bool(span_text) and span_text[0].islower()
        is_dense_body = len(span_text) > 60 or len(span_text.split()) > 6
        is_body_like = is_continuation or is_dense_body

        should_flag = False
        flag_reason = ""

        # Header
        if matches_header_band:
            if is_body_like or not normalized_header_samples:
                should_flag = False
            else:
                similarity_threshold = (
                    _FILTER_SHORT_TEXT_THRESHOLD
                    if len(normalized_span_text) < _FILTER_SHORT_TEXT_LENGTH
                    else _FILTER_FUZZY_MATCH_THRESHOLD
                )
                for sample in normalized_header_samples:
                    similarity = _text_similarity(normalized_span_text, sample)
                    if similarity >= similarity_threshold:
                        should_flag = True
                        flag_reason = f"global_header_match_{similarity:.2f}"
                        break

        # Footer (only if not already flagged by header)
        if (not should_flag) and matches_footer_band:
            if is_body_like or not normalized_footer_samples:
                should_flag = False
            else:
                similarity_threshold = (
                    _FILTER_SHORT_TEXT_THRESHOLD
                    if len(normalized_span_text) < _FILTER_SHORT_TEXT_LENGTH
                    else _FILTER_FUZZY_MATCH_THRESHOLD
                )
                for sample in normalized_footer_samples:
                    similarity = _text_similarity(normalized_span_text, sample)
                    if similarity >= similarity_threshold:
                        should_flag = True
                        flag_reason = f"global_footer_match_{similarity:.2f}"
                        break

        if should_flag:
            # E2 taxonomy: global band matches are high-confidence → requires_review=False
            _flag_candidate(span, flag_reason, requires_review=False)
            flagged_band_matches += 1
        else:
            kept_band_matches += 1

    if trace_id:
        logger.debug(
            "[%s] Global band signals applied: flagged=%d kept_by_shield=%d total=%d",
            trace_id, flagged_band_matches, kept_band_matches, len(classified_spans)
        )

# Wrapper for process.py
def refine_roles_across_document(
        page_outputs: list[dict],
        trace_id: str = None
) -> None:
    """
    Document-scope role refinement pass.

    Applies content-flow outlier logic (including bibliography detection)
    to cache-original spans before Stage 2 windowing.
    """
    if not page_outputs:
        return

    all_spans = []
    for page in page_outputs:
        spans = page.get("classified_spans")
        if spans:
            all_spans.extend(spans)

    if all_spans:
        _refine_roles_via_content_flow(all_spans, trace_id=trace_id)


def normalize_header_footer_across_document(
        page_outputs: List[Dict],
        trace_id: str = None
) -> None:
    """
    Normalize header/footer bands across the entire document.

    Orchestration steps:
        1. Compute global header/footer bands (frequency-based)
        2. Build sample text summaries for each band
        3. Apply filtering to each page using semantic matching
        4. Update page structures with global bands

    Args:
        page_outputs: List of page data dictionaries.
        trace_id: Optional trace ID for logging.

    Mutates:
        Each page's 'content', 'excluded', and 'structure' in place.
    """
    if not page_outputs:
        return

    # =========================================================================
    # STEP 1: Compute global bands
    # =========================================================================
    # HARDENED: Calculate Global Median Font Size on the fly for bucket stability
    # This enables the resolution-agnostic optimizations in _compute_global_header_footer_bands
    all_fonts = []
    for p in page_outputs:
        # Collect from both content and previously excluded spans (if any)
        for s in p.get("content", []) + p.get("excluded", []):
            fs = s.get("font_size", 0)
            if fs > 0:
                all_fonts.append(fs)

    global_median_fs = 10.0
    if all_fonts:
        all_fonts.sort()
        global_median_fs = all_fonts[len(all_fonts) // 2]

    # Pass the calculated median to the band detector
    # This hardens BOTH headers and footers because they use the same bucketing math
    global_bands = _compute_global_header_footer_bands(
        page_outputs,
        trace_id=trace_id,
        global_median_font_size=global_median_fs
    )

    global_header_bands = global_bands["header_bands"]
    global_footer_bands = global_bands["footer_bands"]

    # =========================================================================
    # STEP 2: Build sample text summaries
    # =========================================================================
    band_summaries = _summarize_document_bands(
        page_outputs,
        global_header_bands,
        global_footer_bands
    )

    # Extract sample texts for filtering (Flatten lists)
    header_samples: List[str] = []
    for h in band_summaries["headers"]:
        header_samples.extend(h.get("samples", []))

    footer_samples: List[str] = []
    for f in band_summaries["footers"]:
        footer_samples.extend(f.get("samples", []))

    if trace_id:
        logger.debug(
            "[%s] Band samples: %d header, %d footer",
            trace_id, len(header_samples), len(footer_samples)
        )

    # =========================================================================
    # STEP 3: Apply filtering to each page
    # =========================================================================
    total_excluded = 0

    for page_idx, page_data in enumerate(page_outputs):
        # Get page height from metadata if available
        page_height = page_data.get("metadata", {}).get("height")

        # Filter spans
        filtered_spans, newly_excluded = _filter_spans_by_global_bands(
            page_data,
            global_header_bands,
            global_footer_bands,
            header_samples=header_samples,
            footer_samples=footer_samples,
            page_height=page_height,
            trace_id=trace_id
        )

        # Update page data
        page_data["content"] = filtered_spans

        # Merge newly excluded with existing excluded
        existing_excluded = page_data.get("excluded", [])
        page_data["excluded"] = existing_excluded + newly_excluded

        total_excluded += len(newly_excluded)

        # =====================================================================
        # STEP 4: Update structure with global bands
        # =====================================================================
        if "structure" not in page_data:
            page_data["structure"] = {}

        page_data["structure"]["header_bands"] = global_header_bands
        page_data["structure"]["footer_bands"] = global_footer_bands

    # =========================================================================
    # FINAL LOGGING
    # =========================================================================
    if trace_id:
        logger.info(
            "[%s] Header/footer normalization complete: %d spans excluded across %d pages",
            trace_id, total_excluded, len(page_outputs)
        )


# ✦                  ✦                  ✦                  ✦
# ✦───────────── 6 TTS Preparation  ─────────────✦
# ✦                  ✦                  ✦                  ✦

def _split_into_prosodic_clauses(
        text: str,
        sentence_metadata: Dict = None,
        trace_id: str = None,
) -> List[str]:
    """
    Split a long, clause-dense sentence into decoder-safe prosodic clauses.

    CONTRACT:
        - Narration-only transformation (no semantic/span/chunk changes)
        - Clauses are intended for sequential TTS generation with micro-gaps
        - Fail-open: returns [text] if splitting is unsafe or unnecessary

    SAFETY GUARDRAILS:
        - Minimum 4 words per clause (no tiny fragments)
        - Maximum 4 clauses total (decoder pacing)
        - Preserves connector words with following clause ("and joints" not "joints")

    Args:
        text: The sanitized sentence text to analyze.
        sentence_metadata: Optional RONC metadata (for future authority checks).
        trace_id: Optional trace ID for logging.

    Returns:
        List[str] - always a list, even if single element [text].
        Returns [text] unchanged if:
            - Text is short/simple enough
            - No valid split points found
            - Splitting would create fragments < 5 words
            - Would produce > 4 clauses

    Example:
        Input:  "The third or deep pain, arising from viscera, musculature
                 and joints, is also poorly localized, can be chronic and
                 is often associated with referred pain."

        Output: ["The third or deep pain",
                 "arising from viscera, musculature and joints",
                 "is also poorly localized",
                 "can be chronic and is often associated with referred pain."]
    """
    # -------------------------------------------------------------------------
    # GUARD: Skip if short/simple
    # -------------------------------------------------------------------------
    if not text or not isinstance(text, str):
        return [text] if text else []

    text = text.strip()
    # Treat semicolons as comma-equivalent for prosodic boundary detection only
    text_for_split = text.replace(";", ",")
    comma_count = text_for_split.count(",")

    # No commas = no clause boundaries to split on
    if comma_count < _TTS_PROSODIC_SPLIT_COMMA_THRESHOLD:
        return [text]

    # Short text = no splitting needed
    if len(text) <= _TTS_PROSODIC_SPLIT_CHAR_THRESHOLD:
        return [text]

    # -------------------------------------------------------------------------
    # Try each pattern (strongest to weakest) until one produces valid split
    # -------------------------------------------------------------------------
    for pattern in _TTS_PROSODIC_CLAUSE_PATTERNS:
        splits = pattern.split(text_for_split)

        # Pattern didn't match (need at least 3 parts: before, connector, after)
        if len(splits) < 3:
            if trace_id:
                logger.debug(
                    "[%s] Clause split rejected: pattern did not match (len(splits)=%d)",
                    trace_id, len(splits)
                )
            continue

        # Reconstruct clauses, keeping connector with following text
        clauses = []
        buf = splits[0].rstrip().lstrip()

        i = 1
        while i < len(splits) - 1:
            connector = splits[i]
            remainder = splits[i + 1] if i + 1 < len(splits) else ""

            # Finalize current buffer as a clause
            if buf:
                clauses.append(buf.rstrip(',').strip())

            # Start new buffer with connector + remainder
            buf = f"{connector.strip()} {remainder}".strip()
            i += 2

        # Don't forget the final buffer
        if buf:
            clauses.append(buf.strip())

        # Deduplicate accidental repeats (defensive)
        clauses = [c for i, c in enumerate(clauses) if i == 0 or c != clauses[i - 1]]
        # -----------------------------------------------------------------
        # SAFETY VALIDATION
        # -----------------------------------------------------------------
        if len(clauses) <= 1:
            if trace_id:
                logger.debug("[%s] Clause split rejected: pattern produced single clause", trace_id)
            continue  # No real split, try next pattern
        if len(clauses) > _TTS_PROSODIC_MAX_CLAUSES:
            if trace_id:
                logger.debug("[%s] Clause split rejected: %d clauses exceeds max %d",
                             trace_id, len(clauses), _TTS_PROSODIC_MAX_CLAUSES)
            continue  # Too many clauses, try next pattern

        # Check minimum word count for each clause
        all_safe = True
        for clause in clauses:
            word_count = len(clause.strip(".,;:!?()[]\"' ").split())
            if word_count < _TTS_PROSODIC_MIN_CLAUSE_WORDS:
                all_safe = False
                break

        if not all_safe:
            if trace_id:
                logger.debug("[%s] Clause split rejected: clause below min %d words",
                             trace_id, _TTS_PROSODIC_MIN_CLAUSE_WORDS)
            continue  # Fragment too small, try next pattern

        # Guard: never increase total text length materially
        if sum(len(c) for c in clauses) > len(text) * 1.1:
            if trace_id:
                logger.debug(
                    "[%s] Clause split rejected: length inflation (sum=%d > 1.1*%d)",
                    trace_id,
                    sum(len(c) for c in clauses),
                    len(text)
                )
            continue

        # -----------------------------------------------------------------
        # SUCCESS: Valid split found
        # -----------------------------------------------------------------
        # Preserve terminal punctuation on final clause
        original_terminal = text.rstrip()[-1:] if text.rstrip() else ''
        if original_terminal in '.!?' and clauses:
            final = clauses[-1].rstrip('.!?,;:')
            clauses[-1] = final + original_terminal

        if trace_id:
            logger.debug(
                "[%s] Prosodic clause split: pattern=%s, %d clauses, sizes=%s",
                trace_id,
                pattern.pattern[:30],
                len(clauses),
                [(len(c), c[:40]) for c in clauses]
            )

        return clauses

    # -------------------------------------------------------------------------
    # FAIL-OPEN: No pattern produced valid split
    # -------------------------------------------------------------------------
    return [text]


def _sanitize_for_tts(
        text: str,
        role: str = TextRole.BODY.value,
        add_terminal_punct: bool = True,
        change_tracker: Dict = None,
        trace_id: str = None,
        sentence_metadata: Dict = None
) -> str:
    """
    Prepare text for TTS audio generation.

    ARCHITECTURAL GUARDRAIL:
    This method is the SOLE authority for pronunciation transformations.
    It is the ONLY place allowed to change how text sounds.

    This method must NOT:
      - Perform structural normalization (owned by _clean_spans)
      - Perform stream shaping (owned by _reconstruct_text_for_segmentation)
      - Perform sentence segmentation or healing (owned by _segment_sentences)

    Operations performed:
        1. Smart case normalization (preserves acronyms)
        2. Subscript/superscript expansion
        3. Unit notation expansion
        4. Symbol substitution
        5. Role-aware noise removal (post-segmentation safe)
        6. Empty/orphan bracket removal
        7. Final spacing safety cleanup
        8. Terminal punctuation enforcement
        9. RONC-aware decoder runaway mitigation (v1.4.0)

    Args:
        text: Sentence text to sanitize.
        role: Text role for role-aware noise removal (defaults to BODY).
        add_terminal_punct: Whether to enforce terminal punctuation.
        change_tracker: Optional mutation tracking dictionary.
        trace_id: Optional trace ID for logging.
        sentence_metadata: Optional sentence dict containing RONC signals
            (source_unit_ids, _source_spans) for intelligent split decisions.

    Returns:
        TTS-safe sanitized text.
    """
    # This method MUST be applied ONLY to finalized sentences.
    # It MUST NOT operate on provisional or pre-segmentation text.

    # =========================================================================
    # HANDLE EMPTY INPUT
    # =========================================================================
    if not text:
        if change_tracker is not None:
            change_tracker.update({
                "original_length": 0,
                "sanitized_length": 0,
                "length_delta": 0,
                "modifications": []
            })
        return ""

    original_length = len(text)
    modifications: List[str] = []

    # =========================================================================
    # STEP 0: Punctuation spacing normalization (FIX v6.4)
    # =========================================================================
    # Wire up _normalize_punctuation_spacing (defined line 14254, never called).
    # Fixes TTS decoder crashes caused by space-padded punctuation (" ; ", " . ")
    # that cause the model to loop indefinitely (max_decoder_steps overflow).
    original = text
    text = _normalize_punctuation_spacing(text)
    if text != original:
        modifications.append("punctuation_spacing_normalized")

    # =========================================================================
    # STEP 1: Smart case normalization (prevent TTS from "shouting")
    # =========================================================================
    if text.isupper() and len(text) > _TTS_ACRONYM_LENGTH_THRESHOLD:
        # Check if it's likely an acronym (no spaces, short)
        if " " in text or len(text) > _TTS_LONG_TEXT_THRESHOLD:
            text = _smart_title_case(text)
            modifications.append("case_normalized")

    # =========================================================================
    # STEP 2: Subscript expansion (for chemical formulas and notation)
    # =========================================================================
    original = text

    def expand_chemical_subscript(match: re.Match) -> str:
        """Expand chemical formula subscripts: H₂O → H 2 O."""
        letter = match.group(1)
        subscripts = match.group(2)
        digits = "".join(SUBSCRIPT_DIGITS.get(c, c) for c in subscripts)
        return f"{letter} {digits}"

    text = _CHEMICAL_SUBSCRIPT_PATTERN.sub(expand_chemical_subscript, text)

    # Handle any remaining standalone subscripts
    text = text.translate(_SUBSCRIPT_TABLE)

    if text != original:
        modifications.append("subscript_expanded")

    # =========================================================================
    # STEP 3: Superscript expansion
    # =========================================================================
    original = text

    # Ordinal superscripts: 1ˢᵗ → 1st, 2ⁿᵈ → 2nd, 3ʳᵈ → 3rd, 4ᵗʰ → 4th
    text = _TTS_ORDINAL_1ST_PATTERN.sub(r"\1st", text)
    text = _TTS_ORDINAL_2ND_PATTERN.sub(r"\1nd", text)
    text = _TTS_ORDINAL_3RD_PATTERN.sub(r"\1rd", text)
    text = _TTS_ORDINAL_NTH_PATTERN.sub(r"\1th", text)

    # Common mathematical superscripts with verbal equivalents
    text = _TTS_SQUARED_PATTERN.sub(r"\1 squared", text)
    text = _TTS_CUBED_PATTERN.sub(r"\1 cubed", text)

    # Handle remaining superscripts → regular characters
    text = text.translate(_SUPERSCRIPT_TABLE)

    if text != original:
        modifications.append("superscript_expanded")

    # =========================================================================
    # STEP 4: Unit notation expansion (BEFORE blanket substitutions)
    # Converts: 50/mm² → 50 per mm², 10/s → 10 per s
    # =========================================================================
    original_step35 = text
    text = _TTS_UNIT_SLASH_PATTERN.sub(r'\1 per \2', text)

    if text != original_step35:
        modifications.append("unit_notation_expanded")

    # =========================================================================
    # STEP 5: Character substitutions (symbols → words)
    # =========================================================================
    original = text

    for char, substitution in TTS_SUBSTITUTIONS.items():
        if char == "/":
            text = text.replace(" / ", substitution)  # Spaced separator
            text = re.sub(r'(?<=[a-zA-Z])/(?=[a-zA-Z])', substitution, text)  # Word/Word
            text = re.sub(r'(?<=[a-zA-Z])/\s+', f'{substitution} ', text)  # NEW: Word/ space
        elif char in text:
            text = text.replace(char, substitution)

    if text != original:
        modifications.append("symbols_substituted")

    # =========================================================================
    # STEP 6: Remove empty and orphan brackets
    # =========================================================================
    original = text

    # Empty brackets: (), [], {}
    text = _CLEAN_EMPTY_BRACKETS_PATTERN.sub("", text)

    # Brackets containing only punctuation: (,), [.], etc.
    text = _PUNCT_ONLY_BRACKETS_PATTERN.sub("", text)

    if text != original:
        modifications.append("brackets_cleaned")

    # =========================================================================
    # STEP 6.5: Drop leading glyph-noise tokens (narration-only hygiene)
    #
    # Motivation:
    # Some PDFs yield stray single-letter glyph spans from attributions like
    # "FROM WIKIBOOKS" that get reconstructed as "F W Our ...".
    #
    # Scope:
    # - Only affects spoken text (this method only)
    # - Does NOT alter semantics, CIDs, or span roles
    #
    # Conservative rule:
    # - Only remove 1-2 character ALL-CAPS alphabetic tokens
    # - Only when they occur at the very beginning of the sentence
    # =========================================================================
    original = text

    stripped = text.lstrip()
    lead = ''
    if stripped[:1] in '"\'([{':
        lead = stripped[:1]
        stripped = stripped[1:].lstrip()

    parts = stripped.split()
    removed = []

    # Remove up to 3 leading glyph-noise tokens (covers "F W", "FW", etc.)
    while parts and len(removed) < 3:
        tok = parts[0]
        core = tok.strip('.,:;!?"\'()[]{}')

        # Preserve known legitimate initialisms
        if core in ('US', 'UK', 'EU', 'UN', 'AI'):
            break

        if len(core) <= 2 and core.isalpha() and core.isupper():
            removed.append(tok)
            parts = parts[1:]
            continue
        break

    if removed:
        rebuilt = ('{} '.format(lead) if lead else '') + ' '.join(parts)
        text = rebuilt.strip()
        modifications.append('leading_glyph_noise_removed')
        if trace_id:
            logger.debug(
                '[%s] Leading glyph-noise removed: removed=%r before=%r after=%r',
                trace_id, removed, original[:60], text[:60]
            )

    # =========================================================================
    # STEP 6.3: Number-word decompounding (TEXT_DEHYPHENATE casualty repair)
    # =========================================================================
    original = text
    text = _NUMBER_DECOMPOUND_PATTERN.sub(r"\1-\2", text)
    if text != original:
        modifications.append("number_word_decompounded")

    # =========================================================================
    # STEP 7: Role-aware noise removal (post-segmentation safe)
    # =========================================================================
    _NOISE_REMOVAL_ALLOWED_ROLES = frozenset({
        TextRole.BODY.value,
    })

    effective_role = role

    if effective_role in _NOISE_REMOVAL_ALLOWED_ROLES:
        text_before_noise = text

        for noise in _NOISE_SUBSTRINGS:
            if noise.lower() in text.lower():
                pattern = re.compile(
                    rf"(^|\s){re.escape(noise)}(\s|$)",
                    re.IGNORECASE
                )
                text = pattern.sub(" ", text)

        for noise_pattern in _NOISE_PATTERNS:
            text = noise_pattern.sub("", text)

        if text != text_before_noise:
            modifications.append("noise_removed")
            if trace_id:
                logger.debug(
                    "[%s] Noise removed: '%s' -> '%s'",
                    trace_id,
                    text_before_noise[:50],
                    text[:50]
                )

    # =========================================================================
    # STEP 7.5: Final spacing safety cleanup (non-structural)
    # =========================================================================
    text = _WHITESPACE_PATTERN.sub(" ", text)

    # =========================================================================
    # STEP 7.6: Pre-punctuation whitespace collapse (terminal safety net)
    #
    # Steps 5-7 can re-introduce space-before-punctuation artifacts that
    # Step 0 (_normalize_punctuation_spacing) already cleaned:
    #   - Step 6: bracket removal "word []." → "word ."
    #   - Step 7: noise removal "word noise ." → "word ."
    # This is the final-position application — no subsequent step mutates spacing.
    # =========================================================================
    text = _PRE_PUNCT_WHITESPACE_PATTERN.sub(r'\1', text)
    text = text.lstrip('.,;:!?()[]')

    # =========================================================================
    # STEP 7.9: Heading prosody enhancement
    #
    # Colons after headings create a natural prosodic pause before section content.
    # Most TTS engines interpret colons as a structural cue with a slight pitch
    # reset and timing gap—ideal for audible section boundaries.
    #
    # This is a narration-only transformation; semantic.json is unaffected.
    #
    # ORDERING: Must run BEFORE Step 8 (terminal punctuation enforcement) so
    # the colon replaces any existing terminal punct cleanly.
    # =========================================================================
    effective_role = (
        sentence_metadata.get("role")
        if isinstance(sentence_metadata, dict) and sentence_metadata.get("role")
        else role
    )

    if effective_role in ("heading", "subheading", "title"):
        text = text.rstrip(".!?:") + ":"
        modifications.append("heading_prosody_colon")

    # =========================================================================
    # STEP 8: Terminal punctuation enforcement (decoder runaway mitigation)
    # =========================================================================
    if add_terminal_punct and text:
        needs_punct = True

        # Already ends with terminal punctuation
        if text[-1] in ".!?:;":
            needs_punct = False
        # Ends with punctuation before closing quote: ." ?) !'
        elif len(text) >= 2 and text[-2] in ".!?" and text[-1] in "\"')]}\\":
            needs_punct = False
        # Ends with ellipsis
        elif text.endswith("..."):
            needs_punct = False
        # Ends with closing quote that follows terminal punct further back
        elif len(text) >= 3 and text[-3] in ".!?" and text[-2:] in [
            '."', ".'", ".)", '?"', "?'", "?)", '!"', "!'", "!)"
        ]:
            needs_punct = False

        # =====================================================================
        # PATCH 8A: Don't add period to short fragments ending with incomplete words
        # These are lead-in fragments that need merging, not sentence termination.
        # Threshold (8 words) mirrors _semantic_completeness_action logic.
        # =====================================================================
        if needs_punct:
            words = text.split()
            if words and len(words) <= 8:
                last_word = words[-1].lower().rstrip(",:;\"'")
                if last_word in _STITCH_INCOMPLETE_ENDINGS:
                    needs_punct = False
                    modifications.append("incomplete_ending_preserved")

        if needs_punct:
            text = text + "."
            modifications.append("terminal_punct_added")

    # =========================================================================
    # POPULATE CHANGE TRACKER (if provided)
    # =========================================================================
    if change_tracker is not None:
        sanitized_length = len(text)
        change_tracker.update({
            "original_length": original_length,
            "sanitized_length": sanitized_length,
            "length_delta": sanitized_length - original_length,
            "modifications": modifications,
        })

    # =========================================================================
    # LOGGING
    # =========================================================================
    if trace_id and modifications:
        logger.debug(
            "[%s] TTS sanitized: len %d→%d (delta=%+d), mods=%s, text='%s...'",
            trace_id,
            original_length,
            len(text),
            len(text) - original_length,
            modifications,
            text[:50]
        )

    # -------------------------------------------------------------------------
    # v1.5.1: Decoder-Safe Prosodic Clause Segmentation (stable trigger + defaults)
    # -------------------------------------------------------------------------
    # IMPORTANT:
    # - Trigger is computed from ORIGINAL metrics so sanitation cannot suppress it.
    # - change_tracker always receives explicit defaults to prevent "missing key" ambiguity.
    # -------------------------------------------------------------------------

    original_comma_count = (sentence_metadata.get("_original_comma_count")
                            if isinstance(sentence_metadata, dict)
                               and sentence_metadata.get("_original_comma_count") is not None
                            else None)
    original_len = (sentence_metadata.get("_original_length")
                    if isinstance(sentence_metadata, dict)
                       and sentence_metadata.get("_original_length") is not None
                    else None)

    # If not pre-populated by caller, compute from the pre-sanitized input `text` argument.
    # At this point in the function, `text` has been mutated; we preserved original_length earlier.
    # Use original_length from the start of this function as the authoritative original length.
    if original_len is None:
        original_len = original_length
    if original_comma_count is None:
        # We do NOT have the pre-sanitized string anymore; best available is count from the
        # original input at entry-time. Caller may optionally populate _original_comma_count.
        # Fall back to current count if not provided.
        original_comma_count = text.count(",")

    # Always set defaults in tracker (prevents silent missing keys downstream)
    if change_tracker is not None:
        change_tracker.setdefault("needs_clause_splitting", False)
        change_tracker.setdefault("prosodic_clauses", None)

    decoder_risk = (
            (original_comma_count >= 4 and original_len > 180)
            or
            (original_comma_count >= 2 and original_len > 120)
            or
            (original_comma_count >= _TTS_PROSODIC_SPLIT_COMMA_THRESHOLD
             and original_len > _TTS_PROSODIC_SPLIT_CHAR_THRESHOLD)
            or
            (original_comma_count == 0
             and original_len > _TTS_PROACTIVE_SPLIT_CHARS)  # ← new monolith gate
    )

    if decoder_risk:
        # ────────────────────────────────────────────────────────────
        # MONOLITH PATH (zero-comma long sentences)
        # Do NOT call prosodic splitter — flag Stage 3 proactive split.
        # ────────────────────────────────────────────────────────────
        if original_comma_count == 0:
            if change_tracker is not None:
                change_tracker["needs_clause_splitting"] = True
            modifications.append("monolith_decoder_split")
            if trace_id:
                logger.debug(
                    "[%s] Monolith decoder-risk flagged (orig_len=%d)",
                    trace_id,
                    original_len,
                )
        else:
            prosodic_clauses = _split_into_prosodic_clauses(
                text=text,
                sentence_metadata=sentence_metadata,
                trace_id=trace_id,
            )
            if len(prosodic_clauses) > 1:
                if change_tracker is not None:
                    change_tracker["prosodic_clauses"] = prosodic_clauses
                    change_tracker["needs_clause_splitting"] = True
                modifications.append("prosodic_clause_segmentation")
                if trace_id:
                    logger.debug(
                        "[%s] Prosodic clause metadata: %d clauses (orig_len=%d, orig_commas=%d, final_len=%d)",
                        trace_id,
                        len(prosodic_clauses),
                        original_len,
                        original_comma_count,
                        len(text),
                    )
    return text


def _is_tts_viable_span(span: Dict, trace_id: str = None) -> bool:
    """
    Determine if a span should participate in TTS ordering and reconstruction.

    SINGLE AUTHORITY for span-level TTS viability.
    Mirrors ALL exclusion gates in _reconstruct_text_for_segmentation.

    Returns:
        True if span should participate in TTS stream, False otherwise.
    """
    if not isinstance(span, dict):
        return False

    role = span.get("role", "")

    # ─────────────────────────────────────────────────────────────────
    # GATE 1: Explicit rescue override (highest priority)
    # ─────────────────────────────────────────────────────────────────
    if span.get("_tts_rescued", False):
        return True

    # ─────────────────────────────────────────────────────────────────
    # GATE 2: Never-override roles (page numbers, etc.)
    # ─────────────────────────────────────────────────────────────────
    if role in _SEMANTIC_NEVER_OVERRIDE_ROLES:
        return False

    # ─────────────────────────────────────────────────────────────────
    # GATE 3: Semantic disposition (matches reconstruction contract)
    # ─────────────────────────────────────────────────────────────────
    disp = span.get("_semantic_disposition")
    if disp == _SEM_DISP_EXCLUDED:
        return False
    if disp == _SEM_DISP_INTERRUPTION:
        return False

    # ─────────────────────────────────────────────────────────────────
    # GATE 4.5 (RONC v2.2): Margin Truth Enforcement (FAIL-CLOSED)
    #
    # Lossless window tags non-viable margin/sidebar spans with
    # _stage1_nonviable_hint=True. These spans must NOT participate in
    # ordering/reconstruction, even if semantic disposition is included.
    #
    # Additionally, any explicit margin layout_stream must fail closed.
    # This preserves structural flows and prevents reconstruction contamination.
    # ─────────────────────────────────────────────────────────────────
    if span.get("_stage1_nonviable_hint", False):
        return False

    stream = span.get("layout_stream", "")
    if isinstance(stream, str) and stream.startswith("margin"):
        # FIX v6.5b: Exempt structural punctuation from margin rejection
        # Punctuation like commas between italicized terms gets margin_right
        # layout_stream but is essential for sentence structure.
        text = (span.get("cleaned_text") or "").strip()
        if len(text) <= 3 and text and not any(c.isalnum() for c in text):
            pass  # Allow punctuation to continue to later gates
        else:
            return False

    # ─────────────────────────────────────────────────────────────────
    # GATE 4: TTS exclusion (with chain protection mirror)
    # Must match reconstruction's chain protection logic exactly
    # ─────────────────────────────────────────────────────────────────
    if span.get("_tts_excluded", False):
        contract = span.get("_ronc_contract") or {}
        authority = contract.get("authority")
        is_chain_protected = (
                span.get("_a2_qualified", False) or
                authority in (_RONC_V2_AUTHORITY_STRONG, _RONC_V2_AUTHORITY_WEAK) or
                span.get("_ronc_atomic_unit_id") is not None
        )
        if not is_chain_protected:
            return False
        # Chain-protected spans fall through (eligible)

    # ─────────────────────────────────────────────────────────────────
    # GATE 4.9: Atomic Isolation Guard (P15 - Structure-Based Filter)
    # ─────────────────────────────────────────────────────────────────
    # Enforces the "Semantic Inclusion Contract".
    # A body span must have SEMANTIC BACKING to be included.
    #
    # Backing = (Authority is strong/weak) OR (Belongs to Atomic Unit).
    #
    # If a span is an ORPHAN (No Authority + No Unit), it is noise
    # (labels, artifacts), unless explicitly rescued.
    # ─────────────────────────────────────────────────────────────────
    if role == TextRole.BODY.value and not span.get("_tts_rescued", False):
        contract = span.get("_ronc_contract") or {}
        authority = contract.get("authority")
        unit_id = span.get("_ronc_atomic_unit_id")

        # Check for Semantic Backing (Is it connected to the story?)
        has_backing = (
                authority in ("strong", "weak") or
                unit_id is not None
        )

        if not has_backing:
            # It is a Semantic & Structural Orphan.
            text = (span.get("cleaned_text") or "").strip()

            # FIX v6.5: Preserve structural punctuation (commas, periods, etc.)
            # Punctuation-only spans have no semantic backing by design,
            # but they're essential for sentence structure.
            # Pattern: 1-3 chars, all punctuation, no alphanumeric content.
            if len(text) <= 3 and text and not any(c.isalnum() for c in text):
                return True  # Punctuation is structurally necessary

            # DEFENSE-IN-DEPTH:
            # If a massive block of text (>60 chars) is isolated, RONC likely failed.
            # We keep it for manual review. We only auto-kill isolated spans
            # that fit the profile of artifacts/labels (<60 chars).
            # This safely kills "Force (Golgi tendon organ)" (26 chars) and "Gamma bias".
            if len(text) < 60:
                # FIX v6.5: Whitelist grammatical punctuation.
                if len(text) <= 3 and text and not any(c.isalnum() for c in text):
                    return True

                return False
    # ─────────────────────────────────────────────────────────────────
    # GATE 5: Non-viable roles (sidebar, figure, table, etc.)
    # ─────────────────────────────────────────────────────────────────
    if role in _TTS_NON_VIABLE_ROLES:
        # P12 FIX: Stream Authority bypass
        # If span is physically in body_col stream with substantial text,
        # stream authority trumps geometric role classification.
        # Prevents valid body text from being rejected due to figure overlap.
        text = (span.get("cleaned_text") or "").strip()
        stream = str(span.get("layout_stream") or "")

        if len(text) > 25 and stream.startswith("body_col"):
            return True  # Stream authority trumps role classification

        return False

    # ─────────────────────────────────────────────────────────────────────
    # GATE 6: Positive inclusion (TTS-orderable roles)
    # FAIL-CLOSED: Unknown roles excluded from ordering with warning
    # ─────────────────────────────────────────────────────────────────────
    if role in _TTS_ORDERABLE_ROLES:
        return True

    # Unknown/empty role - FAIL-CLOSED for ordering safety
    if trace_id:
        logger.warning(
            "[%s] _is_span_tts_viable: Unknown role excluded from ordering. "
            "role=%r text=%r",
            trace_id, role, (span.get("cleaned_text") or "")[:30]
        )
    return False


# ─────────────────────────────────────────────────────────────
# FINAL TTS VIABILITY — SENTENCE LEVEL
# ─────────────────────────────────────────────────────────────

def _is_tts_viable(
        text: str,
        role: str = None,
        *,
        is_continuation: bool = False,
        trace_id: str = None,
) -> Tuple[bool, str]:
    """
    Final sentence-level TTS viability gate.
    Operates on reconstructed sentence text.
    Returns: (is_viable: bool, reason: str)
    """

    if not text or not isinstance(text, str):
        return False, "empty_text"

    text = text.strip()
    if not text:
        return False, "whitespace_only"

    if len(text) < 2:
        return False, "too_short"

    # Role-based hard exclusions (mirrors span gate, but sentence-safe)
    if role and role in _TTS_NON_VIABLE_ROLES:
        return False, f"non_viable_role:{role}"

    # Prevent obvious garbage from being spoken
    if text.count("(") != text.count(")"):
        return False, "unbalanced_parentheses"

    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars < max(3, int(len(text) * 0.15)):
        return False, "low_alpha_density"

    return True, "ok"


def _passes_narration_gate(sp: Dict) -> bool:
    """
    Final narration admission gate.

    Runs AFTER semantic resolution.
    This is the ONLY place spans are removed from TTS reconstruction.
    """
    if not isinstance(sp, dict):
        return False

    # (A) Soft exclusion
    if sp.get("_tts_excluded", False):
        return False

    # (B) Role viability
    # RONC/rescue override: Respect RONC must_include for substantial body-column text.
    # Guards prevent rescuing garbage (short fragments, figure captions, non-body streams).
    if sp.get("role") in _TTS_NON_VIABLE_ROLES:
        is_rescued = sp.get("_tts_rescued", False)

        # Check RONC protection with guards
        is_ronc_protected = False
        contract = sp.get("_ronc_contract") or {}
        protection = contract.get("protection") or {}

        # ─────────────────────────────────────────────────────────────────
        # P11 FIX: Stream Authority (Resolves Type A - Split Brain)
        # If span is physically in body_col stream with substantial text,
        # protect it regardless of role. This prevents inside_figure rejection
        # for valid body text that was geometrically misclassified.
        # ─────────────────────────────────────────────────────────────────
        text = (sp.get("cleaned_text") or "").strip()
        stream = sp.get("layout_stream", "")

        if len(text) > 25 and str(stream).startswith("body_col"):
            # P11: Only rescue geometric misclassifications (e.g. inside_figure),
            # not authoritative role assignments such as FOOTNOTE.
            if sp.get("role") in ("inside_figure", "diagram_label"):
                is_ronc_protected = True

        # Fallback: Explicit RONC must_include protection
        elif protection.get("must_include", False):
            is_ronc_protected = True

        if not is_rescued and not is_ronc_protected:
            return False

    # ─────────────────────────────────────────────────────────────────
    # (B2) Caption admissibility guard (v10.0)
    #
    # Captions are narratable ONLY when associated with a figure or
    # table (figure_index >= 0). Running headers misclassified as
    # captions have figure_index == -1 (no figure nearby) and must
    # be blocked. _associate_captions_to_figures runs in Stage 1,
    # so figure_index is available on all caption spans by this point.
    #
    # This catches "E. Elahi et al." at bbox_y 30.9 — classified as
    # caption, no figure association, should never narrate.
    # ─────────────────────────────────────────────────────────────────
    if sp.get("role") == TextRole.CAPTION.value:
        figure_idx = sp.get("figure_index")
        if figure_idx is None or figure_idx < 0:
            return False

    # ─────────────────────────────────────────────────────────────────
    # Spans carrying any structural exclusion reason (document_metadata,
    # publisher_metadata, header_band, footer_band, header_artifact)
    # must not narrate. flags are set by _soft_classify_spans
    # and routed through the exclusion candidate mechanism.
    # catches structural spans even if upstream
    # flag translation (structural authority gate) did not fully
    # exclude them. Respects _exclusion_protected (lowercase
    # continuations, drop caps).
    # ─────────────────────────────────────────────────────────────────
    candidate_reasons = set(sp.get("_exclusion_candidate_reasons") or [])
    if candidate_reasons & _STRUCTURAL_EXCLUSION_REASONS:
        if not sp.get("_exclusion_protected", False):
            return False

    text = (sp.get("cleaned_text") or "").strip()
    if not text:
        return False

    # (C) Short single-word diagram labels
    if " " not in text and len(text) < _TTS_HARD_GATE_SHORT_WORD_MAX_CHARS:
        if sp.get("role") not in _TTS_HARD_GATE_SHORT_WORD_ALLOWED_ROLES:
            return False

    # (D) Promo / meta content
    text_lower = text.lower()
    for kw in _TTS_HARD_GATE_PROMO_KEYWORDS:
        if kw in text_lower:
            return False

    # (E) Margin stream belt-and-suspenders
    stream = sp.get("layout_stream", "")
    if isinstance(stream, str) and stream.startswith(_TTS_HARD_GATE_MARGIN_STREAM_PREFIX):
        return False

    return True



def _finalize_chunk(
        all_chunks: List[Dict],
        sentences: List[Dict],
        avg_chars_sec: float,
        trace_id: str = None
) -> None:
    """
    Finalize a chunk with per-sentence timing calculation.

    Each sentence gets duration proportional to its character count,
    enabling precise highlighting synchronization.

    Args:
        all_chunks: List of all chunks (mutated in place).
        sentences: List of sentence dictionaries for this chunk.
        avg_chars_sec: Average characters per second for timing.
        trace_id: Optional trace ID for logging.

    Mutates:
        Appends new chunk to all_chunks if viable.
    """
    if not sentences:
        return

    # =========================================================================
    # FILTER NON-VIABLE SENTENCES
    # =========================================================================
    # IMPORTANT (INVARIANT):
    # Phase 8 must never override semantic authority.
    # Once _has_semantic_authority is set upstream, it is final.
    # If semantic resolution (Phase 5) included a sentence, it MUST survive.
    # =========================================================================
    viable_sentences: List[Dict] = []
    rejection_counts: Dict[str, int] = {}

    for s in sentences:
        tts_text = s.get("tts_text", "")
        sent_role = s.get("role", TextRole.BODY.value)
        is_continuation = s.get("is_continuation", False)

        # PATCH P0: Respect semantic authority from upstream resolution
        has_authority = s.get("_has_semantic_authority", False)

        is_viable, reason = _is_tts_viable(
            tts_text,
            role=sent_role,
            is_continuation=is_continuation,
            trace_id=trace_id
        )

        # PATCH P0: Authority overrides viability rejection
        if is_viable or has_authority:
            viable_sentences.append(s)
        else:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

            if trace_id:
                logger.debug(
                    "[%s] Sentence rejected: reason=%s, text='%s...'",
                    trace_id, reason, tts_text[:40]
                )

    # =========================================================================
    # DEBUG: Authority violation assertion
    # =========================================================================
    if trace_id and __debug__:
        viable_ids = {id(v) for v in viable_sentences}
        for s in sentences:
            if s.get("_has_semantic_authority") and id(s) not in viable_ids:
                logger.error(
                    "[%s] AUTHORITY VIOLATION: authoritative sentence dropped: '%s...'",
                    trace_id, s.get("tts_text", "")[:40]
                )

    if not viable_sentences:
        if trace_id:
            logger.debug(
                "[%s] Chunk discarded: no viable sentences (rejections=%s)",
                trace_id, rejection_counts
            )
        return

    # =========================================================================
    # VALIDATE FULL CHUNK TEXT
    # =========================================================================
    full_text = " ".join(s["tts_text"] for s in viable_sentences)

    is_viable, reason = _is_tts_viable(full_text, trace_id=trace_id)

    # PATCH P0: Chunk-level authority check
    # If ANY sentence in this chunk has semantic authority, do not reject
    has_any_authority = any(
        s.get("_has_semantic_authority", False)
        for s in viable_sentences
    )

    if not is_viable and not has_any_authority:
        if trace_id:
            logger.debug(
                "[%s] Full chunk rejected: reason=%s",
                trace_id, reason
            )
        return

    # =========================================================================
    # TIMING CALCULATION
    # =========================================================================
    chunk_id = len(all_chunks)
    chunk_start_time = all_chunks[-1]["end_time"] if all_chunks else 0.0
    total_chars = len(full_text)
    chunk_duration = total_chars / avg_chars_sec if total_chars > 0 else 0.0

    # Per-sentence timing (proportional to character count)
    sentence_cursor = chunk_start_time

    for sent in viable_sentences:
        sent_text = sent.get("tts_text", "")
        sent_chars = len(sent_text)

        if total_chars > 0:
            sent_duration = chunk_duration * (sent_chars / total_chars)
        else:
            sent_duration = 0.0

        sent["start_time"] = sentence_cursor
        sent["duration_seconds"] = sent_duration
        sent["end_time"] = sentence_cursor + sent_duration
        sentence_cursor = sent["end_time"]

    # =========================================================================
    # V1.7: Reconcile chunk duration with prosodic gap compensation
    #
    # Sentence timing may include micro-gaps for clause-aware TTS.
    # Chunk end_time must match final sentence's end_time for citation
    # lookup to work correctly across the full audio duration.
    # =========================================================================
    actual_chunk_end = sentence_cursor
    actual_chunk_duration = actual_chunk_end - chunk_start_time

    # =========================================================================
    # BUILD CHUNK RECORD
    # =========================================================================
    pages: set = {
        s.get("page_number") for s in viable_sentences
        if s.get("page_number") is not None
    }
    sorted_pages = sorted(pages)
    primary_page = sorted_pages[0] if sorted_pages else None

    all_chunks.append({
        "chunk_id": chunk_id,
        "page": primary_page,
        "pages": sorted_pages,
        "text": full_text,
        "start_time": chunk_start_time,
        "end_time": actual_chunk_end,
        "duration_seconds": actual_chunk_duration,
        "sentences": [
            _prepare_sentence_for_serialization(s)
            for s in viable_sentences
        ],
    })

    if trace_id:
        logger.debug(
            "[%s] Chunk %d finalized: %d sentences, %.2fs duration",
            trace_id, chunk_id, len(viable_sentences), chunk_duration
        )


# ✦                  ✦                  ✦                  ✦
# ✦───────────── 7 Orchestrators ─────────────✦
# ✦                  ✦                  ✦                  ✦

def extract_page(
        doc: "fitz.Document",
        page_num: int,
        trace_id: str = None,
        global_median_line_height: float = None,  # NEW
        global_median_font_size: float = None  # NEW
) -> Dict:
    """
    STAGE 1: Extraction & Structure Analysis.

    HARDENED v1.9.0:
        1. Added TEXT_PRESERVE_LIGATURES and TEXT_PRESERVE_WHITESPACE flags
           to support downstream kerning analysis (Phase 5 preparation).
        2. Implemented Context-Aware Dynamic Sorting (Step 4.5 + Step 5):
           - Table content: tight tolerance (3px) preserves row integrity
           - Prose content: loose tolerance (8 px) catches italic baseline drift
        3. Uses int() floor division for predictable bucket boundaries
           (eliminates Python banker's rounding edge cases).

    Pipeline:
        1. Extract raw spans from PyMuPDF (with ligature/whitespace preservation)
        2. Detect subscripts (baseline-based)
        3. Detect page regions (figures, tables, links)
        4. Detect columns (layout grid)
        4.5. Tag table content (context for dynamic sorting) — NEW
        5. Sort spans (reading order with context-aware tolerance) — MODIFIED
        6. Detect paragraphs (text blocks)
        7. Filter spans (remove artifacts)
        8. Clean spans (normalize text)
        9. Assign roles (classify content)
        10. Associate captions to figures
        11. Assign span indices

    Args:
        global_median_font_size:
        global_median_line_height:
        doc: Open PyMuPDF Document object.
        page_num: Zero-based page index.
        trace_id: Optional trace ID for observability logging.

    Returns:
        Dict containing full structural data (Stage 1 Output).
    """
    page = doc.load_page(page_num)

    # =========================================================================
    # STEP 1: Extract Raw Spans (The "Clay")
    # =========================================================================
    # FIX (Phase 1): Preserve ligatures and whitespace for downstream kerning fix.
    # - TEXT_DEHYPHENATE: Rejoins hyphenated words across lines
    # - TEXT_PRESERVE_LIGATURES: Keeps fi, fl, ff, ffi, ffl as intended glyphs
    # - TEXT_PRESERVE_WHITESPACE: Preserves original spacing for kerning analysis
    # - clip=page.rect: Prevents off-screen artifact contamination
    text_page = page.get_text(
        "rawdict",
        flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE,
        clip=page.rect
    )
    raw_spans = _flatten_to_raw_spans(text_page, page_num)

    # =========================================================================
    # STEP 2: Detect Subscripts (baseline-based)
    # =========================================================================
    _detect_subscripts(raw_spans, trace_id)

    # =========================================================================
    # STEP 3: Detect Page Regions (The "Skeleton")
    # =========================================================================
    regions = _detect_page_regions(
        page,
        raw_spans,
        global_median_font_size=global_median_font_size,
        trace_id=trace_id,
    )

    # Figures are already tuples from _detect_page_regions
    figure_tuples: List[BboxTuple] = regions.get("figures", [])
    regions["figure_tuples"] = figure_tuples

    # =========================================================================
    # STEP 4: Detect Columns (The "Grid")
    # =========================================================================
    _detect_columns(
        raw_spans,
        figure_tuples,
        regions.get("tables", []),
        page.rect.width,
        trace_id
    )

    if trace_id:
        column_indices = {s.get("column_index", 0) for s in raw_spans}
        logger.debug(
            "[%s] Page %d: %d columns detected",
            trace_id, page_num + 1, len(column_indices)
        )

    # =========================================================================
    # STEP 4.5: Tag Table Content (Context for Dynamic Sorting)
    # =========================================================================
    # Pre-tag spans for context-aware sorting. This is an O(N×M) operation
    # where N = number of spans and M = number of tables, but both are small.
    #
    # Table spans get tight tolerance (3px) to preserve row structure.
    # Prose spans get loose tolerance (8px) to handle italic baseline drift.
    #
    # The '_is_table_content' flag is internal (underscore prefix) and will
    # not appear in the final output schema.
    table_rects = [fitz.Rect(t["bbox"]) for t in regions.get("tables", []) if "bbox" in t]
    for span in raw_spans:
        if table_rects:
            span_center_x = (span["bbox"][0] + span["bbox"][2]) / 2
            span_center_y = (span["bbox"][1] + span["bbox"][3]) / 2
            # HARDENED: Check center OR intersection to catch edge-straddling spans
            span_rect = fitz.Rect(span["bbox"])
            span["_is_table_content"] = any(
                r.contains(fitz.Point(span_center_x, span_center_y)) or r.intersects(span_rect)
                for r in table_rects
            )

        else:
            span["_is_table_content"] = False

    # =========================================================================
    # STEP 5: Sort Spans (The "Flow") — Context-Aware Dynamic Tolerance
    # =========================================================================
    # Fuzzy row grouping with context-aware tolerance:
    # - Table content: tight (3px) to preserve row integrity (WD1 safe)
    # - Prose content: loose (8px) to catch italic/superscript drift
    #
    # Uses int() floor division instead of round() for predictable bucket
    # boundaries. Python's round() uses banker's rounding (round half to even)
    # which creates unpredictable edge cases at bucket boundaries.
    #
    # Sort priority:
    # 1. Column index (left-to-right columns)
    # 2. Row bucket (top-to-bottom within tolerance)
    # 3. X position (left-to-right within row)
    raw_spans.sort(
        key=lambda s: (
            s.get("column_index", 0),
            int(
                s.get("baseline_y", s.get("bbox", (0, 0, 0, 0))[1])
                / _get_span_sort_tolerance(s)
            ) * _get_span_sort_tolerance(s),
            s.get("bbox", (0, 0, 0, 0))[0]
        )
    )

    # =========================================================================
    # STEP 6: Detect Paragraphs (The "Blocks") — requires sorted spans
    # =========================================================================
    _detect_paragraphs(
        raw_spans,
        trace_id=trace_id,
        global_median_line_height=global_median_line_height
    )

    # =========================================================================
    # STEP 7: Soft Classification (Schema v2.0 - Lossless)
    # =========================================================================
    import copy

    # P0 AMENDMENT 4: Deep copy raw_spans to preserve immutability
    raw_spans_immutable = raw_spans  # Keep original reference
    classified_spans = [copy.deepcopy(s) for s in raw_spans]

    classified_spans = _soft_classify_spans(
        classified_spans,
        regions,
        page_width=page.rect.width,
        page_height=page.rect.height,
        trace_id=trace_id,
    )

    # =========================================================================
    # STEP 8: Clean Spans (The "Skin") — operates on classified copies
    # =========================================================================
    _clean_spans(classified_spans)

    # =========================================================================
    # STEP 9: Assign Roles (The "Classifier")
    # =========================================================================
    _assign_roles(
        classified_spans,
        regions,
        page_width=page.rect.width,
        page_height=page.rect.height,
        trace_id=trace_id,
        global_baseline_font_size=global_median_font_size
    )

    # Associate Captions
    _associate_captions_to_figures(
        classified_spans,
        figure_tuples,
        page_height=page.rect.height,
        trace_id=trace_id
    )

    # Assign span_index (must be last)
    for idx, span in enumerate(classified_spans):
        span["span_index"] = idx

    # =========================================================================
    # STEP 11.1: Assign canonical span identity (Stage 2 authority anchor)
    # =========================================================================
    # This ID must be assigned in Stage 1 so it:
    #   - persists into *_raw.json
    #   - survives deep copies and windowing
    #   - allows Option A RONC remapping
    # Format: P{page_idx}:{local_idx}
    # page_num here is ZERO-BASED and aligned with Stage 2 enumerate(raw_data_list)

    for local_idx, span in enumerate(classified_spans):
        span["_canonical_span_id"] = f"P{page_num}:{local_idx}"

    # =========================================================================
    # STEP 12: Derive Legacy Content (Backward Compatibility)
    # =========================================================================
    legacy_content = _derive_legacy_content(classified_spans)
    legacy_excluded = [s for s in classified_spans if
                       s.get("_exclusion_candidate") and not s.get("_exclusion_protected")]

    return {
        "metadata": {
            "page_number": page_num + 1,
            "width": page.rect.width,
            "height": page.rect.height,
            "global_median_font_size": global_median_font_size,
        },
        "structure": regions,
        # === SCHEMA v2.0 ===
        "raw_spans": raw_spans_immutable,  # Immutable extraction output
        "classified_spans": classified_spans,  # All spans with soft flags
        # === LEGACY (deprecated) ===
        "content": legacy_content,  # Derived for backward compat
        "excluded": legacy_excluded,  # Derived for backward compat
        "_schema_version": "2.0",
    }


def _passes_inline_continuation_gate(anchor: Dict, target: Dict) -> bool:
    """
    Determines if target span is likely inline continuation of anchor.

    Gates:
      1) Anchor lacks terminal punctuation
      2) Target starts lowercase or with connective
      3) Target doesn't match obvious annotation/caption patterns
    """
    anchor_text = (anchor.get("cleaned_text") or anchor.get("raw_text") or "").strip()
    target_text = (target.get("cleaned_text") or target.get("raw_text") or "").strip()

    if not anchor_text or not target_text:
        return False

    # Gate 1: anchor must NOT end a sentence
    if anchor_text[-1] in ".!?":
        return False

    # Gate 2: target looks like continuation
    first_char = target_text[0]
    first_word = target_text.split()[0].lower() if target_text.split() else ""
    connectives = {
        "and", "or", "but", "in", "to", "for", "the", "a", "an", "of",
        "is", "are", "was", "were", "that", "which", "with", "as", "by",
    }

    if not (first_char.islower() or first_word in connectives):
        return False

    # Gate 3: disqualify obvious caption/annotation patterns
    if target_text.startswith(("Figure ", "Table ", "Note:", "See ")):
        return False

    return True


def _semantic_completeness_action(
        text: str,
        has_authority: bool = False,
        trace_id: str = None
) -> tuple[str, Optional[str]]:
    """
    Decide what to do with a candidate sentence for TTS quality:
    AUTHORITY CONTRACT:
        If has_authority is True, this function MUST return ("keep", None).
        No heuristic rule is permitted to skip or merge authoritative sentences.
      - "keep": sentence is fine
      - "merge_next": likely a lead-in fragment; merge into next sentence
      - "skip": broken clause/label that should not be spoken

    Conservative: prefer merge_next over skip when uncertain.
    """
    if has_authority:
        return "keep", None

    if not text:
        return "skip", "empty"

    text = text.strip()
    if not text:
        return "skip", "empty"

    lower = text.lower()
    words = text.split()
    wc = len(words)

    # -------------------------------------------------------------------------
    # Rule 1: Dangling preposition endings (short sentences only)
    # -------------------------------------------------------------------------
    DANGLING_PREP = ("to", "of", "for", "with", "by", "in", "on", "at", "from")
    if wc <= 8:
        for w in DANGLING_PREP:
            if lower.endswith(f" {w}.") or lower.endswith(f" {w}"):
                return "merge_next", f"dangling_preposition:{w}"

    # -------------------------------------------------------------------------
    # Rule 2: Dangling copula/auxiliary endings (very short only)
    # -------------------------------------------------------------------------
    DANGLING_VERB = (
        "is", "are", "was", "were", "has", "have", "been",
        "called", "named", "termed", "known"
    )
    # Original threshold (1-6 words): all dangling verbs
    if wc <= 6:
        for w in DANGLING_VERB:
            if lower.endswith(f" {w}.") or lower.endswith(f" {w}"):
                return "merge_next", f"dangling_verb:{w}"
    # PATCH 9A: Surgical extension (7-8 words) for empirically proven failures:
    #   "The central region of each intrafusal fiber has." (8 words)
    #   "The much slower, highly affective component is called." (8 words)
    elif wc <= 8:
        for w in ("has", "called"):  # Only proven failure verbs
            if lower.endswith(f" {w}.") or lower.endswith(f" {w}"):
                return "merge_next", f"dangling_verb:{w}"

    # -------------------------------------------------------------------------
    # Rule 3: Lowercase-leading short fragments (mid-sentence spill)
    # -------------------------------------------------------------------------
    if text[0].islower() and wc <= 7:
        return "merge_next", "lowercase_fragment"

    # -------------------------------------------------------------------------
    # Rule 4: Valid short sentences (dialogue, discourse, utterances)
    # -------------------------------------------------------------------------
    if lower in VALID_SHORT_SENTENCES:
        return "keep", None

    # -------------------------------------------------------------------------
    # Rule 5: Very short label-like phrases → merge (not skip)
    # -------------------------------------------------------------------------
    if wc <= 3 and text.endswith("."):
        return "merge_next", "short_label_candidate"

    return "keep", None


def _order_spans_by_semantic_chains(spans: List[Dict], trace_id: str = None) -> List[Dict]:
    """
    Reorder spans to respect semantic chains using weighted contract signals.

    PHASE 2.8 CONTRACT:
        Input spans are assumed to be TTS-viable only.
        Non-TTS roles (sidebar, figure, table) must be filtered upstream.
        This ensures atomic units contain only semantically coherent content.

    Ordering hierarchy:
      1. Atomic unit grouping (spans in same unit stay together)
      2. Chain role (anchor → member → tail)
      3. A2 edge chain position (follow qualified links)
      4. Link confidence (higher confidence = more trusted)
      5. Original position (fallback for ties)

    Does NOT filter or modify spans — only reorders.
    """
    if not spans:
        return spans

    # Build lookup
    spans_by_id = {
        sp.get("_canonical_span_id"): sp
        for sp in spans
        if sp.get("_canonical_span_id")
    }

    # Preserve original indices for fallback ordering
    original_index = {sp.get("_canonical_span_id"): i for i, sp in enumerate(spans)}

    _chain_depth_cache = {}

    def chain_depth(sp: Dict) -> float:
        """Count hops back to chain anchor. Only follows qualified edges for reliability."""
        span_cid = sp.get("_canonical_span_id")
        if span_cid in _chain_depth_cache:
            return _chain_depth_cache[span_cid]

        depth = 0.0
        current = sp
        visited = {span_cid}

        while current:
            prev_id = current.get("_a2_edge_prev_id")
            if not prev_id or prev_id in visited:
                break

            prev_sp = spans_by_id.get(prev_id)
            if not prev_sp:
                break

            # Do not traverse across atomic unit boundaries
            if (
                    sp.get("_ronc_atomic_unit_id") is not None
                    and prev_sp.get("_ronc_atomic_unit_id") != sp.get("_ronc_atomic_unit_id")
            ):
                break

            if current.get("_a2_qualified"):
                depth += 1
            else:
                depth += 0.5

            visited.add(prev_id)
            current = prev_sp

        _chain_depth_cache[span_cid] = depth
        return depth

    def link_confidence(sp: Dict) -> float:
        """Extract link confidence from RONC contract."""
        contract = sp.get("_ronc_contract") or {}
        links = contract.get("links") or {}
        prev_link = links.get("prev") or {}

        confidence = prev_link.get("confidence", 0)
        if prev_link.get("mutual"):
            confidence += 0.1

        return confidence

    def authority_score(sp: Dict) -> int:
        """Authority contributes to ordering stability."""
        contract = sp.get("_ronc_contract") or {}
        auth = contract.get("authority")
        return _RONC_V2_AUTHORITY_NUMERIC.get(auth, 0)

    def sort_key(sp: Dict) -> tuple:
        """Composite sort key using all contract signals."""
        cid = sp.get("_canonical_span_id")
        unit_id = sp.get("_ronc_atomic_unit_id")
        role = sp.get("_ronc_atomic_role")

        # Atomic units may span pages; unit_id dominates ordering.
        # Page number stabilizes fallback ordering when unit is missing.
        # RONC v2.1: Deterministic geometric fallback for unlinked spans
        unit_group = (
            sp.get("page_number", 0),
            sp.get("block_id", 0),
            sp.get("line_index", 0),
            sp.get("span_index_in_line", 0),
        )

        role_priority = _RONC_ROLE_PRIORITY.get(role, 3)
        depth = chain_depth(sp)
        conf_val = link_confidence(sp)
        conf = -conf_val
        auth = -authority_score(sp)

        # Explicit RONC links should dominate authority-only ordering (small stable nudge)
        if conf_val > 0:
            auth -= 1

        orig_pos = original_index.get(cid, float("inf"))

        if trace_id:
            sp["_ordering_debug"] = {
                "unit": unit_id,
                "role": role,
                "depth": depth,
                "conf": conf_val,
                "auth": authority_score(sp),
                "orig": orig_pos,
            }

        return unit_group, role_priority, depth, conf, auth, orig_pos

    ordered = sorted(spans, key=sort_key)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX (UPDATED): Preserve visual order of ALL INCLUDED spans within
    # RONC atomic unit visual ranges.
    #
    # Atomic units define a contiguous *range* in original visual order, not a
    # contiguous *block* in output order. Any INCLUDED span whose original
    # position lies between the first and last member of a unit must not be
    # leapfrogged, regardless of unit membership.
    # ─────────────────────────────────────────────────────────────────────────
    if ordered:
        orig_pos = original_index

        # Collect visual ranges for units with >=2 members
        unit_ranges = {}
        for sp in ordered:
            uid = sp.get("_ronc_atomic_unit_id")
            cid = sp.get("_canonical_span_id")
            if uid is None or cid is None:
                continue
            p = orig_pos.get(cid)
            if p is None:
                continue
            lo, hi = unit_ranges.get(uid, (p, p))
            unit_ranges[uid] = (min(lo, p), max(hi, p))

        visited = set()
        stabilized = []

        # Helper: INCLUDED spans only (excluded spans never reached this function)
        def is_included(span):
            return span.get("_semantic_disposition") != _SEM_DISP_EXCLUDED

        for sp in ordered:
            cid = sp.get("_canonical_span_id")
            if cid in visited:
                continue

            uid = sp.get("_ronc_atomic_unit_id")
            # If not a multi-member unit, emit as-is
            if uid is None or uid not in unit_ranges:
                stabilized.append(sp)
                visited.add(cid)
                continue

            lo, hi = unit_ranges[uid]

            # Emit ALL INCLUDED spans whose original positions fall within [lo..hi],
            # in original visual order, regardless of unit membership.
            for sp2 in sorted(
                    (x for x in ordered if x.get("_canonical_span_id") not in visited),
                    key=lambda x: orig_pos.get(x.get("_canonical_span_id"), float("inf"))
            ):
                p2 = orig_pos.get(sp2.get("_canonical_span_id"))
                if p2 is not None and lo <= p2 <= hi and is_included(sp2):
                    stabilized.append(sp2)
                    visited.add(sp2.get("_canonical_span_id"))

        # Append any remaining spans (outside all unit ranges)
        for sp in ordered:
            cid = sp.get("_canonical_span_id")
            if cid not in visited:
                stabilized.append(sp)
                visited.add(cid)

        ordered = stabilized

    # Log reordering details
    if trace_id:
        reorder_count = 0
        reorder_details = []

        for new_idx, sp in enumerate(ordered):
            cid = sp.get("_canonical_span_id")
            old_idx = original_index.get(cid)
            if old_idx is not None and old_idx != new_idx:
                reorder_count += 1
                reorder_details.append({
                    "cid": cid,
                    "text": (sp.get("cleaned_text") or "")[:20],
                    "old": old_idx,
                    "new": new_idx,
                    "unit": sp.get("_ronc_atomic_unit_id"),
                    "role": sp.get("_ronc_atomic_role"),
                })

        if reorder_count > 0:
            logger.info(
                "[%s] Chain Ordering: %d/%d spans reordered",
                trace_id, reorder_count, len(spans)
            )
            for detail in reorder_details[:10]:
                logger.debug(
                    "[%s]   %s '%s' moved %d→%d (unit=%s, role=%s)",
                    trace_id,
                    detail["cid"],
                    detail["text"],
                    detail["old"],
                    detail["new"],
                    detail["unit"],
                    detail["role"],
                )
            if len(reorder_details) > 10:
                logger.debug("[%s]   ... and %d more moves", trace_id, len(reorder_details) - 10)

    return ordered


def compile_tts_ready_content(
        raw_data_list: List[Dict],
        trace_id: str = None
) -> Dict:
    """
    STAGE 2: Transform extracted page data into TTS-ready chunks.

    MODIFIED v3.2: Added cross-page truncation healing after stitching.
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # PIPELINE INVARIANT (v9.5):
    #
    # 1. Canonical IDs (_canonical_span_id) are IMMUTABLE after Stage 1 assignment.
    #    They must NEVER be mutated, synced, or reassigned.
    #
    # 2. All span lookups into page_span_cache MUST use CID-based search, NEVER
    #    position-based indexing. Lists may be sorted/filtered at any phase.
    #
    # Violation of these invariants causes systematic text drift (+1 shift pattern).
    # ═══════════════════════════════════════════════════════════════════════════

    if trace_id:
        logger.info("[%s] compile_tts_ready_content: Starting Stage 2", trace_id)

    # =========================================================================
    # STEP 1: Compute Global Header/Footer Bands
    # =========================================================================
    global_bands = _compute_global_header_footer_bands(raw_data_list)

    doc_band_summary = _summarize_document_bands(
        raw_data_list,
        global_bands["header_bands"],
        global_bands["footer_bands"]
    )

    document_headers = doc_band_summary["headers"]
    document_footers = doc_band_summary["footers"]

    if trace_id:
        logger.debug(
            "[%s] Global bands: %d headers, %d footers",
            trace_id, len(document_headers), len(document_footers)
        )

    # -------------------------------------------------------------------------
    # PHASE 3.0 (DOCUMENT-SCOPE): Content-Flow Outlier Refinement
    #
    # IMPORTANT:
    # - This refinement is document-context dependent (font stats, block gaps).
    # - It MUST run once on the full span set, not per-page.
    # - It mutates span roles only (no exclusions, no deletions).
    # -------------------------------------------------------------------------

    page_span_cache: Dict[int, List[Dict]] = {}
    page_metadata_cache: Dict[int, Dict] = {}

    # Collect processed window spans for semantic artifact
    # These contain P6 same-line promotions that don't exist in page_span_cache originals
    processed_spans_collector: Dict[str, Dict] = {}

    for page_idx, page_data in enumerate(raw_data_list):
        page_num = page_data.get("metadata", {}).get("page_number")

        header_sample_texts = [h.get("sample_text", "") for h in document_headers]
        footer_sample_texts = [f.get("sample_text", "") for f in document_footers]
        page_height = page_data.get("metadata", {}).get("height", _FILTER_DEFAULT_PAGE_HEIGHT)

        schema_version = page_data.get("_schema_version", "1.0")

        if schema_version == "2.0":
            classified = page_data.get("classified_spans", []) or []

            _apply_global_band_signals(
                classified,
                global_bands["header_bands"],
                global_bands["footer_bands"],
                header_samples=header_sample_texts,
                footer_samples=footer_sample_texts,
                trace_id=trace_id,
            )

            # For debugging parity with legacy fields
            page_data["excluded_by_global_bands"] = [
                s for s in classified
                if s.get("_exclusion_candidate")
                   and not s.get("_exclusion_protected")
                   and any(
                    (r.startswith("global_header_match_") or r.startswith("global_footer_match_"))
                    for r in (s.get("_exclusion_candidate_reasons") or [])
                )
            ]

            spans_for_processing = classified

        else:
            filtered_spans, excluded_spans = _filter_spans_by_global_bands(
                page_data,
                global_bands["header_bands"],
                global_bands["footer_bands"],
                header_samples=header_sample_texts,
                footer_samples=footer_sample_texts,
                page_height=page_height,
                trace_id=trace_id
            )
            page_data["excluded_by_global_bands"] = excluded_spans
            spans_for_processing = filtered_spans

        # =====================================================================
        # PHASE 3.0: Content-Flow Outlier Refinement
        # (legacy mutates roles; in Schema v2.0 we can later replace with signals)
        # =====================================================================

        bbox_invalid_spans = [s for s in spans_for_processing if not s.get("bbox_is_valid", True)]
        page_data["bbox_invalid_spans"] = bbox_invalid_spans

        spans_for_text = list(spans_for_processing)

        if not spans_for_text:
            continue
        # Assigned ONCE per page, BEFORE any windowing or semantic mutation
        for local_idx, sp in enumerate(spans_for_text):
            # Canonical ID MUST already exist from Stage 1 (IMMUTABLE)
            if "_canonical_span_id" not in sp:
                if trace_id:
                    logger.error(
                        "[%s] Stage 2 contract violation: missing _canonical_span_id (page_idx=%d local_idx=%d)",
                        trace_id, page_idx, local_idx
                    )
                raise RuntimeError(
                    "Stage 2 contract violation: missing _canonical_span_id (must be assigned in Stage 1)"
                )

        # cache
        page_span_cache[page_idx] = spans_for_text
        page_metadata_cache[page_idx] = page_data.get("metadata", {})

        # PHASE 1.5: Continuity-Aware Role Resolution (Stream-First Foundation)
        _apply_continuity_role_resolution(spans_for_text, trace_id=trace_id)

        # PHASE 0: Annotate non-viable spans (NON-AUTHORITATIVE)
        # IMPORTANT:
        # - Phase 0 must NOT make final _tts_excluded decisions.
        # - Authoritative inclusion/exclusion is handled later by
        #   _translate_exclusion_flags (post-semantic).
        # - Here we ONLY record role-based candidates for audit/debug.

        excluded_by_role = []

        for span in spans_for_text:
            role = span.get("role", TextRole.BODY.value)

            if role in _TTS_NON_VIABLE_ROLES:
                excluded_by_role.append(span)

        # DO NOT overwrite page_span_cache[page_idx] here.
        # page_span_cache[page_idx] was already populated earlier with the lossless list.
        # page_span_cache[page_idx] = spans_for_text  # <-- intentionally NOT done

        # Keep metadata cache as before
        page_metadata_cache[page_idx] = {
            "page_num": page_num,
            "continuity": page_data.get("continuity", {}),
        }

    all_sentences: List[Dict] = []
    global_sentence_index = 0
    # ─────────────────────────────────────────────────────────────
    # RONC v2.1: Global atomic unit ID namespace
    # Each semantic window creates local unit IDs starting at 0.
    # Track a monotonically increasing offset so IDs remain unique
    # across all windows.
    # ─────────────────────────────────────────────────────────────
    global_unit_id_offset = 0

    for page_idx in range(len(raw_data_list)):
        spans_for_text = page_span_cache.get(page_idx, [])

        if not spans_for_text:
            continue

        metadata = page_metadata_cache.get(page_idx, {})
        page_num = metadata.get("page_num")
        continuity = metadata.get("continuity", {})

        # =====================================================================
        # PHASE 2.0: Build sliding window from cache
        # Window includes prev_tail + current + next_head
        # All spans are deep copies with _page_local_idx tags
        # =====================================================================
        window_spans, page_span_range = _build_sliding_window_spans(
            page_span_cache, page_idx, trace_id
        )

        if not window_spans:
            continue

        _resolve_semantic_continuity(window_spans, page_span_range, trace_id=trace_id)

        # ═══════════════════════════════════════════════════════════════════════
        # RONC v2.0 — Contract-Maker Architecture
        # Must run AFTER semantic continuity resolution, BEFORE exclusion translation
        #
        # Pipeline: Boundary Profiling → Candidate Pools → Affinity Scoring →
        #           Link Reconciliation → Protection Assignment → Legacy Derivation
        #
        # Output: Every span receives _ronc_contract + legacy fields
        # ═══════════════════════════════════════════════════════════════════════
        _ronc_audit = _build_ronc_contract_v2(
            window_spans,
            trace_id=trace_id,
            unit_id_offset=global_unit_id_offset,
        )

        # Increment offset by the number of units created in this window
        global_unit_id_offset += _ronc_audit.get("phase_6", {}).get("units_created", 0)

        # ---------------------------------------------------------------------
        # RONC PERSISTENCE: propagate contract and legacy fields to canonical spans
        # ---------------------------------------------------------------------
        for sp in window_spans:
            # Only current-page spans participate in canonical authority
            if sp.get("_window_position") != "current":
                continue

            source_page_idx = sp.get("_source_page_idx")
            if source_page_idx is None:
                continue

            # Resolve canonical span via CID
            cid = sp.get("_canonical_span_id")
            if not cid:
                continue

            # ─────────────────────────────────────────────────────────────────
            # FIX v9.5: CID-based canonical lookup
            # INVARIANT: Never trust list position after any sort/filter.
            # Position-based access causes +1 text shift when spans are filtered.
            # ─────────────────────────────────────────────────────────────────
            canonical_list = page_span_cache.get(source_page_idx)
            if not canonical_list:
                continue

            canonical_sp = next(
                (s for s in canonical_list if s.get("_canonical_span_id") == cid),
                None
            )
            if canonical_sp is None:
                if trace_id:
                    logger.debug(
                        "[%s] RONC persistence: CID %s not found in cache page %d",
                        trace_id, cid, source_page_idx
                    )
                continue

            # ═══════════════════════════════════════════════════════════════
            # RONC v2.0: Persist full contract (authoritative)
            # ═══════════════════════════════════════════════════════════════
            contract = sp.get("_ronc_contract")
            if contract:
                canonical_sp["_ronc_contract"] = contract

            # ═══════════════════════════════════════════════════════════════
            # Legacy fields (derived from contract, for backward compatibility)
            # ═══════════════════════════════════════════════════════════════
            canonical_sp["_ronc_atomic_unit_id"] = sp.get("_ronc_atomic_unit_id")
            canonical_sp["_ronc_atomic_role"] = sp.get("_ronc_atomic_role")
            canonical_sp["_ronc_break_after"] = sp.get("_ronc_break_after")
            canonical_sp["_ronc_rescue_applied"] = sp.get("_ronc_rescue_applied")

            # A2 continuity signals (preserved)
            canonical_sp["a2_continues_from_previous"] = sp.get("a2_continues_from_previous")
            canonical_sp["a2_continues_to_next"] = sp.get("a2_continues_to_next")

        # =====================================================================
        # PHASE 2.0.5–2.0.8: ADJUDICATION CHECKPOINT (Lead Option C)
        #
        # Contract guarantees at this point:
        #   1. Every span has _tts_excluded explicitly set (True or False)
        #   2. Every span has _semantic_disposition set (or loudly defaulted)
        #   3. Line-coherent stream inheritance applied (semantic-gated)
        #   4. Exclusion decisions synced back to page cache (audit/debug)
        #
        # Reconstruction input (window_spans_for_text) is CLEAN.
        # =====================================================================

        # 1) Translate soft exclusion + semantic signals into authoritative flags
        _translate_exclusion_flags(window_spans, trace_id=trace_id)

        # 2) Apply line-coherent stream inheritance (semantic-gated)
        #    NOTE: This MUST respect _tts_excluded
        _apply_line_coherent_streams(window_spans, trace_id=trace_id)

        # 3) Sync adjudicated span state back to page cache (audit + convergence)
        _sync_window_spans_back_to_cache(
            window_spans,
            page_span_cache,
            trace_id=trace_id
        )

        # =====================================================================
        # PHASE 1.5b: Post-Semantic Role Reconciliation
        #
        # Re-run continuity resolution now that _semantic_disposition is available.
        # This catches inside_figure spans that are semantically included but
        # lacked adjacency evidence in the first pass (Stage 2.4).
        #
        # WHY HERE:
        # - _semantic_disposition is now populated (from _resolve_semantic_continuity)
        # - Cache spans have been synced with semantic signals
        # - Role changes here will propagate to semantic artifact via collector sync
        #
        # INVARIANT: This is idempotent — already-reconciled spans are skipped.
        # =====================================================================
        cache_spans_for_reconciliation = page_span_cache.get(page_idx, [])
        if cache_spans_for_reconciliation:
            reconciled_count = _apply_continuity_role_resolution(
                cache_spans_for_reconciliation,
                trace_id=trace_id
            )
            if reconciled_count > 0:
                if trace_id:
                    logger.info(
                        "[%s] Phase 1.5b: Post-semantic reconciliation overrode %d spans on page %d",
                        trace_id, reconciled_count, page_idx + 1
                    )
                # Sync role changes back to window_spans for collector consistency
                # Build CID lookup for cache spans with role changes
                cache_role_by_cid = {
                    s.get("_canonical_span_id"): s.get("role")
                    for s in cache_spans_for_reconciliation
                    if s.get("role") == TextRole.BODY.value
                }
                for sp in window_spans:
                    if sp.get("_window_position") == "current":
                        cid = sp.get("_canonical_span_id")
                        if cid and cid in cache_role_by_cid:
                            sp["role"] = cache_role_by_cid[cid]
                            sp["_continuity_override"] = True

        # Collect current-page spans with P6 modifications for semantic artifact
        # window_spans contains deep copies with P6 same-line promotions applied
        # NOTE: Phase 1.5b role reconciliation is now reflected in these copies
        for sp in window_spans:
            if sp.get("_window_position") == "current":
                cid = sp.get("_canonical_span_id")
                if cid:
                    processed_spans_collector[cid] = sp

        # 4) NARRATION ADMISSION GATE (v4.0)
        # This is the ONLY place spans are removed from TTS reconstruction.
        # All semantic decisions (RONC, A2, rescue) have already completed.
        # Cache integrity, ordering, and indices are preserved upstream.
        # See _passes_narration_gate() for gate conditions.
        admitted_spans = [sp for sp in window_spans if _passes_narration_gate(sp)]

        # 5) CHAIN-AWARE ORDERING
        # Ordering for reconstruction is owned by _reconstruct_text_for_segmentation.
        # Do NOT pre-order here to avoid double-ordering tie-breaker drift.
        window_spans_for_text = admitted_spans

        if trace_id:
            excluded_by_gate = len(window_spans) - len(window_spans_for_text)
            if excluded_by_gate > 0:
                logger.debug(
                    "[%s] Narration Gate: %d/%d spans admitted (%d filtered)",
                    trace_id, len(window_spans_for_text), len(window_spans), excluded_by_gate
                )

        # Reconstruct text from window (cross-page aware)
        window_text, span_map, char_to_span = _reconstruct_text_for_segmentation(
            window_spans_for_text, trace_id=trace_id
        )

        # ─────────────────────────────────────────────────────────────────
        # PRE-SEGMENTATION: Orphaned bracket neutralization
        # Citation exclusion can leave unmatched [ in the text stream.
        # pysbd suppresses sentence breaks inside bracket pairs, causing
        # pathologically long sentences. Neutralize unmatched [ with space
        # (1:1 replacement preserves char_to_span alignment).
        # ─────────────────────────────────────────────────────────────────
        _open_brackets = window_text.count("[")
        _close_brackets = window_text.count("]")
        if _open_brackets > _close_brackets:
            _chars = list(window_text)
            _stack = []
            for _i, _ch in enumerate(_chars):
                if _ch == '[':
                    _stack.append(_i)
                elif _ch == ']':
                    if _stack:
                        _stack.pop()
            for _pos in _stack:
                _chars[_pos] = ' '
            window_text = ''.join(_chars)
            if trace_id:
                logger.debug(
                    "[%s] Bracket neutralization: %d orphaned [ replaced "
                    "(open=%d, close=%d)",
                    trace_id, len(_stack), _open_brackets, _close_brackets
                )

        # Segment on full window text
        window_sentences = _segment_sentences(
            window_text,
            char_to_span,
            window_spans_for_text,
            trace_id=trace_id
        )

        # Filter to sentences starting in current page, remap indices
        page_sentences = _filter_sentences_to_page(
            window_sentences,
            window_spans_for_text,
            page_span_range,
            page_idx,
            page_num,
            trace_id
        )

        # Tag sentences with continuity info (page_number already set by filter)
        for sent in page_sentences:
            sent["in_continued_table"] = continuity.get("has_continued_table", False)
            sent["in_continued_figure"] = continuity.get("has_continued_figure", False)

            # Inject role from dominant span
            # =================================================================
            # PHASE 2.8: Role Injection from Dominant Span
            #
            # CRITICAL FIX: Use spans_for_text (the list that produced indices),
            # not filtered_spans (which includes non-viable roles like sidebar).
            #
            # Proof chain:
            #   1. _segment_sentences receives spans_for_text as all_spans param
            #   2. char_to_span maps chars to indices in that list
            #   3. span_start_index = sorted_spans[0] from char_to_span
            #   4. Therefore span_start_index indexes spans_for_text
            #
            # Bug: filtered_spans[idx] != spans_for_text[idx] when non-viable
            #      roles are excluded. 51.6% of sentences had wrong role.
            # =================================================================
            start_idx = sent.get("span_start_index", 0)

            # Defense-in-depth: Validate index type and bounds
            if not isinstance(start_idx, int) or start_idx < 0:
                if trace_id:
                    logger.warning(
                        "[%s] Invalid span_start_index: %r, defaulting to BODY",
                        trace_id, start_idx
                    )
                sent["role"] = TextRole.BODY.value
            elif start_idx >= len(window_spans_for_text):
                if trace_id:
                    logger.warning(
                        "[%s] span_start_index %d >= len(window_spans_for_text) %d, defaulting to BODY",
                        trace_id, start_idx, len(window_spans_for_text)
                    )
                sent["role"] = TextRole.BODY.value
            else:
                role_from_span = window_spans_for_text[start_idx].get("role", TextRole.BODY.value)

                # =============================================================
                # PHASE 2.8.1: Caption Continuity Guard
                #
                # Caption is unique: stitch-skipped but TTS-viable.
                # Captions containing body-like prose should not break stitch.
                #
                # Rationale:
                #   - _STITCH_SKIP_ROLES includes caption (breaks stitch)
                #   - _TTS_NON_VIABLE_ROLES excludes caption (passes TTS)
                #   - Body prose mislabeled as caption would fragment output
                #
                # Heuristic: Body-like text is substantial (>40 chars) and
                # has sentence termination punctuation.
                # =============================================================
                if role_from_span == TextRole.CAPTION.value:
                    sent_text = sent.get("text", "")
                    has_body_characteristics = (
                            len(sent_text) > 40 and
                            sent_text.rstrip()[-1:] in ".!?"
                    )
                    if has_body_characteristics:
                        role_from_span = TextRole.BODY.value
                sent["role"] = role_from_span

            # =================================================================
            # PHASE 2.8.2: SEMANTIC AUTHORITY PROPAGATION (CRITICAL)
            #
            # If a sentence is derived from any span that semantic resolution
            # explicitly INCLUDED (or inside_figure rescued), downstream heuristics
            # are NOT allowed to skip it.
            #
            # WHY HERE:
            # - Here we still have window_spans_for_text in scope.
            # - Later (chunking loop), we only have sentences, not their source spans.
            # =================================================================
            start_idx = sent.get("span_start_index", 0)
            end_idx = sent.get("span_end_index", start_idx)
            if not isinstance(end_idx, int) or end_idx < start_idx:
                end_idx = start_idx

            # Respect pre-existing authority (RONC or semantic resolution)
            has_semantic_authority = sent.get("_has_semantic_authority", False)

            for sp_idx in range(start_idx, min(end_idx + 1, len(window_spans_for_text))):
                sp = window_spans_for_text[sp_idx]
                if (
                        sp.get("_inside_figure_rescued")
                        or (
                        sp.get("_semantic_disposition") == _SEM_DISP_INCLUDED
                        and sp.get("_semantic_confidence", 0.0) >= 0.6
                )
                ):
                    has_semantic_authority = True
                    break

            sent["_has_semantic_authority"] = has_semantic_authority

            # --------------------------------------------------------------
            # Semantic projection (read-only carry-forward)
            # --------------------------------------------------------------
            start_idx = sent.get("span_start_index", 0)
            end_idx = sent.get("span_end_index", start_idx)

            source_spans = window_spans_for_text[
                           start_idx: min(end_idx + 1, len(window_spans_for_text))
                           ]

            _inject_sentence_semantic_projection(sent, source_spans)

            all_sentences.append(sent)

    # =========================================================================
    # PHASE 2 CLOSURE: Same-Page Truncation Healing
    # Repairs sentences cut at auxiliary verbs, articles, prepositions
    # Must run BEFORE cross-page stitching for cleaner stitch input
    # =========================================================================
    all_sentences = _heal_truncated_sentences(all_sentences, trace_id)

    # Cross-page stitching
    all_sentences = _stitch_cross_page_sentences(all_sentences, trace_id)

    # =========================================================================
    # NEW v3.2: Cross-Page Truncation Healing
    # Repairs sentences cut at auxiliary verbs, articles, prepositions
    # Must run AFTER stitching, BEFORE chunking
    # =========================================================================
    all_sentences = _heal_cross_page_truncations(all_sentences, trace_id)

    if trace_id:
        logger.debug("[%s] Total sentences after healing: %d", trace_id, len(all_sentences))

    # =========================================================================
    # STEP 3: Chunk Globally (Paragraph + Role + Length + Viability)
    # =========================================================================
    all_chunks: List[Dict] = []
    current_chunk_sentences: List[Dict] = []
    current_chunk_char_count = 0
    last_para_idx: Optional[int] = None
    skipped_count = 0
    prev_sentence_text = ""

    pending_merge_prefix = ""  # v1.3.2: Accumulator for merge_next fragments

    for i, sent in enumerate(all_sentences):
        sent_role = sent.get("role", TextRole.BODY.value)

        # ==============================================================
        # V1.7: Prosodic Clause Metadata Hook (Stage 2 → Stage 3)
        # ==============================================================
        # Persist original metrics so _sanitize_for_tts can make stable decoder-risk decisions.
        raw_sentence_text = sent.get("text", "") or ""
        sent["_original_length"] = len(raw_sentence_text)
        sent["_original_comma_count"] = raw_sentence_text.count(",")

        tts_change_tracker = {}

        tts_text = _sanitize_for_tts(
            raw_sentence_text,
            role=sent_role,
            trace_id=trace_id,
            sentence_metadata=sent,
            change_tracker=tts_change_tracker,
        )

        # Persist tracker for sentence finalization
        sent["_tts_change_tracker"] = tts_change_tracker

        # =====================================================================
        # PATCH 8B: Strip trailing punctuation from fragment before merging
        # Safety net for merge_next fragments that may have acquired punctuation.
        # =====================================================================
        if pending_merge_prefix:
            clean_prefix = pending_merge_prefix.rstrip().rstrip(".!?")
            tts_text = clean_prefix + " " + tts_text.lstrip()
            pending_merge_prefix = ""  # Clear after use

        # Viability Check logic...
        is_dialogue = False
        if prev_sentence_text:
            last_char = prev_sentence_text.rstrip()[-1] if prev_sentence_text.rstrip() else ""
            is_dialogue = last_char in _ORCHESTRATOR_DIALOGUE_TRIGGERS

        is_short_response = (
                len(tts_text) < _DIALOGUE_SHORT_RESPONSE_CHARS and
                tts_text.strip()[-1:] in ".!?"
        )
        force_keep = is_dialogue and is_short_response

        # =====================================================================
        # v1.3.2: Semantic completeness action (pre-viability)
        # Returns: "keep" | "merge_next" | "skip"
        # =====================================================================
        # SEMANTIC AUTHORITY INVARIANT:
        has_authority = sent.get("_has_semantic_authority", False)

        # FIX v5.6: Structural Role Guard (Chunking Layer)
        # Headings are structural boundaries — they must NEVER be merged or
        # skipped by the semantic completeness heuristic, regardless of length.
        # This ensures "Cutaneous receptors." (20 chars) is preserved as its
        # own sentence rather than being merged into the following body text.
        is_structural_role = sent_role in ("heading", "subheading", "title")

        if is_structural_role:
            action, sem_reason = "keep", "structural_role_protected"
        else:
            # v1.3.2: Semantic completeness action (pre-viability)
            action, sem_reason = _semantic_completeness_action(
                tts_text,
                has_authority=has_authority,
                trace_id=trace_id
            )

        if action == "merge_next" and not force_keep and not has_authority:
            if trace_id:
                logger.debug(
                    "[%s] Fragment marked for merge-next: %r (reason: %s)",
                    trace_id, tts_text[:80], sem_reason
                )
            # Accumulate into pending prefix for next valid sentence
            pending_merge_prefix = (
                pending_merge_prefix.rstrip() + " " + tts_text.lstrip()
                if pending_merge_prefix else tts_text
            )
            skipped_count += 1
            continue
        if action == "skip" and not force_keep and not has_authority:
            if trace_id:
                logger.debug(
                    "[%s] Skipping incomplete sentence: %r (reason: %s)",
                    trace_id, tts_text[:80], sem_reason
                )
            skipped_count += 1
            pending_merge_prefix = ""  # Don't carry garbage forward
            prev_sentence_text = tts_text
            continue
        # Existing viability gate remains authoritative
        viable, reject_reason = _is_tts_viable(tts_text, sent_role, trace_id=trace_id)
        if not viable and not force_keep and not has_authority:
            skipped_count += 1
            prev_sentence_text = tts_text
            continue

        prev_sentence_text = tts_text
        sent["tts_text"] = tts_text
        sent["global_index"] = global_sentence_index
        sent["sentence_id"] = f"S{global_sentence_index:05d}"
        global_sentence_index += 1

        para_idx = sent.get("paragraph_index")
        projected_len = current_chunk_char_count + len(tts_text)
        prev_sent = current_chunk_sentences[-1] if current_chunk_sentences else None

        # --- CHUNKING RULES ---

        # Rule 1: Paragraph boundary
        if (
                last_para_idx is not None and
                para_idx is not None and
                para_idx != last_para_idx and
                current_chunk_sentences
        ):
            if current_chunk_char_count >= _CHUNK_MIN_CHARS:
                _finalize_chunk(all_chunks, current_chunk_sentences, _AVG_CHARS_PER_SEC, trace_id)
                current_chunk_sentences = []
                current_chunk_char_count = 0

        # Rule 2: Role boundary
        elif current_chunk_sentences and _is_role_boundary(prev_sent, sent):
            if current_chunk_char_count >= _CHUNK_MIN_CHARS:
                _finalize_chunk(all_chunks, current_chunk_sentences, _AVG_CHARS_PER_SEC, trace_id)
                current_chunk_sentences = []
                current_chunk_char_count = 0

        # Rule 3: Length limit (With Hard Ceiling Fix)
        elif projected_len > _CHUNK_MAX_CHARS and current_chunk_sentences:
            prev_for_continuation = all_sentences[i - 1] if i > 0 else None
            is_continuation = (
                    prev_for_continuation is not None and
                    _is_cross_page_continuation(prev_for_continuation, sent)
            )

            # Force break if absolute max exceeded, even if it's a continuation
            if projected_len > _CHUNK_ABSOLUTE_MAX_CHARS or not is_continuation:
                _finalize_chunk(all_chunks, current_chunk_sentences, _AVG_CHARS_PER_SEC, trace_id)
                current_chunk_sentences = []
                current_chunk_char_count = 0

        # Add sentence to current chunk
        current_chunk_sentences.append(sent)
        current_chunk_char_count += len(tts_text)
        last_para_idx = para_idx

    # =========================================================================
    # FLUSH REMAINDER
    # =========================================================================
    if current_chunk_sentences:
        if current_chunk_char_count >= _CHUNK_MIN_CHARS:
            # Normal finalization for sufficiently large remainder
            _finalize_chunk(all_chunks, current_chunk_sentences, _AVG_CHARS_PER_SEC, trace_id)
        elif all_chunks:
            # Merge tiny final chunk (must preserve serialized sentence contract)
            prev_chunk = all_chunks[-1]

            # Finalize the remainder sentences into a temporary chunk so sentences are serialized
            tmp_chunks: List[Dict] = []
            _finalize_chunk(tmp_chunks, current_chunk_sentences, _AVG_CHARS_PER_SEC, trace_id)

            if tmp_chunks:
                prev_chunk["sentences"].extend(tmp_chunks[0]["sentences"])

                # Recompute merged chunk text from serialized sentences
                merged_text = " ".join(
                    s.get("text", "")
                    for s in prev_chunk["sentences"]
                )
                prev_chunk["text"] = merged_text

                # Recompute pages/page from merged serialized sentences
                pages = {s.get("page_number") for s in prev_chunk["sentences"] if
                         s.get("page_number") is not None}
                sorted_pages = sorted(pages)
                prev_chunk["pages"] = sorted_pages
                prev_chunk["page"] = sorted_pages[0] if sorted_pages else prev_chunk.get("page")

                # Recompute timing from last sentence (authoritative after serialization)
                if prev_chunk["sentences"]:
                    last = prev_chunk["sentences"][-1]
                    prev_chunk["end_time"] = last.get("end_time", prev_chunk.get("end_time"))
                    prev_chunk["duration_seconds"] = float(prev_chunk["end_time"]) - float(
                        prev_chunk["start_time"])
        else:
            # No existing chunks to merge into — finalize as standalone
            _finalize_chunk(all_chunks, current_chunk_sentences, _AVG_CHARS_PER_SEC, trace_id)

    total_duration = all_chunks[-1]["end_time"] if all_chunks else 0.0

    # =========================================================================
    # PAGE TURN MARKERS (Sentence-Preserving, Word-Accurate)
    #
    # Purpose:
    #   Derive precise page-turn points for sentences that span multiple pages,
    #   without introducing word-level highlighting or altering sentence timing.
    #
    # Invariants:
    #   - Sentence-level timing (start_time / end_time) remains authoritative.
    #   - Sentences are NOT split or mutated.
    #   - Uses only existing provenance (_source_span_ids → processed_spans_collector).
    #
    # Output:
    #   page_turn_markers[] with interpolated turn_time inside sentence window.
    # =========================================================================

    page_turn_markers = []

    # Iterate finalized chunks and sentences (post-healing, post-chunking)
    for chunk in all_chunks:
        chunk_id = chunk.get("chunk_id")
        sentences = chunk.get("sentences", []) or []

        for sent in sentences:
            source_ids = (
                    sent.get("_source_span_ids")
                    or sent.get("source_cids")
                    or []
            )
            if not source_ids or not isinstance(source_ids, list):
                continue

            # Collect spans participating in this sentence with page numbers
            spans = []
            for cid in source_ids:
                sp = processed_spans_collector.get(cid)
                if not isinstance(sp, dict):
                    continue
                page_num = sp.get("page_number")
                if page_num is None:
                    continue
                spans.append(sp)

            # Group spans by page
            page_to_spans = {}
            for sp in spans:
                p = sp.get("page_number")
                page_to_spans.setdefault(p, []).append(sp)

            # Only interested in cross-page sentences
            if len(page_to_spans) <= 1:
                continue

            # Sentence timing authority
            sent_start = sent.get("start_time")
            sent_end = sent.get("end_time")
            sent_text = sent.get("text", "") or ""

            if (
                    not isinstance(sent_start, (int, float))
                    or not isinstance(sent_end, (int, float))
                    or sent_end <= sent_start
                    or not sent_text
            ):
                continue

            sentence_duration = float(sent_end) - float(sent_start)
            sentence_len = len(sent_text)

            if sentence_len <= 0:
                continue

            # Order pages as they appear in the sentence (by span order)
            # NOTE: _source_span_ids order already reflects reconstruction order
            ordered_pages = []
            for sp in spans:
                p = sp.get("page_number")
                if p not in ordered_pages:
                    ordered_pages.append(p)

            # For each page boundary inside this sentence (exclude last page)
            for i in range(len(ordered_pages) - 1):
                page_current = ordered_pages[i]
                page_next = ordered_pages[i + 1]

                # Identify spans belonging to the current page, in sentence order
                page_spans = [sp for sp in spans if sp.get("page_number") == page_current]
                if not page_spans:
                    continue

                # Compute total span character length for this sentence (SPAN SPACE)
                total_span_chars = 0
                for sp_all in spans:
                    txt_all = sp_all.get("cleaned_text", "") or ""
                    total_span_chars += len(txt_all)

                if total_span_chars <= 0:
                    continue  # defensive: cannot compute ratio

                # Compute character boundary within span space (current page)
                boundary_chars = 0
                for sp in page_spans:
                    txt = sp.get("cleaned_text", "") or ""
                    boundary_chars += len(txt)

                # Clamp boundary to total span chars
                boundary_chars = min(max(boundary_chars, 0), total_span_chars)

                # Proportional interpolation inside sentence timing (SPAN SPACE)
                ratio = boundary_chars / float(total_span_chars)
                ratio = min(max(ratio, 0.0), 1.0)

                turn_time = sent_start + ratio * sentence_duration

                # ---------------------------------------------------------
                # DUPLICATE PAGE-TURN GUARD
                #
                # Rare edge case: a sentence's spans may revisit a page
                # (e.g., P2 → P3 → P2 → P3). We must not emit multiple
                # turn markers for the same (page, sentence) pair.
                # ---------------------------------------------------------
                existing_keys = {
                    (m.get("page"), m.get("sentence_global_index"))
                    for m in page_turn_markers
                }
                marker_key = (page_next, sent.get("global_index"))
                if marker_key in existing_keys:
                    continue

                page_turn_markers.append({
                    "page": page_next,
                    "chunk_id": chunk_id,
                    "sentence_global_index": sent.get("global_index"),
                    "turn_time": round(float(turn_time), 3),
                })

    if trace_id:
        logger.info(
            "[%s] Stage 2 complete: %d chunks, %d sentences, %d skipped, %.1fs duration",
            trace_id, len(all_chunks), global_sentence_index, skipped_count, total_duration
        )

    # ═══════════════════════════════════════════════════════════════════════
    # SEMANTIC EXPORT (Belt-and-Suspenders)
    # Dict mutations already propagate via reference chain, but this ensures
    # the list pointer in raw_data_list matches page_span_cache exactly.
    # Only assigns if list identity differs (avoids unnecessary reassignment).
    # ═══════════════════════════════════════════════════════════════════════
    for page_idx, enriched_spans in page_span_cache.items():
        if page_idx < len(raw_data_list):
            current_list = raw_data_list[page_idx].get("classified_spans")
            if current_list is not enriched_spans:
                raw_data_list[page_idx]["classified_spans"] = enriched_spans

    return {
        "processing": {
            "total_chunks": len(all_chunks),
            "total_sentences": global_sentence_index,
            "skipped_sentences": skipped_count,
            "total_estimated_duration_seconds": total_duration,
        },
        "document_headers": document_headers,
        "document_footers": document_footers,
        "chunks": all_chunks,
        "processed_spans": processed_spans_collector,
        "page_turn_markers": page_turn_markers,
    }