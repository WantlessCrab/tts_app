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
import unicodedata
import logging
import fitz  # PyMuPDF
import pysbd
import ftfy
import math
import sys
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, Field, model_validator
from collections import Counter
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

    # Structural/Layout Roles
    HEADER_ARTIFACT = "header_artifact"
    EMPTY = "empty"
    INSIDE_FIGURE = "inside_figure"
    FIGURE_LABEL = "figure_label"
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
        default=14.0,
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
        default=3,
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
_LAYOUT_DEFAULT_PAGE_WIDTH: float = 612.0

# --- Column Index Reservations (Margin Isolation) ---
_COLUMN_INDEX_LEFT_MARGIN: int = -1  # Left sidebar/margin content
_COLUMN_INDEX_RIGHT_MARGIN: int = 100  # Right sidebar/margin content

# --- Configuration Constants ---
ENABLE_DIAGRAM_LABEL_FILTER = True

# --- Ingestion Constants ---

# PyMuPDF block types
_PYMUPDF_TEXT_BLOCK_TYPE: int = 0

# Group 0 Constants — ADD
VALID_SHORT_WORDS: frozenset = frozenset({
    # 1-letter words
    'a', 'i',
    # 2-letter words (common)
    'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is',
    'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we',
    # 3-letter words (very common)
    'and', 'are', 'but', 'can', 'did', 'for', 'get', 'got', 'had', 'has',
    'her', 'him', 'his', 'how', 'its', 'let', 'may', 'new', 'nor', 'not',
    'now', 'old', 'one', 'our', 'out', 'own', 'per', 'put', 'run', 'say',
    'see', 'set', 'she', 'the', 'too', 'try', 'two', 'use', 'was', 'way',
    'who', 'why', 'yet', 'you',
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

# --- Filter Threshold Constants ---

_FILTER_DEFAULT_PAGE_HEIGHT: float = 792.0  # US Letter
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
TTS_SUBSTITUTIONS: Dict[str, str] = {
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
    "‘": "‘",
    "’": "’",
    """: '"',
    """: '"',
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

SUBSCRIPT_DIGITS: Dict[str, str] = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")",
    "ₐ": "a", "ₑ": "e", "ₒ": "o", "ₓ": "x", "ₔ": "schwa",
    "ₕ": "h", "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n",
    "ₚ": "p", "ₛ": "s", "ₜ": "t",
}

SUPERSCRIPT_DIGITS: Dict[str, str] = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")",
    "ⁿ": "n", "ⁱ": "i",
}


# --- TTS Sanitization Thresholds ---

_TTS_ACRONYM_LENGTH_THRESHOLD: int = 4
_TTS_LONG_TEXT_THRESHOLD: int = 20

# --- TTS Viability: Role Gate ---

_TTS_NON_VIABLE_ROLES: frozenset[str] = frozenset({
    TextRole.SIDEBAR.value,
    TextRole.FOOTNOTE.value,
    TextRole.INSIDE_FIGURE.value,
    TextRole.FIGURE_LABEL.value,
    TextRole.TABLE_CELL.value,
    TextRole.CODE.value,
    TextRole.HYPERLINK.value,
    TextRole.EMPTY.value,
    TextRole.HEADER_ARTIFACT.value,
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
_CHUNK_MAX_CHARS: int = 600
_CHUNK_ABSOLUTE_MAX_CHARS: int = 650  # Hard ceiling, never exceed even for continuations
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

# Quote normalization mapping
_CLEAN_QUOTE_MAP: dict[str, str] = {
    "\u201c": '"', "\u201d": '"',  # Curly double quotes
    "\u2018": "'", "\u2019": "'",  # Curly single quotes
    "\u00ab": '"', "\u00bb": '"',  # Guillemets
    "\u201e": '"', "\u201f": '"',  # German/Eastern European quotes
    "\u2039": "'", "\u203a": "'",  # Single guillemets
    "\u201a": "'", "\u201b": "'",  # Low single quotes
    "\u301d": '"', "\u301e": '"',  # CJK quotes
    "\u300c": '"', "\u300d": '"',  # Japanese corner brackets
    "`": "'",  # Backtick to apostrophe
}

# Dash normalization mapping
_CLEAN_DASH_MAP: dict[str, str] = {
    "\u2010": "-",  # Hyphen
    "\u2011": "-",  # Non-breaking hyphen
    "\u2012": "-",  # Figure dash
    "\u2013": "-",  # En dash
    "\u2014": ", ",  # Em dash (preserve for pauses)
    "\u2015": ", ",  # Horizontal bar
    "\u2212": "-",  # Minus sign
    "\ufe58": ", ",  # Small em dash
    "\ufe63": "-",  # Small hyphen-minus
    "\uff0d": "-",  # Fullwidth hyphen-minus
}

FORCED_LABEL_TERMS = frozenset({
    'force', 'length', 'velocity', 'feedback', 'error',
    'signal', 'control', 'gain', 'loop', 'input', 'output',
    'driving', 'afferents', 'efferents',
})

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
_NOISE_SUBSTRINGS: Tuple[str, ...] = (
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
_GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT = 0.15  # Widen search to top/bottom 15%
_GLOBAL_BAND_MERGE_TOLERANCE = 5
_GLOBAL_BAND_MIN_PAGE_FRACTION: float = 0.25
_GLOBAL_BAND_ROUNDING_PRECISION: int = -1  # round(y, -1) → nearest 10
_GLOBAL_TEXT_MIN_PAGE_FRACTION: float = 0.5  # Text must appear on 50%+ of pages (NEW)

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

# Layout thresholds (mirrored from PageLayoutConfig defaults for module use)
_LAYOUT_HEADER_THRESHOLD_Y: float = 0.20
_LAYOUT_FOOTER_THRESHOLD_Y: float = 0.80

# --- Paragraph Detection ---

_PARA_MAX_LINE_GAP: float = 50.0  # Maximum reasonable line gap
_PARA_DEFAULT_LINE_HEIGHT: float = 12.0
_PARA_GAP_MULTIPLIER: float = 1.8  # Gap > this * line_height = new para

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
})

DIALOGUE_TRIGGER_ENDINGS: frozenset[str] = frozenset({"?", ":"})

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
_LABEL_TECHNICAL_TERMS: frozenset = frozenset({
    # Control systems terminology
    'signal', 'control', 'feedback', 'force', 'length', 'velocity',
    'error', 'driving', 'input', 'output', 'gain', 'loop',
    # Neuroscience terminology
    'afferents', 'efferents', 'neuron', 'neurons', 'receptor',
    'receptors', 'muscle', 'spindle', 'tendon', 'organ', 'interneurons',
    # Greek letter designators (common in scientific diagrams)
    'primary', 'secondary', 'alpha', 'beta', 'gamma', 'delta',
})

_LABEL_PROSE_INDICATORS: frozenset = frozenset({
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

COMMON_VERBS = frozenset({
    "is", "are", "was", "were", "has", "have", "had",
    "can", "could", "will", "would", "shows", "displays",
    "contains", "indicates", "represents"
})

GREEK_WHITELIST: frozenset = frozenset({
    # Lowercase Greek
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
    'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
    # Uppercase Greek
    'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ',
    'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
    # Common scientific word forms
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda',
    'mu', 'sigma', 'omega', 'pi',
})
# --- Layout Detection ---

_CAPTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
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
    r'[.!?]["\'""'')\]]*$'
)

_CAPTION_START_PATTERN: re.Pattern[str] = re.compile(
    r"^(Figure|Fig\.|Table|Tab\.|Chart|Graph|Diagram|Exhibit)\s*\d+[.:)\- ]"
    r"|^(Figure|Fig\.|Table|Tab\.)\s+[IVXLCDM]+\b"
    r"|^(Fig(ure)?|Table|Panel|Chart|Graph|Plate|Photo|Image)\s*[A-Z0-9]",
    re.IGNORECASE,
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

# --- Cleaning Patterns ---

_EMPTY_BRACKETS_PATTERN: re.Pattern[str] = re.compile(r"[\(\[\{]\s*[\)\]\}]")
_PUNCT_ONLY_BRACKETS_PATTERN: re.Pattern[str] = re.compile(r"[\(\[]\s*[,;:.]\s*[\)\]]")
_WHITESPACE_PATTERN: re.Pattern[str] = re.compile(r"\s+")
_CHEMICAL_SUBSCRIPT_PATTERN: re.Pattern[str] = re.compile(r"([A-Za-z])([₀-₉]+)")

# --- Cleaning Patterns ---

# Safe Control Chars (Keep \n \t \r, remove others)
_CLEAN_CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

# Quote/Dash Normalization Table
# Maps fancy quotes/dashes to standard ASCII for better TTS compatibility.
_CLEAN_TRANS_TABLE = str.maketrans({
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


# ✦───────────── 6 DERIVED CONSTANTS ─────────────✦

_SUBSCRIPT_TABLE: dict[int, str] = str.maketrans(SUBSCRIPT_DIGITS)
_SUPERSCRIPT_TABLE: dict[int, str] = str.maketrans(SUPERSCRIPT_DIGITS)


# ✦───────────── 7 MODULE INITIALIZATION ─────────────✦

# Lazy-loaded sentence segmenter (initialized on first use)
_SENTENCE_SEGMENTER: Optional[pysbd.Segmenter] = None


# ✦────────────────────✦────────────────────✦
#                ✿   METHODS  ✿
# ✦────────────────────✦────────────────────✦

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
            if _rects_intersect(span_rect, fig_rect):
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

            if _rects_intersect(span_rect, table_rect):
                return True

    return False


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
    exclusion_rects = list(figure_tuples)
    for table in tables:
        if table.get("bbox"):
            exclusion_rects.append(_to_bbox_tuple(table["bbox"]))

    # =========================================================================
    # Step 2: Detect Margin Boundaries
    # =========================================================================
    body_left, body_right = _detect_margin_boundaries(spans, page_width, trace_id=trace_id)

    # =========================================================================
    # Step 3: Calculate X-Centroids and Tag Margin Content
    # =========================================================================
    x_centroids = []
    centroid_map = {}
    span_rect_map = {}

    for span in spans:
        span_rect = _span_to_rect(span)
        if span_rect is None:
            span["column_index"] = 0
            continue

        cx = (span_rect[0] + span_rect[2]) / 2
        centroid_map[id(span)] = cx
        span_rect_map[id(span)] = span_rect

        # Tag margin content (used in Step 7 for isolation)
        is_margin = (cx < body_left or cx > body_right)
        span["is_margin_content"] = is_margin

        # Only body content (non-margin, non-excluded) contributes to grid detection
        in_exclusion = any(_rects_intersect(span_rect, ex) for ex in exclusion_rects)
        if not in_exclusion and not is_margin:
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
    # Step 5: Threshold Calculation (Geometry Trust)
    # =========================================================================
    base_threshold = max(page_width * 0.05, 20.0)
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
            if (page_width * 0.25) < midpoint < (page_width * 0.75):
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

    for span in spans:
        cx = centroid_map.get(id(span))
        span_rect = span_rect_map.get(id(span))

        if cx is None:
            span["column_index"] = 0
            continue

        # MARGIN ISOLATION: Reserved column indices for margin content
        if span.get("is_margin_content"):
            if cx < body_left:
                span["column_index"] = _COLUMN_INDEX_LEFT_MARGIN
                margin_left_count += 1
            else:
                span["column_index"] = _COLUMN_INDEX_RIGHT_MARGIN
                margin_right_count += 1
            continue

        # BODY CONTENT: Assign based on detected boundaries
        if num_columns == 1:
            span["column_index"] = 0
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


def _detect_margin_boundaries(
        spans: List[Dict],
        page_width: float,
        page_height: float = None,
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
    # CONFIGURATION: Adaptive bin count
    # =========================================================================
    num_bins = max(
        _MARGIN_MIN_BINS,
        min(_MARGIN_MAX_BINS, int(page_width / _MARGIN_TARGET_BIN_WIDTH))
    )
    bin_width = page_width / num_bins

    # =========================================================================
    # BUILD WEIGHTED HISTOGRAM
    # =========================================================================
    bins: List[float] = [0.0] * num_bins
    span_count = 0

    for span in spans:
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

        center_x = (span_rect[0] + span_rect[2]) / 2

        # Clamp to valid bin range
        bin_idx = max(0, min(int(center_x / bin_width), num_bins - 1))

        # Weight by character count (optional)
        if _MARGIN_WEIGHT_BY_CHARS:
            span_text = span.get("cleaned_text") or span.get("raw_text") or ""
            weight = max(1, len(span_text))
        else:
            weight = 1

        bins[bin_idx] += weight
        span_count += 1

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
    # CALCULATE DENSITY THRESHOLD
    # =========================================================================
    total_weight = sum(bins)
    avg_weight = total_weight / num_bins if num_bins > 0 else 1

    ratio_threshold = max_count * _MARGIN_DENSITY_RATIO
    avg_threshold = avg_weight * _MARGIN_AVG_WEIGHT_MULTIPLIER

    density_threshold = max(
        _MARGIN_MIN_DENSITY_COUNT,
        max(ratio_threshold, avg_threshold)
    )

    # =========================================================================
    # FIND BODY BOUNDARIES (CENTRALITY BIAS)
    # Strategy: Scan OUTWARDS from center to find the main content block.
    # Stops at the first "gap" (low density bin), excluding detached sidebars.
    # =========================================================================

    center_idx = num_bins // 2
    anchor_bin = center_idx

    # 1. FIND ANCHOR: If center is empty (e.g. image), find nearest dense bin
    if bins[center_idx] < density_threshold:
        for offset in range(1, center_idx + 1):
            left, right = center_idx - offset, center_idx + offset
            if 0 <= left < num_bins and bins[left] >= density_threshold:
                anchor_bin = left
                break
            if 0 <= right < num_bins and bins[right] >= density_threshold:
                anchor_bin = right
                break

    # 2. EXPAND LEFT (with gap tolerance)
    body_left_bin = anchor_bin
    gap_count = 0
    max_gap = 1  # Allow 1 empty bin (~30-50px) for internal figures

    for i in range(anchor_bin, -1, -1):
        if bins[i] >= density_threshold:
            gap_count = 0
            body_left_bin = i
        else:
            gap_count += 1
            if gap_count > max_gap:
                break

    # 3. EXPAND RIGHT (with gap tolerance)
    body_right_bin = anchor_bin
    gap_count = 0

    for i in range(anchor_bin, num_bins):
        if bins[i] >= density_threshold:
            gap_count = 0
            body_right_bin = i
        else:
            gap_count += 1
            if gap_count > max_gap:
                break

    # =========================================================================
    # CONVERT TO COORDINATES
    # =========================================================================
    body_left = body_left_bin * bin_width
    body_right = (body_right_bin + 1) * bin_width

    # =========================================================================
    # APPLY MARGIN CONSTRAINTS
    # =========================================================================
    min_margin = page_width * _MARGIN_MIN_RATIO
    max_margin = page_width * _MARGIN_MAX_RATIO

    # Clamp body_left: must be at least min_margin, at most max_margin
    body_left = max(min_margin, min(body_left, max_margin))

    # Clamp body_right: must be at least (page_width - max_margin)
    body_right = min(
        page_width - min_margin,
        max(body_right, page_width - max_margin)
    )

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
        logger.debug(
            "[%s] Margin detection: left=%.0f, right=%.0f, bins=[%s], "
            "threshold=%.1f, spans=%d",
            trace_id, body_left, body_right, bin_viz, density_threshold, span_count
        )

    return body_left, body_right



# ✦                  ✦                  ✦                  ✦
# ✦───────────── 2 Raw Extraction & Classification ─────────────✦
# ✦                  ✦                  ✦                  ✦


# ✦────── a. Ingestion & Cleaning ──────✦


def _flatten_to_raw_spans(
        text_page: Dict,
        page_num: int,
        trace_id: str = None
) -> List[Dict]:
    """
    Flatten PyMuPDF block/line/span hierarchy into a linear list of span dicts.

    This is the ingestion boundary — all bbox values are normalized to tuple
    format (x0, y0, x1, y1) per architectural standard.

    Args:
        text_page: PyMuPDF text dict from page.get_text("dict").
        page_num: Zero-indexed page number.
        trace_id: Optional trace ID for logging.

    Returns:
        List of span dictionaries with standardized structure.

    Note:
        Malformed blocks/lines/spans are logged and skipped (fail-fast but resilient).
    """
    raw_spans: List[Dict] = []
    skipped_count = 0

    blocks = text_page.get("blocks", [])

    for block_idx, block in enumerate(blocks):
        # Skip non-text blocks (images, drawings)
        if block.get("type") != _PYMUPDF_TEXT_BLOCK_TYPE:
            continue

        lines = block.get("lines")
        if not lines:
            continue

        for line_idx, line in enumerate(lines):
            spans = line.get("spans")
            if not spans:
                continue

            for span_idx, span in enumerate(spans):
                try:
                    # Extract and validate bbox
                    bbox_raw = span.get("bbox")
                    if not bbox_raw or len(bbox_raw) < 4:
                        skipped_count += 1
                        continue

                    x0, y0, x1, y1 = bbox_raw[0], bbox_raw[1], bbox_raw[2], bbox_raw[3]

                    # Validate dimensions
                    if x1 <= x0 or y1 <= y0:
                        skipped_count += 1
                        continue

                    # Standardized bbox tuple (architectural contract)
                    bbox: BboxTuple = (float(x0), float(y0), float(x1), float(y1))

                    # Extract baseline/origin
                    origin = span.get("origin", (x0, y1))
                    baseline_y = float(origin[1])
                    line_y_band = round(baseline_y, 1)

                    # Extract text
                    raw_text = span.get("text", "")

                    raw_spans.append({
                        "raw_text": raw_text,
                        "cleaned_text": None,
                        "bbox": bbox,  # TUPLE: (x0, y0, x1, y1)
                        "font_size": float(span.get("size", 0)),
                        "font": span.get("font", ""),
                        "flags": span.get("flags", 0),
                        "color": span.get("color", 0),
                        "origin": origin,
                        "baseline_y": baseline_y,
                        "line_y_band": line_y_band,
                        "page_number": page_num + 1,
                        "column_index": 0,
                        "paragraph_index": 0,
                        "is_paragraph_start": False,
                        "role": TextRole.BODY.value,  # Enum value for compatibility
                        "char_offset": 0,
                        "block_id": None,
                        "is_subscript": False,
                        "figure_index": None,
                    })

                except (KeyError, TypeError, IndexError) as e:
                    skipped_count += 1
                    if trace_id:
                        logger.warning(
                            "[%s] Skipped malformed span at block=%d, line=%d, span=%d: %s",
                            trace_id, block_idx, line_idx, span_idx, e
                        )
                    continue

    if trace_id and skipped_count > 0:
        logger.warning(
            "[%s] Ingestion: skipped %d malformed spans",
            trace_id, skipped_count
        )

    return raw_spans


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
    drop_cap_zone_threshold = effective_page_height * _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT

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

        if not text:
            span["exclusion_reason"] = _REASON_EMPTY
            excluded.append(span)
            continue

        # =====================================================================
        # WHITELIST CHECK — Greek/scientific symbols bypass ALL filters
        # Must be checked BEFORE any exclusion logic
        # =====================================================================
        text_lower = text.lower()
        if text_lower in GREEK_WHITELIST or text in GREEK_WHITELIST:
            valid.append(span)
            continue

        # Standardized Y-band rounding
        y_band = round(span_y / y_band_precision) * y_band_precision

        keep = True
        reason = ""

        # ---------------------------------------------------------------------
        # FILTER 1: Header/Footer Bands (Global Detection)
        # ---------------------------------------------------------------------
        if y_band in header_bands:
            text_stripped = text.strip()
            is_drop_cap = len(
                text_stripped) == 1 and text_stripped.isalpha() and text_stripped.isupper()

            if not is_drop_cap:
                keep = False
                reason = _REASON_HEADER_BAND

        elif y_band in footer_bands:
            text_stripped = text.strip()
            is_drop_cap = len(
                text_stripped) == 1 and text_stripped.isalpha() and text_stripped.isupper()

            if not is_drop_cap:
                keep = False
                reason = _REASON_FOOTER_BAND

        # ---------------------------------------------------------------------
        # FILTER 2: Header Zone Artifacts
        # FIX v3.3: Corrected threshold + unified large-font protection
        # ---------------------------------------------------------------------
        elif span_y < header_artifact_zone_y:
            # Single alpha characters deferred to Filter 5 (Drop Cap / Initial logic)
            text_stripped = text.strip()
            if len(text_stripped) == 1 and text_stripped.isalpha():
                pass  # Defer to Filter 5
            else:
                is_short = len(text) < _FILTER_SHORT_TEXT_LENGTH
                is_uppercase = text.isupper()
                is_fragment = len(text) < _FILTER_FRAGMENT_THRESHOLD

                # Large font = significant content (Drop Caps, titles, callouts)
                is_large_font = span.get("font_size", 0) >= drop_cap_font_threshold

                is_protected = (
                        text in PROTECTED_SHORT_WORDS or
                        _is_protected_acronym(text) or
                        is_large_font
                )

                if not is_protected:
                    if (is_short and is_uppercase) or (is_fragment and not text.isalpha()):
                        has_no_vowels = not any(c in text.upper() for c in _FILTER_VOWELS)
                        is_likely_artifact = has_no_vowels or len(
                            text) <= _FILTER_VERY_SHORT_THRESHOLD

                        if is_likely_artifact or is_uppercase:
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
                is_caption_candidate = False
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
        elif ENABLE_DIAGRAM_LABEL_FILTER and keep and _is_diagram_label(span, figure_tuples):
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
                    # UNCONDITIONAL PROTECTION:
                    # Single uppercase letters (F, W, A.) are rarely noise.
                    if trace_id:
                        logger.debug("[%s] Single uppercase protected: '%s'", trace_id, text_clean)
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
        if keep and len(text) <= _FILTER_SHORT_FRAGMENT_THRESHOLD:
            is_letter = text.isalpha()
            is_protected = text in PROTECTED_SHORT_WORDS

            if not (is_letter or is_protected):
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
    In-place text cleaning for extracted spans.

    UPDATED v1.8.1:
        1. Added noise substring removal (Fix B)
        2. Added noise pattern removal
        3. Preserved Greek letter handling (constants fix)
    """
    if not spans:
        return

    cleaned_count = 0
    erased_count = 0
    noise_removed_count = 0

    for span in spans:
        if not isinstance(span, dict):
            continue

        text = span.get("raw_text", "")
        original_text = text

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
        # STEP 6: Noise Substring Removal (v1.8.1)
        # =====================================================================
        text_before_noise = text

        # Remove known noise substrings (case-insensitive)
        for noise in _NOISE_SUBSTRINGS:
            if noise.lower() in text.lower():
                pattern = re.compile(rf"(^|\s){re.escape(noise)}(\s|$)", re.IGNORECASE)
                text = pattern.sub(" ", text)

        # Remove noise patterns (regex)
        for pattern in _NOISE_PATTERNS:
            text = pattern.sub("", text)

        if text != text_before_noise:
            noise_removed_count += 1
            if trace_id:
                logger.debug(
                    "[%s] Noise removed from span: '%s' -> '%s'",
                    trace_id, text_before_noise[:50], text[:50]
                )

        # =====================================================================
        # STEP 7: Whitespace Normalization (Final)
        # =====================================================================
        text = _WHITESPACE_PATTERN.sub(" ", text).strip()

        # =====================================================================
        # STORE RESULT
        # =====================================================================
        span["cleaned_text"] = text

        if text != original_text:
            cleaned_count += 1
            if not text and original_text.strip():
                erased_count += 1
                if trace_id:
                    logger.warning(
                        "[%s] Span erased completely by cleaner: '%s' -> ''",
                        trace_id, original_text
                    )

    if trace_id and (cleaned_count > 0 or noise_removed_count > 0):
        logger.debug(
            "[%s] Cleaned %d/%d spans (%d erased, %d noise removed)",
            trace_id, cleaned_count, len(spans), erased_count, noise_removed_count
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
        if not any(c in clean.upper() for c in _FILTER_VOWELS):
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
        baselines = [
            s.get("baseline_y")
            for s in band_spans
            if s.get("baseline_y") is not None
        ]

        if not baselines:
            continue

        # Compute median baseline
        sorted_baselines = sorted(baselines)
        median_baseline = sorted_baselines[len(sorted_baselines) // 2]

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

    ...docstring unchanged...
    """
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
                raw_bbox = cell.bbox
                if len(raw_bbox) >= 4:
                    cell_bbox = (
                        float(raw_bbox[0]),
                        float(raw_bbox[1]),
                        float(raw_bbox[2]),
                        float(raw_bbox[3])
                    )
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
                    candidate_figures.append(
                        (expanded.x0, expanded.y0, expanded.x1, expanded.y1)
                    )
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

        for span in raw_spans:
            span_bbox = span.get("bbox")
            if span_bbox is None or len(span_bbox) < 4:
                continue
            span_x0, span_y0, span_x1, span_y1 = span_bbox
            span_cx = (span_x0 + span_x1) / 2
            span_cy = (span_y0 + span_y1) / 2

            if fig_x0 <= span_cx <= fig_x1 and fig_y0 <= span_cy <= fig_y1:
                span_text = span.get("raw_text", "")
                total_text_length += len(span_text)
                word_count = len(span_text.split())
                has_sentence_punct = any(
                    c in span_text for c in _REGION_SENTENCE_PUNCTUATION
                )
                if (word_count >= _REGION_PROSE_MIN_WORD_COUNT or
                        (len(span_text) > _REGION_PROSE_MIN_TEXT_LENGTH and has_sentence_punct)):
                    prose_span_count += 1

        if (prose_span_count >= _REGION_FIGURE_MAX_PROSE_SPANS or
                total_text_length > _REGION_FIGURE_MAX_TEXT_LENGTH):
            if trace_id:
                logger.debug(
                    "[%s] Rejecting false-positive figure at (%.0f,%.0f): "
                    "%d prose spans, %d chars",
                    trace_id, fig_x0, fig_y0, prose_span_count, total_text_length
                )
            # REVERTED v3.7: Do not append.
            # Rejecting this "figure" allows the text inside (likely body prose)
            # to be processed normally. This saves Page 1 content from being
            # swallowed by page borders.
        else:
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

    # Calculate threshold (reserved for future body content detection)
    band_threshold = max(
        _REGION_MIN_BAND_COUNT,
        int(len(raw_spans) * _REGION_BAND_THRESHOLD_RATIO)
    )

    for y, count in y_bands.items():
        norm_y = y / page_height if page_height > 0 else 0

        # FIXED v3.3: Use Global Constants for consistent Zone Definitions (15%)
        # Was: _LAYOUT_HEADER_THRESHOLD_Y (20%) / _LAYOUT_FOOTER_THRESHOLD_Y (80%)
        is_header_zone = norm_y < _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT
        is_footer_zone = norm_y > (1.0 - _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT)

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

    return regions

def _assign_block_ids(spans: List[Dict]) -> None:
    """
    V1.1: Assigns a stable block_id for each contiguous paragraph-flow group.

    For V1.1, block_id == paragraph_index.
    Future V1.2 can refine to merge multi-paragraph blocks (e.g., heading + body).
    """
    for s in spans:
        s["block_id"] = s.get("paragraph_index", 0)


def _detect_paragraphs(spans: List[Dict], trace_id: str = None) -> None:
    """
    Assign paragraph_index to spans.

    CRITICAL: Assumes 'spans' are ALREADY sorted by reading order (Fuzzy Row).
    Does NOT re-sort, to preserve the 'Visual Line' grouping from Stage 1.
    """
    if not spans:
        return

    # =========================================================================
    # GROUP BY COLUMN (Preserving Input Order)
    # =========================================================================
    columns: Dict[int, List[Dict]] = {}

    for span in spans:
        col_idx = span.get("column_index", 0)
        if col_idx not in columns:
            columns[col_idx] = []
        columns[col_idx].append(span)

    # 🛑 DELETED: The internal sort that undid our Fuzzy Sort work.
    # We trust 'extract_page' has already sorted spans by (Column, FuzzyY, X).

    # =========================================================================
    # CALCULATE MEDIAN LINE HEIGHT
    # =========================================================================
    line_heights: List[float] = []

    for col_spans in columns.values():
        for i in range(1, len(col_spans)):
            prev_bbox = col_spans[i - 1].get("bbox")
            curr_bbox = col_spans[i].get("bbox")

            if prev_bbox is None or curr_bbox is None:
                continue

            prev_y = prev_bbox[1]  # y0
            curr_y = curr_bbox[1]  # y0
            gap = curr_y - prev_y

            # Only count gaps that look like line breaks (positive but small)
            # Negative gap = same line (inline fonts). Large gap = paragraph break.
            if 0 < gap < _PARA_MAX_LINE_GAP:
                line_heights.append(gap)

    if line_heights:
        sorted_heights = sorted(line_heights)
        median_line_height = sorted_heights[len(sorted_heights) // 2]
    else:
        median_line_height = _PARA_DEFAULT_LINE_HEIGHT

    # Paragraph gap threshold
    para_gap_threshold = median_line_height * _PARA_GAP_MULTIPLIER

    # =========================================================================
    # ASSIGN PARAGRAPH INDICES
    # =========================================================================
    global_para_idx = 0

    for col_idx in sorted(columns.keys()):
        col_spans = columns[col_idx]
        if not col_spans:
            continue

        # First span starts a new paragraph
        col_spans[0]["paragraph_index"] = global_para_idx

        # Initialize prev_y to the BOTTOM of the first span
        first_bbox = col_spans[0].get("bbox")
        prev_bottom = first_bbox[3] if first_bbox else 0

        for i in range(1, len(col_spans)):
            curr_bbox = col_spans[i].get("bbox")

            if curr_bbox is None:
                col_spans[i]["paragraph_index"] = global_para_idx
                continue

            curr_top = curr_bbox[1]  # y0

            # Gap = Distance from bottom of previous line to top of current line
            # Inline text (Bold/Italic) will have curr_top < prev_bottom (Negative Gap)
            gap = curr_top - prev_bottom

            # New paragraph if gap exceeds threshold
            if gap > para_gap_threshold:
                global_para_idx += 1

            col_spans[i]["paragraph_index"] = global_para_idx

            # Update prev_bottom for next iteration
            prev_bottom = curr_bbox[3]

        # Increment for next column to ensure separation
        global_para_idx += 1

    if trace_id:
        logger.debug(
            "[%s] Paragraph detection: %d paragraphs across %d columns (Height: %.1f, Threshold: %.1f)",
            trace_id, global_para_idx, len(columns), median_line_height, para_gap_threshold
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

    # Check punctuation density
    text = (span.get("cleaned_text") or "").strip()
    if not text:
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
            has_parenthetical = '(' in span_text and ')' in span_text

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
        trace_id: str = None
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

    Mutates:
        Each span receives 'role' key.
    """
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

    # Caption chain tracking
    caption_chain_start_y: Optional[float] = None
    caption_chain_start_font_size: Optional[float] = None
    caption_chain_length: int = 0

    # =========================================================================
    # MAIN CLASSIFICATION LOOP
    # =========================================================================
    for span in spans:
        role = TextRole.BODY.value

        # Extract span properties (tuple bbox)
        bbox = span.get("bbox")
        if bbox is None or len(bbox) < 4:
            span["role"] = TextRole.EMPTY.value
            previous_role = TextRole.EMPTY.value
            continue

        span_x0, span_y0, span_x1, span_y1 = bbox
        span_x = span_x0
        span_y = span_y0

        font_size = span.get("font_size", baseline_font_size)
        font_name = span.get("font", "").lower()
        span_text = (span.get("cleaned_text") or "").strip()

        span_rect = _span_to_rect(span)
        if span_rect is None:
            span["role"] = TextRole.EMPTY.value
            previous_role = TextRole.EMPTY.value
            continue

        word_count = len(span_text.split()) if span_text else 0
        char_count = len(span_text)

        # Skip empty spans
        if not span_text:
            span["role"] = TextRole.EMPTY.value
            previous_role = TextRole.EMPTY.value
            continue

        # =====================================================================
        # PRIORITY 1: Table Cell
        # =====================================================================
        for t_rect in table_rects:
            if _rects_intersect(span_rect, t_rect):
                role = TextRole.TABLE_CELL.value
                break

        # =====================================================================
        # PRIORITY 2: Inside Figure
        # =====================================================================
        if role == TextRole.BODY.value:
            for f_rect in figure_rects:
                if _rects_intersect(span_rect, f_rect):
                    role = TextRole.INSIDE_FIGURE.value
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

                if matches_label or matches_stats:
                    role = TextRole.FIGURE_LABEL.value
                elif is_short and word_count <= _ROLE_FIGURE_LABEL_SHORT_WORD_COUNT:
                    role = TextRole.FIGURE_LABEL.value
                elif is_short and is_small_font:
                    role = TextRole.FIGURE_LABEL.value

        # =====================================================================
        # PRIORITY 4: Caption (with guards)
        # =====================================================================
        if role == TextRole.BODY.value:
            # Explicit caption pattern match — starts new chain
            is_explicit_caption = any(
                pattern.match(span_text)
                for pattern in _COMPILED_CAPTION_PATTERNS
            )

            if is_explicit_caption:
                role = TextRole.CAPTION.value
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

        # =====================================================================
        # PRIORITY 6: Heading
        # =====================================================================
        if role == TextRole.BODY.value:
            is_large = font_size >= baseline_font_size * _ROLE_HEADING_FONT_RATIO
            is_bold = any(
                hint in font_name
                for hint in ("bold", "heavy", "black")
            )
            is_short = (
                    word_count <= _ROLE_HEADING_MAX_WORDS and
                    char_count <= _ROLE_HEADING_MAX_CHARS
            )
            is_title_case = span_text and (span_text[0].isupper() or span_text.isupper())
            no_terminal_punct = span_text and span_text[-1] not in _ROLE_TERMINAL_PUNCTUATION

            if is_short and is_title_case and no_terminal_punct:
                if is_large or is_bold:
                    role = TextRole.HEADING.value

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

            if is_medium and is_short and is_title_case:
                role = TextRole.SUBHEADING.value
            elif is_italic and is_short and is_title_case and word_count >= 2:
                role = TextRole.SUBHEADING.value

        # =====================================================================
        # PRIORITY 8: Footnote
        # =====================================================================
        if role == TextRole.BODY.value:
            is_small = font_size < baseline_font_size * _ROLE_FOOTNOTE_FONT_RATIO
            is_at_bottom = span_y > footer_zone_y

            if is_small and is_at_bottom:
                role = TextRole.FOOTNOTE.value
            elif char_count <= 2 and font_size < baseline_font_size * _ROLE_FOOTNOTE_MARKER_FONT_RATIO:
                if span_text.isdigit() or span_text in _ROLE_FOOTNOTE_MARKER_SYMBOLS:
                    role = TextRole.FOOTNOTE_MARKER.value

        # =====================================================================
        # PRIORITY 9: List Item
        # =====================================================================
        if role == TextRole.BODY.value:
            if _LIST_ITEM_PATTERN.match(span_text):
                role = TextRole.LIST_ITEM.value

        # =====================================================================
        # PRIORITY 10: Code
        # =====================================================================
        if role == TextRole.BODY.value:
            is_monospace = any(hint in font_name for hint in CODE_FONT_HINTS)

            if is_monospace:
                role = TextRole.CODE.value
            elif char_count > _ROLE_CODE_MIN_CHAR_COUNT:
                code_chars = _ROLE_CODE_PUNCT_PATTERN.findall(span_text)
                if len(code_chars) / char_count > _ROLE_CODE_PUNCT_RATIO:
                    role = TextRole.CODE.value

        # =====================================================================
        # PRIORITY 11: Hyperlink
        # =====================================================================
        if role == TextRole.BODY.value:
            if span_text in link_uris or span_text.startswith(("http", "www.")):
                role = TextRole.HYPERLINK.value

        # =====================================================================
        # PRIORITY 12: Inline Equation
        # =====================================================================
        if role == TextRole.BODY.value:
            if _is_inline_equation(span_text, span):
                role = TextRole.INLINE_EQUATION.value

        # =====================================================================
        # PRIORITY 13: Subscript / Superscript
        # =====================================================================
        if role == TextRole.BODY.value:
            is_very_small = font_size < baseline_font_size * _ROLE_VERY_SMALL_FONT_RATIO

            if is_very_small:
                if span.get("is_subscript"):
                    role = TextRole.SUBSCRIPT.value
                    span["merge_with_adjacent"] = True
                elif span.get("flags", 0) & PyMuPDFFlag.SUPERSCRIPT:
                    role = TextRole.SUPERSCRIPT.value
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
                elif is_uppercase and not _is_protected_acronym(span_text):
                    if char_count < _ROLE_VERY_SHORT_UPPERCASE_THRESHOLD:
                        if not any(c in span_text for c in ".,;:"):
                            role = TextRole.HEADER_ARTIFACT.value

        # =====================================================================
        # ASSIGN ROLE AND UPDATE STATE
        # =====================================================================
        span["role"] = role

        previous_role = role
        previous_y = span_y
        previous_x = span_x
        previous_text = span_text
        previous_font_size = font_size

        if trace_id and role not in (TextRole.BODY.value, TextRole.EMPTY.value):
            logger.debug(
                "[%s] Role: '%s' -> %s (font=%.1f, baseline=%.1f)",
                trace_id, span_text[:25], role, font_size, baseline_font_size
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

            # Determine direction
            vert_overlap = not (span_rect[3] < f_rect[1] or span_rect[1] > f_rect[3])

            if f_rect[3] <= span_rect[1]:
                direction = "above"
            elif f_rect[1] >= span_rect[3]:
                direction = "below"
            elif vert_overlap and not horiz_overlap:
                # Vertically aligned but horizontally separated -> Side layout
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

            if adjusted_dist < best_dist:
                best_dist = adjusted_dist
                best_idx = idx
                best_direction = direction

        # Apply distance threshold
        if best_idx is not None and best_dist <= max_caption_distance:
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
            x_aligned = abs(prev_x0 - curr_x0) < _CONTINUITY_X_ALIGNMENT_TOLERANCE

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

                    # Horizontal overlap check
                    x_overlap = not (prev_fig_x1 < curr_fig_x0 or prev_fig_x0 > curr_fig_x1)

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
        start_idx: int
) -> Optional[int]:
    """
    Find next sentence with body-like role, skipping headers/captions.

    Limited to _STITCH_MAX_LOOKAHEAD to prevent distant false matches.

    HARDENED v1.8:
        1. Explicit bounds checking on input.
        2. Safe .get() access for role.
        3. Null sentence handling.

    Args:
        sentences: List of sentence dictionaries.
        start_idx: Index to start searching from.

    Returns:
        Index of next body-like sentence, or None if not found.
    """
    # Safety: Input validity
    if not sentences or start_idx < 0 or start_idx >= len(sentences):
        return None

    # Enforce lookahead limit
    end_idx = min(start_idx + _STITCH_MAX_LOOKAHEAD, len(sentences))

    for j in range(start_idx, end_idx):
        sent = sentences[j]
        if not sent:
            continue

        role = sent.get("role", TextRole.BODY.value)

        # Check against skip set (O(1) lookup)
        if role not in _STITCH_SKIP_ROLES:
            return j

    return None

def _stitch_helper_columns_match(prev: Dict, next_s: Dict) -> bool:
    """
    Check if column indices match between sentences.

    Exception: If pages differ, column index mismatch is allowed
    (layout may shift between pages, e.g., sidebar on one page only).

    Args:
        prev: Previous sentence dictionary.
        next_s: Next sentence dictionary.

    Returns:
        True if columns match or cross-page exception applies.
    """
    prev_page = prev.get("page_number")
    next_page = next_s.get("page_number")

    # Cross-page: ignore column index (layout shift is common)
    if prev_page != next_page:
        return True

    # Same page: columns MUST match
    prev_col = prev.get("column_index")
    next_col = next_s.get("column_index")

    if prev_col is None or next_col is None:
        return False

    return prev_col == next_col



def _is_cross_page_continuation(
        prev_sent: Dict,
        next_sent: Dict,
        prev_spans: List[Dict] = None,
        next_spans: List[Dict] = None
) -> bool:
    """
    Heuristic: Should these two sentences be chunked together across a page break?

    This is a wrapper around _stitch_helper_should_merge for API compatibility.

    Args:
        prev_sent: Previous sentence dictionary.
        next_sent: Next sentence dictionary.
        prev_spans: Optional span list for additional checks.
        next_spans: Optional span list for additional checks.

    Returns:
        True if sentences should be merged across page break.
    """
    if not prev_sent or not next_sent:
        return False

    should_merge, _ = _stitch_helper_should_merge(
        prev_sent,
        next_sent,
        prev_spans,
        next_spans
    )

    return should_merge


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

            # Find next body-like sentence
            next_idx = _stitch_helper_find_next(all_sentences, i + 1)

            if next_idx is not None:
                next_sent = all_sentences[next_idx]

                # Defensive: Validate next_sent
                if not next_sent or not isinstance(next_sent, dict):
                    result.append(curr)
                    i += 1
                    continue

                should, reason = _stitch_helper_should_merge(curr, next_sent)

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
    merged_text = f"{t1} {t2}"

    # Create merged sentence (shallow copy is sufficient)
    merged = curr.copy()
    merged["text"] = merged_text
    merged["is_stitched"] = True

    # Update span indices
    merged["span_end_index"] = max(
        curr.get("span_end_index", 0),
        next_sent.get("span_end_index", 0)
    )

    # Update character indices (critical for timing alignment)
    if "char_end_index" in next_sent:
        merged["char_end_index"] = next_sent["char_end_index"]

    # Merge source spans if present
    curr_sources = curr.get("source_spans", [])
    next_sources = next_sent.get("source_spans", [])
    if curr_sources or next_sources:
        merged["source_spans"] = curr_sources + next_sources

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

    return merged


def _stitch_helper_should_merge(
        prev: Dict,
        next_s: Dict,
        prev_spans: List[Dict] = None,
        next_spans: List[Dict] = None
) -> Tuple[bool, str]:
    """
    Determine if two sentences should be merged.

    ARCHITECTURAL CHANGES:
        v2.0: Y-gap guard, stricter same-page role check
        v2.1: PRE-COMPUTE moved up, dynamic span gap, linguistic overrides

    Processing Order:
        0. PRE-COMPUTE: Text analysis (first/last word, signals)
        1. RULE 0: Empty text check
        2. RULE 1: Span proximity (dynamic tolerance for incomplete endings)
        3. RULE 1.5: Geometric Y-gap check
        4. RULE 2: Column alignment
        5. RULE 3: Role consistency (with linguistic override)
        6. RULE 4: Terminal punctuation (with 4 overrides)
        7. RULE 5: Span-based role blocking
        8. RULE 6: Drop cap reassembly
        9. Continuation signal detection (A-E)

    Terminal Punctuation Overrides:
        1. Incomplete ending + lowercase start
        2. Incomplete ending alone (artifact punctuation)
        3. Gerund (-ing) or possessive ('s) ending
        4. Lowercase start + tight Y-gap

    Args:
        prev: Previous sentence dictionary.
        next_s: Next sentence dictionary.
        prev_spans: Optional span list for additional checks.
        next_spans: Optional span list for additional checks.

    Returns:
        Tuple of (should_merge, reason_string).
        Reason format includes prefix for same-page stitches.
    """
    prev_text = (prev.get("text") or "").rstrip()
    next_text = (next_s.get("text") or "").lstrip()

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

    # =========================================================================
    # RULE 0: Empty Text Check
    # =========================================================================
    if not prev_text or not next_text:
        return False, "empty_text"

    # Track page context
    prev_page = prev.get("page_number")
    next_page = next_s.get("page_number")
    is_same_page = (prev_page == next_page)

    # =========================================================================
    # RULE 1: Span Proximity Check
    # Prevents merging sentences from distant content blocks
    # =========================================================================
    prev_span_end = prev.get("span_end_index", 0)
    next_span_start = next_s.get("span_start_index", 0)
    span_gap = next_span_start - prev_span_end

    allowed_gap = _STITCH_MAX_SPAN_GAP
    if has_incomplete_ending:
        allowed_gap = _STITCH_MAX_SPAN_GAP * 3  # Increase tolerance (e.g., 20 -> 60)

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

        # Only check positive gaps (next is below prev)
        # Negative gaps mean next is above prev (reordering issue, allow merge)
        if y_gap > _STITCH_MAX_Y_GAP_SAME_PAGE:
            return False, f"y_gap_too_large:{y_gap:.1f}"

    # =========================================================================
    # RULE 2: Column Alignment
    # =========================================================================
    if not _stitch_helper_columns_match(prev, next_s):
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
            # FIXED v2.1: Linguistic Override
            # If the sentence is clearly cut off, ignore role/font changes.
            # Example: "The ends of the" (Body) -> "bone" (Misclassified as Caption)
            if not has_incomplete_ending and not has_lowercase_start:
                return False, "same_page_role_mismatch"
        else:
            # Cross-page: Only block non-body + non-body
            # Rationale: Layout shifts between pages are common
            if prev_role != TextRole.BODY.value and next_role != TextRole.BODY.value:
                return False, "role_mismatch"

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

        # OVERRIDE 2 (v2.1): Artifact Punctuation on Incomplete Endings
        # Example: "The ends of the." (Period is likely PDF noise)
        if has_incomplete_ending:
            reason = "same_page:punctuation_artifact_override" if is_same_page else "punctuation_artifact_override"
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

    # =========================================================================
    # RULE 5: Span-based Role Blocking (Optional)
    # =========================================================================
    if prev_spans and next_spans:
        prev_span_idx = prev.get("span_end_index", 0)
        next_span_idx = next_s.get("span_start_index", 0)

        if prev_span_idx < len(prev_spans) and next_span_idx < len(next_spans):
            span_prev_role = prev_spans[prev_span_idx].get("role", TextRole.BODY.value)
            span_next_role = next_spans[next_span_idx].get("role", TextRole.BODY.value)

            if span_prev_role in _STITCH_BLOCKING_ROLES or span_next_role in _STITCH_BLOCKING_ROLES:
                return False, "span_role_blocked"
    # =========================================================================
    # RULE 6: Drop Cap / Initial Reassembly (SURGICAL FIX v3.5)
    # Handles: "F" + "ROM" -> "FROM"
    # =========================================================================
    if is_same_page and len(prev_text) == 1 and prev_text.isupper() and prev_text.isalpha():
        # Condition: Physically adjacent spans (gap=1)
        # Condition: Vertically aligned (y_gap ~ 0)
        if span_gap <= 1:
            prev_y_end = prev.get("end_y") or (prev.get("bbox", [0, 0, 0, 0])[3])
            next_y_start = next_s.get("bbox", [0, 0, 0, 0])[1]
            y_diff = abs(next_y_start - prev_y_end)

            # Allow slight vertical misalignment (drop caps often offset)
            # but require horizontal flow
            if y_diff < 20.0:
                return True, "same_page:drop_cap_reassembly"
    # =========================================================================
    # CONTINUATION SIGNAL DETECTION (Standard — no terminal punct)
    # =========================================================================

    # Signal A: Lowercase start (STRONG)
    if has_lowercase_start:
        reason = "same_page:lowercase_start" if is_same_page else "lowercase_start"
        return True, reason

    # Signal B: Continuation word
    if first_word in _STITCH_CONTINUATION_WORDS:
        reason = "same_page:continuation_word" if is_same_page else "continuation_word"
        return True, reason

    # Signal C: Incomplete ending (even without lowercase — weaker signal)
    if has_incomplete_ending:
        reason = "same_page:incomplete_ending" if is_same_page else "incomplete_ending"
        return True, reason

    # Signal D: Continuing punctuation
    if first_char in _STITCH_CONTINUING_PUNCT and first_char not in _STITCH_NEW_SENTENCE_PUNCT:
        reason = "same_page:continuing_punct" if is_same_page else "continuing_punct"
        return True, reason

    # Signal E: Number continuation
    if first_char.isdigit() and last_word in _STITCH_NUMBER_CONTEXT_WORDS:
        reason = "same_page:number_continuation" if is_same_page else "number_continuation"
        return True, reason

    # No signals found
    return False, "same_page_no_signal" if is_same_page else "no_signal"

# ✦────── c. Sentence Segmentation ──────✦

def _reconstruct_text_for_segmentation(
        spans: List[Dict],
        trace_id: str = None
) -> Tuple[str, List[Dict], List[int]]:
    """
    Intelligent text reconstruction with Source Map for sentence segmentation.

    Creates a character-to-span mapping that enables O(1) lookup from any
    character position to its source span, eliminating brittle find() approaches.

    HARDENED v2.1:
        1. Smart letter-spacing detection (fixes "F ROM W IKIBOOKS")
        2. Page-aware joining (page breaks → space, not newline)
        3. Defensive null handling
        4. NEW: Margin status tracking — margin boundary changes trigger breaks

    Break Priorities:
        1. Column change (same page) → "\n\n"
        2. Margin status change (same page) → "\n\n" (NEW v2.1)
        3. Role breaks (BREAK_BEFORE/AFTER roles) → "\n\n"
        4. Page change → " " (preserve sentence flow)
        5. Paragraph change (same page) → "\n"
        6. Smart joining (letter-spacing detection)
        7. Default word boundary → " "

    Args:
        spans: List of span dictionaries with cleaned_text.
        trace_id: Optional trace ID for logging.

    Returns:
        Tuple of:
            - full_text: Reconstructed text for pysbd
            - span_map: List of span boundary records
            - char_to_span: Character index → span index mapping (SOURCE MAP)
    """
    if not spans:
        return "", [], []

    full_text = ""
    span_map: List[Dict] = []
    char_to_span: List[int] = []

    # Track previous span for boundary detection
    prev_column: Optional[int] = None
    prev_para: Optional[int] = None
    prev_role: Optional[str] = None
    prev_page: Optional[int] = None
    prev_text: str = ""
    prev_margin: Optional[bool] = None  # NEW v2.1: Track margin status

    letter_spacing_joins = 0
    margin_breaks = 0  # NEW v2.1: Metric for logging

    for i, span in enumerate(spans):
        if not isinstance(span, dict):
            continue

        text = (span.get("cleaned_text") or "")
        if not text:
            continue

        curr_column = span.get("column_index", 0)
        curr_para = span.get("paragraph_index", 0)
        curr_role = span.get("role", TextRole.BODY.value)
        curr_page = span.get("page_number", 0)
        curr_margin = span.get("is_margin_content", False)  # NEW v2.1

        # Determine prefix based on boundaries
        prefix = ""
        if full_text:
            # =================================================================
            # PRIORITY 1: Column change (same page) = hard break
            # =================================================================
            if prev_column is not None and curr_column != prev_column and curr_page == prev_page:
                prefix = "\n\n"

            # =================================================================
            # PRIORITY 1.5: Margin status change (same page) = hard break (NEW v2.1)
            # Defense-in-depth: catches margin bleeding even if column detection
            # fails to assign different indices.
            # =================================================================
            elif prev_margin is not None and curr_margin != prev_margin and curr_page == prev_page:
                prefix = "\n\n"
                margin_breaks += 1

            # =================================================================
            # PRIORITY 2: Role-based breaks
            # =================================================================
            elif curr_role in _RECONSTRUCT_BREAK_BEFORE_ROLES:
                prefix = "\n\n"
            elif prev_role in _RECONSTRUCT_BREAK_AFTER_ROLES:
                prefix = "\n\n"

            # =================================================================
            # PRIORITY 3: Page change = space (preserve sentence flow)
            # =================================================================
            elif prev_page is not None and curr_page != prev_page:
                prefix = " "

            # =================================================================
            # PRIORITY 4: Paragraph change on same page = soft break
            # =================================================================
            elif prev_para is not None and curr_para != prev_para:
                prefix = "\n"

            # =================================================================
            # PRIORITY 5: Smart joining (letter-spacing detection)
            # =================================================================
            elif not full_text.endswith(" ") and not text.startswith(" "):

                # -------------------------------------------------------------
                # HARDENING: Hyphen-space normalization across line breaks
                # Example: "bio- \n logical" → "bio-logical"
                # NOTE: We intentionally preserve the hyphen to avoid
                #       corrupting character index mappings.
                # -------------------------------------------------------------
                if prev_text.endswith("-") and text and text[0].islower():
                    prefix = ""  # remove space only, keep hyphen
                    letter_spacing_joins += 1

                elif _should_join_without_space(prev_text, text):
                    prefix = ""  # No space — join directly
                    letter_spacing_joins += 1

                else:
                    prefix = " "  # Normal word boundary

        # Add prefix characters to source map (map to previous span or -1)
        prev_span_idx = span_map[-1]["span_index"] if span_map else -1
        for _ in prefix:
            char_to_span.append(prev_span_idx)

        # Build span map entry
        start_idx = len(full_text) + len(prefix)
        full_text += prefix + text
        end_idx = len(full_text)

        # Add text characters to source map (map to current span)
        for _ in text:
            char_to_span.append(i)

        span_map.append({
            "start": start_idx,
            "end": end_idx,
            "span_index": i,
            "column_index": curr_column,
            "paragraph_index": curr_para,
            "role": curr_role,
        })

        # Update tracking
        prev_column = curr_column
        prev_para = curr_para
        prev_role = curr_role
        prev_page = curr_page
        prev_text = text
        prev_margin = curr_margin  # NEW v2.1

    if trace_id:
        logger.debug(
            "[%s] Reconstructed text: %d chars, %d spans, source map size: %d, "
            "letter-spacing joins: %d, margin breaks: %d",
            trace_id, len(full_text), len(span_map), len(char_to_span),
            letter_spacing_joins, margin_breaks
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
        has_sentence_punct = text.rstrip()[-1] in '.!?' if text.rstrip() else False

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
                "bbox": bbox,
                "center_x": (bbox[0] + bbox[2]) / 2,
                "center_y": (bbox[1] + bbox[3]) / 2,
            })

    if len(label_candidates) < 2:
        return []

    # =========================================================================
    # Step 2: Spatial Clustering (Simple Proximity Grouping)
    # =========================================================================
    CLUSTER_DISTANCE = 80

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

                    # FIXED v3.5: Column-Aware Clustering
                    # Prevent jumping across column gutters.
                    # Text flows horizontally; large X gaps suggest different columns.
                    COLUMN_GUTTER_GUARD = 60
                    if dx > COLUMN_GUTTER_GUARD:
                        continue

                    # Euclidean distance for local proximity
                    distance = (dx ** 2 + dy ** 2) ** 0.5

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
    MIN_DIMENSION_SPREAD = 50

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

    healed: List[Dict] = []
    i = 0
    heal_count = 0

    while i < len(sentences):
        curr = sentences[i]

        # Skip if consumed by a previous lookahead merge
        if curr.get("_consumed"):
            i += 1
            continue

        text = curr.get("text", "").strip()

        # =====================================================================
        # CORRECTED: Only process sentences ending with PERIOD
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
            healed.append(curr)
            i += 1
            continue

        # =====================================================================
        # EXTENDED LOOKAHEAD: Search for continuation (hopping over artifacts)
        # =====================================================================
        curr_page = curr.get("page_number")
        found_continuation = False
        MAX_LOOKAHEAD = 5

        for j in range(i + 1, min(i + 1 + MAX_LOOKAHEAD, len(sentences))):
            next_sent = sentences[j]

            # Stop if we hit a new page
            if next_sent.get("page_number") != curr_page:
                break

            # Skip if already consumed by a previous merge
            if next_sent.get("_consumed"):
                continue

            next_text = next_sent.get("text", "").strip()
            if not next_text:
                continue

            # CONTINUATION SIGNAL: Lowercase start = strong match
            if next_text[0].islower():
                # Perform Merge
                merged_text = text_no_period + " " + next_text

                merged = curr.copy()
                merged["text"] = merged_text
                merged["tts_text"] = merged_text
                merged["span_end_index"] = next_sent.get("span_end_index",
                                                         curr.get("span_end_index"))
                merged["char_end"] = next_sent.get("char_end", curr.get("char_end"))
                merged["healed_truncation"] = True

                healed.append(merged)
                heal_count += 1
                found_continuation = True

                # Mark the future sentence as consumed so the main loop skips it
                sentences[j]["_consumed"] = True

                if trace_id:
                    logger.debug("[%s] Truncation healed (gap=%d): '%s' + '%s'",
                                 trace_id, j - i, last_word, next_text[:10])
                break

        if found_continuation:
            i += 1
        else:
            healed.append(curr)
            i += 1

    if trace_id and heal_count:
        logger.info("[%s] Truncation healing: %d sentences repaired", trace_id, heal_count)

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

    healed: List[Dict] = []
    skip_until: int = -1
    heal_count = 0

    for i, curr in enumerate(all_sentences):
        if i <= skip_until:
            continue

        text = curr.get("text", "").strip()

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
            is_continuation = first_char.islower() or len(peek_text.split()) <= 3

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

        curr_page = curr.get("page_number", 1)
        next_page = next_sent.get("page_number", 1)
        if curr_page != next_page:
            merged["crosses_pages"] = True
            merged["page_range"] = [curr_page, next_page]

        healed.append(merged)
        heal_count += 1
        skip_until = merge_target_idx

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
    Sentence segmentation with Source Map lookup and Fragment Healing.

    Uses char_to_span for O(1) mapping from character positions to source spans.
    Includes robust 4-method fallback chain to minimize sentence drops.

    HARDENED v3.0:
        1. Deterministic role resolution
        2. Bbox propagation for highlighting
        3. tts_text field for downstream compatibility
        4. Improved cursor tracking for normalized matches
        5. Fragment Healing post-processing
        6. NEW: Truncation Healing (repairs cuts at aux verbs/articles)
        7. NEW: Punctuation Normalization

    Alignment Methods:
        1. direct: Exact string find (fastest)
        2. normalized: Whitespace-normalized match
        3. fuzzy: First N words match
        4. fallback: Cursor position (never drops)

    Post-Processing:
        - Fragment Healing: Merges orphan fragments with following sentences
        - Truncation Healing: Repairs sentences cut at auxiliary verbs
        - Punctuation Normalization: Fixes spacing around brackets/punctuation

    Args:
        full_text: Reconstructed text from all spans.
        char_to_span: Character-to-span index mapping.
        all_spans: List of all span dictionaries.
        trace_id: Optional trace ID for logging.

    Returns:
        List of sentence dictionaries with text, span indices, and metadata.
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
        span_indices: set = set()
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
                roles.add(span.get("role", TextRole.BODY.value))

        # Handle unmapped sentences
        if not span_indices:
            if trace_id:
                logger.warning(
                    "[%s] Sentence #%d has no mapped spans: '%s...'",
                    trace_id, sent_idx, sent[:40]
                )
            processed_sentences.append({
                "text": sent,
                "tts_text": sent,
                "span_start_index": 0,
                "span_end_index": 0,
                "paragraph_index": 0,
                "column_index": 0,
                "page_number": 1,
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
            primary_role = TextRole.BODY.value
            for priority_role in role_priority:
                if priority_role in roles:
                    primary_role = priority_role
                    break
            else:
                primary_role = sorted(roles)[0]
        else:
            primary_role = first_span.get("role", TextRole.BODY.value)

        primary_page = first_span.get("page_number", 1)

        # =====================================================================
        # STEP 3.5: Compute sentence bbox
        # =====================================================================
        sentence_bbox = _compute_sentence_bbox(all_spans, sorted_spans)

        # =====================================================================
        # STEP 4: Build sentence record
        # =====================================================================
        sentence_record: Dict = {
            "text": sent,
            "tts_text": sent,
            "span_start_index": start_span_idx,
            "span_end_index": end_span_idx,
            "paragraph_index": first_span.get("paragraph_index", 0),
            "column_index": primary_column,
            "page_number": primary_page,
            "role": primary_role,
            "char_start": start_char,
            "char_end": end_char,
            "alignment_method": alignment_method,
        }

        if sentence_bbox:
            sentence_record["bbox"] = sentence_bbox

        if len(pages) > 1:
            sentence_record["crosses_pages"] = True
            sentence_record["page_range"] = sorted(pages)

        if len(columns) > 1:
            sentence_record["crosses_columns"] = True
            sentence_record["column_range"] = sorted(columns)

        sentence_record["source_span_count"] = len(span_indices)
        processed_sentences.append(sentence_record)

    # =========================================================================
    # POST-PROCESSING: Fragment Healing (v3.0 — Linguistic + Buffered)
    # =========================================================================

    # NOTE:
    # - Forward-only merging via delayed emission (pending buffer)
    # - Uses global _HEALING_CUT_INDICATORS for consistency
    # - Metadata always populated

    healed_sentences: List[Dict] = []
    pending: Optional[Dict] = None
    healed_count = 0

    def _finalize_sentence(sent: Dict) -> None:
        # Ensure schema stability
        sent.setdefault("healing_applied", "none")
        sent["alignment_confidence"] = sent.get("alignment_method", "direct")
        healed_sentences.append(sent)

    for curr in processed_sentences:
        curr_text = (curr.get("text") or "").strip()

        # Always initialize metadata
        curr.setdefault("healing_applied", "none")
        curr["alignment_confidence"] = curr.get("alignment_method", "direct")

        if not pending:
            pending = curr
            continue

        pending_text = (pending.get("text") or "").strip()

        # -------------------------------
        # Linguistic Signals
        # -------------------------------
        pending_words = pending_text.rstrip(".!?").split()
        last_word = pending_words[-1].lower() if pending_words else ""

        # REFERENCE GLOBAL CONSTANT HERE
        ends_with_cut = last_word in _HEALING_CUT_INDICATORS

        has_terminal_punct = pending_text.endswith((".", "!", "?"))
        starts_lowercase = bool(curr_text) and curr_text[0].islower()

        # Is the PENDING sentence a fragment that needs the CURRENT one?
        is_fragment_candidate = (ends_with_cut or not has_terminal_punct)

        # -------------------------------
        # Merge Decision (Forward-only)
        # -------------------------------
        if is_fragment_candidate and starts_lowercase:
            # Merge curr into pending
            merged_text = pending_text.rstrip(".!?") + " " + curr_text.lstrip()

            pending["text"] = merged_text
            pending["tts_text"] = merged_text
            pending["healing_applied"] = "fragment"
            healed_count += 1
            pending["source_sentence_count"] = (
                    pending.get("source_sentence_count", 1)
                    + curr.get("source_sentence_count", 1)
            )

            # Span / char continuity (conservative)
            pending["span_end_index"] = curr.get(
                "span_end_index", pending.get("span_end_index")
            )
            pending["char_end"] = curr.get(
                "char_end", pending.get("char_end")
            )

            # Do not finalize pending yet; it might need to eat the next sentence too
            continue

        # -------------------------------
        # No merge → finalize pending
        # -------------------------------
        _finalize_sentence(pending)
        pending = curr

    # Flush final pending sentence
    if pending:
        _finalize_sentence(pending)

    processed_sentences = healed_sentences

    # =========================================================================
    # POST-PROCESSING: Truncation Healing (NEW v3.0)
    # =========================================================================
    # Repairs sentences cut at auxiliary verbs, articles, or prepositions.
    # Only heals sentences ending with PERIOD (not ?, !, :).
    # =========================================================================
    processed_sentences = _heal_truncated_sentences(processed_sentences, trace_id)

    # =========================================================================
    # POST-PROCESSING: Punctuation Normalization (NEW v3.0)
    # =========================================================================
    # Fixes spacing artifacts around punctuation marks.
    # "( text )" → "(text)"
    # =========================================================================
    for sent in processed_sentences:
        sent["text"] = _normalize_punctuation_spacing(sent["text"])
        sent["tts_text"] = _normalize_punctuation_spacing(sent.get("tts_text", sent["text"]))

    # =========================================================================
    # LOGGING
    # =========================================================================
    if trace_id:
        methods: Dict[str, int] = {}
        for s in processed_sentences:
            m = s.get("alignment_method", "unknown")
            methods[m] = methods.get(m, 0) + 1
        logger.debug(
            "[%s] Segmentation: %d sentences from %d raw, %d empty skipped, "
            "alignment methods=%s, failures=%d, healed=%d",
            trace_id, len(processed_sentences), len(raw_sentences),
            empty_skipped, methods, alignment_failures, healed_count
        )

    return processed_sentences


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
    if not text:
        return ""

    # Remove space after opening brackets
    text = re.sub(r'([(\[{])\s+', r'\1', text)

    # Remove space before closing brackets
    text = re.sub(r'\s+([)\]}])', r'\1', text)

    # Remove space before comma/period/colon/semicolon
    text = re.sub(r'\s+([,.;:])', r'\1', text)

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

    When whitespace is collapsed during normalization, the normalized length
    differs from the actual span length. This function walks through full_text
    to find where the normalized content actually ends.

    Args:
        full_text: Original text with original whitespace.
        start: Starting position in full_text.
        normalized_target: Normalized (whitespace-collapsed) target string.

    Returns:
        End position in full_text.
    """
    target_idx = 0
    text_idx = start

    while target_idx < len(normalized_target) and text_idx < len(full_text):
        # Skip extra whitespace in full_text
        while text_idx < len(full_text) and full_text[text_idx].isspace():
            if target_idx < len(normalized_target) and normalized_target[target_idx].isspace():
                # Both have whitespace — advance both
                target_idx += 1
                text_idx += 1
                # Skip any additional whitespace in full_text
                while text_idx < len(full_text) and full_text[text_idx].isspace():
                    text_idx += 1
                break
            else:
                # Extra whitespace in full_text only — skip it
                text_idx += 1

        # Match non-whitespace characters
        if target_idx < len(normalized_target) and text_idx < len(full_text):
            if not normalized_target[target_idx].isspace():
                target_idx += 1
                text_idx += 1

    return text_idx


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
        trace_id: str = None
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
        footer_zone_min_y = page_height * (1.0 - _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT)

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

            # SAFETY FIX: Explicit Binning (Round to nearest 10px bucket)
            # Uses _REGION_Y_BAND_ROUNDING (10) for unambiguous grouping
            rounded_y = int(round(span_y_top / _REGION_Y_BAND_ROUNDING) * _REGION_Y_BAND_ROUNDING)

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

    # Merge nearby bands
    header_band_pages, header_band_texts = merge_nearby_bands(
        header_band_pages, header_band_texts, _GLOBAL_BAND_MERGE_TOLERANCE
    )
    footer_band_pages, footer_band_texts = merge_nearby_bands(
        footer_band_pages, footer_band_texts, _GLOBAL_BAND_MERGE_TOLERANCE
    )

    # Apply frequency threshold
    min_floor = 2 if total_pages > 1 else 1
    frequency_threshold = max(1, int(total_pages * _GLOBAL_BAND_MIN_PAGE_FRACTION))

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

    for span in all_spans:
        # Skip already-tagged non-body roles
        current_role = span.get("role", TextRole.BODY.value)
        if current_role in {TextRole.HEADER.value, TextRole.FOOTER.value,
                            TextRole.PAGE_NUMBER.value}:
            continue

        bbox = span.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        page_num = span.get("page_number", 1)
        page_height = page_heights.get(page_num, 800)

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
        if in_header_band:
            span["role"] = TextRole.HEADER.value
            span["is_global_header"] = True
            span["header_match_reason"] = "band"
            tagged_header += 1
        elif in_footer_band:
            span["role"] = TextRole.FOOTER.value
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

    band_to_sample_header: Dict[int, Optional[str]] = {y: None for y in header_set}
    band_to_sample_footer: Dict[int, Optional[str]] = {y: None for y in footer_set}

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

            # Capture first sample for each band
            if y_band in header_set and band_to_sample_header.get(y_band) is None:
                band_to_sample_header[y_band] = text

            if y_band in footer_set and band_to_sample_footer.get(y_band) is None:
                band_to_sample_footer[y_band] = text

    # Build output summaries
    headers_summary: List[Dict] = [
        {"y": y, "sample_text": band_to_sample_header.get(y) or ""}
        for y in sorted(header_set)
    ]

    footers_summary: List[Dict] = [
        {"y": y, "sample_text": band_to_sample_footer.get(y) or ""}
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

        if matches_header_band:
            if not normalized_header_samples:
                # No samples available — exclude based on band position only
                should_exclude = True
                exclusion_reason = "header_band_no_samples"
            else:
                # Check similarity against header samples
                similarity_threshold = (
                    _FILTER_SHORT_TEXT_THRESHOLD
                    if len(normalized_span_text) < _FILTER_SHORT_TEXT_LENGTH
                    else _FILTER_FUZZY_MATCH_THRESHOLD
                )

                for sample in normalized_header_samples:
                    similarity = _text_similarity(normalized_span_text, sample)
                    if similarity >= similarity_threshold:
                        should_exclude = True
                        exclusion_reason = f"header_match_{similarity:.2f}"
                        break

        if not should_exclude and matches_footer_band:
            if not normalized_footer_samples:
                # No samples available — exclude based on band position only
                should_exclude = True
                exclusion_reason = "footer_band_no_samples"
            else:
                # Check similarity against footer samples
                similarity_threshold = (
                    _FILTER_SHORT_TEXT_THRESHOLD
                    if len(normalized_span_text) < _FILTER_SHORT_TEXT_LENGTH
                    else _FILTER_FUZZY_MATCH_THRESHOLD
                )

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
    global_bands = _compute_global_header_footer_bands(page_outputs)
    global_header_bands = global_bands["header_bands"]
    global_footer_bands = global_bands["footer_bands"]

    if trace_id:
        logger.info(
            "[%s] Global bands detected: %d header, %d footer",
            trace_id, len(global_header_bands), len(global_footer_bands)
        )

    # If no global bands, nothing to normalize
    if not global_header_bands and not global_footer_bands:
        return

    # =========================================================================
    # STEP 2: Build sample text summaries
    # =========================================================================
    band_summaries = _summarize_document_bands(
        page_outputs,
        global_header_bands,
        global_footer_bands
    )

    # Extract sample texts for filtering
    header_samples: List[str] = [
        h["sample_text"] for h in band_summaries["headers"]
        if h.get("sample_text")
    ]

    footer_samples: List[str] = [
        f["sample_text"] for f in band_summaries["footers"]
        if f.get("sample_text")
    ]

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

def _sanitize_for_tts(
        text: str,
        add_terminal_punct: bool = True,
        change_tracker: Dict = None,
        trace_id: str = None
) -> str:
    """
    TTS text sanitization with optional change tracking.

    Converts text to TTS-safe format. Optionally tracks all modifications
    for timing synchronization (when change_tracker dict is provided).

    Operations:
        1. Smart case normalization (prevent TTS screaming)
        2. Subscript/superscript expansion (H₂O → H 2 O)
        3. Character substitutions (symbols → words)
        4. Unicode normalization (NFKC)
        5. Empty/orphan bracket removal
        6. Space normalization
        7. Terminal punctuation enforcement

    Args:
        text: Raw text to sanitize.
        add_terminal_punct: Whether to add period if missing.
        change_tracker: Optional dict to populate with change metadata.
        trace_id: Optional trace ID for logging.

    Returns:
        TTS-safe sanitized text.
    """
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
    # STEP 1: Smart case normalization (prevent TTS from "shouting")
    # =========================================================================
    if text.isupper() and len(text) > _TTS_ACRONYM_LENGTH_THRESHOLD:
        # Check if it's likely an acronym (no spaces, short)
        if " " in text or len(text) > _TTS_LONG_TEXT_THRESHOLD:
            text = text.capitalize()
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
        return f"{letter} {digits} "

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
    # STEP 3.5: Unit notation expansion (BEFORE blanket substitutions)
    # Converts: 50/mm² → 50 per mm², 10/s → 10 per s
    # =========================================================================
    original_step35 = text
    text = _TTS_UNIT_SLASH_PATTERN.sub(r'\1 per \2', text)

    if text != original_step35:
        modifications.append("unit_notation_expanded")

    # =========================================================================
    # STEP 4: Character substitutions (symbols → words)
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
    # STEP 5: Unicode normalization (NFKC)
    # =========================================================================
    original = text
    text = unicodedata.normalize("NFKC", text)

    if text != original:
        modifications.append("unicode_normalized")

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
    # STEP 7: Space normalization
    # =========================================================================
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    # =========================================================================
    # STEP 8: Terminal punctuation enforcement (decoder runaway mitigation)
    # =========================================================================
    if add_terminal_punct and text:
        needs_punct = True

        # Already ends with terminal punctuation
        if text[-1] in ".!?":
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

    return text

def _is_tts_viable(
        text: str,
        role: str = None,
        is_continuation: bool = False,
        include_captions: bool = False,
        include_headings: bool = True,
        trace_id: str = None
) -> Tuple[bool, str]:
    """
    Gatekeeper to reject content unsuitable for TTS.

    MODIFIED v2.2:
        Added Gate 8 (Disconnected Label Sequence) to catch diagram labels
        that escape upstream detection.

    Eight-gate filtering:
        1. Role Gate - excludes non-TTS roles
        2. Punctuation Noise Gate - rejects punct-only text
        3. Whitelist Gate - allows known short sentences
        4. Length Gate - rejects too-short content
        5. Lowercase/Continuation Gate - handles fragments
        6. Truncation Gate - detects cut-off sentences
        7. Garble Gate - detects merged-column artifacts
        8. Disconnected Label Gate - detects diagram label sequences (NEW v2.2)

    Args:
        text: Text to evaluate.
        role: Content role (uses TextRole enum values).
        is_continuation: Whether this is marked as continuation.
        include_captions: Whether to include caption roles.
        include_headings: Whether to include heading roles.
        trace_id: Optional trace ID for logging.

    Returns:
        Tuple of (is_viable, rejection_reason).
    """
    # Default role
    if role is None:
        role = TextRole.BODY.value

    # =========================================================================
    # GATE 1: Role Gate
    # =========================================================================
    if role in _TTS_NON_VIABLE_ROLES:
        return False, "role_excluded"

    if role == TextRole.CAPTION.value and not include_captions:
        return False, "caption_excluded"

    if role in (TextRole.HEADING.value, TextRole.SUBHEADING.value) and not include_headings:
        return False, "heading_excluded"

    # =========================================================================
    # BASIC VALIDATION
    # =========================================================================
    stripped = text.strip()
    if not stripped or stripped.isdigit():
        return False, "noise"

    # =========================================================================
    # GATE 2: Punctuation Noise Gate
    # =========================================================================
    if all(c in _TTS_PUNCT_CHARS for c in stripped):
        return False, "punct_only"

    # =========================================================================
    # GATE 3: Whitelist Gate (Scientific terms, short answers)
    # =========================================================================
    if stripped.lower() in VALID_SHORT_SENTENCES:
        return True, ""

    # =========================================================================
    # PRE-COMPUTE METRICS
    # =========================================================================
    alpha = sum(1 for c in stripped if c.isalpha())
    words = stripped.split()
    word_count = len(words)
    has_terminal = stripped[-1] in ".!?" if stripped else False
    char_count = len(stripped)

    # =========================================================================
    # GATE 4: Length Gate
    # =========================================================================
    if char_count < _TTS_MIN_CHAR_COUNT and alpha < _TTS_MIN_ALPHA_COUNT and not has_terminal:
        return False, "too_short_chars"

    if word_count == 1 and char_count < _TTS_MIN_SINGLE_WORD_CHARS and not has_terminal:
        return False, "too_short_single_word"

    # =========================================================================
    # GATE 5: Lowercase/Continuation Gate (The "And" Fix)
    # =========================================================================
    if stripped and stripped[0].islower():
        # ALLOW: Marked as continuation by the Stitcher
        if is_continuation:
            pass
        else:
            first_word = words[0].rstrip(".,;:!?").lower()
            # ALLOW: Known lowercase terms (pH, mRNA, iPhone)
            if first_word in _TTS_VALID_LOWERCASE_STARTERS:
                pass
            # ALLOW: Substantial sentence (likely bad segmentation, but worth reading)
            elif has_terminal and word_count >= _TTS_SUBSTANTIAL_WORD_COUNT:
                pass
            # ALLOW: Long fragment (better to read than drop)
            elif word_count >= _TTS_LONG_FRAGMENT_WORD_COUNT:
                pass
            else:
                return False, "fragment_lowercase"

    # =========================================================================
    # GATE 6: Truncation Gate
    # =========================================================================
    if not has_terminal and words:
        last_word = words[-1].lower()
        # Use shared truncation word set from Subgroup 4.2
        if last_word in _STITCH_INCOMPLETE_ENDINGS:
            # Allow if very substantial (likely intentional fragment for stitching)
            if not (word_count >= _TTS_TRUNCATION_MIN_WORDS and
                    char_count >= _TTS_TRUNCATION_MIN_CHARS):
                return False, "truncation_word"

    # =========================================================================
    # GATE 7: Garble Gate (Cap Transitions)
    # =========================================================================
    if word_count >= _TTS_GARBLE_MIN_WORDS:
        caps = 0
        for i in range(1, len(words)):
            prev = words[i - 1]
            curr = words[i]
            if (prev and curr and prev[-1].islower() and
                    curr[0].isupper() and prev[-1] not in ".!?"):
                caps += 1
        ratio = caps / (word_count - 1) if word_count > 1 else 0
        # If >40% of words have weird mid-sentence caps, it's garbage
        if word_count >= _TTS_GARBLE_LONG_WORD_COUNT and ratio > _TTS_GARBLE_CAPS_RATIO:
            return False, "garble_caps"
        elif word_count < _TTS_GARBLE_LONG_WORD_COUNT and caps >= _TTS_GARBLE_CAPS_COUNT:
            return False, "garble_caps"

    # =========================================================================
    # GATE 8: Disconnected Label Sequence (NEW v2.2)
    # =========================================================================
    # Detects patterns like "Label A. Label B. Label C." which are diagram
    # labels that weren't caught by _is_diagram_label (e.g., figure not detected).
    #
    # Characteristics of diagram label sequences:
    #   - Multiple short "sentences" punctuated with periods
    #   - Average words per sentence <= 2.5
    #   - No prose flow words connecting them
    #
    # Example matches:
    #   - "Force control signal. Driving signal. Length control signal."
    #   - "10. 20. 30. X-Axis."
    #
    # Example non-matches:
    #   - "He said hello. She said goodbye." (prose flow with pronouns)
    #   - "The experiment worked. The results were clear." (prose with articles)
    # =========================================================================
    if 3 <= word_count <= 25:
        # Split on sentence-ending punctuation
        potential_labels = [s.strip() for s in re.split(r'[.!?]', stripped) if s.strip()]

        if len(potential_labels) >= 3:
            # Calculate average words per "sentence"
            total_words_in_labels = sum(len(label.split()) for label in potential_labels)
            avg_words_per_label = total_words_in_labels / len(potential_labels)

            # Multiple very short "sentences" = likely diagram labels
            if avg_words_per_label <= 2.5:
                # Additional check: no connecting/prose words between "sentences"
                prose_indicators = {'and', 'or', 'but', 'the', 'a', 'an', 'is', 'are',
                                    'was', 'were', 'he', 'she', 'it', 'they', 'this',
                                    'that', 'which', 'who', 'have', 'has', 'had'}

                words_in_labels = [word.lower() for label in potential_labels
                                   for word in label.split()]
                prose_word_count = sum(1 for w in words_in_labels if w in prose_indicators)
                prose_ratio = prose_word_count / len(words_in_labels) if words_in_labels else 0

                # If less than 20% prose words, it's likely disconnected labels
                if prose_ratio < 0.20:
                    if trace_id:
                        logger.debug(
                            "[%s] _is_tts_viable: '%s' rejected (disconnected_labels: "
                            "%d fragments, %.1f avg words, %.0f%% prose)",
                            trace_id, stripped[:40], len(potential_labels),
                            avg_words_per_label, prose_ratio * 100
                        )
                    return False, "disconnected_labels"

    return True, ""


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
    viable_sentences: List[Dict] = []
    rejection_counts: Dict[str, int] = {}

    for s in sentences:
        tts_text = s.get("tts_text", "")
        sent_role = s.get("role", TextRole.BODY.value)
        is_continuation = s.get("is_continuation", False)

        is_viable, reason = _is_tts_viable(
            tts_text,
            role=sent_role,
            is_continuation=is_continuation,
            trace_id=trace_id
        )

        if is_viable:
            viable_sentences.append(s)
        else:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

            if trace_id:
                logger.debug(
                    "[%s] Sentence rejected: reason=%s, text='%s...'",
                    trace_id, reason, tts_text[:40]
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
    if not is_viable:
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
        "end_time": chunk_start_time + chunk_duration,
        "duration_seconds": chunk_duration,
        "sentences": viable_sentences,
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
        trace_id: str = None
) -> Dict:
    """
    STAGE 1: Extraction & Structure Analysis.

    HARDENED v1.9.0:
        1. Added TEXT_PRESERVE_LIGATURES and TEXT_PRESERVE_WHITESPACE flags
           to support downstream kerning analysis (Phase 5 preparation).
        2. Implemented Context-Aware Dynamic Sorting (Step 4.5 + Step 5):
           - Table content: tight tolerance (3px) preserves row integrity
           - Prose content: loose tolerance (8px) catches italic baseline drift
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
        "dict",
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
    regions = _detect_page_regions(page, raw_spans)

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
            span["_is_table_content"] = any(
                r.contains(fitz.Point(span_center_x, span_center_y))
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
    _detect_paragraphs(raw_spans)

    # =========================================================================
    # STEP 7: Filter Spans (The "Shield")
    # =========================================================================
    valid_spans, excluded_spans = _filter_spans(raw_spans, regions)

    # =========================================================================
    # STEP 8: Clean Spans (The "Skin")
    # =========================================================================
    _clean_spans(valid_spans)

    # =========================================================================
    # STEP 9: Assign Roles (The "Classifier")
    # =========================================================================
    _assign_roles(
        valid_spans,
        regions,
        page_width=page.rect.width,
        page_height=page.rect.height,
        trace_id=trace_id
    )

    # =========================================================================
    # STEP 10: Associate Captions (The "Links")
    # =========================================================================
    _associate_captions_to_figures(
        valid_spans,
        figure_tuples,
        page_height=page.rect.height,
        trace_id=trace_id
    )

    # =========================================================================
    # STEP 11: Assign span_index (Sacred Index — must be last)
    # =========================================================================
    for idx, span in enumerate(valid_spans):
        span["span_index"] = idx

    return {
        "metadata": {
            "page_number": page_num + 1,
            "width": page.rect.width,
            "height": page.rect.height,
        },
        "structure": regions,
        "content": valid_spans,
        "excluded": excluded_spans
    }


def compile_tts_ready_content(
        raw_data_list: List[Dict],
        trace_id: str = None
) -> Dict:
    """
    STAGE 2: Transform extracted page data into TTS-ready chunks.

    MODIFIED v3.2: Added cross-page truncation healing after stitching.
    """
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

    # =========================================================================
    # STEP 2: Process Each Page — Filter, Segment, Collect
    # =========================================================================
    all_sentences: List[Dict] = []
    global_sentence_index = 0

    for page_data in raw_data_list:
        page_num = page_data.get("metadata", {}).get("page_number")

        header_sample_texts = [h.get("sample_text", "") for h in document_headers]
        footer_sample_texts = [f.get("sample_text", "") for f in document_footers]
        page_height = page_data.get("metadata", {}).get("height", _FILTER_DEFAULT_PAGE_HEIGHT)

        filtered_spans, excluded_spans = _filter_spans_by_global_bands(
            page_data,
            global_bands["header_bands"],
            global_bands["footer_bands"],
            header_samples=header_sample_texts,
            footer_samples=footer_sample_texts,
            page_height=page_height,
            trace_id=trace_id
        )

        # Store excluded for debugging/analysis
        page_data["excluded_by_global_bands"] = excluded_spans

        if not filtered_spans:
            continue

        tts_viable_spans = []
        for span in filtered_spans:
            # [cite_start]Default to BODY if role is missing [cite: 5]
            role = span.get("role", TextRole.BODY.value)
            if role not in _TTS_NON_VIABLE_ROLES:
                tts_viable_spans.append(span)

        filtered_spans = tts_viable_spans

        if not filtered_spans:
            continue

        full_text, span_map, char_to_span = _reconstruct_text_for_segmentation(
            filtered_spans, trace_id
        )

        page_sentences = _segment_sentences(
            full_text, char_to_span, filtered_spans, trace_id
        )

        # Detect structural continuity
        continuity = page_data.get("continuity", {})

        # Tag sentences with page info and continuity
        for sent in page_sentences:
            sent["page_number"] = page_num
            sent["in_continued_table"] = continuity.get("has_continued_table", False)
            sent["in_continued_figure"] = continuity.get("has_continued_figure", False)

            # Inject role from dominant span
            start_idx = sent.get("span_start_index", 0)
            if start_idx < len(filtered_spans):
                sent["role"] = filtered_spans[start_idx].get("role", TextRole.BODY.value)
            else:
                sent["role"] = TextRole.BODY.value

            all_sentences.append(sent)

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

    for i, sent in enumerate(all_sentences):
        tts_text = _sanitize_for_tts(sent.get("text", ""), trace_id=trace_id)
        sent_role = sent.get("role", TextRole.BODY.value)

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

        viable, reject_reason = _is_tts_viable(tts_text, sent_role, trace_id=trace_id)

        if not viable and not force_keep:
            skipped_count += 1
            prev_sentence_text = tts_text
            continue

        prev_sentence_text = tts_text
        sent["tts_text"] = tts_text
        sent["global_index"] = global_sentence_index
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
            _finalize_chunk(all_chunks, current_chunk_sentences, _AVG_CHARS_PER_SEC, trace_id)
        elif all_chunks:
            # Merge tiny final chunk
            prev_chunk = all_chunks[-1]
            prev_chunk["sentences"].extend(current_chunk_sentences)

            merged_text = " ".join(
                s.get("tts_text", s.get("text", ""))
                for s in prev_chunk["sentences"]
            )
            prev_chunk["text"] = merged_text

            new_pages = {s.get("page_number") for s in current_chunk_sentences if
                         s.get("page_number") is not None}
            existing_pages = set(prev_chunk.get("pages", []))
            prev_chunk["pages"] = sorted(existing_pages | new_pages)

            prev_chunk["duration_seconds"] = len(merged_text) / _AVG_CHARS_PER_SEC
            prev_chunk["end_time"] = prev_chunk["start_time"] + prev_chunk["duration_seconds"]
        else:
            _finalize_chunk(all_chunks, current_chunk_sentences, _AVG_CHARS_PER_SEC, trace_id)

    total_duration = all_chunks[-1]["end_time"] if all_chunks else 0.0

    if trace_id:
        logger.info(
            "[%s] Stage 2 complete: %d chunks, %d sentences, %d skipped, %.1fs duration",
            trace_id, len(all_chunks), global_sentence_index, skipped_count, total_duration
        )

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
    }