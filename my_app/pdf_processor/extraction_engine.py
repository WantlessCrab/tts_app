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

    'a', 'i', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in',
    'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us',
    'we', 'am', 'are', 'and', 'the', 'for', 'but', 'not', 'you', 'all',
    'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'his', 'how',
    'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'did',
    'get', 'let', 'put', 'say', 'she', 'too', 'use', 'own', 'per',
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

# PHASE 1.5: Continuity Override Configuration

# Roles that can be overridden by stream continuity (geometry-based roles only)
_CONTINUITY_OVERRIDE_CANDIDATES: frozenset = frozenset({
    TextRole.INSIDE_FIGURE.value,
    TextRole.FIGURE_LABEL.value,
})

# Hard semantic veto patterns - spans matching these are NEVER promoted to body
_CONTINUITY_VETO_PATTERNS: tuple = (
    "figure", "fig.", "fig ", "table", "chart", "graph", "diagram",
    "source:", "note:", "©", "http://", "https://", "www.", "doi:",
    "page ", "p. ", "pp.", "vol.", "chapter", "section", "appendix",
)

# Maximum Y-gap (points) between spans for adjacency (prevents cross-block contagion)
_CONTINUITY_MAX_Y_GAP: float = 30.0

# Terminal punctuation that definitively ends a sentence
# NOTE: Excludes ':' (introduces clauses) and ';' (joins clauses) per lead review
_CONTINUITY_TERMINAL_CHARS: str = ".!?"

# Number of spans to include from adjacent pages in the window
_WINDOW_TAIL_SPAN_COUNT: int = 10  # Last N spans from previous page
_WINDOW_HEAD_SPAN_COUNT: int = 10  # First N spans from next page

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

# Ragged-Edge Magnet
MAGNET_GAP_EM = 2.0

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
_STITCH_MAX_NEGATIVE_Y_GAP = 25.0  # px, defensive threshold

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

_PARA_ABBREVIATIONS = {
    "dr", "mr", "ms", "mrs", "prof", "fig", "eq",
    "st", "vs", "etc", "e.g", "i.e", "cf", "al"
}

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
                span_role in ("figure_label", "caption", "inside_figure", "table_cell") or
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

    for (line_id, row_key), group in vlg_groups.items():
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

            # Safe-harbor threshold
            if min_gap <= (MAGNET_GAP_EM * fs):
                m["is_margin_content"] = False
                m["column_index"] = nearest_col
                m["layout_stream"] = f"body_col_{nearest_col}"
                magnet_reclassified += 1

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

    for (line_id, row_key), group in vlg_groups.items():
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
                        trace_id, ratio, primary_weight, secondary_weight, body_left_bin, body_right_bin
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
        text_page: Dict,
        page_num: int,
        trace_id: str = None
) -> List[Dict]:
    """
    Flatten PyMuPDF block/line/span hierarchy into a linear list of span dicts.

    Phase 1.5:  Deterministic order stabilization (row_key → x1 → x0 → index)
    Phase 1.75: Horizontal adjacency signaling (metadata-only, lossless)
    A2-V:       Vertical continuation detection within block
    """

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

            line_text_canonical = "".join(s.get("text", "") for s in spans)

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

                curr_text = (curr.get("text") or "").rstrip()
                next_text = (nxt.get("text") or "").lstrip()

                if not curr_text or not next_text:
                    continue

                if curr_text.endswith((".", "?", "!")):
                    continue

                if curr_text.endswith("-"):
                    horizontal_links[i] = ("hyphen_wrap", "a2_horizontal_hyphen")
                else:
                    horizontal_links[i] = ("space_join", "a2_horizontal_same_row")

                horizontal_link_count += 1

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
                    raw_text = span.get("text", "")

                    # =========================================================
                    # REVISION A2-V: Vertical Continuation Detection
                    # First span of line only — links to prior line tail if safe
                    # =========================================================
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

                        # Guard 2: Previous span must not end with hard sentence boundary
                        prev_tail_stripped = prev_text.rstrip(" '\"”’)]}")
                        prev_hard_end = prev_tail_stripped.endswith((".", "?", "!"))

                        # Guard 3: Context-aware uppercase handling (RELAXED for scientific text)
                        # Allow uppercase continuation if previous line is clearly unfinished
                        curr_starts_upper = len(curr_text) > 0 and curr_text[0].isupper()
                        prev_clearly_unfinished = prev_tail_stripped.endswith(
                            (",", ";", ":", "-", "—", "(", "[", "{")
                        )
                        # Block uppercase only if it genuinely looks like a new sentence
                        curr_looks_new_sentence = (
                                curr_starts_upper and
                                not prev_clearly_unfinished and
                                not prev_text.endswith("-")  # Hyphen wrap is always safe
                        )

                        # Guard 4: Geometry sanity (vertical gap + indent within tolerance)
                        allow_geom = False
                        prev_bbox = carry_tail.get("bbox")

                        # PHASE 0/1: Geometry checks only valid if current bbox exists
                        if bbox is not None and prev_bbox is not None and len(prev_bbox) >= 4:
                            v_gap = bbox[1] - prev_bbox[3]  # curr_y0 - prev_y1
                            indent_dx = abs(bbox[0] - prev_bbox[0])  # x-offset
                            fs = float(carry_tail.get("font_size") or 10.0)

                            max_v_gap = max(2.0, fs * 1.8)
                            max_indent = max(12.0, fs * 3.5)

                            allow_geom = (v_gap <= max_v_gap) and (indent_dx <= max_indent)

                        # Set metadata if all guards pass (NO geometry mutation)
                        if same_block and allow_geom and not prev_hard_end and not curr_looks_new_sentence:
                            carry_tail["a2_continues_to_next"] = True

                            if prev_text.endswith("-"):
                                carry_tail["a2_continuation_mode"] = "hyphen_wrap"
                                carry_tail["a2_continuation_reason"] = "a2_vertical_hyphen"
                            else:
                                carry_tail["a2_continuation_mode"] = "space_join"
                                carry_tail["a2_continuation_reason"] = "a2_vertical_same_block"

                            vertical_link_count += 1

                        # CRITICAL: Clear carry_tail after processing to prevent unbounded chaining
                        # Each vertical link applies to exactly one line transition
                        carry_tail = None

                    # 2. Append new span if not merged
                    new_span_entry = {
                        "raw_text": raw_text,
                        "cleaned_text": None,
                        "bbox": bbox,
                        "bbox_is_valid": (bbox is not None),
                        "bbox_invalid_reason": bbox_invalid_reason,
                        "font_size": float(span.get("size", 0)),
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
                        "is_subscript": False,
                        "figure_index": None,
                    }

                    raw_spans.append(new_span_entry)
                    prev_span = new_span_entry

                except (KeyError, TypeError, IndexError):
                    skipped_count += 1
                    # HARDENED: Fallback for malformed spans
                    raw_spans.append({
                        "raw_text": span.get("text", ""),
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
            "[%s] Ingestion: emitted %d horizontal + %d vertical continuation links (a2_*)",
            trace_id, horizontal_link_count, vertical_link_count
        )

    return raw_spans

def _build_lines_from_spans(
        spans: List[Dict],
        trace_id: str = None
) -> Dict[str, Dict]:
    """
    PHASE 1: Build Line abstractions from flat span list.

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

    if trace_id:
        logger.debug(
            "[%s] Built %d line abstractions from %d spans",
            trace_id, len(lines), len(spans)
        )

    return lines


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
        # =====================================================================
        text_lower = text.lower()
        if text_lower in GREEK_WHITELIST or text in GREEK_WHITELIST:
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

        # FIXED v3.3: Use Global Constants for consistent Zone Definitions (15%)
        # Was: _LAYOUT_HEADER_THRESHOLD_Y (20%) / _LAYOUT_FOOTER_THRESHOLD_Y (80%)
        is_header_zone = norm_y < _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT
        is_footer_zone = norm_y > (1.0 - _GLOBAL_HEADER_FOOTER_EXCLUSION_PERCENT)

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
    Assign paragraph_index to spans.
    Phase A Hardened: Uses "Mixed Layout Protection" and global context.
    """
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
    _PARA_MAX_LINE_GAP = 50.0
    _PARA_DEFAULT_LINE_HEIGHT = 12.0
    _PARA_GAP_MULTIPLIER = 1.25

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
            # VISUAL GAP ZONES (THREE-TIER MODEL)
            # ------------------------------------------------------------------
            visual_break_threshold = median_line_height * 1.5
            borderline_threshold = para_gap_threshold * 1.25

            is_visual_break = gap > visual_break_threshold
            is_borderline = para_gap_threshold < gap <= borderline_threshold

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

            # ------------------------------------------------------------------
            # FINALIZE PARAGRAPH INDEX
            # ------------------------------------------------------------------
            if start_new_paragraph:
                global_para_idx += 1

            curr_span["paragraph_index"] = global_para_idx

            # PHASE 2.3 HARDEN: Track max bottom across same-line spans.
            # Prevents artificially large gaps when inline spans vary in height.
            if start_new_paragraph or not same_line:
                prev_bottom = curr_bbox[3]
            else:
                prev_bottom = max(prev_bottom, curr_bbox[3])

        # Advance paragraph index after each column
        global_para_idx += 1


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

            # HARDENED: Allow longer headings if visual signal is strong (Bold/Large)
            is_strong_visual = (is_large or is_bold)
            relaxed_length = word_count <= (_ROLE_HEADING_MAX_WORDS * 2.0)

            # GUARD: Headings should not start with lowercase (unless all caps)
            # This protects sentence continuations from being promoted
            starts_like_sentence = span_text and span_text[0].islower()
            if starts_like_sentence and not span_text.isupper():
                is_strong_visual = False  # Force strict checks or fail
            elif span_text.isupper() and word_count > _ROLE_HEADING_MAX_WORDS:
                is_strong_visual = False


            if (is_short or (
                    is_strong_visual and relaxed_length)) and is_title_case and no_terminal_punct:
                if is_strong_visual:
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
            elif is_italic and is_short and is_title_case and word_count >= 2:
                if is_structurally_independent:
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

            if is_monospace and "symbol" not in font_name and "wingdings" not in font_name:
                role = TextRole.CODE.value
            elif char_count > _ROLE_CODE_MIN_CHAR_COUNT:
                code_chars = _ROLE_CODE_PUNCT_PATTERN.findall(span_text)
                if len(code_chars) / char_count > _ROLE_CODE_PUNCT_RATIO:
                    role = TextRole.CODE.value

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

        previous_role = role
        previous_y = span_y
        previous_x = span_x
        previous_text = span_text

        if trace_id and role not in (TextRole.BODY.value, TextRole.EMPTY.value):
            logger.debug(
                "[%s] Role: '%s' -> %s (font=%.1f, baseline=%.1f)",
                trace_id, span_text[:25], role, font_size, baseline_font_size
            )


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

    # =========================================================================
    # GUARD: Ensure reading order before continuity analysis
    # Continuity checks use spans[i-1] / spans[i+1], so order is critical.
    # Sort by (page, column, block, line, span_in_line) for stable ordering.
    # =========================================================================
    spans.sort(key=lambda s: (
        s.get("page_number", 0),
        s.get("column_index", 0),
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

        # Decision: Override if in active flow
        in_active_flow = continues_from or continues_to

        if in_active_flow:
            # CONTINUITY WINS: Preserve stream integrity
            span["_continuity_override"] = True
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
    PHASE 2.0: Build a sliding window of spans for cross-page aware segmentation.

    Uses PRE-PASS cached spans to ensure all spans have identical viability
    processing (Phase 1.5 continuity, Phase 1.3 rescue, etc.).

    Window structure:
    - Last N spans from previous page (deep copied, tagged)
    - ALL spans from current page (deep copied, tagged with _page_local_idx)
    - First N spans from next page (deep copied, tagged)

    Args:
        page_span_cache: Dict mapping page_idx → processed spans_for_text
        current_page_idx: Index of current page being processed
        trace_id: Optional trace ID for logging

    Returns:
        Returns: Tuple of:
            - window_spans: Combined span list for the window (deep copies)
            - page_span_range: (start_idx, end_idx) marking current page in window


    Invariants:
        - All spans are deep copies (no mutation of cache)
        - Current page spans tagged with _page_local_idx for remapping
        - All spans tagged with _source_page_idx for attribution
    """
    import copy

    def _copy_span_for_window(span: Dict) -> Dict:
        """
        Phase 2 invariant: Window spans MUST be isolated from cached spans.

        We always deep-copy here to prevent:
          - cross-page mutation
          - tag leakage (_page_local_idx, _source_page_idx)
          - Heisenbugs during multi-pass stitching

        NOTE:
          Deep-copy cost is negligible because window size is bounded
          (prev_tail + current + next_head).
        """
        return copy.deepcopy(span)

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
    if current_page_idx > 0:
        prev_spans = page_span_cache.get(current_page_idx - 1, [])
        if prev_spans:
            tail_spans = prev_spans[-_WINDOW_TAIL_SPAN_COUNT:]
            for span in tail_spans:
                role = span.get("role", "")
                layout_stream = span.get("layout_stream", "")

                if (
                    role == TextRole.SIDEBAR.value
                    or (isinstance(layout_stream, str) and layout_stream.startswith("margin"))
                ):
                    if not span.get("_tts_promoted_to_body_stream", False):
                        continue

                span_copy = _copy_span_for_window(span)
                span_copy["_window_position"] = "prev_tail"
                span_copy["_source_page_idx"] = current_page_idx - 1
                window_spans.append(span_copy)
                prev_tail_count += 1

    # =========================================================================
    # CURRENT PAGE (deep copy, tagged with _page_local_idx)
    # =========================================================================
    page_start_idx = len(window_spans)

    for page_local_idx, span in enumerate(current_spans):
        role = span.get("role", "")
        layout_stream = span.get("layout_stream", "")

        if(
            role == TextRole.SIDEBAR.value
            or (isinstance(layout_stream, str) and layout_stream.startswith("margin"))
        ):
            if not span.get("_tts_promoted_to_body_stream", False):
                continue

        span_copy = _copy_span_for_window(span)
        span_copy["_window_position"] = "current"
        span_copy["_source_page_idx"] = current_page_idx
        span_copy["_page_local_idx"] = page_local_idx
        window_spans.append(span_copy)

    page_end_idx = len(window_spans)

    # =========================================================================
    # NEXT PAGE HEAD (deep copy, tagged)
    # =========================================================================
    next_head_count = 0
    if current_page_idx + 1 in page_span_cache:
        next_spans = page_span_cache.get(current_page_idx + 1, [])
        if next_spans:
            head_spans = next_spans[:_WINDOW_HEAD_SPAN_COUNT]
            for span in head_spans:
                role = span.get("role", "")
                layout_stream = span.get("layout_stream", "")

                if (
                    role == TextRole.SIDEBAR.value
                    or (isinstance(layout_stream, str) and layout_stream.startswith("margin"))
                ):
                    if not span.get("_tts_promoted_to_body_stream", False):
                        continue

                span_copy = _copy_span_for_window(span)
                span_copy["_window_position"] = "next_head"
                span_copy["_source_page_idx"] = current_page_idx + 1
                window_spans.append(span_copy)
                next_head_count += 1

    if trace_id:
        logger.debug(
            "[%s] Phase 2.0: Window for page %d: prev_tail=%d, current=%d, next_head=%d, total=%d",
            trace_id,
            current_page_num,
            prev_tail_count,
            page_end_idx - page_start_idx,
            next_head_count,
            len(window_spans)
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
        - span_start_index and span_end_index are page-local (not window)
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
        page_sent["span_start_index"] = page_local_start
        page_sent["span_end_index"] = max(page_local_start, page_local_end)
        page_sent["page_number"] = page_num

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
    sorted_bids = sorted(blocks.keys(), key=lambda b: blocks[b]["y0"])
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

    # ---- B) Single-pass scoring ----
    for span in spans:
        bbox = span.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        # Respect structural roles (don’t “override ground truth”)
        role = span.get("role", TextRole.BODY.value)
        if role in protected_roles:
            continue

        # Respect explicit protection flags if present in your pipeline
        if span.get("filter_protected") is True:
            continue

        # Text for semantic patterns
        text = (span.get("cleaned_text") or span.get("raw_text") or "").strip().lower()
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
        next_sent

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

    # Connector words that cannot terminate a sentence meaningfully
    CONNECTOR_WORDS = {
        "and", "or", "but", "nor", "so", "yet", "for",
        "to", "of", "in", "with", "by", "at", "from",
    }

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
                            stripped.lower() in CONNECTOR_WORDS and
                            trailing_punct
                    ):
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
                        span_gap: Optional[int] = None

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

    # REVISION B3-M: Deterministic continuation from A2 (with adjacency verification)
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
            fs = float(prev.get("font_size") or 10.0)
            # Allow gap up to ~3 lines (generous to account for A4 removing headers/footers)
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
        connector_words = {
            "and", "or", "of", "the", "a", "an", "in", "on",
            "at", "to", "for", "with", "by"
        }

        prev_ends_connector = (
                _last_word_norm in connector_words or
                prev_text.rstrip().endswith((",", ";", ":"))
        )

        if not prev_ends_connector:
            common_starters = {
                "The", "A", "An", "It", "This", "These", "Those",
                "He", "She", "They", "But", "However", "Therefore",
                "Furthermore", "In", "On", "Figure", "Table"
            }

            if head_first_word in common_starters:
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
    if has_lowercase_start:
        if is_structurally_adjacent:
            reason = "same_page:lowercase_start" if is_same_page else "lowercase_start"
            return True, reason

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

def _reconstruct_text_for_segmentation(
        spans: List[Dict],
        trace_id: str = None
) -> Tuple[str, List[Dict], List[int]]:
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
    # PHASE 2.5: Enforce reading order before reconstruction
    # CRITICAL: Use integer fields to avoid string sort bugs ("1:3:10" < "1:3:2")
    # =========================================================================
    # 1. Attach original index: (orig_idx, span)
    indexed_spans = list(enumerate(spans))

    # 2. Sort by geometry/structure using the span object (item[1])
    sorted_spans = sorted(indexed_spans, key=lambda item: (
        item[1].get('page_number', 0),
        item[1].get('column_index', 0),
        item[1].get('block_id', 0),
        item[1].get('line_index', 0),
        item[1].get('span_index_in_line', 0),
    ))

    full_text = ""
    span_map: List[Dict] = []
    char_to_span: List[int] = []

    # Track previous span for boundary detection
    prev_column: Optional[int] = None
    prev_para: Optional[int] = None
    prev_page: Optional[int] = None
    prev_text: str = ""

    # REVISION B: Initialize reference tracker for A2 signals
    prev_span_ref: Optional[Dict] = None

    a2_signal_joins = 0

    for _, (orig_idx, span) in enumerate(sorted_spans):
        if not isinstance(span, dict):
            continue

        # =====================================================================
        # PHASE 2.7 GUARD 0: Inline-Role Override (Chameleon Logic)
        # Rescues valid body text mislabeled as Sidebar/Subheading/Heading.
        # CRITICAL: Do NOT rescue actual margin content.
        # =====================================================================
        effective_role = span.get("role", "")
        if effective_role in {
            TextRole.SIDEBAR.value,
            TextRole.SUBHEADING.value,
            TextRole.HEADING.value,
        }:
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
        if effective_role in {
            TextRole.SIDEBAR.value,
            TextRole.FOOTNOTE.value,
            TextRole.TABLE_CELL.value,
            TextRole.INSIDE_FIGURE.value,
            TextRole.FIGURE_LABEL.value,
            TextRole.CODE.value,
            TextRole.HEADER_ARTIFACT.value,
        }:
            continue

        text = (span.get("cleaned_text") or "")
        if not text:
            continue

        curr_column = span.get("column_index", 0) or 0
        curr_para = span.get("paragraph_index", 0) or 0
        curr_page = span.get("page_number", 0)

        # Determine prefix based on boundaries
        prefix = ""
        if full_text:
            # =================================================================
            # PRIORITY 0 (PHASE 2.7): UNIVERSAL A2 OVERRIDE — SUPREME AUTHORITY
            # If upstream proved linguistic continuity, we MUST weld.
            # =================================================================
            if prev_span_ref and prev_span_ref.get("a2_continues_to_next", False):
                prefix = " "
                a2_signal_joins += 1

            # =================================================================
            # PRIORITY 1: Column change (same page) = hard break
            # =================================================================
            elif (
                    prev_column is not None and
                    curr_column != prev_column and
                    curr_page == prev_page
            ):
                prefix = "\n\n"

            # =================================================================
            # PRIORITY 2: Page change = space (preserve sentence flow)
            # =================================================================
            elif prev_page is not None and curr_page != prev_page:
                prefix = " "

            # =================================================================
            # PRIORITY 3: Paragraph change on same page = soft break
            # =================================================================
            elif prev_para is not None and curr_para != prev_para:
                prefix = "\n"

            # =================================================================
            # PRIORITY 4: Default join (no structural break)
            # =================================================================
            else:
                prefix = " "

        # GUARDRAIL: prefix must be structural whitespace only
        if prefix not in ("", " ", "\n", "\n\n"):
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
        })

        # Update tracking
        prev_column = curr_column
        prev_para = curr_para
        prev_page = curr_page
        prev_text = text

        # REVISION B: Track span for next iteration
        prev_span_ref = span

    if trace_id:
        logger.debug(
            "[%s] Reconstructed text: %d chars, %d spans, source map size: %d, "
            "a2_signal_joins: %d",
            trace_id, len(full_text), len(span_map), len(char_to_span),
            a2_signal_joins
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
    # LOGGING
    # =========================================================================
    if trace_id:
        methods: Dict[str, int] = {}
        for s in processed_sentences:
            m = s.get("alignment_method", "unknown")
            methods[m] = methods.get(m, 0) + 1
        logger.debug(
            "[%s] Segmentation: %d sentences from %d raw, %d empty skipped, "
            "alignment methods=%s, failures=%d",
            trace_id, len(processed_sentences), len(raw_sentences),
            empty_skipped, methods, alignment_failures
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

    for span in all_spans:
        # HARDENED: Only tag BODY spans as global header/footer.
        # Prevents clobbering validated HEADINGS, CAPTIONS, or TABLES.
        current_role = span.get("role", TextRole.BODY.value)
        if current_role != TextRole.BODY.value:
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

def _sanitize_for_tts(
        text: str,
        role: str = TextRole.BODY.value,
        add_terminal_punct: bool = True,
        change_tracker: Dict = None,
        trace_id: str = None
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

    Args:
        text: Sentence text to sanitize.
        role: Text role for role-aware noise removal (defaults to BODY).
        add_terminal_punct: Whether to enforce terminal punctuation.
        change_tracker: Optional mutation tracking dictionary.
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
            "global_median_font_size": global_median_font_size,  # Store for Stage 2
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
    page_span_cache: Dict[int, List[Dict]] = {}
    page_metadata_cache: Dict[int, Dict] = {}

    for page_idx, page_data in enumerate(raw_data_list):
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

        # =====================================================================
        # PHASE 3.0: Content-Flow Outlier Refinement
        # Detects single-page footers, attributions, isolated metadata that
        # escaped global band detection. Mutates roles to FOOTNOTE/HEADER_ARTIFACT.
        # Must run AFTER global bands, BEFORE role annotation loop.
        # =====================================================================

        _refine_roles_via_content_flow(filtered_spans, trace_id=trace_id)

        # PHASE 0: Track bbox-invalid spans for auditability
        bbox_invalid_spans = [s for s in filtered_spans if not s.get("bbox_is_valid", True)]
        page_data["bbox_invalid_spans"] = bbox_invalid_spans

        # PHASE 1: Preserve lossless text assembly while protecting geometry math
        spans_for_text = list(filtered_spans)
        spans_for_geometry = [s for s in filtered_spans if s.get("bbox_is_valid", True)]

        if not spans_for_text:
            continue

        # PHASE 1.5: Continuity-Aware Role Resolution (Stream-First Foundation)
        _apply_continuity_role_resolution(spans_for_text, trace_id=trace_id)

        # PHASE 0: Annotate non-viable spans instead of silently deleting them.
        excluded_by_role = []

        for span in spans_for_text:
            role = span.get("role", TextRole.BODY.value)

            if role in _TTS_NON_VIABLE_ROLES:
                span["_tts_excluded"] = True
                span["_tts_exclude_reason"] = role
                excluded_by_role.append(span)
            else:
                span["_tts_excluded"] = False
                span["_tts_exclude_reason"] = None

        # Record exclusions for audit/debug (lossless contract)
        page_data["excluded_by_role"] = excluded_by_role

        # =====================================================================
        # PHASE 1.3: Line-Aware Rescue Rail
        # Rescue excluded spans if they are on a line with viable BODY content.
        # Prevents inline fragments like "), and pain (" from being lost.
        # =====================================================================
        lines = _build_lines_from_spans(spans_for_text, trace_id)

        viable_line_ids = set()
        for line_id, line in lines.items():
            if any(
                    s.get("role") == TextRole.BODY.value and not s.get("_tts_excluded", False)
                    for s in line["spans"]
            ):
                viable_line_ids.add(line_id)

        rescued_count = 0
        for span in spans_for_text:
            if span.get("_tts_excluded", False):
                if span.get("line_id") in viable_line_ids:
                    span["_tts_excluded"] = False
                    span["_tts_rescued"] = True
                    span["_tts_rescue_reason"] = "line_has_body_content"
                    rescued_count += 1

                    # ===========================================================================
                    # PHASE 1.3 HARDEN: Document-Adaptive Inline Detection + Full Normalization (E3-v8)
                    # ===========================================================================
                    span_role = span.get("role", "")
                    span_layout = span.get("layout_stream", "")

                    is_margin_classified = (
                            span_role == TextRole.SIDEBAR.value or
                            (isinstance(span_layout, str) and span_layout.startswith("margin"))
                    )

                    # IMPORTANT: define unconditionally
                    is_traditional_inline = False
                    is_adaptive_inline = False
                    is_ratio_fallback_inline = False
                    detection_method = None

                    if is_margin_classified:
                        # Signal 1: traditional
                        is_traditional_inline = span.get("span_index_in_line", 0) > 0
                        if is_traditional_inline:
                            detection_method = "traditional"

                        # Signal 2/3: adaptive / ratio fallback (dominant stream, column-safe)
                        if not is_traditional_inline:
                            line_id = span.get("line_id")
                            line = lines.get(line_id) if isinstance(lines, dict) else None

                            if line and line.get("spans"):
                                body_spans_by_stream = {}
                                for s in line["spans"]:
                                    if (
                                            s is not span and
                                            s.get("role") == TextRole.BODY.value and
                                            not s.get("_tts_excluded", False)
                                    ):
                                        ls = s.get("layout_stream", "")
                                        if isinstance(ls, str) and ls.startswith("body_col"):
                                            body_spans_by_stream.setdefault(ls, []).append(s)

                                if body_spans_by_stream:
                                    dominant_stream = sorted(
                                        body_spans_by_stream.keys(),
                                        key=lambda k: (-len(body_spans_by_stream[k]),
                                                       _extract_column_number(k))
                                    )[0]

                                    body_spans = body_spans_by_stream[dominant_stream]

                                    span_bbox = span.get("bbox") or [0, 0, 0, 0]
                                    span_x0, span_x1 = span_bbox[0], span_bbox[2]

                                    xs = []
                                    for b in body_spans:
                                        bb = b.get("bbox") or [0, 0, 0, 0]
                                        xs.extend([bb[0], bb[2]])
                                    body_x_min, body_x_max = min(xs), max(xs)
                                    has_x_overlap = (span_x0 < body_x_max and span_x1 > body_x_min)

                                    nearest_dist = float("inf")
                                    for b in body_spans:
                                        b_bbox = b.get("bbox") or [0, 0, 0, 0]
                                        b_x0, b_x1 = b_bbox[0], b_bbox[2]
                                        if span_x1 <= b_x0:
                                            dist = b_x0 - span_x1
                                        elif span_x0 >= b_x1:
                                            dist = span_x0 - b_x1
                                        else:
                                            dist = 0.0
                                        nearest_dist = min(nearest_dist, dist)

                                    # Adaptive (2+ body peers): median gap * multiplier, BUT CAPPED by page-width ratio
                                    if len(body_spans) >= 2:
                                        body_spans_sorted = sorted(
                                            body_spans,
                                            key=lambda s: (s.get("bbox") or [0, 0, 0, 0])[0]
                                        )
                                        gaps = []
                                        for i in range(len(body_spans_sorted) - 1):
                                            left = body_spans_sorted[i].get("bbox") or [0, 0, 0, 0]
                                            right = body_spans_sorted[i + 1].get("bbox") or [0, 0,
                                                                                             0, 0]
                                            gap = right[0] - left[2]
                                            if gap > 0:
                                                gaps.append(gap)

                                        if gaps:
                                            gaps_sorted = sorted(gaps)
                                            median_gap = gaps_sorted[len(gaps_sorted) // 2]

                                            page_width = page_data.get("metadata", {}).get("width",
                                                                                           _LAYOUT_DEFAULT_PAGE_WIDTH)
                                            adaptive_threshold = min(
                                                median_gap * _HARDEN_GAP_TOLERANCE_MULTIPLIER,
                                                page_width * _HARDEN_MAX_INLINE_WIDTH_RATIO,
                                            )

                                            if has_x_overlap and nearest_dist <= adaptive_threshold:
                                                is_adaptive_inline = True
                                                detection_method = "adaptive"

                                    # Single-peer fallback: ratio-based threshold
                                    if (not is_adaptive_inline) and len(body_spans) == 1:
                                        page_width = page_data.get("metadata", {}).get("width",
                                                                                       _LAYOUT_DEFAULT_PAGE_WIDTH)
                                        ratio_threshold = page_width * _HARDEN_SINGLE_PEER_WIDTH_RATIO

                                        if has_x_overlap and nearest_dist <= ratio_threshold:
                                            is_ratio_fallback_inline = True
                                            detection_method = "ratio_fallback"

                    # --- NORMALIZATION (applied if any inline signal fires) ---
                    if is_margin_classified and (
                            is_traditional_inline or is_adaptive_inline or is_ratio_fallback_inline):
                        line_id = span.get("line_id")

                        body_peer_stream = None
                        body_peer_column = None
                        dominant_body_stream = None

                        try:
                            line = lines.get(line_id) if isinstance(lines, dict) else None
                            if line and line.get("spans"):
                                # Step 1: dominant body stream on this line
                                stream_counts = {}
                                for s in line["spans"]:
                                    if (
                                            s.get("role") == TextRole.BODY.value and
                                            not s.get("_tts_excluded", False)
                                    ):
                                        ls = s.get("layout_stream", "")
                                        if isinstance(ls, str) and ls.startswith("body_col"):
                                            stream_counts[ls] = stream_counts.get(ls, 0) + 1

                                if stream_counts:
                                    max_count = max(stream_counts.values())
                                    candidates = [k for k, v in stream_counts.items() if
                                                  v == max_count]
                                    dominant_body_stream = \
                                    sorted(candidates, key=_extract_column_number)[0]

                                # Step 2: v7 best-peer selection:
                                # dominant match > bbox distance > span_index_in_line distance
                                span_bbox = span.get("bbox") or [0, 0, 0, 0]
                                span_x0, span_x1 = span_bbox[0], span_bbox[2]
                                span_sil = span.get("span_index_in_line", 0)

                                best_peer = None
                                best_peer_matches_dominant = False
                                best_peer_box_distance = float("inf")
                                best_peer_sil_distance = float("inf")

                                for peer in line["spans"]:
                                    if (
                                            peer is not span and
                                            peer.get("role") == TextRole.BODY.value and
                                            not peer.get("_tts_excluded", False)
                                    ):
                                        ls = peer.get("layout_stream", "")
                                        if not (isinstance(ls, str) and ls.startswith("body_col")):
                                            continue
                                        if dominant_body_stream and ls != dominant_body_stream:
                                            continue

                                        peer_matches_dominant = (
                                                    ls == dominant_body_stream) if dominant_body_stream else False

                                        peer_bbox = peer.get("bbox") or [0, 0, 0, 0]
                                        peer_x0, peer_x1 = peer_bbox[0], peer_bbox[2]

                                        if span_x1 <= peer_x0:
                                            box_dist = peer_x0 - span_x1
                                        elif span_x0 >= peer_x1:
                                            box_dist = span_x0 - peer_x1
                                        else:
                                            box_dist = 0.0

                                        peer_sil = peer.get("span_index_in_line", 0)
                                        sil_dist = abs(peer_sil - span_sil)

                                        is_better = False
                                        if peer_matches_dominant and not best_peer_matches_dominant:
                                            is_better = True
                                        elif peer_matches_dominant == best_peer_matches_dominant:
                                            if box_dist < best_peer_box_distance:
                                                is_better = True
                                            elif box_dist == best_peer_box_distance:
                                                if sil_dist < best_peer_sil_distance:
                                                    is_better = True

                                        if is_better:
                                            best_peer = peer
                                            best_peer_matches_dominant = peer_matches_dominant
                                            best_peer_box_distance = box_dist
                                            best_peer_sil_distance = sil_dist

                                if best_peer:
                                    body_peer_stream = best_peer.get("layout_stream")
                                    body_peer_column = best_peer.get("column_index")

                        except Exception:
                            body_peer_stream = None
                            body_peer_column = None
                            dominant_body_stream = None

                        # Apply full normalization (NO hardcoded column fallback)
                        if body_peer_stream:
                            span["role"] = TextRole.BODY.value
                            span["layout_stream"] = body_peer_stream
                            span["is_margin_content"] = False

                            if body_peer_column is not None:
                                span["column_index"] = body_peer_column
                                span.pop("_tts_promotion_incomplete_reason", None)
                            else:
                                span[
                                    "_tts_promotion_incomplete_reason"] = "peer_column_index_unavailable"

                            span["_tts_promotion_reason"] = "inline_sidebar_promoted_to_body"
                            span["_tts_promoted_to_body_stream"] = True
                            span["_tts_inline_detection_method"] = detection_method

        if trace_id and rescued_count:
            logger.info(
                "[%s] Phase 1.3: Rescued %d inline spans via line coherence",
                trace_id, rescued_count
            )

        # Continue with filtering (rescued spans now included)
        spans_for_text = [s for s in spans_for_text if not s.get("_tts_excluded", False)]

        # Cache the processed spans for Pass 2
        page_span_cache[page_idx] = spans_for_text
        page_metadata_cache[page_idx] = {
            "page_num": page_num,
            "continuity": page_data.get("continuity", {}),
        }
    all_sentences: List[Dict] = []
    global_sentence_index = 0

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

        # Reconstruct text from window (cross-page aware)
        window_text, window_span_map, window_char_to_span = _reconstruct_text_for_segmentation(
            window_spans, trace_id
        )

        # Segment on full window text
        window_sentences = _segment_sentences(
            window_text, window_char_to_span, window_spans, trace_id
        )

        # Filter to sentences starting in current page, remap indices
        page_sentences = _filter_sentences_to_page(
            window_sentences, window_spans, page_span_range,
            page_idx, page_num, trace_id
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
            elif start_idx >= len(spans_for_text):
                if trace_id:
                    logger.warning(
                        "[%s] span_start_index %d >= len(spans_for_text) %d, defaulting to BODY",
                        trace_id, start_idx, len(spans_for_text)
                    )
                sent["role"] = TextRole.BODY.value
            else:
                role_from_span = spans_for_text[start_idx].get("role", TextRole.BODY.value)

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