# nlp/utils.py
"""
Optimized utility helpers for Physician Notetaker NLP pipeline.

Design goals:
- Precompile regexes once at import time to avoid repeated compilation.
- Use single-pass operations where possible.
- Provide BODYPART_RX (string) for compatibility with existing modules,
  while exposing compiled BODYPART_RE for internal fast matching.
"""

import re
from typing import List, Optional, Tuple

# Configuration / Constants

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20
}

# Anatomical terms (ordered longest -> shortest avoids partial matches)
BODY_PARTS = [
    "upper back", "lower back", "back",
    "head", "face", "eye", "ear", "nose", "mouth", "jaw",
    "neck",
    "chest", "breast", "sternum",
    "flank",
    "shoulder", "arm", "elbow", "wrist", "hand", "finger",
    "hip", "thigh", "knee", "leg", "ankle", "foot", "toe",
    "abdomen", "stomach", "belly", "groin",
    "pelvis", "throat",
    "front", "anterior", "posterior"
]

# main bodypart regex string (kept as string for other modules' expectations)
def _build_bodypart_pattern() -> str:
    parts = sorted(BODY_PARTS, key=lambda s: -len(s))
    escaped = [re.escape(p) for p in parts]
    return r'(' + r'|'.join(escaped) + r')'

BODYPART_RX = _build_bodypart_pattern()
# compiled for local fast use
BODYPART_RE = re.compile(BODYPART_RX, re.I)

SYMPTOM_KEYWORDS = r'(pain|ache|aches|hurt|hurting|soreness|discomfort|stiffness|tightness|tenderness|pressure)'
SYMPTOM_KEYWORDS_RE = re.compile(SYMPTOM_KEYWORDS, re.I)

# speaker line patterns (precompiled for speed)
_PATIENT_LINE_RE = re.compile(r'^(?:patient|pt)\s*[:\-]\s*(.+)$', re.I)
_DOCTOR_LINE_RE = re.compile(r'^(?:doctor|physician|dr)\s*[:\-]\s*(.+)$', re.I)
_SPEAKER_TAG_RE = re.compile(r'^(?:physician|doctor|dr|patient|pt)\s*[:\-]\s*', re.I)

# combined regex to extract sessions: matches either digits or written-number word followed by sessions
_SESSION_WORDS = r'(?:sessions|session|visits|visit)'
_SESSION_NUMBERWORDS = r'(' + '|'.join(map(re.escape, NUMBER_WORDS.keys())) + r')'
_SESSION_RE = re.compile(
    rf'\b(?:(\d{{1,3}})|{_SESSION_NUMBERWORDS})\s+(?:physio|physiotherapy|therapy)?\s*{_SESSION_WORDS}\b',
    re.I
)

# duration patterns used in find_duration_near
_DURATION_RE = re.compile(
    r'\b(for|since)\s+((\d+)|' + '|'.join(map(re.escape, NUMBER_WORDS.keys())) + r')\s+(day|days|week|weeks|month|months|year|years)\b',
    re.I
)
_SINCE_RE = re.compile(r'\b(?:from|since)\s+([A-Za-z0-9 ,]{3,60}?)\b', re.I)


# Basic line / speaker utils

def split_lines(text: str) -> List[str]:
    """
    Split text into non-empty trimmed lines preserving order.
    O(n) in text length.
    """
    if not text:
        return []
    # splitlines handles different newline styles efficiently
    return [line.strip() for line in text.splitlines() if line and line.strip()]


def patient_lines(text: str) -> List[str]:
    """
    Return the content of lines labeled 'Patient:' or 'Pt:' (case-insensitive).
    """
    out: List[str] = []
    for line in split_lines(text):
        m = _PATIENT_LINE_RE.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


def doctor_lines(text: str) -> List[str]:
    """
    Return content of lines labeled 'Doctor:' / 'Physician:' / 'Dr:' (case-insensitive).
    """
    out: List[str] = []
    for line in split_lines(text):
        m = _DOCTOR_LINE_RE.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


def strip_speaker_tags(text: str) -> str:
    """
    Remove leading speaker tags at the start of lines and return cleaned text (preserves newlines).
    This is linear in text length and avoids multiple regex compilations.
    """
    if not text:
        return ""
    cleaned_lines: List[str] = []
    for line in split_lines(text):
        cleaned_lines.append(_SPEAKER_TAG_RE.sub('', line).strip())
    return "\n".join(cleaned_lines)


# Number / duration helpers

def extract_number_of_sessions(text: str) -> Optional[int]:
    """
    Robustly extract a number of sessions from the transcript.
    Handles:
      - '10 sessions', 'ten sessions', '12 physiotherapy visits', 'had ten therapy sessions'
    Uses a single compiled regex for speed: O(n) time, O(1) additional space.
    """
    if not text:
        return None
    m = _SESSION_RE.search(text)
    if not m:
        return None
    # If numeric group (group 1) matched, return it; else return number-word mapping
    num_digits = m.group(1)
    if num_digits:
        try:
            return int(num_digits)
        except ValueError:
            return None
    num_word = m.group(2)
    if num_word:
        return NUMBER_WORDS.get(num_word.lower())
    return None


def find_duration_near(text: str, span_start: int, span_end: int, window_chars: int = 60) -> Optional[str]:
    """
    Search for duration expressions near a text span.
    - We extract a small window around the span and search two precompiled patterns.
    - window_chars is small and constant in practice, so this function is O(1) with respect to the transcript.
    """
    if not text:
        return None
    start = max(0, span_start - window_chars)
    end = min(len(text), span_end + window_chars)
    window = text[start:end]

    m = _DURATION_RE.search(window)
    if m:
        return m.group(0).strip()
    m2 = _SINCE_RE.search(window)
    if m2:
        # return the full match (e.g., "since last month" or "from January 5")
        return m2.group(0).strip()
    return None


# Small normalizers / formatters

_TREATMENT_MAP = {
    "physio": "physiotherapy",
    "pt": "physiotherapy"  # in case shorthand appears
}


def normalize_treatment(t: str) -> str:
    """
    Normalize treatment token strings to canonical forms.
    - lowercases, replaces common synonyms, strips whitespace.
    """
    if not t:
        return t
    s = t.strip().lower()
    for k, v in _TREATMENT_MAP.items():
        # word boundary replace to avoid partial matches
        s = re.sub(rf'\b{k}\b', v, s, flags=re.I)
    return s


def title_case(s: str) -> str:
    """
    Safe title-casing for display.
    Keeps numeric tokens and punctuation as-is; capitalizes alphabetic words.
    Not a full localization-aware function, but fast and deterministic.
    """
    if not s:
        return s
    parts = s.split()
    out_parts = []
    for w in parts:
        if w.isalpha():
            out_parts.append(w.capitalize())
        else:
            # try to keep acronyms and alphanumerics intact but capitalize purely alphabetic segment
            out_parts.append(w[0].upper() + w[1:] if w[0].isalpha() else w)
    return " ".join(out_parts)
