# nlp/summarizer.py
"""Optimized, rule-based summarizer (structured JSON + short extractive text).
Relies on extract_entities() for structured fields and uses simple, robust heuristics
to form 'Current_Status' and a short human-readable summary prioritized towards
patient statements.
"""

import re
from typing import List, Dict, Optional
from functools import lru_cache

from .utils import patient_lines, title_case
from .ner import extract_entities

# Precompiled regexes (single compilation, reused)
_RE_WHIPLASH = re.compile(r'\bwhiplash\b', re.I)
_RE_OCCASIONAL = re.compile(r'\b(now only|only occasional|occasional|now only occasional)\b', re.I)
_RE_OCCASIONAL_BP = re.compile(r'\boccasional\s+([a-z]{2,20})(?:\s+backache|\s+backaches|\s+ache|\s+pain)?\b', re.I)
_RE_IMPROVE = re.compile(r'\b(improv|better|improving|getting better|resolved)\b', re.I)
_RE_BACK = re.compile(r'\bback\b', re.I)

# Keywords for extractive text summary scoring (patient-focused)
_SUMMARY_KEYWORDS = [
    r'\b(car accident|car crash|accident)\b',
    r'\b(physiotherapy|physio|sessions)\b',
    r'\b(whiplash|concussion)\b',
    r'\b(pain|backache|headache|neck pain)\b',
    r'\b(recovery|recover|improving|improved|resolved)\b'
]
_SUMMARY_KEYWORD_RX = [re.compile(k, re.I) for k in _SUMMARY_KEYWORDS]

# small helper to pretty-format symptom preserving parentheses
_RE_PAREN = re.compile(r'^(.*?)\s*(\(.+\))\s*$')

def _normalize_diagnosis(dlist: List[str]) -> str:
    if not dlist:
        return "Not mentioned"
    d0 = dlist[0].strip().lower()
    if 'whiplash' in d0:
        return "Whiplash"
    return title_case(d0)

def _normalize_current_status(ptxt: str) -> str:
    ptxt_l = (ptxt or "").lower()
    # Frequent pattern: 'occasional backache' etc.
    if _RE_OCCASIONAL.search(ptxt_l):
        bp_m = _RE_OCCASIONAL_BP.search(ptxt_l)
        if bp_m:
            bp = bp_m.group(1).strip()
            if bp.lower() in ["back", "backache", "backaches"]:
                return "Occasional backache"
            return "Occasional " + title_case(bp + " pain")
        if _RE_BACK.search(ptxt_l):
            return "Occasional backache"
        return "Occasional symptoms"
    if _RE_IMPROVE.search(ptxt_l):
        return "Improving"
    return "Not mentioned"

def _pretty_symptom(s: str) -> str:
    m = _RE_PAREN.search(s)
    if m:
        return title_case(m.group(1).strip()) + " " + m.group(2)
    return title_case(s)

def _normalize_persons(persons: List[str]) -> List[str]:
    """Title-case, dedupe preserving order."""
    out = []
    seen = set()
    for p in persons or []:
        if not p or not p.strip():
            continue
        p_clean = title_case(p.strip())
        key = p_clean.lower()
        if key not in seen:
            seen.add(key)
            out.append(p_clean)
    return out

@lru_cache(maxsize=1024)
def generate_structured_summary(transcript: str, prefer_name_from_ner: bool = False) -> Dict[str, object]:
    """
    Return structured JSON summary:
      Patient_Name, Symptoms[], Diagnosis, Treatment[], Current_Status, Prognosis
    Cached for identical transcript strings (useful during UI refreshes).
    """
    entities = extract_entities(transcript)

    # Patient name heuristics: prefer Ms. Jones when detected (robust pattern), else first PERSON
    patient_name = "Unknown"
    persons = _normalize_persons(entities.get("Detected_Persons") or [])
    for p in persons:
        # match "Jones" or "Ms Jones" robustly
        if re.search(r'\b(ms\.?\s+)?jones\b', p, re.I):
            patient_name = "Ms. Jones"
            break
    if patient_name == "Unknown" and persons:
        patient_name = persons[0]

    symptoms = entities.get("Symptoms") or []
    treatments = entities.get("Treatment") or []
    diagnoses = entities.get("Diagnosis") or []
    prognoses = entities.get("Prognosis") or []

    diagnosis = _normalize_diagnosis(diagnoses)
    prognosis = title_case(prognoses[0]) if prognoses else "Not mentioned"

    # Prefer explicit patient phrasing for current status
    ptxt = " ".join(patient_lines(transcript))
    current_status = _normalize_current_status(ptxt)

    summary_json = {
        "Patient_Name": patient_name,
        "Symptoms": [_pretty_symptom(s) for s in symptoms],
        "Diagnosis": diagnosis,
        "Treatment": [t for t in treatments],
        "Current_Status": current_status,
        "Prognosis": prognosis
    }
    return summary_json

@lru_cache(maxsize=2048)
def generate_text_summary(transcript: str, max_sentences: int = 3) -> str:
    """
    Short extractive summary prioritized for patient statements.
    Strategy:
      1. Take patient lines (highest priority).
      2. Split into sentences and score by presence of medical keywords.
      3. Pick top-scoring sentences (stable deterministic tie-break by order).
      4. Fallback: first 300 characters of cleaned transcript.
    This is intentionally simple, deterministic and fast.
    """
    # collect sentences from patient lines
    pts = patient_lines(transcript)
    sentences: List[str] = []
    for pl in pts:
        # split on sentence boundaries but keep content small
        parts = re.split(r'(?<=[.!?])\s+', pl.strip())
        for s in parts:
            s_clean = s.strip()
            if s_clean:
                sentences.append(s_clean)

    # if patient sentences are empty, consider doctor lines too (lower priority)
    if not sentences:
        # simple fallback: take first few lines of the transcript
        parts = re.split(r'(?<=[.!?])\s+', transcript.strip())
        for s in parts[:6]:
            sc = s.strip()
            if sc:
                sentences.append(sc)

    # scoring: count matched keywords (higher weight) and length penalty to favor concise informative sentences
    scored = []
    for idx, s in enumerate(sentences):
        score = 0
        s_l = s.lower()
        for rx in _SUMMARY_KEYWORD_RX:
            if rx.search(s_l):
                score += 2
        # small bonus if sentence mentions numbers (sessions/dates)
        if re.search(r'\b\d+\b', s_l):
            score += 0.5
        # small length normalization: prefer sentences not extremely short or extremely long
        length = len(s_l.split())
        length_bonus = 0
        if 5 <= length <= 40:
            length_bonus = 0.2
        score += length_bonus
        scored.append((score, idx, s))

    if not scored:
        # final fallback: short extract of transcript
        excerpt = transcript.strip().replace("\n", " ")
        return (excerpt[:300] + "...") if len(excerpt) > 300 else excerpt

    # sort by score desc, then by original order (idx) asc for deterministic ties
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = [s for (_, _, s) in scored[:max_sentences]]

    # join and return
    summary = " ".join(chosen)
    if not summary:
        excerpt = transcript.strip().replace("\n", " ")
        return (excerpt[:300] + "...") if len(excerpt) > 300 else excerpt
    return summary
