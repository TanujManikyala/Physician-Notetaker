# nlp/ner.py
"""
Optimized rule-based medical entity extractor with optional HF NER context.
Public function:
    extract_entities(text: str) -> dict
"""

from typing import List, Tuple, Dict
import re
from functools import lru_cache

# Optional transformers usage - fallback if not available
try:
    from transformers import pipeline
except Exception:
    pipeline = None

from .utils import (
    strip_speaker_tags,
    extract_number_of_sessions,
    BODYPART_RX,
    SYMPTOM_KEYWORDS,
    normalize_treatment,
    find_duration_near,
    title_case,
)

# ---------- Configuration & precompiled regexes ----------
# Patterns (extendable)
DIAGNOSIS_PATTERNS = [r'whiplash', r'concussion', r'strain', r'sprain']
PROGNOSIS_PATTERNS = [r'full recovery', r'recovery expected', r'improving', r'no long[- ]term damage']
TREATMENT_TERMS = [
    r'physiotherapy', r'physio', r'physical therapy', r'painkillers', r'analgesic', r'analgesia',
    r'ibuprofen', r'paracetamol', r'naproxen', r'aspirin'
]

# Compile grouped regexes once
_DIAGNOSIS_RX = re.compile(r'\b(' + r'|'.join(DIAGNOSIS_PATTERNS) + r')\b', re.I)
_PROGNOSIS_RX = re.compile(r'\b(' + r'|'.join(PROGNOSIS_PATTERNS) + r')\b', re.I)
_TREATMENT_RX = re.compile(r'\b(' + r'|'.join([re.escape(t) for t in TREATMENT_TERMS]) + r')\b', re.I)

# Symptom patterns compiled
_MULTI_SYMPTOM_RX = re.compile(
    rf'\b({BODYPART_RX}(?:\s*(?:and|&|,)\s*{BODYPART_RX})+)\b[^.]*?\b{SYMPTOM_KEYWORDS}\b',
    re.I
)
_SINGLE_SYMPTOM_RX = re.compile(rf'\b{BODYPART_RX}\b[^.]*?\b{SYMPTOM_KEYWORDS}\b', re.I)
_BODYPART_RX = re.compile(BODYPART_RX, re.I)

# Route / A&E patterns
_AE_RX = re.compile(r'([A-Z][A-Za-z0-9 &\-]{2,60}?)\s+(Accident(?:\s+and\s+Emergency| and Emergency| & Emergency| & A&E|A&E))', re.I)
_ROUTE_RX = re.compile(r'\bfrom\s+([A-Z][A-Za-z0-9 \-]{2,60}?)\s+to\s+([A-Z][A-Za-z0-9 \-]{2,60}?)\b', re.I)

# HF NER initialization: lazy
_ner_pipe = None
if pipeline is not None:
    try:
        _ner_pipe = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True, framework="pt")
    except Exception:
        _ner_pipe = None


# ---------- helper functions ----------
def _merge_ner_tokens(ner_results) -> Tuple[List[str], List[str], List[str]]:
    """
    Merge token-level ner results into readable strings.
    Returns (orgs, persons, locs).
    """
    orgs: List[str] = []
    persons: List[str] = []
    locs: List[str] = []
    if not ner_results:
        return orgs, persons, locs

    merged = []
    cur_group = None
    cur_text = None

    for ent in ner_results:
        group = ent.get("entity_group") or ent.get("entity") or ent.get("label") or ""
        word_raw = ent.get("word") or ent.get("text") or ent.get("token_str") or ""
        if not word_raw:
            continue
        orig = word_raw
        # remove subword markers
        if orig.startswith("##"):
            chunk = orig.replace("##", "")
        else:
            chunk = orig.replace("▁", "").replace("Ġ", "")
        chunk = chunk.strip()
        if not chunk:
            continue
        if cur_group is not None and group == cur_group:
            # merge tokens
            if orig.startswith("##"):
                cur_text += chunk
            else:
                cur_text += " " + chunk
        else:
            if cur_group is not None and cur_text:
                merged.append((cur_group, cur_text.strip()))
            cur_group = group
            cur_text = chunk
    if cur_group is not None and cur_text:
        merged.append((cur_group, cur_text.strip()))

    # classify merged groups
    seen_orgs, seen_per, seen_locs = set(), set(), set()
    for g, text in merged:
        clean_text = re.sub(r'\s+', ' ', text).strip(" ,.;:")
        if not clean_text:
            continue
        g_up = (g or "").upper()
        if "ORG" in g_up and clean_text not in seen_orgs:
            orgs.append(clean_text); seen_orgs.add(clean_text)
        elif "PER" in g_up and clean_text not in seen_per:
            persons.append(clean_text); seen_per.add(clean_text)
        elif "LOC" in g_up and clean_text not in seen_locs:
            locs.append(clean_text); seen_locs.add(clean_text)
        else:
            # fallback heuristics
            if "ORG" in g_up and clean_text not in seen_orgs:
                orgs.append(clean_text); seen_orgs.add(clean_text)
            elif "LOC" in g_up and clean_text not in seen_locs:
                locs.append(clean_text); seen_locs.add(clean_text)
            elif "PER" in g_up and clean_text not in seen_per:
                persons.append(clean_text); seen_per.add(clean_text)
    return orgs, persons, locs


def _canonicalize_symptoms(symptoms: List[str]) -> List[str]:
    # prefer entries with durations, dedupe by bodypart preserving first-seen order
    seen: Dict[str, Dict] = {}
    order: List[str] = []
    for idx, s in enumerate(symptoms):
        if not s or not s.strip():
            continue
        s_norm = re.sub(r'\bhurt\b', 'pain', s.strip(), flags=re.I)
        s_norm = re.sub(r'\baches\b', 'ache', s_norm, flags=re.I)
        dur_m = re.search(r'\(([^)]+)\)', s_norm)
        dur = dur_m.group(1).strip() if dur_m else None
        bp_m = _BODYPART_RX.search(s_norm)
        bp = bp_m.group(0).lower() if bp_m else s_norm.split()[0].lower()
        prev = seen.get(bp)
        if not prev:
            seen[bp] = {"label": s_norm, "has_dur": bool(dur), "first_idx": idx, "dur": dur}
            order.append(bp)
        else:
            if (not prev["has_dur"]) and dur:
                seen[bp] = {"label": s_norm, "has_dur": True, "first_idx": prev["first_idx"], "dur": dur}
    out = []
    for bp in order:
        item = seen[bp]
        label = re.sub(r'\s*\(.*\)\s*$', '', item["label"]).strip()
        if item["has_dur"] and item["dur"]:
            out.append(f"{label} ({item['dur']})")
        else:
            out.append(label)
    return out


def _split_possible_combined_locations(locs: List[str]) -> List[str]:
    out: List[str] = []
    for loc in locs:
        parts = re.split(r'\b(?:\s+to\s+|,|\s*-\s*|/|\s+and\s+|\s+&\s+)\b', loc)
        for p in parts:
            p = p.strip()
            if p and p not in out:
                out.append(p)
    return out


# ---------- core rule-based extractors (regex-driven) ----------
def _find_symptoms_with_context(text: str) -> Tuple[List[str], List[dict]]:
    """Find bodypart+symptom phrases with optional duration and ambiguity flags."""
    found: List[str] = []
    ambiguities: List[dict] = []
    # 1) multi-bodypart phrases (e.g., "neck and back pain")
    for m in _MULTI_SYMPTOM_RX.finditer(text):
        parts_block = m.group(1)
        span = m.span()
        parts = re.findall(BODYPART_RX, parts_block, re.I)
        duration = find_duration_near(text, span[0], span[1])
        for bp in parts:
            bp_l = bp.lower()
            lab = f"{bp_l} pain"
            if duration:
                lab += f" ({duration})"
            if bp_l in ("front", "anterior"):
                ambiguities.append({"term": bp_l, "note": "unspecified anterior location; clarify"})
            if lab not in found:
                found.append(lab)

    # 2) single bodypart occurrences with symptom words nearby
    for m in _SINGLE_SYMPTOM_RX.finditer(text):
        span = m.span()
        bp_m = _BODYPART_RX.search(m.group(0))
        if not bp_m:
            continue
        bp = bp_m.group(0).lower()
        lab = f"{bp} pain"
        duration = find_duration_near(text, span[0], span[1])
        if duration:
            lab += f" ({duration})"
        if bp in ("front", "anterior"):
            ambiguities.append({"term": bp, "note": "unspecified anterior location; clarify"})
        if lab not in found:
            found.append(lab)

    # 3) fallback looser scan: "X hurt" or nearby pain words
    if not found:
        for m in re.finditer(r'\b' + BODYPART_RX + r'\b', text, re.I):
            bp = m.group(0).lower()
            ws = max(0, m.start() - 40)
            we = min(len(text), m.end() + 40)
            window = text[ws:we]
            if re.search(SYMPTOM_KEYWORDS, window, re.I):
                dur = find_duration_near(text, m.start(), m.end())
                lbl = f"{bp} pain"
                if dur:
                    lbl += f" ({dur})"
                if lbl not in found:
                    found.append(lbl)
                if bp in ("front", "anterior"):
                    ambiguities.append({"term": bp, "note": "unspecified anterior location; clarify"})

    final_symptoms = _canonicalize_symptoms(found)
    return final_symptoms, ambiguities


def _find_treatments(text: str) -> List[str]:
    found: List[str] = []
    # use compiled _TREATMENT_RX
    for m in _TREATMENT_RX.finditer(text):
        found.append(m.group(0))
    # explicit sessions
    n = extract_number_of_sessions(text)
    if n:
        found.append(f"{n} physiotherapy sessions")
    # normalize & dedupe while preserving order
    normalized: List[str] = []
    seen = set()
    for t in found:
        tt = normalize_treatment(t)
        if tt not in seen:
            seen.add(tt)
            normalized.append(tt)
    return normalized


def _find_diag_prognosis(text: str) -> Tuple[List[str], List[str]]:
    diagnoses: List[str] = []
    prognoses: List[str] = []
    # run compiled diagnosis/prognosis regexes
    for m in _DIAGNOSIS_RX.finditer(text):
        val = m.group(0).strip()
        if val not in diagnoses:
            diagnoses.append(val)
    for m in _PROGNOSIS_RX.finditer(text):
        val = m.group(0).strip()
        if val not in prognoses:
            prognoses.append(val)
    return diagnoses, prognoses


# ---------- main extractor with caching ----------
@lru_cache(maxsize=256)
def _extract_entities_cached(clean_text: str) -> dict:
    # internal function expects cleaned text (speaker tags removed)
    symptoms, ambiguities = _find_symptoms_with_context(clean_text)
    treatments = _find_treatments(clean_text)
    diagnoses, prognoses = _find_diag_prognosis(clean_text)

    orgs_raw, persons_raw, locs_raw = [], [], []
    if _ner_pipe:
        try:
            ner_results = _ner_pipe(clean_text)
            orgs_raw, persons_raw, locs_raw = _merge_ner_tokens(ner_results)
        except Exception:
            orgs_raw, persons_raw, locs_raw = [], [], []

    # augment orgs/locs via heuristics
    if locs_raw:
        locs_raw = _split_possible_combined_locations(locs_raw)

    m_ae = _AE_RX.search(clean_text)
    if m_ae:
        full = m_ae.group(0).strip()
        if full.lower() not in [o.lower() for o in orgs_raw]:
            orgs_raw.append(full)

    m_route = _ROUTE_RX.search(clean_text)
    if m_route:
        a = m_route.group(1).strip(); b = m_route.group(2).strip()
        if a and a not in locs_raw: locs_raw.append(a)
        if b and b not in locs_raw: locs_raw.append(b)

    # remove sentence-like fragments from orgs (e.g., "I went to Moss Bank ...")
    def _is_fragment(s: str) -> bool:
        return bool(re.match(r'^(i\s+went|i\s+was|went\s+to|i\s+went\s+to|i\s+have\s+been)\b', s.strip(), re.I))

    orgs_clean = [o for o in (orgs_raw or []) if o and not _is_fragment(o)]
    persons_clean = [p for p in (persons_raw or []) if p and p.strip()]
    locs_clean = [l for l in (locs_raw or []) if l and l.strip()]

    def _norm_titlecase_list(lst: List[str]) -> List[str]:
        out = []
        seen = set()
        for v in lst:
            v2 = re.sub(r'\s+', ' ', v.strip())
            if not v2:
                continue
            pretty = title_case(v2)
            key = pretty.lower()
            if key not in seen:
                seen.add(key)
                out.append(pretty)
        return out

    orgs = _norm_titlecase_list(orgs_clean)
    persons = _norm_titlecase_list(persons_clean)
    locs = _norm_titlecase_list(locs_clean)

    def _pretty_sym(s: str) -> str:
        m = re.search(r'^(.*?)\s*(\(.+\))\s*$', s)
        if m:
            return title_case(m.group(1).strip()) + " " + m.group(2)
        return title_case(s)

    symptoms_out = [_pretty_sym(s) for s in symptoms]
    treatments_out = [t for t in treatments]
    diagnoses_out = [d for d in diagnoses]
    prognoses_out = [p for p in prognoses]

    return {
        "Symptoms": symptoms_out,
        "Treatment": treatments_out,
        "Diagnosis": diagnoses_out,
        "Prognosis": prognoses_out,
        "Detected_Orgs": orgs,
        "Detected_Persons": persons,
        "Detected_Locations": locs,
        "Ambiguities": ambiguities
    }


def extract_entities(text: str) -> dict:
    """
    Public API: accepts raw transcript (with speaker tags) and returns structured entities.
    This function strips speaker tags, normalizes whitespace, and uses an LRU cache keyed
    by cleaned text for repeated calls.
    """
    if not text:
        return {
            "Symptoms": [], "Treatment": [], "Diagnosis": [], "Prognosis": [],
            "Detected_Orgs": [], "Detected_Persons": [], "Detected_Locations": [], "Ambiguities": []
        }
    clean = strip_speaker_tags(text).strip()
    if not clean:
        return {
            "Symptoms": [], "Treatment": [], "Diagnosis": [], "Prognosis": [],
            "Detected_Orgs": [], "Detected_Persons": [], "Detected_Locations": [], "Ambiguities": []
        }
    # use cached internal function
    return _extract_entities_cached(clean)
