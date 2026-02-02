# nlp/keywords.py
"""
Optimized clinical keyword/keyphrase extractor
------------------------------------------------
• Rule-first for high-precision clinical terms
• Lightweight RAKE-style scoring (no ML, no deps)
• Deterministic, CPU-safe
• Eliminates filler fragments ("and pain", "on pain", "don t")
• Public API unchanged

extract_keywords(text: str, top_n: int = 8) -> List[(phrase, score)]
"""

import re
from collections import Counter
from typing import List, Tuple

# -------------------- CONSTANTS --------------------

_STOPWORDS = {
    "the","and","is","in","it","of","to","a","i","that","was","for","on","with","at","by","an","be",
    "this","are","as","have","had","but","not","or","from","you","they","we","he","she","my","me",
    "so","if","then","after","before","your","yours","like","just","ok","okay","yeah","well",
    "thanks","thank","please","really","very","um","uh","got","get","going","went","dont","didnt",
    "cant","could","would","should","might","may","also","else","there","here","who","what","when",
    "where","how","any","some","thing","things","one","two","three","first","second","were",
    "anything","good","say","yes","no","did","do","does","feel","felt","ms","mr","mrs","doctor",
    "physician","patient","pt","im","ive","youre","its","oh","right"
}

_MEDICAL = {
    "whiplash","physiotherapy","physio","pain","neck","back","head","injury",
    "painkiller","painkillers","analgesic","backache","headache",
    "session","sessions","accident","emergency","a&e","seatbelt","steering"
}

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]", re.I)

# -------------------- NORMALIZATION --------------------

def _normalize(text: str) -> str:
    text = text.lower().replace("’", "").replace("'", "")
    return re.sub(r"\s+", " ", text).strip()

# -------------------- RULE EXTRACTION --------------------

def _rule_extract(text: str) -> List[Tuple[str, float]]:
    out = []

    if re.search(r"\bcar\b[^.]{0,60}\b(accident|hit|crash|rear)\b", text):
        out.append(("car accident", 1.0))

    if "whiplash" in text:
        out.append(("whiplash", 1.0))

    m = re.search(r"\b(\d+)\s+sessions?\s+(physiotherapy|physio)\b", text)
    if m:
        out.append((f"physiotherapy ({m.group(1)} sessions)", 1.0))
    elif "physiotherapy" in text or "physio" in text:
        out.append(("physiotherapy", 0.95))

    if re.search(r"\bneck\b[^.]{0,30}\b(pain|ache|stiff)", text):
        out.append(("neck pain", 0.95))

    if re.search(r"\bback\b[^.]{0,30}\b(pain|ache|stiff|backache)", text):
        out.append(("back pain", 0.95))

    if re.search(r"\b(hit|hit my)\b[^.]{0,20}\bhead\b", text) or "steering wheel" in text:
        out.append(("head injury", 0.9))

    if re.search(r"\bpainkiller|analgesic|ibuprofen|paracetamol\b", text):
        out.append(("painkillers", 0.9))

    if re.search(r"\baccident\s+and\s+emergency|a&e\b", text):
        out.append(("Accident and Emergency", 0.9))

    return out

# -------------------- PHRASE BUILDING (RAKE-LITE) --------------------

def _build_phrases(text: str) -> List[str]:
    text = _NON_WORD_RE.sub(" ", text)
    tokens = _TOKEN_RE.findall(text)

    phrases, curr = [], []
    for t in tokens:
        if t in _STOPWORDS:
            if curr:
                phrases.append(" ".join(curr))
                curr = []
        else:
            curr.append(t)
    if curr:
        phrases.append(" ".join(curr))

    seen, out = set(), []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            toks = p.split()
            if len(toks) == 1 and toks[0] not in _MEDICAL:
                continue
            out.append(p)
    return out

# -------------------- SCORING --------------------

def _score_phrases(phrases: List[str]) -> List[Tuple[str, float]]:
    freq = Counter()
    degree = Counter()

    for p in phrases:
        words = p.split()
        for w in words:
            freq[w] += 1
            degree[w] += len(words)

    scores = {}
    for w in freq:
        scores[w] = degree[w] / freq[w]

    ranked = []
    for p in phrases:
        words = p.split()
        s = sum(scores[w] for w in words)
        if any(w in _MEDICAL for w in words):
            s *= 1.6
        if len(words) > 1:
            s *= 1.2
        ranked.append((p, s))

    ranked.sort(key=lambda x: x[1], reverse=True)
    max_s = ranked[0][1] if ranked else 1.0
    return [(p, round(s / max_s, 3)) for p, s in ranked]

# -------------------- PUBLIC API --------------------

def extract_keywords(text: str, top_n: int = 8) -> List[Tuple[str, float]]:
    text = _normalize(text)
    if not text:
        return []

    rules = _rule_extract(text)
    phrases = _build_phrases(text)
    scored = _score_phrases(phrases)

    out, seen = [], set()

    for p, s in rules:
        k = p.lower()
        if k not in seen:
            out.append((p, s))
            seen.add(k)

    for p, s in scored:
        if len(out) >= top_n:
            break
        k = p.lower()
        if k in seen:
            continue
        out.append((p.title(), s))
        seen.add(k)

    return out[:top_n]
