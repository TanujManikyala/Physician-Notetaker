# nlp/intent.py
from typing import List, Optional, Dict, Tuple, Any
import threading
import re
from functools import lru_cache
import hashlib

try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None  # will be checked later

# ---- Configuration ----
_DEFAULT_LABELS = [
    "Reporting symptoms",
    "Seeking reassurance",
    "Discussing treatment",
    "Asking about prognosis",
    "Reporting history",
    "Other"
]

# Heuristic keyword sets for fallback (lowercase)
_LABEL_KEYWORDS = {
    "Reporting symptoms": {"pain", "hurt", "ache", "backache", "headache", "neck", "stiffness", "soreness"},
    "Seeking reassurance": {"worried", "anxious", "nervous", "concerned", "reassure", "afraid"},
    "Discussing treatment": {"physio", "physiotherapy", "treatment", "sessions", "therapy", "medication"},
    "Asking about prognosis": {"when", "how long", "prognos", "recover", "recovery", "future", "outcome"},
    "Reporting history": {"last", "ago", "since", "history", "previous", "weeks", "months"},
    "Other": set()
}

# compiled phrase regex for multi-word checks (for speed)
_PHRASE_PATTERNS = {
    "Asking about prognosis": re.compile(r'\b(how long|when will|when can i|recover|recovery|prognos)\b', re.I),
    "Discussing treatment": re.compile(r'\b(physio|physiotherapy|treatment|sessions|therapy)\b', re.I),
    "Reporting symptoms": re.compile(r'\b(pain|ache|hurt|backache|headache|neck)\b', re.I)
}

# model lazy-init / cache
_classifier = None
_classifier_lock = threading.Lock()


def _init_hf_classifier():
    """
    Lazy initialize the HF classifier if possible. Thread-safe.
    """
    global _classifier
    if _classifier is not None:
        return _classifier
    if pipeline is None:
        _classifier = None
        return None
    with _classifier_lock:
        if _classifier is None:
            try:
                _classifier = pipeline("zero-shot-classification",
                                       model="facebook/bart-large-mnli",
                                       framework="pt")
            except Exception:
                _classifier = None
    return _classifier


# LRU cache for HF outputs (keeps last 256 queries)
@lru_cache(maxsize=256)
def _cached_hf_classify(text_hash: str, candidate_labels_key: Tuple[str, ...], multi_class: bool) -> Tuple[Any, ...]:
    """
    Internal helper: calls HF pipeline. text_hash is SHA256 of the text (to keep cache key small).
    We still need the raw text to pass to the pipeline; decode text from hash is impossible,
    so we call this via wrapper that provides raw text (see classify_intent below).
    This function is only memoizing by hash+labels; actual call is performed in wrapper.
    """
    # This function will not be called directly; wrapper will fill in via direct call below.
    raise RuntimeError("_cached_hf_classify should not be called directly")


def _make_text_hash(text: str) -> str:
    """Return a short SHA256 hex digest for caching keys."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _heuristic_intent(text: str, candidate_labels: List[str]) -> Dict[str, Any]:
    """
    Fast heuristic-based labeling when HF model is unavailable or disabled.
    - Tokenizes to a small set of lowercased alphanumeric tokens.
    - Also applies simple phrase regex checks for multi-word cues.
    """
    tl = text.lower()
    # fast token set (split on non-alpha)
    tokens = set(re.findall(r"[a-z]{2,30}", tl))

    labels_found: List[str] = []
    scores: List[float] = []

    for lbl in candidate_labels:
        lbl_score = 0.0
        # phrase pattern (if exists)
        pat = _PHRASE_PATTERNS.get(lbl)
        if pat and pat.search(tl):
            lbl_score = max(lbl_score, 0.9)
        # token matching
        kwset = _LABEL_KEYWORDS.get(lbl, set())
        if kwset and tokens & kwset:
            lbl_score = max(lbl_score, 0.85)
        if lbl_score > 0:
            labels_found.append(lbl)
            scores.append(round(lbl_score, 3))

    if not labels_found:
        # fallback: pick the most likely by counting token hits
        counts: Dict[str, int] = {}
        for lbl in candidate_labels:
            c = 0
            for w in _LABEL_KEYWORDS.get(lbl, ()):
                if w in tokens:
                    c += 1
            counts[lbl] = c
        # choose label with max count (or "Other")
        best_lbl = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        labels_found = [best_lbl]
        scores = [round(0.5 + counts[best_lbl] * 0.1, 3)]

    return {"sequence": text, "labels": labels_found, "scores": scores}


def classify_intent(text: str,
                    candidate_labels: Optional[List[str]] = None,
                    threshold: float = 0.28,
                    multi_class: bool = True,
                    use_hf_if_available: bool = True) -> Dict[str, Any]:
    """
    Classify intent using HF zero-shot model when available; otherwise use efficient heuristics.

    - text: input transcript (string)
    - candidate_labels: list of labels (defaults provided)
    - threshold: min score for including a label (HF path)
    - multi_class: passed to HF pipeline
    - use_hf_if_available: if False, forces heuristic path even if HF is installed.

    Returns: {"sequence": text, "labels": [...], "scores":[...]}
    """
    if candidate_labels is None:
        candidate_labels = _DEFAULT_LABELS

    if not text or not text.strip():
        return {"sequence": text, "labels": ["Other"], "scores": [0.0]}

    # If HF not requested or not installed -> use heuristics
    if not use_hf_if_available:
        return _heuristic_intent(text, candidate_labels)

    # Try to init HF classifier lazily
    clf = _init_hf_classifier()
    if clf is None:
        return _heuristic_intent(text, candidate_labels)

    # Use caching by hashing text + labels tuple
    text_hash = _make_text_hash(text)
    labels_key = tuple(candidate_labels)

    # We cannot directly use lru_cache with raw text (could be large) if we want a smaller key;
    # we'll keep a local in-memory cache mapping (text_hash, labels_key, multi_class) -> result.
    # But simplest: call the pipeline (heavy) and rely on HF's internal cache if any.
    try:
        result = clf(text, candidate_labels, multi_class=multi_class)
    except Exception:
        # If HF inference fails for any reason, fallback to heuristics
        return _heuristic_intent(text, candidate_labels)

    labels_out: List[str] = []
    scores_out: List[float] = []
    for lbl, sc in zip(result.get("labels", []), result.get("scores", [])):
        if sc >= threshold:
            labels_out.append(lbl)
            scores_out.append(round(float(sc), 3))

    if not labels_out and result.get("labels"):
        labels_out = [result["labels"][0]]
        scores_out = [round(float(result["scores"][0]), 3)]

    return {"sequence": result.get("sequence", text), "labels": labels_out, "scores": scores_out}
