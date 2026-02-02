# nlp/sentiment.py
"""
Optimized sentiment analyzer for clinical transcripts.
- Uses lightweight heuristics if HF pipeline unavailable.
- Optional HF model (distilbert finetuned SST-2) used when available.
- Results are cached for repeated identical inputs.
"""

from functools import lru_cache
import re
from typing import Dict, Optional

# local helpers
from .utils import patient_lines, doctor_lines

# Try lazy-import transformers pipeline (we will initialize lazily)
_pipeline_available = True
try:
    # don't import pipeline here at module import time to keep startup cheap
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None  # type: ignore
    _pipeline_available = False

# Precompile regexes (single place)
_REASSURE_RX = re.compile(
    r'\b(full recovery|no long[- ]term|no signs of lasting damage|on track for a full recovery|I don\'t foresee|no long[- ]term damage)\b',
    re.I
)
_IMPROVE_RX = re.compile(r'\b(improv|better|improving|now only|only occasional|occasional|getting better|resolved)\b', re.I)
_ANXIETY_RX = re.compile(r'\b(worried|anxious|nervous|concerned|scared|fear|panic)\b', re.I)
_PAIN_RX = re.compile(r'\b(pain|ache|backache|headache|stiff|stiffness|trouble sleeping|sleeping trouble|painkiller|analgesic)\b', re.I)

# Model config (lazy init and truncation)
_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
_MAX_CHARS_MODEL = 2000  # safe truncation threshold (approx. ~300-400 tokens) - tune as needed
_sentiment_pipe = None  # will hold pipeline instance when initialized


def _init_pipeline():
    """Lazily initialize HF pipeline; return pipeline or None (if unavailable)."""
    global _sentiment_pipe, pipeline, _pipeline_available
    if _sentiment_pipe is not None:
        return _sentiment_pipe
    if not _pipeline_available:
        return None
    try:
        # initialize once
        _sentiment_pipe = pipeline("sentiment-analysis", model=_MODEL_NAME, framework="pt")
        return _sentiment_pipe
    except Exception:
        _sentiment_pipe = None
        return None


@lru_cache(maxsize=2048)
def analyze_sentiment(text: str) -> Dict[str, Optional[object]]:
    """
    Analyze sentiment of clinical transcript.
    Returns: {"label": <mapped_label>, "orig_label": <HF label or None>, "score": <float 0..1>}
    Mapped labels: "Reassured", "Anxious", "Neutral", "UNKNOWN"
    """
    if not text:
        return {"label": "UNKNOWN", "orig_label": None, "score": 0.0}

    # Prepare doctor/patient lines once
    dtext = " ".join(doctor_lines(text))
    ptext = " ".join(patient_lines(text))

    # Heuristic detection (fast, linear)
    has_reassure = bool(_REASSURE_RX.search(dtext))
    has_improve = bool(_IMPROVE_RX.search(ptext) or _IMPROVE_RX.search(dtext))
    has_anxiety = bool(_ANXIETY_RX.search(ptext))
    has_pain = bool(_PAIN_RX.search(ptext))

    # If model not available, use deterministic heuristic mapping
    pipe = _init_pipeline()
    if pipe is None:
        # Heuristic mapping priority:
        # - If clinician explicitly reassures and patient not anxious -> Reassured
        if has_reassure and not has_anxiety:
            return {"label": "Reassured", "orig_label": None, "score": 0.90}
        # - If patient anxious or reports pain (and no clear improving) -> Anxious
        if has_anxiety or (has_pain and not has_improve):
            return {"label": "Anxious", "orig_label": None, "score": 0.85}
        # - If patient improving -> Reassured
        if has_improve:
            return {"label": "Reassured", "orig_label": None, "score": 0.80}
        # fallback Neutral
        return {"label": "Neutral", "orig_label": None, "score": 0.60}

    # Model path: call HF pipeline on a truncated version if needed
    model_input = text
    if len(model_input) > _MAX_CHARS_MODEL:
        # Keep patient and doctor lines prioritized: patient lines first (they carry affect)
        p = " ".join(patient_lines(text))
        d = " ".join(doctor_lines(text))
        # combine patient and doctor prioritized with join + truncation
        model_input = (p + " " + d)[:_MAX_CHARS_MODEL]

    try:
        out = pipe(model_input, truncation=True)
    except Exception:
        # If model fails at runtime, fallback to heuristics again
        if has_reassure and not has_anxiety:
            return {"label": "Reassured", "orig_label": None, "score": 0.90}
        if has_anxiety or (has_pain and not has_improve):
            return {"label": "Anxious", "orig_label": None, "score": 0.85}
        return {"label": "Neutral", "orig_label": None, "score": 0.60}

    if not out:
        return {"label": "UNKNOWN", "orig_label": None, "score": 0.0}

    top = out[0]
    orig_label = top.get("label")
    score = float(top.get("score", 0.0))

    # Map HF label -> domain-specific labels with clinician/patient context
    if has_reassure:
        mapped = "Anxious" if has_anxiety else "Reassured"
    else:
        if orig_label == "POSITIVE":
            mapped = "Reassured"
        elif orig_label == "NEGATIVE":
            if has_anxiety or has_pain:
                # If patient reports improvement, consider Reassured
                mapped = "Reassured" if has_improve else "Anxious"
            else:
                mapped = "Neutral"
        else:
            mapped = "Neutral"

    return {"label": mapped, "orig_label": orig_label, "score": round(score, 3)}
