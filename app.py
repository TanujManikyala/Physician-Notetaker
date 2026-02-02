# app.py (fixed + optimized)
import streamlit as st
import hashlib
import time
from typing import Any, Dict, List, Tuple

st.set_page_config(page_title="Physician Notetaker", layout="wide")
st.title("🩺 Physician Notetaker")

# Sidebar controls
sidebar = st.sidebar
sidebar.header("Options / Performance")
max_keywords = sidebar.slider("Max keywords", 1, 20, 8)
use_hf_models = sidebar.checkbox("Use HF models (transformers/torch)", value=True)
st.sidebar.caption("Tip: uncheck HF models for faster runs on CPU or when transformers/torch are not installed.")

# Pipeline steps to run
st.sidebar.markdown("**Pipeline steps to run**")
DEFAULT_STEPS = ["Entities", "Summary", "Keywords", "Sentiment", "Intent", "SOAP"]
steps_selected = st.sidebar.multiselect("Select stages", DEFAULT_STEPS, default=DEFAULT_STEPS)

# Load sample text
try:
    with open("data/sample_transcript.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()
except Exception:
    sample_text = ""

transcript = st.text_area("Paste clinical transcript here:", value=sample_text, height=360)

# helper: transcript hash for logging/cache keys if needed
def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# RESOURCE: load heavy modules once (unserializable objects allowed)

@st.cache_resource(show_spinner=False)
def _load_nlp_modules(use_hf: bool) -> Dict[str, Any]:
    """
    Lazily import and return module objects. Cached as a resource (not pickled).
    """
    import importlib
    mods: Dict[str, Any] = {}
    # Import modules. Each module handles its own fallback if HF models are missing.
    mods["ner"] = importlib.import_module("nlp.ner")
    mods["summarizer"] = importlib.import_module("nlp.summarizer")
    mods["keywords"] = importlib.import_module("nlp.keywords")
    mods["sentiment"] = importlib.import_module("nlp.sentiment")
    mods["intent"] = importlib.import_module("nlp.intent")
    mods["soap"] = importlib.import_module("nlp.soap")
    return mods


# CACHED RESULTS (serializable outputs) - use st.cache_data

@st.cache_data(ttl=3600, show_spinner=False)
def cached_extract_entities(transcript: str, use_hf: bool) -> dict:
    mods = _load_nlp_modules(use_hf)
    try:
        return mods["ner"].extract_entities(transcript)
    except Exception as e:
        # safe fallback
        return {"error": f"Entity extraction failed: {e}"}

@st.cache_data(ttl=3600, show_spinner=False)
def cached_structured_summary(transcript: str, use_hf: bool) -> dict:
    mods = _load_nlp_modules(use_hf)
    try:
        return mods["summarizer"].generate_structured_summary(transcript, prefer_name_from_ner=True)
    except Exception as e:
        return {"error": f"Structured summary failed: {e}"}

@st.cache_data(ttl=3600, show_spinner=False)
def cached_text_summary(transcript: str, use_hf: bool) -> str:
    mods = _load_nlp_modules(use_hf)
    try:
        return mods["summarizer"].generate_text_summary(transcript)
    except Exception:
        return transcript[:500]

@st.cache_data(ttl=3600, show_spinner=False)
def cached_keywords(transcript: str, top_n: int, use_hf: bool) -> List[Tuple[str, float]]:
    mods = _load_nlp_modules(use_hf)
    try:
        return mods["keywords"].extract_keywords(transcript, top_n=top_n)
    except Exception:
        import re
        tokens = re.findall(r'\b(physiotherapy|physio|whiplash|pain|neck|back|accident|injury|sessions|painkillers)\b', transcript, re.I)
        uniq = []
        for t in tokens:
            if t.lower() not in [u.lower() for u in uniq]:
                uniq.append(t)
            if len(uniq) >= top_n:
                break
        return [(t, 1.0) for t in uniq]

@st.cache_data(ttl=3600, show_spinner=False)
def cached_sentiment(transcript: str, use_hf: bool) -> dict:
    mods = _load_nlp_modules(use_hf)
    try:
        return mods["sentiment"].analyze_sentiment(transcript)
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {e}"}

@st.cache_data(ttl=3600, show_spinner=False)
def cached_intent(transcript: str, candidate_labels: Tuple[str, ...], use_hf: bool) -> dict:
    mods = _load_nlp_modules(use_hf)
    try:
        return mods["intent"].classify_intent(transcript, candidate_labels=list(candidate_labels))
    except Exception as e:
        return {"error": f"Intent classification failed: {e}"}

@st.cache_data(ttl=3600, show_spinner=False)
def cached_soap(transcript: str, use_hf: bool) -> dict:
    mods = _load_nlp_modules(use_hf)
    try:
        return mods["soap"].generate_soap_note(transcript)
    except Exception as e:
        return {"error": f"SOAP generation failed: {e}"}


# UI: Run analysis when clicked

if st.button("Analyze") and transcript.strip():
    t0 = time.time()
    st.info("Running pipeline — running only selected steps. Results are cached for this transcript.")

    # Entities
    if "Entities" in steps_selected:
        with st.spinner("Extracting medical entities..."):
            entities = cached_extract_entities(transcript, use_hf_models)
        st.subheader("1) Extracted Medical Entities (NER + Rules)")
        st.json(entities)
    else:
        entities = None

    # Structured summary + short text
    if "Summary" in steps_selected:
        with st.spinner("Generating structured summary..."):
            structured = cached_structured_summary(transcript, use_hf_models)
        st.subheader("2) Structured Medical Summary (JSON)")
        st.json(structured)

        with st.expander("Short text summary"):
            text_summary = cached_text_summary(transcript, use_hf_models)
            st.write(text_summary)

    # Keywords
    if "Keywords" in steps_selected:
        with st.spinner("Extracting keywords..."):
            kws = cached_keywords(transcript, top_n=max_keywords, use_hf=use_hf_models)
        st.subheader("3) Keywords / Keyphrases")
        try:
            st.write([k for k, s in kws])
        except Exception:
            st.write(kws)

    # Sentiment
    if "Sentiment" in steps_selected:
        with st.spinner("Analyzing sentiment (patient tone)..."):
            sent = cached_sentiment(transcript, use_hf_models)
        st.subheader("4) Sentiment (Patient tone)")
        st.json(sent)

    # Intent
    if "Intent" in steps_selected:
        with st.spinner("Classifying intent (zero-shot)..."):
            labels = ("Reporting symptoms", "Seeking reassurance", "Discussing treatment",
                      "Asking about prognosis", "Reporting history", "Other")
            intent = cached_intent(transcript, labels, use_hf_models)
        st.subheader("5) Intent (zero-shot)")
        st.json(intent)

    # SOAP
    if "SOAP" in steps_selected:
        with st.spinner("Generating SOAP note..."):
            soap = cached_soap(transcript, use_hf_models)
        st.subheader("6) SOAP Note (Rule-based draft)")
        st.json(soap)

    ttotal = time.time() - t0
    st.success(f"Done — pipeline finished in {ttotal:.2f}s (cached results will make repeated runs fast).")
