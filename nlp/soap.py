# nlp/soap.py
from typing import List, Optional, Dict, Any
import re

from .utils import patient_lines, doctor_lines, extract_number_of_sessions, BODYPART_RX, title_case
from .ner import extract_entities

# Precompile regexes once at module import (faster than compiling per-call)
_RE_SEVERITY = re.compile(
    r'\b(severe|severely|constant|constantly|incapacit|unbearable|intense|debilitat|intractable)\b', re.I
)
_RE_IMPACT = re.compile(
    r'\b(trouble sleeping|had to take a week off|took a week off|took a week|off work|unable to|couldn\'t work|limited|regularly taking|regularly)\b',
    re.I
)
_RE_IMPROVING = re.compile(r'\b(improv|better|improving|resolved|now only|only occasional|occasional|getting better)\b', re.I)
_RE_ANALGESIC = re.compile(r'\b(painkill(?:er|ers)?|analgesic|ibuprofen|paracetamol|naproxen|aspirin)\b', re.I)

# Useful patterns to detect physical exam phrases (doctor findings)
_RE_PHYSICAL_FINDING = re.compile(
    r'\b(range of motion|full range|no tenderness|no sign of lasting damage|no signs of lasting|everything looks good|no evidence of damage|no tenderness)\b',
    re.I
)
_RE_EXAM_MENTION = re.compile(r'\b(exam|physical examination|physical exam)\b', re.I)

# Helper for chief complaint generation
def _short_chief_from_symptoms(symptoms_list: List[str]) -> Optional[str]:
    """
    Build concise chief complaint like:
      - "Neck pain"
      - "Neck and Back pain"
      - "Neck, Back and Head pain"
    Preserves the order they were detected.
    """
    if not symptoms_list:
        return None
    bodyparts = []
    seen = set()
    for s in symptoms_list:
        # strip possible duration parentheses e.g. "back pain (for four weeks)"
        core = re.sub(r'\s*\(.*\)\s*$', '', s).strip()
        m = re.search(BODYPART_RX, core, re.I)
        if m:
            bp = m.group(0).strip().capitalize()
        else:
            # fallback: first token
            bp = core.split()[0].capitalize()
        if bp not in seen:
            seen.add(bp)
            bodyparts.append(bp)
    if not bodyparts:
        return None
    if len(bodyparts) == 1:
        return f"{bodyparts[0]} pain"
    if len(bodyparts) == 2:
        return f"{bodyparts[0]} and {bodyparts[1]} pain"
    # 3+ parts -> "A, B and C pain"
    return f"{', '.join(bodyparts[:-1])} and {bodyparts[-1]} pain"

# Compute severity from a couple of signals
def _compute_severity(transcript: str, history: Optional[str]) -> str:
    t = (transcript or "")
    h = (history or "")
    has_severe = bool(_RE_SEVERITY.search(t))
    has_impact = bool(_RE_IMPACT.search(t)) or bool(_RE_IMPACT.search(h))
    has_improving = bool(_RE_IMPROVING.search(t)) or bool(_RE_IMPROVING.search(h))
    has_analgesic = bool(_RE_ANALGESIC.search(t)) or bool(_RE_ANALGESIC.search(h))
    if has_severe and not has_improving:
        return "Severe"
    if has_impact and has_improving:
        return "Mild-to-moderate (improving)"
    if has_impact or has_analgesic:
        return "Mild-to-moderate"
    if has_improving:
        return "Mild"
    return "Not specified"

def generate_soap_note(transcript: str) -> Dict[str, Any]:
    """
    Produce a rule-based SOAP note from a clinical transcript.

    Returns a dict with keys: Subjective, Objective, Assessment, Plan
    """
    # 1) Extract entities once (rules + optional HF NER inside)
    entities = extract_entities(transcript)
    symptoms = entities.get("Symptoms", []) or []

    # 2) History: prefer patient lines for HPI
    plines = patient_lines(transcript)
    history = " ".join(plines) if plines else None

    # 3) Chief complaint: derive succinct phrase from symptoms
    chief = _short_chief_from_symptoms(symptoms)

    # 4) Objective: find an explicit exam/finding line from doctor statements.
    physical_exam = None
    # scan each doctor line once; pick the best candidate
    for line in doctor_lines(transcript):
        text = line.strip()
        # ignore instruction lines like "Let's go ahead and do..."
        if re.search(r"\b(let's|lets|we will|we'll|go ahead|we should|we will do)\b", text, re.I):
            continue
        # prefer explicit findings
        if _RE_PHYSICAL_FINDING.search(text):
            physical_exam = re.sub(r'(?i)^(doctor|physician|dr)\s*[:\-]\s*', '', text).strip()
            break
    # fallback: pick any line mentioning 'exam' or 'physical examination'
    if physical_exam is None:
        for line in doctor_lines(transcript):
            if _RE_EXAM_MENTION.search(line) and not re.search(r"\b(let's|lets|go ahead)\b", line, re.I):
                physical_exam = re.sub(r'(?i)^(doctor|physician|dr)\s*[:\-]\s*', '', line).strip()
                break

    # 5) Assessment (diagnosis): prefer explicit diagnosis extracted; otherwise heuristic
    diagnosis = "Not specified"
    if entities.get("Diagnosis"):
        # already normalized in extractor
        diagnosis = entities["Diagnosis"][0].title()
    else:
        # heuristics on history
        if history and re.search(r'\bwhiplash\b', history, re.I):
            diagnosis = "Whiplash injury"
        elif history and re.search(r'\bneck\b|\bback\b|\bpain\b', history, re.I):
            diagnosis = "Post-accident musculoskeletal pain"

    # 6) Severity
    severity = _compute_severity(transcript, history)

    # 7) Plan: assemble recommended items based on treatments & keywords
    plan: List[str] = []
    treatments_list = entities.get("Treatment", []) or []

    # Use set membership to avoid rerunning regexes if possible
    treatments_lower = {t.lower() for t in treatments_list}

    # physiotherapy
    if any('physio' in t or 'physiotherapy' in t for t in treatments_lower) or re.search(r'\bphysio|physiotherapy\b', transcript, re.I):
        n_sessions = extract_number_of_sessions(transcript)
        if n_sessions:
            plan.append(f"Continue/monitor physiotherapy as needed ({n_sessions} sessions documented).")
        else:
            plan.append("Continue/monitor physiotherapy as needed.")

    # analgesics
    if any(re.search(r'\b(painkill(?:er|ers)?|analgesic|ibuprofen|paracetamol|naproxen|aspirin)\b', t, re.I) for t in treatments_list) \
       or re.search(r'\b(painkill(?:er|ers)?|analgesic|ibuprofen|paracetamol|naproxen|aspirin)\b', transcript, re.I):
        plan.append("Analgesics PRN for pain control.")

    # baseline follow-up guidance
    plan.append("Follow up if symptoms worsen or persist beyond expected recovery time.")

    # 8) Build SOAP structure
    soap = {
        "Subjective": {
            "Chief_Complaint": chief,
            "History_of_Present_Illness": history,
            "Ambiguities": entities.get("Ambiguities", []),
        },
        "Objective": {
            "Physical_Exam": physical_exam,
            "Observations": None
        },
        "Assessment": {
            "Diagnosis": diagnosis,
            "Severity": severity
        },
        "Plan": {
            "Treatment": plan,
            "Follow_Up": "Return if symptoms worsen or do not improve"
        }
    }
    return soap
