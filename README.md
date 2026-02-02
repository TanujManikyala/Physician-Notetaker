# Physician Notetaker — README

A lightweight Streamlit app that converts clinical transcripts into structured medical outputs (NER, keywords, summaries, sentiment, intent, SOAP notes).
This README explains how to set up the project using **one** `requirements.txt` (single requirements file), run the app, and resolve common environment issues.

---

## Quick summary

* **Repository entrypoint:** `app.py`
* **Single requirements file:** `requirements.txt` (one file only)
* **Python:** **3.10** or **3.11** recommended
* **Launch:** `streamlit run app.py`

---

## Project layout (important files)

```
.
├─ app.py
├─ requirements.txt        # single requirements file (see below)
├─ README.md               # this file
├─ nlp/
│  ├─ keywords.py
│  ├─ ner.py
│  ├─ sentiment.py
│  ├─ intent.py
│  ├─ soap.py
│  ├─ summarizer.py
│  └─ utils.py
└─ data/
   └─ sample_transcript.txt
```

---

## Single `requirements.txt`

**Place only one file** named `requirements.txt` at the project root. Use this single file for `pip install -r requirements.txt`.

Below is a recommended `requirements.txt` that enables full functionality (NER, zero-shot intent, KeyBERT keyword extraction). If you want a lighter install for CPU-only or to avoid heavy deps, see the *Light / CPU* section below.

**requirements.txt**

```text
streamlit>=1.20.0
transformers>=4.30.0
torch>=2.0.0        # CPU/GPU PyTorch (choose GPU wheel if you have CUDA)
sentence-transformers>=2.2.2
keybert>=0.7.0
scikit-learn>=1.0
numpy>=1.24
regex>=2023.6.3
# Optional: use tf-keras if using TensorFlow backend / TF integrations in transformers
# If you see Keras 3 incompatibility errors, install tf-keras:
# tf-keras>=2.13.1
```

> **Notes**
>
> * Keep only **this** file as your single `requirements.txt`. Remove any other `requirements-*.txt` files to avoid confusion.
> * Uncomment `tf-keras` line (remove `#`) **only** if you plan to use TF-backed transformers features and you get an error that mentions Keras/Tensorflow incompatibility.
> * `sentence-transformers` and `keybert` pull heavier dependencies. If you do not need KeyBERT, you can remove `sentence-transformers`/`keybert` from `requirements.txt` and the fallback keyword extractor will be used.

---

## Setup instructions

1. **Clone repo**

   ```bash
   git clone <your-repo-url>
   cd physician-notetaker
   ```

2. **Create a Python virtual environment**

   ```bash
   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # or Windows (cmd)
   .venv\Scripts\activate
   ```

3. **Install dependencies (single requirements file)**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

5. Open the URL printed by Streamlit (usually `http://localhost:8501`) in your browser.

---

## Light / CPU-only install (recommended on low-resource machines)

If you do **not** have GPUs and want to avoid heavy installs:

* Edit `requirements.txt` before installing and **remove** these packages:

  * `sentence-transformers`
  * `keybert`
* Keep `transformers` and `torch` only if you want model-backed features. You can also skip `transformers` and rely on rule-based fallbacks (the app supports fallbacks when HF models are not available).

Then run:

```bash
pip install -r requirements.txt
```

While running the app, **uncheck** `Use HF models (transformers/torch)` in the sidebar to force fallback, faster execution, and avoid transformer loading.

---

## Common issues & troubleshooting

### 1. `Keras 3` / `transformers` import error

Error message example:

```
Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers. Please install the backwards-compatible tf-keras package with `pip install tf-keras`.
```

**Fixes**

* Option A (recommended if you want TF integration):
  `pip install tf-keras` (and re-run `pip install -r requirements.txt` if you uncomment the tf-keras line).
* Option B (if you don't want TF): uncheck `Use HF models` in the Streamlit sidebar and restart the app. The app will use rule-based fallbacks and lighter code paths.

### 2. KeyBERT / sentence-transformers heavy install

* KeyBERT requires `sentence-transformers` and other heavy deps. If you experience installs failing or very long installs:

  * Remove `keybert` and `sentence-transformers` from `requirements.txt`.
  * The app will fall back to the built-in RAKE-like extractor in `nlp/keywords.py`.

### 3. `streamlit` cache errors (unserializable objects)

* If you cache heavy model objects or pipelines, use `st.cache_resource` (server state) instead of `st.cache_data`. The repo uses simple caching; altering caching behavior may be necessary if you persist transformer objects.

### 4. If the app is slow on CPU

* Uncheck `Use HF models (transformers/torch)` in the sidebar.
* Increase the `Max keywords` slider to reduce overhead? (higher number increases keyword scoring work).
* Consider running with a GPU-enabled environment for transformer workloads.

---

## Recommended development workflow

1. Create and activate venv.
2. Install `requirements.txt`.
3. Run `streamlit run app.py` and open the UI.
4. Paste your transcript in the UI and toggle stages (Entities, Keywords...).
5. For debugging the keyword extractor directly:

   ```bash
   python -c "from nlp.keywords import extract_keywords; print(extract_keywords(open('data/sample_transcript.txt').read(), top_n=8))"
   ```

---

## Developer / Submission notes

* Keep **only** one `requirements.txt` at the repo root (remove other `requirements-*.txt`).
* The `nlp` package contains pure-Python fallbacks so the app still works when `transformers` / `keybert` are not installed.
* If you plan to deploy to a server (Heroku/Streamlit Cloud), include the single `requirements.txt` and ensure the target environment has sufficient memory for optional heavy libs (or disable HF models).



---

## License & contact

* This project contains code snippets and utilities built for demonstration and clinical note drafting. Adjust for local policies and clinical governance before any production/clinical use.
* For questions or issues, add an issue in the repo or contact the project maintainer.

---