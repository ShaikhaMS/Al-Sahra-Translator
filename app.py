"""Al-Sahra Translator: live Transformer translation with a choice of models.

Run (macOS):
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    flask --app app run --port 5050
Then open http://127.0.0.1:5050
"""
import gc
import os
import re
import threading
import time
from collections import OrderedDict

from flask import Flask, jsonify, render_template, request

import metrics
from languages import (ISO_TO_NLLB, M2M100_CODES, MBART50_CODES, NLLB_LANGUAGES,
                       is_rtl)

try:  # optional: automatic language detection
    import langid
except ImportError:
    langid = None

# ---------------------------------------------------------------------------
# Transformers the user can choose from. "family" decides how languages are coded.
# ---------------------------------------------------------------------------
MODELS = OrderedDict([
    ("nllb-600m", dict(id="facebook/nllb-200-distilled-600M", family="nllb",
                       label="NLLB-200 (600M)", note="Meta · encoder–decoder Transformer · 200 languages · fastest · 2.5 GB download")),
    ("nllb-1.3b", dict(id="facebook/nllb-200-distilled-1.3B", family="nllb",
                       label="NLLB-200 (1.3B)", note="Meta · larger encoder–decoder Transformer · 200 languages · most accurate, slower · 5.5 GB download")),
    ("m2m-418m", dict(id="facebook/m2m100_418M", family="m2m",
                      label="M2M-100 (418M)", note="Meta · many-to-many Transformer · 100 languages · fast · 1.9 GB download")),
    ("mbart-50", dict(id="facebook/mbart-large-50-many-to-many-mmt", family="mbart",
                      label="mBART-50 (611M)", note="Meta · pre-trained denoising Transformer · 50 languages · 2.4 GB download")),
])
DEFAULT_MODEL = "nllb-600m"
# How many Transformers stay in memory at once (set SAHRA_MAX_LOADED=1 on an 8 GB Mac)
MAX_LOADED = int(os.environ.get("SAHRA_MAX_LOADED", "2"))
MAX_CHARS = 5000
app = Flask(__name__)


def model_code(family, code):
    """App language code (NLLB style) -> the code a model family expects."""
    if family == "m2m":
        return M2M100_CODES.get(code)
    if family == "mbart":
        return MBART50_CODES.get(code)
    return code


def lang_token(family, mcode):
    """Special token that tells the decoder which language to write."""
    return f"__{mcode}__" if family == "m2m" else mcode


class Translator:
    """One loaded model. Translates sentence by sentence and caches results,
    so live typing only re-translates the sentence that changed."""

    def __init__(self, key):
        self.key = key
        self.spec = MODELS[key]
        self.family = self.spec["family"]
        self.languages = {}
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.device = "cpu"

    def load(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        if torch.backends.mps.is_available():      # Apple Silicon GPU
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        # Half precision on a GPU: loads faster and uses half the memory
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.spec["id"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.spec["id"], torch_dtype=dtype, low_cpu_mem_usage=True).to(self.device).eval()
        vocab = self.tok.get_vocab()
        for code, name in NLLB_LANGUAGES.items():
            mcode = model_code(self.family, code)
            if mcode and lang_token(self.family, mcode) in vocab:
                self.languages[code] = name
        self.warm_up()

    def warm_up(self):
        """Run one tiny translation so the first real one is fast.
        Falls back to full precision if half precision misbehaves."""
        try:
            ok = self._generate(["Hello."], "eng_Latn", "fra_Latn", 1)[0].strip()
        except Exception:
            ok = ""
        if not ok and self.model.dtype != self.torch.float32:
            self.model = self.model.float()
            self._generate(["Hello."], "eng_Latn", "fra_Latn", 1)

    def _generate(self, segments, src, tgt, beams):
        s, t = model_code(self.family, src), model_code(self.family, tgt)
        self.tok.src_lang = s
        batch = self.tok(segments, return_tensors="pt", padding=True,
                         truncation=True, max_length=256).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **batch, num_beams=beams, max_new_tokens=256,
                forced_bos_token_id=self.tok.convert_tokens_to_ids(lang_token(self.family, t)))
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def translate(self, segments, src, tgt, beams=1):
        keys = [(s, src, tgt, beams) for s in segments]
        todo = list(dict.fromkeys(s for s, k in zip(segments, keys) if k not in self.cache))
        if todo:
            with self.lock:  # one generation at a time
                for s, t in zip(todo, self._generate(todo, src, tgt, beams)):
                    self.cache[(s, src, tgt, beams)] = t
            while len(self.cache) > 5000:
                self.cache.popitem(last=False)
        return [self.cache[k] for k in keys], len(todo)


class ModelManager:
    """Loads Transformers in the background and keeps the most recently used
    ones in memory, so switching back to one is instant."""

    def __init__(self, max_loaded=MAX_LOADED):
        self.max_loaded = max_loaded
        self.loaded = OrderedDict()   # key -> Translator, most recent last
        self.current = None           # key the user has selected
        self.loading = set()          # keys being loaded right now
        self.errors = {}
        self.lock = threading.Lock()

    def select(self, key):
        with self.lock:
            self.current = key
            self.errors.pop(key, None)
            if key in self.loaded:                # already in memory: instant
                self.loaded.move_to_end(key)
                return
            if key in self.loading:
                return
            self.loading.add(key)
        threading.Thread(target=self._load, args=(key,), daemon=True).start()

    def _load(self, key):
        try:
            t = Translator(key)
            t.load()
        except Exception as exc:  # shown on the page
            with self.lock:
                self.loading.discard(key)
                self.errors[key] = f"{type(exc).__name__}: {exc}"
            return
        with self.lock:
            self.loading.discard(key)
            self.loaded[key] = t
            if key != self.current:
                self.loaded.move_to_end(key, last=False)
            while len(self.loaded) > self.max_loaded:   # free the least recently used
                old = next(k for k in self.loaded if k != self.current)
                del self.loaded[old]
        gc.collect()

    @property
    def translator(self):
        return self.loaded.get(self.current)

    @property
    def ready(self):
        return self.current in self.loaded

    @property
    def error(self):
        return self.errors.get(self.current)


manager = ModelManager()

SENT_SPLIT = re.compile(r"(?<=[.!?؟۔。！？])\s+")


def split_text(text):
    """Lines -> sentences, keeping the line structure for the output."""
    return [[s for s in SENT_SPLIT.split(line.strip()) if s] for line in text.split("\n")]


def detect(text, languages):
    if langid is None or len(text.strip()) < 4:
        return None
    iso, _ = langid.classify(text)
    code = ISO_TO_NLLB.get(iso)
    return code if code in languages else None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def models():
    return jsonify(default=DEFAULT_MODEL,
                   models=[{"key": k, "label": m["label"], "note": m["note"]} for k, m in MODELS.items()])


@app.post("/api/model")
def choose_model():
    key = (request.get_json(force=True) or {}).get("model")
    if key not in MODELS:
        return jsonify(error=f"Unknown model '{key}'."), 400
    manager.select(key)
    return status()


@app.get("/api/status")
def status():
    t = manager.translator if manager.ready else None
    langs = sorted(({"code": c, "name": n, "rtl": is_rtl(c)} for c, n in t.languages.items()),
                   key=lambda x: x["name"]) if t else []
    return jsonify(model=manager.current, ready=manager.ready, loading=sorted(manager.loading),
                   loaded=list(manager.loaded), error=manager.error, languages=langs,
                   detection=langid is not None)


@app.post("/api/translate")
def api_translate():
    data = request.get_json(force=True) or {}
    if not manager.ready or data.get("model", manager.current) != manager.current:
        return jsonify(error=manager.error or "The Transformer is still loading."), 503
    t = manager.translator
    text = (data.get("text") or "")[:MAX_CHARS]
    src, tgt = data.get("source", "auto"), data.get("target", "arb_Arab")
    beams = 4 if data.get("quality") == "better" else 1
    detected = None
    if src == "auto":
        detected = detect(text, t.languages) or "eng_Latn"
        src = detected
    for code in (src, tgt):
        if code not in t.languages:
            return jsonify(error="This Transformer does not support the selected language."), 400
    lines = split_text(text)
    flat = [s for line in lines for s in line]
    start = time.perf_counter()
    if src == tgt or not flat:
        out, new = flat, 0
    else:
        out, new = t.translate(flat, src, tgt, beams)
    seconds = time.perf_counter() - start
    it = iter(out)
    translation = "\n".join(" ".join(next(it) for _ in line) for line in lines)
    return jsonify(translation=translation, detected=detected, source=src, rtl=is_rtl(tgt),
                   seconds=round(seconds, 3), sentences=len(flat), translated_now=new)


@app.post("/api/score")
def api_score():
    """BLEU and chrF of a translation against a reference (see metrics.py).
    Both are given one sentence per line, in the same order."""
    data = request.get_json(force=True) or {}
    hyps = [h.strip() for h in (data.get("hypothesis") or "").splitlines() if h.strip()]
    refs = [r.strip() for r in (data.get("reference") or "").splitlines() if r.strip()]
    if not hyps or len(hyps) != len(refs):
        return jsonify(error="Give one reference line for each translated line."), 400
    return jsonify(metrics.score(hyps, refs))


# Load the default model as soon as the server starts
manager.select(DEFAULT_MODEL)

if __name__ == "__main__":
    app.run(port=5050, debug=False)
