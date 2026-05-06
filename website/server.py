from __future__ import annotations

import argparse
import html
import json
import math
import mimetypes
import re
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_A_PATH = MODEL_DIR / "best_model_a.joblib"
MODEL_B_PATH = MODEL_DIR / "best_model_b.joblib"
WORD2VEC_PATH = MODEL_DIR / "word2vec_150d.joblib"
FUSION_PATH = MODEL_DIR / "fusion_weights.json"

MODEL_A_FEATURES = [
    "num_urls",
    "num_emails",
    "num_ips",
    "num_html_tags",
    "num_exclamations",
    "num_questions",
    "uppercase_ratio",
    "total_words",
    "avg_word_length",
    "suspicious_word_count",
    "num_replies_forwards",
    "num_subdomains",
]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\S+@\S+")
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HTML_TAG_PATTERN = re.compile(r"<\s*(html|a\s|script|div)", re.IGNORECASE)
RE_FWD_PATTERN = re.compile(r"\b(?:RE|FWD):", re.IGNORECASE)
URL_HOST_PATTERN = re.compile(r"(?:https?://)?(?:www\.)?([^/\s:]+)", re.IGNORECASE)

ALLOWED_TEXT_CHARS_RE = re.compile(r"[^a-z0-9]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WORD_KEY_RE = re.compile(r"[^a-z0-9]+")
WHITESPACE_RE = re.compile(r"\s+")

SUSPICIOUS_WORDS = {
    "urgent",
    "verify",
    "click",
    "free",
    "winner",
    "update",
    "password",
    "confirm",
    "limited",
    "offer",
    "claim",
    "account",
    "suspended",
    "security",
    "login",
    "unlock",
    "congratulations",
}

_pipeline = None
_pipeline_lock = threading.Lock()


class _GensimCompat:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


def _call_on_class_only(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


def _install_gensim_compat_modules() -> None:
    """Provide enough class names to unpickle the saved Word2Vec vectors."""

    for loaded_name in list(sys.modules):
        if loaded_name == "gensim" or loaded_name.startswith("gensim."):
            del sys.modules[loaded_name]

    module_names = [
        "gensim",
        "gensim.models",
        "gensim.models.word2vec",
        "gensim.models.keyedvectors",
        "gensim.models.callbacks",
        "gensim.models.phrases",
        "gensim.models.doc2vec",
        "gensim.models.fasttext",
        "gensim.models.word2vec_inner",
        "gensim.models.fasttext_inner",
        "gensim.utils",
        "gensim.interfaces",
    ]

    for name in module_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    for name in module_names:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(sys.modules[parent_name], child_name, sys.modules[name])

    class_names = [
        "Word2Vec",
        "KeyedVectors",
        "Word2VecKeyedVectors",
        "Vocab",
        "CallbackAny2Vec",
        "SaveLoad",
        "BaseTopicModel",
        "TransformationABC",
        "Phrases",
        "FrozenPhrases",
        "Doc2Vec",
        "Doc2VecKeyedVectors",
        "FastText",
        "FastTextKeyedVectors",
    ]

    for module_name in module_names:
        module = sys.modules[module_name]
        for class_name in class_names:
            if not hasattr(module, class_name):
                setattr(
                    module,
                    class_name,
                    type(class_name, (_GensimCompat,), {"__module__": module_name}),
                )

    sys.modules["gensim.utils"].call_on_class_only = _call_on_class_only


def _load_word2vec(path: Path):
    try:
        return joblib.load(path)
    except ImportError as exc:
        exc_name = getattr(exc, "name", "") or ""
        if "gensim" not in exc_name and "gensim" not in str(exc).lower():
            raise
        _install_gensim_compat_modules()
        return joblib.load(path)


def _load_fusion_config() -> dict[str, float]:
    defaults = {"w_a": 0.2, "w_b": 0.8, "threshold": 0.5}
    if not FUSION_PATH.exists():
        return defaults

    with FUSION_PATH.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    config = defaults | {
        key: float(loaded[key])
        for key in defaults
        if key in loaded and _is_number(loaded[key])
    }

    total_weight = config["w_a"] + config["w_b"]
    if total_weight > 0:
        config["w_a"] = config["w_a"] / total_weight
        config["w_b"] = config["w_b"] / total_weight
    return config


def _is_number(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def get_pipeline() -> dict:
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        missing = [
            str(path.relative_to(PROJECT_ROOT))
            for path in [MODEL_A_PATH, MODEL_B_PATH, WORD2VEC_PATH, FUSION_PATH]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing model artifact(s): {', '.join(missing)}")

        model_a = joblib.load(MODEL_A_PATH)
        model_b = joblib.load(MODEL_B_PATH)
        word2vec = _load_word2vec(WORD2VEC_PATH)
        word_vectors = word2vec.wv

        _pipeline = {
            "model_a": model_a,
            "model_b": model_b,
            "word_vectors": word_vectors,
            "fusion": _load_fusion_config(),
            "vector_size": int(getattr(word_vectors, "vector_size", 150)),
        }
        return _pipeline


def safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def combine_subject_body(subject, body) -> str:
    subject_text = safe_text(subject).strip()
    body_text = safe_text(body).strip()
    if subject_text and body_text:
        return f"{subject_text}\n{body_text}"
    return subject_text or body_text


def count_subdomains(url_host: str) -> int:
    parts = url_host.split(".")
    return max(0, len(parts) - 2)


def extract_structural_features(text: str) -> dict[str, float]:
    text = str(text)
    urls = URL_PATTERN.findall(text)
    words = text.split()
    alpha_chars = [char for char in text if char.isalpha()]
    text_lower = text.lower()

    subdomain_total = 0
    for url in urls:
        host_match = URL_HOST_PATTERN.search(url)
        if host_match:
            subdomain_total += count_subdomains(host_match.group(1))

    return {
        "num_urls": len(urls),
        "num_emails": len(EMAIL_PATTERN.findall(text)),
        "num_ips": len(IP_PATTERN.findall(text)),
        "num_html_tags": len(HTML_TAG_PATTERN.findall(text)),
        "num_exclamations": text.count("!"),
        "num_questions": text.count("?"),
        "uppercase_ratio": sum(1 for char in alpha_chars if char.isupper()) / max(len(alpha_chars), 1),
        "total_words": len(words),
        "avg_word_length": sum(len(word) for word in words) / max(len(words), 1),
        "suspicious_word_count": sum(text_lower.count(word) for word in SUSPICIOUS_WORDS),
        "num_replies_forwards": len(RE_FWD_PATTERN.findall(text)),
        "num_subdomains": subdomain_total,
    }


def clean_email_text(text: str, remove_stopwords: bool = True) -> str:
    cleaned = html.unescape(safe_text(text))
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = cleaned.lower()
    cleaned = ALLOWED_TEXT_CHARS_RE.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()

    if not remove_stopwords or not cleaned:
        return cleaned

    kept_tokens = []
    for token in cleaned.split():
        word_key = WORD_KEY_RE.sub("", token)
        if word_key and word_key in ENGLISH_STOP_WORDS:
            continue
        kept_tokens.append(token)
    return " ".join(kept_tokens)


def document_vector(tokens: list[str], word_vectors, vector_size: int) -> np.ndarray:
    key_to_index = getattr(word_vectors, "key_to_index", {})
    vectors = getattr(word_vectors, "vectors")
    indices = [key_to_index[token] for token in tokens if token in key_to_index]

    if not indices:
        return np.zeros(vector_size, dtype=np.float32)

    return np.asarray(vectors[indices], dtype=np.float32).mean(axis=0)


def phishing_probability(model, feature_matrix: np.ndarray) -> float:
    probabilities = model.predict_proba(feature_matrix)[0]
    classes = list(getattr(model, "classes_", []))
    try:
        phishing_index = classes.index(1)
    except ValueError:
        phishing_index = len(probabilities) - 1
    return float(probabilities[phishing_index])


def build_reasons(features: dict[str, float], score_a: float, score_b: float, is_phishing: bool) -> list[str]:
    structural_reasons = []

    if features["num_urls"] > 0:
        structural_reasons.append(f"Found {features['num_urls']} link(s), which the structural model treats as important context.")
    if features["num_subdomains"] >= 2:
        structural_reasons.append("Detected deep subdomains, a common sign of link disguise.")
    if features["suspicious_word_count"] >= 2:
        structural_reasons.append("Found repeated security or account-pressure words.")
    if features["num_ips"] > 0:
        structural_reasons.append("Found a raw IP address, which is unusual in normal email links.")
    if features["num_html_tags"] > 0:
        structural_reasons.append("Found embedded HTML tags in the message text.")
    if features["uppercase_ratio"] >= 0.35 and features["total_words"] >= 5:
        structural_reasons.append("The message uses unusually heavy capitalization.")
    if features["num_exclamations"] + features["num_questions"] >= 3:
        structural_reasons.append("The message uses high-pressure punctuation.")

    if is_phishing:
        reasons = structural_reasons[:]
        if score_b >= 0.65:
            reasons.append("The text model sees language patterns close to known phishing messages.")
        if score_a >= 0.65:
            reasons.append("The structural model also returned a high phishing probability.")
        if not reasons:
            reasons.append("The combined model score crossed the saved decision threshold.")
        return reasons[:5]

    reasons = []
    if score_b <= 0.35:
        reasons.append("The text model sees low similarity to known phishing language.")
    if not structural_reasons:
        reasons.append("No links, IP addresses, HTML tags, or deep subdomains were found.")
    else:
        reasons.append("Some structural signals were present, but the fused score stayed below the saved threshold.")
    if score_a >= 0.65:
        reasons.append("Model A was elevated on structure, but the saved fusion weighting kept the final result legitimate.")

    return reasons[:5]


def analyze_email(subject: str, body: str) -> dict:
    pipeline = get_pipeline()
    raw_text = combine_subject_body(subject, body)
    cleaned_text = clean_email_text(raw_text)
    tokens = cleaned_text.split()

    features = extract_structural_features(raw_text)
    model_a_input = np.asarray([[features[name] for name in MODEL_A_FEATURES]], dtype=float)
    model_b_input = document_vector(tokens, pipeline["word_vectors"], pipeline["vector_size"]).reshape(1, -1)

    score_a = phishing_probability(pipeline["model_a"], model_a_input)
    score_b = phishing_probability(pipeline["model_b"], model_b_input)

    fusion = pipeline["fusion"]
    fusion_score = float((fusion["w_a"] * score_a) + (fusion["w_b"] * score_b))
    is_phishing = fusion_score >= fusion["threshold"]

    return {
        "decision": "phishing" if is_phishing else "legitimate",
        "scores": {
            "model_a": score_a,
            "model_b": score_b,
            "fusion": fusion_score,
            "threshold": fusion["threshold"],
        },
        "fusion": fusion,
        "features": features,
        "tokens_used": len(tokens),
        "reasons": build_reasons(features, score_a, score_b, is_phishing),
    }


def status_payload() -> dict:
    pipeline = get_pipeline()
    return {
        "loaded": True,
        "models_path": str(MODEL_DIR),
        "artifacts": {
            "model_a": MODEL_A_PATH.name,
            "model_b": MODEL_B_PATH.name,
            "word2vec": WORD2VEC_PATH.name,
            "fusion": FUSION_PATH.name,
        },
        "fusion": pipeline["fusion"],
        "model_a_features": MODEL_A_FEATURES,
        "model_b_vector_size": pipeline["vector_size"],
    }


class WebsiteHandler(BaseHTTPRequestHandler):
    server_version = "PhishingDetector/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(status_payload())
            return

        self._serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self._send_json({"error": "Unknown endpoint."}, status=404)
            return

        try:
            body = self._read_json()
            result = analyze_email(body.get("subject", ""), body.get("body", ""))
        except json.JSONDecodeError:
            self._send_json({"error": "Request body must be valid JSON."}, status=400)
            return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return

        self._send_json(result)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        return json.loads(raw_body or "{}")

    def _send_json(self, payload: dict, status: int = 200):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _serve_static(self, request_path: str):
        safe_path = unquote(request_path).lstrip("/")
        if not safe_path:
            safe_path = "index.html"

        file_path = (SITE_ROOT / safe_path).resolve()
        if not file_path.is_file() or SITE_ROOT not in file_path.parents and file_path != SITE_ROOT:
            self._send_json({"error": "File not found."}, status=404)
            return

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(host: str, port: int) -> None:
    get_pipeline()
    server = ThreadingHTTPServer((host, port), WebsiteHandler)
    print(f"Phishing detector running at http://{host}:{port}")
    print(f"Using model artifacts from {MODEL_DIR}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phishing detector website.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
