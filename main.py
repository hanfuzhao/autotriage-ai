"""AutoTriage AI - interactive web app for NHTSA complaint component
classification. Loads the three trained models and serves inference only
(no training) behind a small Flask JSON API + single-page UI.

Endpoints:
    GET  /          -> the single-page app
    GET  /health    -> liveness probe (used by the keep-alive workflow)
    GET  /api/examples -> curated demo complaints
    POST /api/predict  -> {text, model} -> prediction, top-k, explanation, triage
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from scripts.model import ClassicalModel, NaiveBaseline, TextCNNModel

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------
MODELS: dict[str, object] = {}
LABELS: list[str] = []
DEPLOYED = "deep"

# Safety-criticality tier per component, used to colour the triage banner.
CRITICALITY = {
    "SERVICE BRAKES": "critical", "AIR BAGS": "critical", "STEERING": "critical",
    "SEATS/SEAT BELTS": "critical", "FUEL/PROPULSION SYSTEM": "critical",
    "VEHICLE SPEED CONTROL": "critical", "STRUCTURE": "critical",
    "POWER TRAIN": "high", "ELECTRICAL SYSTEM": "high", "ENGINE": "high",
    "DRIVER ASSISTANCE (ADAS)": "high",
    "EXTERIOR LIGHTING": "moderate", "VISIBILITY/WIPER": "moderate",
    "SUSPENSION": "moderate",
}

EXAMPLES = [
    {"title": "Sudden power loss",
     "text": "While driving on the highway at about 65 mph the engine suddenly lost power and the vehicle decelerated. The check engine light came on and the car nearly stalled in traffic before I could pull over."},
    {"title": "Brakes to the floor",
     "text": "When I pressed the brake pedal it went almost to the floor with very little stopping force. I had to pump the brakes several times to slow down. This has happened twice now, both times in stop and go traffic."},
    {"title": "Phantom braking",
     "text": "The automatic emergency braking activates for no reason on an empty road, slamming the brakes when there is no car or obstacle ahead. The forward collision warning also beeps constantly. Very dangerous on the freeway."},
    {"title": "Airbag warning light",
     "text": "The airbag warning light stays illuminated on the dashboard. The dealer says the passenger occupancy sensor is faulty and the airbag may not deploy in a crash. Waiting weeks for the part to arrive."},
    {"title": "Headlights flicker",
     "text": "The exterior headlights flicker and sometimes shut off completely while driving at night. Dealer replaced a bulb but the low beam still cuts out randomly, leaving me driving in the dark."},
    {"title": "Steering pulls hard",
     "text": "The electric power steering suddenly gets very heavy and the wheel pulls to the right. There is a loud clunk when turning at low speed and a warning appeared on the dash about the steering assist."},
]


def load_models() -> None:
    """Load whichever trained models are present; degrade gracefully."""
    global LABELS, DEPLOYED
    meta_path = MODELS_DIR / "labels.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        LABELS = meta.get("classes", [])
        DEPLOYED = meta.get("deployed_model", "deep")
    loaders = {
        "naive": (NaiveBaseline, "naive.pkl"),
        "classical": (ClassicalModel, "classical.pkl"),
        "deep": (TextCNNModel, "deep_textcnn.pt"),
    }
    for name, (cls, fname) in loaders.items():
        path = MODELS_DIR / fname
        if path.exists():
            try:
                MODELS[name] = cls.load(path)
            except Exception as err:  # noqa: BLE001
                app.logger.warning("could not load %s: %s", name, err)
    if DEPLOYED not in MODELS and MODELS:
        DEPLOYED = "deep" if "deep" in MODELS else next(iter(MODELS))


def _triage(label: str) -> dict:
    tier = CRITICALITY.get(label, "moderate")
    routing = {
        "critical": "Route to the safety-critical review queue for immediate triage.",
        "high": "Assign to the responsible powertrain/electronics engineering team.",
        "moderate": "Queue for standard component-team review.",
    }[tier]
    return {"tier": tier, "routing": routing}


def _predict_with(model, name: str, text: str) -> dict:
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    order = sorted(range(len(classes)), key=lambda i: proba[i], reverse=True)
    top_k = [{"label": classes[i], "prob": float(proba[i])} for i in order[:3]]
    result = {
        "model": name,
        "prediction": classes[order[0]],
        "confidence": float(proba[order[0]]),
        "top_k": top_k,
        "all_probs": {classes[i]: float(proba[i]) for i in order},
    }
    if name in ("classical", "deep") and hasattr(model, "explain"):
        try:
            result["highlights"] = [
                {"token": tok, "weight": round(w, 4)}
                for tok, w in model.explain(text, k=10) if w > 0
            ]
        except Exception:  # noqa: BLE001
            result["highlights"] = []
    return result


@app.route("/")
def index():
    return render_template("index.html", labels=LABELS, deployed=DEPLOYED,
                           n_models=len(MODELS))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": len(MODELS),
                    "deployed": DEPLOYED, "classes": len(LABELS)})


@app.route("/api/examples")
def examples():
    return jsonify(EXAMPLES)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    which = payload.get("model") or DEPLOYED
    if len(text) < 15:
        return jsonify({"error": "Please enter a longer complaint description (at least a sentence)."}), 400
    if not MODELS:
        return jsonify({"error": "No trained models available. Run `python setup.py` first."}), 503
    if which not in MODELS:
        which = DEPLOYED

    main_result = _predict_with(MODELS[which], which, text)
    main_result["triage"] = _triage(main_result["prediction"])
    # side-by-side snapshot of all three models
    compare = {
        name: {
            "prediction": (r := _predict_with(model, name, text))["prediction"],
            "confidence": r["confidence"],
        }
        for name, model in MODELS.items()
    }
    main_result["compare"] = compare
    return jsonify(main_result)


load_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
