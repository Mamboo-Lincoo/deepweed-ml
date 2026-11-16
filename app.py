"""
DeepWeed AI model service.

Implements steps 4-7 of the system flow:
  4 preprocess(image)  ->  5 predict(preprocessed)  ->  6 forwardPass()  ->  7 top3Species(conf)

Two interchangeable model backends (MODEL_BACKEND):
  keras   TensorFlow + models/weed_classifier.h5       (local development; needs ~1 GB RAM)
  tflite  LiteRT + models/weed_classifier.tflite       (light: fits Render's free 512 MB instances)
  auto    (default) keras if TensorFlow and the .h5 are present, otherwise tflite

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import hmac
import importlib.util
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger("deepweed-ml")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "models"))
MODEL_FILE = os.getenv("MODEL_FILE", "weed_classifier.h5")
TFLITE_FILE = os.getenv("TFLITE_FILE", "weed_classifier.tflite")
IMG_SIZE = (224, 224)  # from the model's input shape (None, 224, 224, 3)
API_KEY = os.getenv("ML_API_KEY", "")  # when set, /predict requires header  x-api-key: <ML_API_KEY>


def _meta_preprocess():
    try:
        return json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8")).get("preprocess")
    except Exception:
        return None


# The model has NO rescaling layer inside, so inputs must be scaled like during training:
#   "rescale"   -> pixels / 255        "mobilenet" -> pixels / 127.5 - 1
PREPROCESS = (os.getenv("PREPROCESS") or _meta_preprocess() or "rescale").lower()


class KerasRunner:
    name = "keras"

    def __init__(self):
        import tensorflow as tf  # imported here so the light deployment never needs TensorFlow

        path = MODEL_DIR / MODEL_FILE
        log.info("Loading Keras model %s ...", path)
        self.model = tf.keras.models.load_model(path, compile=False)
        self.n_out = self.model.output_shape[-1]

    def predict(self, x):
        return self.model.predict(x, verbose=0)[0]


class TfliteRunner:
    name = "tflite"

    def __init__(self):
        path = MODEL_DIR / TFLITE_FILE
        if not path.exists():
            raise RuntimeError(
                f"{path} not found. Run `python convert_to_tflite.py` once (see DEPLOY.md) and commit models/{TFLITE_FILE}."
            )
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:  # full TensorFlow installed instead
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
        log.info("Loading TFLite model %s ...", path)
        self.interp = Interpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.n_out = int(self.out["shape"][-1])

    def predict(self, x):
        self.interp.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"])[0]


def choose_backend() -> str:
    b = os.getenv("MODEL_BACKEND", "auto").lower()
    if b in ("keras", "tflite"):
        return b
    has_tf = importlib.util.find_spec("tensorflow") is not None
    return "keras" if has_tf and (MODEL_DIR / MODEL_FILE).exists() else "tflite"


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    indices = json.loads((MODEL_DIR / "class_indices.json").read_text(encoding="utf-8"))
    # class_indices.json is the authoritative index -> name mapping from training
    by_index = {v: k for k, v in indices.items()}
    classes = [by_index[i] for i in range(len(by_index))]

    runner = KerasRunner() if choose_backend() == "keras" else TfliteRunner()
    if runner.n_out != len(classes):
        raise RuntimeError(f"Model has {runner.n_out} outputs but {len(classes)} class names were found")

    runner.predict(np.zeros((1, *IMG_SIZE, 3), dtype="float32"))  # warm-up: first real request is fast
    state.update(runner=runner, classes=classes)
    log.info("Model ready: backend=%s classes=%d preprocess=%s", runner.name, len(classes), PREPROCESS)
    yield
    state.clear()


app = FastAPI(title="DeepWeed ML service", version="1.1.0", lifespan=lifespan)


def preprocess(raw: bytes) -> np.ndarray:
    """Step 4: decode, fix orientation, RGB, resize to 224x224, scale."""
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Unreadable image") from exc
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    x = np.asarray(img, dtype="float32")
    x = x / 127.5 - 1.0 if PREPROCESS == "mobilenet" else x / 255.0
    return np.expand_dims(x, 0)


@app.get("/health")
def health():
    ready = "runner" in state
    return {
        "status": "ok" if ready else "loading",
        "backend": state["runner"].name if ready else None,
        "classes": len(state.get("classes", [])),
        "preprocess": PREPROCESS,
    }


@app.post("/predict")
async def predict(request: Request):
    if API_KEY and not hmac.compare_digest(request.headers.get("x-api-key", ""), API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if "runner" not in state:
        raise HTTPException(status_code=503, detail="Model is still loading")
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty body")

    t0 = time.perf_counter()
    x = preprocess(raw)                                   # 4
    probs = state["runner"].predict(x)                    # 5/6 forward pass
    top = np.argsort(probs)[::-1][:3]                     # 7 top-3
    classes = state["classes"]
    return {
        "model": state["runner"].name,
        "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        "top3": [
            {
                "class_index": int(i),
                "class_name": classes[int(i)],
                "confidence": round(float(probs[i]) * 100, 2),
            }
            for i in top
        ],
    }
