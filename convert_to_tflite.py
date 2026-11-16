"""
Convert the trained Keras model (models/weed_classifier.h5) to TensorFlow Lite for the mobile app,
so the phone can identify weeds with NO internet connection.

Run it once, in the same virtual environment as the model service (it needs TensorFlow):

    cd ml-service
    python convert_to_tflite.py                       # float16 weights (recommended)
    python convert_to_tflite.py --preprocess mobilenet   # if the model was trained with preprocess_input
    python convert_to_tflite.py --image path/to/weed.jpg # also compare against a real photo

It writes into ../mobile-app/assets/model/:
    weed_classifier.tflite   the on-device model
    model_meta.json          class names, input size, preprocessing (the app reads this)

IMPORTANT: --preprocess must match what the server uses (PREPROCESS in app.py, default "rescale").
"""
import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MODELS = HERE / "models"
IMG_SIZE = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "weed_classifier.h5"))
    ap.add_argument("--out", default=str(HERE.parent / "mobile-app" / "assets" / "model"))
    ap.add_argument("--preprocess", choices=["rescale", "mobilenet"], default="rescale",
                    help="rescale = pixels/255 ; mobilenet = pixels/127.5-1 (must match the server)")
    ap.add_argument("--quant", choices=["float16", "dynamic", "none"], default="float16",
                    help="float16: half-size, same accuracy. dynamic: smallest. none: full float32")
    ap.add_argument("--image", help="optional photo to compare Keras vs TFLite predictions on")
    ap.add_argument("--no-ml-copy", action="store_true", help="do not also copy the model into ml-service/models (used by the hosted model service)")
    args = ap.parse_args()

    import tensorflow as tf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    indices = json.loads((MODELS / "class_indices.json").read_text(encoding="utf-8"))
    by_index = {v: k for k, v in indices.items()}
    classes = [by_index[i] for i in range(len(by_index))]

    print("Loading Keras model …")
    model = tf.keras.models.load_model(args.model, compile=False)
    if model.output_shape[-1] != len(classes):
        sys.exit(f"Model has {model.output_shape[-1]} outputs but {len(classes)} classes")

    # Keras 3 → SavedModel → TFLite is the most reliable path
    with tempfile.TemporaryDirectory() as tmp:
        saved = str(Path(tmp) / "saved_model")
        try:
            model.export(saved)
            converter = tf.lite.TFLiteConverter.from_saved_model(saved)
        except Exception as exc:  # older TF/Keras without Model.export
            print("model.export failed (%s); converting directly" % exc)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if args.quant == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif args.quant == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Converting (%s) …" % args.quant)
        tflite_bytes = converter.convert()

    tflite_path = out / "weed_classifier.tflite"
    tflite_path.write_bytes(tflite_bytes)
    print("Wrote %s (%.1f MB)" % (tflite_path, len(tflite_bytes) / 1e6))
    if not args.no_ml_copy:
        (MODELS / "weed_classifier.tflite").write_bytes(tflite_bytes)
        print("Also copied to %s (used by the hosted model service)" % (MODELS / "weed_classifier.tflite"))

    # ---- verify: TFLite must agree with Keras
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    print("TFLite input %s %s, output %s" % (inp["shape"].tolist(), np.dtype(inp["dtype"]).name, outp["shape"].tolist()))

    def run_tflite(x):
        interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
        interp.invoke()
        return interp.get_tensor(outp["index"])[0]

    def scale(arr):
        return arr / 127.5 - 1.0 if args.preprocess == "mobilenet" else arr / 255.0

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        x = scale(rng.integers(0, 256, (1, IMG_SIZE, IMG_SIZE, 3)).astype("float32"))
        worst = max(worst, float(np.abs(model.predict(x, verbose=0)[0] - run_tflite(x)).max()))
    print("Max probability difference on random inputs: %.5f" % worst)
    if worst > 0.05:
        print("WARNING: difference is larger than expected; try --quant none")

    if args.image:
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        x = scale(np.expand_dims(np.asarray(img, dtype="float32"), 0))
        pk, pt = model.predict(x, verbose=0)[0], run_tflite(x)
        top = lambda p: [(classes[i], round(float(p[i]) * 100, 1)) for i in np.argsort(p)[::-1][:3]]
        print("Keras  top-3:", top(pk))
        print("TFLite top-3:", top(pt))

    meta = {
        "installed": True,
        "input_size": IMG_SIZE,
        "preprocess": args.preprocess,
        "quantization": args.quant,
        "classes": classes,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.no_ml_copy:
        (MODELS / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote model_meta.json. Commit ml-service/models/weed_classifier.tflite to deploy the model service; build the mobile app as in mobile-app/README.md.")


if __name__ == "__main__":
    main()
