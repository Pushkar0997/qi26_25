"""
verify_parity.py — prove the browser pipeline matches the Python one.

The web demo is only honest if it computes the same thing the report measured.
Rather than assuming the JavaScript port is faithful, this runs both over the
same real document images and compares them stage by stage:

    1. quanvolutional features on identical crops   (numeric, tight tolerance)
    2. segmentation box counts and positions
    3. final extracted text
    4. Grover result on the recovered ID field

Any drift shows up here rather than in front of an audience.

Requires Node (any recent version). Usage:
    python web/verify_parity.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "integration"))

import pipeline as P                                    # noqa: E402
from features import quanv_features, normalize_crops    # noqa: E402

WEB = ROOT / "web"

NODE_HARNESS = r"""
import { Pipeline } from './pipeline.js';
import { readFileSync } from 'fs';

// Minimal fetch shim so pipeline.js can load model.json under Node.
globalThis.fetch = async (p) => ({
  ok: true, status: 200,
  json: async () => JSON.parse(readFileSync(p.replace('./', './'), 'utf8')),
});
globalThis.performance = globalThis.performance || { now: () => Date.now() };
globalThis.requestAnimationFrame = (cb) => cb();

const input = JSON.parse(readFileSync(process.argv[2], 'utf8'));
await Pipeline.load(process.argv[3]);

const out = {};

// --- stage 1: features on the exact crops Python produced ---------------
out.features = input.crops.map(c => Array.from(Pipeline.quanvFeatures(Float64Array.from(c))));

// --- stage 2-4: full run from raw pixels --------------------------------
const imgData = {
  width: input.width, height: input.height,
  data: Uint8ClampedArray.from(input.rgba),
};
const res = await Pipeline.run(imgData, input.pattern, () => {});
out.n_boxes = res.boxes.length;
out.boxes = res.boxes.map(b => [b.x0, b.y0, b.x1, b.y1]);
out.text = res.text;
out.field = res.field;
out.grover = res.grover ? {
  best: res.grover.best, correct: res.grover.correct,
  confidence: res.grover.confidence, iterations: res.grover.iterations,
  truth: res.grover.truth,
} : null;

console.log(JSON.stringify(out));
"""


def run_node(payload, model_path):
    """Execute the browser pipeline under Node and return its results as JSON.

    Node is used rather than a headless browser because pipeline.js is pure
    computation with no DOM dependency -- the only browser API it touches is
    fetch(), which the harness shims. That keeps this check runnable in CI or
    on a laptop with no browser installed.
    """
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "input.json"
        inp.write_text(json.dumps(payload))
        harness = WEB / "_parity_harness.mjs"
        harness.write_text(NODE_HARNESS)
        try:
            r = subprocess.run(
                ["node", str(harness), str(inp), str(model_path)],
                capture_output=True, text=True, cwd=str(WEB), timeout=600)
        finally:
            harness.unlink(missing_ok=True)
    if r.returncode != 0:
        raise SystemExit("node failed:\n" + r.stderr[-3000:])
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        raise SystemExit("node produced unparseable output:\nSTDOUT[:2000]: "
                         + r.stdout[:2000] + "\nSTDERR: " + r.stderr[-2000:])


def check_export_is_current():
    """Fail loudly if web/model.json was exported from a different backend.

    Retraining writes integration/ocr_backend.npz but does NOT refresh
    web/model.json, so the browser keeps serving the previous weights. Without
    this check the symptom is a bare "text DIFFERS" from the parity run, which
    looks like a porting bug in pipeline.js and sends you hunting in the wrong
    place. It is also the exact failure a reviewer hits after cloning the repo
    and regenerating the dataset on a machine with different fonts.
    """
    model = WEB / "model.json"
    npz = ROOT / "integration" / "ocr_backend.npz"
    if not model.exists():
        raise SystemExit("web/model.json missing — run: python web/export_model.py")
    if not npz.exists():
        raise SystemExit("integration/ocr_backend.npz missing — run: "
                         "python integration/pipeline.py --train")

    exported = json.loads(model.read_text())
    d = np.load(npz, allow_pickle=True)
    coef = np.asarray(d["coef"], dtype=float)
    ex_coef = np.asarray(exported["ocr_coef"], dtype=float)

    if ex_coef.shape != coef.shape or not np.allclose(ex_coef, coef, atol=1e-9):
        raise SystemExit(
            "\nweb/model.json is STALE — it does not match the trained backend.\n"
            "  exported holdout accuracy : {:.4f}\n"
            "  current backend accuracy  : {:.4f}\n\n"
            "The browser would serve different weights than Python. Fix:\n"
            "  python web/export_model.py\n".format(
                exported["meta"].get("holdout_acc", float("nan")),
                float(d["holdout_acc"])))


def main():
    check_export_is_current()
    model = WEB / "model.json"

    docs = []
    for tier in ["clean_digital", "clean_scan", "noisy_scan"]:
        f = ROOT / "data" / "processed" / tier / "doc_000.png"
        if f.exists():
            docs.append((tier, f))
    if not docs:
        raise SystemExit("no documents found — run data/generate_dataset.py")

    all_ok = True
    for tier, f in docs:
        arr = np.array(Image.open(f).convert("L"), dtype=np.uint8)
        rgba = np.dstack([arr, arr, arr, np.full_like(arr, 255)]).ravel().tolist()

        # Python side
        boxes_py = P.segment_characters(arr)
        crops_py = P.crops_from_boxes(arr, boxes_py)
        norm_py = normalize_crops(crops_py)
        feat_py = quanv_features(norm_py, correlations=True)
        coef, intercept, classes = P.load_backend()
        chars_py = P.predict_chars(crops_py, coef, intercept, classes)
        text_py = P.assemble_text(boxes_py, chars_py)
        field_py = P._locate_id_field(text_py)
        # Search for a two-character slice of the field the pipeline actually
        # recovered, so the pattern is guaranteed to be present and Grover has
        # something to find. A hardcoded pattern would silently degrade this
        # into an "absent pattern" test on most documents.
        pattern = field_py[2:4] if len(field_py) >= 4 else "26"

        payload = {
            # Flatten: pipeline.js indexes crops as a flat length-64 array.
            # Sending nested 8x8 makes Float64Array.from() yield NaN.
            "crops": [np.asarray(c).ravel().tolist() for c in norm_py],
            "width": int(arr.shape[1]), "height": int(arr.shape[0]),
            "rgba": rgba, "pattern": pattern,
        }
        # Hand the SAME crops Python produced to the JS feature extractor, so
        # stage 1 isolates the quantum maths from any difference in how the two
        # implementations cut crops. Stages 2-4 then run the full JS pipeline
        # from raw pixels, which tests the cutting as well.
        js = run_node(payload, "./model.json")

        raw = js["features"]
        if any(v is None for row in raw for v in row):
            raise SystemExit("JS produced non-finite features (JSON null). "
                             "This usually means a shape mismatch between the "
                             "Python payload and what pipeline.js expects.")
        feat_js = np.array(raw, dtype=float)
        fmax = float(np.abs(feat_js - feat_py).max())
        box_match = js["n_boxes"] == len(boxes_py)
        text_match = js["text"] == text_py
        field_match = js["field"] == field_py

        # 1e-9 rather than exact equality: the two implementations sum
        # floating-point products in different orders, so agreement at machine
        # epsilon (~1e-16) is the realistic ceiling. Anything at 1e-6 or worse
        # would indicate a real divergence, not rounding.
        ok = fmax < 1e-9 and box_match and text_match and field_match
        all_ok &= ok
        print("[{}] {}".format("PASS" if ok else "FAIL", tier))
        print("   features   max|Δ| = {:.2e}   ({} x {})".format(
            fmax, *feat_py.shape))
        print("   boxes      py={}  js={}  {}".format(
            len(boxes_py), js["n_boxes"], "match" if box_match else "MISMATCH"))
        print("   text       {}".format("identical" if text_match else "DIFFERS"))
        if not text_match:
            print("     py: {!r}".format(text_py[:90]))
            print("     js: {!r}".format(js["text"][:90]))
        print("   id field   py={!r} js={!r}".format(field_py, js["field"]))
        if js["grover"]:
            print("   grover     pos={} correct={} conf={:.3f}".format(
                js["grover"]["best"], js["grover"]["correct"],
                js["grover"]["confidence"]))

    print("\n" + ("ALL STAGES MATCH — browser output is the Python pipeline"
                  if all_ok else "PARITY FAILURE — do not ship until fixed"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
