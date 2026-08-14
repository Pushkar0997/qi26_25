"""
export_model.py — export the trained pipeline to plain JSON for the web demo.

WHY THIS EXISTS
---------------
The browser demo must reproduce the Python pipeline exactly, or the site is
showing something other than what the report measured. Rather than
reimplementing the model in JavaScript from the paper, every learned or fixed
numeric artifact is exported here and consumed verbatim by the JS, so there is
one source of truth.

What gets exported:
  filter_unitary   16x16 complex matrix (real/imag), the quanvolutional filter
  ocr_coef         38x160 logistic-regression weights
  ocr_intercept    38 biases
  classes          the 38 character labels, in coef row order
  constants        crop size, patch size, stride, charset

What the browser does NOT need:
  Qiskit          the filter is a fixed unitary; applying it is a matmul
  scikit-learn    inference is coef @ features + intercept
  scipy           median filter and connected components are reimplemented in JS

Usage:
    python web/export_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "integration"))

from features import _filter_unitary, initial_angles          # noqa: E402

MODEL = ROOT / "integration" / "ocr_backend.npz"
OUT = ROOT / "web" / "model.json"


def main():
    if not MODEL.exists():
        raise SystemExit(
            "integration/ocr_backend.npz not found.\n"
            "Run:  python integration/pipeline.py --train")

    d = np.load(MODEL, allow_pickle=True)

    # The trained logistic-regression head: one weight row per character class.
    coef = np.asarray(d["coef"], dtype=float)
    intercept = np.asarray(d["intercept"], dtype=float)

    # Class labels in the SAME ROW ORDER as coef, so the browser can map an
    # argmax straight back to a character. Order is not alphabetical by
    # accident of sklearn -- it is whatever np.unique produced during training,
    # which is why it is exported rather than reconstructed.
    classes = [str(c) for c in d["classes"]]

    # The quanvolutional filter as a single 16x16 unitary. Extracting it here
    # means the browser never needs Qiskit: applying the filter is one
    # matrix-vector product. Seed 42 and depth 2 must match what trained the
    # head above, or the features and the weights describe different circuits.
    U = _filter_unitary(4, 42, 2)

    payload = {
        "meta": {
            "note": "Exported from the trained pipeline. Do not hand-edit.",
            "n_classes": len(classes),
            "feature_dim": int(coef.shape[1]),
            "holdout_acc": float(d["holdout_acc"]),
        },
        "constants": {
            "crop_size": 8,
            "patch_size": 2,
            "stride": 2,
            "n_qubits": 4,
            "correlations": True,
        },
        # Full float64 precision, deliberately not rounded.
        # Rounding to 6 decimals shifted features by ~2e-6, which was enough to
        # flip the argmax on borderline glyphs in the degraded tiers and made
        # the browser decode one character differently from Python. The file is
        # ~2x larger and gzips well; exact agreement is worth more than the KB.
        "filter_unitary_real": U.real.tolist(),
        "filter_unitary_imag": U.imag.tolist(),
        "ocr_coef": coef.tolist(),
        "ocr_intercept": intercept.tolist(),
        "classes": classes,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    kb = OUT.stat().st_size / 1024
    print("wrote {}  ({:.0f} KB)".format(OUT, kb))
    print("  classes      {}".format(len(classes)))
    print("  feature dim  {}".format(coef.shape[1]))
    print("  unitary      {}x{}".format(*U.shape))


if __name__ == "__main__":
    main()
