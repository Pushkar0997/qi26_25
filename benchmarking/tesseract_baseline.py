"""
tesseract_baseline.py — an external, non-quantum OCR baseline.

WHY THIS EXISTS
---------------
The benchmark compares quantum features against classical features against raw
pixels. All three share this project's segmentation front end and its logistic
regression head, so they answer "which feature map is best inside this
pipeline?" — not "is this pipeline any good?"

Without an external reference, a CER of 6.5% has no anchor. Tesseract is the
obvious one: a mature, freely available engine that has been developed since
1985. Comparing against it is the question a sceptical reader asks first, and
the answer is unflattering, which is precisely why it belongs in the report.

Tesseract is given the same document images and scored with the same CER
function, so the comparison is like for like at the document level. It is not a
controlled comparison of feature maps — Tesseract brings its own segmentation,
language model and training data — and the report says so.

REQUIREMENTS
    pip install pytesseract
    plus the tesseract binary:
      Ubuntu/Debian  sudo apt-get install tesseract-ocr
      macOS          brew install tesseract
      Windows        https://github.com/UB-Mannheim/tesseract/wiki
    Colab            !apt-get -qq install tesseract-ocr

If the binary is missing the script exits with a clear message rather than
failing partway.

Usage:
    python benchmarking/tesseract_baseline.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "integration"))
sys.path.insert(0, str(ROOT / "track_b_search" / "oracle"))

import pipeline as P                                    # noqa: E402

TIERS = ["clean_digital", "clean_scan", "noisy_scan",
         "clear_handwriting", "degraded_handwriting"]


def check_tesseract():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return pytesseract
    except ImportError:
        raise SystemExit(
            "pytesseract is not installed.\n  pip install pytesseract")
    except Exception as e:
        raise SystemExit(
            "pytesseract is installed but the tesseract binary was not found.\n"
            "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
            "  macOS:         brew install tesseract\n"
            "  Windows:       https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Original error: {}".format(e))


def normalise(text):
    """Fold Tesseract's output into the same alphabet the pipeline uses.

    The dataset charset is uppercase A-Z, digits, hyphen and colon. Tesseract
    reports mixed case and a wider punctuation set, so comparing raw output
    would penalise it for differences the ground truth cannot express. Upcasing
    and dropping out-of-charset characters removes that unfairness.
    """
    keep = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-: \n")
    return "".join(c for c in text.upper() if c in keep)


def main():
    pytesseract = check_tesseract()
    print("tesseract", pytesseract.get_tesseract_version(), "\n")

    coef, intercept, classes = P.load_backend()
    rows = {}

    header = "{:24s} {:>12s} {:>12s} {:>14s} {:>14s}".format(
        "tier", "quantum CER", "tesseract", "quantum ID", "tesseract ID")
    print(header)
    print("-" * len(header))

    for tier in TIERS:
        q_cer, t_cer, q_id, t_id = [], [], [], []

        # Both systems see the identical image file. Tesseract is given the
        # original PNG rather than the binarised or cropped intermediates, since
        # its own preprocessing is part of what is being compared.
        for f in sorted((ROOT / "data/processed" / tier).glob("doc_*.png")):
            meta = json.loads(f.with_suffix(".json").read_text())
            truth = meta["text"]
            true_hex = "".join(c for c in meta["doc_id"].upper()
                               if c in "0123456789ABCDEF")

            # This project's pipeline.
            arr = np.array(Image.open(f).convert("L"), dtype=np.uint8)
            boxes = P.segment_characters(arr)
            crops = P.crops_from_boxes(arr, boxes)
            qtext = P.assemble_text(boxes, P.predict_chars(
                crops, coef, intercept, classes))
            # Same CER function for both, so the comparison is like for like
            # even though the two systems reach their text very differently.
            q_cer.append(P.cer(qtext, truth))
            q_id.append(1.0 if P._locate_id_field(qtext) == true_hex else 0.0)

            # Tesseract, on the same image.
            ttext = normalise(pytesseract.image_to_string(Image.open(f)))
            t_cer.append(P.cer(ttext, truth))
            t_id.append(1.0 if P._locate_id_field(ttext) == true_hex else 0.0)

        rows[tier] = {
            "quantum_cer": float(np.mean(q_cer)),
            "tesseract_cer": float(np.mean(t_cer)),
            "quantum_id_exact": float(np.mean(q_id)),
            "tesseract_id_exact": float(np.mean(t_id)),
            "n_docs": len(q_cer),
        }
        r = rows[tier]
        print("{:24s} {:>11.1%} {:>11.1%} {:>13.0%} {:>13.0%}".format(
            tier, r["quantum_cer"], r["tesseract_cer"],
            r["quantum_id_exact"], r["tesseract_id_exact"]))

    out = ROOT / "benchmarking" / "tesseract_baseline.json"
    out.write_text(json.dumps(rows, indent=2))
    print("\nwrote", out)

    print("\nNote: this is a document-level comparison, not a controlled one.")
    print("Tesseract brings its own segmentation, language model and training")
    print("data; the pipeline here uses a 4-qubit filter over 8x8 crops.")


if __name__ == "__main__":
    main()
