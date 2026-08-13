"""
pipeline.py — end-to-end: document image -> extracted text -> Grover pattern search.

This is the Week 3 deliverable ("working Python script extracting text from
sample documents") joined to the Week 4 deliverable (Grover string search),
which is the point at which the project stops being two disconnected tracks.

Stages:
    1. SEGMENT    projection-profile segmentation of the page into lines, then
                  lines into character boxes
    2. ENCODE     each character crop -> quantum state -> quanvolutional layer
                  (Track A) -> 64-dimensional feature vector
    3. CLASSIFY   lightweight classical backend (multinomial logistic
                  regression) maps features -> characters. This is the
                  "lightweight classical OCR back-end" the brief asks for: the
                  feature extraction is quantum, the decision layer is not.
    4. SEARCH     the recovered DOCUMENT ID field is fed to the Grover
                  comparator oracle (Track B) to locate a pattern

Usage:
    python integration/pipeline.py --train                       # fit + cache the backend
    python integration/pipeline.py --doc data/processed/clean_scan/doc_000.png --pattern 26
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "integration"))
sys.path.insert(0, str(ROOT / "track_b_search" / "oracle"))

from features import quanv_features, normalize_crops, square_pad_resize          # noqa: E402
from comparator_oracle import Alphabet, grover_search, check_phase  # noqa: E402

DATA = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "integration" / "ocr_backend.npz"
CROP = 8


# --------------------------------------------------------------------------
# 1. Segmentation
# --------------------------------------------------------------------------

def binarize(arr):
    """Otsu threshold. Returns a boolean ink mask (True = ink)."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = w_b = 0.0
    best_t, best_var = 128, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return arr <= best_t          # ink is dark


def clean_ink(ink, min_component=6):
    """Drop connected components too small to be a character stroke.

    The noisy tiers add toner speckle and stain discs. After thresholding these
    survive as small blobs which, left in place, make every row of the page
    register as containing ink, so line detection collapses into one 420-row
    'line'. Size filtering removes them without touching real glyphs.
    """
    from scipy import ndimage
    lab, n = ndimage.label(ink)
    if n == 0:
        return ink
    sizes = ndimage.sum(ink, lab, range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_component
    return keep[lab]


def preprocess_page(arr):
    """Median filter, threshold, then remove speckle. Returns an ink mask."""
    from scipy import ndimage
    smoothed = ndimage.median_filter(arr, size=3)
    return clean_ink(binarize(smoothed))


def segment_characters(arr, min_ink=4):
    """Projection-profile segmentation -> list of (line_idx, x0, y0, x1, y1).

    Rows containing ink form lines; within a line, columns containing ink form
    characters, with blank column runs acting as separators. This is the
    classical, non-learned part of the front end — deliberately simple, and its
    failure modes on rotated/noisy pages are reported rather than patched over.
    """
    ink = preprocess_page(arr)
    boxes = []

    # Adaptive row threshold: a real text line marks a meaningful fraction of
    # the page width. A fixed '>1 pixel' test is what let residual noise turn
    # the whole page into a single line on the degraded tiers.
    row_ink = ink.sum(axis=1)
    row_active = row_ink > max(2, int(0.008 * ink.shape[1]))
    lines, start = [], None
    for y, a in enumerate(row_active):
        if a and start is None:
            start = y
        elif not a and start is not None:
            if y - start >= 4:
                lines.append((start, y))
            start = None
    if start is not None and len(row_active) - start >= 4:
        lines.append((start, len(row_active)))

    for li, (y0, y1) in enumerate(lines):
        band = ink[y0:y1]
        col_ink = band.sum(axis=0)
        col_active = col_ink > 0
        cs = None
        for x, a in enumerate(col_active):
            if a and cs is None:
                cs = x
            elif not a and cs is not None:
                if band[:, cs:x].sum() >= min_ink and (x - cs) >= 2:
                    ys = np.where(band[:, cs:x].any(axis=1))[0]
                    boxes.append((li, cs, y0 + ys.min(), x, y0 + ys.max() + 1))
                cs = None
        if cs is not None and band[:, cs:].sum() >= min_ink:
            ys = np.where(band[:, cs:].any(axis=1))[0]
            boxes.append((li, cs, y0 + ys.min(), band.shape[1], y0 + ys.max() + 1))

    return split_wide_boxes(boxes, ink)


def split_wide_boxes(boxes, ink, ratio=1.8, valley_frac=0.45):
    """Split merged glyphs, but only at genuine valleys in the ink profile.

    On low-DPI scans, downsampling plus blur closes the gap between adjacent
    glyphs, so the projection returns one box covering several characters.

    Splitting such a box into equal-width slices is tempting but wrong: on the
    degraded tiers it over-segments badly, cutting single characters apart and
    pushing the error rate above 100% through pure insertions. Instead we cut
    only where the column ink profile genuinely dips - a thin bridge between
    two glyphs - and require the dip to be well below the box average. A box
    with no such valley is left intact, so a clean wide 'M' survives.
    """
    if len(boxes) < 3:
        return boxes
    widths = np.array([x1 - x0 for (_, x0, _, x1, _) in boxes], dtype=float)
    ref = float(np.percentile(widths, 60))
    if ref <= 0:
        return boxes

    out = []
    for (li, x0, y0, x1, y1) in boxes:
        w = x1 - x0
        if w <= ratio * ref:
            out.append((li, x0, y0, x1, y1))
            continue

        col = ink[y0:y1, x0:x1].sum(axis=0).astype(float)
        if col.size < 6 or col.mean() <= 0:
            out.append((li, x0, y0, x1, y1))
            continue

        margin = max(2, int(0.25 * ref))
        cuts = []
        for c in range(margin, len(col) - margin):
            if col[c] < valley_frac * col.mean() and \
               col[c] <= col[max(0, c - 2):c + 3].min():
                if not cuts or c - cuts[-1] >= margin:
                    cuts.append(c)

        if not cuts:
            out.append((li, x0, y0, x1, y1))
            continue

        prev = 0
        for c in cuts + [len(col)]:
            if c - prev >= 2:
                out.append((li, x0 + prev, y0, x0 + c, y1))
            prev = c
    return out


def crops_from_boxes(arr, boxes, pad=1):
    out = []
    H, W = arr.shape
    for (_, x0, y0, x1, y1) in boxes:
        y0, y1 = max(0, y0 - pad), min(H, y1 + pad)
        x0, x1 = max(0, x0 - pad), min(W, x1 + pad)
        out.append(square_pad_resize(arr[y0:y1, x0:x1].astype(np.uint8), CROP))
    return np.array(out) if out else np.zeros((0, CROP, CROP))


# --------------------------------------------------------------------------
# 2-3. Feature extraction + classical backend
# --------------------------------------------------------------------------

def load_training_data(tiers=None, holdout_docs=2):
    """Character crops from every tier. The backend is trained across all
    quality levels so it is not tuned to clean input only."""
    tiers = tiers or [d.name for d in sorted(DATA.iterdir()) if d.is_dir()]
    X, y = [], []
    for t in tiers:
        f = DATA / t / "chars.npz"
        if not f.exists():
            continue
        d = np.load(f)
        X.append(d["crops"])
        y.extend(list(d["labels"]))
    return np.concatenate(X, axis=0), np.array(y)


def train_backend(seed=26, shots=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = load_training_data()
    Xn = normalize_crops(X)
    F = quanv_features(Xn, shots=shots)

    Ftr, Fte, ytr, yte = train_test_split(
        F, y, test_size=0.25, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=5.0)
    clf.fit(Ftr, ytr)
    acc = clf.score(Fte, yte)

    np.savez(MODEL_PATH, coef=clf.coef_, intercept=clf.intercept_,
             classes=clf.classes_, holdout_acc=acc)
    print("backend trained on {} crops | held-out char accuracy {:.1%}".format(
        len(X), acc))
    return clf, acc


def load_backend():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("run with --train first")
    d = np.load(MODEL_PATH, allow_pickle=True)
    return d["coef"], d["intercept"], d["classes"]


def predict_chars(crops, coef, intercept, classes, shots=None):
    if len(crops) == 0:
        return []
    F = quanv_features(normalize_crops(crops), shots=shots)
    logits = F @ coef.T + intercept
    return [classes[i] for i in logits.argmax(axis=1)]


# --------------------------------------------------------------------------
# Text assembly + accuracy
# --------------------------------------------------------------------------

def assemble_text(boxes, chars, space_gap=1.6):
    """Rebuild lines, inserting spaces where the horizontal gap between
    consecutive characters is unusually large."""
    lines = {}
    for (li, x0, _, x1, _), ch in zip(boxes, chars):
        lines.setdefault(li, []).append((x0, x1, ch))

    out = []
    for li in sorted(lines):
        items = sorted(lines[li])
        if not items:
            continue
        # Word gaps are wide outliers among mostly-uniform letter gaps, so the
        # threshold is set from the gap distribution itself rather than from
        # glyph width, which varies far too much between 'I' and 'M'.
        gaps = [items[k][0] - items[k - 1][1] for k in range(1, len(items))]
        if gaps:
            med_gap = float(np.median(gaps))
            thresh = max(med_gap + space_gap * (np.std(gaps) + 1e-6),
                         med_gap * 2.0, 3.0)
        else:
            thresh = 1e9
        s = items[0][2]
        for k in range(1, len(items)):
            if (items[k][0] - items[k - 1][1]) > thresh:
                s += " "
            s += items[k][2]
        out.append(s)
    return "\n".join(out)


def cer(pred, truth):
    """Character error rate via Levenshtein distance, normalised by truth length."""
    p, t = pred.replace("\n", ""), truth.replace("\n", "")
    prev = list(range(len(t) + 1))
    for i, pc in enumerate(p, 1):
        cur = [i]
        for j, tc in enumerate(t, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pc != tc)))
        prev = cur
    return prev[-1] / max(1, len(t))


# --------------------------------------------------------------------------
# 4. Grover search on the recovered ID
# --------------------------------------------------------------------------

HEX = Alphabet("0123456789ABCDEF")


def _locate_id_field(text):
    """Pick the line most likely to be the DOCUMENT ID and return its hex part.

    Keyword matching on the label is unreliable, because the label itself is
    OCR output and comes back as things like 'OOCUMENT IO:'. Instead we score
    every line by the digit density of its value portion (whatever follows the
    last colon) and take the winner. This is layout-driven extraction of the
    kind used on forms and invoices, and it degrades gracefully when the label
    text is misread.
    """
    best, best_score = "", -1.0
    for line in text.upper().split("\n"):
        value = line.rsplit(":", 1)[-1] if ":" in line else line
        hexpart = "".join(c for c in value if c in "0123456789ABCDEF")
        if len(hexpart) < 3:
            continue
        digits = sum(c.isdigit() for c in hexpart)
        score = digits / len(hexpart) + 0.05 * len(hexpart)
        if score > best_score:
            best, best_score = hexpart, score
    return best


def search_extracted_id(text, pattern, max_len=16):
    """Run the Track B oracle over the hex-representable part of the extracted
    text. Restricted to a hex charset because a 38-symbol alphabet needs 6
    qubits per character, which puts even a short window past what we can
    simulate — see docs/benchmarking notes."""
    field = _locate_id_field(text)[:max_len]
    if len(field) < len(pattern) + 1:
        return {"searchable_field": field, "status": "field too short"}
    if any(c not in HEX.code for c in pattern.upper()):
        return {"searchable_field": field, "status": "pattern not hex"}

    flipped, truth, ok, leak = check_phase(field, pattern.upper(), HEX)
    if not truth:
        return {"searchable_field": field, "status": "pattern absent",
                "oracle_marked": flipped}

    best, counts, iters = grover_search(field, pattern.upper(), HEX, shots=2048)
    conf = max(counts.values()) / sum(counts.values())
    return {
        "searchable_field": field,
        "pattern": pattern.upper(),
        "true_positions": truth,
        "grover_position": best,
        "correct": best in truth,
        "confidence": round(conf, 4),
        "iterations": iters,
        "phase_check_ok": ok,
        "uncompute_leakage": leak,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_document(png_path, pattern, shots=None, verbose=True):
    png_path = Path(png_path)
    arr = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
    meta_path = png_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    boxes = segment_characters(arr)
    crops = crops_from_boxes(arr, boxes)
    coef, intercept, classes = load_backend()
    chars = predict_chars(crops, coef, intercept, classes, shots=shots)
    text = assemble_text(boxes, chars)

    result = {
        "file": str(png_path),
        "tier": meta.get("tier"),
        "n_boxes": len(boxes),
        "n_truth_chars": len(meta.get("char_labels", [])),
        "extracted_text": text,
    }
    if "text" in meta:
        result["ground_truth"] = meta["text"]
        result["cer"] = round(cer(text, meta["text"]), 4)

    if pattern:
        result["grover"] = search_extracted_id(text, pattern)

    if verbose:
        print("=" * 62)
        print("FILE :", png_path.name, "| TIER:", result["tier"])
        print("SEGMENTED: {} boxes (truth {} chars)".format(
            result["n_boxes"], result["n_truth_chars"]))
        print("-" * 62)
        print("EXTRACTED:")
        print(text)
        if "ground_truth" in result:
            print("-" * 62)
            print("GROUND TRUTH:")
            print(result["ground_truth"])
            print("-" * 62)
            print("CER: {:.1%}".format(result["cer"]))
        if pattern:
            print("-" * 62)
            print("GROVER SEARCH:")
            for k, v in result["grover"].items():
                print("  {:20s} {}".format(k, v))
        print("=" * 62)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--doc", type=str)
    ap.add_argument("--pattern", type=str, default="")
    ap.add_argument("--shots", type=int, default=None,
                    help="simulate finite measurement shots (default: exact)")
    args = ap.parse_args()

    if args.train:
        train_backend(shots=args.shots)
    if args.doc:
        run_document(args.doc, args.pattern, shots=args.shots)
    if not args.train and not args.doc:
        ap.print_help()


if __name__ == "__main__":
    main()
