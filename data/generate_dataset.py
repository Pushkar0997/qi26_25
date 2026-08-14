"""
generate_dataset.py — synthetic document dataset across five quality tiers.

Why synthetic instead of scraped real documents:
  1. Ground-truth text is exact and free. Hand-labelling 200 scraped documents
     is not possible in the time we have, and without labels we cannot compute
     character error rate, which is the headline accuracy number in the report.
  2. Degradation is a controlled variable. Because every tier renders the SAME
     source text, any accuracy difference between tiers is attributable to image
     quality alone, not to content difficulty. That is a cleaner experiment than
     five unrelated piles of real documents.
  3. It is reproducible from a seed, so the mentor can regenerate our exact
     dataset.

Honest limitation to state in the report: the "handwriting" tiers are a
italic-font-plus-per-glyph-jitter proxy for handwriting variability, not real
handwriting. Real handwritten data (IAM Handwriting Database) is the correct
extension if the project continues past the deadline.

Outputs, per tier:
    data/processed/<tier>/doc_XXX.png     full page image
    data/processed/<tier>/doc_XXX.json    ground-truth text + char boxes
    data/processed/<tier>/chars/          labelled character crops (.npy bundle)

Usage:
    python data/generate_dataset.py --docs-per-tier 12 --seed 26
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Crop preprocessing is shared with the inference path so that training crops
# and runtime crops are produced by identical code. Divergence here silently
# destroys accuracy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integration"))
from features import square_pad_resize  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TIERS = [
    "clean_digital",
    "clean_scan",
    "noisy_scan",
    "clear_handwriting",
    "degraded_handwriting",
]

# Characters that get their own bounding box and become classifier training
# data. Space is excluded (nothing to classify), everything else is a class.
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:"

# Font resolution is deliberately cross-platform. The printed tiers want a
# neutral sans face; the handwriting-proxy tiers want an italic serif, since
# slanted, connected-looking strokes are what makes the proxy plausible.
#
# The Windows entries are not arbitrary substitutes: the Liberation faces were
# designed as metric-compatible replacements for Arial and Times New Roman, so
# Arial / Times New Roman Italic are the closest possible match to the Linux
# rendering. Glyph outlines still differ slightly between them, so accuracy
# figures regenerated on a different OS may shift by a fraction of a percent.
# The resolved font files are recorded in manifest.json for exactly this reason.

FONT_PRINT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
    "LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",                                       # Windows
    "arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",                     # macOS
    "DejaVuSans.ttf",
]

FONT_HAND_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",  # Linux
    "LiberationSerif-Italic.ttf",
    "C:/Windows/Fonts/timesi.ttf",                                      # Windows
    "timesi.ttf",
    "C:/Windows/Fonts/georgiai.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",    # macOS
    "DejaVuSerif-Italic.ttf",
]


def resolve_font(candidates, kind):
    """Return the first candidate PIL can actually open.

    Existence checks are not enough — a path can exist but be a format PIL
    cannot load — so each candidate is opened for real. Bare filenames are
    included because PIL searches the system font directories for those, which
    is how this finds fonts on Windows without hardcoding a drive letter.
    """
    for cand in candidates:
        try:
            ImageFont.truetype(cand, 12)
            return cand
        except (OSError, IOError):
            continue
    raise SystemExit(
        "\nCould not find a usable {} font. Tried:\n  {}\n\n"
        "Pass one explicitly, e.g.:\n"
        "  python data/generate_dataset.py --font-{} C:/Windows/Fonts/arial.ttf\n"
        .format(kind, "\n  ".join(candidates),
                "print" if kind == "print" else "hand")
    )

PAGE_W, PAGE_H = 620, 420
FONT_SIZE = 22
LINE_H = 46
MARGIN_X, MARGIN_Y = 40, 34

CROP_SIZE = 8  # character crops are downsampled to 8x8 for the quantum stage

# Fields used to build each document. The DOC ID is the target that the
# Track B Grover search looks for downstream, so it follows a fixed,
# searchable format.
SUBJECTS = [
    "QUANTUM DOCUMENT PIPELINE",
    "ENCODING EFFICIENCY REVIEW",
    "GROVER SEARCH BENCHMARK",
    "FEATURE MAP ANALYSIS",
    "OCR BACKEND HANDOFF",
]
DEPARTMENTS = ["RESEARCH", "ARCHIVES", "OPERATIONS", "COMPLIANCE"]


# --------------------------------------------------------------------------
# Document content
# --------------------------------------------------------------------------

def make_document_text(rng):
    """Build one document's lines plus the doc ID that Track B will search for.

    The ID format QI##-XXXX is deliberate: it mixes digits with hex letters, so
    the downstream identifier extractor has to distinguish it from a pure-digit
    field like REF. That distinction turned out to matter -- an earlier scoring
    rule based on digit density alone preferred REF and searched the wrong
    field entirely.
    """
    doc_id = "QI{}-{}{}{}{}".format(
        rng.integers(10, 100),
        *[rng.choice(list("ABCDEF0123456789")) for _ in range(4)],
    )
    lines = [
        "QWORLD RESEARCH INSTITUTE",
        "DOCUMENT ID: {}".format(doc_id),
        "DEPT: {}".format(rng.choice(DEPARTMENTS)),
        "SUBJECT: {}".format(rng.choice(SUBJECTS)),
        "REF: {:04d}   PAGE: {:02d}".format(
            int(rng.integers(0, 10000)), int(rng.integers(1, 20))
        ),
    ]
    return lines, doc_id


# --------------------------------------------------------------------------
# Rendering
#
# We render the page image and, in parallel, an integer "index map" in which
# every character's pixels are painted with that character's index + 1.
# Any geometric transform (rotation, skew, rescale) is applied to BOTH with
# nearest-neighbour resampling, so character bounding boxes stay exactly
# correct no matter how badly we warp the page. Photometric degradation
# (blur, noise, contrast) touches only the visible image.
# --------------------------------------------------------------------------

def render_page(lines, handwriting, rng, font_print, font_hand):
    """Return (grayscale page as uint8, index map as int32, list of char labels)."""
    font_path = font_hand if handwriting else font_print
    font = ImageFont.truetype(font_path, FONT_SIZE)

    page = Image.new("L", (PAGE_W, PAGE_H), color=255)
    idx_map = Image.new("I", (PAGE_W, PAGE_H), color=0)
    draw_page = ImageDraw.Draw(page)
    draw_idx = ImageDraw.Draw(idx_map)

    labels = []          # labels[i] is the character for index i
    char_index = 0

    y = MARGIN_Y
    for line in lines:
        x = MARGIN_X
        for ch in line:
            if ch == " ":
                x += font.getlength(" ")
                continue

            # Per-glyph jitter simulates the natural inconsistency of
            # handwriting: baseline wobble, slight rotation, size variation.
            # These three together are what a font alone cannot fake -- printed
            # text has identical glyph shapes at identical baselines, and it is
            # that regularity, not the letterforms, that makes it easy to
            # segment. Perturbing position/rotation/scale per character
            # reproduces the part of handwriting that actually breaks OCR.
            if handwriting:
                dx = rng.normal(0, 1.4)
                dy = rng.normal(0, 2.2)
                angle = rng.normal(0, 5.0)
                scale = rng.normal(1.0, 0.07)
            else:
                dx = dy = angle = 0.0
                scale = 1.0

            glyph_font = font
            if abs(scale - 1.0) > 1e-3:
                glyph_font = ImageFont.truetype(
                    font_path, max(10, int(FONT_SIZE * scale))
                )

            # Draw the glyph on its own tile so it can be rotated
            # independently, then paste. Rotating the whole page instead would
            # rotate every glyph by the same angle, which is a scanning
            # artifact, not handwriting variation. The same tile mask paints the
            # index map, which is what keeps the ground-truth boxes exact.
            tw = int(glyph_font.getlength(ch)) + 12
            th = FONT_SIZE * 2
            tile = Image.new("L", (tw, th), color=0)
            ImageDraw.Draw(tile).text((6, 4), ch, fill=255, font=glyph_font)

            if abs(angle) > 1e-3:
                tile = tile.rotate(angle, resample=Image.BILINEAR, expand=False)

            px, py = int(x + dx), int(y + dy)
            # Ink is dark on light paper, so paste black through the glyph mask.
            page.paste(0, (px, py), tile)

            mask = tile.point(lambda v: 255 if v > 96 else 0)
            char_index += 1
            draw_idx.bitmap((px, py), mask, fill=char_index)

            labels.append(ch)
            x += font.getlength(ch) * (scale if handwriting else 1.0)

        y += LINE_H

    del draw_page, draw_idx
    return np.array(page, dtype=np.uint8), np.array(idx_map, dtype=np.int32), labels


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------

def geometric_degrade(page, idx_map, tier, rng):
    """Rotation / low-DPI resampling. Applied identically to page and index map."""
    img = Image.fromarray(page, mode="L")
    imap = Image.fromarray(idx_map, mode="I")

    if tier == "clean_digital":
        return img, imap

    angle = {
        "clean_scan": rng.normal(0, 0.5),
        "noisy_scan": rng.normal(0, 2.2),
        "clear_handwriting": rng.normal(0, 0.8),
        "degraded_handwriting": rng.normal(0, 1.8),
    }[tier]

    if abs(angle) > 1e-3:
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
        imap = imap.rotate(angle, resample=Image.NEAREST, fillcolor=0)

    # Low-DPI scan: downsample then upsample, permanently destroying detail.
    # This is the single most damaging degradation in the set, and the reason
    # noisy_scan segments so poorly: once adjacent glyphs blur into each other
    # at 45% scale, no amount of upsampling separates them again. It models a
    # genuinely common failure -- documents scanned at low DPI and later
    # enlarged.
    if tier in ("noisy_scan", "degraded_handwriting"):
        f = 0.45 if tier == "noisy_scan" else 0.55
        small = (int(PAGE_W * f), int(PAGE_H * f))
        img = img.resize(small, Image.BILINEAR).resize(
            (PAGE_W, PAGE_H), Image.BILINEAR
        )
        # Index map is NOT downsampled — boxes stay exact.

    return img, imap


def photometric_degrade(img, tier, rng):
    """Blur, sensor noise, contrast loss, blotches. Visible image only."""
    if tier == "clean_digital":
        return np.array(img, dtype=np.uint8)

    params = {
        "clean_scan":           dict(blur=0.4, noise=4,  contrast=1.00, blotch=0),
        "noisy_scan":           dict(blur=1.1, noise=22, contrast=0.72, blotch=14),
        "clear_handwriting":    dict(blur=0.5, noise=6,  contrast=0.95, blotch=0),
        "degraded_handwriting": dict(blur=1.4, noise=26, contrast=0.55, blotch=22),
    }[tier]

    if params["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(params["blur"]))

    arr = np.array(img, dtype=np.float32)

    # Contrast reduction pulls everything toward mid-grey. This is what makes
    # faded photocopies genuinely hard to read, and it is why the pipeline
    # normalises each crop before encoding: the intensity-to-angle map would
    # otherwise compress a faded character into a narrow band of rotation
    # angles, nearly indistinguishable from blank paper.
    if params["contrast"] < 1.0:
        arr = 255 - (255 - arr) * params["contrast"]

    if params["noise"] > 0:
        arr += rng.normal(0, params["noise"], arr.shape)

    # Blotches: toner speckle and paper stains, as random dark/light discs.
    for _ in range(params["blotch"]):
        cy, cx = rng.integers(0, PAGE_H), rng.integers(0, PAGE_W)
        r = int(rng.integers(2, 9))
        val = float(rng.choice([rng.integers(0, 90), rng.integers(200, 256)]))
        yy, xx = np.ogrid[:PAGE_H, :PAGE_W]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        arr[disc] = arr[disc] * 0.25 + val * 0.75

    return np.clip(arr, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Character crop extraction
# --------------------------------------------------------------------------

def extract_crops(page_arr, idx_map_arr, labels, pad=2):
    """Cut each labelled character out of the degraded page, resize to 8x8.

    Boxes come from the index map, so they remain pixel-accurate even after
    rotation. Crops are taken from the DEGRADED page, so the classifier sees
    exactly the noise the pipeline will face at inference time.
    """
    crops, crop_labels, boxes = [], [], []

    # Index i corresponds to labels[i-1]; the map stores index+1 so that 0 can
    # mean "no glyph here".
    for i, ch in enumerate(labels, start=1):
        ys, xs = np.where(idx_map_arr == i)

        # Fewer than 6 pixels means the glyph was rotated off-page or destroyed
        # by degradation. Dropped rather than kept, because a near-empty crop
        # labelled with a real character is a mislabelled training example --
        # actively worse than a missing one.
        if len(ys) < 6:
            continue

        y0, y1 = max(0, ys.min() - pad), min(page_arr.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(page_arr.shape[1], xs.max() + pad + 1)
        if (y1 - y0) < 4 or (x1 - x0) < 3:
            continue

        crops.append(square_pad_resize(page_arr[y0:y1, x0:x1], CROP_SIZE))
        crop_labels.append(ch)
        boxes.append([int(x0), int(y0), int(x1), int(y1)])

    return np.array(crops), crop_labels, boxes


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-per-tier", type=int, default=12)
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--out", type=str, default="data/processed")
    ap.add_argument("--font-print", type=str, default=None,
                    help="override the printed-tier font (path or system name)")
    ap.add_argument("--font-hand", type=str, default=None,
                    help="override the handwriting-proxy font")
    args = ap.parse_args()

    font_print = args.font_print or resolve_font(FONT_PRINT_CANDIDATES, "print")
    font_hand = args.font_hand or resolve_font(FONT_HAND_CANDIDATES, "hand")
    print("fonts: print={}\n       hand ={}\n".format(font_print, font_hand))

    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out)
    manifest = {"seed": args.seed, "charset": CHARSET, "crop_size": CROP_SIZE,
                "font_print": font_print, "font_hand": font_hand,
                "platform": sys.platform, "tiers": {}}

    for tier in TIERS:
        tier_dir = out_root / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        handwriting = "handwriting" in tier

        all_crops, all_labels = [], []
        docs_meta = []

        for d in range(args.docs_per_tier):
            lines, doc_id = make_document_text(rng)
            page, idx_map, labels = render_page(lines, handwriting, rng,
                                                font_print, font_hand)
            img, imap = geometric_degrade(page, idx_map, tier, rng)
            page_arr = photometric_degrade(img, tier, rng)
            idx_arr = np.array(imap, dtype=np.int32)

            crops, crop_labels, boxes = extract_crops(page_arr, idx_arr, labels)

            name = "doc_{:03d}".format(d)
            Image.fromarray(page_arr, mode="L").save(tier_dir / (name + ".png"))
            with open(tier_dir / (name + ".json"), "w") as f:
                json.dump({
                    "tier": tier,
                    "doc_id": doc_id,
                    "lines": lines,
                    "text": "\n".join(lines),
                    "char_labels": crop_labels,
                    "char_boxes": boxes,
                }, f, indent=2)

            all_crops.append(crops)
            all_labels.extend(crop_labels)
            docs_meta.append({"file": name + ".png", "doc_id": doc_id,
                              "n_chars": len(crop_labels)})

        crops_arr = np.concatenate(all_crops, axis=0)
        np.savez_compressed(
            tier_dir / "chars.npz",
            crops=crops_arr,
            labels=np.array(all_labels),
        )

        manifest["tiers"][tier] = {
            "n_docs": args.docs_per_tier,
            "n_chars": int(crops_arr.shape[0]),
            "docs": docs_meta,
        }
        print("{:22s}  {:3d} docs  {:5d} char crops".format(
            tier, args.docs_per_tier, crops_arr.shape[0]))

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nManifest written to", out_root / "manifest.json")


if __name__ == "__main__":
    main()
