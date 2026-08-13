"""
run_benchmark.py — produces every quantitative result in the final report.

Experiments:
    A. Feature-extractor comparison, per quality tier, at matched dimensionality
    B. End-to-end character error rate per tier, including segmentation quality
    C. Quantum resource cost: encoding, quanvolutional layer, Grover oracle
    D. Grover oracle scaling with text length (the O(N) data-loading result)
    E. Shot-noise sensitivity of the quantum feature stage

Design rule followed throughout: when comparing quantum against classical, hold
feature dimensionality fixed. Otherwise a win is indistinguishable from simply
handing one method a wider representation.

Usage:
    python benchmarking/run_benchmark.py --out benchmarking/results
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "integration"))
sys.path.insert(0, str(ROOT / "track_b_search" / "oracle"))

from features import (quanv_features, classical_conv, raw_pixels,      # noqa: E402
                      normalize_crops, random_entangling_circuit)
from comparator_oracle import Alphabet, oracle_resources               # noqa: E402
import pipeline as P                                                   # noqa: E402

from sklearn.linear_model import LogisticRegression                    # noqa: E402
from sklearn.model_selection import train_test_split                   # noqa: E402

DATA = ROOT / "data" / "processed"
TIERS = ["clean_digital", "clean_scan", "noisy_scan",
         "clear_handwriting", "degraded_handwriting"]


def load_tier(tier):
    d = np.load(DATA / tier / "chars.npz")
    return d["crops"], d["labels"]


def fit_score(F, y, seed=26):
    a, b, c, d = train_test_split(F, y, test_size=0.25, random_state=seed,
                                  stratify=y)
    clf = LogisticRegression(max_iter=3000, C=5.0).fit(a, c)
    return clf.score(b, d)


# --------------------------------------------------------------------------
# A. Feature extractors, per tier
# --------------------------------------------------------------------------

def experiment_a():
    print("\n[A] Feature extractor comparison, per tier")
    extractors = {
        "quanv (marginals)":      lambda X: quanv_features(X),
        "quanv (+ZZ corr)":       lambda X: quanv_features(X, correlations=True),
        "classical conv (dim-matched)": lambda X: classical_conv(X, n_filters=10),
        "raw pixels":             lambda X: raw_pixels(X),
    }
    rows = {}
    for tier in TIERS:
        X, y = load_tier(tier)
        Xn = normalize_crops(X)
        rows[tier] = {}
        for name, fn in extractors.items():
            F = fn(Xn)
            rows[tier][name] = {"acc": round(float(fit_score(F, y)), 4),
                                "dim": int(F.shape[1])}
        print("  {:22s} ".format(tier) + "  ".join(
            "{}={:.3f}".format(k.split()[0], v["acc"]) for k, v in rows[tier].items()))

    # pooled across all tiers
    X = np.concatenate([load_tier(t)[0] for t in TIERS])
    y = np.concatenate([load_tier(t)[1] for t in TIERS])
    Xn = normalize_crops(X)
    rows["ALL_TIERS"] = {}
    for name, fn in extractors.items():
        F = fn(Xn)
        rows["ALL_TIERS"][name] = {"acc": round(float(fit_score(F, y)), 4),
                                   "dim": int(F.shape[1])}
    print("  {:22s} ".format("ALL TIERS") + "  ".join(
        "{}={:.3f}".format(k.split()[0], v["acc"]) for k, v in rows["ALL_TIERS"].items()))
    return rows


# --------------------------------------------------------------------------
# B. End-to-end CER per tier
# --------------------------------------------------------------------------

def experiment_b(max_docs=8):
    print("\n[B] End-to-end pipeline, per tier")
    if not P.MODEL_PATH.exists():
        P.train_backend()
    coef, intercept, classes = P.load_backend()

    rows = {}
    for tier in TIERS:
        cers, seg_ratio, id_hits = [], [], []
        for f in sorted((DATA / tier).glob("doc_*.png"))[:max_docs]:
            meta = json.loads(f.with_suffix(".json").read_text())
            from PIL import Image
            arr = np.array(Image.open(f).convert("L"), dtype=np.uint8)
            boxes = P.segment_characters(arr)
            crops = P.crops_from_boxes(arr, boxes)
            chars = P.predict_chars(crops, coef, intercept, classes)
            text = P.assemble_text(boxes, chars)
            cers.append(P.cer(text, meta["text"]))
            seg_ratio.append(len(boxes) / max(1, len(meta["char_labels"])))

            # Task-level success: was the document's identifier recovered
            # exactly? CER measures average character quality, but the actual
            # job here is retrieving a specific field, and a single wrong
            # character makes the ID useless. Reporting both separates
            # "mostly readable" from "actually usable".
            true_hex = "".join(c for c in meta.get("doc_id", "").upper()
                               if c in "0123456789ABCDEF")
            id_hits.append(1.0 if P._locate_id_field(text) == true_hex else 0.0)
        rows[tier] = {
            "cer_mean": round(float(np.mean(cers)), 4),
            "cer_std": round(float(np.std(cers)), 4),
            "segmentation_recall_proxy": round(float(np.mean(seg_ratio)), 3),
            "id_exact_match": round(float(np.mean(id_hits)), 3),
            "n_docs": len(cers),
        }
        print("  {:22s} CER={:.1%} +/- {:.1%}   seg={:.2f}   ID exact={:.0%}".format(
            tier, rows[tier]["cer_mean"], rows[tier]["cer_std"],
            rows[tier]["segmentation_recall_proxy"], rows[tier]["id_exact_match"]))
    return rows


# --------------------------------------------------------------------------
# C. Quantum resource cost
# --------------------------------------------------------------------------

def experiment_c():
    print("\n[C] Quantum resource cost")
    from qiskit import transpile
    res = {}

    filt = random_entangling_circuit(4, seed=42, depth=2)
    t = transpile(filt, basis_gates=["u", "cx"], optimization_level=1)
    res["quanv_filter"] = {"qubits": 4, "depth": t.depth(),
                           "cx": t.count_ops().get("cx", 0),
                           "circuits_per_8x8_char": 16}
    print("  quanv filter: 4 qubits, depth {}, cx {}, 16 circuits/char".format(
        t.depth(), t.count_ops().get("cx", 0)))

    hexa = Alphabet("0123456789ABCDEF")
    r = oracle_resources("A7F3B41C9D05E2F8", "26", hexa)
    res["grover_oracle_16char"] = r
    print("  grover oracle (16-char hex text, 2-char pattern): "
          "{} qubits, depth {}, cx {}".format(r["qubits"], r["depth"], r["cx"]))
    return res


# --------------------------------------------------------------------------
# D. Grover scaling
# --------------------------------------------------------------------------

def experiment_d():
    print("\n[D] Grover oracle scaling with text length")
    hexa = Alphabet("0123456789ABCDEF")
    base = "A7F3B41C9D05E2F8"
    rows = []
    for N in [8, 12, 16, 24, 32, 48, 64]:
        t = (base * ((N // len(base)) + 1))[:N]
        t = t[:-2] + "26"
        rows.append(oracle_resources(t, "26", hexa))
        print("  N={:3d}  qubits={:2d}  depth={:6d}  cx={:6d}".format(
            N, rows[-1]["qubits"], rows[-1]["depth"], rows[-1]["cx"]))
    N = np.array([r["n_text"] for r in rows], dtype=float)
    cx = np.array([r["cx"] for r in rows], dtype=float)
    exponent = float(np.polyfit(np.log(N), np.log(cx), 1)[0])
    print("  -> CX scaling exponent vs N: {:.2f}".format(exponent))
    return {"rows": rows, "cx_exponent": round(exponent, 3)}


# --------------------------------------------------------------------------
# E. Shot-noise sensitivity
# --------------------------------------------------------------------------

def experiment_e(shot_levels=(64, 256, 1024, 4096, None)):
    print("\n[E] Shot-noise sensitivity of the quantum feature stage")
    X = np.concatenate([load_tier(t)[0] for t in TIERS])
    y = np.concatenate([load_tier(t)[1] for t in TIERS])
    Xn = normalize_crops(X)
    out = {}
    for s in shot_levels:
        F = quanv_features(Xn, shots=s)
        acc = float(fit_score(F, y))
        out["exact" if s is None else str(s)] = round(acc, 4)
        print("  shots={:>6}  acc={:.3f}".format("exact" if s is None else s, acc))
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_markdown(results, path):
    L = ["# Benchmark Results", "",
         "Generated by `benchmarking/run_benchmark.py`. All numbers measured, "
         "none estimated.", ""]

    L += ["## A. Feature extractors (character accuracy)", ""]
    names = list(results["A"]["ALL_TIERS"].keys())
    L.append("| Tier | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for tier, row in results["A"].items():
        L.append("| {} | ".format(tier) + " | ".join(
            "{:.1%} (d={})".format(row[n]["acc"], row[n]["dim"]) for n in names) + " |")

    L += ["", "## B. End-to-end character error rate", "",
          "| Tier | CER | std | segmentation ratio | document ID exact |",
          "|---|---|---|---|---|"]
    for tier, r in results["B"].items():
        L.append("| {} | {:.1%} | {:.1%} | {:.2f} | {:.0%} |".format(
            tier, r["cer_mean"], r["cer_std"], r["segmentation_recall_proxy"],
            r["id_exact_match"]))

    L += ["", "## C. Quantum resource cost", "", "```",
          json.dumps(results["C"], indent=2), "```"]

    L += ["", "## D. Grover oracle scaling", "",
          "| text length N | qubits | depth | CX |", "|---|---|---|---|"]
    for r in results["D"]["rows"]:
        L.append("| {} | {} | {} | {} |".format(
            r["n_text"], r["qubits"], r["depth"], r["cx"]))
    L += ["", "Measured CX scaling exponent vs N: **{:.2f}**. A value near 1 or "
          "above confirms that loading classical text into the oracle costs at "
          "least linear gate count, so the O(sqrt(N)) query advantage does not "
          "become an end-to-end speedup for stored unstructured text."
          .format(results["D"]["cx_exponent"])]

    L += ["", "## E. Shot-noise sensitivity", "", "| shots | accuracy |", "|---|---|"]
    for k, v in results["E"].items():
        L.append("| {} | {:.1%} |".format(k, v))

    Path(path).write_text("\n".join(L) + "\n")
    print("\nWrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="benchmarking/results")
    ap.add_argument("--skip", type=str, default="", help="comma list, e.g. B,E")
    args = ap.parse_args()
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    t0 = time.time()
    results = {}
    if "A" not in skip: results["A"] = experiment_a()
    if "B" not in skip: results["B"] = experiment_b()
    if "C" not in skip: results["C"] = experiment_c()
    if "D" not in skip: results["D"] = experiment_d()
    if "E" not in skip: results["E"] = experiment_e()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(results, indent=2))
    write_markdown(results, str(out) + ".md")
    print("Total runtime: {:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
