"""
make_figures.py — regenerate every figure in the report from committed results.

Reads only benchmarking/*.json, so the figures cannot drift out of step with
the numbers in the report: if the benchmark is re-run, re-running this script
is the whole update.

Usage:
    python benchmarking/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
# Agg (write-to-file, no window) must be selected BEFORE pyplot is imported.
# Without it this script fails on a headless machine or over ssh, which is
# exactly where figures tend to get regenerated.
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarking"
FIGS = BENCH / "figures"

# The embedded newlines split each tick label over two lines, which is what lets
# five tier names sit under a 9-inch axis without rotating them to 45 degrees.
TIER_LABELS = {
    "clean_digital": "clean\ndigital",
    "clean_scan": "clean\nscan",
    "noisy_scan": "noisy\nscan",
    "clear_handwriting": "clear\nhandwr.",
    "degraded_handwriting": "degraded\nhandwr.",
}
# A fixed order rather than whatever the JSON happens to iterate in. Every figure
# then reads left to right as increasing degradation, so tiers stay comparable
# from one figure to the next.
ORDER = ["clean_digital", "clean_scan", "noisy_scan",
         "clear_handwriting", "degraded_handwriting"]


def load(name):
    """Read a results file, returning None if the experiment has not been run.

    Returning None rather than raising lets the script produce whatever figures
    it can. That matters because the experiments live in different scripts
    (run_benchmark.py, train_filter.py, encoding_scaling.py) and a user may
    reasonably have run only some of them.
    """
    p = BENCH / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fig_accuracy_by_tier(res):
    A = res["A"]
    methods = list(A["ALL_TIERS"].keys())

    # Grouped bars: one cluster per tier, one bar per method. Bar width is
    # derived from the method count so the clusters stay separated no matter how
    # many extractors are compared.
    x = np.arange(len(ORDER))
    w = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(methods):
        vals = [A[t][m]["acc"] for t in ORDER]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w,
               label="{} (d={})".format(m, A["ALL_TIERS"][m]["dim"]))
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in ORDER])
    ax.set_ylabel("character accuracy")
    # Start the axis at 0.5, not 0. All methods score well above chance, and a
    # zero baseline would compress the differences that matter into a few
    # pixels. Flagged here because truncated axes can mislead -- the intent is
    # legibility, and the numeric table in the report carries the exact values.
    ax.set_ylim(0.5, 1.02)
    ax.axhline(A["ALL_TIERS"]["raw pixels"]["acc"], ls=":", c="grey", lw=1)
    ax.set_title("Feature extractors at matched dimensionality\n"
                 "(dotted line = raw-pixel accuracy, all tiers pooled)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "accuracy_by_tier.png", dpi=150)
    plt.close(fig)


def fig_grover_scaling(res):
    rows = res["D"]["rows"]
    N = np.array([r["n_text"] for r in rows], float)
    cx = np.array([r["cx"] for r in rows], float)
    # The exponent is read from the results file rather than refitted here, so the
    # number in the title cannot disagree with the one quoted in the report.
    exp = res["D"]["cx_exponent"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(N, cx, "o-", label="measured CX count")

    # Reference lines anchored at the first measured point, so all three curves
    # start together and only their SLOPES differ. On log-log axes a power law
    # is a straight line, so "measured sits above linear" is directly readable
    # as "grows faster than linearly".
    ref = cx[0] * (N / N[0])
    ax.loglog(N, ref, "--", c="grey", label="linear reference (slope 1)")
    ax.loglog(N, cx[0] * np.sqrt(N / N[0]), ":", c="green",
              label=r"$\sqrt{N}$ reference (slope 0.5)")
    ax.set_xlabel("text length N (characters)")
    ax.set_ylabel("CX gates per oracle call")
    ax.set_title("Grover oracle cost is dominated by data loading\n"
                 "fitted slope = {:.2f}".format(exp))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "grover_scaling.png", dpi=150)
    plt.close(fig)


def fig_shot_noise(res):
    E = res["E"]
    # "exact" is the noiseless statevector result and has no shot count, so it
    # is drawn as a horizontal reference rather than a point on the x axis.
    keys = [k for k in E if k != "exact"]
    keys.sort(key=int)          # dict order is insertion order, not numeric
    vals = [E[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    # Log x-axis because the shot levels double: on a linear axis everything below
    # 1024 collapses against the left edge.
    ax.semilogx([int(k) for k in keys], vals, "o-")
    ax.axhline(E["exact"], ls="--", c="grey",
               label="exact statevector ({:.1%})".format(E["exact"]))
    ax.set_xlabel("measurement shots per patch")
    ax.set_ylabel("character accuracy")
    ax.set_title("Shot budget needed to approach noiseless features")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "shot_noise.png", dpi=150)
    plt.close(fig)


def fig_cer(res):
    B = res["B"]
    x = np.arange(len(ORDER))
    cer = [B[t]["cer_mean"] for t in ORDER]
    err = [B[t]["cer_std"] for t in ORDER]
    seg = [B[t]["segmentation_recall_proxy"] for t in ORDER]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # The error bars are the standard deviation ACROSS DOCUMENTS, not a confidence
    # interval. They are wide on the degraded tiers because failure there is
    # per-document and close to all-or-nothing, not a uniform accuracy drop.
    ax.bar(x, cer, 0.55, yerr=err, capsize=4, color="#b44", label="CER")
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in ORDER])
    ax.set_ylabel("character error rate")
    ax.grid(axis="y", alpha=0.3)

    # Twin axis: CER and segmentation ratio have different units and ranges, but
    # plotting them together is the point -- the visual correlation between them
    # is the evidence that the front end, not the classifier, drives end-to-end
    # error.
    ax2 = ax.twinx()
    ax2.plot(x, seg, "o--", c="#248", label="segmentation ratio")
    ax2.set_ylabel("segmentation ratio (1.0 = ideal)")
    ax2.set_ylim(0, 1.3)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("End-to-end error tracks segmentation quality,\n"
                 "not character-classification quality")
    fig.tight_layout()
    fig.savefig(FIGS / "cer_by_tier.png", dpi=150)
    plt.close(fig)


def fig_variational(var):
    labels = ["quantum\nuntrained", "quantum\ntrained",
              "classical\nuntrained", "classical\ntrained", "raw\npixels"]
    # Untrained entries are plain floats while trained ones carry a whole result
    # dict, hence the asymmetric indexing. All five are accuracies on the SAME
    # sealed test partition, which is what makes the bars comparable at all.
    vals = [var["untrained"]["quantum"], var["trained"]["quantum"]["final_test_acc"],
            var["untrained"]["classical"], var["trained"]["classical"]["final_test_acc"],
            var["untrained"]["raw_pixels"]]
    cols = ["#7aa", "#166", "#caa", "#844", "#888"]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(range(5), vals, 0.6, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, "{:.3f}".format(v), ha="center", fontsize=9)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=9)
    # Truncated y-axis again, for the reason given in fig_accuracy_by_tier: every
    # bar is above 0.85 and the differences under discussion are around 0.01. The
    # value labels drawn above each bar carry the exact numbers, so nothing hides.
    ax.set_ylim(0.85, max(vals) + 0.02)
    ax.set_ylabel("accuracy on sealed test partition")
    ax.axhline(var["untrained"]["raw_pixels"], ls=":", c="grey", lw=1)
    ax.set_title("Training the filter does not close the gap\n"
                 "(quantum {:+.3f}, classical {:+.3f})".format(
                     var["deltas"]["quantum"], var["deltas"]["classical"]))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "variational_training.png", dpi=150)
    plt.close(fig)


def fig_encoding(enc):
    rows = enc["rows"]
    px = [r["pixels"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))

    a1.plot(px, [r["frqi_qubits"] for r in rows], "o-", label="FRQI")
    a1.plot(px, [r["neqr_qubits"] for r in rows], "s-", label="NEQR")
    a1.set_xscale("log", base=2)
    a1.set_xlabel("pixels per patch")
    a1.set_ylabel("total qubits")
    a1.set_title("Qubit cost")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)

    a2.plot(px, [r["qubit_ratio"] for r in rows], "o-", c="#844")
    a2.set_xscale("log", base=2)
    a2.set_xlabel("pixels per patch")
    a2.set_ylabel("NEQR / FRQI qubit ratio")
    a2.set_title("Ratio NARROWS with patch size\n(Week 1 predicted widening)")
    a2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS / "encoding_scaling.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []

    res = load("results.json")
    if res:
        fig_accuracy_by_tier(res); made.append("accuracy_by_tier.png")
        fig_grover_scaling(res);   made.append("grover_scaling.png")
        fig_shot_noise(res);       made.append("shot_noise.png")
        fig_cer(res);              made.append("cer_by_tier.png")
    else:
        print("! benchmarking/results.json missing — run run_benchmark.py first")

    # Each results file is handled separately because the three experiments live in
    # three different scripts. A missing file prints a hint and costs one figure,
    # rather than aborting the whole regeneration.
    var = load("variational_results.json")
    if var:
        fig_variational(var); made.append("variational_training.png")
    else:
        print("! variational_results.json missing — run train_filter.py")

    enc = load("encoding_scaling.json")
    if enc:
        fig_encoding(enc); made.append("encoding_scaling.png")
    else:
        print("! encoding_scaling.json missing — run encoding_scaling.py")

    for m in made:
        print("  wrote figures/{}".format(m))
    print("\n{} figures in {}".format(len(made), FIGS))


if __name__ == "__main__":
    main()
