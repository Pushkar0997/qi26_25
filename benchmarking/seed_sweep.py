"""
seed_sweep.py — repeat Experiment A across many train/test splits.

WHY THIS EXISTS
---------------
Experiment A in run_benchmark.py reports a single split (seed 26). That is fine
for a point estimate but it cannot tell a reader which differences are real.
The headline comparison contains two gaps of very different character:

    quanv (+ZZ) 92.2%  vs  classical conv 91.9%   -> 0.3 points
    quanv (+ZZ) 92.2%  vs  raw pixels     93.6%   -> 1.4 points

Reported as bare numbers, both look like results. Repeating over many splits
shows that the first is indistinguishable from noise while the second is not,
which is exactly the distinction the report's conclusions rest on.

Paired testing is used throughout: every extractor is scored on the SAME splits,
so the comparison is within-split and removes split-to-split variance, which is
the dominant source of noise here.

Usage:
    python benchmarking/seed_sweep.py --seeds 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "integration"))

from features import (quanv_features, classical_conv, raw_pixels,   # noqa: E402
                      normalize_crops)
from sklearn.linear_model import LogisticRegression                 # noqa: E402
from sklearn.model_selection import train_test_split                # noqa: E402

TIERS = ["clean_digital", "clean_scan", "noisy_scan",
         "clear_handwriting", "degraded_handwriting"]


def load_all():
    X = np.concatenate([np.load(ROOT / "data/processed" / t / "chars.npz")["crops"]
                        for t in TIERS])
    y = np.concatenate([np.load(ROOT / "data/processed" / t / "chars.npz")["labels"]
                        for t in TIERS])
    return X, y


def paired_t(diffs):
    """Two-sided paired t-test against a null of zero mean difference.

    Implemented directly rather than pulled from scipy.stats so the whole
    computation is visible: with n paired observations, t = mean / (sd / sqrt(n))
    on n-1 degrees of freedom. The p-value uses the survival function of the t
    distribution, which scipy provides.
    """
    from scipy import stats
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(t), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", type=str, default="benchmarking/seed_sweep.json")
    args = ap.parse_args()

    X, y = load_all()

    # Per-crop contrast normalisation, exactly as the benchmark does it.
    # Omitting this changes the result enough to reverse the headline finding,
    # because the degraded tiers compress into a narrow intensity band without
    # it and every extractor is then measured on different inputs.
    Xn = normalize_crops(X)
    print("crops: {}, classes: {}".format(X.shape, len(np.unique(y))))

    # Features are computed ONCE. Only the train/test split varies across seeds,
    # which is the source of variance being measured; recomputing features would
    # add nothing but time.
    feats = {
        "quanv (marginals)": quanv_features(Xn),
        "quanv (+ZZ corr)": quanv_features(Xn, correlations=True),
        "classical conv": classical_conv(Xn, n_filters=10),
        "raw pixels": raw_pixels(Xn),
    }

    scores = {k: [] for k in feats}
    print("\nrunning {} splits ...".format(args.seeds))

    # Seeds 0..n-1 rather than random draws, so the sweep is reproducible and a
    # reviewer re-running it lands on the same splits.
    for s in range(args.seeds):
        for name, F in feats.items():
            a, b, c, d = train_test_split(F, y, test_size=0.25,
                                          random_state=s, stratify=y)
            scores[name].append(
                LogisticRegression(max_iter=3000, C=5.0).fit(a, c).score(b, d))
        if (s + 1) % 10 == 0:
            print("  {}/{}".format(s + 1, args.seeds))

    print("\n{:22s} {:>8s} {:>8s} {:>16s}".format(
        "extractor", "mean", "std", "95% interval"))
    summary = {}
    for name, v in scores.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        summary[name] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                         "lo": float(lo), "hi": float(hi),
                         "scores": [float(x) for x in v]}
        print("{:22s} {:8.4f} {:8.4f}   [{:.4f}, {:.4f}]".format(
            name, v.mean(), v.std(ddof=1), lo, hi))

    # The two comparisons the report's conclusions actually rest on.
    print("\npaired comparisons across the same {} splits:".format(args.seeds))
    # Only the comparisons the report's conclusions depend on. Testing every
    # pair would invite multiple-comparison problems for no benefit: three of
    # the six pairs are not claims the report makes.
    comparisons = [
        ("quanv (+ZZ corr)", "classical conv"),
        ("raw pixels", "quanv (+ZZ corr)"),
        ("raw pixels", "classical conv"),
        ("quanv (+ZZ corr)", "quanv (marginals)"),
    ]
    tests = {}
    for a, b in comparisons:
        d = np.array(scores[a]) - np.array(scores[b])
        t, p = paired_t(d)
        wins = int((d > 0).sum())
        verdict = ("significant" if p < 0.01 else
                   "not distinguishable from noise")
        tests["{} - {}".format(a, b)] = {
            "mean_diff": float(d.mean()), "t": t, "p": p,
            "wins": wins, "n": len(d), "verdict": verdict}
        print("  {:>18s} - {:<18s} {:+.4f}  p={:<9.2g} {}/{} splits  {}".format(
            a, b, d.mean(), p, wins, len(d), verdict))

    out = ROOT / args.out
    out.write_text(json.dumps({"n_seeds": args.seeds, "summary": summary,
                               "tests": tests}, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
