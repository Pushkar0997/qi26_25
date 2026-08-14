> **Superseded.** This is the plan written on 10 August, retained as a record of
> how the remaining work was scheduled. Results, conclusions and final status are
> in [`final_report.md`](final_report.md).

---

# Completion Plan — Aug 10 to Aug 13

**Project:** Hybrid QML and Quantum String Algorithms for Efficient Document Extraction and Processing
**QIntern 2026 | QWorld | Mentor:** Potluri Krishna Priyatham
**Repo:** github.com/Pushkar0997/qi26_25
**Written:** August 10, 2026 — 4 days to deadline

---

## Status at the time of writing

The project was scoped for three participants. In practice the two track
assignments were not taken up and the mentor was unreachable from early July, so
the remaining work was consolidated into a single-person scope. This plan was
written on that basis.

At the point of writing, the following had just been completed: the synthetic
document dataset, a comparator-based Grover oracle replacing the earlier
marked-state version, a batched quanvolutional feature extractor, the end-to-end
pipeline, and the benchmark suite.

## Scope: what is in, what is cut

Cutting scope is not failure here — it is the only way the remaining work fits
four days. Every cut is recorded so the report can state it explicitly.

**In scope (will be completed and reported):**

- Phase 1 — FRQI/NEQR encoding comparison, extended to 4×4
- Phase 2 — Quanvolutional feature extraction, plus the trained (variational) variant
- Phase 3 — Integration: working end-to-end extraction script
- Phase 4 — Grover string matching with a real comparator oracle
- Phase 6 — Benchmarking and final report

**Cut, with reason (state these in the report, do not hide them):**

| Original scope | Decision | Reason |
|---|---|---|
| 200-document dataset | Cut to 60 synthetic docs | Hand-labelling scraped documents is not possible in 4 days, and without labels CER cannot be computed at all. Synthetic gives exact ground truth and controlled degradation. |
| Real handwriting | Cut, proxy used | Italic font + per-glyph jitter stands in for handwriting variability. IAM Handwriting DB named as the correct extension. |
| Week 5: quantum string alignment / OCR noise cleaning | Cut | Depends on a high-accuracy OCR stage that did not exist at this point; attempting it would have produced a broken component rather than a result. |
| LLM baseline comparison | Cut | No API budget or time. Classical-OCR baseline is retained and is the more informative control anyway. |

---

## Daily milestones

Each day has a single hard deliverable. If a day's deliverable is not committed
by end of day, the next day starts by finishing it, and the *last* item in the
plan gets dropped rather than everything sliding.

### Aug 10 — completed

- Dataset generator + 6,027 labelled crops across 5 tiers
- Real comparator oracle replacing the hardcoded-answer starter oracle
- Quanvolutional extractor + classical baselines at matched dimensionality
- End-to-end pipeline script
- Benchmark harness, all 5 experiments producing measured numbers

### Aug 11 — Close the two open scientific questions

1. **Train the quantum filter (Phase 2's actual deliverable).**
   The brief asks for a *variational* circuit; the implementation at this point
   used a fixed random filter, and the benchmark shows it losing to raw pixels (88.7% vs 94.8%).
   The open question is whether that gap is caused by the filter being quantum
   or by it being untrained. Optimise the filter's rotation angles against
   classification loss and re-measure.
   *Definition of done:* trained-filter accuracy recorded in the results table
   next to the untrained number, either way it comes out.

2. **FRQI/NEQR at 4×4** — the Week 1 summary still says "need to re-run at
   larger patch sizes". Extend to 16 pixels, confirm whether the qubit/CX gap
   widens as predicted.
   *Definition of done:* `docs/week1_frqi_neqr_technical_summary.md` updated with
   4×4 numbers and the hypothesis marked confirmed or refuted.

3. **Notebooks.** Convert `comparator_oracle.py` into a notebook with the three
   verification stages shown as output cells. Mentors read notebooks; a `.py`
   file is less legible as a deliverable.

### Aug 12 — Figures and full report draft

1. Plots into `benchmarking/figures/`: accuracy by tier, Grover CX scaling
   (log-log, with the 1.39 exponent fitted), shot-noise curve, CER by tier.
2. Complete `docs/final_report.md` — every section written, no placeholders.
3. Re-run `run_benchmark.py` end to end from a clean checkout to confirm
   reproducibility.

*Definition of done:* report readable start to finish by someone who has not
seen the repo.

### Aug 13 — Wrap

1. Notebook cleanup, clear all stale outputs, re-run top to bottom.
2. README updated to describe what the project actually does and how to run it.
3. Final read-through of the report.
4. Submit.

**Reserve the last 3 hours for submission mechanics.** Do not schedule work into
them.

---

## Command reference

```bash
# regenerate the dataset (deterministic, seeded)
python data/generate_dataset.py --docs-per-tier 12 --seed 26

# train the classical OCR backend on quantum features
python integration/pipeline.py --train

# run one document end to end, with Grover search for a pattern
python integration/pipeline.py --doc data/processed/clean_scan/doc_000.png --pattern 38

# full benchmark suite -> benchmarking/results.{json,md}
python benchmarking/run_benchmark.py

# Track B oracle staged verification
python track_b_search/oracle/comparator_oracle.py
```

---

*Last updated: August 10, 2026*

---

## Outcome

All items above were completed. Work not anticipated by this plan was added
afterwards: six runnable notebooks, a browser demo verified to produce
byte-identical output to the Python pipeline, a noise and hardware-limits
analysis, and cross-platform reproducibility work.

Two items resolved differently from the expectation recorded here. The
variational filter experiment initially showed a 1.4-point gain that proved to be
data leakage; with a sealed test partition the gain is 0.1 points. And the
largest single improvement in the project — a 6.6x reduction in document error
rate — came from correcting a train/serve skew that this plan does not mention,
because it had not yet been identified.

*Last updated: August 14, 2026*