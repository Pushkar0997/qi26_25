# Completion Plan — Aug 10 to Aug 13

**Project:** Hybrid QML and Quantum String Algorithms for Efficient Document Extraction and Processing
**QIntern 2026 | QWorld | Mentor:** Potluri Krishna Priyatham
**Repo:** github.com/Pushkar0997/qi26_25
**Written:** August 10, 2026 — 4 days to deadline

---

## Status, stated plainly

Prior plans (Aug 6 sprint plan, reset plan) assumed a three-person team. That
assumption did not hold: the repository has 5 commits, all from one author, and
the last was August 2. Track A and Track B were never picked up by their
assigned owners.

This plan therefore drops the two-track division entirely and is written for one
person. It is not a re-issue of the previous split with tighter deadlines — that
approach has already been tried twice and did not produce commits.

**What changed today (Aug 10).** The critical path was unblocked in one session.
The following now exist, run, and are verified:

| Component | State |
|---|---|
| `data/generate_dataset.py` | 60 documents, 6,027 labelled character crops, 5 quality tiers |
| `track_b_search/oracle/comparator_oracle.py` | Real in-circuit comparator oracle, 3-stage verification passing |
| `integration/features.py` | Quanvolutional extractor + matched classical baselines |
| `integration/pipeline.py` | End-to-end: image → text → Grover search |
| `benchmarking/run_benchmark.py` | 5 experiments, all numbers measured |

The project moved from "Phases 3–6 not started" to "Phases 3–6 running with
results" in a single day. What remains is genuinely finishable by Aug 13.

---

## The one thing that must be said to the mentor today

Send this before the next check-in, not on Aug 13. Suggested wording:

> The project is on track for submission on Aug 13, but I want to flag the team
> situation honestly rather than at the deadline. The repository shows all
> commits from me; my two teammates have not contributed code or responded to
> the last several check-ins. I have restructured the remaining work as a
> single-person scope and the pipeline is now running end to end with
> benchmark results. I would rather tell you this now than have it surface
> when you look at the commit history.

This protects you. A mentor who learns about it on the 13th has no way to help;
one who learns on the 10th does.

---

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
| Week 5: quantum string alignment / OCR noise cleaning | Cut | Depends on a working high-accuracy OCR stage that we do not have. Attempting it would produce a broken component rather than a result. |
| LLM baseline comparison | Cut | No API budget or time. Classical-OCR baseline is retained and is the more informative control anyway. |

---

## Daily milestones

Each day has a single hard deliverable. If a day's deliverable is not committed
by end of day, the next day starts by finishing it, and the *last* item in the
plan gets dropped rather than everything sliding.

### Aug 10 (today) — DONE

- [x] Dataset generator + 6,027 labelled crops across 5 tiers
- [x] Real comparator oracle replacing the hardcoded-answer starter oracle
- [x] Quanvolutional extractor + classical baselines at matched dimensionality
- [x] End-to-end pipeline script
- [x] Benchmark harness, all 5 experiments producing measured numbers
- [ ] **Commit and push everything.** Nothing above counts until it is pushed.
- [ ] Send the mentor note above.

### Aug 11 — Close the two open scientific questions

1. **Train the quantum filter (Phase 2's actual deliverable).**
   The brief asks for a *variational* circuit; what we have is a fixed random
   filter, and the benchmark shows it losing to raw pixels (88.7% vs 94.8%).
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

## Meeting cadence

Every alternate day, 7:30 PM IST, as before: **Aug 10, Aug 12**, wrap on Aug 13.

Given that the previous cadence produced no attendance, the rule for these is:

- The meeting happens whether or not anyone else joins.
- If a teammate joins and wants work, give them something **off the critical
  path** — figures, report proofreading, notebook cleanup. Do not hand over a
  blocking item four days out; a task that arrives incomplete on Aug 13 costs
  more than doing it yourself on Aug 11.
- Log attendance. The report's contribution section should reflect what actually
  happened, stated neutrally and without editorialising.

---

## Working rules for the next four days

1. **Push the same day.** Eight days of local-only work is what produced the
   current situation. An unpushed commit is invisible to the mentor.
2. **Measured numbers only.** Every figure in the report traces to
   `benchmarking/results.json`. No estimates, no "approximately".
3. **Negative results are results.** The benchmark currently shows the quantum
   feature layer losing to raw pixels. That is reportable, and honestly framed
   it is stronger than an unsupported claim of advantage. Do not tune the
   experiment until quantum wins.
4. **If something won't make it, write it down on Aug 12, not Aug 13.** A
   documented "we attempted X, here is what we found and why it did not
   complete" is a legitimate part of a research report.

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
