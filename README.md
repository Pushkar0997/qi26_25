# qi26_25 — Hybrid QML and Quantum String Algorithms for Document Extraction

**QIntern 2026 | QWorld**
**Mentor:** Potluri Krishna Priyatham
**Team:** Pushkar Kumar (orchestration, integration, benchmarking), Klasik Taidi (Track A), Hoang Dinh Duy Anh (Track B)

---

## What this is

A hybrid document-intelligence pipeline in which two stages are quantum and the
rest is classical, benchmarked honestly against classical baselines.

```
document image
  -> projection-profile segmentation           (classical)
     -> 8x8 character crops
        -> RY encoding + quanvolutional layer  (QUANTUM — Track A)
           -> 64/160-dim features
              -> logistic-regression backend   (classical OCR head)
                 -> extracted text
                    -> Grover comparator search (QUANTUM — Track B)
                       -> pattern position
```

**Read `docs/final_report.md` first** — it contains every result, with the
figures in `benchmarking/figures/`.

## Headline findings

The results are largely negative, and deliberately reported as such:

- The quanvolutional layer performs at **parity with a dimension-matched
  classical convolution** (92.2% vs 91.9%), and **both lose to raw pixels**
  (93.6%). Training the filter does not close the gap — in the reference run it
  degraded the quantum filter by 1.1 points.
- Grover string matching works correctly, but oracle CX cost grows with a
  measured exponent of **1.39** in text length, so the √N query advantage does
  not survive data loading for stored classical text.
- The pipeline's dominant error source is the **classical segmentation front
  end**, not either quantum stage.
- The Week 1 prediction that the FRQI/NEQR qubit gap widens with patch size is
  **refuted**: it narrows, 3.33x -> 2.00x.

## Quick start

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

python data/generate_dataset.py --docs-per-tier 12 --seed 26   # ~8MB, gitignored
python integration/pipeline.py --train                          # trains OCR backend
python integration/pipeline.py --doc data/processed/clean_scan/doc_000.png --pattern 38
python benchmarking/run_benchmark.py                            # all results, ~35s
```

Full reproduction sequence including the variational and encoding experiments is
in `docs/final_report.md` §11.

## Layout

```
data/generate_dataset.py                     synthetic 5-tier dataset, seeded
integration/features.py                      quanvolutional + classical extractors
integration/pipeline.py                      end-to-end image -> text -> search
track_a_vision/encoding/encoding_scaling.py  FRQI vs NEQR across patch sizes
track_a_vision/quanvolutional/train_filter.py variational filter training
track_b_search/oracle/comparator_oracle.py   in-circuit Grover comparator
benchmarking/run_benchmark.py                5-experiment suite
benchmarking/make_figures.py                 figures from committed results
docs/final_report.md                         the report
```

## Notes on reproducibility

- The dataset (~8 MB) and trained backend are gitignored and regenerable; only
  `manifest.json` is committed, recording seed, charset, fonts and platform.
- Font rendering differs between operating systems, which shifts absolute
  accuracy by a fraction of a percent. Comparative conclusions were verified
  across two platforms and are unchanged.
- Figures are generated from `benchmarking/*.json`, so they cannot disagree with
  the reported numbers.

---
*Last updated: August 13, 2026*