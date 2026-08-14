# qi26_25 — Hybrid QML and Quantum String Algorithms for Document Extraction

**QIntern 2026 | QWorld**
**Mentor:** Potluri Krishna Priyatham
**Author:** Pushkar Kumar — all commits in this repository are single-authored.
**Originally scoped for three:** Klasik Taidi (Track A) and Hoang Dinh Duy Anh
(Track B) were assigned tracks that were not taken up; see Section 11 of the
report.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Chosen for consistency with Qiskit,
which this project builds on, and for its explicit patent grant.

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
- Measured over 30 splits, quantum-vs-classical parity is a **null result**
  (p=0.23, 15/30 splits won) while the raw-pixel lead is real (p=6e-12, 30/30).
- Against **Tesseract** on the same images, this pipeline is 3-4x worse on clean
  input (6.6% vs 2.0% character error). It is not competitive with mature
  classical OCR; the contribution is the measured characterisation of why.
- The pipeline's dominant error source is the **classical segmentation front
  end**, not either quantum stage. Clean-document CER is 6.5-8.8%, and **100% of
  clean digital documents have their identifier recovered exactly** end-to-end
  through both quantum stages (50% on clean scans, 0% on degraded input).
- The largest single improvement came from fixing a **train/serve skew** (crops
  cut from ground-truth bounds in training vs segmentation bounds at inference),
  which cut clean-document CER 6.6x. It was found because a *better* feature set
  produced *worse* end-to-end output - isolated-character accuracy turned out to
  be a misleading proxy for pipeline quality.
- The Week 1 prediction that the FRQI/NEQR qubit gap widens with patch size is
  **refuted**: it narrows, 3.33x -> 2.00x.

## Live demo

**https://pushkar0997.github.io/qi26_25/** — upload a document image and watch
each pipeline stage run in the browser. No image leaves the machine.

The demo is not a re-implementation: it loads the exact trained weights and
filter unitary from this repository, and `web/verify_parity.py` asserts
stage by stage that it produces byte-identical output to the Python pipeline
(features agree to 5.6e-16; decoded text is identical).

## Notebooks

Six runnable notebooks in `notebooks/`, in order. Each is self-contained: the
first cell clones this repo and installs dependencies if needed, so they open
directly in Colab with no setup.

| Notebook | Contents |
|---|---|
| `00_start_here` | Environment, dataset, one full pipeline run |
| `01_encoding_frqi_neqr` | Pixels to quantum states; the refuted Week 1 hypothesis |
| `02_quanvolutional_features` | The quantum feature extractor and why it loses to raw pixels |
| `03_grover_comparator` | A real string-matching oracle; why sqrt(N) does not survive |
| `04_full_pipeline` | All nine stages in detail, plus the two bugs that cost most |
| `05_noise_and_runtimes` | Shot/gate/readout noise, GPU, simulation limits |

Total 162 cells, all executed with outputs saved.

## Quick start

The dataset and the trained model are committed, so nothing needs to be
generated before verifying any result:

```bash
git clone https://github.com/Pushkar0997/qi26_25.git
cd qi26_25
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

python benchmarking/run_benchmark.py     # all reported results, ~30s
python web/verify_parity.py              # confirms the demo matches Python
python benchmarking/seed_sweep.py        # significance over 30 splits
```

Run one document end to end:

```bash
python integration/pipeline.py --doc data/processed/clean_scan/doc_000.png --pattern 38
```

**The dataset should not be regenerated except to change it deliberately.** Generation
renders text with whatever fonts the machine provides, and different fonts
change segmentation: on a machine with thinner glyph rendering, clean-document
character error rose from 6.5% to 63.7% in testing. The generator refuses to
overwrite committed data without `--force` for this reason. Regenerating also
requires retraining and re-exporting:

```bash
python data/generate_dataset.py --docs-per-tier 12 --seed 26 --force
python integration/pipeline.py --train
python web/export_model.py
```

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

- The dataset (~8 MB) and the trained backend (20 KB) **are committed**, so a
  fresh clone reproduces every reported number without generating or training
  anything. They were previously gitignored as derived data; that was wrong,
  because generation depends on machine-installed fonts and training is not
  bit-reproducible across BLAS builds.
- `data/processed/manifest.json` records the seed, charset, fonts and platform
  used for the reported figures.
- Figures are generated from the committed results files, so they cannot
  disagree with the tables in the report.
- Continuous integration runs the benchmark, the Grover oracle self-test and the
  browser-parity check on every push.

## Limitations

Stated here rather than only in the report, because they bound what the results
mean:

- **Synthetic data.** 60 rendered documents, one font per tier. Exact ground
  truth is what makes character error rate computable at all, but real scans
  differ.
- **The handwriting tiers are a proxy**, not handwriting: an italic serif face
  with per-glyph position, rotation and scale jitter. Real handwritten data
  (IAM database) is the correct extension.
- **Small scale.** 6,023 characters, 8x8 crops, a 36-class charset and a 4-qubit
  filter. Conclusions about quanvolutional layers are conclusions about this
  regime, not about the approach in general.
- **Three of five tiers fail.** Clean digital and clean scan work; noisy scans
  and both handwriting tiers have character error rates of 52-88% and recover
  no identifiers. The pipeline is usable on clean digital input and not usable
  on degraded input.

---
*Last updated: August 14, 2026*