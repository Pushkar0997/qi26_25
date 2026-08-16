# Reports

LaTeX sources and compiled PDFs for the QIntern 2026 deliverables.

| File | Contents | Pages |
|---|---|---|
| `qi26_25_final_report` | Full project report | 14 |
| `week1_encoding` | FRQI vs NEQR; the refuted scaling prediction | 3 |
| `week2_features` | Quanvolutional layer, readout design, variational training | 3 |
| `week3_integration` | End-to-end pipeline; the train/serve skew | 3 |
| `week4_grover` | Comparator oracle; why sqrt(N) does not survive | 3 |
| `week5_cleaning` | **Not delivered** — what was cut and why | 3 |
| `week6_benchmarking` | Five experiments, significance, external baseline | 4 |

`weekly_preamble.tex` holds the shared formatting for the six weekly reports.

## Building

Requires a TeX distribution with `pdflatex`.

```bash
pdflatex qi26_25_final_report.tex   # run twice for the table of contents
pdflatex qi26_25_final_report.tex

for f in week*.tex; do pdflatex "$f"; done
```

## Note on figures

Every number in these reports is measured and appears in the repository, in
`benchmarking/results.json`, `seed_sweep.json`, `tesseract_baseline.json` and
`encoding_scaling.json`. If the benchmark is re-run and any figure changes, the
reports need updating to match — the same constraint that continuous integration
enforces for `docs/final_report.md`.
