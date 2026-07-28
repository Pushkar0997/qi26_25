# qi26_25 — Hybrid QML and Quantum String Algorithms for Efficient Document Extraction and Processing

**QIntern 2026 | QWorld**
**Mentor:** Potluri Krishna Priyatham
**Team:** Pushkar Kumar (orchestration + integration), Klasik Taidi (Track A), Hoang Dinh Duy Anh (Track B)
**Timeline:** July 1 – August 13, 2026

---

## What this project does

Builds a resource-efficient document intelligence pipeline that replaces parts of a classical LLM-based OCR pipeline with quantum-native components:

1. **FRQI/NEQR quantum image encoding** — represent document image patches as quantum states
2. **Quanvolutional layers** — quantum feature extraction, standing in for classical CNN filters
3. **Grover-based string matching** — quadratic-speedup pattern search for structured identifiers (e.g., ID numbers)
4. **Benchmarking** — quantum pipeline vs. classical OCR vs. LLM-based parsing, on accuracy and compute cost

This is exploratory, honest benchmarking work — not a claim that quantum beats classical. See `docs/` for the technical write-ups as they land.

---

## Repo structure

```
qi26_25/
├── track_a_vision/          # Klasik — encoding + quantum feature extraction
│   ├── encoding/             # FRQI + NEQR implementations
│   ├── quanvolutional/       # Quanvolutional layer implementations
│   └── ocr_integration/      # Handoff to classical OCR backend
├── track_b_search/           # Anh — Grover string matching
│   └── oracle/                # Oracle construction, toy → scaled tests
├── data/
│   ├── raw/                   # Original collected documents (not committed if large — see .gitignore)
│   └── processed/             # Sorted by quality tier
│       ├── clean_digital/
│       ├── clean_scan/
│       ├── noisy_scan/
│       ├── clear_handwriting/
│       └── degraded_handwriting/
├── notebooks/                # Shared exploratory notebooks
├── integration/               # Phase 3 — full pipeline (Pushkar)
├── benchmarking/              # Phase 5 — comparison results, plots
└── docs/                      # Technical summaries, weekly notes, final report
```

---

## Working norms

- Each track owner works primarily in their own folder. Commit directly to `main` — team is small and timeline is tight, we're not doing heavyweight branching.
- Commit messages should reference the phase/task (e.g. `[Phase 1] FRQI encoding on 8x8 patch, qubit count + fidelity check`) — these become source material for the final report.
- Notebooks in `notebooks/` are for exploration; once something's stable, move the clean version into the relevant track folder.
- Large data files: keep out of git, note the source/download instructions in `data/README.md` instead.

See the full phase-by-phase plan in the shared project plan doc (outside this repo).

---

*Last updated: July 26, 2026*
