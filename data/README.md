# Data

Documents with the same/semantically similar content, deliberately varying in quality and format, for pipeline benchmarking.

## Target: ~200 files (working subset of 30–40 acceptable to start)

## Quality tiers (`processed/`)

| Folder | Description |
|---|---|
| `clean_digital/` | Born-digital PDFs/DOCX, no scanning artifacts |
| `clean_scan/` | Scanned printed documents, high DPI (300+) |
| `noisy_scan/` | Scanned printed documents, low DPI / skewed / noisy |
| `clear_handwriting/` | Legible handwritten documents |
| `degraded_handwriting/` | Partially legible / low-contrast handwritten documents |

## Formats to cover

PDF, DOCX/DOC, PNG/JPG/TIFF, TXT, RTF

## Collection status

- [ ] Working subset (30–40 files) across all tiers
- [ ] Full set (~200 files)

## Sources

Track download links / dataset sources used here as they're added (e.g. IAM Handwriting DB, RVL-CDIP, synthetic augmentation via Augraphy, etc.) — keeps the final report's data section easy to write.

**Note:** raw/processed files are gitignored (see root `.gitignore`) — don't commit large binary datasets directly. Log sources/links here instead.
