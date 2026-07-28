# Track A — Vision (FRQI/NEQR → Quanvolutional → OCR handoff)

**Owner:** Klasik
**Supporting:** Pushkar (reference prototypes + review)

## Scope

1. **`encoding/`** — FRQI and NEQR pixel-to-qubit encoding, implemented in Qiskit on small grayscale image patches (start 4×4 or 8×8). Compare qubit count and reconstruction accuracy between the two schemes.
2. **`quanvolutional/`** — Quanvolutional layer: encode a patch, apply a fixed (often random) entangling circuit, measure, output becomes the feature map value. Test against a known-simple image (e.g. a clean vertical edge) before moving to real document patches.
3. **`ocr_integration/`** — Hand off quantum feature-extraction output to a lightweight classical OCR backend.

## Deliverables

- Working FRQI + NEQR encoding notebook, with qubit count + fidelity comparison
- Technical summary: encoding efficiency vs. classical normalization
- Quanvolutional layer notebook with feature map outputs on sample patches
- Notes on anything that doesn't hand off cleanly to the OCR backend — this is a legitimate finding, document it

## Reference

Pushkar has (or will have) a small working FRQI/NEQR prototype in `notebooks/` — use it as a sanity check, not a copy source. If your numbers diverge significantly from the reference, that's worth a quick sync before going further.
