# Week 1 Technical Summary — FRQI vs. NEQR Encoding

**Status:** Draft, based on verified implementation. Numbers below are from a single
4-pixel (2×2) test patch — need to re-run at larger patch sizes (Aug 4×4 →) before
this is final.

## What we built

Implemented both FRQI and NEQR pixel-to-qubit encoding schemes in Qiskit and
verified correctness via simulation:

- **FRQI:** one color qubit whose rotation angle encodes pixel brightness,
  entangled with position qubits via multi-controlled RY gates.
- **NEQR:** exact binary pixel value across 8 color qubits, entangled with
  position qubits via multi-controlled X gates.

## Verification method

- **NEQR:** measured the full circuit (4096 shots), decoded (position, color)
  pairs from the bitstrings, compared against ground truth.
- **FRQI:** estimated `P(color=1 | position=i) = sin²(θᵢ)` from measurement
  statistics (8192 shots), inverted to recover `θᵢ`, converted back to a pixel
  value estimate.

## Results — 2×2 patch (4 pixels), pixel values [0, 255, 128, 64]

**NEQR — exact reconstruction, all 4 pixels:**

| Position | Expected | Recovered |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 255 | 255 |
| 2 | 128 | 128 |
| 3 | 64 | 64 |

**FRQI — approximate reconstruction (8192 shots):**

| Position | True pixel | Recovered (estimated) |
|---|---|---|
| 0 | 0 | 0.0 |
| 1 | 255 | 255.0 |
| 2 | 128 | 128.5 |
| 3 | 64 | 65.5 |

FRQI recovery is close but not exact — error is on the order of 1-2 pixel
values out of 255 at this shot count, and would tighten further with more
shots but never reach zero error.

## Qubit / gate-count comparison

| | FRQI | NEQR |
|---|---|---|
| Total qubits | 3 | 10 |
| Circuit depth (transpiled, u+cx basis) | 38 | 103 |
| CX gate count | 24 | 60 |
| Reconstruction | Approximate | Exact |

**On this patch, NEQR costs 3.3× the qubits and ~2.5× the CX gates of FRQI**,
in exchange for exact rather than approximate reconstruction.

## Why this matters for our pipeline

NEQR's qubit cost scales with bit depth (8 qubits per pixel, fixed) while
FRQI's stays at 1 color qubit regardless of patch size — so this gap should
widen, not shrink, as we move to realistic patch sizes. For OCR specifically,
we lean toward NEQR being worth the extra qubit cost, since character
recognition needs exact pixel values — small brightness errors compounding
across a multi-stage pipeline (encoding → quanvolutional layer → OCR) could
plausibly corrupt which character gets read. This is a hypothesis to test
once the quanvolutional layer stage is built, not yet a confirmed result.

## Next steps

1. Re-run this comparison at 4×4 (16 pixels) and, if feasible on simulation,
   8×8 — confirm the qubit/gate-count gap direction holds at scale.
2. Test on a real document image crop instead of a synthetic patch.
3. Feed both encodings into the quanvolutional layer (Phase 2) and compare
   downstream feature-extraction quality, not just encoding fidelity in
   isolation.

---
*Pushkar Kumar — July 28, 2026*
