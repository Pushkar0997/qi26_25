"""
encoding_scaling.py — FRQI vs NEQR resource comparison across patch sizes.

Closes the open item in docs/week1_frqi_neqr_technical_summary.md, which
recorded 2x2 numbers and stated the hypothesis that the qubit and gate-count
gap between the two encodings widens with patch size, without testing it.

The prediction, restated precisely:
  - FRQI colour cost is FIXED at 1 qubit regardless of pixel count.
  - NEQR colour cost is FIXED at bit_depth (8) qubits regardless of pixel count.
  - Both position registers grow as ceil(log2(n_pixels)).
So the qubit RATIO should NARROW as patches grow (the constant 8-vs-1 colour gap
is amortised over a growing shared position register), while the absolute gate
counts should both grow roughly linearly in pixel count.

Note this is the opposite of what the Week 1 summary predicted. The summary said
the gap "should widen, not shrink" — that reasoning conflated the qubit gap with
the gate-count gap. This script measures both and reports which way each goes.

Usage:
    python track_a_vision/encoding/encoding_scaling.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import RYGate, XGate

ROOT = Path(__file__).resolve().parent.parent.parent


def build_frqi_circuit(patch):
    """Unchanged from frqi_neqr_starter.ipynb."""
    flat = patch.flatten()
    n_pixels = len(flat)
    n_pos = int(np.ceil(np.log2(n_pixels)))
    angles = (np.pi / 2) * (flat / 255.0)

    pos = QuantumRegister(n_pos, "pos")
    color = QuantumRegister(1, "color")
    qc = QuantumCircuit(pos, color)
    qc.h(pos)

    for i, theta in enumerate(angles):
        bits = format(i, "0{}b".format(n_pos))
        for j, b in enumerate(bits):
            if b == "0":
                qc.x(pos[j])
        qc.append(RYGate(2 * theta).control(n_pos), list(pos) + [color[0]])
        for j, b in enumerate(bits):
            if b == "0":
                qc.x(pos[j])
    return qc, n_pos + 1


def build_neqr_circuit(patch, bit_depth=8):
    """Unchanged from frqi_neqr_starter.ipynb."""
    flat = patch.flatten()
    n_pixels = len(flat)
    n_pos = int(np.ceil(np.log2(n_pixels)))

    pos = QuantumRegister(n_pos, "pos")
    color = QuantumRegister(bit_depth, "color")
    qc = QuantumCircuit(pos, color)
    qc.h(pos)

    for i, val in enumerate(flat):
        bp = format(i, "0{}b".format(n_pos))
        bv = format(int(val), "0{}b".format(bit_depth))
        for j, b in enumerate(bp):
            if b == "0":
                qc.x(pos[j])
        for k, b in enumerate(bv):
            if b == "1":
                qc.append(XGate().control(n_pos), list(pos) + [color[k]])
        for j, b in enumerate(bp):
            if b == "0":
                qc.x(pos[j])
    return qc, n_pos + bit_depth


def measure(qc):
    t = transpile(qc, basis_gates=["u", "cx"], optimization_level=1)
    ops = t.count_ops()
    return {"depth": t.depth(), "cx": ops.get("cx", 0),
            "total_gates": sum(ops.values())}


def main():
    rng = np.random.default_rng(26)
    rows = []

    print("{:>7} {:>8} {:>18} {:>18} {:>10}".format(
        "patch", "pixels", "FRQI (q/depth/cx)", "NEQR (q/depth/cx)", "qubit x"))

    for side in [2, 4, 8]:
        patch = rng.integers(0, 256, (side, side))

        f_qc, f_q = build_frqi_circuit(patch)
        n_qc, n_q = build_neqr_circuit(patch)
        f, n = measure(f_qc), measure(n_qc)

        row = {
            "patch": "{0}x{0}".format(side), "pixels": side * side,
            "frqi_qubits": f_q, "frqi_depth": f["depth"], "frqi_cx": f["cx"],
            "neqr_qubits": n_q, "neqr_depth": n["depth"], "neqr_cx": n["cx"],
            "qubit_ratio": round(n_q / f_q, 3),
            "cx_ratio": round(n["cx"] / max(1, f["cx"]), 3),
        }
        rows.append(row)
        print("{:>7} {:>8} {:>18} {:>18} {:>10.2f}".format(
            row["patch"], row["pixels"],
            "{}/{}/{}".format(f_q, f["depth"], f["cx"]),
            "{}/{}/{}".format(n_q, n["depth"], n["cx"]),
            row["qubit_ratio"]))

    print("\nQubit ratio (NEQR/FRQI) across sizes: " +
          " -> ".join("{:.2f}".format(r["qubit_ratio"]) for r in rows))
    print("CX ratio   (NEQR/FRQI) across sizes: " +
          " -> ".join("{:.2f}".format(r["cx_ratio"]) for r in rows))

    qr = [r["qubit_ratio"] for r in rows]
    verdict = ("NARROWS" if qr[-1] < qr[0] else
               "WIDENS" if qr[-1] > qr[0] else "FLAT")
    print("\nQubit-cost ratio {} with patch size.".format(verdict))
    print("The Week 1 summary predicted the gap would widen; measured, the "
          "qubit ratio {} because FRQI and NEQR share an identically growing "
          "position register while their colour registers stay constant."
          .format(verdict.lower()))

    out = ROOT / "benchmarking" / "encoding_scaling.json"
    out.write_text(json.dumps({"rows": rows, "qubit_ratio_verdict": verdict},
                              indent=2))
    print("\nWrote", out)


if __name__ == "__main__":
    main()
