"""
features.py — feature extractors for the OCR stage.

Three extractors share one interface, so the benchmark can swap them and
attribute any accuracy difference to the feature stage alone:

    quanv_features(...)     quantum: quanvolutional layer (Track A)
    classical_conv(...)     classical: random convolution filters, matched in
                            shape and parameter count to the quantum filter
    raw_pixels(...)         no feature extraction, flattened pixels

WHY THE QUANTUM PATH IS FAST HERE
---------------------------------
The starter notebook runs one Aer job per patch. At 6k characters x 16 patches
that is ~96k simulator jobs, which is hours of wall clock and makes the whole
benchmark impossible to iterate on.

The filter is a FIXED circuit, so its unitary U is the same for every patch. We
extract U from Qiskit once, then apply it with numpy. The encoding produces a
product state, so the input statevector for a patch is a Kronecker product of
four single-qubit states, and every patch in the dataset can be pushed through
U as one batched matrix multiply.

This is not an approximation. It is the same linear algebra Aer performs, minus
the per-job overhead and minus shot noise. `verify_against_qiskit()` asserts
agreement with the notebook's shot-based implementation. Sampling noise can be
reintroduced with `shots=...`, which draws from the exact outcome distribution.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


# --------------------------------------------------------------------------
# Quantum filter
# --------------------------------------------------------------------------

def random_entangling_circuit(n_qubits, seed=42, depth=2):
    """The Track A filter, unchanged from quanvolutional_starter.ipynb."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(depth):
        for q in range(n_qubits):
            qc.ry(rng.uniform(0, 2 * np.pi), q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(n_qubits - 1, 0)   # ring entanglement
    return qc


def entangling_circuit_from_angles(angles, n_qubits, depth=2):
    """Same architecture as random_entangling_circuit, but with the RY angles
    supplied rather than drawn from an RNG. This is what makes the filter
    variational: the angles become trainable parameters."""
    angles = np.asarray(angles, dtype=float).reshape(depth, n_qubits)
    qc = QuantumCircuit(n_qubits)
    for d in range(depth):
        for q in range(n_qubits):
            qc.ry(float(angles[d, q]), q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(n_qubits - 1, 0)
    return qc


def initial_angles(n_qubits, seed=42, depth=2):
    """The angles the untrained random filter happens to use, so training can
    start from exactly the baseline configuration and any improvement is
    attributable to optimisation alone."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(depth):
        out.extend(rng.uniform(0, 2 * np.pi, n_qubits))
        # the CX ring consumes no randomness; kept in step with the original
    return np.array(out)


def _filter_unitary(n_qubits, seed, depth):
    return np.asarray(Operator(random_entangling_circuit(n_qubits, seed, depth)).data)


def _unitary_from_angles(angles, n_qubits, depth):
    return np.asarray(Operator(
        entangling_circuit_from_angles(angles, n_qubits, depth)).data)


def _encode_batch(patches):
    """Patches -> batched statevectors, matching qc.ry(pi*v/255, i) encoding.

    Each qubit i starts in |0> and receives RY(theta_i), giving
    [cos(theta/2), sin(theta/2)]. The joint state is the Kronecker product.
    Qiskit is little-endian: qubit 0 is the LEAST significant index, so the
    product must be built from the highest-index qubit down.
    """
    patches = np.asarray(patches, dtype=float)
    n = patches.shape[0]
    flat = patches.reshape(n, -1)
    theta = np.pi * (flat / 255.0)
    c, s = np.cos(theta / 2), np.sin(theta / 2)

    n_qubits = flat.shape[1]
    state = np.ones((n, 1), dtype=complex)
    for q in range(n_qubits - 1, -1, -1):          # little-endian ordering
        qubit = np.stack([c[:, q], s[:, q]], axis=1)
        state = (state[:, :, None] * qubit[:, None, :]).reshape(n, -1)
    return state


def quanv_patch_probs(patches, n_qubits=4, seed=42, depth=2, shots=None, rng=None,
                      correlations=False, angles=None):
    """Per-qubit P(measure 1), optionally plus <Z_i Z_j> pairwise correlations.

    Output width is n_qubits, or n_qubits + C(n_qubits, 2) when correlations
    are enabled."""
    U = (_unitary_from_angles(angles, n_qubits, depth) if angles is not None
         else _filter_unitary(n_qubits, seed, depth))
    states = _encode_batch(patches) @ U.T
    probs = np.abs(states) ** 2                     # (n, 2**n_qubits)

    if shots is not None:
        # Reintroduce measurement noise by sampling the exact distribution.
        rng = rng or np.random.default_rng(0)
        sampled = np.empty_like(probs)
        for i in range(probs.shape[0]):
            p = np.clip(probs[i], 0, None)
            sampled[i] = rng.multinomial(shots, p / p.sum()) / shots
        probs = sampled

    # Marginalise: qubit q is bit q of the outcome index (little-endian).
    idx = np.arange(probs.shape[1])
    bits = np.stack([(idx >> q) & 1 for q in range(n_qubits)], axis=0)  # (nq, 2**nq)

    out = [probs[:, bits[q] == 1].sum(axis=1) for q in range(n_qubits)]

    if correlations:
        # Single-qubit marginals discard every correlation the entangling layer
        # creates: the filter produces a 2**n-outcome distribution and keeping
        # only n marginals throws most of it away. <Z_i Z_j> recovers the
        # pairwise structure, which is the part of the output that a product
        # state could not have produced.
        z = 1.0 - 2.0 * bits                      # map bit 0/1 -> +1/-1
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                out.append((probs * (z[i] * z[j])).sum(axis=1))

    return np.stack(out, axis=1)


def quanv_features(images, patch_size=2, n_qubits=4, seed=42, depth=2,
                   stride=None, shots=None, correlations=False, angles=None):
    """Apply the quanvolutional layer across a batch of images.

    images: (n, H, W) uint8/float. Returns (n, out_h * out_w * n_qubits).
    """
    images = np.asarray(images, dtype=float)
    if images.ndim == 2:
        images = images[None, ...]
    n, H, W = images.shape
    stride = stride or patch_size
    out_h = (H - patch_size) // stride + 1
    out_w = (W - patch_size) // stride + 1

    # Collect every patch from every image, run one batched pass, reshape back.
    patches = np.empty((n * out_h * out_w, patch_size * patch_size))
    k = 0
    for oy in range(out_h):
        for ox in range(out_w):
            y, x = oy * stride, ox * stride
            block = images[:, y:y + patch_size, x:x + patch_size]
            patches[k * n:(k + 1) * n] = block.reshape(n, -1)
            k += 1

    probs = quanv_patch_probs(patches, n_qubits=n_qubits, seed=seed,
                              depth=depth, shots=shots,
                              correlations=correlations, angles=angles)
    n_feat = probs.shape[1]
    probs = probs.reshape(out_h * out_w, n, n_feat)
    return probs.transpose(1, 0, 2).reshape(n, -1)


# --------------------------------------------------------------------------
# Classical baselines
# --------------------------------------------------------------------------

def classical_conv(images, patch_size=2, n_filters=4, seed=42, stride=None):
    """Random-filter classical convolution, matched to the quantum layer.

    Same patch size, same stride, same number of output channels, same
    untrained-random-weights setup. This is the control: if the quantum layer
    scores better, it must be the quantum feature map doing the work and not
    simply the extra dimensionality of the representation.
    """
    images = np.asarray(images, dtype=float) / 255.0
    if images.ndim == 2:
        images = images[None, ...]
    n, H, W = images.shape
    stride = stride or patch_size
    out_h = (H - patch_size) // stride + 1
    out_w = (W - patch_size) // stride + 1

    rng = np.random.default_rng(seed)
    filters = rng.normal(0, 1, (n_filters, patch_size * patch_size))

    feats = np.empty((n, out_h * out_w, n_filters))
    k = 0
    for oy in range(out_h):
        for ox in range(out_w):
            y, x = oy * stride, ox * stride
            block = images[:, y:y + patch_size, x:x + patch_size].reshape(n, -1)
            feats[:, k, :] = np.tanh(block @ filters.T)   # bounded, like probs
            k += 1
    return feats.reshape(n, -1)


def classical_conv_from_weights(images, weights, patch_size=2, stride=None):
    """Classical convolution with explicit (trainable) filter weights, so the
    classical control can be optimised on exactly the same footing as the
    quantum filter. Training only the quantum side would invert the unfairness
    the untrained comparison was designed to avoid."""
    images = np.asarray(images, dtype=float) / 255.0
    if images.ndim == 2:
        images = images[None, ...]
    n, H, W = images.shape
    stride = stride or patch_size
    out_h = (H - patch_size) // stride + 1
    out_w = (W - patch_size) // stride + 1
    weights = np.asarray(weights, dtype=float).reshape(-1, patch_size * patch_size)

    feats = np.empty((n, out_h * out_w, weights.shape[0]))
    k = 0
    for oy in range(out_h):
        for ox in range(out_w):
            y, x = oy * stride, ox * stride
            block = images[:, y:y + patch_size, x:x + patch_size].reshape(n, -1)
            feats[:, k, :] = np.tanh(block @ weights.T)
            k += 1
    return feats.reshape(n, -1)


def raw_pixels(images):
    images = np.asarray(images, dtype=float) / 255.0
    if images.ndim == 2:
        images = images[None, ...]
    return images.reshape(images.shape[0], -1)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def square_pad_resize(arr, size=8, background=255):
    """Pad a character box to square, THEN resize. Shared by dataset generation
    and inference so training and test crops are produced identically.

    Resizing a box straight to NxN destroys aspect ratio, which for OCR is not a
    cosmetic loss: a colon is two small dots in a narrow tall box, and stretched
    to a square it becomes two thick horizontal bands that closely resemble an
    E. Padding to square first keeps width-to-height information intact, and the
    padding margin itself encodes how narrow the original glyph was.
    """
    from PIL import Image as _Image
    h, w = arr.shape
    side = max(h, w)
    canvas = np.full((side, side), background, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = arr
    return np.array(_Image.fromarray(canvas, mode="L").resize(
        (size, size), _Image.BILINEAR))


def normalize_crops(crops):
    """Per-crop min-max stretch to the full 0-255 range.

    Necessary because the degraded tiers reduce contrast globally: without this
    a faded crop and a blank crop look nearly identical to the encoder, since
    RY(pi*v/255) maps a narrow band of v to a narrow band of angles. This is
    standard OCR preprocessing and is applied identically to every extractor,
    so it does not advantage any one of them.
    """
    c = np.asarray(crops, dtype=float)
    flat = c.reshape(len(c), -1)
    lo = flat.min(axis=1, keepdims=True)
    hi = flat.max(axis=1, keepdims=True)
    span = np.maximum(hi - lo, 1e-6)
    return (((flat - lo) / span) * 255.0).reshape(c.shape)


# --------------------------------------------------------------------------
# Correctness check against the original shot-based implementation
# --------------------------------------------------------------------------

def verify_against_qiskit(n_trials=6, n_qubits=4, shots=200000, tol=0.01):
    """Confirm the fast path reproduces the notebook's Aer-based quanv_filter."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    rng = np.random.default_rng(7)
    filt = random_entangling_circuit(n_qubits, seed=42, depth=2)
    sim = AerSimulator()
    max_err = 0.0

    for _ in range(n_trials):
        patch = rng.integers(0, 256, (2, 2))

        # Reference: build and run the circuit exactly as the notebook does.
        qc = QuantumCircuit(n_qubits)
        for i, val in enumerate(patch.flatten().astype(float)):
            qc.ry(np.pi * (val / 255.0), i)
        qc.compose(filt, inplace=True)
        qc.measure_all()
        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()

        p1 = np.zeros(n_qubits)
        for bits, c in counts.items():
            full = bits.replace(" ", "")[::-1]
            for i in range(n_qubits):
                if full[i] == "1":
                    p1[i] += c
        p1 /= shots

        fast = quanv_patch_probs(patch[None, ...], n_qubits=n_qubits)[0]
        max_err = max(max_err, float(np.abs(p1 - fast).max()))

    return max_err, max_err < tol


if __name__ == "__main__":
    err, ok = verify_against_qiskit()
    print("max deviation from Aer shot-based quanv: {:.4f} -> {}".format(
        err, "MATCH" if ok else "MISMATCH"))

    import time
    imgs = np.random.default_rng(0).integers(0, 256, (2000, 8, 8))
    t0 = time.time()
    f = quanv_features(imgs)
    print("quanv on {} images -> {} features in {:.2f}s".format(
        len(imgs), f.shape[1], time.time() - t0))
