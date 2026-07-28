# Track B — Search (Grover-based String Matching)

**Owner:** Anh
**Supporting:** Pushkar (toy reference build + review)

## Scope

Build a Grover oracle that flips the phase of position `i` in a text if and only if the pattern `P` matches starting at position `i`.

## Suggested approach

1. Start toy: 8-character text, 2-character pattern (3 position qubits). Build a comparator circuit checking character-by-character equality, combine with a multi-controlled phase flip.
2. Verify the phase flip fires **only** at the correct position, using a statevector simulator — check this before adding the diffusion operator.
3. Add the diffusion operator, confirm amplitude amplification toward the correct position over repeated iterations.
4. Scale pattern/text length gradually. Track where gate count/circuit depth becomes the practical bottleneck — this is a real, citable finding for the final report even if scaling stalls.

## Deliverables

- Working toy-case oracle + diffusion notebook, verified via statevector simulator
- Notes on gate-count scaling behavior as pattern length grows
- Functional Grover-based string search simulation (per original project deliverable)

## Reference

Pushkar will build the same toy case independently — compare notes rather than working in isolation, since this is a genuinely tricky piece to get right the first time.
