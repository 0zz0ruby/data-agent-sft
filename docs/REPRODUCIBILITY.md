# Reproducibility audit — 2026-10-03

## Verified locally

- Re-ran the final selection script on the original 11,707-record input.
- Exact removals: 207; near-duplicate removals: 199; selected split: 2,000/500.
- Reproduced train and validation JSONL files match the originals byte-for-byte.
- Train SHA256: `251f24b27a21f3241df27e87b8812c5365acf1ef7659272fcf009535d0170944`.
- Validation SHA256: `b65a4907198c7427ead5c65759ecc4e57943bfcee97eb6af321690004a7916a3`.
- Model archive hash matches the server backup manifest:
  `1d710e23cfa810d86f9c7716f6d24221716851f5518188c85aac692751570597`.
- Three downstream calculation/narrative workflows pass 25 regression assertions.

## Recorded training evidence

The server archive contains checkpoint weights/config/tokenizer, metrics,
experiment summary, Python version, package versions and NVIDIA information.
The published result summary is sanitized; personal server paths are omitted.
Original evidence and model weights remain in ignored `artifacts/`.

Recorded environment: Python 3.12.3; PyTorch 2.5.1+cu124;
Transformers 5.17.0; Ray 2.58.0; Accelerate 1.15.0; RTX 4090 D.
The unused vision tower is frozen by default in this text-only training setup.

## Not verified

The full training run has not been repeated during cleanup: the local validation
machine has CPU-only PyTorch, not the original GPU environment. Identical losses
are therefore not claimed. GPU smoke training, generated-answer benchmarking,
and large-scale load testing remain outstanding. Dependency ranges in the
training requirements should be paired with the recorded versions above.

Selection reproducibility is established; end-to-end training reproducibility
is documented but not independently demonstrated by a second full run.
