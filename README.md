# DataAgent-SFT

**Data curation, supervised fine-tuning and evidence-grounded deployment of a
small data-analysis language model.** CrossMetric AI is a downstream case study,
not the primary training project.

## Research objective

Can a compact Qwen model be adapted to produce structured data-analysis
responses using a curated subset of analytical trajectories? This repository
documents selection, assistant-token supervision, training and validation loss,
checkpoint inference, and the limitations encountered in deployment.

The model proposes mathematical formulations, analysis workflows and Python
code. Generated code is displayed, not executed. Correct-looking prose does
not guarantee a correct analysis.

## Pipeline

```text
DataMind trajectories -> measurable quality scores -> exact/near deduplication
 -> source-capped selection -> group-disjoint split -> assistant-turn encoding
 -> Qwen text-backbone SFT -> held-out loss -> streaming demo
 -> CrossMetric: deterministic facts + guarded model interpretation
```

### Data curation

The recorded run processed 11,707 trajectories. Quality scoring combines
complexity, code syntax, execution-consistency, structural and text signals.
Execution-consistency signals are parsed from existing traces; selection does
not sandbox and re-execute every program.

| Stage | Count |
|---|---:|
| Input | 11,707 |
| After hash deduplication (207 removed) | 11,500 |
| After TF-IDF character n-gram near-deduplication (199 removed) | 11,301 |
| Selected with ranking/diversity constraints | 2,500 |
| Train / validation | 2,000 / 500 |

The remaining 8,801 candidates were not all invalid; target size and a per-source
cap of six constrain selection. Recorded group, ID and message-hash overlaps:
zero. The split uses candidate search to balance quality/difficulty distributions.
This is **deterministic reward scoring**, not a trained reward model or RL.
The optional external GLM judge was disabled. No embedding/K-means claim is made.

### Fine-tuning

Base: Qwen3.5-0.8B. Text-backbone supervised fine-tuning, with the unused vision
tower frozen by default. This is not LoRA, but also not an update of every
multimodal parameter. Chat templates encode one example per assistant turn.
System/user/context tokens are masked with `-100`; assistant target tokens
contribute to loss. Long contexts are left-windowed to retain the target.

The trainer uses Ray TorchTrainer, gradient accumulation/checkpointing, baseline
validation, optimizer-step scheduling and validation-scored checkpoint retention.

| Recorded setting/result | Value |
|---|---:|
| Learning rate / seed | 2e-5 / 42 |
| Epochs / maximum sequence length | 1 / 2,048 |
| Per-device batch / accumulation | 1 / 8 |
| Optimizer steps | 1,475 |
| Baseline validation loss | 0.731607 |
| Final validation loss | 0.488760 |
| Final training loss | 0.504752 |

Validation loss decreased about 33.2%. This is teacher-forced token loss, not
code runnability, task accuracy or evidence of real business performance.
No measured task-success rate is claimed. These metrics are from the archived
server run, not a new training run on the laptop.

## Reproduce

Recorded server: Python 3.12.3, PyTorch 2.5.1+cu124, Transformers 5.17.0,
Ray 2.58.0, Accelerate 1.15.0; one RTX 4090 D. Use a separate environment.
Install the CUDA-compatible PyTorch build, then:

```bash
pip install -r requirements-training.txt
python training/select_trajectories.py --input /path/to/datamind_12k.json --output-dir data/training/selected --glm-limit 0
python evaluation/test_training.py
python training/train_qwen_ray.py --learning-rates 2e-5 --epochs 1 --max-length 2048 --seed 42
```

For an integration smoke run, add `--smoke`. It is not a replacement for the full
experiment. Raw DataMind trajectories and weights are not included in Git.
Acquire upstream data/model with permission. Supply a local base-model snapshot
using `--model /path/to/base --local-files-only` to avoid revision drift.
Inspect `--help` for resource and output options.

Reproducibility evidence and remaining limitations:
[Reproducibility report](docs/REPRODUCIBILITY.md). Identical floating-point
training results are not guaranteed across hardware/library versions.

## Run the model demo

```bash
pip install -r requirements-demo.txt
python demo/data_agent_app.py --checkpoint-path /path/to/fine-tuned/model --local-files-only --server-name 127.0.0.1 --server-port 6006
```

For remote inference, run on your laptop:

```bash
ssh -N -L 7861:127.0.0.1:6006 -p <port> <user>@<host>
```

Open `http://127.0.0.1:7861`; keep server and tunnel alive. Do not commit SSH keys.

## Downstream application: CrossMetric AI

The bilingual commerce prototype demonstrates three guarded decision workflows
and Monte Carlo scenarios. Python establishes facts; the model explains them.
Unsupported narratives fall back to deterministic text. Such fallback must not
be counted as model reasoning success. Trends have weaker narrative protection.

See [Application README](applications/crossmetric/README.md).

```bash
python applications/crossmetric/evaluation/guardrail_regression.py
```

100,000 connections / 10,000 API RPS / 500 LLM streams are architecture design
targets only, not benchmarked throughput. Historical UCI transaction revenue is
observed; operational cost/channel/inventory fields are simulated.

## Structure

```text
training/                 final selection and SFT scripts
demo/                     streaming model demonstration
evaluation/               recorded metrics and executable checks
docs/                     reproduction notes and model card
applications/crossmetric/ downstream product and demo data
artifacts/                ignored local models, raw evidence and outputs
```

## Limitations and responsible release

Observed failures include invented metrics, contradictory financial reasoning
and repetition. Guardrails are heuristic, not comprehensive correctness proofs.
The project does not execute model-generated code or automate financial actions.
Base-vs-fine-tuned generated-answer evaluation remains future work.

Original code has no selected open-source license yet. Upstream data/model terms
are separate. Raw datasets, model weights, credentials and personal server paths
are excluded from Git. Development/documentation used LLM assistance.
