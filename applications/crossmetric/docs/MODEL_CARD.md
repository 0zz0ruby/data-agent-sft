# CrossMetric Data Agent

Base: Qwen3.5-0.8B. Adaptation: full-parameter supervised fine-tuning on selected
DataMind trajectories. Intended use: human-reviewed analytical interpretations.
Weights are not distributed here.

11,707 trajectories were scored using complexity, code syntax, execution
consistency, structure and text quality. Hash and TF-IDF deduplication preceded
quality/diversity selection and a group-disjoint 2,000/500 split. The optional
GLM judge was disabled. The deterministic reward is not a learned reward model.

Training uses chat templates, assistant-only loss masking, gradient accumulation,
checkpointing and Ray. Recorded settings: LR 2e-5, sequence length 2,048, one
epoch, 1,475 optimizer steps. Baseline validation loss: 0.731607; final validation
loss: 0.488760; final training loss: 0.504752. These recorded measurements were
not rerun during cleanup. No task-success percentage is claimed.

Python computes facts for three protected workflows. A heuristic validator
rejects unsupported numbers/entities and selected contradictions; rejection
uses a safe template. This is not a comprehensive semantic proof. Trend
Diagnosis has weaker protection.

The model has produced invented metrics, incorrect financial interpretations
and repetitive text. It must not autonomously perform financial actions.
Operational demo fields are simulated. Review upstream model/DataMind terms
before redistributing weights or training trajectories.
