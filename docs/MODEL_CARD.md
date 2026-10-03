# DataAgent-SFT model card

Base: Qwen3.5-0.8B. Text-backbone supervised fine-tuning without LoRA; unused
vision tower frozen. Curated DataMind trajectories: 2,000 train / 500 validation.
No learned reward model or reinforcement learning was used.

Recorded LR 2e-5, seed 42, one epoch, sequence length 2,048, accumulation eight.
Held-out token loss: 0.731607 before training, 0.488760 after training.
No answer-accuracy or code-execution success rate has been established.

Intended use: research and human-reviewed data-analysis assistance. Generated
code is not automatically executed. Known risks: fabricated numbers, invalid
reasoning, repetition and distribution shift. CrossMetric guards some narratives
using deterministic findings and fallback; this does not certify model quality.

Weights are separately prepared from the verified server backup. Their Hub
publication requires authenticated access and review of upstream license terms.
Training data is not redistributed here. No open-source code license is selected.
