#!/usr/bin/env python
"""Ray Train fine-tuning for Qwen3.5-0.8B on selected DataMind trajectories.

Key properties:
- real Ray ``TorchTrainer`` execution on CPU or one GPU;
- one SFT example per assistant turn, with assistant-only loss;
- validation loss before training and after every epoch;
- correct optimizer-step scheduling with gradient accumulation;
- Ray-managed checkpoints scored by validation loss;
- optional learning-rate comparison using the same validation split.

Run ``python training/train_qwen_ray.py --help`` for the smoke/search workflow.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SELECTED = REPO_ROOT / "data" / "training" / "selected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_SELECTED / "train_qwen_2k.jsonl",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=DEFAULT_SELECTED / "val_qwen_500.jsonl",
    )
    parser.add_argument("--storage-path", type=Path, default=REPO_ROOT / "artifacts" / "ray_results")
    parser.add_argument("--experiment-name", default="qwen35-datamind-sft")
    parser.add_argument(
        "--learning-rates",
        default="1e-5,2e-5,5e-5",
        help="Comma-separated validation search grid; use one value for a final run.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-dataloader-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument(
        "--freeze-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the unused vision tower for this text-only dataset.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU/GPU integration test: 20 train trajectories, 10 validation, one LR/epoch.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str, limit: int = 0) -> List[List[Dict[str, str]]]:
    trajectories: List[List[Dict[str, str]]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            messages = row.get("messages", [])
            cleaned = [
                {"role": str(m["role"]), "content": str(m["content"])}
                for m in messages
                if isinstance(m, dict) and "role" in m and "content" in m
            ]
            if cleaned:
                trajectories.append(cleaned)
            if limit and len(trajectories) >= limit:
                break
    return trajectories


def template_ids(tokenizer: Any, messages: Sequence[Dict[str, str]], generation: bool) -> List[int]:
    ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=generation,
    )
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    size = min(len(left), len(right))
    for index in range(size):
        if left[index] != right[index]:
            return index
    return size


def _flat_int_list(value: Any) -> List[int]:
    """Normalize a tokenizer tensor/list field to one flat Python list."""

    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def assistant_mask_ids(
    tokenizer: Any, messages: Sequence[Dict[str, str]]
) -> Optional[Tuple[List[int], List[int]]]:
    """Return template IDs and the assistant mask when the template supports it."""

    try:
        encoded = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except (TypeError, ValueError):
        return None

    input_ids = _flat_int_list(encoded["input_ids"])
    for key in ("assistant_masks", "assistant_mask"):
        if key in encoded:
            mask = _flat_int_list(encoded[key])
            if len(mask) == len(input_ids):
                return input_ids, mask
    return None


def last_mask_span(mask: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Find the final contiguous assistant-content span as [start, end)."""

    end = len(mask)
    while end > 0 and not mask[end - 1]:
        end -= 1
    if end == 0:
        return None
    start = end - 1
    while start > 0 and mask[start - 1]:
        start -= 1
    return start, end


def last_subsequence_span(haystack: Sequence[int], needle: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Find the final exact occurrence of needle in haystack."""

    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start, start + len(needle)
    return None


def encode_assistant_turns(
    trajectories: Sequence[Sequence[Dict[str, str]]],
    tokenizer: Any,
    max_length: int,
) -> Tuple[List[Dict[str, List[int]]], Dict[str, int]]:
    """Expand each trajectory into history -> next assistant response examples."""

    examples: List[Dict[str, List[int]]] = []
    stats = {
        "trajectories": len(trajectories),
        "assistant_turns": 0,
        "examples": 0,
        "left_truncated": 0,
        "dropped_target_too_long": 0,
        "template_alignment_fallbacks": 0,
        "content_alignment_fallbacks": 0,
        "dropped_alignment_failures": 0,
    }
    for messages in trajectories:
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            stats["assistant_turns"] += 1
            prefix = messages[:index]
            complete = messages[: index + 1]
            masked = assistant_mask_ids(tokenizer, complete)
            target_span: Optional[Tuple[int, int]] = None
            if masked is not None:
                full_ids, mask = masked
                target_span = last_mask_span(mask)
            else:
                full_ids = template_ids(tokenizer, complete, generation=False)

            # Some third-party chat templates do not expose generation masks.
            # In that case, locate the final assistant content exactly inside the
            # fully templated sequence.  This is safer than treating the entire
            # trajectory as the target when generation prompts do not align.
            if target_span is None:
                content_ids = _flat_int_list(
                    tokenizer(str(message["content"]), add_special_tokens=False)["input_ids"]
                )
                target_span = last_subsequence_span(full_ids, content_ids)
                stats["content_alignment_fallbacks"] += 1

            if target_span is None:
                prompt_ids = template_ids(tokenizer, prefix, generation=True)
                label_start = common_prefix_length(prompt_ids, full_ids)
                if label_start < max(1, min(len(prompt_ids), len(full_ids)) // 2):
                    stats["dropped_alignment_failures"] += 1
                    continue
                target_span = (label_start, len(full_ids))
                stats["template_alignment_fallbacks"] += 1

            label_start, label_end = target_span
            target_length = label_end - label_start
            if target_length <= 0:
                continue
            # Truncating away the target itself corrupts supervision.  Drop an
            # assistant response that cannot fit instead of silently training on a tail.
            if target_length >= max_length:
                stats["dropped_target_too_long"] += 1
                continue

            if len(full_ids) > max_length:
                crop = len(full_ids) - max_length
                full_ids = full_ids[crop:]
                label_start -= crop
                label_end -= crop
                stats["left_truncated"] += 1
            label_start = max(0, label_start)
            label_end = min(len(full_ids), label_end)
            if label_end <= label_start:
                stats["dropped_alignment_failures"] += 1
                continue
            labels = [-100] * len(full_ids)
            labels[label_start:label_end] = full_ids[label_start:label_end]
            if not any(label != -100 for label in labels):
                continue
            examples.append({"input_ids": full_ids, "labels": labels})

    stats["examples"] = len(examples)
    return examples, stats


class AssistantTurnDataset(Dataset):
    def __init__(self, examples: Sequence[Dict[str, List[int]]]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.examples[index]


def load_qwen_model(model_source: str, dtype: torch.dtype, local_files_only: bool) -> Any:
    """Load the multimodal Qwen3.5 class while keeping a clear version error."""

    try:
        from transformers import AutoModelForMultimodalLM

        model_class = AutoModelForMultimodalLM
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText

            model_class = AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError(
                "This Qwen3.5 checkpoint needs a recent Transformers release with "
                "AutoModelForMultimodalLM (or AutoModelForImageTextToText)."
            ) from exc
    return model_class.from_pretrained(
        model_source,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )


def freeze_vision_parameters(model: torch.nn.Module) -> int:
    frozen = 0
    vision_markers = ("visual", "vision_tower", "vision_model", "model.visual")
    for name, parameter in model.named_parameters():
        if any(marker in name.lower() for marker in vision_markers):
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def reduce_pair(value_sum: float, weight_sum: float, device: torch.device) -> Tuple[float, float]:
    tensor = torch.tensor([value_sum, weight_sum], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0].item()), float(tensor[1].item())


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        count = int((batch["labels"] != -100).sum().item())
        weighted_loss += float(outputs.loss.item()) * count
        supervised_tokens += count
    weighted_loss, supervised_tokens_float = reduce_pair(weighted_loss, supervised_tokens, device)
    return weighted_loss / max(1.0, supervised_tokens_float)


def train_loop_per_worker(config: Dict[str, Any]) -> None:
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import prepare_data_loader, prepare_model
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup

    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    seed_everything(int(config["seed"]) + rank)
    use_gpu = bool(config["use_gpu"] and torch.cuda.is_available())
    device = torch.device("cuda", torch.cuda.current_device()) if use_gpu else torch.device("cpu")
    dtype = torch.bfloat16 if use_gpu and torch.cuda.is_bf16_supported() else (
        torch.float16 if use_gpu else torch.float32
    )

    resume_checkpoint = train.get_checkpoint()
    resume_context = (
        resume_checkpoint.as_directory() if resume_checkpoint is not None else contextlib.nullcontext(None)
    )
    with resume_context as resume_dir:
        model_source = config["model"]
        state: Optional[Dict[str, Any]] = None
        if resume_dir:
            resume_model = Path(resume_dir) / "model"
            resume_state = Path(resume_dir) / "trainer_state.pt"
            if resume_model.exists():
                model_source = str(resume_model)
            if resume_state.exists():
                state = torch.load(resume_state, map_location="cpu", weights_only=False)

        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=bool(config["local_files_only"] or resume_dir),
            padding_side="right",
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_qwen_model(
            model_source,
            dtype=dtype,
            local_files_only=bool(config["local_files_only"] or resume_dir),
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        frozen_vision = freeze_vision_parameters(model) if config["freeze_vision"] else 0

        train_trajectories = read_jsonl(config["train_file"], config["train_limit"])
        val_trajectories = read_jsonl(config["val_file"], config["val_limit"])
        train_examples, train_stats = encode_assistant_turns(
            train_trajectories, tokenizer, config["max_length"]
        )
        val_examples, val_stats = encode_assistant_turns(
            val_trajectories, tokenizer, config["max_length"]
        )
        if not train_examples or not val_examples:
            raise RuntimeError(
                "Tokenization produced an empty training or validation dataset: "
                f"train_stats={train_stats}, val_stats={val_stats}"
            )

        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )
        generator = torch.Generator().manual_seed(int(config["seed"]))
        train_loader = DataLoader(
            AssistantTurnDataset(train_examples),
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collator,
            num_workers=config["num_dataloader_workers"],
            pin_memory=use_gpu,
            generator=generator,
        )
        val_loader = DataLoader(
            AssistantTurnDataset(val_examples),
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collator,
            num_workers=config["num_dataloader_workers"],
            pin_memory=use_gpu,
        )
        train_loader = prepare_data_loader(train_loader)
        val_loader = prepare_data_loader(val_loader)
        model = prepare_model(model)

        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        grad_accum = int(config["gradient_accumulation_steps"])
        optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
        total_optimizer_steps = max(1, optimizer_steps_per_epoch * int(config["epochs"]))
        warmup_steps = int(total_optimizer_steps * float(config["warmup_ratio"]))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
        start_epoch = 0
        global_step = 0
        if state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"])
            global_step = int(state["global_step"])

        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(
                json.dumps(
                    {
                        "rank/world_size": f"{rank}/{world_size}",
                        "device": str(device),
                        "dtype": str(dtype),
                        "parameters_total": total,
                        "parameters_trainable": trainable,
                        "parameters_frozen_vision": frozen_vision,
                        "train_tokenization": train_stats,
                        "validation_tokenization": val_stats,
                        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                        "total_optimizer_steps": total_optimizer_steps,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        baseline_val_loss = evaluate(model, val_loader, device)
        train.report(
            metrics={
                "stage": "baseline",
                "epoch": start_epoch,
                "train_loss": float("nan"),
                "val_loss": baseline_val_loss,
                "baseline_val_loss": baseline_val_loss,
                "learning_rate": config["learning_rate"],
                "global_step": global_step,
            }
        )

        for epoch in range(start_epoch, int(config["epochs"])):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            supervised_tokens = 0
            for batch_index, batch in enumerate(train_loader, 1):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                raw_loss = outputs.loss
                (raw_loss / grad_accum).backward()
                count = int((batch["labels"] != -100).sum().item())
                loss_sum += float(raw_loss.detach().item()) * count
                supervised_tokens += count

                should_step = batch_index % grad_accum == 0 or batch_index == len(train_loader)
                if should_step:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, config["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            loss_sum, supervised_tokens_float = reduce_pair(loss_sum, supervised_tokens, device)
            train_loss = loss_sum / max(1.0, supervised_tokens_float)
            val_loss = evaluate(model, val_loader, device)
            metrics = {
                "stage": "train",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "baseline_val_loss": baseline_val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "configured_learning_rate": config["learning_rate"],
                "global_step": global_step,
            }

            with tempfile.TemporaryDirectory(prefix="ray-qwen-checkpoint-") as checkpoint_dir:
                checkpoint = None
                if rank == 0:
                    checkpoint_root = Path(checkpoint_dir)
                    model_dir = checkpoint_root / "model"
                    model_to_save = unwrap_model(model)
                    model_to_save.save_pretrained(model_dir, safe_serialization=True)
                    tokenizer.save_pretrained(model_dir)
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "metrics": metrics,
                            "config": config,
                        },
                        checkpoint_root / "trainer_state.pt",
                    )
                    (checkpoint_root / "metrics.json").write_text(
                        json.dumps(metrics, indent=2), encoding="utf-8"
                    )
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(metrics=metrics, checkpoint=checkpoint)


def result_summary(result: Any, learning_rate: float) -> Dict[str, Any]:
    checkpoints = []
    for entry in getattr(result, "best_checkpoints", []) or []:
        if isinstance(entry, tuple):
            checkpoint, metrics = entry
        else:
            checkpoint = getattr(entry, "checkpoint", None)
            metrics = getattr(entry, "metrics", {})
        checkpoints.append(
            {
                "checkpoint": str(checkpoint),
                "val_loss": metrics.get("val_loss"),
                "epoch": metrics.get("epoch"),
            }
        )
    scored = [row for row in checkpoints if isinstance(row.get("val_loss"), (int, float))]
    best = min(scored, key=lambda row: row["val_loss"]) if scored else None
    return {
        "learning_rate": learning_rate,
        "result_path": str(result.path),
        "last_metrics": dict(result.metrics),
        "retained_checkpoints": checkpoints,
        "best_checkpoint": best,
    }


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.train_limit = 20
        args.val_limit = 10
        args.epochs = 1
        args.max_length = min(args.max_length, 1024)
        args.learning_rates = args.learning_rates.split(",")[0]
    learning_rates = [float(value.strip()) for value in args.learning_rates.split(",") if value.strip()]
    if not learning_rates:
        raise ValueError("--learning-rates is empty")
    if not args.train_file.exists() or not args.val_file.exists():
        raise FileNotFoundError(
            "Selected training files were not found. Run training/select_trajectories.py "
            "or pass --train-file and --val-file explicitly."
        )

    use_gpu = torch.cuda.is_available() and not args.cpu
    if use_gpu and args.num_workers != 1:
        raise ValueError("This reproducible single-GPU workflow requires --num-workers 1")
    args.storage_path.mkdir(parents=True, exist_ok=True)

    import ray
    from ray import train
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    ray.init(ignore_reinit_error=True)
    summaries = []
    try:
        for learning_rate in learning_rates:
            config = {
                "model": args.model,
                "train_file": str(args.train_file.resolve()),
                "val_file": str(args.val_file.resolve()),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_length": args.max_length,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "num_dataloader_workers": args.num_dataloader_workers,
                "seed": args.seed,
                "train_limit": args.train_limit,
                "val_limit": args.val_limit,
                "local_files_only": args.local_files_only,
                "freeze_vision": args.freeze_vision,
                "gradient_checkpointing": args.gradient_checkpointing,
                "use_gpu": use_gpu,
                "learning_rate": learning_rate,
            }
            suffix = f"lr-{learning_rate:.0e}".replace("+", "")
            run_config = RunConfig(
                name=f"{args.experiment_name}-{suffix}",
                storage_path=str(args.storage_path.resolve()),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=2,
                    checkpoint_score_attribute="val_loss",
                    checkpoint_score_order="min",
                ),
            )
            trainer = TorchTrainer(
                train_loop_per_worker=train_loop_per_worker,
                train_loop_config=config,
                scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=use_gpu),
                run_config=run_config,
            )
            result = trainer.fit()
            summaries.append(result_summary(result, learning_rate))
    finally:
        ray.shutdown()

    valid = [row for row in summaries if row["best_checkpoint"]]
    chosen = min(valid, key=lambda row: row["best_checkpoint"]["val_loss"]) if valid else None
    output = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "gpu" if use_gpu else "cpu",
        "model": args.model,
        "train_file": str(args.train_file.resolve()),
        "val_file": str(args.val_file.resolve()),
        "runs": summaries,
        "selected_by_validation_loss": chosen,
    }
    summary_path = args.storage_path / "hyperparameter_summary.json"
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Summary written to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
