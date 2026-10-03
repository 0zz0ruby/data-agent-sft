#!/usr/bin/env python
"""Reproducible DataMind-12K trajectory selection for CrossMetric AI.

The pipeline is deliberately auditable:
1. score trajectory complexity and measurable quality signals;
2. remove exact and task-level near duplicates;
3. select a quality/diversity pool with a per-source cap;
4. make an exact-size, group-disjoint, level-stratified 2,000/500 split;
5. save Qwen-ready JSONL plus metadata and a manifest.

The optional GLM judge augments the deterministic reward.  It never replaces the
measurable scores and all responses are cached for reproducibility.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "training" / "datamind_12k.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "training" / "selected"
DEFAULT_GLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

TAG_RE = {
    "think": re.compile(r"<think>.*?</think>", re.I | re.S),
    "code": re.compile(r"<code>.*?</code>", re.I | re.S),
    "interpreter": re.compile(r"<interpreter>.*?</interpreter>", re.I | re.S),
    "answer": re.compile(r"<answer>.*?</answer>", re.I | re.S),
}
FENCED_CODE_RE = re.compile(r"```(?:python|py|sql)?\s*(.*?)```", re.I | re.S)
CODE_TAG_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.I | re.S)
ERROR_RE = re.compile(r"\b(traceback|exception|syntaxerror|nameerror|failed|error:)\b", re.I)
SUCCESS_RE = re.compile(
    r"\b(code run successfully|executed successfully|output has been saved|"
    r"the output content is|rows? returned|query executed successfully)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--max-per-group", type=int, default=6)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.93)
    parser.add_argument("--dedup-neighbors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--glm-limit",
        type=int,
        default=0,
        help="Judge this many top candidates with GLM (0 disables API calls).",
    )
    parser.add_argument("--glm-model", default="glm-4.7-flash")
    parser.add_argument("--glm-url", default=DEFAULT_GLM_URL)
    parser.add_argument("--glm-timeout", type=float, default=45.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug only: process the first N source records (0 means all).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_source(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("Expected the DataMind snapshot to be a JSON list")
    return data


def messages_of(item: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = item.get("messages") or item.get("trajectory") or []
    clean: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if role and isinstance(content, str):
            clean.append({"role": role, "content": content})
    return clean


def sample_id(item: Dict[str, Any], index: int) -> str:
    return str(item.get("id") or item.get("task_id") or f"row-{index:06d}")


def group_key(item: Dict[str, Any], sid: str) -> str:
    extra = item.get("extra_info") if isinstance(item.get("extra_info"), dict) else {}
    value = (
        item.get("filename")
        or item.get("db_id")
        or extra.get("db_id")
        or re.sub(r"[_-]\d+$", "", sid)
    )
    return str(value).strip().lower()


def level_of(item: Dict[str, Any]) -> str:
    return str(item.get("level") or item.get("difficulty") or "Unknown").strip() or "Unknown"


def task_prompt(messages: Sequence[Dict[str, str]]) -> str:
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"].strip()
        if "<interpreter>" not in content.lower():
            return content
    return ""


def normalize_prompt(text: str) -> str:
    text = text.lower()
    text = re.sub(r"`[^`]+`", " CODE ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " NUM ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def extract_code_blocks(messages: Sequence[Dict[str, str]]) -> List[str]:
    text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    blocks = FENCED_CODE_RE.findall(text)
    if not blocks:
        blocks = CODE_TAG_RE.findall(text)
    cleaned: List[str] = []
    for block in blocks:
        block = re.sub(r"^```(?:python|py|sql)?\s*|```$", "", block.strip(), flags=re.I)
        if block:
            cleaned.append(block.strip())
    return cleaned


def python_syntax_score(blocks: Sequence[str]) -> float:
    if not blocks:
        return 0.0
    valid = 0
    checked = 0
    for block in blocks:
        # SQL statements embedded as execute_sql(...) are valid Python.  A raw SQL
        # block is treated as unchecked rather than incorrectly marked invalid.
        if re.match(r"^\s*(select|with|pragma|insert|update|delete)\b", block, re.I):
            continue
        checked += 1
        try:
            # Dataset code can contain regex strings such as "\d" that are valid
            # today but emit SyntaxWarning on newer Python releases.  The warning is
            # not a syntax failure, so keep the audit log quiet and score it normally.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(block)
            valid += 1
        except SyntaxError:
            pass
    if checked == 0:
        return 0.5
    return valid / checked


def execution_score(messages: Sequence[Dict[str, str]]) -> Tuple[float, int, int]:
    interpreter = [m["content"] for m in messages if "<interpreter>" in m["content"].lower()]
    if not interpreter:
        return 0.0, 0, 0
    successes = sum(bool(SUCCESS_RE.search(text)) for text in interpreter)
    errors = sum(bool(ERROR_RE.search(text)) for text in interpreter)
    neutral = max(0, len(interpreter) - successes - errors)
    score = (successes + 0.45 * neutral) / max(1, len(interpreter))
    # A trajectory that recovers from an error and still reaches an answer contains
    # useful debugging supervision; do not discard it outright.
    has_final_answer = any(TAG_RE["answer"].search(m["content"]) for m in messages)
    if errors and has_final_answer:
        score = min(1.0, score + 0.15)
    return float(score), successes, errors


def bounded_length_quality(length: int, low: int, ideal: int, high: int) -> float:
    if length <= 0:
        return 0.0
    if length < low:
        return length / low
    if length <= high:
        return 1.0
    return max(0.0, 1.0 - (length - high) / max(high, ideal))


def build_record(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    messages = messages_of(item)
    sid = sample_id(item, index)
    prompt = task_prompt(messages)
    normalized = normalize_prompt(prompt)
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    all_text = "\n".join(m["content"] for m in messages)
    blocks = extract_code_blocks(messages)
    exec_score, successes, errors = execution_score(messages)
    tag_presence = {name: float(bool(pattern.search(all_text))) for name, pattern in TAG_RE.items()}
    completeness = float(np.mean(list(tag_presence.values())))
    last_answer = ""
    for message in reversed(assistant_messages):
        match = TAG_RE["answer"].search(message["content"])
        if match:
            last_answer = match.group(0)
            break

    complexity_raw = (
        1.2 * math.log1p(len(assistant_messages))
        + 0.9 * math.log1p(len(blocks))
        + 0.8 * math.log1p(successes + errors)
        + 0.45 * math.log1p(len(all_text) / 1000.0)
    )
    return {
        "index": index,
        "id": sid,
        "group": group_key(item, sid),
        "level": level_of(item),
        "messages": messages,
        "original": item,
        "task_prompt": prompt,
        "normalized_prompt": normalized,
        "prompt_hash": stable_hash(normalized),
        "message_hash": stable_hash(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
        "metrics": {
            "assistant_turns": len(assistant_messages),
            "code_blocks": len(blocks),
            "interpreter_turns": successes + errors,
            "successful_interpreter_turns": successes,
            "error_interpreter_turns": errors,
            "characters": len(all_text),
            "complexity_raw": complexity_raw,
            "syntax_score": python_syntax_score(blocks),
            "execution_consistency_score": exec_score,
            "completeness_score": completeness,
            "prompt_quality_score": bounded_length_quality(len(prompt), 12, 300, 2400),
            "final_answer_quality_score": bounded_length_quality(len(last_answer), 20, 350, 3000),
            **{f"has_{name}": value for name, value in tag_presence.items()},
        },
        "glm": {"score": None, "reason": "", "api_success": False},
    }


def percentile_ranks(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="stable")
    ranks = np.empty(len(values_array), dtype=np.float64)
    ranks[order] = np.arange(len(values_array), dtype=np.float64)
    if len(values_array) <= 1:
        return np.ones(len(values_array), dtype=np.float64)
    return ranks / (len(values_array) - 1)


def score_records(records: List[Dict[str, Any]]) -> None:
    complexity = percentile_ranks([r["metrics"]["complexity_raw"] for r in records])
    for record, complexity_score in zip(records, complexity):
        metrics = record["metrics"]
        metrics["complexity_score"] = float(complexity_score)
        metrics["deterministic_reward"] = float(
            0.20 * complexity_score
            + 0.20 * metrics["syntax_score"]
            + 0.20 * metrics["execution_consistency_score"]
            + 0.20 * metrics["completeness_score"]
            + 0.10 * metrics["prompt_quality_score"]
            + 0.10 * metrics["final_answer_quality_score"]
        )
        metrics["final_reward"] = metrics["deterministic_reward"]


def request_glm_score(
    record: Dict[str, Any], api_key: str, model: str, url: str, timeout: float
) -> Dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests before enabling --glm-limit") from exc

    evidence = {
        "task": record["task_prompt"][:1800],
        "last_assistant": next(
            (m["content"][-2200:] for m in reversed(record["messages"]) if m["role"] == "assistant"),
            "",
        ),
        "automatic_metrics": record["metrics"],
    }
    prompt = (
        "Evaluate this data-agent trajectory for correctness, reasoning quality, and "
        "instructional usefulness. Return JSON only with keys score (integer 1-5) "
        "and reason (one short sentence). Do not reward verbosity.\n\n"
        + json.dumps(evidence, ensure_ascii=False)
    )
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 160,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    payload = json.loads(content)
    score = int(payload["score"])
    if score < 1 or score > 5:
        raise ValueError(f"GLM score outside 1-5: {score}")
    return {"score": score / 5.0, "reason": str(payload.get("reason", "")), "api_success": True}


def apply_glm_judge(records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if args.glm_limit <= 0:
        return
    api_key = os.environ.get("ZHIPU_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("--glm-limit requires the ZHIPU_API_KEY environment variable")

    cache_path = args.output_dir / "glm_judge_cache.jsonl"
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                cache[row["message_hash"]] = row["glm"]

    candidates = sorted(records, key=lambda r: r["metrics"]["deterministic_reward"], reverse=True)
    candidates = candidates[: min(args.glm_limit, len(candidates))]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as stream:
        for position, record in enumerate(candidates, 1):
            cached = cache.get(record["message_hash"])
            if cached:
                record["glm"] = cached
            else:
                try:
                    record["glm"] = request_glm_score(
                        record, api_key, args.glm_model, args.glm_url, args.glm_timeout
                    )
                except Exception as exc:  # API failure is evidence, not silent fallback.
                    record["glm"] = {"score": None, "reason": repr(exc), "api_success": False}
                stream.write(
                    json.dumps(
                        {"message_hash": record["message_hash"], "glm": record["glm"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stream.flush()
                time.sleep(0.15)
            if record["glm"]["api_success"]:
                automatic = record["metrics"]["deterministic_reward"]
                record["metrics"]["final_reward"] = 0.75 * automatic + 0.25 * record["glm"]["score"]
            print(f"[GLM] {position}/{len(candidates)}", end="\r", flush=True)
    print()


def exact_deduplicate(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_messages = set()
    seen_prompts = set()
    for record in sorted(records, key=lambda r: r["metrics"]["final_reward"], reverse=True):
        if (
            record["id"] in seen_ids
            or record["message_hash"] in seen_messages
            or (record["normalized_prompt"] and record["prompt_hash"] in seen_prompts)
        ):
            continue
        seen_ids.add(record["id"])
        seen_messages.add(record["message_hash"])
        if record["normalized_prompt"]:
            seen_prompts.add(record["prompt_hash"])
        kept.append(record)
    return kept, len(records) - len(kept)


def near_deduplicate(
    records: List[Dict[str, Any]], threshold: float, neighbors: int
) -> Tuple[List[Dict[str, Any]], int]:
    """Remove near-duplicate tasks using TF-IDF cosine neighbors.

    Only the task prompt is embedded.  This is intentionally called task-level
    near-deduplication, not full-trajectory semantic deduplication.
    """

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    texts = [record["normalized_prompt"] or record["task_prompt"] for record in records]
    if len(texts) < 2:
        return records, 0
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=45000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    n_neighbors = min(max(2, neighbors), len(records))
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1)
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix, return_distance=True)

    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for row, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, other in zip(row_distances[1:], row_indices[1:]):
            if 1.0 - float(distance) >= threshold:
                union(row, int(other))

    clusters: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(records)):
        clusters[find(index)].append(index)
    kept_indices = []
    for members in clusters.values():
        best = max(members, key=lambda i: records[i]["metrics"]["final_reward"])
        kept_indices.append(best)
    kept = [records[i] for i in sorted(kept_indices)]
    return kept, len(records) - len(kept)


def choose_quality_diversity_pool(
    records: List[Dict[str, Any]], size: int, max_per_group: int
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    for record in sorted(records, key=lambda r: r["metrics"]["final_reward"], reverse=True):
        if group_counts[record["group"]] >= max_per_group:
            continue
        selected.append(record)
        group_counts[record["group"]] += 1
        if len(selected) == size:
            break
    if len(selected) != size:
        raise RuntimeError(
            f"Only {len(selected)} samples survived selection; need {size}. "
            "Increase --max-per-group or lower the near-duplicate threshold."
        )
    return selected


def distribution(records: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def split_objective(
    validation: Sequence[Dict[str, Any]], all_records: Sequence[Dict[str, Any]]
) -> float:
    target_ratio = len(validation) / len(all_records)
    total_levels = Counter(record["level"] for record in all_records)
    val_levels = Counter(record["level"] for record in validation)
    level_error = sum(
        abs(val_levels[level] - count * target_ratio) / max(1.0, count * target_ratio)
        for level, count in total_levels.items()
    )
    rewards = np.asarray([r["metrics"]["final_reward"] for r in all_records])
    val_rewards = np.asarray([r["metrics"]["final_reward"] for r in validation])
    reward_error = abs(float(val_rewards.mean() - rewards.mean())) / max(float(rewards.std()), 1e-8)
    return level_error + reward_error


def group_stratified_exact_split(
    records: List[Dict[str, Any]], val_size: int, seed: int, trials: int = 8000
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    group_names = list(groups)
    rng = random.Random(seed)
    best_names: Optional[set[str]] = None
    best_objective = float("inf")

    # Randomized group order plus exact-size skipping gives many group-disjoint
    # candidates.  Keep the one closest to the full pool's level/reward distribution.
    for _ in range(trials):
        rng.shuffle(group_names)
        chosen: List[str] = []
        count = 0
        for name in group_names:
            group_size = len(groups[name])
            if count + group_size <= val_size:
                chosen.append(name)
                count += group_size
            if count == val_size:
                break
        if count != val_size:
            continue
        candidate = [record for name in chosen for record in groups[name]]
        objective = split_objective(candidate, records)
        if objective < best_objective:
            best_objective = objective
            best_names = set(chosen)

    if best_names is None:
        raise RuntimeError(
            "Could not construct an exact group-disjoint validation set. "
            "Try a different seed or --max-per-group."
        )
    validation = [record for record in records if record["group"] in best_names]
    train = [record for record in records if record["group"] not in best_names]
    return train, validation


def qwen_row(record: Dict[str, Any]) -> Dict[str, Any]:
    return {"messages": record["messages"]}


def metadata_row(record: Dict[str, Any], split: str) -> Dict[str, Any]:
    row = dict(record["original"])
    row["messages"] = record["messages"]
    row["_selection"] = {
        "split": split,
        "id": record["id"],
        "group": record["group"],
        "level": record["level"],
        "prompt_hash": record["prompt_hash"],
        "message_hash": record["message_hash"],
        "metrics": record["metrics"],
        "glm": record["glm"],
    }
    return row


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_audit_csv(records: Sequence[Dict[str, Any]], split: str, path: Path) -> None:
    fieldnames = [
        "split",
        "id",
        "group",
        "level",
        "final_reward",
        "deterministic_reward",
        "complexity_score",
        "syntax_score",
        "execution_consistency_score",
        "completeness_score",
        "assistant_turns",
        "code_blocks",
        "interpreter_turns",
        "characters",
        "glm_score",
        "glm_api_success",
        "glm_reason",
    ]
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for record in records:
            metrics = record["metrics"]
            writer.writerow(
                {
                    "split": split,
                    "id": record["id"],
                    "group": record["group"],
                    "level": record["level"],
                    "final_reward": f"{metrics['final_reward']:.8f}",
                    "deterministic_reward": f"{metrics['deterministic_reward']:.8f}",
                    "complexity_score": f"{metrics['complexity_score']:.8f}",
                    "syntax_score": f"{metrics['syntax_score']:.8f}",
                    "execution_consistency_score": f"{metrics['execution_consistency_score']:.8f}",
                    "completeness_score": f"{metrics['completeness_score']:.8f}",
                    "assistant_turns": metrics["assistant_turns"],
                    "code_blocks": metrics["code_blocks"],
                    "interpreter_turns": metrics["interpreter_turns"],
                    "characters": metrics["characters"],
                    "glm_score": record["glm"]["score"],
                    "glm_api_success": record["glm"]["api_success"],
                    "glm_reason": record["glm"]["reason"],
                }
            )


def validate_split(
    train_records: Sequence[Dict[str, Any]], val_records: Sequence[Dict[str, Any]], args: argparse.Namespace
) -> None:
    if len(train_records) != args.train_size or len(val_records) != args.val_size:
        raise AssertionError("Split sizes are incorrect")
    train_groups = {r["group"] for r in train_records}
    val_groups = {r["group"] for r in val_records}
    if train_groups & val_groups:
        raise AssertionError("Group leakage detected")
    train_ids = {r["id"] for r in train_records}
    val_ids = {r["id"] for r in val_records}
    if train_ids & val_ids:
        raise AssertionError("ID leakage detected")
    train_hashes = {r["message_hash"] for r in train_records}
    val_hashes = {r["message_hash"] for r in val_records}
    if train_hashes & val_hashes:
        raise AssertionError("Message leakage detected")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_hash = sha256_file(args.input)
    source = load_source(args.input)
    source_count = len(source)
    if args.limit:
        source = source[: args.limit]
    print(f"[load] {len(source):,}/{source_count:,} records; SHA256={input_hash}")

    records = [build_record(item, index) for index, item in enumerate(source)]
    score_records(records)
    apply_glm_judge(records, args)
    records, exact_removed = exact_deduplicate(records)
    print(f"[exact dedup] removed {exact_removed:,}; retained {len(records):,}")
    records, near_removed = near_deduplicate(
        records, threshold=args.near_duplicate_threshold, neighbors=args.dedup_neighbors
    )
    print(f"[task near-dedup] removed {near_removed:,}; retained {len(records):,}")

    selected_size = args.train_size + args.val_size
    selected = choose_quality_diversity_pool(records, selected_size, args.max_per_group)
    train_records, val_records = group_stratified_exact_split(selected, args.val_size, args.seed)
    validate_split(train_records, val_records, args)

    outputs = {
        "train_qwen": args.output_dir / "train_qwen_2k.jsonl",
        "val_qwen": args.output_dir / "val_qwen_500.jsonl",
        "train_metadata": args.output_dir / "train_metadata_2k.jsonl",
        "val_metadata": args.output_dir / "val_metadata_500.jsonl",
        "audit": args.output_dir / "selection_audit.csv",
        "manifest": args.output_dir / "selection_manifest.json",
    }
    write_jsonl((qwen_row(r) for r in train_records), outputs["train_qwen"])
    write_jsonl((qwen_row(r) for r in val_records), outputs["val_qwen"])
    write_jsonl((metadata_row(r, "train") for r in train_records), outputs["train_metadata"])
    write_jsonl((metadata_row(r, "validation") for r in val_records), outputs["val_metadata"])
    if outputs["audit"].exists():
        outputs["audit"].unlink()
    write_audit_csv(train_records, "train", outputs["audit"])
    write_audit_csv(val_records, "validation", outputs["audit"])

    manifest = {
        "pipeline_version": "2.0",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "input": {
            "path": str(args.input.resolve()),
            "sha256": input_hash,
            "source_count": source_count,
            "processed_count": len(source),
        },
        "selection": {
            "seed": args.seed,
            "train_size": args.train_size,
            "validation_size": args.val_size,
            "max_per_group": args.max_per_group,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "dedup_neighbors": args.dedup_neighbors,
            "exact_duplicates_removed": exact_removed,
            "near_duplicates_removed": near_removed,
            "glm_model": args.glm_model if args.glm_limit else None,
            "glm_limit": args.glm_limit,
        },
        "checks": {
            "group_overlap": sorted(
                {r["group"] for r in train_records} & {r["group"] for r in val_records}
            ),
            "id_overlap_count": len(
                {r["id"] for r in train_records} & {r["id"] for r in val_records}
            ),
            "message_hash_overlap_count": len(
                {r["message_hash"] for r in train_records}
                & {r["message_hash"] for r in val_records}
            ),
        },
        "distributions": {
            "train_level": distribution(train_records, "level"),
            "validation_level": distribution(val_records, "level"),
            "train_groups": len({r["group"] for r in train_records}),
            "validation_groups": len({r["group"] for r in val_records}),
            "train_reward_mean": float(
                np.mean([r["metrics"]["final_reward"] for r in train_records])
            ),
            "validation_reward_mean": float(
                np.mean([r["metrics"]["final_reward"] for r in val_records])
            ),
        },
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    with outputs["manifest"].open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print(f"[done] train={len(train_records):,} validation={len(val_records):,}")
    print(f"[done] group overlap={manifest['checks']['group_overlap']}")
    print(f"[done] output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
