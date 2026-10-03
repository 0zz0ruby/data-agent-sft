"""Local Data Agent web demo powered by a Qwen3.5-0.8B checkpoint.

Users may upload a CSV, inspect its schema, choose an analysis task, and ask
the model for a mathematical abstraction, an analysis pipeline, Python code,
and conclusions. Model-generated code is displayed but is never executed.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from threading import Thread
from typing import Any, Iterator

# Keep the multi-gigabyte model cache on the project drive rather than the
# relatively small Windows system drive. Respect an explicit user override.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, TextIteratorStreamer


DEFAULT_CHECKPOINT = "Qwen/Qwen3.5-0.8B"
MAX_CONTEXT_CHARS = 14_000

SYSTEM_PROMPT = """You are a Data Agent for users without expert programming or
data-science knowledge. Respond in the same language as the user. For each task:

1. Clarify the analytical objective and assumptions.
2. Abstract the problem mathematically: variables, target, constraints, and metrics.
3. Propose an end-to-end data-analysis pipeline.
4. Provide executable Python using pandas/numpy/scikit-learn/matplotlib when useful.
5. Explain how to validate the result and identify limitations or data risks.
6. End with a concise, plain-language conclusion.

Use Markdown headings. Never claim that generated code has been executed. Never
invent values that are absent from the supplied dataset summary. If information is
missing, state what is missing and provide a defensible next step.
"""

TASK_GUIDANCE = {
    "自动选择": "Choose the most appropriate data-analysis workflow for the request.",
    "探索性数据分析": "Focus on data quality, descriptive statistics, distributions, relationships, and anomalies.",
    "预测建模": "Define target/features, preprocessing, train-validation design, baseline, model, and evaluation metrics.",
    "数学抽象": "Translate the real-world question into variables, equations, objectives, assumptions, and constraints.",
    "可视化": "Recommend suitable charts and provide clear matplotlib/seaborn code with labels and interpretation.",
    "假设检验": "State hypotheses, assumptions, test statistic, significance level, effect size, and interpretation.",
}

EXAMPLES = [
    ["探索性数据分析", "请检查数据质量，总结关键分布和变量关系，并给出可视化代码。"],
    ["预测建模", "请设计一个预测方案，包括特征工程、数据划分、基线模型和评价指标。"],
    ["数学抽象", "把这个业务问题转换成数学建模问题，明确变量、目标函数、约束和假设。"],
    ["假设检验", "如何比较A/B两组的差异？请说明检验流程、效应量和Python代码。"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B Data Agent Web Demo")
    parser.add_argument("-c", "--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU inference")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def _model_class() -> Any:
    try:
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM
    except ImportError:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText


def load_model(args: argparse.Namespace) -> tuple[Any, Any, str]:
    use_gpu = torch.cuda.is_available() and not args.cpu_only
    device = "cuda" if use_gpu else "cpu"
    dtype = torch.bfloat16 if use_gpu and torch.cuda.is_bf16_supported() else (
        torch.float16 if use_gpu else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        local_files_only=args.local_files_only,
        padding_side="left",
    )
    model = _model_class().from_pretrained(
        args.checkpoint_path,
        dtype=dtype,
        device_map="auto" if use_gpu else "cpu",
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def _csv_path(file_value: Any) -> Path:
    if isinstance(file_value, str):
        return Path(file_value)
    name = getattr(file_value, "name", None)
    if name:
        return Path(name)
    raise ValueError("无法读取上传的文件路径。")


def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, nrows=10_000), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"CSV编码无法识别：{last_error}")


def build_data_context(frame: pd.DataFrame, filename: str, encoding: str) -> str:
    missing = frame.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    missing_text = missing.head(20).to_string() if not missing.empty else "None"
    numeric = frame.select_dtypes(include="number")
    describe_text = (
        numeric.describe().transpose().round(4).head(30).to_string()
        if not numeric.empty
        else "No numeric columns"
    )
    context = f"""Uploaded dataset summary (computed locally, not generated):
- File: {filename}
- Parsed encoding: {encoding}
- Loaded shape: {frame.shape[0]} rows x {frame.shape[1]} columns

Columns and dtypes:
{frame.dtypes.to_string()}

Missing-value counts (non-zero, top 20):
{missing_text}

Numeric descriptive statistics (up to 30 columns):
{describe_text}

First 8 rows:
{frame.head(8).to_csv(index=False)}
"""
    return context[:MAX_CONTEXT_CHARS]


def inspect_csv(file_value: Any) -> tuple[pd.DataFrame, str, str]:
    if file_value is None:
        return pd.DataFrame(), "尚未上传CSV。也可以直接输入通用数据分析问题。", ""
    try:
        path = _csv_path(file_value)
        if path.suffix.lower() != ".csv":
            raise ValueError("请上传.csv文件。")
        frame, encoding = _read_csv(path)
        context = build_data_context(frame, path.name, encoding)
        missing_cells = int(frame.isna().sum().sum())
        summary = (
            f"**已读取 `{path.name}`**  \n"
            f"{len(frame):,}行 × {len(frame.columns):,}列｜"
            f"编码 `{encoding}`｜缺失单元格 {missing_cells:,}个  \n"
            "下方仅预览前20行；模型会收到字段、缺失值、描述统计和前8行样例。"
        )
        return frame.head(20), summary, context
    except Exception as exc:
        raise gr.Error(f"CSV读取失败：{exc}") from exc


def clear_csv() -> tuple[None, pd.DataFrame, str, str]:
    return None, pd.DataFrame(), "尚未上传CSV。也可以直接输入通用数据分析问题。", ""


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def stream_answer(
    model: Any,
    tokenizer: Any,
    query: str,
    history: list[tuple[str, str]],
    data_context: str,
    task_type: str,
    max_new_tokens: int,
) -> Iterator[str]:
    system = SYSTEM_PROMPT
    if data_context:
        system += "\n\n" + data_context
    system += "\n\nTask emphasis: " + TASK_GUIDANCE[task_type]
    messages = [{"role": "system", "content": text_content(system)}]
    for old_query, old_answer in history:
        messages.append({"role": "user", "content": text_content(old_query)})
        messages.append({"role": "assistant", "content": text_content(old_answer)})
    messages.append({"role": "user", "content": text_content(query)})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.eos_token_id,
    }
    error: list[BaseException] = []

    def generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except BaseException as exc:
            error.append(exc)

    thread = Thread(target=generate, daemon=True)
    thread.start()
    for text in streamer:
        yield text
    thread.join()
    if error:
        raise RuntimeError(str(error[0])) from error[0]


def launch_demo(args: argparse.Namespace, model: Any, tokenizer: Any, device: str) -> None:
    def predict(
        query: str,
        task_type: str,
        data_context: str,
        chatbot: list[dict[str, str]] | None,
        history: list[tuple[str, str]] | None,
    ) -> Iterator[tuple[list[dict[str, str]], list[tuple[str, str]]]]:
        query = (query or "").strip()
        if not query:
            query = "请根据已上传的数据设计一个完整、可验证的分析方案。"
        chatbot = list(chatbot or [])
        history = list(history or [])
        chatbot.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": ""},
            ]
        )
        answer = ""
        try:
            for piece in stream_answer(
                model, tokenizer, query, history, data_context, task_type, args.max_new_tokens
            ):
                answer += piece
                chatbot[-1] = {"role": "assistant", "content": answer}
                yield chatbot, history
        except Exception as exc:
            answer += f"\n\n⚠️ 生成失败：`{exc}`"
            chatbot[-1] = {"role": "assistant", "content": answer}
        history.append((query, answer))
        yield chatbot, history

    def reset_chat() -> tuple[list[Any], list[Any]]:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [], []

    css = """
    .gradio-container {max-width: 1180px !important; margin: auto;}
    .hero {padding: 18px 22px; border-radius: 16px; background: linear-gradient(120deg,#eef5ff,#f7f2ff);}
    .status {font-size: 0.92rem; color: #475569;}
    """
    with gr.Blocks(title="Data Agent Demo") as demo:
        gr.HTML(
            f"""<div class="hero"><h1>Data Agent · 数据分析智能助手</h1>
            <p>自然语言问题 → 数学抽象 → 分析流程 → Python代码 → 结论</p>
            <p class="status">模型：{args.checkpoint_path}｜设备：{device.upper()}｜
            代码仅生成、不在本机自动执行</p></div>"""
        )
        data_context = gr.State("")
        chat_history = gr.State([])
        with gr.Row():
            with gr.Column(scale=4):
                csv_file = gr.File(label="1. 上传CSV（可选）", file_types=[".csv"], type="filepath")
            with gr.Column(scale=2):
                clear_data_button = gr.Button("清除数据")
        data_summary = gr.Markdown("尚未上传CSV。也可以直接输入通用数据分析问题。")
        preview = gr.Dataframe(label="数据预览（前20行）", interactive=False, wrap=True)
        with gr.Row():
            task_type = gr.Dropdown(
                list(TASK_GUIDANCE), value="自动选择", label="2. 选择分析任务"
            )
            query = gr.Textbox(
                lines=3,
                label="3. 描述你的问题",
                placeholder="例如：哪些因素与销售额最相关？应该如何建模验证？",
            )
        chatbot = gr.Chatbot(label="Data Agent 输出", height=540, render_markdown=True)
        with gr.Row():
            submit = gr.Button("开始分析", variant="primary")
            clear_chat_button = gr.Button("清空对话")
        gr.Examples(EXAMPLES, inputs=[task_type, query], label="示例任务")
        gr.Markdown(
            "*Data Agent Web Demo · Adapted from the official Qwen Gradio demo · "
            "Uploaded data stays in the local Gradio session.*"
        )

        csv_file.change(inspect_csv, inputs=[csv_file], outputs=[preview, data_summary, data_context])
        clear_data_button.click(
            clear_csv, outputs=[csv_file, preview, data_summary, data_context]
        )
        submit.click(
            predict,
            inputs=[query, task_type, data_context, chatbot, chat_history],
            outputs=[chatbot, chat_history],
            show_progress=True,
        ).then(lambda: "", outputs=[query])
        query.submit(
            predict,
            inputs=[query, task_type, data_context, chatbot, chat_history],
            outputs=[chatbot, chat_history],
            show_progress=True,
        ).then(lambda: "", outputs=[query])
        clear_chat_button.click(reset_chat, outputs=[chatbot, chat_history])

    demo.queue(default_concurrency_limit=1).launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_name=args.server_name,
        server_port=args.server_port,
        css=css,
    )


def main() -> None:
    args = parse_args()
    print(f"Loading {args.checkpoint_path} ...")
    model, tokenizer, device = load_model(args)
    print(f"Model loaded on {device}. Opening http://{args.server_name}:{args.server_port}")
    launch_demo(args, model, tokenizer, device)


if __name__ == "__main__":
    main()
