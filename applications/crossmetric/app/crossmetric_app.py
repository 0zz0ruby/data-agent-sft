"""CrossMetric AI: evidence-first analytics copilot for cross-border commerce.

The UI computes auditable metrics locally and asks the fine-tuned Qwen Data
Agent to explain them. Generated Python is displayed, never executed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "crossmetric_demo_transactions.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "dataset_summary.json"
OBSERVED = {
    "invoice_id", "sku", "description", "quantity", "invoice_timestamp",
    "unit_price_gbp", "customer_id", "country", "observed_net_revenue_gbp",
}
SIMULATED = {
    "channel", "sim_unit_cost_gbp", "sim_allocated_ad_spend_gbp",
    "sim_platform_fee_gbp", "sim_fulfilment_cost_gbp", "sim_cogs_gbp",
    "sim_contribution_profit_gbp",
}
TASK_LABELS = {
    "English": {
        "Profit Leak": "Profit Leak",
        "Marketing Efficiency": "Marketing Efficiency",
        "Inventory Risk": "Inventory Risk",
        "Trend Diagnosis": "Trend Diagnosis",
    },
    "中文": {
        "Profit Leak": "利润泄漏",
        "Marketing Efficiency": "营销效率",
        "Inventory Risk": "库存风险",
        "Trend Diagnosis": "趋势诊断",
    },
}
DEFAULT_QUESTIONS = {
    "English": {
        "Profit Leak": "Which category is leaking contribution profit, and what should I validate first?",
        "Marketing Efficiency": "Which paid channel is least efficient, and what should I validate before reallocating budget?",
        "Inventory Risk": "Which SKU has the strongest sustained demand concentration, and what inventory data should I validate?",
        "Trend Diagnosis": "What material trend or anomaly should I investigate first?",
    },
    "中文": {
        "Profit Leak": "哪个品类的贡献利润流失最严重？我应该优先验证哪些实际成本字段？",
        "Marketing Efficiency": "哪个付费渠道的效率最低？在调整预算前应核验哪些数据？",
        "Inventory Risk": "哪个 SKU 的持续需求集中度最高？我应该核验哪些库存数据？",
        "Trend Diagnosis": "当前最值得优先调查的趋势或异常是什么？",
    },
}


def canonical_task(value: str) -> str:
    for labels in TASK_LABELS.values():
        for canonical, localized in labels.items():
            if value in {canonical, localized}:
                return canonical
    return "Profit Leak"


def chinese(language: str) -> bool:
    return language == "中文"


def default_question(language: str, task: str) -> str:
    return DEFAULT_QUESTIONS[language][canonical_task(task)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    p.add_argument("--server-name", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=6007)
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def load_frame(path: str | None = None) -> pd.DataFrame:
    source = Path(path) if path else DEFAULT_DATA
    df = pd.read_csv(source, low_memory=False)
    df["invoice_timestamp"] = pd.to_datetime(df["invoice_timestamp"], errors="coerce")
    df["is_return"] = df["is_return"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def money(x: float) -> str:
    return f"£{x:,.0f}"


def compute_evidence(df: pd.DataFrame, task: str, language: str = "English") -> tuple[str, pd.DataFrame, Any, dict[str, Any]]:
    revenue = float(df["observed_net_revenue_gbp"].sum())
    profit = float(df["sim_contribution_profit_gbp"].sum())
    ad_spend = float(df["sim_allocated_ad_spend_gbp"].sum())
    orders = int(df["invoice_id"].nunique())
    returns = float(df["is_return"].mean())
    margin = profit / revenue if revenue else 0.0
    decision: dict[str, Any] = {"task": task}

    if task == "Profit Leak":
        table = (df.groupby("category", as_index=False)
                 .agg(revenue=("observed_net_revenue_gbp", "sum"),
                      contribution_profit=("sim_contribution_profit_gbp", "sum"),
                      cogs=("sim_cogs_gbp", "sum"),
                      ad_spend=("sim_allocated_ad_spend_gbp", "sum"),
                      platform_fee=("sim_platform_fee_gbp", "sum"),
                      fulfilment=("sim_fulfilment_cost_gbp", "sum"),
                      return_line_rate=("is_return", "mean")))
        table["margin"] = table["contribution_profit"] / table["revenue"].replace(0, np.nan)
        table["margin_gap_pp"] = (margin - table["margin"]) * 100
        table["estimated_profit_leakage"] = (
            margin * table["revenue"] - table["contribution_profit"]
        ).clip(lower=0)
        eligible = table.loc[table["revenue"] > 0].sort_values(
            ["estimated_profit_leakage", "margin"], ascending=[False, True]
        )
        if eligible.empty:
            raise ValueError("Profit Leak requires at least one category with positive revenue")
        target = eligible.iloc[0]
        decision = {
            "task": task,
            "target_category": str(target["category"]),
            "target_revenue": float(target["revenue"]),
            "target_profit": float(target["contribution_profit"]),
            "target_margin": float(target["margin"]),
            "portfolio_margin": float(margin),
            "margin_gap_pp": float(target["margin_gap_pp"]),
            "estimated_profit_leakage": float(target["estimated_profit_leakage"]),
            "cogs_share": float(target["cogs"] / target["revenue"]),
            "ad_share": float(target["ad_spend"] / target["revenue"]),
            "fulfilment_share": float(target["fulfilment"] / target["revenue"]),
            "return_line_rate": float(target["return_line_rate"]),
            "known_categories": [str(value) for value in table["category"].tolist()],
            "cost_basis": "simulated",
        }
        table = table.sort_values(
            ["estimated_profit_leakage", "margin"], ascending=[False, True]
        ).round(4)
        fig = px.bar(
            table,
            x="category",
            y="estimated_profit_leakage",
            color="margin",
            title=("各品类估算利润泄漏（模拟成本口径）" if chinese(language)
                   else "Estimated profit leakage by category (simulated cost basis)"),
        )
    elif task == "Marketing Efficiency":
        table = (df.groupby("channel", as_index=False)
                 .agg(revenue=("observed_net_revenue_gbp", "sum"),
                      contribution_profit=("sim_contribution_profit_gbp", "sum"),
                      ad_spend=("sim_allocated_ad_spend_gbp", "sum"),
                      orders=("invoice_id", "nunique")))
        table["roas"] = table["revenue"] / table["ad_spend"].replace(0, np.nan)
        table["profit_per_ad_pound"] = (
            table["contribution_profit"] / table["ad_spend"].replace(0, np.nan)
        )
        table["profit_per_order"] = table["contribution_profit"] / table["orders"].replace(0, np.nan)
        eligible = table.loc[(table["ad_spend"] > 0) & (table["orders"] > 0)].sort_values(
            ["profit_per_ad_pound", "roas"], ascending=[True, True]
        )
        if eligible.empty:
            raise ValueError("Marketing Efficiency requires at least one channel with positive ad spend")
        target = eligible.iloc[0]
        decision = {
            "task": task,
            "target_entity": str(target["channel"]),
            "target_kind": "channel",
            "target_revenue": float(target["revenue"]),
            "target_profit": float(target["contribution_profit"]),
            "target_ad_spend": float(target["ad_spend"]),
            "target_roas": float(target["roas"]),
            "target_profit_per_ad_pound": float(target["profit_per_ad_pound"]),
            "target_profit_per_order": float(target["profit_per_order"]),
            "known_entities": [str(value) for value in table["channel"].tolist()],
            "selection_rule": "lowest contribution profit per advertising pound among paid channels",
            "cost_basis": "simulated",
        }
        table = table.sort_values(
            ["profit_per_ad_pound", "roas"], ascending=[True, True], na_position="last"
        ).round(4)
        fig = px.bar(
            table.loc[table["ad_spend"] > 0],
            x="channel",
            y="profit_per_ad_pound",
            color="roas",
            title=("付费渠道每广告英镑贡献利润" if chinese(language)
                   else "Contribution profit per advertising pound (paid channels)"),
        )
    elif task == "Inventory Risk":
        all_skus = (df.loc[~df["is_return"]]
                    .groupby(["sku", "description"], as_index=False)
                    .agg(units=("quantity", "sum"),
                         revenue=("observed_net_revenue_gbp", "sum"),
                         contribution_profit=("sim_contribution_profit_gbp", "sum"),
                         orders=("invoice_id", "nunique"),
                         transaction_lines=("invoice_id", "size")))
        minimum_lines = 10
        eligible = all_skus.loc[
            (all_skus["transaction_lines"] >= minimum_lines) & (all_skus["units"] > 0)
        ].sort_values(["units", "orders"], ascending=[False, False])
        if eligible.empty:
            raise ValueError("Inventory Risk requires a SKU with sustained non-return demand")
        target = eligible.iloc[0]
        decision = {
            "task": task,
            "target_entity": str(target["sku"]),
            "target_kind": "SKU",
            "target_description": str(target["description"]),
            "target_units": float(target["units"]),
            "target_orders": int(target["orders"]),
            "target_transaction_lines": int(target["transaction_lines"]),
            "minimum_transaction_lines": minimum_lines,
            "known_entities": [str(value) for value in all_skus["sku"].tolist()],
            "selection_rule": "highest non-return unit demand among SKUs with sustained transaction frequency",
            "inventory_status_available": False,
        }
        table = eligible.head(15).round(3)
        fig = px.bar(table, x="sku", y="units", hover_data=["description"],
                     title=("需要核查库存的持续高需求 SKU" if chinese(language)
                            else "Sustained-demand SKUs for inventory validation"))
    else:
        daily = df.groupby(df["invoice_timestamp"].dt.to_period("M").astype(str), as_index=False).agg(
            revenue=("observed_net_revenue_gbp", "sum"),
            contribution_profit=("sim_contribution_profit_gbp", "sum"),
        )
        table = daily.tail(18).round(3)
        fig = px.line(table, x="invoice_timestamp", y=["revenue", "contribution_profit"],
                      title=("月度观测收入与模拟贡献利润" if chinese(language)
                             else "Monthly observed revenue and simulated contribution profit"))

    if chinese(language):
        evidence = (
            f"数据行数={len(df):,}；订单数={orders:,}；观测净收入={money(revenue)}；"
            f"模拟贡献利润={money(profit)}；模拟利润率={margin:.1%}；"
            f"模拟广告分摊={money(ad_spend)}；退货交易行比例={returns:.2%}。"
        )
    else:
        evidence = (
            f"Rows={len(df):,}; Orders={orders:,}; Observed net revenue={money(revenue)}; "
            f"Simulated contribution profit={money(profit)}; Simulated margin={margin:.1%}; "
            f"Simulated allocated ad spend={money(ad_spend)}; Return-line rate={returns:.2%}."
        )
    return evidence, table, fig, decision


def monte_carlo(df: pd.DataFrame, demand_pct: float, ad_cost_pct: float,
                return_pct: float, language: str = "English", trials: int = 1500) -> tuple[str, Any]:
    base_revenue = float(df["observed_net_revenue_gbp"].sum())
    base_profit = float(df["sim_contribution_profit_gbp"].sum())
    base_ads = float(df["sim_allocated_ad_spend_gbp"].sum())
    base_returns = float(df.loc[df["is_return"], "observed_net_revenue_gbp"].abs().sum())
    rng = np.random.default_rng(20261002)
    demand = rng.normal(1 + demand_pct / 100, 0.07, trials)
    ad_multiplier = rng.normal(1 + ad_cost_pct / 100, 0.05, trials)
    return_multiplier = rng.normal(1 + return_pct / 100, 0.06, trials)
    incremental_revenue = base_revenue * (demand - 1)
    variable_margin = max((base_profit + base_ads) / max(base_revenue, 1), 0.05)
    profits = (base_profit + incremental_revenue * variable_margin
               - base_ads * (ad_multiplier - 1)
               - base_returns * (return_multiplier - 1))
    p5, p50, p95 = np.percentile(profits, [5, 50, 95])
    loss_prob = float((profits < 0).mean())
    if chinese(language):
        summary = (f"{trials:,} 次固定随机种子模拟 · P5 {money(p5)} · 中位数 {money(p50)} · "
                   f"P95 {money(p95)} · 亏损概率 {loss_prob:.1%}。"
                   "这是规划模拟，不是预测保证。")
    else:
        summary = (f"{trials:,} seeded trials · P5 {money(p5)} · median {money(p50)} · "
                   f"P95 {money(p95)} · probability of loss {loss_prob:.1%}. "
                   "This is a planning simulation, not a forecast guarantee.")
    chart = px.histogram(
        x=profits, nbins=45,
        title=("情景分布：模拟贡献利润" if chinese(language)
               else "Scenario distribution: simulated contribution profit"),
    )
    chart.update_xaxes(title="GBP")
    return summary, chart


def verified_finding(decision: dict[str, Any], language: str = "English") -> str:
    task = decision.get("task")
    if task == "Profit Leak":
        if chinese(language):
            return (
                "### 已验证结论\n"
                f"**{decision['target_category']}** 的估算利润率缺口最大。其模拟贡献利润率为 "
                f"**{decision['target_margin']:.2%}**，组合整体利润率为 "
                f"**{decision['portfolio_margin']:.2%}**，相差 "
                f"**{decision['margin_gap_pp']:.2f} 个百分点**，对应约 "
                f"**{money(decision['estimated_profit_leakage'])}** 的基准贡献利润缺口。\n\n"
                "*收入来自观测交易样本；成本、广告、平台费和履约字段为模拟数据。*"
            )
        return (
            "### Verified finding\n"
            f"**{decision['target_category']}** has the largest estimated margin-based profit leakage. "
            f"Its simulated contribution margin is **{decision['target_margin']:.2%}**, versus the "
            f"portfolio margin of **{decision['portfolio_margin']:.2%}**. The margin gap is "
            f"**{decision['margin_gap_pp']:.2f} percentage points**, corresponding to approximately "
            f"**{money(decision['estimated_profit_leakage'])}** of contribution profit below the "
            "portfolio-margin benchmark.\n\n"
            "*Revenue is observed from the transaction sample; cost, advertising, fee and "
            "fulfilment inputs are simulated.*"
        )
    if task == "Marketing Efficiency":
        if chinese(language):
            return (
                "### 已验证结论\n"
                f"在广告支出大于零的渠道中，**{decision['target_entity']}** 的模拟广告效率最低："
                f"每投入 £1 广告费产生 **£{decision['target_profit_per_ad_pound']:.2f}** 的贡献利润。"
                f"其观测收入广告比为 **{decision['target_roas']:.2f}x**，模拟每订单贡献利润为 "
                f"**£{decision['target_profit_per_order']:.2f}**。\n\n"
                "*收入来自观测数据；渠道标签、广告分摊和贡献成本字段为模拟数据。*"
            )
        return (
            "### Verified finding\n"
            f"Among channels with positive advertising spend, **{decision['target_entity']}** has "
            "the lowest simulated contribution profit per advertising pound: "
            f"**£{decision['target_profit_per_ad_pound']:.2f} per £1 of ad spend**. Its observed "
            f"revenue-to-ad-spend ratio is **{decision['target_roas']:.2f}x**, and its simulated "
            f"contribution profit per order is **£{decision['target_profit_per_order']:.2f}**.\n\n"
            "*Revenue is observed; channel labels, advertising allocation and contribution-cost "
            "inputs are simulated.*"
        )
    if task == "Inventory Risk":
        if chinese(language):
            return (
                "### 已验证结论\n"
                f"**SKU {decision['target_entity']} — {decision['target_description']}** 在至少有 "
                f"**{decision['minimum_transaction_lines']} 条交易记录**的 SKU 中，持续非退货需求最强。"
                f"该商品共售出 **{decision['target_units']:,.0f} 件**，涉及 "
                f"**{decision['target_orders']:,} 个订单**。\n\n"
                "*这是需求集中信号，并不代表库存已经不足或即将缺货；当前缺少在手库存与补货周期数据。*"
            )
        return (
            "### Verified finding\n"
            f"**SKU {decision['target_entity']} — {decision['target_description']}** has the strongest "
            "sustained non-return demand among SKUs with at least "
            f"**{decision['minimum_transaction_lines']} transaction lines**. It records "
            f"**{decision['target_units']:,.0f} units** across **{decision['target_orders']:,} orders**.\n\n"
            "*This is a demand-concentration signal, not confirmation of low stock or an imminent "
            "stockout; on-hand inventory and replenishment lead time are unavailable.*"
        )
    return ""


def build_prompt(task: str, question: str, evidence: str, table: pd.DataFrame,
                 decision: dict[str, Any], language: str = "English") -> str:
    compact = table.head(12).to_csv(index=False)
    if task in {"Profit Leak", "Marketing Efficiency", "Inventory Risk"}:
        if task == "Profit Leak":
            selected = decision["target_category"]
            authoritative_payload = {
                "selected_category": selected,
                "selection_rule": "largest positive margin-based estimated profit leakage",
                "cost_basis": "simulated",
            }
            task_rules = (
                "Explain that cost, advertising, fee and fulfilment inputs are simulated assumptions.\n"
                "Recommend validation of actual landed cost, advertising allocation, platform fees, "
                "fulfilment cost and returns."
            )
            task_rules_zh = "说明成本、广告、平台费与履约字段均为模拟假设，并建议核验真实采购成本、广告分摊、平台费、履约成本与退货。"
        elif task == "Marketing Efficiency":
            selected = decision["target_entity"]
            authoritative_payload = {
                "selected_channel": selected,
                "selection_rule": decision["selection_rule"],
                "cost_basis": "simulated",
            }
            task_rules = (
                "Explain that channel attribution, advertising allocation and contribution costs are simulated.\n"
                "Recommend validation of attributed revenue, actual spend, channel fees, conversion tracking "
                "and cohort quality before reallocating budget."
            )
            task_rules_zh = "说明渠道归因、广告分摊与贡献成本均为模拟数据；在调整预算前核验真实广告支出、归因收入、渠道费用、转化追踪和客户质量。"
        else:
            selected = decision["target_entity"]
            authoritative_payload = {
                "selected_sku": selected,
                "description": decision["target_description"],
                "selection_rule": decision["selection_rule"],
                "inventory_status_available": False,
            }
            task_rules = (
                "Describe this only as sustained demand concentration. Do not claim current inventory, "
                "a confirmed stockout, a return rate, or adequate buffer stock.\n"
                "Recommend validation of on-hand inventory, open purchase orders, lead time, reorder point, "
                "seasonality and recent demand."
            )
            task_rules_zh = "只能将其描述为持续需求集中，不得声称已经缺货、库存充足或具有某个退货率；建议核验在手库存、在途采购、供应周期、补货点、季节性和近期需求。"
        authoritative = json.dumps(
            authoritative_payload,
            ensure_ascii=False,
        )
        if chinese(language):
            return f"""你是 CrossMetric AI 的解释层。Python 已完成计算并选定分析对象。不得重新计算、重新排名或替换该结论。

任务：{TASK_LABELS['中文'][task]}
用户问题：{question or '解释已验证结论，并说明下一步应该核验什么。'}
权威决策：{authoritative}
已验证汇总信息：{evidence}

规则：
1. 只能讨论已选定对象：{selected}。
2. 不要重复任何金额、百分比、行数或其他数字，界面会单独显示已验证数字。
3. {task_rules_zh}
4. 不得将相关性描述为因果关系，不得建议自动执行财务操作。
5. 不要输出 HTML、XML、SVG、代码、表格或工具调用标签。

必须使用以下 Markdown 标题：
### 解读
### 建议验证
### 局限性
"""
        return f"""You are the explanation layer of CrossMetric AI. Python has already made
the calculation and selected the entity. You must not recalculate, rank, or
replace that decision.

Task: {task}
User question: {question or 'Explain the verified finding and what to validate.'}
Authoritative decision: {authoritative}
Verified aggregate context: {evidence}

Rules:
1. Discuss only the selected entity: {selected}.
2. Do not repeat any monetary value, percentage, row count, or other number. The UI displays verified numbers separately.
3. {task_rules}
5. Do not claim causality and do not recommend an automatic financial action.
6. Do not output HTML, XML, SVG, code, tables, or tool-call tags.

Use exactly these Markdown headings:
### Interpretation
### Recommended validation
### Limitations
"""
    if chinese(language):
        return f"""你是 CrossMetric AI，一款面向小型跨境电商团队、以证据为基础的数据分析助手。

任务：{TASK_LABELS['中文'].get(task, task)}
用户问题：{question or '找出最值得关注的发现，并提出安全的下一步。'}
已验证汇总信息：{evidence}
支持表格（前 12 行）：
{compact}

请区分观测交易事实与模拟成本/广告假设，不得编造数据或声称因果关系。任何预算、采购或定价变化必须由人工批准。不要输出 HTML、XML、SVG 或工具标签。

使用以下 Markdown 标题：
### 发现
### 证据
### 建议验证
### 局限性
"""
    return f"""You are CrossMetric AI, an evidence-first analytics copilot for a small
cross-border ecommerce team. Answer in concise English unless the user writes Chinese.

Task: {task}
User question: {question or 'Identify the most actionable finding and a safe next step.'}
Verified aggregate evidence: {evidence}
Supporting table (first 12 rows):
{compact}

Rules:
1. Separate observed transaction facts from simulated cost/ad assumptions.
2. State the main finding, evidence, business interpretation, and recommended validation.
3. Do not claim causality from correlation and do not invent unavailable data.
4. Any budget, purchasing, or pricing change requires human approval.
5. Do not output HTML, XML, SVG, tool-call tags, or raw dataframe code.
6. Only provide Python code when the user explicitly asks for code.

Use exactly these Markdown headings:
### Finding
### Evidence
### Recommended validation
### Limitations
"""


def validate_model_response(value: str, decision: dict[str, Any], language: str = "English") -> tuple[bool, str]:
    """Reject a narrative that conflicts with deterministic evidence."""
    task = decision.get("task")
    if task not in {"Profit Leak", "Marketing Efficiency", "Inventory Risk"}:
        return True, "not-applicable"
    lowered = value.casefold()
    required = (
        ["### 解读", "### 建议验证", "### 局限性"]
        if chinese(language)
        else ["### interpretation", "### recommended validation", "### limitations"]
    )
    if any(heading not in lowered for heading in required):
        return False, "missing required narrative sections"
    target = str(decision.get("target_category", decision.get("target_entity", "")))
    if target.casefold() not in lowered:
        return False, "selected entity was omitted"
    if task == "Inventory Risk":
        identifiers = set(re.findall(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]+\b", value))
        if any(identifier.casefold() != target.casefold() for identifier in identifiers):
            return False, "unverified SKU identifier"
    else:
        known = decision.get("known_categories", decision.get("known_entities", []))
        for entity in known:
            if entity.casefold() == target.casefold():
                continue
            if re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", value, flags=re.IGNORECASE):
                return False, f"unverified entity comparison: {entity}"
    numeric_check = re.sub(re.escape(target), "", value, flags=re.IGNORECASE)
    if re.search(r"[£$€%]|\d", numeric_check):
        return False, "model repeated an unverified numeric value"
    contradictions = []
    if task == "Profit Leak":
        contradictions = [
            r"contribution profit.{0,35}(?:higher|greater|exceed).{0,20}revenue",
            r"revenue.{0,35}(?:lower|less).{0,20}contribution profit",
        ]
    elif task == "Marketing Efficiency":
        contradictions = [r"most efficient", r"highly cost-effective", r"best-performing"]
    elif task == "Inventory Risk":
        contradictions = [
            r"has sufficient stock", r"will stock out", r"is out of stock",
            r"inventory is low", r"return-line rate", r"return rate",
        ]
    if any(re.search(pattern, lowered) for pattern in contradictions):
        return False, "financial consistency rule failed"
    return True, "passed"


def safe_narrative(decision: dict[str, Any], language: str = "English") -> str:
    task = decision["task"]
    target = str(decision.get("target_category", decision.get("target_entity", "")))
    if chinese(language):
        if task == "Profit Leak":
            return f"""### 解读
在模拟成本模型下，{target} 的贡献利润率与组合基准的差距最大，因此应优先核查；这并不证明该品类本身就是利润流失的原因。

### 建议验证
使用 {target} 的实际到岸成本发票、广告分摊、平台费用和履约费用替换模拟假设。在调整价格、预算或采购决策前，再检查退货和 SKU 级结果。

### 局限性
交易收入来自观测数据，但利润率计算使用的成本和运营字段为模拟值。该结果仅用于确定核查优先级，采取行动前必须人工复核。"""
        if task == "Marketing Efficiency":
            return f"""### 解读
按照已验证的效率规则，{target} 是最应优先核查的付费渠道。这是基于归因数据的诊断信号，并不能证明该渠道导致了较差的客户经济性。

### 建议验证
将 {target} 的实际平台支出与归因订单和收入进行核对。在调整预算前，检查追踪质量、渠道费用、转化窗口和客户群长期价值。

### 局限性
渠道标签、广告分摊和贡献成本字段为模拟值。该结果用于确定核查顺序，不会自动授权预算调整。"""
        return f"""### 解读
排除稀疏的一次性交易模式后，SKU {target} 显示出最强的持续需求集中度。它应被优先纳入补货核查，但这并不代表当前库存不足。

### 建议验证
在调整补货数量前，检查 SKU {target} 的现有库存、在途采购订单、供应商交付周期、再订货点、季节性和近期需求。

### 局限性
数据集包含交易需求，但不包含权威的现有库存、交付周期或安全库存目标。采取行动前必须由人员完成库存复核。"""
    if task == "Profit Leak":
        return f"""### Interpretation
{target} falls furthest below the portfolio contribution-margin benchmark under the simulated cost model. This makes it the first category to investigate, not proof that the category itself causes the leakage.

### Recommended validation
Replace the simulated assumptions for {target} with actual landed-cost invoices, advertising allocation, platform fees and fulfilment charges. Then review returns and SKU-level results before changing price, budget or purchasing decisions.

### Limitations
The transaction revenue is observed, but the cost and operational fields used in the margin calculation are simulated. The result is a validation priority and requires human review before action."""
    if task == "Marketing Efficiency":
        return f"""### Interpretation
{target} is the first paid channel to investigate under the verified efficiency rule. This is an attribution-based diagnostic signal, not proof that the channel causes poor customer economics.

### Recommended validation
Reconcile actual platform spend with attributed orders and revenue for {target}. Check tracking quality, channel fees, conversion windows and customer-cohort value before changing budget allocation.

### Limitations
Channel labels, advertising allocation and contribution-cost fields are simulated. The result prioritizes validation and does not authorize an automatic budget change."""
    return f"""### Interpretation
SKU {target} shows the strongest sustained demand concentration after excluding sparse, one-off transaction patterns. It is a replenishment-review priority, not confirmation that inventory is currently low.

### Recommended validation
Check on-hand inventory, open purchase orders, supplier lead time, reorder point, seasonality and recent demand for SKU {target} before changing replenishment quantities.

### Limitations
The dataset contains transaction demand but not authoritative on-hand stock, lead time or safety-stock targets. A human inventory review is required before action."""


def clean_model_response(value: str) -> str:
    """Remove trajectory markup that should never reach the product UI."""
    text = str(value).replace("&#x20;", " ")
    # Data-agent trajectories may contain empty/unclosed rendering payloads.
    text = re.sub(r"(?is)```(?:xml|svg)\s*.*?(?:```|\Z)", "", text)
    text = re.sub(r"(?is)</?(?:code|svg|xml)(?:\s+[^>]*)?>", "", text)
    text = re.sub(r"(?im)^\s*(?:svg|xml)\s*$", "", text)
    text = re.sub(r"(?:\n\s*){3,}", "\n\n", text).strip()
    # Keep legitimate Markdown code readable if the user explicitly requested it.
    if text.count("```") % 2:
        text += "\n```"
    if len(text) < 20:
        return "The model did not produce a usable narrative. Please retry with a more specific business question."
    return text


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, local_files_only=args.local_files_only)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, torch_dtype=dtype, local_files_only=args.local_files_only,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer, "CUDA" if torch.cuda.is_available() else "CPU"


CSS = """
.gradio-container {max-width: 1760px !important; padding:20px 26px 42px !important; background:#f5f6f2;}
.hero {background:linear-gradient(135deg,#082f2b,#0f766e); color:white; padding:28px 34px;
 border-radius:20px; margin-bottom:12px; box-shadow:0 14px 35px #0f766e24;}
.hero h1 {font-size:38px; margin:0 0 8px; color:#ffffff !important;}
.hero p {font-size:17px; opacity:.92; margin:4px 0; color:#ddf7f2 !important;}
.eyebrow {letter-spacing:.14em; font-weight:700; color:#99f6e4; font-size:12px;}
.workspace {gap:18px !important; align-items:flex-start !important;}
.workspace-pane {background:#ffffff; border:1px solid #dce5e1; border-radius:18px;
 padding:16px !important; box-shadow:0 10px 28px #0f3d3520;}
.data-pane {position:sticky !important; top:12px; max-height:calc(100vh - 24px);
 overflow-y:auto !important; overflow-x:hidden !important; min-width:0 !important;}
.data-pane > * {max-width:100% !important; min-width:0 !important;}
.analysis-pane {min-height:760px;}
.pane-heading h2 {font-size:21px !important; color:#123c36 !important; margin:0 0 3px !important;}
.pane-heading p {color:#64748b !important; margin:0 0 10px !important; font-size:14px !important;}
.provenance {border-left:5px solid #f59e0b !important; background:#fffbeb !important;
 border-radius:10px; padding:10px 12px !important; margin-top:8px;}
.data-status {background:#ecfdf5 !important; border:1px solid #a7f3d0 !important;
 border-radius:10px; padding:8px 11px !important;}
.recommendation {min-height:300px; padding:8px 12px !important;}
.evidence-strip {background:#eff6ff !important; border-left:4px solid #2563eb !important;
 border-radius:10px; padding:9px 12px !important;}
.primary {background:#ea580c !important; color:white !important; font-weight:700 !important;}
@media (max-width: 1050px) {
 .gradio-container {padding:12px !important;}
 .workspace {display:block !important;}
 .workspace-pane {margin-bottom:16px !important;}
 .data-pane {position:static !important; max-height:none !important; overflow:visible !important;}
}
"""


def hero_html(language: str, device: str) -> str:
    if chinese(language):
        return f"""<div class='hero'><div class='eyebrow'>跨境电商智能决策</div>
        <h1>CrossMetric AI</h1><p>将分散的电商数据转化为可审计的利润决策。</p>
        <p>微调 Qwen 数据智能体 · {device} 推理 · 财务操作须人工批准</p></div>"""
    return f"""<div class='hero'><div class='eyebrow'>CROSS-BORDER COMMERCE INTELLIGENCE</div>
    <h1>CrossMetric AI</h1><p>From fragmented commerce data to auditable profit decisions.</p>
    <p>Fine-tuned Qwen Data Agent · {device} inference · human approval for financial actions</p></div>"""


def provenance_text(language: str, rows: int) -> str:
    if chinese(language):
        return (
            f"**数据来源**  \n观测数据：UCI Online Retail II 交易记录"
            f"（抽样 {rows:,} 行，CC BY 4.0）。  \n"
            "固定随机种子模拟：渠道、销售成本、广告支出、平台费、履约与库存。"
        )
    return (
        f"**Data provenance**  \nObserved: UCI Online Retail II transactions "
        f"({rows:,} sampled rows, CC BY 4.0).  \n"
        "Simulated with fixed seed: channel, COGS, ad spend, fees, fulfilment and inventory."
    )


def main() -> None:
    import torch

    args = parse_args()
    model, tokenizer, device = load_model(args)
    default_df = load_frame(str(args.data_path))
    source_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    def ingest(file_path: str | None, language: str):
        try:
            df = load_frame(file_path) if file_path else default_df.copy()
            missing = (OBSERVED | SIMULATED) - set(df.columns)
            if missing:
                raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
            message = f"已加载 {len(df):,} 行数据。" if chinese(language) else f"Loaded {len(df):,} rows."
            return df, df.head(20), message
        except Exception as exc:
            message = (
                f"上传被拒绝：{exc}。已恢复演示数据。"
                if chinese(language)
                else f"Upload rejected: {exc}. Demo data restored."
            )
            return default_df.copy(), default_df.head(20), message

    def analyze(df: pd.DataFrame, task: str, question: str, language: str):
        task = canonical_task(task)
        evidence, table, fig, decision = compute_evidence(df, task, language)
        prompt = build_prompt(task, question, evidence, table, decision, language)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.05)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response = clean_model_response(response)
        valid, validation_note = validate_model_response(response, decision, language)
        protected_tasks = {"Profit Leak", "Marketing Efficiency", "Inventory Risk"}
        if not valid and task in protected_tasks:
            response = safe_narrative(decision, language)
            validation_note = (
                "已启用安全回答" if chinese(language)
                else f"safe fallback used ({validation_note})"
            )
        verified = evidence
        if task in protected_tasks:
            guardrail = "**叙述校验：** " if chinese(language) else "**Narrative guardrail:** "
            verified = verified_finding(decision, language) + f"\n\n{guardrail}{validation_note}."
            response = verified_finding(decision, language) + "\n\n" + response
        return verified, table, fig, response

    with gr.Blocks(title="CrossMetric AI") as demo:
        # Gradio 6 requires State to be registered inside the Blocks context.
        state = gr.State(default_df)
        with gr.Row():
            language = gr.Radio(
                ["English", "中文"], value="English", label="Language / 语言", scale=1
            )
        hero = gr.HTML(hero_html("English", device))
        with gr.Row(elem_classes="workspace"):
            # Left: a compact, independently scrollable data workspace.
            with gr.Column(scale=5, min_width=430, elem_classes=["workspace-pane", "data-pane"]):
                data_heading = gr.Markdown(
                    "## Data workspace\nUpload, validate and inspect the evidence used by the agent.",
                    elem_classes="pane-heading",
                )
                with gr.Accordion("Data source & provenance · click to upload", open=False) as data_source_acc:
                    upload = gr.File(label="Upload compatible transaction CSV (optional)", type="filepath")
                    load_button = gr.Button("Validate and load data")
                    provenance = gr.Markdown(
                        provenance_text("English", source_summary["rows"]),
                        elem_classes="provenance",
                    )
                with gr.Accordion("Validated preview · first 20 rows", open=True) as preview_acc:
                    preview = gr.Dataframe(
                        value=default_df.head(20),
                        label="Auditable input sample",
                        interactive=False,
                        max_height=390,
                    )
                status = gr.Markdown("**Ready ·** Demo dataset loaded.", elem_classes="data-status")

            # Right: the conversational decision workflow and all generated outputs.
            with gr.Column(scale=7, min_width=620, elem_classes=["workspace-pane", "analysis-pane"]):
                decision_heading = gr.Markdown(
                    "## Decision workspace\nAsk a business question, then inspect the recommendation and its evidence.",
                    elem_classes="pane-heading",
                )
                with gr.Row():
                    task = gr.Dropdown(
                        ["Profit Leak", "Marketing Efficiency", "Inventory Risk", "Trend Diagnosis"],
                        value="Profit Leak",
                        label="Decision workflow",
                        scale=2,
                    )
                    question = gr.Textbox(
                        label="Business question",
                        value=default_question("English", "Profit Leak"),
                        lines=3,
                        scale=3,
                    )
                analyze_btn = gr.Button("Generate evidence-backed recommendation", elem_classes="primary")
                evidence_box = gr.Markdown(
                    "Verified metrics will appear here.",
                    label="Verified evidence",
                    elem_classes="evidence-strip",
                )
                with gr.Tabs():
                    with gr.Tab("Recommendation") as recommendation_tab:
                        answer = gr.Markdown(
                            "Run an analysis to generate a decision memo.",
                            label="Data Agent recommendation",
                            elem_classes="recommendation",
                        )
                    with gr.Tab("Supporting calculation") as calculation_tab:
                        result_table = gr.Dataframe(
                            label="Auditable calculation", interactive=False, max_height=430
                        )
                    with gr.Tab("Decision chart") as chart_tab:
                        result_plot = gr.Plot(label="Decision view")

                with gr.Accordion("Scenario lab · Monte Carlo planning", open=False) as scenario_acc:
                    scenario_desc = gr.Markdown("Stress-test a decision under demand, advertising-cost and return-loss uncertainty.")
                    with gr.Row():
                        demand = gr.Slider(-30, 50, 5, step=1, label="Demand change (%)")
                        ad_cost = gr.Slider(-20, 50, 10, step=1, label="Ad cost change (%)")
                        return_rate = gr.Slider(-20, 50, 5, step=1, label="Return-loss change (%)")
                    scenario_btn = gr.Button("Run 1,500-trial scenario")
                    scenario_text = gr.Markdown()
                    scenario_plot = gr.Plot()

        def switch_language(selected: str, current_task: str):
            zh = chinese(selected)
            canonical = canonical_task(current_task)
            labels = TASK_LABELS[selected]
            return (
                hero_html(selected, device),
                "## 数据工作区\n上传、验证并检查智能体使用的数据证据。" if zh else
                    "## Data workspace\nUpload, validate and inspect the evidence used by the agent.",
                gr.update(label="数据来源与说明 · 点击上传" if zh else "Data source & provenance · click to upload"),
                gr.update(label="上传兼容的交易 CSV（可选）" if zh else "Upload compatible transaction CSV (optional)"),
                gr.update(value="验证并加载数据" if zh else "Validate and load data"),
                provenance_text(selected, source_summary["rows"]),
                gr.update(label="已验证预览 · 前20行" if zh else "Validated preview · first 20 rows"),
                gr.update(label="可审计输入样本" if zh else "Auditable input sample"),
                "**就绪 ·** 已加载演示数据。" if zh else "**Ready ·** Demo dataset loaded.",
                "## 决策工作区\n提出业务问题，并检查建议及其证据。" if zh else
                    "## Decision workspace\nAsk a business question, then inspect the recommendation and its evidence.",
                gr.update(choices=list(labels.values()), value=labels[canonical],
                          label="分析任务" if zh else "Decision workflow"),
                gr.update(label="业务问题" if zh else "Business question",
                          value=default_question(selected, canonical)),
                gr.update(value="生成有证据支持的建议" if zh else "Generate evidence-backed recommendation"),
                gr.update(value="已验证指标将显示在这里。" if zh else "Verified metrics will appear here.",
                          label="已验证证据" if zh else "Verified evidence"),
                gr.update(label="建议" if zh else "Recommendation"),
                gr.update(label="支持计算" if zh else "Supporting calculation"),
                gr.update(label="决策图表" if zh else "Decision chart"),
                gr.update(value="运行分析以生成决策说明。" if zh else "Run an analysis to generate a decision memo.",
                          label="数据智能体建议" if zh else "Data Agent recommendation"),
                gr.update(label="情景实验 · 蒙特卡洛规划" if zh else "Scenario lab · Monte Carlo planning"),
                "在需求、广告成本和退货损失的不确定性下对决策进行压力测试。" if zh else
                    "Stress-test a decision under demand, advertising-cost and return-loss uncertainty.",
                gr.update(label="需求变化（%）" if zh else "Demand change (%)"),
                gr.update(label="广告成本变化（%）" if zh else "Ad cost change (%)"),
                gr.update(label="退货损失变化（%）" if zh else "Return-loss change (%)"),
                gr.update(value="运行1,500次情景模拟" if zh else "Run 1,500-trial scenario"),
            )

        language.change(
            switch_language,
            [language, task],
            [hero, data_heading, data_source_acc, upload, load_button, provenance,
             preview_acc, preview, status, decision_heading, task, question, analyze_btn,
             evidence_box, recommendation_tab, calculation_tab, chart_tab, answer,
             scenario_acc, scenario_desc, demand, ad_cost, return_rate, scenario_btn],
        )
        task.change(lambda selected_task, selected_language: default_question(selected_language, selected_task),
                    [task, language], question)
        load_button.click(ingest, [upload, language], [state, preview, status])
        analyze_btn.click(analyze, [state, task, question, language], [evidence_box, result_table, result_plot, answer])
        scenario_btn.click(monte_carlo, [state, demand, ad_cost, return_rate, language], [scenario_text, scenario_plot])

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        css=CSS,
    )


if __name__ == "__main__":
    main()
