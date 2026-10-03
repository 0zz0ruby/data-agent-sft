"""Prepare an auditable semi-synthetic dataset for the CrossMetric AI demo.

Observed fields come from UCI Online Retail II. Operational fields that the
public dataset does not contain are simulated with a fixed seed and are marked
explicitly in the output. The script never alters the original workbook.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "raw" / "online_retail_ii" / "online_retail_II.xlsx"
OUT = ROOT / "processed"
SEED = 20261002


def stable_fraction(value: object, salt: str = "") -> float:
    digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_category(description: str) -> str:
    text = str(description).upper()
    rules = {
        "Seasonal": ("CHRISTMAS", "EASTER", "HALLOWEEN", "VALENTINE"),
        "Kitchen": ("MUG", "CUP", "PLATE", "BOWL", "KITCHEN", "TEA"),
        "Home Decor": ("LANTERN", "CANDLE", "FRAME", "LIGHT", "DECORATION"),
        "Stationery": ("NOTEBOOK", "PENCIL", "PEN ", "CARD", "PAPER"),
        "Accessories": ("BAG", "PURSE", "NECKLACE", "BRACELET", "UMBRELLA"),
    }
    for category, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "Gifts"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for sheet in ("Year 2009-2010", "Year 2010-2011"):
        frame = pd.read_excel(SOURCE, sheet_name=sheet)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.rename(
        columns={
            "Invoice": "invoice_id",
            "StockCode": "sku",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_timestamp",
            "Price": "unit_price_gbp",
            "Customer ID": "customer_id",
            "Country": "country",
        }
    )
    data["invoice_id"] = data["invoice_id"].astype(str)
    data["sku"] = data["sku"].astype(str)
    data["description"] = data["description"].fillna("Unknown item").str.strip()
    data["country"] = data["country"].fillna("Unknown")
    data["invoice_timestamp"] = pd.to_datetime(data["invoice_timestamp"], errors="coerce")
    data["unit_price_gbp"] = pd.to_numeric(data["unit_price_gbp"], errors="coerce")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data = data.dropna(subset=["invoice_timestamp", "unit_price_gbp", "quantity"])
    data = data[(data["unit_price_gbp"] > 0) & (data["quantity"] != 0)].copy()
    data["is_return"] = data["invoice_id"].str.upper().str.startswith("C") | (data["quantity"] < 0)
    data["category"] = data["description"].map(assign_category)
    data["order_date"] = data["invoice_timestamp"].dt.date.astype(str)
    data["observed_net_revenue_gbp"] = data["quantity"] * data["unit_price_gbp"]

    # Uniform fixed-seed sampling preserves the observed return mix while keeping
    # the interactive demo responsive.
    demo = data.sample(n=min(65_000, len(data)), random_state=SEED).copy()
    demo = demo.sort_values("invoice_timestamp").reset_index(drop=True)

    channels = np.array(["Organic", "Meta Ads", "Google Ads", "TikTok Ads", "Email", "Marketplace"])
    probs = np.array([0.29, 0.19, 0.20, 0.10, 0.10, 0.12])
    rng = np.random.default_rng(SEED)
    demo["channel"] = rng.choice(channels, size=len(demo), p=probs)
    category_cost_base = {
        "Seasonal": 0.43,
        "Kitchen": 0.39,
        "Home Decor": 0.46,
        "Stationery": 0.34,
        "Accessories": 0.42,
        "Gifts": 0.40,
    }
    cost_ratio = np.array(
        [category_cost_base[c] + (stable_fraction(sku, "cogs") - 0.5) * 0.12 for sku, c in zip(demo["sku"], demo["category"])]
    )
    cost_ratio = np.clip(cost_ratio, 0.24, 0.62)
    demo["sim_unit_cost_gbp"] = (demo["unit_price_gbp"] * cost_ratio).round(4)
    paid = demo["channel"].isin(["Meta Ads", "Google Ads", "TikTok Ads"])
    ad_rate = rng.uniform(0.08, 0.24, len(demo))
    demo["sim_allocated_ad_spend_gbp"] = np.where(
        paid & ~demo["is_return"],
        np.maximum(demo["observed_net_revenue_gbp"], 0) * ad_rate,
        0.0,
    ).round(4)
    demo["sim_platform_fee_gbp"] = np.where(
        ~demo["is_return"],
        np.maximum(demo["observed_net_revenue_gbp"], 0) * rng.uniform(0.025, 0.075, len(demo)),
        0.0,
    ).round(4)
    units = demo["quantity"].abs()
    demo["sim_fulfilment_cost_gbp"] = np.where(
        ~demo["is_return"], 1.15 + units * rng.uniform(0.08, 0.22, len(demo)), 0.45
    ).round(4)
    # A return reverses recognized COGS when inventory can be recovered, while
    # still incurring a separate return handling cost.
    demo["sim_cogs_gbp"] = np.where(
        demo["is_return"], -units * demo["sim_unit_cost_gbp"], units * demo["sim_unit_cost_gbp"]
    ).round(4)
    demo["sim_contribution_profit_gbp"] = (
        demo["observed_net_revenue_gbp"]
        - demo["sim_cogs_gbp"]
        - demo["sim_allocated_ad_spend_gbp"]
        - demo["sim_platform_fee_gbp"]
        - demo["sim_fulfilment_cost_gbp"]
    ).round(4)
    demo["observed_source"] = "UCI Online Retail II"
    demo["operational_field_status"] = "cost, ad, fee and fulfilment fields are simulated"
    demo.insert(0, "line_id", np.arange(1, len(demo) + 1))

    transactions_path = OUT / "crossmetric_demo_transactions.csv"
    demo.to_csv(transactions_path, index=False, encoding="utf-8-sig")

    product_daily = (
        demo[~demo["is_return"]]
        .groupby(["sku", "description", "category"], as_index=False)
        .agg(
            units_sold=("quantity", "sum"),
            avg_unit_price_gbp=("unit_price_gbp", "mean"),
            sim_unit_cost_gbp=("sim_unit_cost_gbp", "mean"),
        )
        .sort_values("units_sold", ascending=False)
        .head(500)
    )
    demand_scale = np.maximum(product_daily["units_sold"].to_numpy() / 365.0, 0.2)
    lead_time = rng.integers(4, 29, len(product_daily))
    safety = rng.uniform(1.15, 1.80, len(product_daily))
    product_daily["sim_lead_time_days"] = lead_time
    product_daily["sim_reorder_point_units"] = np.ceil(demand_scale * lead_time * safety).astype(int)
    product_daily["sim_on_hand_units"] = rng.integers(0, np.maximum(product_daily["sim_reorder_point_units"] * 3, 2))
    product_daily["sim_holding_cost_per_unit_month_gbp"] = (
        product_daily["sim_unit_cost_gbp"] * rng.uniform(0.015, 0.035, len(product_daily))
    ).round(4)
    product_daily["operational_field_status"] = "inventory attributes are simulated"
    product_daily.to_csv(OUT / "crossmetric_demo_inventory.csv", index=False, encoding="utf-8-sig")

    ad = demo[demo["channel"].isin(["Meta Ads", "Google Ads", "TikTok Ads"]) & ~demo["is_return"]].copy()
    ad_daily = ad.groupby(["order_date", "channel"], as_index=False).agg(
        sim_spend_gbp=("sim_allocated_ad_spend_gbp", "sum"),
        observed_attributed_revenue_gbp=("observed_net_revenue_gbp", "sum"),
        orders=("invoice_id", "nunique"),
    )
    ad_daily["sim_clicks"] = np.maximum((ad_daily["sim_spend_gbp"] / rng.uniform(0.35, 1.40, len(ad_daily))).round(), 1).astype(int)
    ad_daily["sim_impressions"] = (ad_daily["sim_clicks"] / rng.uniform(0.008, 0.035, len(ad_daily))).round().astype(int)
    ad_daily["operational_field_status"] = "channel assignment, spend, clicks and impressions are simulated"
    ad_daily.to_csv(OUT / "crossmetric_demo_ads.csv", index=False, encoding="utf-8-sig")

    revenue = float(demo["observed_net_revenue_gbp"].sum())
    profit = float(demo["sim_contribution_profit_gbp"].sum())
    summary = {
        "seed": SEED,
        "source": "UCI Online Retail II",
        "doi": "10.24432/C5CG6D",
        "license": "CC BY 4.0",
        "rows": int(len(demo)),
        "orders": int(demo["invoice_id"].nunique()),
        "products": int(demo["sku"].nunique()),
        "countries": int(demo["country"].nunique()),
        "date_min": str(demo["invoice_timestamp"].min()),
        "date_max": str(demo["invoice_timestamp"].max()),
        "observed_net_revenue_gbp": round(revenue, 2),
        "sim_contribution_profit_gbp": round(profit, 2),
        "return_line_rate": round(float(demo["is_return"].mean()), 6),
        "observed_fields": [
            "invoice_id", "sku", "description", "quantity", "invoice_timestamp",
            "unit_price_gbp", "customer_id", "country"
        ],
        "simulated_fields": [
            "channel", "sim_unit_cost_gbp", "sim_allocated_ad_spend_gbp",
            "sim_platform_fee_gbp", "sim_fulfilment_cost_gbp",
            "sim_contribution_profit_gbp", "inventory attributes", "ad clicks and impressions"
        ],
    }
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
