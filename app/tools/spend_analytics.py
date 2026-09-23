"""Spend analytics tool: structured queries over the transactions dataset (pandas).

Operations:
- ``total``     : sum of amount_eur, optionally filtered by category and/or quarter.
- ``top_n``     : top-N groups by total spend, grouped by 'category' or 'vendor'.
- ``anomalies`` : monthly category totals that deviate > 2 std devs from that
                  category's mean monthly total (z-score based).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.tools.base import BaseTool

_Z_THRESHOLD = 2.0


class SpendAnalyticsTool(BaseTool):
    name = "spend_analytics"
    description = (
        "Analyze the spend transactions dataset. operation in [total, top_n, anomalies]. "
        "Optional filters: category, quarter (1-4), month (YYYY-MM). For top_n: by in "
        "[category, vendor], n. Returns numeric results suitable for grounding a final answer."
    )

    def __init__(self, csv_path: str | Path) -> None:
        self._csv_path = Path(csv_path)
        self._df = self._load()

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self._csv_path, parse_dates=["date"])
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.strftime("%Y-%m")
        return df

    def run(self, operation: str = "total", **kwargs: Any) -> dict[str, Any]:
        op = (operation or "total").lower()
        if op == "total":
            return self._total(
                kwargs.get("category"), kwargs.get("quarter"), kwargs.get("month")
            )
        if op == "top_n":
            return self._top_n(kwargs.get("by", "category"), int(kwargs.get("n", 3)))
        if op == "anomalies":
            return self._anomalies()
        return {"operation": op, "error": f"unknown operation '{op}'"}

    # ---- operations ---------------------------------------------------------
    def _total(
        self, category: str | None, quarter: int | None, month: str | None = None
    ) -> dict[str, Any]:
        df = self._df
        if category:
            df = df[df["category"].str.lower() == str(category).lower()]
        if quarter:
            df = df[df["quarter"] == int(quarter)]
        if month:
            df = df[df["month"] == str(month)]
        return {
            "operation": "total",
            "category": category,
            "quarter": quarter,
            "month": month,
            "total_eur": round(float(df["amount_eur"].sum()), 2),
            "count": int(len(df)),
        }

    def _top_n(self, by: str, n: int) -> dict[str, Any]:
        by = by if by in ("category", "vendor") else "category"
        grouped = (
            self._df.groupby(by)["amount_eur"].sum().sort_values(ascending=False).head(n)
        )
        rows = [{"key": k, "total_eur": round(float(v), 2)} for k, v in grouped.items()]
        return {"operation": "top_n", "by": by, "n": n, "rows": rows}

    def _anomalies(self) -> dict[str, Any]:
        monthly = (
            self._df.groupby(["category", "month"])["amount_eur"].sum().reset_index()
        )
        anomalies: list[dict[str, Any]] = []
        for category, grp in monthly.groupby("category"):
            if len(grp) < 3:
                continue
            mean = grp["amount_eur"].mean()
            std = grp["amount_eur"].std(ddof=0)
            if std == 0:
                continue
            for _, row in grp.iterrows():
                z = (row["amount_eur"] - mean) / std
                if abs(z) > _Z_THRESHOLD:
                    anomalies.append(
                        {
                            "category": category,
                            "month": row["month"],
                            "total_eur": round(float(row["amount_eur"]), 2),
                            "z_score": round(float(z), 2),
                        }
                    )
        anomalies.sort(key=lambda a: abs(a["z_score"]), reverse=True)
        return {"operation": "anomalies", "threshold_z": _Z_THRESHOLD, "anomalies": anomalies}
