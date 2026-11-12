"""审核容量约束下的策略评估。

线上风控团队每天能人工审核的交易数量是有限的。
因此“模型有多准”不是唯一问题，更实际的问题是：
在每天固定 N 个名额的前提下，应该按什么顺序把交易送进审核队列？
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from . import config

SegmentSelector = Callable[[pd.DataFrame], pd.Series]

# 排序策略：名称 -> (排序字段, 升序标志)
SORT_STRATEGIES: dict[str, tuple[list[str], list[bool]]] = {
    "amount": (["TransactionAmt"], [False]),
    "score": (["score", "TransactionAmt"], [False, False]),
}


def top_queue(
    part: pd.DataFrame,
    sort_columns: list[str],
    ascending: list[bool],
    capacity: int | None = None,
) -> pd.DataFrame:
    """每天按指定字段排序，取前 capacity 笔进入审核队列。

    末尾追加 TransactionID 作为最终 tie-breaker：金额和分数都可能并列，
    若没有唯一键，截断处谁入选将取决于输入行序，指标会随数据顺序漂移。
    """
    capacity = config.DAILY_REVIEW_CAPACITY if capacity is None else capacity

    columns = list(sort_columns)
    directions = list(ascending)
    if "TransactionID" not in columns:
        columns.append("TransactionID")
        directions.append(True)

    ordered = part.sort_values(
        ["day"] + columns,
        ascending=[True] + directions,
    )
    return ordered.groupby("day", sort=False).head(capacity)


def summarize_queue(part: pd.DataFrame, queue: pd.DataFrame) -> dict:
    """汇总一个审核队列的规模、命中率与召回率。"""
    reviews = len(queue)
    caught = int(queue["isFraud"].sum())
    total_fraud = int(part["isFraud"].sum())
    return {
        "reviews": reviews,
        "fraud_caught": caught,
        "precision": caught / reviews if reviews else 0.0,
        "recall": caught / total_fraud if total_fraud else 0.0,
    }


def capacity_curve(
    periods: dict[str, pd.DataFrame],
    capacities: tuple[int, ...] = (25, 50, 100, 200, 500),
    strategies: dict[str, tuple[list[str], list[bool]]] | None = None,
) -> pd.DataFrame:
    """容量曲线：不同审核名额下，各排序策略的效果。"""
    strategies = strategies or SORT_STRATEGIES
    rows = []
    for period_name, part in periods.items():
        for capacity in capacities:
            for strategy_name, (columns, ascending) in strategies.items():
                queue = top_queue(part, columns, ascending, capacity)
                metrics = summarize_queue(part, queue)
                rows.append(
                    {
                        "period": period_name,
                        "strategy": strategy_name,
                        "capacity": capacity,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def segment_stability(
    periods: dict[str, pd.DataFrame],
    segments: dict[str, SegmentSelector] | None = None,
) -> pd.DataFrame:
    """检查关键风险群体在三个时间段上的欺诈率是否稳定。

    如果某个群体只在训练期高、测试期回落，说明它可能是噪声，
    不应该据此制定规则。
    """
    segments = segments or {
        "产品C + 信用卡": lambda x: (x["ProductCD"] == "C")
        & (x["card6"] == "credit"),
        "产品C + 借记卡": lambda x: (x["ProductCD"] == "C")
        & (x["card6"] == "debit"),
        "W + 借记卡": lambda x: (x["ProductCD"] == "W")
        & (x["card6"] == "debit"),
        "所有信用卡": lambda x: x["card6"] == "credit",
        "gmail 邮箱": lambda x: x["P_emaildomain"] == "gmail.com",
        "hotmail / outlook": lambda x: x["P_emaildomain"].isin(
            ["hotmail.com", "outlook.com"]
        ),
    }

    rows = []
    for label, selector in segments.items():
        row = {"segment": label}
        for period_name, part in periods.items():
            sub = part[selector(part)]
            row[f"{period_name}_transactions"] = len(sub)
            row[f"{period_name}_rate"] = (
                float(sub["isFraud"].mean()) if len(sub) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def score_tier_table(periods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """按固定分数档位统计交易量、欺诈率与欺诈占比。

    固定档位回答的是“阈值定在哪”的问题；
    容量曲线回答的是“名额给多少”的问题，两者互补。
    """
    tiers = [
        ("分数 ≥ 0.20", lambda x: x["score"] >= 0.20),
        ("0.10 ≤ 分数 < 0.20", lambda x: (x["score"] >= 0.10) & (x["score"] < 0.20)),
        ("0.05 ≤ 分数 < 0.10", lambda x: (x["score"] >= 0.05) & (x["score"] < 0.10)),
        ("分数 < 0.05", lambda x: x["score"] < 0.05),
    ]

    rows = []
    for period_name, part in periods.items():
        days = part["day"].nunique()
        total_fraud = int(part["isFraud"].sum())
        for label, selector in tiers:
            sub = part[selector(part)]
            frauds = int(sub["isFraud"].sum())
            rows.append(
                {
                    "period": period_name,
                    "tier": label,
                    "transactions": len(sub),
                    "daily_transactions": len(sub) / days if days else 0.0,
                    "frauds": frauds,
                    "fraud_rate": frauds / len(sub) if len(sub) else 0.0,
                    "fraud_share": frauds / total_fraud if total_fraud else 0.0,
                }
            )
    return pd.DataFrame(rows)


def policy_comparison(
    periods: dict[str, pd.DataFrame],
    capacity: int | None = None,
) -> pd.DataFrame:
    """比较三种“自动拒绝 + 人工审核 + 放行”的三档方案。

    A 激进：高风险直接拒绝，次高风险进入审核池
    B 保守：只拒绝极高风险，其余进入审核池
    C 金额门槛：高风险且金额较大才拒绝，缩小拒绝面
    """
    capacity = config.DAILY_REVIEW_CAPACITY if capacity is None else capacity

    schemes = {
        "A 激进": (
            lambda x: x["score"] >= 0.20,
            lambda x: (x["score"] >= 0.05) & (x["score"] < 0.20),
        ),
        "B 保守": (
            lambda x: x["score"] >= 0.25,
            lambda x: (x["score"] >= 0.05) & (x["score"] < 0.25),
        ),
        "C 金额门槛": (
            lambda x: (x["score"] >= 0.20) & (x["TransactionAmt"] >= 100),
            lambda x: (x["score"] >= 0.05)
            & ~((x["score"] >= 0.20) & (x["TransactionAmt"] >= 100)),
        ),
    }

    rows = []
    for period_name, part in periods.items():
        days = part["day"].nunique()
        total_fraud = int(part["isFraud"].sum())

        for policy_name, (reject_selector, review_selector) in schemes.items():
            reject = part[reject_selector(part)]
            review_pool = part[review_selector(part)]
            queue = top_queue(
                review_pool,
                ["score", "TransactionAmt"],
                [False, False],
                capacity,
            )

            reject_n = len(reject)
            reject_fraud = int(reject["isFraud"].sum())
            review_caught = int(queue["isFraud"].sum())
            total_caught = reject_fraud + review_caught

            rows.append(
                {
                    "period": period_name,
                    "policy": policy_name,
                    "rejects_per_day": reject_n / days if days else 0.0,
                    "reject_precision": reject_fraud / reject_n if reject_n else 0.0,
                    "false_rejects_per_day": (reject_n - reject_fraud) / days
                    if days
                    else 0.0,
                    "reviews_per_day": len(queue) / days if days else 0.0,
                    "total_recall": total_caught / total_fraud
                    if total_fraud
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def weekly_stability(
    window: pd.DataFrame,
    window_days: int = 7,
    capacity: int | None = None,
) -> pd.DataFrame:
    """把验证期与测试期切成 7 天一段，检查策略效果是否随时间漂移。"""
    capacity = config.DAILY_REVIEW_CAPACITY if capacity is None else capacity
    start_day = int(window["day"].min())
    data = window.copy()
    data["segment"] = (data["day"] - start_day) // window_days

    rows = []
    for segment, part in data.groupby("segment", sort=True):
        queue = top_queue(
            part,
            ["score", "TransactionAmt"],
            [False, False],
            capacity,
        )
        metrics = summarize_queue(part, queue)
        rows.append(
            {
                "segment": int(segment) + 1,
                "start_day": int(part["day"].min()),
                "end_day": int(part["day"].max()),
                "days": int(part["day"].nunique()),
                "reviews": metrics["reviews"],
                "fraud_caught": metrics["fraud_caught"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )
    return pd.DataFrame(rows)
