"""规则评估。

一条风控规则上线前必须回答三个问题：
    1. 每天要送审多少笔？（成本）
    2. 能抓到多少欺诈？（召回）
    3. 送审的交易里有多少真的是欺诈？（命中率）
本模块把这三个问题统一成一张可比较的指标表。
"""

from __future__ import annotations

import pandas as pd

from . import config

RULE_LABELS = {
    "amount_threshold": "金额 > 阈值",
    "amount_gt_500": "金额 > 500",
    "product_c": "产品 C",
    "credit_card": "信用卡",
    "product_c_and_credit": "产品 C 且信用卡",
    "product_c_and_credit_and_amount_gt_100": "产品 C 且信用卡且金额 > 100",
}


def build_candidate_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    """构造待评估的候选规则。

    每个规则是一个与 df 等长的布尔序列：True 表示命中、需要送审。
    """
    return {
        "amount_gt_500": df["TransactionAmt"] > 500,
        "product_c": df["ProductCD"] == "C",
        "credit_card": df["card6"] == "credit",
        "product_c_and_credit": (df["ProductCD"] == "C")
        & (df["card6"] == "credit"),
        "product_c_and_credit_and_amount_gt_100": (df["ProductCD"] == "C")
        & (df["card6"] == "credit")
        & (df["TransactionAmt"] > 100),
    }


def evaluate_rule(
    df: pd.DataFrame,
    hit: pd.Series,
    rule_name: str = "",
    threshold: float | None = None,
) -> dict:
    """计算单条规则的成本与效果指标。"""
    total_transactions = len(df)
    total_fraud = int(df["isFraud"].sum())
    days = df["day"].nunique()
    base_rate = float(df["isFraud"].mean())

    is_fraud = df["isFraud"] == 1
    review_count = int(hit.sum())
    fraud_caught = int((hit & is_fraud).sum())

    return {
        "rule": rule_name,
        "threshold": threshold,
        "review_count": review_count,
        "review_share": review_count / total_transactions,
        "daily_review": review_count / days,
        "fraud_caught": fraud_caught,
        "recall": fraud_caught / total_fraud if total_fraud else 0.0,
        "false_alarms": review_count - fraud_caught,
        "precision": fraud_caught / review_count if review_count else 0.0,
        "lift": (fraud_caught / review_count) / base_rate if review_count else 0.0,
    }


def amount_threshold_table(
    df: pd.DataFrame,
    thresholds: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """金额阈值敏感性分析：同一规则只改阈值，观察成本与收益如何变化。"""
    thresholds = thresholds or config.AMOUNT_THRESHOLDS
    rows = [
        evaluate_rule(
            df,
            df["TransactionAmt"] > threshold,
            rule_name=RULE_LABELS["amount_threshold"],
            threshold=threshold,
        )
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def rule_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """横向对比所有候选规则。"""
    rows = [
        evaluate_rule(df, hit, rule_name=name)
        for name, hit in build_candidate_rules(df).items()
    ]
    table = pd.DataFrame(rows)
    table["rule"] = table["rule"].map(RULE_LABELS).fillna(table["rule"])
    return table.sort_values("precision", ascending=False)
