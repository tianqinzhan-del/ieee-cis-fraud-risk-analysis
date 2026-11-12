"""数据概览与风险画像。

风险画像的核心问题：哪个群体的欺诈率显著高于整体水平？
判断一个信号是否值得做成规则，需要同时看四个数：
    交易笔数、欺诈率、提升倍数、欺诈占比。
只看欺诈率会被小样本误导，只看欺诈占比会漏掉小而准的群体。
"""

from __future__ import annotations

import pandas as pd

from . import config


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """输出数据规模、标签分布与时间跨度等基础指标。"""
    total = len(df)
    frauds = int(df["isFraud"].sum())
    rows = [
        {"metric": "transactions", "value": total},
        {"metric": "columns_loaded", "value": df.shape[1] - 1},  # 不含派生列 day
        {"metric": "frauds", "value": frauds},
        {"metric": "fraud_rate", "value": frauds / total},
        {"metric": "first_day", "value": int(df["day"].min())},
        {"metric": "last_day", "value": int(df["day"].max())},
        {"metric": "covered_days", "value": int(df["day"].nunique())},
    ]
    frame = pd.DataFrame(rows)
    # 指标列同时包含计数与比率，用 object 保留各自类型，
    # 避免 pandas 把整数统一提升为浮点（Notebook 中会显示成 590540.0）。
    frame["value"] = frame["value"].astype(object)
    return frame


def missing_rate(df: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """按缺失率从高到低汇总字段缺失情况。"""
    rate = df.isna().mean().sort_values(ascending=False)
    if top is not None:
        rate = rate.head(top)
    rate = rate.rename("missing_rate")
    rate.index.name = "column"
    return rate.reset_index()


def risk_profile(
    df: pd.DataFrame,
    dimensions: list[str],
    min_count: int | None = None,
    top: int | None = None,
) -> pd.DataFrame:
    """按一个或多个维度分组，计算风险画像指标。

    参数：
        dimensions  分组字段，例如 ["ProductCD", "card6"]
        min_count   最小样本量，低于该值的分组会被剔除
        top         仅保留欺诈率最高的前 N 组
    """
    min_count = config.MIN_PROFILE_SIZE if min_count is None else min_count

    base_rate = float(df["isFraud"].mean())
    total_fraud = int(df["isFraud"].sum())

    table = (
        df.groupby(dimensions, dropna=False)
        .agg(
            transactions=("isFraud", "size"),
            frauds=("isFraud", "sum"),
        )
        .assign(
            fraud_rate=lambda x: x["frauds"] / x["transactions"],
        )
    )
    table["lift"] = table["fraud_rate"] / base_rate
    table["fraud_share"] = table["frauds"] / total_fraud
    table["daily_transactions"] = table["transactions"] / df["day"].nunique()

    table = table[table["transactions"] >= min_count]
    table = table.sort_values("fraud_rate", ascending=False)
    if top is not None:
        table = table.head(top)
    return table


def profile_all_dimensions(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """对配置中的全部维度批量生成风险画像，并补充邮箱域名维度。"""
    profiles = {}
    for name, dimensions in config.PROFILE_DIMENSIONS.items():
        profiles[name] = risk_profile(df, dimensions)

    profiles["P_emaildomain"] = risk_profile(
        df,
        ["P_emaildomain"],
        min_count=config.MIN_PROFILE_SIZE,
        top=12,
    )
    return profiles
