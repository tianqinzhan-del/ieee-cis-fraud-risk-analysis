"""数据读取、时间字段构造与交易表/身份表连接。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_transactions(
    columns: list[str] | None = None,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """读取交易表。

    默认只加载分析所需的列：原始表有 400 多列，全量读入既慢又占内存。
    """
    usecols = list(columns) if columns is not None else list(config.TRANSACTION_COLUMNS)
    return pd.read_csv(path or config.TRANSACTION_PATH, usecols=usecols)


def load_identity(path: str | Path | None = None) -> pd.DataFrame:
    """读取身份表。该表只有 41 列、约 14 万行，可以整表载入。"""
    return pd.read_csv(path or config.IDENTITY_PATH)


def add_day(df: pd.DataFrame, dt_col: str = "TransactionDT") -> pd.DataFrame:
    """新增 day 列：把秒级时间戳换算成“数据开始后的第几天”。"""
    out = df.copy()
    out["day"] = (out[dt_col] // config.SECONDS_PER_DAY).astype("int16")
    return out


def split_by_time(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按天把数据切成训练期 / 验证期 / 测试期。

    使用时间切分而非随机切分，是为了模拟真实上线场景：
    规则只能基于过去的数据制定，然后在未来的数据上检验。
    """
    train = df[df["day"] <= config.TRAIN_END_DAY].copy()
    valid = df[
        (df["day"] > config.TRAIN_END_DAY) & (df["day"] <= config.VALID_END_DAY)
    ].copy()
    test = df[df["day"] > config.VALID_END_DAY].copy()
    return {"train": train, "valid": valid, "test": test}


def describe_periods(periods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """汇总各时间段的规模、欺诈数与欺诈率。"""
    rows = []
    for name, part in periods.items():
        rows.append(
            {
                "period": name,
                "days": int(part["day"].nunique()),
                "transactions": len(part),
                "frauds": int(part["isFraud"].sum()),
                "fraud_rate": float(part["isFraud"].mean()),
            }
        )
    return pd.DataFrame(rows)


def merge_identity(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把身份表左连接到交易表，并输出连接质量摘要。

    返回：
        merged   连接后的明细表
        summary  连接前后的规模、键唯一性、覆盖率与行数校验
    """
    rows = [
        {"check": "transaction_rows", "value": len(transactions)},
        {"check": "identity_rows", "value": len(identity)},
        {
            "check": "transaction_id_unique",
            "value": bool(transactions["TransactionID"].is_unique),
        },
        {
            "check": "identity_id_unique",
            "value": bool(identity["TransactionID"].is_unique),
        },
    ]

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    covered = int((merged["_merge"] == "both").sum())
    rows.extend(
        [
            {"check": "merged_rows", "value": len(merged)},
            {"check": "row_count_preserved", "value": len(merged) == len(transactions)},
            {"check": "identity_covered", "value": covered},
            {"check": "identity_coverage_rate", "value": covered / len(merged)},
        ]
    )
    return merged, pd.DataFrame(rows)


def identity_join_profile(merged: pd.DataFrame) -> pd.DataFrame:
    """比较“有身份信息”与“无身份信息”两组交易的欺诈率。"""
    table = (
        # _merge 是 categorical，显式 observed=True 可避免 pandas 2.x
        # 额外输出一个空无一行的 right_only 分组。
        merged.groupby("_merge", observed=True)
        .agg(
            transactions=("isFraud", "size"),
            frauds=("isFraud", "sum"),
        )
        .assign(fraud_rate=lambda x: x["frauds"] / x["transactions"])
    )
    return table


def identity_missing_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """统计 id_* 字段连接后的缺失情况。"""
    id_cols = [c for c in merged.columns if c.startswith("id_")]
    missing = merged[id_cols].isna().mean()
    frame = pd.DataFrame(
        [
            {"metric": "id_columns", "value": len(id_cols)},
            {"metric": "id_01_missing_rate", "value": float(missing["id_01"])},
            {"metric": "median_missing_rate", "value": float(missing.median())},
            {"metric": "max_missing_rate", "value": float(missing.max())},
        ]
    )
    frame["value"] = frame["value"].astype(object)
    return frame
