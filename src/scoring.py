"""基于历史欺诈率的查表打分器。

这是最朴素的评分模型：把历史欺诈率当作风险分。
它的价值不在于精度，而在于提供一个可解释、可上线的基线，
用来回答“在固定审核名额下，按风险排序是否优于按金额排序”。

关键约束：所有历史欺诈率只能在训练期上拟合，验证期与测试期只做打分，
不参与任何参数估计，否则会高估线上效果。
"""

from __future__ import annotations

import pandas as pd

from . import config


def fit_rate_table(
    data: pd.DataFrame,
    keys: list[str],
    min_count: int,
) -> pd.DataFrame:
    """在训练期上计算每个分组的欺诈率。

    样本量低于 min_count 的分组直接丢弃，后续会回退到更粗的维度。
    """
    table = (
        data.groupby(keys, dropna=False)
        .agg(
            transactions=("isFraud", "size"),
            frauds=("isFraud", "sum"),
        )
        .assign(rate=lambda x: x["frauds"] / x["transactions"])
    )
    table = table[table["transactions"] >= min_count]
    return table["rate"].reset_index()


class RateLookupScorer:
    """三级回退的查表打分器。

    优先级：
        1. primary  ：产品 + 卡类型 + 邮箱（最细，样本门槛 50）
        2. fallback ：产品 + 卡类型（较粗，样本门槛 100，覆盖更广）
        3. global   ：训练期整体欺诈率（兜底）
    """

    def __init__(
        self,
        primary_keys: list[str] | None = None,
        fallback_keys: list[str] | None = None,
        min_primary: int | None = None,
        min_fallback: int | None = None,
    ) -> None:
        self.primary_keys = primary_keys or ["ProductCD", "card6", "P_emaildomain"]
        self.fallback_keys = fallback_keys or ["ProductCD", "card6"]
        self.min_primary = (
            config.MIN_RATE_TABLE_SIZE_PRIMARY
            if min_primary is None
            else min_primary
        )
        self.min_fallback = (
            config.MIN_RATE_TABLE_SIZE_FALLBACK
            if min_fallback is None
            else min_fallback
        )
        self.primary_table: pd.DataFrame | None = None
        self.fallback_table: pd.DataFrame | None = None
        self.global_rate: float | None = None

    def fit(self, train: pd.DataFrame) -> "RateLookupScorer":
        """只用训练期数据拟合三张查找表。"""
        self.primary_table = fit_rate_table(
            train, self.primary_keys, self.min_primary
        )
        self.fallback_table = fit_rate_table(
            train, self.fallback_keys, self.min_fallback
        )
        self.global_rate = float(train["isFraud"].mean())
        return self

    def transform(self, part: pd.DataFrame) -> pd.DataFrame:
        """给任意时间段打风险分。"""
        if self.primary_table is None or self.global_rate is None:
            raise RuntimeError("请先调用 fit() 再调用 transform()")

        out = part.copy()
        out = out.merge(self.primary_table, on=self.primary_keys, how="left")
        out = out.rename(columns={"rate": "rate_primary"})
        out = out.merge(self.fallback_table, on=self.fallback_keys, how="left")
        out = out.rename(columns={"rate": "rate_fallback"})

        out["score"] = (
            out["rate_primary"]
            .fillna(out["rate_fallback"])
            .fillna(self.global_rate)
        )
        return out
