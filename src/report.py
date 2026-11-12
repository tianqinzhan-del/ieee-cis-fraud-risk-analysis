"""结果落盘与 Markdown 报告生成。

表格数据以英文列名写入 CSV，便于下游复用；
展示到 report.md 时统一翻译成中文列名并做百分比/倍数格式化。
"""

from __future__ import annotations

import math
import numbers
from pathlib import Path

import pandas as pd

from . import config

COLUMN_LABELS = {
    "column": "字段",
    "metric": "指标",
    "value": "数值",
    "missing_rate": "缺失率",
    "period": "期间",
    "strategy": "排序方式",
    "capacity": "每日名额",
    "reviews": "送审笔数",
    "fraud_caught": "抓到欺诈",
    "precision": "命中率",
    "recall": "召回率",
    "transactions": "交易笔数",
    "frauds": "欺诈笔数",
    "fraud_rate": "欺诈率",
    "lift": "提升倍数",
    "fraud_share": "欺诈占比",
    "daily_transactions": "日均交易",
    "daily_review": "日均送审",
    "review_count": "送审笔数",
    "review_share": "送审占比",
    "false_alarms": "误伤笔数",
    "rule": "规则",
    "threshold": "阈值",
    "check": "检查项",
    "segment": "分组",
    "start_day": "起始天",
    "end_day": "结束天",
    "days": "天数",
    "tier": "分数档",
    "policy": "方案",
    "rejects_per_day": "日均拒绝",
    "reject_precision": "拒绝命中率",
    "false_rejects_per_day": "日均误拒",
    "reviews_per_day": "日均送审",
    "total_recall": "总召回率",
    "_merge": "连接结果",
}

PERIOD_LABELS = {"train": "训练期", "valid": "验证期", "test": "测试期"}
SORT_STRATEGY_LABELS = {"amount": "金额", "score": "风险分"}
MERGE_LABELS = {"left_only": "无身份信息", "both": "有身份信息"}


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _format_cell(value) -> str:
    if _is_missing(value):
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, numbers.Integral):
        return f"{int(value):,}"
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isnan(number):
            return "-"
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:.4f}"
    return str(value)


def render_table(
    df: pd.DataFrame,
    percent_columns: tuple[str, ...] = (),
    ratio_columns: tuple[str, ...] = (),
    int_columns: tuple[str, ...] = (),
    decimal_columns: tuple[str, ...] = (),
    column_labels: dict[str, str] | None = None,
) -> str:
    """把 DataFrame 渲染成 Markdown 表格。"""
    labels = COLUMN_LABELS if column_labels is None else column_labels

    table = df.copy()
    # 分组键可能是单层索引，也可能是多级索引，都要还原成普通列。
    if isinstance(table.index, pd.MultiIndex) or table.index.name is not None:
        table = table.reset_index()
    table.columns = [str(c) for c in table.columns]

    headers = [labels.get(c, c) for c in table.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    # itertuples 保留每列原始类型，避免整数被统一转成浮点。
    for row in table.itertuples(index=False):
        cells = []
        for column, value in zip(table.columns, row):
            if _is_missing(value):
                cells.append("-")
            elif column in percent_columns:
                cells.append(f"{float(value):.2%}")
            elif column in ratio_columns:
                cells.append(f"{float(value):.2f}x")
            elif column in int_columns:
                cells.append(f"{int(value):,}")
            elif column in decimal_columns:
                cells.append(f"{float(value):.1f}")
            else:
                cells.append(_format_cell(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, filename: str) -> Path:
    """把结果表写入 outputs/tables/，保留原始数值类型。

    只有带业务含义的索引（分组键）才写入 CSV，普通行号不写入。
    """
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TABLE_DIR / filename
    # MultiIndex（多维分组）的 .name 为 None，但 .names 有值，必须一并判断。
    has_business_index = (
        isinstance(df.index, pd.MultiIndex) or df.index.name is not None
    )
    df.to_csv(path, index=has_business_index)
    return path


def build_report(title: str, sections: list[tuple[str, str]]) -> str:
    """拼接 Markdown 报告。sections 为 (小标题, 正文) 列表。"""
    parts = [f"# {title}", ""]
    for heading, body in sections:
        parts.extend([f"## {heading}", "", body.strip(), ""])
    return "\n".join(parts)


def write_report(text: str, path: Path | None = None) -> Path:
    path = path or (config.OUTPUT_DIR / "report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
