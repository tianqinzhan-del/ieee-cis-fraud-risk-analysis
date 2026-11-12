"""项目入口：运行完整的支付交易欺诈风险分析。

用法：
    python run_analysis.py

输出：
    outputs/report.md          汇总报告（Markdown）
    outputs/tables/*.csv       各分析环节的明细表
"""

from __future__ import annotations

import logging
import numbers

import pandas as pd

from src import config, data, evaluation, profiling, report, rules, scoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("fraud-analysis")

METRIC_LABELS = {
    "transactions": "交易笔数",
    "columns_loaded": "加载字段数",
    "frauds": "欺诈笔数",
    "fraud_rate": "整体欺诈率",
    "first_day": "起始天",
    "last_day": "结束天",
    "covered_days": "覆盖天数",
    "id_columns": "id_ 字段数",
    "id_01_missing_rate": "id_01 缺失率",
    "median_missing_rate": "id_ 缺失率中位数",
    "max_missing_rate": "id_ 缺失率最大值",
}

CHECK_LABELS = {
    "transaction_rows": "交易表行数",
    "identity_rows": "身份表行数",
    "transaction_id_unique": "交易表 ID 唯一",
    "identity_id_unique": "身份表 ID 唯一",
    "merged_rows": "连接后行数",
    "row_count_preserved": "连接前后行数一致",
    "identity_covered": "命中身份信息的交易数",
    "identity_coverage_rate": "身份信息覆盖率",
}

PERCENT_METRICS = {
    "fraud_rate",
    "id_01_missing_rate",
    "median_missing_rate",
    "max_missing_rate",
}
PERCENT_CHECKS = {"identity_coverage_rate"}
BOOL_CHECKS = {
    "transaction_id_unique",
    "identity_id_unique",
    "row_count_preserved",
}


def _format_scalar(value) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, numbers.Integral):
        return f"{int(value):,}"
    if isinstance(value, numbers.Real):
        number = float(value)
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:.4f}"
    return str(value)


def _format_metric_value(metric: str, value) -> str:
    if metric in PERCENT_METRICS:
        return f"{float(value):.2%}"
    return _format_scalar(value)


def render_metrics(table: pd.DataFrame, key_column: str, label_map: dict[str, str]) -> str:
    """把 (metric/check, value) 两列的指标表渲染成中文 Markdown。"""
    display = table.copy()
    key = display.columns[0]
    display[key] = display[key].map(label_map).fillna(display[key])
    display["value"] = [
        _format_metric_value(metric, value)
        if key == "metric"
        else _format_check_value(metric, value)
        for metric, value in zip(table[key], table["value"])
    ]
    return report.render_table(display)


def _format_check_value(check: str, value) -> str:
    if check in PERCENT_CHECKS:
        return f"{float(value):.2%}"
    if check in BOOL_CHECKS or isinstance(value, bool):
        return "是" if value else "否"
    return _format_scalar(value)


def with_display_labels(table: pd.DataFrame) -> pd.DataFrame:
    """把期间、排序方式、连接结果等内部枚举值翻译成中文。"""
    display = table.copy()
    mappings = {
        "period": report.PERIOD_LABELS,
        "strategy": report.SORT_STRATEGY_LABELS,
        "_merge": report.MERGE_LABELS,
    }
    for column, labels in mappings.items():
        if column in display.columns:
            display[column] = display[column].map(labels).fillna(display[column])
        if display.index.name == column:
            display.index = pd.Index(
                [labels.get(value, value) for value in display.index],
                name=column,
            )
    return display


STABILITY_LABELS = {
    f"{period}_{suffix}": f"{label}{suffix_cn}"
    for period, label in report.PERIOD_LABELS.items()
    for suffix, suffix_cn in (("transactions", "交易笔数"), ("rate", "欺诈率"))
}
REPORT_LABELS = {**report.COLUMN_LABELS, **STABILITY_LABELS}

PROFILE_FORMATS = {
    "percent_columns": ("fraud_rate", "fraud_share"),
    "ratio_columns": ("lift",),
    "int_columns": ("transactions", "frauds"),
    "decimal_columns": ("daily_transactions",),
}


def render_profile(table: pd.DataFrame) -> str:
    return report.render_table(table, **PROFILE_FORMATS)


def main() -> None:
    logger.info("[1/8] 读取交易表并构造时间字段")
    tx = data.load_transactions()
    tx = data.add_day(tx)
    logger.info("      交易 %s 笔，字段 %s 个，覆盖 %s 天",
                f"{len(tx):,}", tx.shape[1], tx["day"].nunique())

    # ---------- 1. 数据概览 ----------
    logger.info("[2/8] 数据概览与缺失率")
    overview = profiling.dataset_overview(tx)
    # 缺失率只统计原始字段；day 是派生列，恒不缺失，计入会稀释结论。
    missing = profiling.missing_rate(tx.drop(columns=["day"]))
    report.save_table(overview, "01_overview.csv")
    report.save_table(missing, "02_missing_rate.csv")

    # ---------- 2. 金额规则 ----------
    logger.info("[3/8] 金额阈值敏感性分析")
    amount_table = rules.amount_threshold_table(tx)
    report.save_table(amount_table, "03_amount_thresholds.csv")

    # ---------- 3. 风险画像 ----------
    logger.info("[4/8] 多维风险画像")
    profiles = profiling.profile_all_dimensions(tx)
    for name, table in profiles.items():
        report.save_table(table, f"04_profile_{name}.csv")

    # ---------- 4. identity 表连接 ----------
    logger.info("[5/8] 连接 identity 表并检查数据质量")
    identity = data.load_identity()
    merged, merge_summary = data.merge_identity(tx, identity)
    join_profile = data.identity_join_profile(merged)
    identity_missing = data.identity_missing_summary(merged)
    report.save_table(merge_summary, "05_identity_merge_summary.csv")
    report.save_table(join_profile, "05_identity_join_profile.csv")
    report.save_table(identity_missing, "05_identity_missing.csv")

    # ---------- 5. 时间切分 ----------
    logger.info("[6/8] 时间切分与群体稳定性检查")
    periods = data.split_by_time(tx)
    period_summary = data.describe_periods(periods)
    stability_by_segment = evaluation.segment_stability(periods)
    report.save_table(period_summary, "06_period_summary.csv")
    report.save_table(stability_by_segment, "06_segment_stability.csv")

    # ---------- 6. 打分与规则 ----------
    logger.info("[7/8] 拟合查表打分器并评估审核策略")
    scorer = scoring.RateLookupScorer().fit(periods["train"])
    scored_periods = {
        "valid": scorer.transform(periods["valid"]),
        "test": scorer.transform(periods["test"]),
    }

    rule_table = rules.rule_comparison_table(tx)
    report.save_table(rule_table, "07_rule_comparison.csv")

    capacity = evaluation.capacity_curve(scored_periods)
    tiers = evaluation.score_tier_table(scored_periods)
    policies = evaluation.policy_comparison(scored_periods)

    report.save_table(capacity, "08_capacity_curve.csv")
    report.save_table(tiers, "09_score_tiers.csv")
    report.save_table(policies, "10_policy_comparison.csv")

    review_window = pd.concat(
        [scored_periods["valid"], scored_periods["test"]], ignore_index=True
    )
    weekly = evaluation.weekly_stability(review_window)
    report.save_table(weekly, "11_weekly_stability.csv")

    # ---------- 7. 生成报告 ----------
    logger.info("[8/8] 生成 Markdown 报告")
    sections = build_sections(
        overview=overview,
        missing=missing,
        amount_table=amount_table,
        profiles=profiles,
        merge_summary=merge_summary,
        join_profile=join_profile,
        identity_missing=identity_missing,
        period_summary=period_summary,
        stability_by_segment=stability_by_segment,
        rule_table=rule_table,
        capacity=capacity,
        tiers=tiers,
        policies=policies,
        weekly=weekly,
        tx=tx,
    )
    text = report.build_report("IEEE-CIS 支付交易欺诈风险分析", sections)
    path = report.write_report(text)

    logger.info("完成。报告：%s", path)
    logger.info("明细表目录：%s", config.TABLE_DIR)


def build_sections(**context) -> list[tuple[str, str]]:
    """把各环节结果组织成报告章节。"""
    tx = context["tx"]
    overview = context["overview"]
    missing = context["missing"]
    amount_table = context["amount_table"]
    profiles = context["profiles"]
    merge_summary = context["merge_summary"]
    join_profile = context["join_profile"]
    identity_missing = context["identity_missing"]
    period_summary = context["period_summary"]
    stability_by_segment = context["stability_by_segment"]
    rule_table = context["rule_table"]
    capacity = context["capacity"]
    tiers = context["tiers"]
    policies = context["policies"]
    weekly = context["weekly"]

    base_rate = float(tx["isFraud"].mean())
    fraud_count = int(tx["isFraud"].sum())

    # 关键结论所需的取数
    amount_500 = amount_table.loc[amount_table["threshold"] == 500].iloc[0]
    product_card = profiles["ProductCD_x_card6"]
    product_c_credit = product_card.loc[("C", "credit")]

    test_capacity = capacity[(capacity["period"] == "test") & (capacity["capacity"] == 100)]
    score_100 = test_capacity[test_capacity["strategy"] == "score"].iloc[0]
    amount_100 = test_capacity[test_capacity["strategy"] == "amount"].iloc[0]

    policy_test = policies[policies["period"] == "test"]
    best_policy = policy_test.sort_values("total_recall", ascending=False).iloc[0]

    weekly_precision_range = (float(weekly["precision"].min()), float(weekly["precision"].max()))
    weekly_recall_range = (float(weekly["recall"].min()), float(weekly["recall"].max()))

    payload = [
        (
            "1. 项目背景",
            f"""
本数据集包含 **{len(tx):,}** 笔支付交易、**{fraud_count:,}** 笔欺诈交易，
整体欺诈率 **{base_rate:.2%}**。正负样本极不均衡，
因此后续评估不使用准确率，而使用命中率（precision）、召回率（recall）
和每日送审量这三个更贴近业务决策的指标。

分析目标：在每天固定人工审核名额的约束下，
比较不同排序与分档策略，给出可解释、可上线的规则配置建议。
""",
        ),
        (
            "2. 数据概览",
            render_metrics(overview, "metric", METRIC_LABELS)
            + "\n\n各字段缺失率：\n\n"
            + report.render_table(missing, percent_columns=("missing_rate",)),
        ),
        (
            "3. 金额规则敏感性",
            f"""
以“交易金额 > 阈值”作为唯一规则，观察不同阈值下的成本与收益。
阈值 500 时送审占比 {amount_500['review_share']:.2%}，
召回率 {amount_500['recall']:.2%}，命中率 {amount_500['precision']:.2%}。

{report.render_table(
    amount_table,
    percent_columns=("review_share", "recall", "precision"),
    ratio_columns=("lift",),
    int_columns=("review_count", "fraud_caught", "false_alarms"),
    decimal_columns=("daily_review",),
)}
""",
        ),
        (
            "4. 风险画像",
            f"""
风险画像同时看四个指标：交易笔数、欺诈率、提升倍数、欺诈占比。
提升倍数 = 该群体欺诈率 / 整体欺诈率（{base_rate:.2%}）。
样本量低于 {config.MIN_PROFILE_SIZE:,} 的分组不纳入结论。

**产品类型 ProductCD**

{render_profile(profiles['ProductCD'])}

**卡组织 card4**

{render_profile(profiles['card4'])}

**卡类型 card6**

{render_profile(profiles['card6'])}

**产品类型 × 卡类型**

{render_profile(profiles['ProductCD_x_card6'])}

其中“产品 C + 信用卡”群体欺诈率 **{product_c_credit['fraud_rate']:.2%}**，
是整体水平的 **{product_c_credit['lift']:.2f} 倍**，
贡献了全部欺诈交易的 **{product_c_credit['fraud_share']:.2%}**。

**付款邮箱域名（欺诈率最高的前 12 个）**

{render_profile(profiles['P_emaildomain'])}
""",
        ),
        (
            "5. 交易表与 identity 表连接",
            "连接质量检查：\n\n"
            + render_metrics(merge_summary, "check", CHECK_LABELS)
            + "\n\n有无身份信息的交易在欺诈率上的差异：\n\n"
            + report.render_table(
                with_display_labels(join_profile),
                percent_columns=("fraud_rate",),
                int_columns=("transactions", "frauds"),
            )
            + "\n\nid_* 字段缺失情况：\n\n"
            + render_metrics(identity_missing, "metric", METRIC_LABELS),
        ),
        (
            "6. 时间切分与群体稳定性",
            f"""
按天切分：训练期第 1–{config.TRAIN_END_DAY} 天，
验证期第 {config.TRAIN_END_DAY + 1}–{config.VALID_END_DAY} 天，
测试期第 {config.VALID_END_DAY + 1} 天之后。
测试期只在最终评估时使用，不参与任何参数选择。

{report.render_table(
    with_display_labels(period_summary),
    percent_columns=("fraud_rate",),
    int_columns=("days", "transactions", "frauds"),
)}

关键风险群体在三段时间的欺诈率与样本量：

{report.render_table(
    stability_by_segment,
    percent_columns=tuple(
        c for c in stability_by_segment.columns if c.endswith("_rate")
    ),
    int_columns=tuple(
        c for c in stability_by_segment.columns if c.endswith("_transactions")
    ),
    column_labels=REPORT_LABELS,
)}
""",
        ),
        (
            "7. 规则横向对比",
            f"""
规则在全量数据上的表现（日均送审按 {tx['day'].nunique()} 天折算）：

{report.render_table(
    rule_table,
    percent_columns=("review_share", "recall", "precision"),
    ratio_columns=("lift",),
    int_columns=("review_count", "fraud_caught", "false_alarms"),
    decimal_columns=("daily_review",),
)}
""",
        ),
        (
            "8. 审核容量曲线",
            f"""
查表打分器使用训练期数据拟合，按“产品 + 卡类型 + 邮箱 → 产品 + 卡类型 → 整体欺诈率”
三级回退。评估时每天只取分数最高的 N 笔送审。

{report.render_table(
    with_display_labels(capacity),
    percent_columns=("precision", "recall"),
    int_columns=("reviews", "fraud_caught"),
)}

测试期、每天 100 个名额时：
按风险分排序命中率 **{score_100['precision']:.2%}**、召回率 **{score_100['recall']:.2%}**；
按金额排序命中率 **{amount_100['precision']:.2%}**、召回率 **{amount_100['recall']:.2%}**。
""",
        ),
        (
            "9. 固定分数分档",
            report.render_table(
                with_display_labels(tiers),
                percent_columns=("fraud_rate", "fraud_share"),
                int_columns=("transactions", "frauds"),
                decimal_columns=("daily_transactions",),
            ),
        ),
        (
            "10. 三档处置方案",
            f"""
三档方案比较：高风险自动拒绝、次高风险进入人工审核池、其余放行。
审核池仍然每天只取前 {config.DAILY_REVIEW_CAPACITY} 笔。

{report.render_table(
    with_display_labels(policies),
    percent_columns=("reject_precision", "total_recall"),
    decimal_columns=("rejects_per_day", "false_rejects_per_day", "reviews_per_day"),
)}

测试期总召回率最高的方案为 **{best_policy['policy']}**，
总召回率 **{best_policy['total_recall']:.2%}**，
日均拒绝 **{best_policy['rejects_per_day']:.2f}** 笔，
日均误拒 **{best_policy['false_rejects_per_day']:.2f}** 笔。
""",
        ),
        (
            "11. 七天稳定性",
            f"""
把验证期与测试期按 7 天分段，检查同一策略的效果是否随时间漂移。
命中率区间 **{weekly_precision_range[0]:.2%}–{weekly_precision_range[1]:.2%}**，
召回率区间 **{weekly_recall_range[0]:.2%}–{weekly_recall_range[1]:.2%}**。

{report.render_table(weekly, percent_columns=("precision", "recall"))}
""",
        ),
        (
            "12. 结论",
            f"""
1. **数据高度不平衡**：整体欺诈率仅 {base_rate:.2%}，
   因此业务上更关心“每天有限名额能抓到多少欺诈”，而不是整体准确率。
2. **单条金额规则的有效性有限**：阈值 500 时命中率 {amount_500['precision']:.2%}，
   仅略高于整体欺诈率 {base_rate:.2%}，召回率 {amount_500['recall']:.2%}，
   说明金额本身不足以区分欺诈，需要与群体特征组合。
3. **“产品 C + 信用卡”是最显著的风险群体**：
   欺诈率 {product_c_credit['fraud_rate']:.2%}，提升 {product_c_credit['lift']:.2f} 倍，
   贡献 {product_c_credit['fraud_share']:.2%} 的欺诈交易。
4. **按风险分排序优于按金额排序**：
   在测试期每天 100 个名额下，召回率由 {amount_100['recall']:.2%}
   提升到 {score_100['recall']:.2%}。
5. **策略效果在观察窗口内波动但未持续恶化**：
   7 天分段的命中率在 {weekly_precision_range[0]:.2%}–{weekly_precision_range[1]:.2%} 之间，
   召回率在 {weekly_recall_range[0]:.2%}–{weekly_recall_range[1]:.2%} 之间。
   可作为规则上线前的基线方案，后续再用监督模型替换查表打分。
""",
        ),
    ]
    return payload


if __name__ == "__main__":
    main()
