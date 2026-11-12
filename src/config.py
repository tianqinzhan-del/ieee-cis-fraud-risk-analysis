"""项目级配置：路径、数据列、时间切分点与评估参数。

所有模块统一从这里读取配置，避免魔法数字散落在各处。
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录默认是项目下的 data/；也可通过环境变量 FRAUD_DATA_DIR 指定。
DATA_DIR = Path(os.environ.get("FRAUD_DATA_DIR", BASE_DIR / "data"))
OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"

TRANSACTION_PATH = DATA_DIR / "train_transaction.csv"
IDENTITY_PATH = DATA_DIR / "train_identity.csv"

# TransactionDT 的单位是秒，除以该常量得到“第几天”。
SECONDS_PER_DAY = 86_400

# 时间切分：前 60% 建模、中间 20% 选规则、最后 20% 只在最终评估时使用。
# 训练期与验证期之间不重叠，避免用未来信息回头调参。
TRAIN_END_DAY = 109
VALID_END_DAY = 145

TRANSACTION_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "isFraud",
]

# 风险画像的最小样本量：样本不足的分组不进入结论，避免小样本噪声。
MIN_PROFILE_SIZE = 1_000

# 查表打分器的最小样本量。
# 三维组合更稀疏，门槛设为 50 以保留足够细的覆盖率；
# 二维是回退层，被大量样本共用，门槛提高到 100 保证回退值本身可靠。
MIN_RATE_TABLE_SIZE_PRIMARY = 50
MIN_RATE_TABLE_SIZE_FALLBACK = 100

# 每天可用于人工审核的名额。
DAILY_REVIEW_CAPACITY = 100

# 金额规则阈值敏感性分析使用的阈值。
AMOUNT_THRESHOLDS = (100, 200, 500, 1_000, 2_000)

# 风险画像维度：展示名 -> 分组字段。
PROFILE_DIMENSIONS = {
    "ProductCD": ["ProductCD"],
    "card4": ["card4"],
    "card6": ["card6"],
    "ProductCD_x_card6": ["ProductCD", "card6"],
}
