# 数据目录

本项目使用 [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
数据集。从 Kaggle 下载后，把以下两个文件放入本目录：

- `train_transaction.csv`
- `train_identity.csv`

也可以通过环境变量指向其他数据目录：

```bash
# Linux / macOS
export FRAUD_DATA_DIR=/path/to/data
python run_analysis.py
```

```powershell
# Windows PowerShell
$env:FRAUD_DATA_DIR = "D:\data"
python run_analysis.py
```

数据文件体积较大，不纳入版本控制。
