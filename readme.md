project_name/
├── data/                   # 数据存储
│   ├── raw/                # 原始数据
│   │   ├── train.csv
│   │   ├── test.csv
│   ├── processed/          # 处理后的数据
│       ├── train_processed.npy
│       ├── test_processed.npy
├── data_processing/        # 数据处理模块
│   ├── __init__.py
│   ├── data_loader.py      # 数据加载代码
│   ├── data_preprocessing.py  # 数据预处理代码
│   ├── feature_engineering.py  # 特征工程代码
├── models/                 # 模型定义和训练代码
│   ├── __init__.py
│   ├── model_definition.py  # 模型定义代码
│   ├── model_training.py   # 模型训练代码
│   ├── model_evaluation.py  # 模型评估代码
├── experiments/            # 实验配置和结果
│   ├── configs/            # 实验配置文件
│   │   ├── experiment_1.yaml
│   ├── results/            # 实验结果
│       ├── experiment_1/
│           ├── metrics.csv
│           ├── model.pth
├── utils/                  # 工具函数和辅助代码
│   ├── __init__.py
│   ├── logging_utils.py    # 日志记录工具
│   ├── visualization_utils.py  # 可视化工具
├── tests/                  # 测试代码
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_model_definition.py
├── main.py                 # 项目主程序
├── requirements.txt        # 项目依赖的 Python 包
├── README.md               # 项目说明文档