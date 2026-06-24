# 北京市二手房成交价格多维度分析与可视化

数据可视化课程大作业。

## 运行环境

- Python 3.10+
- Windows / macOS / Linux

## 快速启动

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行数据采集（下载 Kaggle 数据集）
python src/data_collection.py

# 5. 运行数据预处理
python src/preprocessing.py

# 6. 运行分析与建模（生成图表至 figures/）
python src/analysis.py

# 7. 启动 Streamlit 应用
streamlit run app/streamlit_app.py

# 8. 访问 http://localhost:8501
```

## AI 分析配置（可选）

```bash
# 1. 复制配置文件
cp .env.example .env

# 2. 编辑 .env 填入你的 DeepSeek API Key
DEEPSEEK_API_KEY=sk-xxx

# 3. 重启 Streamlit，进入 Tab 4 使用 AI 分析功能
```

## 题目

北京市二手房成交价格多维度分析与可视化

## 数据来源

Kaggle 公开数据集 `ruiqurm/lianjia`（链家北京二手房成交记录 2002-2018）
