"""
AI 自动分析模块。
使用 DeepSeek API 对筛选后的数据进行智能分析，生成自然语言洞察。
"""

import os
import logging
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件（从项目根目录）
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
load_dotenv(os.path.join(_project_root, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def get_client() -> Optional[OpenAI]:
    """
    创建 DeepSeek API 客户端。

    Returns
    -------
    OpenAI or None
        如果未配置 API Key 则返回 None。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    if not api_key or api_key == "sk-your-api-key-here":
        logger.warning("DeepSeek API Key 未配置，请在 .env 文件中设置 DEEPSEEK_API_KEY")
        return None

    return OpenAI(api_key=api_key, base_url=base_url)


def summarize_dataframe(df: pd.DataFrame) -> str:
    """
    生成 DataFrame 的统计摘要文本，供 AI 分析使用。

    Parameters
    ----------
    df : pd.DataFrame
        筛选后的数据。

    Returns
    -------
    str
        格式化的统计摘要。
    """
    lines = []
    lines.append(f"数据量: {len(df):,} 条记录")

    # 数值列统计
    for col, label in [
        ("totalPrice", "总价(万元)"),
        ("price", "单价(元/㎡)"),
        ("square", "面积(㎡)"),
        ("house_age", "房龄(年)"),
    ]:
        if col in df.columns:
            s = df[col].dropna()
            lines.append(
                f"{label}: 均值={s.mean():.1f}, 中位数={s.median():.1f}, "
                f"标准差={s.std():.1f}, 最小={s.min():.1f}, 最大={s.max():.1f}"
            )

    # 行政区分组
    if "district_name" in df.columns and "price" in df.columns:
        district_stats = (
            df.groupby("district_name")["price"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )
        lines.append("\n各行政区均价排名:")
        for idx, row in district_stats.iterrows():
            lines.append(f"  {idx}: 均价={row['mean']:.0f}元/㎡, 成交量={row['count']:,}")

    # 户型分布
    if "layout_label" in df.columns:
        top_layouts = df["layout_label"].value_counts().head(5)
        lines.append("\n热门户型 (Top 5):")
        for layout, count in top_layouts.items():
            lines.append(f"  {layout}: {count:,} 套 ({count/len(df)*100:.1f}%)")

    # 装修分布
    if "renovation_name" in df.columns:
        renv = df["renovation_name"].value_counts()
        lines.append("\n装修状况分布:")
        for name, count in renv.items():
            lines.append(f"  {name}: {count:,} 套 ({count/len(df)*100:.1f}%)")

    # 年份趋势
    if "trade_year" in df.columns and "price" in df.columns:
        yearly = df.groupby("trade_year")["price"].agg(["mean", "count"])
        yearly = yearly.sort_index()
        lines.append("\n年度趋势:")
        for year, row in yearly.iterrows():
            lines.append(f"  {int(year)}年: 均价={row['mean']:.0f}元/㎡, 成交量={row['count']:,}")

    return "\n".join(lines)


def generate_analysis(
    df: pd.DataFrame,
    analysis_type: str = "全面分析",
) -> str:
    """
    使用 DeepSeek API 对数据进行智能分析。

    Parameters
    ----------
    df : pd.DataFrame
        要分析的数据（筛选后）。
    analysis_type : str
        分析类型提示。

    Returns
    -------
    str
        AI 生成的分析文本（Markdown 格式）。
    """
    client = get_client()
    if client is None:
        return (
            "⚠️ **DeepSeek API Key 未配置**\n\n"
            "请按以下步骤配置：\n"
            "1. 复制 `.env.example` 为 `.env`\n"
            "2. 在 `.env` 中填入你的 `DEEPSEEK_API_KEY`\n"
            "3. 重启 Streamlit 应用\n\n"
            "当前显示为静态统计摘要。"
        )

    data_summary = summarize_dataframe(df)

    prompt = f"""你是一位资深的数据分析师，擅长解读中国房地产市场数据。
请基于以下北京二手房成交数据的统计摘要，写一份专业的分析报告。

分析类型：{analysis_type}

数据统计摘要：
```
{data_summary}
```

请用 Markdown 格式输出，包含以下部分（如数据不支持则跳过）：
1. **市场概况**：总体价格水平和交易活跃度的综合评价（3-5句）
2. **价格分布特征**：价格分化程度、高低价区域的差异
3. **区域差异分析**：各区之间的均价差距及可能原因
4. **户型偏好**：市场主流户型及其特征
5. **时间趋势**：如果有年份数据，分析价格随时间的变化趋势
6. **购房建议**：基于数据分析，给出2-3条实用的购房建议
7. **关键洞察**：用 bullet points 列出3-5个最重要的发现

注意：
- 语言简洁专业，避免废话
- 数字引用要与统计摘要一致
- 总字数控制在 400-600 字
- 使用适当的 Markdown 格式（标题、列表、加粗）"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的中国房地产市场数据分析师，擅长从数据中提炼有价值的洞察。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        result = response.choices[0].message.content
        logger.info("AI 分析生成成功")
        return result
    except Exception as e:
        logger.error(f"DeepSeek API 调用失败: {e}")
        return f"❌ AI 分析请求失败: {e}\n\n请检查 API Key 和网络连接后重试。"
