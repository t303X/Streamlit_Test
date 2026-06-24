"""
数据采集模块。
从 Kaggle 下载链家北京二手房成交记录数据集 (ruiqurm/lianjia)。
该数据集为 Kaggle 公开数据，包含 2002-2018 年北京约 31.8 万条成交记录。
"""

import os
import logging

import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Kaggle 数据集标识
DATASET_REF = "ruiqurm/lianjia"
RAW_FILENAME = "ershoufang_raw.csv"


def get_raw_dir():
    """返回 data/raw/ 目录的绝对路径（项目根目录下的 data/raw/）。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(current_dir), "data", "raw")


def download_dataset() -> str:
    """
    从 Kaggle 下载链家二手房数据集，并复制到 data/raw/。

    Returns
    -------
    str
        本地 CSV 文件的绝对路径。
    """
    raw_dir = get_raw_dir()
    os.makedirs(raw_dir, exist_ok=True)
    dst_path = os.path.join(raw_dir, RAW_FILENAME)

    # 如果已存在则跳过下载
    if os.path.exists(dst_path):
        logger.info(f"数据文件已存在: {dst_path}")
        return dst_path

    logger.info(f"正在从 Kaggle 下载数据集 {DATASET_REF}...")
    kaggle_path = kagglehub.dataset_download(DATASET_REF)
    src_file = os.path.join(kaggle_path, "new.csv")

    # 复制到项目 data/raw/
    import shutil
    shutil.copy2(src_file, dst_path)
    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    logger.info(f"数据集已保存: {dst_path} ({size_mb:.1f} MB)")
    return dst_path


def generate_overview(csv_path: str):
    """
    输出数据概览：行列数、字段、缺失情况、基本统计。

    Parameters
    ----------
    csv_path : str
        房源 CSV 文件路径。
    """
    import pandas as pd

    print("\n" + "=" * 65)
    print("  北京市二手房成交数据 — 概览")
    print("=" * 65)

    df = pd.read_csv(csv_path, encoding="gbk", low_memory=False)
    print(f"  总记录数:   {len(df):,} 条")
    print(f"  总字段数:   {len(df.columns)} 个")

    # 从 URL 提取城市
    if "url" in df.columns:
        df["_city"] = df["url"].str.extract(r"//([a-z]+)\.")
        cities = df["_city"].value_counts()
        print(f"  城市分布:   {dict(cities)}")

    # 时间范围
    if "tradeTime" in df.columns:
        df["tradeTime"] = pd.to_datetime(df["tradeTime"])
        print(f"  时间跨度:   {df['tradeTime'].min().date()} ~ {df['tradeTime'].max().date()}")

    # 字段概览
    print(f"\n  {'字段名':<22s} {'类型':<10s} {'缺失数':>8s}  {'缺失率':>7s}  说明")
    print(f"  {'-'*60}")
    field_descriptions = {
        "url": "链家成交URL",
        "id": "房源ID",
        "Lng": "经度",
        "Lat": "纬度",
        "Cid": "小区ID",
        "tradeTime": "成交时间",
        "DOM": "挂牌天数",
        "followers": "关注人数",
        "totalPrice": "总价(万元)",
        "price": "单价(元/㎡)",
        "square": "面积(㎡)",
        "livingRoom": "卧室数",
        "drawingRoom": "客厅数",
        "kitchen": "厨房数",
        "bathRoom": "卫生间数",
        "floor": "楼层信息",
        "buildingType": "建筑类型",
        "constructionTime": "建筑年代",
        "renovationCondition": "装修状况",
        "buildingStructure": "建筑结构",
        "ladderRatio": "梯户比",
        "elevator": "有无电梯",
        "fiveYearsProperty": "是否满五年",
        "subway": "是否近地铁",
        "district": "行政区编码",
        "communityAverage": "小区均价(元/㎡)",
    }

    for col in df.columns:
        if col.startswith("_"):
            continue
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()
        missing_rate = missing / len(df) * 100
        desc = field_descriptions.get(col, "")
        print(f"  {col:<22s} {dtype:<10s} {missing:>8,d}  {missing_rate:>6.2f}%  {desc}")

    # 价格统计
    print(f"\n  核心数值统计:")
    for col, label in [("totalPrice", "总价(万元)"), ("price", "单价(元/㎡)"), ("square", "面积(㎡)")]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                print(f"    {label}: 均值={s.mean():.1f}, "
                      f"中位数={s.median():.1f}, "
                      f"最小={s.min():.1f}, "
                      f"最大={s.max():.1f}, "
                      f"标准差={s.std():.1f}")

    print(f"\n  小区数: {df['Cid'].nunique():,}" if "Cid" in df.columns else "")
    print("=" * 65 + "\n")


def main():
    """数据采集主入口：从 Kaggle 下载数据集并输出概览。"""
    # 下载数据集
    csv_path = download_dataset()

    # 输出数据概览
    generate_overview(csv_path)


if __name__ == "__main__":
    main()
