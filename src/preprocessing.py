"""
数据预处理模块。
负责清洗链家北京二手房成交数据：缺失值处理、异常值剔除、
编码字段映射、类型统一、特征工程，最终输出到 data/processed/。
"""

import os
import re
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 随机种子，保证可复现
RANDOM_STATE = 42

# ---- 行政区编码映射（基于各行政区平均经纬度推断） ----
# 重要：原始映射中 code 2 与 code 10 被互换，导致最高价区域被错误标注为"丰台区"。
# 经核对各编码对应数据的均价排序，code 2 实为丰台区（南城，均价较低），
# code 10 实为西城区（金融街/北京核心，均价全市最高）。下方已修正。
DISTRICT_MAP = {
    1: "东城区",
    2: "丰台区",       # 修正：原误标为西城区
    3: "亦庄开发区",
    4: "大兴区",
    5: "房山区",
    6: "昌平区",
    7: "朝阳区",
    8: "海淀区",
    9: "石景山区",
    10: "西城区",      # 修正：原误标为丰台区
    11: "通州区",
    12: "门头沟区",
    13: "顺义区",
}


def verify_district_mapping(df):
    """
    打印各行政区编码的经纬度中心，便于人工核对映射是否正确。
    应在 map_encoded_fields 之后调用。
    """
    print("\n  行政区编码 - 经纬度中心 校验（与北京地图对照）:")
    print(f"  {'编码':<4} {'名称':<10} {'平均经度':>9} {'平均纬度':>9} {'记录数':>10}")
    for code, name in sorted(DISTRICT_MAP.items()):
        sub = df[df["district"] == code]
        if len(sub) == 0:
            continue
        lng = sub["Lng"].mean()
        lat = sub["Lat"].mean()
        print(f"  {code:<4} {name:<10} {lng:>9.4f} {lat:>9.4f} {len(sub):>10,d}")
    print("  （西城区中心约 116.37, 39.91；丰台区中心约 116.29, 39.85）\n")

# 装修状况映射
RENOVATION_MAP = {
    0: "其他",
    1: "毛坯",
    2: "简装修",
    3: "精装修",
    4: "豪华装修",
}

# 建筑类型映射
BUILDING_TYPE_MAP = {
    1.0: "板楼",
    2.0: "塔楼",
    3.0: "板塔结合",
    4.0: "平房",
}

# 建筑结构映射
BUILDING_STRUCTURE_MAP = {
    0: "未知",
    1: "砖混结构",
    2: "钢混结构",
    3: "砖木结构",
    4: "钢结构",
    5: "框架结构",
    6: "混合结构",
}


def get_raw_path():
    """返回原始数据文件路径。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(current_dir), "data", "raw", "ershoufang_raw.csv")


def get_processed_path():
    """返回处理后数据文件路径。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(current_dir), "data", "processed", "ershoufang_clean.csv")


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    加载原始 CSV 数据，处理编码问题。

    Parameters
    ----------
    filepath : str
        原始 CSV 文件路径。

    Returns
    -------
    pd.DataFrame
    """
    logger.info("加载原始数据...")
    df = pd.read_csv(filepath, encoding="gbk", low_memory=False)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")
    return df


def fix_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复混合类型列，统一转换为数值型。

    - livingRoom: 剔除 '#NAME?' 等异常值，转 int
    - drawingRoom: 提取数值部分，转 int
    - bathRoom: 剔除年份等异常值，转 int
    - floor: 从乱码文本中提取含有的数值信息

    Parameters
    ----------
    df : pd.DataFrame
        原始数据。

    Returns
    -------
    pd.DataFrame
    """
    logger.info("修复列类型...")

    # livingRoom: 强制转数值，异常值变为 NaN
    df["livingRoom"] = pd.to_numeric(df["livingRoom"], errors="coerce")
    before = df["livingRoom"].isna().sum()

    # drawingRoom: 提取数值部分（有些值如 "客厅 6" 含中文）
    df["drawingRoom"] = df["drawingRoom"].astype(str).str.extract(r"(\d+)", expand=False)
    df["drawingRoom"] = pd.to_numeric(df["drawingRoom"], errors="coerce")

    # bathRoom: 剔除年份值（>1000 的是年份误填），转数值
    df["bathRoom"] = pd.to_numeric(df["bathRoom"], errors="coerce")
    df.loc[df["bathRoom"] > 1000, "bathRoom"] = np.nan
    df.loc[df["bathRoom"] > 10, "bathRoom"] = np.nan  # 超过10个卫生间不合理

    logger.info(f"  livingRoom 异常值: {before} 行")
    return df


def fix_floor_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理 floor 字段。
    floor 列中文部分因编码问题乱码，但末尾数字代表总楼层。
    提取数字部分作为 total_floors 特征。

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("处理 floor 列...")

    # 提取 floor 中的数字作为总楼层参考
    df["total_floors"] = df["floor"].astype(str).str.extract(r"(\d+)", expand=False)
    df["total_floors"] = pd.to_numeric(df["total_floors"], errors="coerce")

    # 过滤不合理值
    df.loc[df["total_floors"] > 80, "total_floors"] = np.nan
    df.loc[df["total_floors"] < 1, "total_floors"] = np.nan

    valid = df["total_floors"].notna().sum()
    logger.info(f"  提取到总楼层: {valid} 条 ({valid/len(df)*100:.1f}%)")
    return df


def map_encoded_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    将数字编码字段映射为中文标签。

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("映射编码字段...")

    # 行政区
    df["district_name"] = df["district"].map(DISTRICT_MAP)
    unmapped = df["district_name"].isna().sum()
    if unmapped > 0:
        logger.warning(f"  行政区未映射: {unmapped} 条")

    # 装修状况
    df["renovation_name"] = df["renovationCondition"].map(RENOVATION_MAP)

    # 建筑类型
    df["building_type_name"] = df["buildingType"].map(BUILDING_TYPE_MAP)
    # 对于非标准值（如 0.5, 0.333 等），标记为"其他"
    df["building_type_name"] = df["building_type_name"].fillna("其他")

    # 建筑结构
    df["structure_name"] = df["buildingStructure"].map(BUILDING_STRUCTURE_MAP)
    df["structure_name"] = df["structure_name"].fillna("未知")

    logger.info("  编码字段映射完成")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理异常值和极端值。

    基于房地产实务经验为核心数值字段设定合理边界，
    超出边界的记录视为数据质量问题，予以直接剔除。

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("处理异常值...")
    outliers_removed = 0

    # 总价异常：保留 10万 ~ 3000万 之间的记录
    # 0.1万、1万等明显是数据错误
    mask_before = len(df)
    df = df[(df["totalPrice"] >= 10) & (df["totalPrice"] <= 3000)]
    outliers_removed += mask_before - len(df)

    # 单价异常：保留 5000 ~ 150000 元/㎡ 之间的记录
    mask_before = len(df)
    df = df[(df["price"] >= 5000) & (df["price"] <= 150000)]
    outliers_removed += mask_before - len(df)

    # 面积异常：保留 15 ~ 500 ㎡ 之间的记录
    # 极小面积可能是车位/地下室，极大面积可能是别墅或数据错误
    mask_before = len(df)
    df = df[(df["square"] >= 15) & (df["square"] <= 500)]
    outliers_removed += mask_before - len(df)

    # 建筑年代：保留 1950 ~ 2016 之间的记录
    df["constructionTime"] = pd.to_numeric(df["constructionTime"], errors="coerce")
    mask_before = len(df)
    df = df[
        ((df["constructionTime"] >= 1950) & (df["constructionTime"] <= 2016))
        | df["constructionTime"].isna()
    ]
    outliers_removed += mask_before - len(df)

    # 卧室数：保留 1 ~ 8
    mask_before = len(df)
    df = df[(df["livingRoom"] >= 1) & (df["livingRoom"] <= 8)]
    outliers_removed += mask_before - len(df)

    # 客厅数：保留 0 ~ 5
    mask_before = len(df)
    df = df[(df["drawingRoom"] >= 0) & (df["drawingRoom"] <= 5)]
    outliers_removed += mask_before - len(df)

    # 卫生间数：保留 0 ~ 6
    mask_before = len(df)
    df = df[(df["bathRoom"] >= 0) & (df["bathRoom"] <= 6)]
    outliers_removed += mask_before - len(df)

    logger.info(f"  剔除异常值: {outliers_removed} 条")
    logger.info(f"  剩余数据: {len(df)} 条")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：创建新的分析特征。

    - trade_year: 成交年份
    - trade_month: 成交月份
    - house_age: 成交时房龄（成交年份 - 建筑年份）
    - unit_layout: 户型标签（X室X厅）
    - price_per_sqm_category: 单价区间分类
    - total_price_category: 总价区间分类
    - area_category: 面积区间分类

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("特征工程...")

    # 时间特征
    df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")
    df["trade_year"] = df["tradeTime"].dt.year
    df["trade_month"] = df["tradeTime"].dt.month

    # 房龄（成交年份 - 建筑年份）
    df["constructionTime"] = pd.to_numeric(df["constructionTime"], errors="coerce")
    df["house_age"] = df["trade_year"] - df["constructionTime"]
    # 过滤负房龄（数据错误）
    df.loc[df["house_age"] < 0, "house_age"] = np.nan
    df.loc[df["house_age"] > 100, "house_age"] = np.nan

    # 户型标签
    df["layout_label"] = (
        df["livingRoom"].fillna(0).astype(int).astype(str)
        + "室"
        + df["drawingRoom"].fillna(0).astype(int).astype(str)
        + "厅"
    )

    # DOM（挂牌天数）缺失值填充中位数
    median_dom = df["DOM"].median()
    df["DOM_filled"] = df["DOM"].fillna(median_dom)

    # 总价区间
    price_bins = [0, 150, 300, 500, 800, 1200, float("inf")]
    price_labels = ["<150万", "150-300万", "300-500万", "500-800万", "800-1200万", ">1200万"]
    df["total_price_category"] = pd.cut(
        df["totalPrice"], bins=price_bins, labels=price_labels, right=False
    )

    # 单价区间
    unit_bins = [0, 30000, 50000, 70000, 90000, 120000, float("inf")]
    unit_labels = ["<3万", "3-5万", "5-7万", "7-9万", "9-12万", ">12万"]
    df["unit_price_category"] = pd.cut(
        df["price"], bins=unit_bins, labels=unit_labels, right=False
    )

    # 面积区间
    area_bins = [0, 50, 70, 90, 120, 150, float("inf")]
    area_labels = ["<50㎡", "50-70㎡", "70-90㎡", "90-120㎡", "120-150㎡", ">150㎡"]
    df["area_category"] = pd.cut(
        df["square"], bins=area_bins, labels=area_labels, right=False
    )

    logger.info("  特征工程完成")
    return df


def generate_comparison(before: pd.DataFrame, after: pd.DataFrame):
    """
    输出清洗前后对比报告。

    Parameters
    ----------
    before : pd.DataFrame
        原始数据。
    after : pd.DataFrame
        清洗后数据。
    """
    print("\n" + "=" * 65)
    print("  数据预处理 — Before/After 对比")
    print("=" * 65)

    print(f"\n  记录数:")
    print(f"    清洗前: {len(before):,} 条")
    print(f"    清洗后: {len(after):,} 条")
    print(f"    剔除:   {len(before) - len(after):,} 条 ({(1 - len(after)/len(before))*100:.1f}%)")

    print(f"\n  关键字段统计对比:")
    for col, label in [
        ("totalPrice", "总价(万元)"),
        ("price", "单价(元/㎡)"),
        ("square", "面积(㎡)"),
        ("livingRoom", "卧室数"),
        ("constructionTime", "建筑年代"),
    ]:
        if col in before.columns and col in after.columns:
            b = pd.to_numeric(before[col], errors="coerce").dropna()
            a = pd.to_numeric(after[col], errors="coerce").dropna()
            if len(b) == 0 or len(a) == 0:
                continue
            print(f"\n  {label}:")
            print(f"    清洗前: 均值={b.mean():.1f}, 中位数={b.median():.1f}, "
                  f"最小={b.min():.1f}, 最大={b.max():.1f}")
            print(f"    清洗后: 均值={a.mean():.1f}, 中位数={a.median():.1f}, "
                  f"最小={a.min():.1f}, 最大={a.max():.1f}")

    # 新增字段
    new_cols = set(after.columns) - set(before.columns)
    if new_cols:
        print(f"\n  新增特征 ({len(new_cols)} 个): {', '.join(sorted(new_cols))}")

    print("=" * 65 + "\n")


def main():
    """数据预处理主入口。"""
    raw_path = get_raw_path()
    processed_path = get_processed_path()
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # 1. 加载原始数据
    df = load_raw_data(raw_path)
    df_before = df.copy()

    # 2. 修复列类型
    df = fix_column_types(df)

    # 3. 处理 floor 列
    df = fix_floor_column(df)

    # 4. 映射编码字段
    df = map_encoded_fields(df)

    # 4b. 打印行政区映射校验信息（便于人工核对经纬度）
    verify_district_mapping(df)

    # 5. 剔除异常值
    df = handle_outliers(df)

    # 6. 特征工程
    df = feature_engineering(df)

    # 7. 重置索引
    df = df.reset_index(drop=True)

    # 8. 保存清洗后数据
    df.to_csv(processed_path, index=False, encoding="utf-8-sig")
    logger.info(f"清洗后数据已保存: {processed_path}")
    logger.info(f"  最终数据量: {len(df):,} 行, {len(df.columns)} 列")

    # 9. 输出对比报告
    generate_comparison(df_before, df)

    return df


if __name__ == "__main__":
    main()
