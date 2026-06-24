"""
数据分析与建模模块。
产出：描述性统计、聚类分析、回归建模，所有图表保存至 figures/。
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from viz_utils import set_style, save_figure, get_figures_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
FIGURES_DIR = get_figures_dir()


def load_clean_data():
    """加载清洗后数据。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(current_dir), "data", "processed", "ershoufang_clean.csv")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")
    logger.info(f"加载清洗数据: {len(df)} 行")
    return df


# ============================================================
# 3.1 描述性统计分析
# ============================================================

def descriptive_stats(df: pd.DataFrame):
    """输出描述性统计结果（集中趋势 + 离散程度）。"""
    print("\n" + "=" * 65)
    print("  3.1 描述性统计分析")
    print("=" * 65)

    for col, label in [
        ("totalPrice", "总价(万元)"),
        ("price", "单价(元/㎡)"),
        ("square", "面积(㎡)"),
        ("house_age", "房龄(年)"),
        ("DOM", "挂牌天数"),
    ]:
        s = df[col].dropna()
        print(f"\n  {label}:")
        print(f"    样本量: {len(s):,}")
        print(f"    均值={s.mean():.1f}, 中位数={s.median():.1f}")
        print(f"    标准差={s.std():.1f}, 偏度={s.skew():.2f}, 峰度={s.kurtosis():.2f}")
        print(f"    最小={s.min():.1f}, Q1={s.quantile(0.25):.1f}, Q3={s.quantile(0.75):.1f}, 最大={s.max():.1f}")

    # 行政区统计
    if "district_name" in df.columns:
        print(f"\n  各行政区成交均价 (元/㎡):")
        district_stats = df.groupby("district_name")["price"].agg(["mean", "median", "count"])
        district_stats = district_stats.sort_values("mean", ascending=False)
        for idx, row in district_stats.iterrows():
            print(f"    {idx}: 均价={row['mean']:.0f}, 中位数={row['median']:.0f}, 成交量={row['count']:,}")

    print("=" * 65 + "\n")


# ---- 图 3-1: 总价分布 ----
def fig_total_price_dist(df: pd.DataFrame):
    """图 3-1: 北京市二手房成交总价分布直方图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    prices = df["totalPrice"]
    ax.hist(prices, bins=80, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(prices.median(), color="red", linestyle="--", linewidth=1.5, label=f"中位数: {prices.median():.0f}万")
    ax.axvline(prices.mean(), color="orange", linestyle="--", linewidth=1.5, label=f"均值: {prices.mean():.0f}万")
    ax.set_xlabel("总价（万元）", fontsize=12)
    ax.set_ylabel("成交套数", fontsize=12)
    # 不在图中嵌入"图 3-1"标题，由 Word 文档下方的题注承担；保留简短信息标题。
    ax.set_title("北京市二手房成交总价分布直方图", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1500)
    save_figure(fig, "fig_3_1_total_price_dist.png")
    plt.close(fig)


# ---- 图 3-2: 单价分布 ----
def fig_unit_price_dist(df: pd.DataFrame):
    """图 3-2: 北京市二手房成交单价分布直方图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    prices = df["price"]
    ax.hist(prices, bins=80, color="#55A868", edgecolor="white", alpha=0.85)
    ax.axvline(prices.median(), color="red", linestyle="--", linewidth=1.5, label=f"中位数: {prices.median():.0f}元/㎡")
    ax.axvline(prices.mean(), color="orange", linestyle="--", linewidth=1.5, label=f"均值: {prices.mean():.0f}元/㎡")
    ax.set_xlabel("单价（元/㎡）", fontsize=12)
    ax.set_ylabel("成交套数", fontsize=12)
    ax.set_title("北京市二手房成交单价分布直方图", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 140000)
    save_figure(fig, "fig_3_2_unit_price_dist.png")
    plt.close(fig)


# ---- 图 3-3: 面积分布 ----
def fig_area_dist(df: pd.DataFrame):
    """图 3-3: 北京市二手房成交面积分布直方图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    areas = df["square"]
    ax.hist(areas, bins=80, color="#C44E52", edgecolor="white", alpha=0.85)
    ax.axvline(areas.median(), color="blue", linestyle="--", linewidth=1.5, label=f"中位数: {areas.median():.0f}㎡")
    ax.axvline(areas.mean(), color="orange", linestyle="--", linewidth=1.5, label=f"均值: {areas.mean():.0f}㎡")
    ax.set_xlabel("面积（㎡）", fontsize=12)
    ax.set_ylabel("成交套数", fontsize=12)
    ax.set_title("北京市二手房成交面积分布直方图", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 250)
    save_figure(fig, "fig_3_3_area_dist.png")
    plt.close(fig)


# ---- 图 3-4: 行政区价格箱线图 ----
def fig_price_boxplot_by_district(df: pd.DataFrame):
    """图 3-4: 各行政区成交单价箱线图。"""
    fig, ax = plt.subplots(figsize=(14, 6))
    # 按均价排序
    order = df.groupby("district_name")["price"].median().sort_values(ascending=False).index
    sns.boxplot(
        x="district_name", y="price", data=df, order=order,
        palette="Set2", ax=ax, showfliers=False,
        hue="district_name", legend=False,
    )
    ax.set_xlabel("行政区", fontsize=12)
    ax.set_ylabel("单价（元/㎡）", fontsize=12)
    ax.set_title("北京市各行政区二手房成交单价箱线图", fontsize=13)
    plt.xticks(rotation=30, ha="right")
    ax.axhline(df["price"].median(), color="red", linestyle="--", linewidth=1, alpha=0.7, label="全市中位数")
    ax.legend(fontsize=9)
    save_figure(fig, "fig_3_4_price_boxplot_by_district.png")
    plt.close(fig)


# ---- 图 3-5: 年度价格趋势 ----
def fig_price_trend_by_year(df: pd.DataFrame):
    """图 3-5: 北京市二手房年度成交均价与成交量趋势。"""
    yearly = df.groupby("trade_year").agg(
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        count=("price", "count"),
    ).reset_index()
    yearly = yearly[yearly["trade_year"] >= 2008]  # 过滤早期稀疏年份

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(yearly["trade_year"], yearly["avg_price"], "o-", color="#4C72B0", linewidth=2, markersize=6, label="均价")
    ax1.plot(yearly["trade_year"], yearly["median_price"], "s--", color="#55A868", linewidth=2, markersize=6, label="中位数价")
    ax2.bar(yearly["trade_year"], yearly["count"], alpha=0.3, color="#C44E52", label="成交量")

    ax1.set_xlabel("年份", fontsize=12)
    ax1.set_ylabel("单价（元/㎡）", fontsize=12)
    ax2.set_ylabel("成交量（套）", fontsize=12)
    # 用实际年份范围而非硬编码 (2008-2016)
    year_min = int(yearly["trade_year"].min())
    year_max = int(yearly["trade_year"].max())
    ax1.set_title(f"北京市二手房年度成交均价与成交量趋势 ({year_min}-{year_max})", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    save_figure(fig, "fig_3_5_price_trend_by_year.png")
    plt.close(fig)


# ---- 图 3-6: 户型分布 ----
def fig_layout_distribution(df: pd.DataFrame):
    """图 3-6: 北京市二手房成交户型分布。"""
    total = len(df)
    layout_counts = df["layout_label"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(layout_counts))
    bars = ax.barh(layout_counts.index[::-1], layout_counts.values[::-1], color=colors[::-1])
    ax.set_xlabel("成交套数", fontsize=12)
    ax.set_ylabel("户型", fontsize=12)
    ax.set_title("北京市二手房成交户型分布 (Top 15)", fontsize=13)
    # 同时标注数量和占比，便于报告引用真实百分比
    for bar, val in zip(bars, layout_counts.values[::-1]):
        pct = val / total * 100
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({pct:.1f}%)", va="center", fontsize=8)
    save_figure(fig, "fig_3_6_layout_distribution.png")
    plt.close(fig)


# ---- 图 3-7: 房龄分布 ----
def fig_house_age_dist(df: pd.DataFrame):
    """图 3-7: 北京市二手房成交房龄分布。"""
    ages = df["house_age"].dropna()
    ages = ages[(ages >= 0) & (ages <= 60)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ages, bins=60, color="#8C564B", edgecolor="white", alpha=0.85)
    ax.axvline(ages.median(), color="blue", linestyle="--", linewidth=1.5, label=f"中位数: {ages.median():.0f}年")
    ax.set_xlabel("房龄（年）", fontsize=12)
    ax.set_ylabel("成交套数", fontsize=12)
    ax.set_title("北京市二手房成交房龄分布", fontsize=13)
    ax.legend(fontsize=10)
    save_figure(fig, "fig_3_7_house_age_dist.png")
    plt.close(fig)


# ============================================================
# 3.2 数据建模
# ============================================================

# ---- 图 3-8: 相关性热力图 ----
def fig_correlation_heatmap(df: pd.DataFrame):
    """图 3-8: 数值特征相关性热力图。"""
    # 列名→中文映射，避免图中英文与正文中文不统一
    col_zh = {
        "totalPrice": "总价",
        "price": "单价",
        "square": "面积",
        "livingRoom": "卧室数",
        "drawingRoom": "客厅数",
        "bathRoom": "卫生间数",
        "house_age": "房龄",
        "total_floors": "总楼层",
        "DOM": "挂牌天数",
        "Lng": "经度",
        "Lat": "纬度",
        "communityAverage": "小区均价",
    }
    # 去掉 ladderRatio—— 该列实际取值方差极小（与所有变量相关系数≈0），无分析价值
    numeric_cols = list(col_zh.keys())
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()
    # 用中文重命名
    corr_zh = corr.rename(index=col_zh, columns=col_zh)

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_zh, dtype=bool), k=1)
    sns.heatmap(
        corr_zh, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("北京市二手房数值特征相关性热力图", fontsize=13)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    save_figure(fig, "fig_3_8_correlation_heatmap.png")
    plt.close(fig)


# ---- 图 3-9: K-Means 行政区聚类 ----
def fig_cluster_analysis(df: pd.DataFrame):
    """
    图 3-9: 基于各行政区的房价与交易特征进行 K-Means 聚类，
    将行政区划分为不同层级。
    """
    # 按行政区聚合特征
    agg = df.groupby("district_name").agg(
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        avg_total_price=("totalPrice", "mean"),
        avg_area=("square", "mean"),
        avg_age=("house_age", "mean"),
        transaction_count=("price", "count"),
        avg_dom=("DOM", "median"),
    ).dropna()

    # 标准化
    features = ["avg_price", "avg_total_price", "avg_area", "avg_age", "transaction_count", "avg_dom"]
    X = agg[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 聚类 (k=3: 高/中/低)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    agg["cluster"] = kmeans.fit_predict(X_scaled)

    # 按均价排序给聚类命名
    cluster_order = agg.groupby("cluster")["avg_price"].mean().sort_values(ascending=False).index
    cluster_names = {cluster_order[0]: "高房价核心区", cluster_order[1]: "中等房价区", cluster_order[2]: "低房价外围区"}
    agg["cluster_name"] = agg["cluster"].map(cluster_names)

    logger.info("聚类结果:")
    for name, group in agg.groupby("cluster_name"):
        logger.info(f"  {name}: {list(group.index)}")

    # 可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {"高房价核心区": "#C44E52", "中等房价区": "#55A868", "低房价外围区": "#4C72B0"}
    for cname, group in agg.groupby("cluster_name"):
        ax.scatter(
            group["avg_price"], group["avg_total_price"],
            s=group["transaction_count"] / 50, c=colors.get(cname, "gray"),
            alpha=0.7, edgecolors="black", linewidth=0.5, label=cname,
        )
        for dist, row in group.iterrows():
            ax.annotate(dist, (row["avg_price"], row["avg_total_price"]),
                        fontsize=7, ha="center", va="bottom", alpha=0.8)

    ax.set_xlabel("平均单价（元/㎡）", fontsize=12)
    ax.set_ylabel("平均总价（万元）", fontsize=12)
    ax.set_title("北京市行政区二手房市场聚类分析 (K-Means, k=3)", fontsize=13)
    ax.legend(fontsize=9, title="聚类结果", title_fontsize=10)
    save_figure(fig, "fig_3_9_cluster_analysis.png")
    plt.close(fig)

    return agg


# ---- 图 3-10: 房价预测模型 ----
def fig_price_prediction(df: pd.DataFrame):
    """
    图 3-10: 基于随机森林的二手房单价预测模型。
    展示特征重要性与预测效果。
    """
    logger.info("训练房价预测模型 (Random Forest)...")

    # 准备特征
    model_df = df.dropna(subset=["price", "square", "livingRoom", "drawingRoom",
                                  "bathRoom", "house_age", "total_floors", "Lng", "Lat",
                                  "district_name"]).copy()

    if len(model_df) < 1000:
        logger.warning("建模数据不足")
        return

    # 特征与目标
    feature_cols = ["square", "livingRoom", "drawingRoom", "bathRoom",
                    "house_age", "total_floors", "Lng", "Lat", "district_name"]
    X = model_df[feature_cols]
    y = model_df["price"]

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
    )

    # 预处理：数值特征标准化 + 类别特征独热编码
    numeric_features = ["square", "livingRoom", "drawingRoom", "bathRoom",
                        "house_age", "total_floors", "Lng", "Lat"]
    categorical_features = ["district_name"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info(f"  模型性能: R²={r2:.4f}, MAE={mae:.0f}元/㎡, RMSE={rmse:.0f}元/㎡")

    # ---- 特征重要性图 ----
    rf = model.named_steps["regressor"]
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
    cat_features = list(ohe.get_feature_names_out(categorical_features))
    all_features = numeric_features + cat_features
    importances = rf.feature_importances_

    # 取 Top 15
    feat_imp = pd.DataFrame({"feature": all_features, "importance": importances})
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

    # 将英文特征名翻译为中文标签，并把 OneHot 编码生成的 "district_name_XXX" 简化为 "XXX"
    feat_zh_map = {
        "square": "面积", "livingRoom": "卧室数", "drawingRoom": "客厅数",
        "bathRoom": "卫生间数", "house_age": "房龄", "total_floors": "总楼层",
        "Lng": "经度", "Lat": "纬度",
    }
    def zh_name(f):
        if f.startswith("district_name_"):
            return f.replace("district_name_", "")
        return feat_zh_map.get(f, f)
    feat_imp["feature_zh"] = feat_imp["feature"].map(zh_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 特征重要性
    ax1.barh(feat_imp["feature_zh"][::-1], feat_imp["importance"][::-1], color="#4C72B0")
    ax1.set_xlabel("重要性", fontsize=12)
    ax1.set_title("(a) 特征重要性 (Top 15)", fontsize=13)

    # 预测 vs 实际散点图
    sample = np.random.RandomState(RANDOM_STATE).choice(len(y_test), min(2000, len(y_test)), replace=False)
    ax2.scatter(y_test.iloc[sample], y_pred[sample], alpha=0.3, s=5, color="#55A868")
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=1.5)
    ax2.set_xlabel("实际单价（元/㎡）", fontsize=12)
    ax2.set_ylabel("预测单价（元/㎡）", fontsize=12)
    ax2.set_title(f"(b) 预测 vs 实际 (R²={r2:.3f}, MAE={mae:.0f})", fontsize=13)

    save_figure(fig, "fig_3_10_price_prediction.png")
    plt.close(fig)

    return r2, mae, rmse


# ---- 图 3-11: 空间分布散点图 ----
def fig_spatial_distribution(df: pd.DataFrame):
    """图 3-11: 北京市二手房成交空间分布（按单价着色）。

    叠加各行政区中心点与二环、四环、六环近似圆作为空间参照，
    使纯散点图具有可读的地理语义。
    """
    # 抽样以避免过度绘制
    sample = df.sample(n=min(20000, len(df)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(
        sample["Lng"], sample["Lat"],
        c=sample["price"], cmap="RdYlGn_r", s=2, alpha=0.5, vmax=120000,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label("单价（元/㎡）", fontsize=11)

    # 叠加各行政区中心（用均值经纬度近似中心）
    if "district_name" in df.columns:
        centroids = df.groupby("district_name").agg(
            lng=("Lng", "mean"), lat=("Lat", "mean"),
        )
        for name, row in centroids.iterrows():
            ax.scatter(row["lng"], row["lat"], marker="*", s=80,
                       c="black", edgecolors="white", linewidth=0.6, zorder=5)
            ax.annotate(name, (row["lng"], row["lat"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="black",
                        path_effects=None, zorder=6)

    # 二环（约半径 4km）、四环（约 7.5km）、六环（约 15km），以天安门为圆心
    import matplotlib.patches as mpatches
    center_lng, center_lat = 116.397, 39.908
    # 经纬度→km：北京附近 1° lat ≈ 111km，1° lng ≈ 85km
    for radius_km, label in [(4, "二环"), (7.5, "四环"), (15, "六环")]:
        dlat = radius_km / 111
        dlng = radius_km / 85
        circle = mpatches.Ellipse(
            (center_lng, center_lat), 2 * dlng, 2 * dlat,
            fill=False, linestyle="--", linewidth=1.0, edgecolor="gray", alpha=0.6, zorder=4,
        )
        ax.add_patch(circle)
        ax.annotate(label, (center_lng + dlng * 0.7, center_lat + dlat * 0.7),
                    fontsize=8, color="gray", alpha=0.8, zorder=4)

    ax.set_xlabel("经度", fontsize=12)
    ax.set_ylabel("纬度", fontsize=12)
    ax.set_title("北京市二手房成交空间分布（按单价着色，叠加行政区中心与环线）", fontsize=12)
    ax.set_aspect("equal")
    save_figure(fig, "fig_3_11_spatial_distribution.png")
    plt.close(fig)


# ---- 图 3-12: 装修状况与价格 ----
def fig_renovation_price(df: pd.DataFrame):
    """图 3-12: 不同装修状况下的单价分布。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df.groupby("renovation_name")["price"].median().sort_values(ascending=False).index
    sns.boxplot(
        x="renovation_name", y="price", data=df, order=order,
        palette="Set2", ax=ax, showfliers=False,
        hue="renovation_name", legend=False,
    )
    ax.set_xlabel("装修状况", fontsize=12)
    ax.set_ylabel("单价（元/㎡）", fontsize=12)
    ax.set_title("不同装修状况下的二手房单价分布", fontsize=13)
    # 打印各装修等级的中位数 / 样本数，便于报告引用真实数字
    print("\n  装修状况-单价中位数:")
    for rn in order:
        sub = df[df["renovation_name"] == rn]["price"]
        print(f"    {rn}: 中位数={sub.median():.0f}元/㎡, 均值={sub.mean():.0f}元/㎡, n={len(sub):,}")
    save_figure(fig, "fig_3_12_renovation_price.png")
    plt.close(fig)


# ---- 图 3-13: 面积-总价关系 ----
def fig_area_price_relationship(df: pd.DataFrame):
    """图 3-13: 面积与总价关系散点图，按行政区着色，并叠加各区线性拟合直线。

    叠加回归直线后，正文中"不同行政区斜率不同"的论断有了可视化支撑。
    """
    # 为每个区独立抽样，避免成交量小的区域被冲掉
    fig, ax = plt.subplots(figsize=(11, 7))
    # 选取均价排序两端及中段的代表性区，便于对比"核心 vs 外围"的斜率差异
    target_districts = ["西城区", "东城区", "海淀区", "朝阳区",
                        "丰台区", "通州区", "昌平区", "房山区"]
    sample_per_district = 1500
    color_palette = sns.color_palette("tab10", len(target_districts))

    for dist, color in zip(target_districts, color_palette):
        sub = df[df["district_name"] == dist]
        if len(sub) == 0:
            continue
        sub_sample = sub.sample(n=min(sample_per_district, len(sub)),
                                random_state=RANDOM_STATE)
        ax.scatter(sub_sample["square"], sub_sample["totalPrice"],
                   alpha=0.25, s=6, color=color, label=dist)
        # 各区独立线性拟合
        x = sub["square"].values
        y = sub["totalPrice"].values
        valid = (x > 0) & (x < 300) & (y > 0) & (y < 2000) & np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 30:
            continue
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
        xline = np.linspace(15, 250, 50)
        yline = slope * xline + intercept
        ax.plot(xline, yline, color=color, linewidth=2.0, alpha=0.95)

    ax.set_xlabel("面积（㎡）", fontsize=12)
    ax.set_ylabel("总价（万元）", fontsize=12)
    ax.set_title("北京市二手房面积与总价关系（按行政区着色，附线性拟合）", fontsize=12)
    ax.legend(fontsize=9, markerscale=3, title="行政区", loc="upper left")
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 2000)
    save_figure(fig, "fig_3_13_area_price_scatter.png")
    plt.close(fig)


# ============================================================
# 主入口
# ============================================================

def main():
    """数据分析与可视化主入口。"""
    set_style()
    df = load_clean_data()

    # ---- 3.1 描述性统计 ----
    descriptive_stats(df)

    logger.info("生成描述性统计图表...")
    fig_total_price_dist(df)
    fig_unit_price_dist(df)
    fig_area_dist(df)
    fig_price_boxplot_by_district(df)
    fig_price_trend_by_year(df)
    fig_layout_distribution(df)
    fig_house_age_dist(df)

    # ---- 3.2 数据建模 ----
    logger.info("生成建模分析图表...")
    fig_correlation_heatmap(df)
    fig_cluster_analysis(df)
    fig_price_prediction(df)
    fig_spatial_distribution(df)
    fig_renovation_price(df)
    fig_area_price_relationship(df)

    # 列出所有已生成图表
    logger.info(f"\n已生成 {len(os.listdir(FIGURES_DIR))} 张图表于 {FIGURES_DIR}:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith(".png"):
            size_kb = os.path.getsize(os.path.join(FIGURES_DIR, f)) / 1024
            logger.info(f"  {f} ({size_kb:.0f} KB)")

    print("\n[analysis] 分析完成，所有图表已保存至 figures/ 目录。")


if __name__ == "__main__":
    main()
