"""
Streamlit 交互式 Web 应用。
北京市二手房成交数据多维度分析与可视化平台。
启动命令: streamlit run app/streamlit_app.py
"""

import os
import sys

# 最先加载 .env（必须在 ai_analysis 被导入之前）
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(os.path.abspath(_env_path), override=True)

# 确保 src/ 在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz_utils import set_style, get_figures_dir
from ai_analysis import generate_analysis, get_client

# ---- 页面配置 ----
st.set_page_config(
    page_title="北京二手房价格分析平台",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 加载数据 ----
@st.cache_data
def load_data():
    """加载清洗后的数据，带缓存。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), "data", "processed", "ershoufang_clean.csv")
    df = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
    df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")
    df["trade_year"] = df["tradeTime"].dt.year
    return df


df = load_data()

# ============================================================
# 侧边栏：数据筛选控件
# ============================================================
st.sidebar.title("🏠 筛选条件")
st.sidebar.markdown("---")

# 筛选 1: 行政区（多选下拉框）
all_districts = sorted(df["district_name"].dropna().unique())
selected_districts = st.sidebar.multiselect(
    "行政区",
    options=all_districts,
    default=all_districts[:5],
    help="选择一个或多个行政区进行分析",
)

# 筛选 2: 价格范围（滑块）
price_min = int(df["totalPrice"].min())
price_max = int(df["totalPrice"].max())
selected_price = st.sidebar.slider(
    "总价范围（万元）",
    min_value=price_min,
    max_value=price_max,
    value=(100, 800),
    step=10,
    help="拖动滑块选择总价范围",
)

# 筛选 3: 面积范围（滑块）
area_min = int(df["square"].min())
area_max = int(df["square"].max())
selected_area = st.sidebar.slider(
    "面积范围（㎡）",
    min_value=area_min,
    max_value=area_max,
    value=(50, 150),
    step=5,
    help="拖动滑块选择面积范围",
)

# 筛选 4: 户型选择（多选下拉框）
all_layouts = sorted(df["layout_label"].dropna().unique(), key=lambda x: str(x))
selected_layouts = st.sidebar.multiselect(
    "户型",
    options=all_layouts,
    default=all_layouts[:8],
    help="选择户型（X室X厅）",
)

# 筛选 5: 年份范围（滑块）
year_min = int(df["trade_year"].min())
year_max = int(df["trade_year"].max())
selected_years = st.sidebar.slider(
    "成交年份",
    min_value=year_min,
    max_value=year_max,
    value=(2012, 2016),
    step=1,
    help="选择成交年份范围",
)

# 应用筛选
mask = (
    df["district_name"].isin(selected_districts)
    & (df["totalPrice"] >= selected_price[0])
    & (df["totalPrice"] <= selected_price[1])
    & (df["square"] >= selected_area[0])
    & (df["square"] <= selected_area[1])
    & df["layout_label"].isin(selected_layouts)
    & (df["trade_year"] >= selected_years[0])
    & (df["trade_year"] <= selected_years[1])
)
filtered_df = df[mask]

st.sidebar.markdown("---")
st.sidebar.metric("筛选后记录数", f"{len(filtered_df):,} 条")
st.sidebar.metric("占总数据比例", f"{len(filtered_df)/len(df)*100:.1f}%")

# ============================================================
# 主页面
# ============================================================
st.title("🏠 北京市二手房成交价格多维度分析平台")
st.markdown(
    "数据来源：链家北京二手房成交记录（2002-2018），"
    f"共 **{len(df):,}** 条记录，当前筛选 **{len(filtered_df):,}** 条。"
)
st.markdown("---")

# ---- 创建标签页 ----
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 数据概览", "📈 描述性统计", "🗺️ 交互式可视化", "🤖 建模分析"]
)

# ============================================================
# Tab 1: 数据概览
# ============================================================
with tab1:
    st.header("数据概览")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总记录数", f"{len(filtered_df):,}")
    with col2:
        st.metric("均价（元/㎡）", f"{filtered_df['price'].mean():,.0f}")
    with col3:
        st.metric("均总价（万元）", f"{filtered_df['totalPrice'].mean():.0f}")
    with col4:
        st.metric("均面积（㎡）", f"{filtered_df['square'].mean():.0f}")

    st.markdown("---")
    st.subheader("数据样本（前 100 行）")
    display_cols = [
        "tradeTime", "district_name", "community", "layout_label",
        "square", "totalPrice", "price", "house_age", "renovation_name",
    ]
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[available_cols].head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("字段说明")
    field_info = pd.DataFrame({
        "字段名": ["tradeTime", "district_name", "layout_label", "square",
                    "totalPrice", "price", "house_age", "renovation_name",
                    "Lng", "Lat", "DOM", "elevator", "subway"],
        "含义": ["成交时间", "行政区", "户型（室厅）", "面积（㎡）",
                  "总价（万元）", "单价（元/㎡）", "房龄（年）", "装修状况",
                  "经度", "纬度", "挂牌天数", "有无电梯", "是否近地铁"],
        "类型": ["日期", "分类", "分类", "数值",
                  "数值", "数值", "数值", "分类",
                  "数值", "数值", "数值", "二值", "二值"],
    })
    st.dataframe(field_info, use_container_width=True, hide_index=True)

# ============================================================
# Tab 2: 描述性统计
# ============================================================
with tab2:
    st.header("描述性统计分析")

    st.subheader("核心指标统计量")
    stat_cols = {
        "totalPrice": "总价（万元）",
        "price": "单价（元/㎡）",
        "square": "面积（㎡）",
        "house_age": "房龄（年）",
        "DOM": "挂牌天数",
    }
    stats_data = []
    for col, label in stat_cols.items():
        if col in filtered_df.columns:
            s = filtered_df[col].dropna()
            stats_data.append({
                "指标": label,
                "样本量": f"{len(s):,}",
                "均值": f"{s.mean():.1f}",
                "中位数": f"{s.median():.1f}",
                "标准差": f"{s.std():.1f}",
                "最小值": f"{s.min():.1f}",
                "最大值": f"{s.max():.1f}",
            })
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("各行政区均价对比")
    district_agg = (
        filtered_df.groupby("district_name")
        .agg(
            均价=("price", "mean"),
            中位数价=("price", "median"),
            成交量=("price", "count"),
        )
        .sort_values("均价", ascending=False)
        .reset_index()
    )
    fig = px.bar(
        district_agg,
        x="district_name",
        y="均价",
        color="成交量",
        text=district_agg["均价"].apply(lambda x: f"{x:,.0f}"),
        color_continuous_scale="Blues",
        title="各行政区成交均价对比",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="行政区", yaxis_title="均价（元/㎡）")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 3: 交互式可视化
# ============================================================
with tab3:
    st.header("交互式可视化分析")

    viz_option = st.selectbox(
        "选择可视化类型",
        [
            "单价分布直方图",
            "总价-面积散点图",
            "年度价格趋势",
            "行政区价格箱线图",
            "空间分布热力图",
            "户型成交量对比",
            "房龄-价格关系",
        ],
    )

    # 抽样以提升交互性能
    sample_size = min(10000, len(filtered_df))
    sample = filtered_df.sample(n=sample_size, random_state=42) if len(filtered_df) > sample_size else filtered_df

    if viz_option == "单价分布直方图":
        fig = px.histogram(
            sample, x="price", nbins=80,
            marginal="box",
            color_discrete_sequence=["#4C72B0"],
            title="二手房成交单价分布",
        )
        fig.update_layout(xaxis_title="单价（元/㎡）", yaxis_title="成交套数")
        fig.add_vline(x=filtered_df["price"].median(), line_dash="dash", line_color="red",
                       annotation_text=f"中位数: {filtered_df['price'].median():,.0f}")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "总价-面积散点图":
        fig = px.scatter(
            sample, x="square", y="totalPrice",
            color="district_name",
            size="price",
            hover_data=["community", "layout_label"],
            opacity=0.6,
            title="面积 vs 总价（按行政区着色，气泡大小=单价）",
        )
        fig.update_layout(xaxis_title="面积（㎡）", yaxis_title="总价（万元）")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "年度价格趋势":
        yearly = (
            filtered_df.groupby("trade_year")
            .agg(均价=("price", "mean"), 中位数=("price", "median"), 成交量=("price", "count"))
            .reset_index()
        )
        yearly = yearly[yearly["trade_year"] >= 2008]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=yearly["trade_year"], y=yearly["均价"], name="均价",
                        mode="lines+markers", line=dict(color="#4C72B0", width=2)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=yearly["trade_year"], y=yearly["中位数"], name="中位数价",
                        mode="lines+markers", line=dict(color="#55A868", width=2, dash="dash")),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(x=yearly["trade_year"], y=yearly["成交量"], name="成交量",
                    marker=dict(color="#C44E52", opacity=0.3)),
            secondary_y=True,
        )
        fig.update_xaxes(title_text="年份")
        fig.update_yaxes(title_text="单价（元/㎡）", secondary_y=False)
        fig.update_yaxes(title_text="成交量（套）", secondary_y=True)
        fig.update_layout(title="年度成交均价与成交量趋势")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "行政区价格箱线图":
        fig = px.box(
            sample, x="district_name", y="price",
            color="district_name",
            title="各行政区单价箱线图",
        )
        fig.update_layout(xaxis_title="行政区", yaxis_title="单价（元/㎡）", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "空间分布热力图":
        st.markdown("💡 提示：可缩放、拖拽地图，悬停查看详情")
        fig = px.density_mapbox(
            sample,
            lat="Lat",
            lon="Lng",
            z="price",
            radius=10,
            center=dict(lat=39.92, lon=116.40),
            zoom=9,
            mapbox_style="carto-positron",
            title="二手房成交单价空间热力图",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "户型成交量对比":
        layout_counts = (
            filtered_df["layout_label"].value_counts().head(15).reset_index()
        )
        layout_counts.columns = ["户型", "成交量"]
        fig = px.bar(
            layout_counts, x="成交量", y="户型",
            orientation="h",
            color="成交量",
            color_continuous_scale="Blues",
            title="热门户型成交量 (Top 15)",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "房龄-价格关系":
        valid_age = sample[(sample["house_age"] >= 0) & (sample["house_age"] <= 50)]
        fig = px.scatter(
            valid_age, x="house_age", y="price",
            color="district_name",
            opacity=0.5,
            trendline="lowess",
            title="房龄 vs 单价（含 LOWESS 趋势线）",
        )
        fig.update_layout(xaxis_title="房龄（年）", yaxis_title="单价（元/㎡）")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 4: 建模分析
# ============================================================
with tab4:
    st.header("数据建模分析")

    st.subheader("K-Means 行政区聚类")
    st.markdown("""
    基于各行政区的 **均价、总价、面积、房龄、成交量** 等特征进行 K-Means 聚类（k=3），
    将北京 13 个行政区划分为三个层级：
    """)

    # 聚类计算（与 analysis.py 保持一致的 6 特征）
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    agg = filtered_df.groupby("district_name").agg(
        avg_price=("price", "mean"),
        avg_total_price=("totalPrice", "mean"),
        avg_area=("square", "mean"),
        avg_age=("house_age", "mean"),
        transaction_count=("price", "count"),
        avg_dom=("DOM", "median"),
    ).dropna()

    features = ["avg_price", "avg_total_price", "avg_area", "avg_age", "transaction_count", "avg_dom"]
    X_scaled = StandardScaler().fit_transform(agg[features].values)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    agg["cluster"] = kmeans.fit_predict(X_scaled)
    cluster_order = agg.groupby("cluster")["avg_price"].mean().sort_values(ascending=False).index
    cluster_names = {
        cluster_order[0]: "🔴 高房价核心区",
        cluster_order[1]: "🟡 中等房价区",
        cluster_order[2]: "🟢 低房价外围区",
    }
    agg["聚类"] = agg["cluster"].map(cluster_names)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(
            agg.reset_index(),
            x="avg_price",
            y="avg_total_price",
            color="聚类",
            size="transaction_count",
            text="district_name",
            title="行政区聚类结果（气泡大小=成交量）",
            color_discrete_map={
                "🔴 高房价核心区": "#C44E52",
                "🟡 中等房价区": "#F0C040",
                "🟢 低房价外围区": "#55A868",
            },
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_title="均价（元/㎡）", yaxis_title="均总价（万元）")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 聚类结果")
        for cname in ["🔴 高房价核心区", "🟡 中等房价区", "🟢 低房价外围区"]:
            members = agg[agg["聚类"] == cname].index.tolist()
            st.markdown(f"**{cname}**")
            for m in members:
                st.markdown(f"  - {m}")
            st.markdown("")

    
    st.markdown("---")
    st.subheader("🤖 AI 智能分析报告")

    ai_type = st.radio(
        "选择分析角度",
        ["全面分析", "价格趋势聚焦", "区域对比聚焦", "购房建议聚焦"],
        horizontal=True,
    )

    client = get_client()
    if client is None:
        st.warning(
            "⚠️ **DeepSeek API Key 未配置**\n\n"
            "1. 复制 `.env.example` 为 `.env`\n"
            "2. 在 `.env` 中填入 `DEEPSEEK_API_KEY=sk-xxx`\n"
            "3. 重启 Streamlit 应用"
        )
        from ai_analysis import summarize_dataframe
        st.markdown("### 当前筛选数据统计摘要")
        st.text(summarize_dataframe(filtered_df))
    else:
        if st.button("🚀 生成 AI 分析报告", type="primary", use_container_width=True):
            with st.spinner("DeepSeek 正在分析数据，请稍候..."):
                report = generate_analysis(filtered_df, analysis_type=ai_type)
            st.markdown("### 📝 AI 分析报告")
            st.markdown(report)
        else:
            st.info("👆 点击上方按钮，DeepSeek 将根据当前筛选数据自动生成分析报告")

    st.markdown("---")
    st.subheader("随机森林房价预测模型")

    # 准备建模数据
    model_df = filtered_df.dropna(subset=[
        "price", "square", "livingRoom", "drawingRoom", "bathRoom",
        "house_age", "total_floors", "Lng", "Lat", "district_name",
    ]).copy()

    if len(model_df) < 1000:
        st.warning("筛选后数据量不足（需 ≥1,000 条），无法训练可靠的预测模型。请放宽筛选条件。")
    else:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        # 若数据量过大，随机抽样以保障交互响应速度
        max_train_size = 50000
        if len(model_df) > max_train_size:
            model_df = model_df.sample(n=max_train_size, random_state=42)

        feature_cols = [
            "square", "livingRoom", "drawingRoom", "bathRoom",
            "house_age", "total_floors", "Lng", "Lat", "district_name",
        ]
        X = model_df[feature_cols]
        y = model_df["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

        numeric_features = [
            "square", "livingRoom", "drawingRoom", "bathRoom",
            "house_age", "total_floors", "Lng", "Lat",
        ]
        categorical_features = ["district_name"]

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ])

        with st.spinner("正在训练随机森林模型…"):
            model = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=100, max_depth=20, random_state=42, n_jobs=-1,
                )),
            ])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.markdown(f"""
        使用 **随机森林回归** 模型预测二手房单价，特征包括：
        面积、卧室数、客厅数、卫生间数、房龄、总楼层、经纬度、行政区
        （基于当前筛选数据训练，训练集 {len(X_train):,} 条，测试集 {len(X_test):,} 条）。

        | 指标 | 数值 |
        |------|------|
        | R²（决定系数） | **{r2:.3f}** |
        | MAE（平均绝对误差） | **{mae:,.0f} 元/㎡** |
        | RMSE（均方根误差） | **{rmse:,.0f} 元/㎡** |
        """)

        # ---- 特征重要性（动态计算，中文标注） ----
        rf = model.named_steps["regressor"]
        ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
        all_features = numeric_features + cat_feature_names
        importances = rf.feature_importances_

        feat_imp = pd.DataFrame({"feature": all_features, "importance": importances})
        feat_imp = feat_imp.sort_values("importance", ascending=False).head(10)

        # 英文特征名 → 中文标签
        feat_zh_map = {
            "square": "面积", "livingRoom": "卧室数", "drawingRoom": "客厅数",
            "bathRoom": "卫生间数", "house_age": "房龄", "total_floors": "总楼层",
            "Lng": "经度", "Lat": "纬度",
        }

        def zh_name(f):
            if f.startswith("district_name_"):
                return f.replace("district_name_", "")
            return feat_zh_map.get(f, f)

        feat_imp["特征"] = feat_imp["feature"].map(zh_name)
        feat_imp["重要性"] = feat_imp["importance"]

        st.subheader("特征重要性分析")
        fig = px.bar(
            feat_imp.iloc[::-1], x="重要性", y="特征", orientation="h",
            color="重要性", color_continuous_scale="Blues",
            title=f"随机森林特征重要性 Top 10（基于 {len(model_df):,} 条训练数据）",
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 页脚
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "数据可视化课程大作业 · 北京市二手房成交价格多维度分析与可视化 · 数据来源：Kaggle (ruiqurm/lianjia)"
    "</div>",
    unsafe_allow_html=True,
)
