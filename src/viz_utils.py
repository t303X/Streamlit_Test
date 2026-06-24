"""
绘图工具函数模块。
统一管理字体、调色板、图片保存等，确保论文所有图表风格一致。

用法：在任何 plotting 之前调用 set_style()。
"""

import os
import platform
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import Optional

# 记录是否已配置
_CONFIGURED = False


def setup_chinese_font():
    """
    配置 matplotlib 中文字体。
    必须在导入 seaborn 和创建任何 figure 之前调用。
    """
    # 查找中文字体文件
    system = platform.system()
    font_path = None

    if system == "Windows":
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
    elif system == "Darwin":
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]

    for fp in candidates:
        if os.path.exists(fp):
            font_path = fp
            break

    if not font_path:
        # 回退：通过字体名查找
        for f in fm.fontManager.ttflist:
            if any(k in f.name for k in ["YaHei", "SimHei", "Heiti", "PingFang"]):
                font_path = f.fname
                break

    if font_path:
        # 添加字体到管理器
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name()
        print(f"[viz_utils] 中文字体: {font_name} ({font_path})")

        # 核心：在 seaborn 之前设置全局 rcParams
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        print("[viz_utils] 警告: 未找到中文字体！")

    # 抑制字体相关的 UserWarning（已知问题：某些特殊字符仍可能缺失）
    warnings.filterwarnings("ignore", message="Glyph.*missing from font")


def set_style():
    """
    统一设置 matplotlib/seaborn 样式。

    调用顺序至关重要：
    1. 先设置 seaborn 样式
    2. 再覆盖字体配置（seaborn 会重置 sans-serif 列表）
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # 1. 先设置 seaborn
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # 2. 再配置字体（覆盖 seaborn 的默认字体）
    setup_chinese_font()

    # 3. 其他全局设置
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.facecolor"] = "white"

    _CONFIGURED = True


def save_figure(
    fig: plt.Figure,
    filename: str,
    figures_dir: Optional[str] = None,
    dpi: int = 300,
):
    """
    保存图表到 figures/ 目录，300dpi，白底。

    Parameters
    ----------
    fig : plt.Figure
        matplotlib 图对象。
    filename : str
        文件名，如 "fig_3_1_amount_dist.png"。
    figures_dir : str, optional
        图表目录路径，默认为项目根目录下的 figures/。
    dpi : int
        分辨率，默认 300。
    """
    if figures_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        figures_dir = os.path.join(os.path.dirname(current_dir), "figures")

    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[viz_utils] 图表已保存: {filepath}")


def get_figures_dir() -> str:
    """返回项目根目录下 figures/ 的绝对路径。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(current_dir), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir
