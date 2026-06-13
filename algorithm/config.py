# ══════════════════════════════════════════════════════════════
# config.py  ·  全局配置与可选技术点
# ══════════════════════════════════════════════════════════════
#
# 可选技术点集中在 §3，每项均有注释说明所有可选值和推荐原因。
# step0_preprocess.py 只依赖 §1/§2/§4/§5，不受可选技术点影响。
#
# ══════════════════════════════════════════════════════════════

import os
import re


# ══════════════════════════════════════════════════════════════
# §1  路径配置
# ══════════════════════════════════════════════════════════════

_HERE         = os.path.dirname(os.path.abspath(__file__))

# ── 输入模式 ──────────────────────────────────────────────────
# "folder" : 原有方式，扫描文件夹内所有 xlsx，每孔可有多段，取最长段
# "excel"  : 单文件方式，读取一个 xlsx，每个 sheet 为人工选定的稳定段
#            sheet 名格式：{孔号}{信号类型}，如 "1pressure"、"6torque"
# "mixed"  : 混合方式，扫描文件夹内所有 xlsx，每孔每个稳定段独立保留，
#            孔号命名为 {原孔号}-{段序号}（如 1-1、1-2），所有段两两配对
INPUT_MODE   = "mixed"

# mixed 模式长稳定段细分：超过此点数的段自动均分为若干 MIXED_MAX_SEG_LEN 点的子段
# 尾部不足 MIXED_MAX_SEG_LEN 的余量丢弃；各孔子段在孔内重排序（-1, -2, -3 ...）
# 设为 None 则关闭细分
MIXED_MAX_SEG_LEN = 400

_env_folder  = os.environ.get("INPUT_FOLDER_NAME")
INPUT_FOLDER = (os.path.join(_HERE, "../../data/2divide", _env_folder)
                if _env_folder
                else os.path.join(_HERE, "../../data/2divide", "2TRD1.6.8.13.RCT1"))
INPUT_EXCEL  = os.path.join(_HERE, "../../data/2divide/mdf/2TRD1.6.8.13divide - mdf1.xlsx")

# 输出目录以数据文件夹命名，不同数据源互不干扰
# excel 模式时以文件名（不含扩展名）作为标识
_DATA_FOLDER  = (os.path.splitext(os.path.basename(INPUT_EXCEL))[0]
                 if INPUT_MODE == "excel"
                 else os.path.basename(os.path.normpath(INPUT_FOLDER)))
OUTPUT_FOLDER = os.path.join(_HERE, f"{_DATA_FOLDER}_output")


def _get_prefix(input_folder):
    folder_name = os.path.basename(os.path.normpath(input_folder))
    m = re.match(r'^([A-Za-z0-9]*?[A-Za-z]+)(\d+[\.\d]*)$', folder_name)
    return m.group(1) if m else folder_name


if INPUT_MODE == "excel":
    _stem = os.path.splitext(os.path.basename(INPUT_EXCEL))[0]
    _m = re.match(r'^(\d*[A-Z]+)', _stem)
    PREFIX = _m.group(1) if _m else _get_prefix(_stem)
else:  # "folder" or "mixed"
    PREFIX = _get_prefix(INPUT_FOLDER)


def make_label(hole_id, signal_type):
    """生成标识符，如 '2TRD_1pressure'"""
    return f"{PREFIX}_{hole_id}{signal_type}"


def make_dir(*parts):
    """拼接路径并创建目录，返回完整路径。"""
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════
# §2  实验设计 · 控制参数
# ══════════════════════════════════════════════════════════════

# "A"：伺服钻速控制  参数为 ucs + drilling_speed + rotation_speed + doc
# "B"：伺服钻压控制  参数为 rotation_speed + drilling_pressure
CONTROL_MODE = "A"

CONTROL_PARAMS = {
    "1":  {"drilling_speed": 85,  "rotation_speed": 100},
    "6":  {"drilling_speed": 85,  "rotation_speed": 250},
    "8":  {"drilling_speed": 110, "rotation_speed": 100},
    "13": {"drilling_speed": 110, "rotation_speed": 200},
}


def get_pair_meta(hole_A, hole_B):
    """
    分析两孔控制参数差异，返回孔对元信息字典：
      is_single_factor : bool   只有 1 个参数不同
      n_diff_params    : int    不同参数数量
      diff_params      : list   不同参数名列表
      fixed_params     : dict   相同参数及其值
      tag              : str    简短标签，如 "单因素[rotation_speed]"
    """
    pa = CONTROL_PARAMS.get(hole_A)
    pb = CONTROL_PARAMS.get(hole_B)
    if pa is None or pb is None:
        return {"is_single_factor": None, "n_diff_params": None,
                "diff_params": [], "fixed_params": {}, "tag": "未知"}
    if not pa and not pb:
        return {"is_single_factor": None, "n_diff_params": None,
                "diff_params": [], "fixed_params": {}, "tag": "未配置控制参数"}
    all_keys  = sorted(set(pa) | set(pb))
    diff_keys = [k for k in all_keys if pa.get(k) != pb.get(k)]
    fixed     = {k: pa[k] for k in all_keys if pa.get(k) == pb.get(k)}
    n_diff    = len(diff_keys)
    if n_diff == 0:
        tag = "同参数"
    elif n_diff == 1:
        tag = f"单因素[{diff_keys[0]}]"
    elif n_diff == 2:
        tag = f"双因素[{','.join(diff_keys)}]"
    else:
        tag = f"多因素[{','.join(diff_keys)}]"
    return {"is_single_factor": n_diff == 1, "n_diff_params": n_diff,
            "diff_params": diff_keys, "fixed_params": fixed, "tag": tag}


# ══════════════════════════════════════════════════════════════
# §3  可选技术点  ← 在此修改切换方法
# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 3-A  第一幕：WPD 小波基
# ─────────────────────────────────────────────────────────────
# 选项：
#   'db4'  — Daubechies-4，紧支撑，适合机械振动瞬态 ★推荐
#   'sym8' — 近似对称，边界效应更小，适合较长平稳段
#   'coif3'— 近似对称，适合光滑缓变信号
#
# 注：必须使用正交小波，保证能量守恒（不可用 VMD）。
WPD_WAVELET      = "db4"
WPD_MIN_NODE_LEN = 32     # 末层节点最少样本数（自适应层数下限保护）
WPD_MAX_LEVEL    = 5      # 层数上限（自适应时不超过此值）


# ─────────────────────────────────────────────────────────────
# 3-B  第二幕：节点分类阈值模式
# ─────────────────────────────────────────────────────────────
# 作用于各节点 Wasserstein 距离序列 {W_i}，找Common Component/公共分量/Differential Component/差异分量分界。
# 选项：
#   'otsu'   — 大津法自适应断点 ★推荐（与全流程 Otsu 逻辑一致）
#   'median' — 以中位数为阈值
#   'mean'   — 以均值为阈值
#   float    — 手动固定阈值，如 0.05
NODE_THRESH_MODE = "otsu"


# ─────────────────────────────────────────────────────────────
# 3-C  第四幕：CWT 小波基（复数 Morlet）
# ─────────────────────────────────────────────────────────────
# 选项：
#   'cmor1.5-1.0' — 带宽1.5，中心频率1.0，时频均衡 ★推荐
#   'cmor2.0-1.0' — 带宽2.0，频率分辨率↑，时间分辨率↓
#   'cmor1.0-1.5' — 带宽1.0，时间分辨率↑，适合瞬态定位
SST_WAVELET  = "cmor1.5-1.0"
SST_N_SCALES = 128     # 尺度数量；64→快速，128→高分辨率（约慢一倍）


# ─────────────────────────────────────────────────────────────
# 3-D  第四·五幕：图像变换方式
# ─────────────────────────────────────────────────────────────
# 选项：
#   'FQI' — 频率-分位图 ★推荐
#     FQI(f,q) = 频率 f 下幅值的第 q 分位数（逆CDF）
#     · 时间轴 → 分位序 q∈[0,1]，两孔图像坐标系完全一致
#     · 保留绝对物理幅值，不归一化
#     · 逐行 L1 距离 = 精确 1D Wasserstein 距离（图像操作与距离计算统一）
#   'PSD' — 功率谱密度（经典对比，信息量是 FQI 的子集）
#     PSD(f) = 时均功率，退化为 1D 曲线，可作辅助验证
IMAGE_TRANSFORM  = "FQI"
FINE_N_QUANTILES = 128    # FQI 分位点数量（建议 64~256）

# FQI 幅值对数变换：True → log(1+amp)，对高分位端（大幅值冲击）加权
# False → 使用原始幅值，保留物理量纲 ★推荐
FQI_LOG_TRANSFORM = False


# ─────────────────────────────────────────────────────────────
# 3-E  第五幕：软掩模模式
# ─────────────────────────────────────────────────────────────
# 将逐频 Wasserstein 距离 W(f) → 软掩模 mask(f)∈[0,1]
# 选项：
#   'otsu_soft'  — Otsu 阈值自适应分界 + 线性软衰减 ★推荐
#     W(f) > Otsu 阈值的频率：线性从 0.5 升至 1（确保Differential Component/差异分量有实质能量）
#     W(f) < Otsu 阈值的频率：线性从 0 升至 0.5（软过渡，无 Gibbs）
#     与第二幕节点分类的 Otsu 逻辑一致，方法论自洽
#     Differential Component/差异分量约含实质性能量，W 距离统计有意义
#   'normalized' — mask = W/max(W)，无超参数，但 W(f) 分布极不均匀时
#                  大多数频率 mask≈0，Differential Component/差异分量能量极低，不适合真实数据
#   'sigmoid'    — mask = σ((W−μ)/(σ_w·scale))，过渡区形状可调
SOFT_MASK_MODE      = "sigmoid"
SIGMOID_SIGMA_SCALE = 1.0    # 仅 sigmoid 模式：< 1 过渡陡，> 1 过渡缓


# ─────────────────────────────────────────────────────────────
# 3-F  验证：跨孔相关性方法
# ─────────────────────────────────────────────────────────────
# 'spearman' — 秩相关，无正态假设 ★推荐
# 'pearson'  — 线性相关，适合近正态分布信号
# V1 辅助：是否在验证图中附加 PSD 对比曲线（不影响主流程）
# # 方案A（默认，论文描述均值层面分离）
# DEMEAN_BEFORE_DECOMPOSE = False
# # 方案B（去均值，论文描述频率层面分离）
# DEMEAN_BEFORE_DECOMPOSE = True
# ─────────────────────────────────────────────────────────────
# 3-G  V3 参数相关性：幅值度量指标
# ─────────────────────────────────────────────────────────────
# 作用：V3 计算每对孔对的"Differential Component/差异分量幅值差"时使用的度量
# 选项：
#   'std' — 标准差（去均值后的波动幅度）★推荐
#     std = sqrt(E[(x-μ)²])，不含均值偏移，真实反映频率成分的波动能量
#     当信号均值远大于波动时（如 pressure: 均值7MPa, 波动0.2MPa），
#     std 才能正确衡量波动差异，RMS ≈ 均值而无法区分
#   'rms' — 均方根（含均值偏移）
#     RMS² = μ² + std²，均值主导时 RMS ≈ μ
#     当信号均值接近0时，RMS ≈ std，两者等价

# ─────────────────────────────────────────────────────────────
# 3-H  去均值分解开关
# ─────────────────────────────────────────────────────────────
# 作用：在 WPD 分解之前先减去各孔自身均值，分解完成后加回。
# 效果：
#   False（默认）：保留原始物理幅值，DC（均值）差异成为Differential Component/差异分量的主要成分
#     → 方法分离的是"孔间均值偏移 + 波动差异"
#   True：去均值后仅分析波动成分，Otsu 不再被 DC 节点主导
#     → 方法分离的是"频率层面的波动差异"，更纯粹
#     → 对应论文主张"从MWD波动中分离地层响应与参数激发响应"
#   建议：同时运行 False 和 True，对比两种结果，论文中两种模式均有意义
DEMEAN_BEFORE_DECOMPOSE = False

# ─────────────────────────────────────────────────────────────
# 3-I  公共分量底限值恢复开关
# ─────────────────────────────────────────────────────────────
# 作用：分解前将参与计算的所有稳定段信号减去全局最小值（底限），
#       分解完成后仅将底限加回到公共分量（Common Component），差异分量不加。
#
# 物理意义：
#   · 底限 = 该信号类型下所有参与孔段的最小信号值
#     代表"地层对钻头的最低基准阻力"，与岩石强度直接相关
#   · 减去底限后，各孔信号均以同一基准点为零点进入分解
#   · 分解结束后底限加回 comm，使公共分量保留岩性相关的绝对幅值信息
#   · 不同岩性的底限不同（硬岩底限高），从而增大跨岩性的 comm 区分度
#   · 差异分量不加底限，保持参数效应的相对表达
#
# 效果：
#   False（默认）：不做底限处理，comm 幅值主要为波动成分，岩性间绝对幅值差小
#   True：comm 恢复岩性基准幅值，跨岩性 Fisher 比提高，岩性识别能力增强
#
# 注：与 DEMEAN_BEFORE_DECOMPOSE 独立，可组合使用
COMM_BASELINE = True

# ══════════════════════════════════════════════════════════════
# §4  输出控制
# ══════════════════════════════════════════════════════════════

SIGNAL_TYPES      = ["pressure", "torque"]
SAVE_INTERMEDIATE = True    # True → 保存中间 npz 数据

# ══════════════════════════════════════════════════════════════
# §5  输出路径（无需修改）
# ══════════════════════════════════════════════════════════════
#
#   output/
OUTPUT_STEP0 = os.path.join(OUTPUT_FOLDER, "step0_preprocessed")
OUTPUT_STEP1 = os.path.join(OUTPUT_FOLDER, "step1_line1")
# 旧名兼容（保留，供 step0 使用）
OUTPUT_LINE1  = OUTPUT_STEP1
OUTPUT_INTERM = OUTPUT_STEP0


def make_pair_dir(step_root, signal_type, label_A, label_B, subdir=None):
    """构建并创建孔对子目录，返回完整路径。"""
    path = os.path.join(step_root, signal_type, f"{label_A}_vs_{label_B}")
    if subdir:
        path = os.path.join(path, subdir)
    os.makedirs(path, exist_ok=True)
    return path

# ─────────────────────────────────────────────────────────────
# 3-I  V7 降维方法（距离地图）
# ─────────────────────────────────────────────────────────────
# 用于 V7 2D距离地图的降维算法，将4孔间的距离矩阵投影到2D平面。
# 选项：
#   'mds'   — 多维尺度分析 ★推荐
#     直接保持Wasserstein距离关系，数学上最忠实，无需额外安装
#   'umap'  — 均匀流形近似（需 pip install umap-learn）
#     保留局部拓扑结构，4个点时结果与MDS差异不大
#   'tsne'  — t-SNE（scikit-learn自带，但4点时不稳定，不推荐）
# 注：4个孔的数据在最多3维空间里，任何降维到2D信息损失都接近0。
