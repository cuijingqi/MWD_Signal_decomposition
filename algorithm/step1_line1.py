# ══════════════════════════════════════════════════════════════
# step1_line1.py  ·  两级渐进分解
# ══════════════════════════════════════════════════════════════
#
# 一级（粗）：WPD → Wasserstein节点分类 → 粗重构
# 二级（细）：CWT → FQI变换 → 双向软掩模 → FFT精确重构
#
# 四分量定义：
#   δ_diff   : 粗Differential Component/差异分量 × mask_δ        → 真Differential Component/差异分量（两孔分布差异显著的频率）
#   δ_common : 粗Differential Component/差异分量 × (1−mask_δ)    → 假Differential Component/差异分量归还Common Component/公共分量
#   ε_diff   : 粗Common Component/公共分量 × mask_ε        → 漏网Differential Component/差异分量
#   ε_common : 粗Common Component/公共分量 × (1−mask_ε)    → 真细Common Component/公共分量
#
# 最终Common Component/公共分量 = ε_common + δ_common
# 最终Differential Component/差异分量 = δ_diff   + ε_diff
#
# 能量守恒：FFT(signal)×mask + FFT(signal)×(1-mask) = FFT(signal)
#           IFFT后精确守恒，误差 < 2e-15（机器精度）
# ══════════════════════════════════════════════════════════════

import os
import math
import warnings
import numpy as np
import pywt
from scipy.stats import wasserstein_distance as _wd1d

from config import (
    WPD_WAVELET, WPD_MIN_NODE_LEN, WPD_MAX_LEVEL,
    NODE_THRESH_MODE,
    SST_WAVELET, SST_N_SCALES,
    IMAGE_TRANSFORM, FINE_N_QUANTILES, FQI_LOG_TRANSFORM,
    SOFT_MASK_MODE, SIGMOID_SIGMA_SCALE,
    SAVE_INTERMEDIATE,
    OUTPUT_STEP1,
    make_pair_dir,
    DEMEAN_BEFORE_DECOMPOSE,
    COMM_BASELINE,
)


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def _adaptive_level(n_samples):
    """L = min(floor(log2(N / min_len)), max_level)"""
    if n_samples < WPD_MIN_NODE_LEN * 2:
        print(f"    ⚠ 信号长度N={n_samples}过短（< {WPD_MIN_NODE_LEN*2}），"
              f"WPD将退化为 level=1（仅2个节点），节点分类可靠性低")
        return 1
    level = int(math.floor(math.log2(n_samples / WPD_MIN_NODE_LEN)))
    return max(1, min(level, WPD_MAX_LEVEL))


def _otsu_threshold(values):
    """大津法一维最优阈值（使用箱中点计算类均值，减少量化偏差）。"""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(np.mean(values))
    mn, mx = values.min(), values.max()
    if mx - mn < 1e-15:
        return float(mn)
    n_bins = min(256, len(values))
    hist, edges = np.histogram(values, bins=n_bins)
    hist = hist.astype(float)
    # 使用箱中点（左右边缘均值）计算类均值，比左边缘更精确
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    total_sum = np.dot(hist, centers)
    best_t, best_var = edges[1], -1.0
    w0 = s0 = 0.0
    for i, h in enumerate(hist):
        w0 += h
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        s0 += h * centers[i]
        mu0 = s0 / w0
        mu1 = (total_sum - s0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_t = edges[i + 1]
    return float(best_t)


def _select_threshold(values, mode):
    """根据 NODE_THRESH_MODE 选择阈值。"""
    if mode == "otsu":
        return _otsu_threshold(values)
    elif mode == "median":
        return float(np.median(values))
    elif mode == "mean":
        return float(np.mean(values))
    else:
        return float(mode)   # 手动数值



# ══════════════════════════════════════════════════════════════
# 第一幕：WPD 频带分解
# ══════════════════════════════════════════════════════════════

def wpd_reconstruct_nodes(signal, wavelet, level):
    """
    WPD 分解并逐节点重建时域信号。
    使用正交小波 + reflect 模式，保证：
      ∑ nodes ≈ signal（能量守恒，误差 < 1e-10）
    返回 nodes(list[ndarray(N,)]), labels(list[str])
    """
    N = len(signal)
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet,
                            mode="reflect", maxlevel=level)
    leaf_nodes = wp.get_level(level, order="freq")
    nodes, labels = [], []
    for node in leaf_nodes:
        wp_z = pywt.WaveletPacket(data=np.zeros(N), wavelet=wavelet,
                                  mode="reflect", maxlevel=level)
        wp_z[node.path] = node.data
        rec = wp_z.reconstruct(update=False)
        if len(rec) > N:
            rec = rec[:N]
        elif len(rec) < N:
            rec = np.pad(rec, (0, N - len(rec)))
        nodes.append(rec)
        labels.append(f"L{level}-{node.path}")
    return nodes, labels


# ══════════════════════════════════════════════════════════════
# 第二幕：Wasserstein 节点分类（取代 ICA）
# ══════════════════════════════════════════════════════════════

def classify_nodes_wasserstein(nodes_A, nodes_B, thresh_mode):
    """
    对每个 WPD 节点，计算孔A/B 幅值分布的 Wasserstein 距离 W_i。
    阈值分类：W_i 小 → Common Component/公共分量节点；W_i 大 → Differential Component/差异分量节点。

    优势：完全无需时间对齐假设，比的是分布而非波形。
         孔A第t秒和孔B第t秒不需要对应同一物理事件。

    返回：
      is_common : bool ndarray(n_nodes,)
      W_vals    : float ndarray(n_nodes,)
      threshold : float
    """
    W_vals = np.array([float(_wd1d(rA, rB))
                       for rA, rB in zip(nodes_A, nodes_B)])
    threshold = _select_threshold(W_vals, thresh_mode)
    is_common = W_vals < threshold
    print(f"    节点W距离: min={W_vals.min():.4f}  max={W_vals.max():.4f}"
          f"  阈值({thresh_mode})={threshold:.4f}"
          f"  Common Component/公共分量={is_common.sum()}/{len(W_vals)}")
    return is_common, W_vals, threshold


# ══════════════════════════════════════════════════════════════
# 第三幕：粗重构
# ══════════════════════════════════════════════════════════════

def coarse_reconstruct(nodes_A, nodes_B, is_common):
    """
    按Common Component/公共分量/Differential Component/差异分量叠加节点。
    WPD 正交性保证：粗Common Component/公共分量 + 粗Differential Component/差异分量 = 原始信号（数值误差 < 1e-10）。
    """
    N = len(nodes_A[0])
    cc_A = np.zeros(N); cc_B = np.zeros(N)
    cd_A = np.zeros(N); cd_B = np.zeros(N)
    for rA, rB, cmn in zip(nodes_A, nodes_B, is_common):
        if cmn:
            cc_A += rA; cc_B += rB
        else:
            cd_A += rA; cd_B += rB
    return cc_A, cc_B, cd_A, cd_B


# ══════════════════════════════════════════════════════════════
# 第四幕：复数小波 CWT（SST 基础）
# ══════════════════════════════════════════════════════════════

def _cwt_complex(signal, wavelet, n_scales):
    """复数小波 CWT，返回 (F×T) 复系数、伪频率数组、尺度数组。
    仅用于 FQI 变换（分布分析），不用于信号重构。"""
    N = len(signal)
    scales = np.geomspace(2, max(N // 2, 4), n_scales)
    coeffs, freqs = pywt.cwt(signal, scales, wavelet)
    return coeffs.astype(complex), freqs, scales


def _fft_split(signal, mask_1d, cwt_freqs):
    """
    FFT 域滤波重构：CWT 推导的掩模 mask(f_cwt) 映射到 FFT 线性频率轴，
    在频域做软乘法后 IFFT 得到两个互补分量。

    能量守恒性质（数学精确）：
      mask_fft + (1 - mask_fft) = 1  →  kept + compl = signal
      最大重构误差 < 2e-15（机器精度）

    与 CWT 软掩模语义一致：
      mask(f) 高 → 该频率保留在 kept 分量
      mask(f) 低 → 该频率保留在 compl 分量

    参数：
      signal    : 时域信号 (N,)
      mask_1d   : CWT 频率对应的软掩模 (F,)，值域 [0,1]
      cwt_freqs : CWT 伪频率数组 (F,)，与 mask_1d 对应
    返回：
      kept  : (N,)  掩模保留分量
      compl : (N,)  掩模互补分量
    """
    N = len(signal)
    S = np.fft.rfft(signal)
    fft_f = np.fft.rfftfreq(N)           # [0, 0.5]，线性频率（归一化）

    # CWT 伪频率按升序排列后插值到 FFT 频率轴
    # DC（f=0）在 CWT 范围之外，用最低 CWT 频率的掩模值外推（不硬编码为 0）。
    # 原因：若 DC 被强制为 mask=0，则信号均值永远回流到Common Component/公共分量，
    #       造成两孔均值差异主导 W(公共分量_A, 公共分量_B)，收缩验证永远失效。
    # 正确做法：差模路（cd）在低频差异大 → 掩模接近 1 → DC 留在 delta_diff；
    #           共模路（cc）在低频相似   → 掩模接近 0 → DC 留在 epsilon_comm。
    idx = np.argsort(cwt_freqs)
    f_sorted = cwt_freqs[idx]
    m_sorted = mask_1d[idx]
    fft_mask = np.interp(fft_f, f_sorted, m_sorted,
                         left=float(m_sorted[0]), right=float(m_sorted[-1]))

    kept  = np.fft.irfft(S * fft_mask,         N)
    compl = np.fft.irfft(S * (1.0 - fft_mask), N)
    return kept, compl


# ══════════════════════════════════════════════════════════════
# 第四·五幕：FQI 变换（核心创新）
# ══════════════════════════════════════════════════════════════

def fqi_transform(sst_matrix, n_quantiles, log_transform=False):
    """
    频率-分位图变换 (Frequency-Quantile Image, FQI)。

        FQI(f, q) = 频率 f 下幅值的第 q 分位数（逆CDF）

    输入 : sst_matrix (F×T) 复值或实值
    输出 : fqi        (F×Q) 实值，绝对物理幅值

    关键性质：
      · 时间轴 → 统一分位序 q∈[0,1]，信号长度无关
      · 两孔图像坐标系完全一致，可逐像素比较
      · 保留绝对幅值（不归一化）
      · 逐行 L1 距离 ≈ 1D Wasserstein 距离的黎曼和近似
        （n_quantiles 个分位点，n_quantiles→T 时收敛至精确值）
    """
    amp = np.abs(sst_matrix)          # (F, T)
    if log_transform:
        amp = np.log1p(amp)           # log(1+amp)，对大幅值加权
    F, T = amp.shape
    q_grid = np.linspace(0, 1, n_quantiles)
    t_grid = np.linspace(0, 1, T)
    fqi = np.zeros((F, n_quantiles))
    for f in range(F):
        fqi[f, :] = np.interp(q_grid, t_grid, np.sort(amp[f, :]))
    return fqi


def psd_curve(sst_matrix):
    """PSD(f) = mean(|SST(f,:)|²)，退化为 1D 频率曲线。"""
    return np.mean(np.abs(sst_matrix) ** 2, axis=1)


def compute_image_and_distance(sst_A, sst_B, mode, n_quantiles, log_transform):
    """
    计算图像表示和逐频率 Wasserstein 距离。

    FQI 模式：
      img_A, img_B : (F, Q)  绝对幅值矩阵
      W(f)         : (F,)    逐行 L1 ≈ 1D Wasserstein 近似（n_quantiles 分位点）

    PSD 模式（备选）：
      img_A, img_B : (F, 1)  功率列向量
      W(f)         : (F,)    归一化功率差
    """
    if mode == "FQI":
        img_A = fqi_transform(sst_A, n_quantiles, log_transform)
        img_B = fqi_transform(sst_B, n_quantiles, log_transform)
        W = np.mean(np.abs(img_A - img_B), axis=1)     # ≈ 1D Wasserstein（n_quantiles 分位近似）
    else:   # PSD
        pA = psd_curve(sst_A)
        pB = psd_curve(sst_B)
        img_A = pA[:, None]
        img_B = pB[:, None]
        W = np.abs(pA - pB) / (np.maximum(pA, pB) + 1e-15)
    return img_A, img_B, W


# ══════════════════════════════════════════════════════════════
# 第五幕：软掩模
# ══════════════════════════════════════════════════════════════

def make_soft_mask(W, mode, sigma_scale=1.0):
    """
    Wasserstein 距离 W(f) → 软掩模 mask(f)∈[0,1]。

    normalized : mask = W/max(W)，无超参数 ★推荐
    sigmoid    : mask = σ((W−μ)/(σ_w·scale))
    otsu_soft  : Otsu 阈值以上线性升至 1，以下线性降至 0
    """
    W = np.asarray(W, dtype=float)
    eps = 1e-15
    if mode == "normalized":
        return W / (W.max() + eps)

    elif mode == "sigmoid":
        mu = np.mean(W)
        sw = np.std(W) * sigma_scale + eps
        return 1.0 / (1.0 + np.exp(-(W - mu) / sw))

    elif mode == "otsu_soft":
        theta = _otsu_threshold(W)
        mask = np.zeros_like(W)
        above = W >= theta
        if above.any():
            wa = W[above]
            mask[above] = 0.5 + 0.5 * (wa - theta) / (wa.max() - theta + eps)
        below = ~above
        if below.any():
            mask[below] = 0.5 * W[below] / (theta + eps)
        return np.clip(mask, 0.0, 1.0)

    else:
        return W / (W.max() + eps)


# ══════════════════════════════════════════════════════════════
# 第六幕：双向细分解 + 四分量 FFT 重构
# ══════════════════════════════════════════════════════════════

def fine_decompose(cc_A, cc_B, cd_A, cd_B):
    """
    从粗Differential Component/差异分量提取真Differential Component/差异分量（δ_diff），从粗Common Component/公共分量提取漏网Differential Component/差异分量（ε_diff），
    剩余部分归还Common Component/公共分量池，最终合并。

    两阶段分离：
      ① CWT → FQI → W(f) → mask(f)   [分析阶段：推导掩模]
      ② FFT × mask_fft → IFFT          [重构阶段：精确能量守恒]

    设计动机：
      · CWT 不可逆（有限离散尺度下 iCWT 误差 30-60%，不可用于重构）
      · FFT 完全可逆（误差 < 2e-15）
      · 两者的频率轴通过线性插值对齐
      · CWT 掩模的物理意义（哪些频率是Differential Component/差异分量）不变

    返回字典：
      sst_diff_A/B, sst_comm_A/B   : CWT 系数矩阵 (F,T)  [仅供可视化]
      img_diff_A/B, img_comm_A/B   : FQI/PSD 图像
      W_diff, W_comm               : 逐频 Wasserstein 距离 (F,)
      mask_delta, mask_epsilon     : 软掩模 (F,) [CWT频率轴]
      delta_diff_A/B               : δ_diff  [FFT精确重构]
      delta_comm_A/B               : δ_common
      epsilon_diff_A/B             : ε_diff
      epsilon_comm_A/B             : ε_common
      final_comm_A/B               : 最终Common Component/公共分量
      final_diff_A/B               : 最终Differential Component/差异分量
      freqs                        : CWT 伪频率数组
    """
    wavelet   = SST_WAVELET
    n_scales  = SST_N_SCALES
    n_q       = FINE_N_QUANTILES
    mask_mode = SOFT_MASK_MODE
    log_t     = FQI_LOG_TRANSFORM
    sig_sc    = SIGMOID_SIGMA_SCALE

    # ── 第四幕：CWT 双路展开（仅用于 FQI 分析）──────
    print("      CWT 展开（粗Differential Component/差异分量 & 粗Common Component/公共分量，双路）...")
    sst_diff_A, freqs, _ = _cwt_complex(cd_A, wavelet, n_scales)
    sst_diff_B, _,     _ = _cwt_complex(cd_B, wavelet, n_scales)
    sst_comm_A, _,     _ = _cwt_complex(cc_A, wavelet, n_scales)
    sst_comm_B, _,     _ = _cwt_complex(cc_B, wavelet, n_scales)

    # ── 第四·五幕：FQI 变换 + 逐频 Wasserstein ──────
    print(f"      {IMAGE_TRANSFORM} 变换 (Q={n_q}, log={log_t})...")
    img_dA, img_dB, W_diff = compute_image_and_distance(
        sst_diff_A, sst_diff_B, IMAGE_TRANSFORM, n_q, log_t)
    img_cA, img_cB, W_comm = compute_image_and_distance(
        sst_comm_A, sst_comm_B, IMAGE_TRANSFORM, n_q, log_t)

    # ── 第五幕：双向软掩模（CWT 频率轴） ────────────
    print(f"      软掩模（{mask_mode}）...")
    mask_delta   = make_soft_mask(W_diff, mask_mode, sig_sc)
    mask_epsilon = make_soft_mask(W_comm, mask_mode, sig_sc)

    # ── 第六幕：四路 FFT 精确重构 ────────────────────
    # 掩模语义不变（高权重=该频率保留），重构方式从 iCWT 改为 FFT
    print("      四路 FFT 域滤波重构（机器精度）...")
    delta_diff_A,   delta_comm_A   = _fft_split(cd_A, mask_delta,   freqs)
    delta_diff_B,   delta_comm_B   = _fft_split(cd_B, mask_delta,   freqs)
    epsilon_diff_A, epsilon_comm_A = _fft_split(cc_A, mask_epsilon, freqs)
    epsilon_diff_B, epsilon_comm_B = _fft_split(cc_B, mask_epsilon, freqs)

    # ── 最终合并 ────────────────────────────────────
    final_comm_A = epsilon_comm_A + delta_comm_A
    final_comm_B = epsilon_comm_B + delta_comm_B
    final_diff_A = delta_diff_A   + epsilon_diff_A
    final_diff_B = delta_diff_B   + epsilon_diff_B

    return dict(
        sst_diff_A=sst_diff_A, sst_diff_B=sst_diff_B,
        sst_comm_A=sst_comm_A, sst_comm_B=sst_comm_B,
        img_diff_A=img_dA, img_diff_B=img_dB,
        img_comm_A=img_cA, img_comm_B=img_cB,
        W_diff=W_diff, W_comm=W_comm,
        mask_delta=mask_delta, mask_epsilon=mask_epsilon,
        delta_diff_A=delta_diff_A,   delta_comm_A=delta_comm_A,
        epsilon_diff_A=epsilon_diff_A, epsilon_comm_A=epsilon_comm_A,
        delta_diff_B=delta_diff_B,   delta_comm_B=delta_comm_B,
        epsilon_diff_B=epsilon_diff_B, epsilon_comm_B=epsilon_comm_B,
        final_comm_A=final_comm_A, final_comm_B=final_comm_B,
        final_diff_A=final_diff_A, final_diff_B=final_diff_B,
        freqs=freqs,
    )


# ══════════════════════════════════════════════════════════════
# 过程图 A / B / C / E / G / H / I / J / K
# ══════════════════════════════════════════════════════════════
# 每张图均包含：
#   标题行1 — 图的目的（这张图是干什么的）
#   标题行2 — 读法提示（怎么看这张图）
#   子图标题 — 该格的具体含义和数值
#   关键注释 — 用 ax.text / fig.text 在图上写明判断依据
# ══════════════════════════════════════════════════════════════

def _recon_err(orig, comm, diff):
    """相对平均绝对误差（%）"""
    n = min(len(orig), len(comm), len(diff))
    recon = comm[:n] + diff[:n]
    return float(np.abs(orig[:n] - recon).mean() /
                 (np.abs(orig[:n]).mean() + 1e-15) * 100)


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def run_line1(pair_data):
    """
    Step1 主流程：两级渐进分解。
    输入  : pair_data  → step0 输出
    输出  : pair_results
    """
    import sys
    print("\n" + "=" * 60)
    print("  Step1：两级渐进分解")
    print(f"  配置：WPD小波={WPD_WAVELET}  节点阈值={NODE_THRESH_MODE}")
    print(f"        CWT小波={SST_WAVELET}  图像变换={IMAGE_TRANSFORM}(Q={FINE_N_QUANTILES})")
    print(f"        掩模模式={SOFT_MASK_MODE}")
    print("  流程：[幕1]WPD频带分解 → [幕2]Wasserstein节点分类")
    print("        → [幕3]粗重构 → [幕4-6]SST+FQI+软掩模+FFT四分量重构")
    print("=" * 60)
    sys.stdout.flush()

    pair_results = {}

    for sig_type, pairs in pair_data.items():
        pair_results[sig_type] = {}
        print(f"\n{'─'*60}")
        print(f"  信号类型: {sig_type}  共 {len(pairs)} 对孔对待处理")
        print(f"{'─'*60}")
        sys.stdout.flush()

        # ── 底限值（COMM_BASELINE）：所有参与孔段的最小信号值 ──────
        if COMM_BASELINE:
            all_mins = [min(d["y_A"].min(), d["y_B"].min())
                        for d in pairs.values()]
            folder_min = float(min(all_mins))
            print(f"  [底限值] COMM_BASELINE=ON  folder_min={folder_min:.6f}")
            print(f"  [底限值] 所有信号分解前减去底限，分解后仅 comm 加回底限")
        else:
            folder_min = 0.0

        for (hA, hB), d in pairs.items():
            y_A     = d["y_A"]
            y_B     = d["y_B"]
            label_A = d["label_A"]
            label_B = d["label_B"]
            meta    = d["meta"]
            tag     = meta.get("tag", "")
            N       = len(y_A)

            print(f"\n  ┌─ {label_A}  vs  {label_B}")
            print(f"  │  孔对类型: {tag}")
            print(f"  │  信号长度: N={N}个样本点（两孔已截齐到相同长度）")
            print(f"  │  孔A信号: 均值={np.mean(y_A):.4f}  std={np.std(y_A):.4f}"
                  f"  范围=[{y_A.min():.4f}, {y_A.max():.4f}]")
            print(f"  │  孔B信号: 均值={np.mean(y_B):.4f}  std={np.std(y_B):.4f}"
                  f"  范围=[{y_B.min():.4f}, {y_B.max():.4f}]")

            # ── 去均值预处理（可选，由 DEMEAN_BEFORE_DECOMPOSE 控制）─────
            mean_A = float(np.mean(y_A))
            mean_B = float(np.mean(y_B))
            if DEMEAN_BEFORE_DECOMPOSE:
                print(f"  │  [去均值] 开关=ON → 分解前各自减去均值，分解后加回")
                print(f"  │           孔A均值={mean_A:.4f} 已减去；孔B均值={mean_B:.4f} 已减去")
                print(f"  │           效果：Otsu不再被DC节点主导，分离真正的频率级波动差异")
                y_A_proc = y_A - mean_A - folder_min
                y_B_proc = y_B - mean_B - folder_min
            else:
                y_A_proc = y_A - folder_min
                y_B_proc = y_B - folder_min
                if COMM_BASELINE:
                    print(f"  │  [底限] 孔A/B各减去 folder_min={folder_min:.4f}")
            sys.stdout.flush()

            pair_dir = make_pair_dir(OUTPUT_STEP1, sig_type, label_A, label_B)

            # ══ 第一幕：WPD 频带分解 ══════════════════════════
            level  = _adaptive_level(N)
            n_node = 2 ** level
            print(f"  │")
            print(f"  ├─ [幕1] WPD小波包分解")
            print(f"  │   目的：将信号正交分解为{n_node}个频带节点，每个节点对应一个频段的时域波形。")
            print(f"  │   参数：分解层数={level}（自适应，由N={N}和最小节点长度{WPD_MIN_NODE_LEN}决定）")
            print(f"  │         节点数={n_node}=2^{level}  小波基={WPD_WAVELET}")
            sys.stdout.flush()

            nodes_A, node_labels = wpd_reconstruct_nodes(y_A_proc, WPD_WAVELET, level)
            nodes_B, _           = wpd_reconstruct_nodes(y_B_proc, WPD_WAVELET, level)

            wpd_err_A = float(np.abs(y_A_proc - sum(nodes_A)).max())
            wpd_err_B = float(np.abs(y_B_proc - sum(nodes_B)).max())
            print(f"  │   WPD重构验证(孔A): 最大误差={wpd_err_A:.2e}  "
                  f"(孔B): 最大误差={wpd_err_B:.2e}")
            for sfx, err in [("A", wpd_err_A), ("B", wpd_err_B)]:
                if err < 1e-10:
                    print(f"  │   ✓ 孔{sfx}误差可忽略（正常<1e-10）。")
                else:
                    print(f"  │   ⚠ 孔{sfx}误差偏大({err:.2e})，请检查信号是否含NaN或异常值。")

            # ══ 第二幕：Wasserstein 节点分类 ══════════════════
            print(f"  │")
            print(f"  ├─ [幕2] Wasserstein距离节点分类")
            print(f"  │   目的：对每个频带节点，计算孔A/B幅值分布的Wasserstein距离W_i，")
            print(f"  │         用{NODE_THRESH_MODE}阈值分为Common Component/公共分量节点（W小=分布相似）和Differential Component/差异分量节点（W大=分布差异大）。")
            print(f"  │   优势：完全不依赖时间对齐——比较的是统计分布形状，非波形逐点匹配。")
            sys.stdout.flush()

            is_common, W_vals, threshold = classify_nodes_wasserstein(
                nodes_A, nodes_B, NODE_THRESH_MODE)

            n_common = int(is_common.sum())
            n_diff   = n_node - n_common
            print(f"  │   结果: Common Component/公共分量节点={n_common}/{n_node}，Differential Component/差异分量节点={n_diff}/{n_node}")
            print(f"  │   W距离: 最小={W_vals.min():.4f}  最大={W_vals.max():.4f}  中位数={np.median(W_vals):.4f}")
            print(f"  │   自动阈值（{NODE_THRESH_MODE}法）={threshold:.4f}")
            print(f"  │   解读：W距离越大=两孔该频段分布差异越大=受控制参数影响越强。")
            if n_common == n_node:
                print(f"  │   ⚠ 全部为Common Component/公共分量节点：两孔信号极度相似，Differential Component/差异分量能量将接近零。")
            elif n_diff == n_node:
                print(f"  │   ⚠ 全部为Differential Component/差异分量节点：两孔信号差异极大，请检查数据质量。")

            # ══ 第三幕：粗重构 ══════════════════════════════════
            print(f"  │")
            print(f"  ├─ [幕3] 粗重构（频带级）")
            print(f"  │   目的：将Common Component/公共分量节点叠加→粗Common Component/公共分量，Differential Component/差异分量节点叠加→粗Differential Component/差异分量。")
            print(f"  │   保证：WPD正交性 → 粗Common Component/公共分量+粗Differential Component/差异分量 = 原始信号（能量100%守恒）。")
            sys.stdout.flush()

            cc_A, cc_B, cd_A, cd_B = coarse_reconstruct(nodes_A, nodes_B, is_common)

            for sfx, orig, comm, diff in [("A", y_A_proc, cc_A, cd_A), ("B", y_B_proc, cc_B, cd_B)]:
                e_c = np.var(comm) / (np.var(orig) + 1e-15) * 100
                e_d = np.var(diff) / (np.var(orig) + 1e-15) * 100
                tot = e_c + e_d
                print(f"  │   孔{sfx}: 粗Common Component/公共分量方差比={e_c:.1f}%  粗Differential Component/差异分量方差比={e_d:.1f}%  合计={tot:.1f}%")
                if abs(tot - 100.0) > 5.0:
                    print(f"  │        ⚠ 合计≠100%（正常现象：var(comm+diff)=var(comm)+var(diff)+2·cov，"
                          f"Common Component/公共分量Differential Component/差异分量之间存在协方差，合计偏差{tot-100:.1f}%）")
            print(f"  │   注：此时是粗糙的频带级分解，下一步在频率级进一步精细化。")

            # ══ 第四-六幕：细分解 ══════════════════════════════
            print(f"  │")
            print(f"  ├─ [幕4-6] 频率级细分解（CWT→FQI→软掩模→FFT精确重构）")
            print(f"  │   CWT尺度数={SST_N_SCALES}  FQI分位点数={FINE_N_QUANTILES}  掩模={SOFT_MASK_MODE}")
            sys.stdout.flush()

            fine = fine_decompose(cc_A, cc_B, cd_A, cd_B)

            # ── 去均值模式：最终结果加回均值 ─────────────────────
            if DEMEAN_BEFORE_DECOMPOSE:
                # Common Component/公共分量 = 共同波动模式，加回各自孔均值后恢复物理量纲
                # Differential Component/差异分量 = 波动差异部分，不加均值（均值差异已在去均值步骤中移除，Differential Component/差异分量只含频率波动）
                fine["final_comm_A"] = fine["final_comm_A"] + mean_A
                fine["final_comm_B"] = fine["final_comm_B"] + mean_B

            # ── 底限值恢复：comm 加回 folder_min，diff 不加 ────────
            if COMM_BASELINE:
                fine["final_comm_A"] = fine["final_comm_A"] + folder_min
                fine["final_comm_B"] = fine["final_comm_B"] + folder_min

            err_A = _recon_err(y_A - folder_min - (mean_A if DEMEAN_BEFORE_DECOMPOSE else 0),
                               fine["final_comm_A"] - folder_min - (mean_A if DEMEAN_BEFORE_DECOMPOSE else 0),
                               fine["final_diff_A"])
            err_B = _recon_err(y_B - folder_min - (mean_B if DEMEAN_BEFORE_DECOMPOSE else 0),
                               fine["final_comm_B"] - folder_min - (mean_B if DEMEAN_BEFORE_DECOMPOSE else 0),
                               fine["final_diff_B"])

            # RMS（均方根）：信号的整体幅值水平，包含均值
            # 方差（var）：信号的波动量，不含均值，更能反映"频率内容"
            # 两者的关系：RMS² = mean² + var  （如果均值较大，RMS≈mean，方差才反映真实波动）
            diff_rms_A  = float(np.sqrt(np.mean(fine["final_diff_A"]**2)))
            comm_rms_A  = float(np.sqrt(np.mean(fine["final_comm_A"]**2)))
            diff_var_A  = float(np.var(fine["final_diff_A"]))
            comm_var_A  = float(np.var(fine["final_comm_A"]))
            orig_var_A  = float(np.var(y_A))
            # 用方差比（不含均值效应）更准确反映频率成分分配
            diff_var_pct = diff_var_A / (orig_var_A + 1e-15) * 100
            comm_var_pct = comm_var_A / (orig_var_A + 1e-15) * 100
            orig_rms_A  = float(np.sqrt(np.mean(y_A**2)))

            print(f"  │   最终重构误差: A={err_A:.6f}%  B={err_B:.6f}%")
            if max(err_A, err_B) < 0.001:
                print(f"  │   ✓ 重构误差为机器精度（FFT精确守恒，正常范围<0.001%）")
            else:
                print(f"  │   ⚠ 重构误差偏大，请检查是否有数据异常")
            print(f"  │   孔A Differential Component/差异分量RMS={diff_rms_A:.4f}  Common Component/公共分量RMS={comm_rms_A:.4f}  原始RMS={orig_rms_A:.4f}")
            print(f"  │       Differential Component/差异分量方差占原始方差={diff_var_pct:.1f}%  Common Component/公共分量方差占原始方差={comm_var_pct:.1f}%")
            print(f"  │   【注】方差比（上行）反映波动能量分配，RMS包含均值偏移，")
            print(f"  │        两者之和可能≠100%（因为diff和comm之间有交叉项），但diff+comm=原始信号精确成立。")
            if diff_var_pct < 5:
                print(f"  │   ⚠ Differential Component/差异分量波动能量极低(<5%)：两孔的波动模式高度相似，参数效应主要表现在均值偏移上。")
            if diff_var_pct > 95:
                print(f"  │   ⚠ Differential Component/差异分量波动能量极高(>95%)：两孔波动模式差异极大，或Common Component/公共分量仅代表极小的共同成分。")


            # ── 全量数据保存（学术论文用）───────────────────────
            if SAVE_INTERMEDIATE:
                import json

                # ① 信号数据 decomposed.npz —— 所有时域信号
                npz_path = os.path.join(pair_dir, "decomposed.npz")
                np.savez_compressed(npz_path,
                    # 原始信号（物理量）
                    y_A=y_A, y_B=y_B,
                    # 去均值后的信号（DEMEAN=True时与y_A不同，False时相同）
                    y_A_proc=y_A_proc, y_B_proc=y_B_proc,
                    # 各孔均值（DEMEAN=True时有意义；False时与y_A均值相同）
                    mean_A=np.array([mean_A]), mean_B=np.array([mean_B]),
                    # 粗分解（频带级）
                    coarse_comm_A=cc_A, coarse_comm_B=cc_B,
                    coarse_diff_A=cd_A, coarse_diff_B=cd_B,
                    # 四个细分量（频率级）
                    delta_diff_A=fine["delta_diff_A"],
                    delta_comm_A=fine["delta_comm_A"],
                    epsilon_diff_A=fine["epsilon_diff_A"],
                    epsilon_comm_A=fine["epsilon_comm_A"],
                    delta_diff_B=fine["delta_diff_B"],
                    delta_comm_B=fine["delta_comm_B"],
                    epsilon_diff_B=fine["epsilon_diff_B"],
                    epsilon_comm_B=fine["epsilon_comm_B"],
                    # 最终结果
                    final_comm_A=fine["final_comm_A"],
                    final_comm_B=fine["final_comm_B"],
                    final_diff_A=fine["final_diff_A"],
                    final_diff_B=fine["final_diff_B"],
                )

                # ② 频谱分析数据 spectral.npz —— 所有频率域中间结果（图F/G/H数据源）
                spectral_path = os.path.join(pair_dir, "spectral.npz")
                np.savez_compressed(spectral_path,
                    # CWT幅值矩阵（图F数据源）形状=(n_scales, N)
                    cwt_amp_diff_A=np.abs(fine["sst_diff_A"]),
                    cwt_amp_diff_B=np.abs(fine["sst_diff_B"]),
                    cwt_amp_comm_A=np.abs(fine["sst_comm_A"]),
                    cwt_amp_comm_B=np.abs(fine["sst_comm_B"]),
                    # CWT频率轴（与cwt_amp_*的行对应）
                    cwt_freqs=fine["freqs"],
                    # FQI图像（图G数据源）形状=(n_scales, n_quantiles)
                    fqi_diff_A=fine["img_diff_A"],
                    fqi_diff_B=fine["img_diff_B"],
                    fqi_comm_A=fine["img_comm_A"],
                    fqi_comm_B=fine["img_comm_B"],
                    # 逐频Wasserstein距离（图H上行数据源）
                    W_diff_f=fine["W_diff"],
                    W_comm_f=fine["W_comm"],
                    # 软掩模（图H下行数据源）
                    mask_delta=fine["mask_delta"],
                    mask_epsilon=fine["mask_epsilon"],
                )

                # ③ WPD节点数据 wpd_nodes.npz —— 图A/B/C数据源
                wpd_path = os.path.join(pair_dir, "wpd_nodes.npz")
                node_arr_A = np.stack(nodes_A)   # (n_nodes, N)
                node_arr_B = np.stack(nodes_B)
                np.savez_compressed(wpd_path,
                    nodes_A=node_arr_A,       # 每行=一个节点的时域重建信号（孔A）
                    nodes_B=node_arr_B,       # 每行=一个节点的时域重建信号（孔B）
                    node_W_vals=W_vals,        # 每节点的Wasserstein距离
                    node_is_common=is_common,  # 每节点的分类（True=Common Component/公共分量）
                    node_threshold=np.array([threshold]),
                    node_level=np.array([level]),
                    node_count=np.array([n_node]),
                )
                # 节点标签单独保存为文本（npz不支持字符串数组）
                labels_path = os.path.join(pair_dir, "wpd_node_labels.txt")
                with open(labels_path, 'w', encoding='utf-8') as f:
                    for lbl in node_labels:
                        f.write(lbl + '\n')

                # ④ 元数据 metadata.json —— 记录本次运行的所有参数配置
                import config as _cfg
                meta_path = os.path.join(pair_dir, "metadata.json")
                meta_dict = {
                    "pair": {
                        "label_A": label_A, "label_B": label_B, "tag": tag,
                        "hole_A": hA, "hole_B": hB,
                        "params_A": _cfg.CONTROL_PARAMS.get(hA, {}),
                        "params_B": _cfg.CONTROL_PARAMS.get(hB, {}),
                        "N_samples": N,
                        "mean_A": round(mean_A, 6),
                        "mean_B": round(mean_B, 6),
                        "is_single_factor": meta.get("is_single_factor"),
                        "diff_params": meta.get("diff_params", []),
                    },
                    "config": {
                        "DEMEAN_BEFORE_DECOMPOSE": _cfg.DEMEAN_BEFORE_DECOMPOSE,
                        "WPD_WAVELET": _cfg.WPD_WAVELET,
                        "WPD_LEVEL": level,
                        "WPD_N_NODES": n_node,
                        "NODE_THRESH_MODE": _cfg.NODE_THRESH_MODE,
                        "NODE_THRESHOLD": round(float(threshold), 6),
                        "N_COMMON_NODES": n_common, "N_DIFF_NODES": n_diff,
                        "SST_WAVELET": _cfg.SST_WAVELET,
                        "SST_N_SCALES": _cfg.SST_N_SCALES,
                        "IMAGE_TRANSFORM": _cfg.IMAGE_TRANSFORM,
                        "FINE_N_QUANTILES": _cfg.FINE_N_QUANTILES,
                        "FQI_LOG_TRANSFORM": _cfg.FQI_LOG_TRANSFORM,
                        "SOFT_MASK_MODE": _cfg.SOFT_MASK_MODE,
                    },
                    "metrics": {
                        "recon_err_A_pct": round(err_A, 8),
                        "recon_err_B_pct": round(err_B, 8),
                        "diff_var_pct_A": round(diff_var_pct, 2),
                        "comm_var_pct_A": round(comm_var_pct, 2),
                        "diff_std_A": round(float(np.std(fine["final_diff_A"])), 6),
                        "diff_std_B": round(float(np.std(fine["final_diff_B"])), 6),
                        "comm_std_A": round(float(np.std(fine["final_comm_A"])), 6),
                        "comm_std_B": round(float(np.std(fine["final_comm_B"])), 6),
                    },
                    "files": {
                        "decomposed.npz": "时域信号：原始/粗分解/四细分量/最终Common Component/公共分量Differential Component/差异分量",
                        "spectral.npz":   "频谱数据：CWT幅值/FQI图像/W(f)距离/软掩模（图F/G/H数据源）",
                        "wpd_nodes.npz":  "WPD节点：各节点时域重建/W距离/分类（图A/B/C数据源）",
                        "wpd_node_labels.txt": "WPD节点标签列表（与wpd_nodes.npz行对应）",
                        "metadata.json":  "本次运行配置参数与数值指标",
                    }
                }
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta_dict, f, ensure_ascii=False, indent=2)

                print(f"  │   decomposed.npz   → 时域信号（原始/粗分解/四细分量/最终）")
                print(f"  │   spectral.npz     → CWT幅值/FQI图像/W(f)/掩模（图F/G/H数据源）")
                print(f"  │   wpd_nodes.npz    → WPD节点重建/W距离/分类（图A/B/C数据源）")
                print(f"  │   metadata.json    → 本次运行配置与指标")

            print(f"  └─ {label_A} vs {label_B} 完成")
            sys.stdout.flush()

            pair_results[sig_type][(hA, hB)] = {
                "y_A": y_A, "y_B": y_B, "label_A": label_A, "label_B": label_B,
                "meta": meta, "tag": tag,
                "coarse_comm_A": cc_A, "coarse_comm_B": cc_B,
                "coarse_diff_A": cd_A, "coarse_diff_B": cd_B,
                "node_W": W_vals, "node_is_common": is_common,
                "node_labels": node_labels, "node_threshold": threshold,
                "delta_diff_A":   fine["delta_diff_A"],
                "delta_comm_A":   fine["delta_comm_A"],
                "epsilon_diff_A": fine["epsilon_diff_A"],
                "epsilon_comm_A": fine["epsilon_comm_A"],
                "delta_diff_B":   fine["delta_diff_B"],
                "delta_comm_B":   fine["delta_comm_B"],
                "epsilon_diff_B": fine["epsilon_diff_B"],
                "epsilon_comm_B": fine["epsilon_comm_B"],
                "final_comm_A": fine["final_comm_A"], "final_comm_B": fine["final_comm_B"],
                "final_diff_A": fine["final_diff_A"], "final_diff_B": fine["final_diff_B"],
                "img_diff_A": fine["img_diff_A"], "img_diff_B": fine["img_diff_B"],
                "img_comm_A": fine["img_comm_A"], "img_comm_B": fine["img_comm_B"],
                "W_diff": fine["W_diff"], "W_comm": fine["W_comm"],
                "mask_delta": fine["mask_delta"], "mask_epsilon": fine["mask_epsilon"],
                "freqs": fine["freqs"],
                "recon_err_A": err_A, "recon_err_B": err_B,
            }

        print(f"\n  ✓ [{sig_type.upper()}] 全部 {len(pairs)} 对处理完成")
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"  [Step1 全部完成]")
    print(f"  每对孔对输出：decomposed.npz / spectral.npz / wpd_nodes.npz / metadata.json")
    print(f"  输出目录: {OUTPUT_STEP1}/")
    print(f"{'='*60}")
    sys.stdout.flush()
    return pair_results


# ── 独立运行入口 ────────────────────────────────────────────────
if __name__ == "__main__":
    from step0_preprocess import run_preprocess
    _, pair_data = run_preprocess()
    run_line1(pair_data)