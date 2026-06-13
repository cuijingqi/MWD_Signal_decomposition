# ══════════════════════════════════════════════════════════════════════
# predict_lithology_5types.py
# 5种岩性分类实验：验证公共分量比原始信号更好地表达岩性关系
#
# 数据：5种岩性文件夹，目标 = 岩性（5类）
# 策略：每孔在每个分解阶段取跨配对均值 → 特征向量
#       按类分层80/20切分，训练集上5折GridSearchCV调参
#       重复10次随机切分取均值和标准差
#       比较 original / final_diff / final_comm 三个阶段
#       同时比较 pressure / torque / pressure+torque 三种特征组合
# ══════════════════════════════════════════════════════════════════════

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT_DIR  = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── 5 种岩性配置 ──────────────────────────────────────────────────────
ROCK_TYPES = [
    {"name": "Sandstone A", "folder": "SandstoneA", "label": 0},
    {"name": "Sandstone B", "folder": "SandstoneB", "label": 1},
    {"name": "Sandstone C", "folder": "SandstoneC", "label": 2},
    {"name": "Granite",     "folder": "Granite",    "label": 3},
    {"name": "Limestone",   "folder": "Limestone",  "label": 4},
]
LABEL_NAMES = {r["label"]: r["name"] for r in ROCK_TYPES}

# ── 10 个精选特征 ─────────────────────────────────────────────────────
ALL_SIGNAL_FEATURES = [
    # ── 振幅 ──
    "mean", "rms", "peak_to_peak",
    # ── 分布 ──
    "std", "variance", "iqr",
    # ── 频谱 ──
    "dominant_freq", "spectral_kurtosis", "spectral_entropy",
    # ── 时序 ──
    "sample_entropy",
]

STAGES_OF_INTEREST = ["original", "final_diff", "final_comm"]

# ── 6 个分类器 ─────────────────────────────────────────
CLASSIFIERS = {
    # ── 线性/判别 ──
    "LDA": (
        LinearDiscriminantAnalysis(solver="lsqr"),
        {"clf__shrinkage": [None, "auto", 0.1, 0.3, 0.5, 0.7]}
    ),
    # ── 近邻 ──
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [1, 3, 5, 7],
         "clf__weights":     ["uniform", "distance"]}
    ),
    # ── 核方法 ──
    "SVM-RBF": (
        SVC(kernel="rbf", random_state=42),
        {"clf__C":     [0.1, 1, 10, 100],
         "clf__gamma": ["scale", "auto", 0.01, 0.1]}
    ),
    # ── 树集成 ──
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {"clf__n_estimators": [50, 100, 200],
         "clf__max_depth":    [3, 5, 10, None],
         "clf__max_features": ["sqrt", "log2"]}
    ),
    # ── 梯度提升 ──
    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
        {"clf__n_estimators": [100, 200],
         "clf__max_depth":    [3, 5, 7],
         "clf__learning_rate":[0.01, 0.1]}
    ),
    "LightGBM": (
        LGBMClassifier(random_state=42, verbose=-1),
        {"clf__n_estimators": [100, 200],
         "clf__max_depth":    [3, 5, 7],
         "clf__learning_rate":[0.01, 0.1]}
    ),
}


# ══════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════════════════════════

def load_all_features():
    """
    加载所有岩性特征：
    - original:    按 (hole, signal) 去重，每孔段1条
    - final_comm/final_diff: 保留每个 (pair, hole) 为独立样本
    """
    records = []
    for rt in ROCK_TYPES:
        csv_path = DATA_DIR / rt["folder"] / "curve_features_per_signal.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["stage"].isin(STAGES_OF_INTEREST)].copy()
        feat_cols = [f for f in ALL_SIGNAL_FEATURES if f in df.columns]

        # original：按 (hole, signal) 去重，取第一条（所有重复行特征相同）
        orig_df = df[df["stage"] == "original"].drop_duplicates(
            subset=["hole", "signal"], keep="first")
        for _, row in orig_df.iterrows():
            rec = {
                "sample_id":  f"{rt['folder']}__{row['hole']}__{row['signal']}__original",
                "hole_id":    row["hole"],
                "pair":       "original",
                "rock_label": rt["label"],
                "rock_name":  rt["name"],
                "stage":      "original",
                "signal":     row["signal"],
            }
            for f in feat_cols:
                rec[f] = row[f]
            records.append(rec)

        # final_comm / final_diff：每个 (pair, hole) 独立保留
        for stage in ["final_diff", "final_comm"]:
            stage_df = df[df["stage"] == stage]
            for _, row in stage_df.iterrows():
                sample_id = f"{rt['folder']}__{row['pair']}__{row['hole']}__{stage}"
                rec = {
                    "sample_id":  sample_id,
                    "hole_id":    row["hole"],
                    "pair":       row["pair"],
                    "rock_label": rt["label"],
                    "rock_name":  rt["name"],
                    "stage":      stage,
                    "signal":     row["signal"],
                }
                for f in feat_cols:
                    rec[f] = row[f]
                records.append(rec)

    return pd.DataFrame(records)


def pivot_to_matrix(df, stage, signals=("pressure", "torque")):
    """
    将每个 (sample_id, stage, signal) 转为样本矩阵。
    sample_id = pair × hole，每个配对产生的共模/差模各自独立。
    多信号时以 (sample_id去signal后缀) 做 inner join。
    """
    feat_cols = [f for f in ALL_SIGNAL_FEATURES if f in df.columns]
    stage_df  = df[df["stage"] == stage].copy()

    if len(signals) == 1:
        sig = signals[0]
        sub = stage_df[stage_df["signal"] == sig].copy()
        sub = sub.dropna(subset=feat_cols).reset_index(drop=True)
        X     = np.nan_to_num(sub[feat_cols].values.astype(float), nan=0.0)
        y     = sub["rock_label"].values.astype(int)
        holes = sub["hole_id"].tolist()
        sids  = sub["sample_id"].tolist()
        return X, y, holes, sids

    # 多信号：pressure + torque，以 (pair, hole_id) 做 inner join
    parts = {}
    for sig in signals:
        sub = stage_df[stage_df["signal"] == sig].copy()
        sub = sub.set_index(["pair", "hole_id"])
        sub = sub[feat_cols + ["rock_label", "rock_name"]].add_prefix(f"{sig}_")
        parts[sig] = sub

    merged = parts[signals[0]]
    for sig in signals[1:]:
        other = parts[sig].drop(
            columns=[f"{sig}_rock_label", f"{sig}_rock_name"], errors="ignore")
        merged = merged.join(other, how="inner")

    merged = merged.reset_index()
    feat_merged = [c for c in merged.columns
                   if c not in ("pair", "hole_id",
                                f"{signals[0]}_rock_label",
                                f"{signals[0]}_rock_name")]
    X     = np.nan_to_num(merged[feat_merged].values.astype(float), nan=0.0)
    y     = merged[f"{signals[0]}_rock_label"].values.astype(int)
    holes = merged["hole_id"].tolist()
    sids  = (merged["pair"] + "__" + merged["hole_id"]).tolist()
    return X, y, holes, sids
    holes = merged["hole"].tolist()
    sids  = (merged["pair"] + "__" + merged["hole"]).tolist()
    return X, y, holes, sids


# ══════════════════════════════════════════════════════════════════════
# 2. 分层 80/20 分类实验
#    每个岩性独立 80/20 切分，再合并为全局训练/测试集
#    超参数：在训练集上 5折 GridSearchCV
#    重复 10 次随机切分取均值 ± 标准差
# ══════════════════════════════════════════════════════════════════════

def stratified_split_per_class(y, test_size=0.2, random_state=0):
    """每个类独立做 80/20 切分，保证每类在测试集里都有样本。"""
    rng = np.random.RandomState(random_state)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * test_size))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())
    return np.array(train_idx), np.array(test_idx)


def run_classification(X, y, holes=None, n_repeats=10):
    X = np.nan_to_num(X, nan=0.0)
    results = {}

    for clf_name, (base_clf, param_grid) in CLASSIFIERS.items():
        acc_list  = []
        f1_list   = []
        n_tested  = np.zeros(len(y), dtype=int)
        n_correct = np.zeros(len(y), dtype=int)
        last_pred = np.full(len(y), -1, dtype=int)

        for seed in range(1, 1 + n_repeats):
            train_idx, test_idx = stratified_split_per_class(
                y, test_size=0.2, random_state=seed)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            n_splits = min(5, np.bincount(y_train).min())
            if n_splits < 2:
                pipe = Pipeline([("scaler", StandardScaler()),
                                 ("clf",    clone(base_clf))])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test).ravel()
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed)
                pipe = Pipeline([("scaler", StandardScaler()),
                                 ("clf",    clone(base_clf))])
                gs = GridSearchCV(pipe, param_grid, cv=cv,
                                  scoring="accuracy", refit=True, n_jobs=1)
                gs.fit(X_train, y_train)
                preds = gs.predict(X_test).ravel()

            acc_list.append(accuracy_score(y_test, preds))
            f1_list.append(f1_score(y_test, preds, average="macro", zero_division=0))
            n_tested[test_idx]  += 1
            n_correct[test_idx] += (preds == y[test_idx]).astype(int)
            last_pred[test_idx]  = preds

        mean_acc  = float(np.mean(acc_list))
        std_acc   = float(np.std(acc_list))
        mean_f1   = float(np.mean(f1_list))
        std_f1    = float(np.std(f1_list))
        miscls    = np.where(n_tested > 0,
                             1.0 - n_correct / np.maximum(n_tested, 1),
                             np.nan)
        results[clf_name] = {
            "acc":      mean_acc,
            "std":      std_acc,
            "f1":       mean_f1,
            "f1_std":   std_f1,
            "preds":    last_pred,
            "n_tested": n_tested,
            "n_correct": n_correct,
            "miscls":   miscls,
        }
        print(f"    {clf_name:14s}  acc={mean_acc:.3f} ± {std_acc:.3f}  f1={mean_f1:.3f} ± {std_f1:.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════
# 3. 特征重要性（ANOVA F 值）
# ══════════════════════════════════════════════════════════════════════

def feature_importance(X, y, feature_names):
    X = np.nan_to_num(X, nan=0.0)
    F, pval = f_classif(X, y)
    fi = pd.DataFrame({"feature": feature_names, "F": F, "pval": pval})
    return fi.sort_values("F", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("5种岩性分类实验（10次重复 80/20切分 + 5折GridSearchCV）")
    print("=" * 65)

    print("\n[1] 加载特征数据 ...")
    df = load_all_features()
    for rt in ROCK_TYPES:
        n = (df["rock_label"] == rt["label"]).sum()
        print(f"    {rt['name']:14s}: {n} 条记录")

    # 信号组合配置
    SIG_CONFIGS = {
        "pressure":       ("pressure",),
        "torque":         ("torque",),
        "pressure+torque": ("pressure", "torque"),
    }

    all_acc_rows = []
    all_detail   = {}

    for sig_key, signals in SIG_CONFIGS.items():
        print(f"\n{'─'*65}")
        print(f"  信号: {sig_key}")
        print(f"{'─'*65}")

        stage_data = {}
        for stage in STAGES_OF_INTEREST:
            X, y, holes, sids = pivot_to_matrix(df, stage, signals)
            stage_data[stage] = {"X": X, "y": y, "holes": holes, "sids": sids}
            dist = np.bincount(y, minlength=len(ROCK_TYPES)).tolist()
            print(f"    [{stage}] n={len(y)}, 分布={dist}")

        clf_results = {}
        for stage in STAGES_OF_INTEREST:
            print(f"\n  [2] 分类 — {stage}")
            d = stage_data[stage]
            clf_results[stage] = run_classification(d["X"], d["y"], d["holes"])

        # 精度汇总
        print(f"\n  [3] 精度汇总 — {sig_key}")
        clf_names = list(CLASSIFIERS.keys())
        header = f"  {'阶段':12s}" + "".join(f"  {n:14s}" for n in clf_names)
        print(header)
        for stage in STAGES_OF_INTEREST:
            row = f"  {stage:12s}"
            for n in clf_names:
                row += f"  {clf_results[stage][n]['acc']:.3f}±{clf_results[stage][n]['std']:.3f}  "
            print(row)

        # 收集精度
        for stage in STAGES_OF_INTEREST:
            for n in clf_names:
                all_acc_rows.append({
                    "signal":    sig_key,
                    "stage":     stage,
                    "model":     n,
                    "accuracy":  clf_results[stage][n]["acc"],
                    "acc_std":   clf_results[stage][n]["std"],
                    "f1_macro":  clf_results[stage][n]["f1"],
                    "f1_std":    clf_results[stage][n]["f1_std"],
                    "n_samples": len(stage_data[stage]["y"]),
                })

        # 逐孔详情
        for stage in STAGES_OF_INTEREST:
            d = stage_data[stage]
            detail = pd.DataFrame({
                "sample_id":  d["sids"],
                "hole_id":    d["holes"],
                "true_label": d["y"],
                "true_name":  [LABEL_NAMES[lb] for lb in d["y"]],
            })
            for n in clf_names:
                detail[f"pred_{n}"]     = clf_results[stage][n]["preds"]
                detail[f"n_tested_{n}"] = clf_results[stage][n]["n_tested"]
                detail[f"n_correct_{n}"]= clf_results[stage][n]["n_correct"]
                detail[f"miscls_{n}"]   = clf_results[stage][n]["miscls"]
            key = f"{sig_key}__{stage}"
            all_detail[key] = detail

    # ── 保存结果 ─────────────────────────────────────────────────────
    acc_df = pd.DataFrame(all_acc_rows)
    acc_df.to_csv(OUT_DIR / "accuracy_5types.csv",
                  index=False, encoding="utf-8-sig")

    for key, detail in all_detail.items():
        detail.to_csv(OUT_DIR / f"detail_5types_{key}.csv",
                      index=False, encoding="utf-8-sig")

    # 最佳结果快速查看
    print(f"\n{'='*65}")
    print("最佳精度（按信号×阶段 max over 所有分类器）")
    print(f"{'='*65}")
    best = (acc_df.groupby(["signal", "stage"])["accuracy"]
            .max().unstack("stage")[STAGES_OF_INTEREST])
    print(best.to_string())

    print(f"\n结果已保存至 {OUT_DIR}")
    print("=" * 65)

    return acc_df


if __name__ == "__main__":
    acc_df = main()
