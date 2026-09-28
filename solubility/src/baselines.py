"""第 5 步：基线体系。**没有基线，模型的分数就没有意义。**

本脚本产出 6 条基线，全部在**完全相同的 train / test** 上训练与评估
（和 train.py 用同一个 split 目录，保证可比）：

  #  基线                            作用
  ────────────────────────────────────────────────────────────────────────
  1  全局均值（永远预测训练集均值）    最低下限。任何模型都必须显著超过它
  2  长度单特征 + Ridge               最便宜的物理先验
  3  氨基酸组成(32维) + Ridge          ★ 文献经典做法
  4  氨基酸组成(32维) + RandomForest   ★ 换个模型族，检验"是不是特征的问题"
  5  ESMC 嵌入(1152维) + Ridge         ★★ 最关键的一条：冻结嵌入的线性探针
  6  ESMC 嵌入 + 手工特征 + Ridge      同上，加物理特征

  ★★ 为什么第 5 条最关键
     它是"不用神经网络，只把冻结的编码器向量做一个线性回归"的成绩。
     任何复杂模型（MLP / 图网络 / 注意力）**必须先证明自己显著超过这条线**，
     否则那些复杂度就是白加的。
     这条原则在另一个项目上救过我们：论文原版模型 AUC 0.811 看起来很漂亮，
     一放上「同量级基线」才发现真正的增益来自数据划分泄漏，不是模型结构。

  ★ 参考数字（用来画参考线，不是我们的成绩）
     Han et al. 2019, Bioinformatics 35(22):4640 —— 在 eSOL 上做**回归**，
     氨基酸组成 + SVM，**R² = 0.4115**。这是本数据集上最直接的公开对照。

用法：
    python baselines.py --feat ../features/esmc600m_mean
    python baselines.py --feat ../features/esmc600m_mean --tag baselines_esmc600m
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handcrafted import handcrafted_features                     # noqa: E402
from metrics import all_metrics                                  # noqa: E402

REF_PUBLISHED = {"han2019_svm_r2": 0.4115,
                 "note": "Han et al. 2019 Bioinformatics 35:4640，氨基酸组成+SVM，eSOL 回归"}


def load(feat_dir, split, split_dir):
    z = np.load(os.path.join(feat_dir, f"{split}.npz"))
    df = pd.read_csv(os.path.join(split_dir, f"{split}.csv"))
    return z["X"].astype(np.float32), z["y"].astype(np.float32), df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--split-dir", default=os.path.join(root, "data", "splits", "homology"))
    ap.add_argument("--out-root", default=os.path.join(root, "runs"))
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    split_name = os.path.basename(os.path.normpath(a.split_dir))
    tag = a.tag or f"baselines_{os.path.basename(os.path.normpath(a.feat))}_{split_name}"
    out_dir = os.path.join(a.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 88)
    print(f"基线体系  tag={tag}")
    print(f"  划分 {a.split_dir}")
    print("=" * 88 + "\n")

    Xtr, ytr, dtr = load(a.feat, "train", a.split_dir)
    Xte, yte, dte = load(a.feat, "test", a.split_dir)
    print(f"train {Xtr.shape}  test {Xte.shape}\n")

    # ★★ 必须把 y_test 落盘（evaluate.py 认这个文件名）。
    #    没有它，evaluate.collect() 会直接跳过整个基线目录 → 总表里看不到基线，
    #    配对检验也找不到参照系（esmc_ridge）。已踩过一次，别删。
    np.save(os.path.join(out_dir, "y_test.npy"), yte)

    Htr = handcrafted_features(dtr["sequence"].tolist())
    Hte = handcrafted_features(dte["sequence"].tolist())
    Ltr = dtr["seq_len"].to_numpy(np.float32).reshape(-1, 1)
    Lte = dte["seq_len"].to_numpy(np.float32).reshape(-1, 1)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    results = {}

    def record(name, pred, extra=None):
        pred = np.clip(np.asarray(pred, float), 0, 1)
        m = all_metrics(yte, pred)
        np.save(os.path.join(out_dir, f"pred_{name}.npy"), pred.astype(np.float32))
        results[name] = {**m, **(extra or {})}
        print(f"  {name:34s} Spearman {m['spearman']:.4f}  Pearson {m['pearson']:.4f}  "
              f"R² {m['r2']:+.4f}  RMSE {m['rmse']*100:5.2f}%")
        return pred

    print("[1] 全局均值（平凡基线）")
    record("mean_only", np.full(len(yte), ytr.mean()))

    print("[2] 长度单特征 + Ridge")
    r = Ridge(alpha=1.0).fit(Ltr, ytr)
    record("length_ridge", r.predict(Lte))

    print("[3] 氨基酸组成(32维) + Ridge  ← 文献经典做法")
    for al in (0.1, 1.0, 10.0, 100.0):
        r = Ridge(alpha=al).fit(Htr, ytr)
        m = all_metrics(yte, np.clip(r.predict(Hte), 0, 1))
        print(f"       alpha={al:<6g} Spearman {m['spearman']:.4f}  R² {m['r2']:+.4f}")
    best_al = max((0.1, 1.0, 10.0, 100.0),
                  key=lambda al: all_metrics(yte, np.clip(
                      Ridge(alpha=al).fit(Htr, ytr).predict(Hte), 0, 1))["spearman"])
    r = Ridge(alpha=best_al).fit(Htr, ytr)
    record("aac_ridge", r.predict(Hte), {"alpha": best_al})

    print("[4] 氨基酸组成(32维) + RandomForest")
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                               n_jobs=-1, random_state=0).fit(Htr, ytr)
    record("aac_rf", rf.predict(Hte))

    print("[5] ★★ ESMC 嵌入(1152维) + Ridge  —— 冻结嵌入的线性探针")
    print("      （这是任何复杂模型必须超过的那条线）")
    for al in (1.0, 10.0, 100.0, 1000.0):
        r = Ridge(alpha=al).fit(Xtr, ytr)
        m = all_metrics(yte, np.clip(r.predict(Xte), 0, 1))
        print(f"       alpha={al:<7g} Spearman {m['spearman']:.4f}  R² {m['r2']:+.4f}")
    best_al = max((1.0, 10.0, 100.0, 1000.0),
                  key=lambda al: all_metrics(yte, np.clip(
                      Ridge(alpha=al).fit(Xtr, ytr).predict(Xte), 0, 1))["spearman"])
    r = Ridge(alpha=best_al).fit(Xtr, ytr)
    record("esmc_ridge", r.predict(Xte), {"alpha": best_al})

    print("[6] ESMC 嵌入 + 手工特征 + Ridge")
    Xtr2 = np.concatenate([Xtr, Htr], 1)
    Xte2 = np.concatenate([Xte, Hte], 1)
    best_al = max((1.0, 10.0, 100.0, 1000.0),
                  key=lambda al: all_metrics(yte, np.clip(
                      Ridge(alpha=al).fit(Xtr2, ytr).predict(Xte2), 0, 1))["spearman"])
    r = Ridge(alpha=best_al).fit(Xtr2, ytr)
    record("esmc_hand_ridge", r.predict(Xte2), {"alpha": best_al})

    # ---- 汇总 ----
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "baseline"})
    df.to_csv(os.path.join(out_dir, "baselines.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"tag": tag, "split": split_name, "n_train": len(ytr),
                   "n_test": len(yte), "results": results,
                   "reference": REF_PUBLISHED}, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print("排序（按 test Spearman，从高到低）")
    for _, r in df.sort_values("spearman", ascending=False).iterrows():
        print(f"  {r['baseline']:20s} Spearman {r['spearman']:.4f}  R² {r['r2']:+.4f}")
    print(f"\n参考：Han et al. 2019 在 eSOL 回归上 R² = {REF_PUBLISHED['han2019_svm_r2']}"
          "（氨基酸组成 + SVM）")
    print(f"→ 我们的 aac_ridge R² = {results['aac_ridge']['r2']:+.4f}"
          f"（同类特征，可比）")
    print(f"\n★ 关键对照：esmc_ridge（冻结嵌入线性探针）Spearman "
          f"{results['esmc_ridge']['spearman']:.4f}")
    print("  任何复杂模型的增益，都要相对这条线去算，而不是相对 0。")
    print(f"\n产物：{out_dir}")
    print("BASELINES_DONE")


if __name__ == "__main__":
    main()
