"""第 6 步：汇总评估 —— 把所有实验拉成一张表，并做**配对检验**。

它会自动找齐 runs/ 下所有跑过的实验（含 baselines_*），对每个实验：
  1. 算 test 上的全套指标 + 独立 bootstrap 95% CI
  2. 与一个**参照系**做配对 bootstrap（默认参照 = 冻结嵌入线性探针 esmc_ridge）
  3. 如果同一配置在 random 与 homology 两套划分上都跑过，额外算出"泄漏的代价"

★ 为什么必须用配对检验而不是比两个 CI
    见 metrics.py 顶部说明。简言之：两个模型在同一批测试样本上误差相关，
    "两个 CI 有重叠 ⇒ 不显著"是错误推理，会把显著性结论报反。

★ 输出
    runs/_summary/all_experiments.csv    每个实验一行的总表
    runs/_summary/paired_vs_ref.csv      配对检验结果
    runs/_summary/leakage_cost.csv       泄漏代价（random − homology）
    runs/_summary/REPORT.md              人可读的报告骨架

用法：
    python evaluate.py
    python evaluate.py --ref esmc_ridge
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import all_metrics, bootstrap_ci, paired_bootstrap   # noqa: E402


def collect(runs_root):
    """扫出所有有 ensemble 预测的实验。"""
    found = []
    for name in sorted(os.listdir(runs_root)):
        d = os.path.join(runs_root, name)
        if not os.path.isdir(d) or name.startswith("_"):
            continue
        yp = os.path.join(d, "y_test.npy")
        if not os.path.exists(yp):
            continue
        # 预测文件：train.py 用 ensemble_test_pred.npy；baselines.py 用 pred_<name>.npy
        preds = {}
        ep = os.path.join(d, "ensemble_test_pred.npy")
        if os.path.exists(ep):
            preds["ensemble"] = np.load(ep)
        for f in os.listdir(d):
            if f.startswith("pred_") and f.endswith(".npy"):
                preds[f[5:-4]] = np.load(os.path.join(d, f))
        if not preds:
            continue
        y = np.load(yp)
        found.append(dict(name=name, dir=d, y=y, preds=preds))
    return found


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(root, "runs"))
    ap.add_argument("--ref", default="esmc_ridge",
                    help="配对检验的参照系（默认冻结嵌入线性探针）")
    ap.add_argument("--pairwise", action=argparse.BooleanOptionalAction, default=True,
                    help="除与参照系比之外，再做候选实验之间两两配对（默认开）")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    out_dir = os.path.join(a.runs, "_summary")
    os.makedirs(out_dir, exist_ok=True)

    exps = collect(a.runs)
    if not exps:
        raise SystemExit(f"在 {a.runs} 下没找到任何预测文件 —— 先跑 train.py / baselines.py")

    print("=" * 100)
    print(f"汇总评估：{len(exps)} 个实验目录")
    print("=" * 100)

    # ---------------- 每个实验：指标 + CI ----------------
    rows = []
    for e in exps:
        for pname, p in e["preds"].items():
            if len(p) != len(e["y"]):
                print(f"  [跳过] {e['name']}/{pname} 长度不符 {len(p)} vs {len(e['y'])}")
                continue
            m = all_metrics(e["y"], p)
            lo, hi = bootstrap_ci(e["y"], p, "spearman", a.n_boot)
            rows.append(dict(experiment=e["name"], pred=pname, n=len(p),
                             spearman=m["spearman"],
                             spearman_lo=lo, spearman_hi=hi,
                             pearson=m["pearson"], r2=m["r2"],
                             rmse_pct=m["rmse"] * 100, mae_pct=m["mae"] * 100))
    tab = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    tab.to_csv(os.path.join(out_dir, "all_experiments.csv"), index=False)

    print("\n" + "-" * 100)
    print("全部结果（按 Spearman 降序；CI 是独立 bootstrap，**不要用它判显著性**）")
    print("-" * 100)
    print(f"{'实验':40s} {'预测':14s} {'Spearman':>9s}  {'95% CI':>20s}  {'R²':>8s}  {'RMSE%':>7s}")
    for _, r in tab.iterrows():
        print(f"{r['experiment'][:40]:40s} {r['pred'][:14]:14s} "
              f"{r['spearman']:9.4f}  [{r['spearman_lo']:.4f},{r['spearman_hi']:.4f}]  "
              f"{r['r2']:+8.4f}  {r['rmse_pct']:7.2f}")

    # ---------------- 配对检验 ----------------
    # ★★★ 关键前提：配对 bootstrap 要求两个模型吃**同一批测试样本**。
    #   本项目同时跑 random 与 homology 两套划分，两套的测试集**大小相同（都 316）但样本不同**，
    #   所以"长度相等就配对"是错的 —— 那样会把两个不同测试集上的分数相减，结果没有意义。
    #   做法：按「测试集标签逐元素完全相同」自动分组（同划分 ⇒ y_test 必然一字不差），
    #        只在组内配对；每组各用自己那份 esmc_ridge 当参照。
    print("\n" + "-" * 100)
    print(f"配对 bootstrap（每组各自的参照系 = {a.ref}）")
    print("判据：差值的 95%CI **跨过 0 = 不显著**")
    print("分组依据：测试集标签逐元素相同（= 同一套划分）")
    print("-" * 100)

    groups = {}
    for e in exps:
        groups.setdefault(e["y"].tobytes(), []).append(e)

    paired_rows = []
    if not a.pairwise:
        print("  （--no-pairwise：只与参照系比）")
    for gi, members in enumerate(groups.values(), 1):
        y = members[0]["y"]
        items = []
        nan_items = []
        for e in members:
            for pname, p in e["preds"].items():
                if len(p) != len(y):
                    continue
                label = f"{e['name']}/{pname}"
                # ★ 常数预测（如全局均值基线）的 Spearman 是 nan —— 秩相关无定义。
                #   放进配对检验会得到 Δ=nan 的垃圾行，必须先剔掉。
                if not np.isfinite(all_metrics(y, p)["spearman"]):
                    nan_items.append(label)
                    continue
                items.append((label, p))
        if nan_items:
            print(f"  [跳过] 预测为常数、指标 nan，不参与配对：{', '.join(nan_items)}")
        if not items:
            continue
        print(f"\n【组 {gi}】测试集 n={len(y)}，实验："
              f"{', '.join(e['name'] for e in members)}")
        ref = next(((l, p) for l, p in items if l.rsplit("/", 1)[-1] == a.ref), None)
        if ref is None:
            print(f"  [警告] 本组没有 {a.ref}，跳过（基线必须先跑，并写出 y_test.npy / pred_{a.ref}.npy）")
            continue
        rl, rp = ref
        print(f"  参照：{rl}   Spearman {all_metrics(y, rp)['spearman']:.4f}")

        pairs = [(l, p, rl, rp) for l, p in items if l != rl]        # 候选 vs 参照
        if a.pairwise:                                              # 候选之间两两
            others = [(l, p) for l, p in items if l != rl]
            pairs += [(l1, p1, l2, p2) for (l1, p1), (l2, p2) in itertools.combinations(others, 2)]

        for cand, pc, ctrl, pb in pairs:
            diff, lo, hi, pv = paired_bootstrap(y, pc, pb, "spearman", a.n_boot)
            significant = (lo > 0 or hi < 0)
            verdict = "显著优于" if lo > 0 else ("显著劣于" if hi < 0 else "无显著差异")
            paired_rows.append(dict(group=gi, n_test=len(y), candidate=cand, control=ctrl,
                                    delta=diff, ci_lo=lo, ci_hi=hi, p=pv,
                                    significant=significant, verdict=verdict))
            print(f"  {cand[:44]:44s} vs {ctrl[:34]:34s} "
                  f"Δ={diff:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={pv:.3f}  {verdict}")

    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(
            os.path.join(out_dir, "paired_vs_ref.csv"), index=False)
    else:
        print("\n  [提示] 没产出任何配对检验行 —— 检查基线是否跑过、参照名是否为"
              f" {a.ref}，以及是否在 all_experiments 里。")

    # ---------------- 泄漏代价 ----------------
    print("\n" + "-" * 100)
    print("泄漏代价：同一配置在 random 与 homology 两套划分上的差距")
    print("-" * 100)
    lk = []
    for e in exps:
        for pname, p in e["preds"].items():
            if "_random" in e["name"]:
                twin = e["name"].replace("_random", "_homology")
                if os.path.isdir(os.path.join(a.runs, twin)):
                    for f in os.listdir(os.path.join(a.runs, twin)):
                        if f == f"pred_{pname}.npy" or f == "ensemble_test_pred.npy":
                            tp = np.load(os.path.join(a.runs, twin, f))
                            ty = np.load(os.path.join(a.runs, twin, "y_test.npy"))
                            if len(tp) == len(p):
                                # 两套划分的测试集样本不同 ⇒ 标签数组必须不同；
                                # 若一字不差，说明其实还是同一套划分，"泄漏代价"没有意义。
                                if np.array_equal(ty, e["y"]):
                                    print(f"  [跳过] {e['name']}: 与 {twin} 的测试集标签相同，"
                                          f"不是两套划分")
                                    continue
                                s1 = all_metrics(e["y"], p)["spearman"]
                                s2 = all_metrics(ty, tp)["spearman"]
                                lk.append(dict(config=e["name"].replace("_random", ""),
                                               pred=pname,
                                               spearman_random=s1,
                                               spearman_homology=s2,
                                               cost=s1 - s2))
                                print(f"  {e['name'][:40]:40s} random {s1:.4f} → "
                                      f"homology {s2:.4f}  **虚高 {s1-s2:+.4f}**")
    if lk:
        pd.DataFrame(lk).to_csv(os.path.join(out_dir, "leakage_cost.csv"), index=False)
    else:
        print("  （没有成对的 random/homology 实验 —— 用同名 tag 加后缀 _random / _homology 跑两遍）")

    print(f"\n汇总产物：{out_dir}")
    print("EVALUATE_DONE")


if __name__ == "__main__":
    main()
