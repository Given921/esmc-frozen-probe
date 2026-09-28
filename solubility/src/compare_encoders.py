#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编码器规模消融 —— 把不同编码器上的同名实验并排比较。

要回答的问题
    分数到底来自「预训练表示的规模」，还是来自「头部设计」？
    600M 上的既有结论是「MLP 未显著超过线性探针」⇒ 分数几乎全来自表示。
    换一个更小的编码器，如果差距也不大，就说明"规模不敏感"。

对照设计（唯一变量 = 编码器）
    mlp_<enc>_hand_homology   头部配置逐字一致（--hand/300轮/sigmoid/3种子）
    baselines_<enc>_homology  同样的基线体系，含线性探针（= 尺子）

★ 配对检验的前提：两个模型必须落在**同一批测试样本、同一顺序**上。
  本脚本会显式断言两次实验的 y_test 完全一致，不一致直接报错而不是默默算错。

用法（项目根目录下）
    python src/compare_encoders.py
    python src/compare_encoders.py --runs runs --encs esmc600m esmc300m
    python src/compare_encoders.py --split homology --out ../output/编码器消融.csv
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import spearman, all_metrics, paired_bootstrap          # noqa: E402

ENCLABEL = {
    "esmc300m": "ESMC-300M (960d)",
    "esmc600m": "ESMC-600M (1152d)",
    "esmc6b": "ESMC-6B (2560d)",
}

# 要并排比较的实验（按"头部配置"分组，每组内的差异只应来自编码器）
CONFIGS = [
    ("mlp_{e}_hand_{s}", "ensemble", "MLP + 手工特征（3 种子集成）"),
    ("mlp_{e}_{s}", "ensemble", "MLP（仅编码器向量）"),
]

# 每个实验结果目录里的关键预测文件
PRED_FILES = {
    "ensemble": "ensemble_test_pred.npy",
    "seed0": "seed0/test_pred.npy",
}


def _np(p):
    return np.load(p) if os.path.exists(p) else None


def load_exp(runs, tag, pred_key):
    """读一个实验的 (y, pred, 单种子列表)。缺文件返回 None。"""
    d = os.path.join(runs, tag)
    y = _np(os.path.join(d, "y_test.npy"))
    pf = PRED_FILES.get(pred_key, pred_key)
    pr = _np(os.path.join(d, pf))
    if y is None or pr is None:
        return None
    singles = []
    sj = os.path.join(d, "summary.json")
    if os.path.exists(sj):
        with open(sj, encoding="utf-8") as fh:
            meta = json.load(fh)
        singles = list(meta.get("single_test_spearman") or [])
    return {"y": y, "pred": pr, "singles": singles, "dir": d}


def load_base(runs, tag, which="esmc_ridge"):
    """读基线（如线性探针 = 尺子）的 (y, pred)。"""
    d = os.path.join(runs, tag)
    y = _np(os.path.join(d, "y_test.npy"))
    pr = _np(os.path.join(d, f"pred_{which}.npy"))
    return {"y": y, "pred": pr, "singles": [], "dir": d} if (y is not None and pr is not None) else None


def s(x):
    return "%.4f" % x if x == x else "nan"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(root, "runs"))
    ap.add_argument("--split", default="homology")
    # ★ 默认自动发现：以前写死 ["esmc300m","esmc600m"]，加进 6B 后必须显式传参，
    #   忘传就会静默只比前两个 —— 少了一条结论却不报错，最难发现。
    ap.add_argument("--encs", nargs="+", default=None,
                    help="默认自动扫描 runs/ 里存在的 esmc300m/esmc600m/esmc6b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="把对比表写成 csv")
    a = ap.parse_args()

    if a.encs is None:
        found = []
        for e in ("esmc300m", "esmc600m", "esmc6b"):
            if os.path.isdir(os.path.join(a.runs, "baselines_%s_%s" % (e, a.split))):
                found.append(e)
        a.encs = found
        print("自动发现编码器：%s" % (a.encs or "（无）"))
    if len(a.encs) < 2:
        print("!! 至少需要两个编码器的结果才能比较（当前 %d 个）" % len(a.encs))
        print("   请先跑：REPO=... TAGNAME=<enc> bash run_encoder_ablation.sh")
        return 1
    pairs_of_encs = list(itertools.combinations(a.encs, 2))
    print("将做 %d 组两两对比：%s"
          % (len(pairs_of_encs), "  ".join("%s×%s" % p for p in pairs_of_encs)))

    rows = []
    print("=" * 96)
    print("编码器规模消融 —— 划分 %s" % a.split)
    print("=" * 96)

    # ---------- 1. 逐个编码器：自身水平 + 与自家尺子的差距 ----------
    print("\n【一】各编码器的绝对水平（test n=?）\n")
    store = {}
    for e in a.encs:
        line = {}
        # 线性探针（尺子）
        bl = load_base(a.runs, "baselines_%s_%s" % (e, a.split))
        if bl:
            m = all_metrics(bl["y"], bl["pred"])
            line["ridge"] = (bl, m)
        # 各实验配置
        for tpl, pkey, _lab in CONFIGS:
            tag = tpl.format(e=e, s=a.split)
            ex = load_exp(a.runs, tag, pkey)
            if ex:
                line[tag] = (ex, all_metrics(ex["y"], ex["pred"]))
        store[e] = line

    hdr = "%-22s %-28s %8s %8s %9s" % ("编码器", "实验", "Spearman", "R²", "单种子 std")
    print(hdr)
    print("-" * 96)
    y_ref = None
    for e in a.encs:
        line = store.get(e, {})
        if not line:
            print("%-22s （无结果，先跑 run_encoder_ablation.sh）" % ENCLABEL.get(e, e))
            continue
        first = True
        for k, (ex, m) in line.items():
            lab = "线性探针（尺子）" if k == "ridge" else next(
                (l for t, _p, l in CONFIGS if t.format(e=e, s=a.split) == k), k)
            sd = ""
            if ex["singles"]:
                sd = "%.4f" % float(np.std(ex["singles"], ddof=1)) if len(ex["singles"]) > 1 else "—"
            print("%-22s %-28s %8s %8s %9s" % (
                ENCLABEL.get(e, e) if first else "", lab, s(m["spearman"]), s(m["r2"]), sd))
            first = False
            rows.append({"encoder": e, "encoder_label": ENCLABEL.get(e, e),
                         "experiment": k, "experiment_label": lab,
                         "n": int(len(ex["y"])), "spearman": m["spearman"], "r2": m["r2"],
                         "single_std": (float(np.std(ex["singles"], ddof=1)) if len(ex["singles"]) > 1 else None)})
            # 记录一份 y 用于一致性校验
            if y_ref is None:
                y_ref = ex["y"]

    # ---------- 2. 跨编码器配对比较（同一配置，只换编码器）----------
    print("\n【二】跨编码器配对检验（同一批测试样本，唯一变量 = 编码器）\n")
    print("%-40s %10s %24s %8s  %s" % ("对比（候选 vs 对照）", "ΔSpearman", "95% CI", "p", "判定"))
    print("-" * 96)
    pair_rows = []

    def compare(name, A, B, ya, yb):
        ya, yb = np.asarray(ya, float), np.asarray(yb, float)
        if not np.allclose(ya, yb):
            print("!! 拒绝比较：两次实验的 y_test 不一致（配对检验前提被破坏）")
            return
        d, lo, hi, pv = paired_bootstrap(ya, A, B, "spearman", n_boot=a.n_boot, seed=0)
        verdict = "显著优于" if lo > 0 else ("显著劣于" if hi < 0 else "无显著差异")
        print("%-40s %+10.4f %24s %8.3f  %s" % (
            name, d, "[%+.4f, %+.4f]" % (lo, hi), pv, verdict))
        pair_rows.append({"comparison": name, "delta": d, "ci_lo": lo, "ci_hi": hi,
                          "p": pv, "significant": bool(lo > 0 or hi < 0), "verdict": verdict})

    # 2.1 尺子跨编码器（所有两两组合）
    for eA, eB in pairs_of_encs:
        A = store.get(eA, {}).get("ridge")
        B = store.get(eB, {}).get("ridge")
        if A and B:
            compare("线性探针（尺子） %s vs %s" % (ENCLABEL.get(eA, eA), ENCLABEL.get(eB, eB)),
                    A[0]["pred"], B[0]["pred"], A[0]["y"], B[0]["y"])

    # 2.2 各配置跨编码器（所有两两组合）
    for tpl, pkey, lab in CONFIGS:
        for eA, eB in pairs_of_encs:
            A = store.get(eA, {}).get(tpl.format(e=eA, s=a.split))
            B = store.get(eB, {}).get(tpl.format(e=eB, s=a.split))
            if A and B:
                compare("%s  %s vs %s" % (lab, ENCLABEL.get(eA, eA), ENCLABEL.get(eB, eB)),
                        A[0]["pred"], B[0]["pred"], A[0]["y"], B[0]["y"])

    # ---------- 3. 每个编码器内部：MLP vs 尺子 ----------
    print("\n【三】各编码器内部：MLP 是否显著超过线性探针（这是本项目的核心问题）\n")
    for e in a.encs:
        line = store.get(e, {})
        rg = line.get("ridge")
        if not rg:
            continue
        for tpl, pkey, lab in CONFIGS:
            tag = tpl.format(e=e, s=a.split)
            ex = line.get(tag)
            if not ex:
                continue
            compare("%s  %s 的 %s vs 尺子" % (ENCLABEL.get(e, e), lab, "MLP"),
                    ex[0]["pred"], rg[0]["pred"], ex[0]["y"], rg[0]["y"])

    # ---------- 落盘 ----------
    if a.out:
        import csv as _csv
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="") as fh:
            if pair_rows:
                w = _csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
                w.writeheader()
                w.writerows(pair_rows)
        print("\n已写出对比表：%s（%d 行）" % (a.out, len(pair_rows)))

    print("\n结论怎么读：")
    print("  · 若「【二】MLP 600M vs 300M」不显著 → 规模不敏感 ⇒ 不值得下 6B")
    print("  · 若显著且 300M 更差       → 规模确实有用 ⇒ 6B 值得下")
    print("COMPARE_DONE")


if __name__ == "__main__":
    main()
