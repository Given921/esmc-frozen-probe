"""阈值扫描汇总 —— 回答"去冗余该选哪个阈值"。

扫描流程（由 run_threshold_scan.sh 驱动）：
    对每个同一性阈值 ID ∈ {20,25,30,35,40}：
      ① 用该阈值做 BLAST 家族聚类 + 家族整组划分 → data/splits_id<ID>/
      ② 量化 test/valid 与 train 在**多档判据**下的同源重叠 → leakage_summary.csv

本脚本把各阈值的产物拼成一张表：**阈值 → 泄漏率 → 划分规模/分布**，
用于回答两个问题：
    · 松到多少，泄漏就不可忽略？（阈值上调 → 泄漏上升，找"从 0 抬头"的拐点）
    · 严到多少，划分就开始失衡？（阈值下调 → 家族变大 → 分组分配变粗，规模/分布漂移）

★ 注意判读口径
    leakage_summary.csv 里的 ge25/ge30/... 是**同一个划分**在**不同判据**下的泄漏率。
    所以"选了 30% 阈值做去冗余"不等于"ge30 那一列是 0" —— 去冗余时用的是
    「同一性 ≥ 阈值 且 覆盖 ≥50%」，而报告列里的 ge30 是「同一性 ≥30%」不设覆盖；
    两者差一个覆盖率条件，前者更宽松 → ge30 列会略高于 0。这是正常的，不是 bug。

用法：
    python scan_summary.py                    # 扫 data/splits_id*
    python scan_summary.py --root data --prefix splits_id
"""
import argparse
import os
import sys

import pandas as pd

THS = (25, 30, 40, 50)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root_default = os.path.abspath(os.path.join(here, "..", "data"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=root_default)
    ap.add_argument("--prefix", default="splits_id")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cands = []
    for name in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, name)
        if not os.path.isdir(d) or not name.startswith(a.prefix):
            continue
        if a.suffix and not name.endswith(a.suffix):
            continue
        tail = name[len(a.prefix):]
        if a.suffix:
            tail = tail[: -len(a.suffix)]
        try:
            thr = float(tail)
        except ValueError:
            continue
        cands.append((thr, name, d))
    cands.sort()

    if not cands:
        raise SystemExit(f"在 {a.root} 下没找到 {a.prefix}<数字> 形式的目录 —— "
                         f"先跑 run_threshold_scan.sh")

    rows = []
    for thr, name, d in cands:
        sd = os.path.join(d, "homology", "split_summary.csv")
        lk = os.path.join(d, "leakage_summary.csv")
        if not os.path.exists(sd):
            print(f"  [跳过] {name} 缺 homology/split_summary.csv")
            continue
        s = pd.read_csv(sd).set_index("split")
        row = dict(threshold=thr,
                   n_train=int(s.loc["train", "n"]),
                   n_valid=int(s.loc["valid", "n"]),
                   n_test=int(s.loc["test", "n"]),
                   train_mean=float(s.loc["train", "mean"]),
                   test_mean=float(s.loc["test", "mean"]))
        if os.path.exists(lk):
            L = pd.read_csv(lk)
            L = L[L["split"] == "homology"]
            for part in ("valid", "test"):
                r = L[L.part == part]
                if len(r):
                    r = r.iloc[0]
                    for t in THS:
                        row[f"{part}_ge{t}"] = float(r[f"ge{t}_pct"])
                    row[f"{part}_worst"] = float(r["worst_pident"])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("threshold")
    out = a.out or os.path.join(a.root, "scan", "threshold_scan.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    print("=" * 104)
    print("去冗余阈值扫描：同一性阈值 → test/valid 的泄漏率 + 划分规模/分布")
    print("=" * 104)
    hdr = (f"{'阈值':>6s} {'n_train':>8s} {'n_valid':>8s} {'n_test':>7s} "
           f"{'train均值':>9s} {'test均值':>9s} " + "".join(f"{'test≥'+str(t):>9s}" for t in THS)
           + f"{'test最严重':>11s}")
    print(hdr)
    print("-" * 104)
    for _, r in df.iterrows():
        cells = "".join(f"{r.get(f'test_ge{t}', float('nan')):9.1f}" for t in THS)
        print(f"{r.threshold:6.0f} {int(r.n_train):8d} {int(r.n_valid):8d} {int(r.n_test):7d} "
              f"{r.train_mean:9.2f} {r.test_mean:9.2f} {cells}"
              f"{r.get('test_worst', float('nan')):11.1f}")
    print("-" * 104)
    if len(df) >= 2:
        print("怎么选：")
        print("  · 从上往下看 test 各列 —— 越是低阈值，泄漏越接近 0；越松的阈值，泄漏抬头。")
        print("  · 看 n_train/n_test 与 train/test 均值 —— 阈值过低时家族过大，")
        print("    分组分配的粒度变粗，三集合规模与分布会开始漂；那是「过严的代价」。")
        print("  · 落在「泄漏已 ≈0」且「规模/分布仍稳」的那档就是合适阈值。")
        print("  · 领域惯例：25%~30% 是蛋白去冗余的常用线（SPARKS/ProStab/MegaScale 同款口径），")
        print("    可以拿它当默认值，再用本表验证它对本数据集确实够严。")
    print(f"\n汇总写出：{out}")
    print("SCAN_SUMMARY_DONE")


if __name__ == "__main__":
    main()
