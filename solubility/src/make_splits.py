"""数据集划分：一次生成「随机划分」和「同源感知划分」两套，用于量化泄漏的代价。

★ 为什么要做两套
  本数据集是 3157 个 **E. coli K-12 蛋白** —— 同一个物种，彼此之间同源程度很高
  （很多是旁系同源、同一操纵子、同一家族）。
  文献里在 eSOL 上做回归的多数工作（含 Han et al. 2019、GraphSol 2020）
  用的是 **随机划分**：把蛋白随机分到 train/test。
  这样 test 里的蛋白大概率能在 train 里找到近亲 → 模型"背近亲"就能拿分，
  报出来的分数是虚的，换到真正的新蛋白上会掉。

  所以本脚本做两套划分：
    random/    —— 随机划分（复现文献口径，作为"虚高上界"）
    homology/  —— 同源感知划分（BLAST 家族整组进同一个集合，**主口径**）
  两者之差 = "同源泄漏带来的虚高"。

★ 判据（同源）
  all-vs-all blastp → 保留「同一性 ≥ ID% 且 (比对长度/两者长度的较小者) ≥ COV」
  → 并查集连边 → 家族 → **家族整组分配**（绝不拆散）
  默认 ID=25%, COV=0.5（与文献常用的 25% 去冗余阈值一致）。

★ 为什么不用 k-mer / 共享片段
  k-mer 要求 k 个残基完全一致，比 BLAST 宽松地多地把同源判成"不同源"（系统性漏判）。
  在另一个项目上实测过：共享 13-mer 判出 0% 泄漏，而 BLAST ≥25% 查出来 26.48%。
  **去冗余必须用比对法（BLAST / MMseqs2）。**

★ 分层
  回归任务没有"正负样本"可平衡，改成对**标签分箱（默认 5 箱）分层**：
  保证 train/valid/test 三个集合的溶解度分布形状一致（均值、分位数接近）。

用法：
    python make_splits.py                        # 两套都做
    python make_splits.py --only homology
    python make_splits.py --id 30 --cov 0.5      # 换阈值
"""
import argparse
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

MAX_TARGET_SEQS = 2000          # all-vs-all 时每个 query 最多留多少命中


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
            return True
        return False


# ---------------------------------------------------------------------------
def blast_families(df, work, identity, coverage, threads):
    """all-vs-all blastp → 并查集家族。返回 {家族id: [行号,...]}"""
    os.makedirs(work, exist_ok=True)
    fa = os.path.join(work, "all.fasta")
    db = os.path.join(work, "alldb")
    tsv = os.path.join(work, f"allvall_{identity:g}_{coverage:g}.tsv")

    with open(fa, "w") as fh:
        for i, s in enumerate(df["sequence"]):
            fh.write(f">s{i}\n{s}\n")

    if not os.path.exists(db + ".pin"):
        print(f"  [1/3] makeblastdb（{len(df)} 条）...")
        subprocess.run(["makeblastdb", "-in", fa, "-dbtype", "prot", "-out", db],
                       check=True, capture_output=True)
    else:
        print("  [1/3] 复用已有 BLAST 库")

    if not os.path.exists(tsv) or os.path.getsize(tsv) == 0:
        print(f"  [2/3] all-vs-all blastp（{threads} 线程）... 最慢的一步")
        t0 = time.time()
        r = subprocess.run(
            ["blastp", "-query", fa, "-db", db, "-out", tsv,
             "-outfmt", "6 qseqid sseqid pident length qlen slen evalue",
             "-evalue", "1e-3", "-num_threads", str(threads),
             "-max_target_seqs", str(MAX_TARGET_SEQS), "-max_hsps", "1"],
            capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write((r.stderr or "")[-2000:])
            raise SystemExit("blastp 失败")
        print(f"        done {time.time() - t0:.0f}s，"
              f"{os.path.getsize(tsv) / 1e6:.1f} MB")
    else:
        print(f"  [2/3] 复用 {os.path.basename(tsv)}")

    print(f"  [3/3] 按 同一性≥{identity:g}% / 覆盖≥{coverage:.0%} 聚家族 ...")
    uf = UnionFind(len(df))
    n_hit = n_edge = 0
    with open(tsv) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            n_hit += 1
            q, s = f[0], f[1]
            if q == s:
                continue
            pid, aln, ql, sl = float(f[2]), int(f[3]), int(f[4]), int(f[5])
            if pid < identity:
                continue
            if min(aln / max(ql, 1), aln / max(sl, 1)) < coverage:
                continue
            if uf.union(int(q[1:]), int(s[1:])):
                n_edge += 1
    fams = defaultdict(list)
    for i in range(len(df)):
        fams[uf.find(i)].append(i)
    print(f"        {n_hit:,} 条命中 → {n_edge:,} 次合并 → **{len(fams)} 个家族**；"
          f"最大 {max(len(v) for v in fams.values())} 条，"
          f"独居 {sum(1 for v in fams.values() if len(v) == 1)} 个")
    return list(fams.values())


def stratified_group_assign(fams, labels, frac, nbins, seed):
    """按家族整组分配，同时让各集合的标签分布一致（分层）。

    策略：先把每个家族按其成员标签的均值归到某个箱；
    家族按大小降序处理（大块先放，避免最后塞不下）；
    每个家族放进「该箱缺口比例最大 且 未超容量」的集合。
    """
    rng = random.Random(seed)
    n = len(labels)
    names = ["train", "valid", "test"]
    assert abs(sum(frac) - 1.0) < 1e-9, "比例之和必须为 1"

    # 全局标签分箱
    edges = np.quantile(labels, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_of = np.digitize(labels, edges[1:-1])

    target_n = {sp: frac[i] * n for i, sp in enumerate(names)}
    target_bin = {sp: defaultdict(float) for sp in names}
    for sp in names:
        for b in range(nbins):
            target_bin[sp][b] = frac[names.index(sp)] * (bin_of == b).sum()

    cur_n = {sp: 0 for sp in names}
    cur_bin = {sp: defaultdict(int) for sp in names}
    out = {sp: [] for sp in names}

    order = sorted(range(len(fams)), key=lambda k: (-len(fams[k]), rng.random()))
    for k in order:
        members = fams[k]
        # 家族归属箱 = 成员标签的中位箱
        b = int(np.median(bin_of[members]))
        best, best_score = None, -1e9
        for sp in names:
            cap_left = target_n[sp] - cur_n[sp]
            if cap_left < len(members):
                continue
            fill = cur_bin[sp][b] / max(target_bin[sp][b], 1.0)     # 该箱已填比例
            score = -fill + 1e-6 * cap_left                        # 越空越该放
            if score > best_score:
                best, best_score = sp, score
        if best is None:                                           # 全部放不下 → 塞给缺口最大的
            best = max(names, key=lambda sp: target_n[sp] - cur_n[sp])
        out[best].extend(members)
        cur_n[best] += len(members)
        cur_bin[best][b] += len(members)
    return {sp: sorted(out[sp]) for sp in names}


def report(name, df, assign, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    print(f"\n  [{name}] 划分结果")
    for sp in ("train", "valid", "test"):
        idx = assign[sp]
        sub = df.iloc[idx]
        sub.to_csv(os.path.join(out_dir, f"{sp}.csv"), index=False)
        v = sub["solubility_pct"]
        rows.append(dict(split=sp, n=len(sub), frac=len(sub) / len(df),
                         mean=v.mean(), std=v.std(), p25=v.quantile(.25),
                         p50=v.median(), p75=v.quantile(.75),
                         lmin=sub["seq_len"].min(), lmed=sub["seq_len"].median()))
        print(f"    {sp:6s} n={len(sub):5d} ({len(sub)/len(df)*100:4.1f}%)  "
              f"溶解度 均值 {v.mean():5.1f} 中位 {v.median():5.1f} "
              f"P25 {v.quantile(.25):5.1f} P75 {v.quantile(.75):5.1f}  "
              f"长度中位 {sub['seq_len'].median():.0f}")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "split_summary.csv"), index=False)
    return pd.DataFrame(rows)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "..", "data", "esol_reg.csv"))
    ap.add_argument("--out-root", default=os.path.join(here, "..", "data", "splits"))
    ap.add_argument("--work", default=os.path.join(here, "..", "data", "blast_work"))
    ap.add_argument("--id", type=float, default=25.0, help="同源判据：同一性 %%")
    ap.add_argument("--cov", type=float, default=0.5, help="同源判据：覆盖率")
    ap.add_argument("--nbins", type=int, default=5, help="标签分箱数（分层用）")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=["random", "homology"], default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    label = df["solubility_pct"].to_numpy(float)
    print("=" * 84)
    print(f"数据集 {len(df)} 条；标签 {label.min():.0f}~{label.max():.0f}%，均值 {label.mean():.1f}")
    print("=" * 84)

    frac = [0.8, 0.1, 0.1]
    summaries = {}

    # ---------- ① 随机划分（每行 = 一个蛋白，所以就是按蛋白随机分） ----------
    if a.only in (None, "random"):
        print("\n[① 随机划分] 蛋白级随机打乱（复现文献口径，预期有同源泄漏）")
        rng = random.Random(a.seed)
        idx = list(range(len(df)))
        rng.shuffle(idx)
        n1 = int(0.8 * len(df))
        n2 = int(0.9 * len(df))
        assign = {"train": sorted(idx[:n1]), "valid": sorted(idx[n1:n2]),
                  "test": sorted(idx[n2:])}
        # 用分层再修一遍分布（同样随机，但标签分布对齐）
        assign = stratified_group_assign([[i] for i in idx], label, frac, a.nbins, a.seed)
        s = report("random", df, assign, os.path.join(a.out_root, "random"))
        summaries["random"] = s

    # ---------- ② 同源感知划分 ----------
    if a.only in (None, "homology"):
        print(f"\n[② 同源感知划分] BLAST 同一性≥{a.id:g}% / 覆盖≥{a.cov:.0%}，家族整组分配")
        fams = blast_families(df, a.work, a.id, a.cov, a.threads)
        assign = stratified_group_assign(fams, label, frac, a.nbins, a.seed)
        s = report("homology", df, assign, os.path.join(a.out_root, "homology"))
        summaries["homology"] = s

        # 家族大小分布落盘（便于写报告）
        with open(os.path.join(a.out_root, "homology", "family_stats.txt"), "w",
                  encoding="utf-8") as fh:
            sizes = sorted((len(f) for f in fams), reverse=True)
            fh.write(f"家族数 {len(fams)}\n")
            fh.write(f"独居家族 {sum(1 for x in sizes if x == 1)}\n")
            fh.write(f"最大家族 {sizes[0]}\n")
            fh.write(f"家族大小前 20: {sizes[:20]}\n")
            fh.write(f"单条家族占比 {sum(1 for x in sizes if x==1)/len(sizes)*100:.1f}%\n")

    print("\n" + "=" * 84)
    print("两套划分都已生成 →", a.out_root)
    print("下一步：python check_leakage.py   （量化 test↔train 的同源重叠）")
    print("MAKE_SPLITS_DONE")


if __name__ == "__main__":
    main()
