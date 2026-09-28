"""泄漏量化：test / valid 里有多少序列能在 train 里找到同源物？

★ 为什么必须做这一步
  "我按蛋白划分了"不等于"没有泄漏"。同一个物种的蛋白彼此高度同源，
  即使按蛋白随机划分，test 的蛋白也常能在 train 里找到近亲（>25% 同一性）。
  模型靠"背近亲"就能拿分 —— 这个分数换到新蛋白上会掉。

  所以划分完之后，**必须实测**交叉集合的同源重叠率，而不是假设它是 0。

判据：同一性 ≥ ID 且 覆盖率 ≥ COV（覆盖率 = 比对长度 / 两条序列长度的较小者）
默认在 25 / 30 / 40 / 50% 四档都报一遍，方便看"泄漏随阈值怎么变"。

用法：
    python check_leakage.py                       # 两套划分都查
    python check_leakage.py --split-dir ../data/splits/homology
"""
import argparse
import os
import subprocess
import sys
import time
from collections import Counter

import pandas as pd

THRESHOLDS = [25, 30, 40, 50]


def read_split(split_dir):
    out = {}
    for sp in ("train", "valid", "test"):
        p = os.path.join(split_dir, f"{sp}.csv")
        if not os.path.exists(p):
            raise SystemExit(f"缺少 {p} —— 先跑 make_splits.py")
        out[sp] = pd.read_csv(p)
    return out


def blast_against_train(split_dir, work, threads):
    """把 train 建库，test+valid blast 回去。返回 tsv 路径。"""
    os.makedirs(work, exist_ok=True)
    parts = read_split(split_dir)
    db_fa = os.path.join(work, "train.fasta")
    q_fa = os.path.join(work, "query.fasta")
    tsv = os.path.join(work, "vs_train.tsv")

    with open(db_fa, "w") as fh:
        for i, r in parts["train"].iterrows():
            fh.write(f">t{i}\n{r['sequence']}\n")
    with open(q_fa, "w") as fh:
        for sp in ("valid", "test"):
            for i, r in parts[sp].iterrows():
                fh.write(f">{sp}_{i}\n{r['sequence']}\n")

    db = os.path.join(work, "traindb")
    if not os.path.exists(db + ".pin"):
        subprocess.run(["makeblastdb", "-in", db_fa, "-dbtype", "prot", "-out", db],
                       check=True, capture_output=True)
    if not os.path.exists(tsv) or os.path.getsize(tsv) == 0:
        t0 = time.time()
        r = subprocess.run(
            ["blastp", "-query", q_fa, "-db", db, "-out", tsv,
             "-outfmt", "6 qseqid sseqid pident length qlen slen evalue",
             "-evalue", "1e-3", "-num_threads", str(threads),
             "-max_target_seqs", "500", "-max_hsps", "1"],
            capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write((r.stderr or "")[-2000:])
            raise SystemExit("blastp 失败")
        print(f"    blastp 用时 {time.time() - t0:.0f}s")
    return tsv, parts


def quantify(tsv, parts, split_dir):
    """按阈值统计：每个 query 是否至少有一个满足判据的 train 命中。"""
    # query -> 最佳命中（按 同一性 与 覆盖率 分别取最大）
    best = {}
    with open(tsv) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            q = f[0]
            pid, aln, ql, sl = float(f[2]), int(f[3]), int(f[4]), int(f[5])
            cov = min(aln / max(ql, 1), aln / max(sl, 1))
            cur = best.get(q)
            if cur is None or (pid, cov) > cur:
                best[q] = (pid, cov)

    return _tally(best, parts)


def quantify_from_allvall(tsv, parts, allfasta):
    """★ 直接复用 all-vs-all 的命中表来量化泄漏（推荐做法）。

    为什么必须提供这个路径 —— 上面那个 quantify 有个隐蔽的不一致：
      · make_splits 用它自己的 all-vs-all（库 = **全部 3157 条**）决定谁和谁同源、谁和谁同组；
      · 而 quantify 走的是单独的 blastp（库 = **只有 train 的 2525 条**）。
      但 BLAST 的 E-value 会**随数据库大小线性缩放**（E ≈ K·m·n·e^(-λS) 里的 n 就是库大小）。
      库从 3157 缩到 2525，同一个比对对的 E-value 会**变小约 1.25 倍**。
      于是处在阈值边缘的命中（例如实测到的 evalue = 1e-3，正好等于 -evalue 1e-3）会出现：
        全库比对 → 1.25e-3 > 1e-3 → 被丢弃 → 划分时判为「不同源」（该连的边没连上）
        小库比对 → 1.00e-3 ≤ 1e-3 → 被保留 → 报泄漏时判为「同源」
      ⇒ 同一个序列对，划分逻辑说不同源、泄漏报告说同源，**自相矛盾**。
      实测：25% 阈值下这个不一致影响 0 条；20% 阈值下影响 1 条（0.32%）。
      它不会让结论翻盘，但会让读者拿着两张表对不上账。

    正确做法：**让量化泄漏和做划分用同一份证据** —— 读 all-vs-all 的命中表。

    ★★ 映射必须用「序列」而不是「行号」！
      两个原因：
        · 划分 csv 是 `to_csv(index=False)` 存的，所以读回来是 0..n-1 的**划分内位置**，
          根本不是原始行号；
        · all-vs-all 里的 `s{i}` 是 `enumerate(df['sequence'])` 的**原始位置**。
      两者混用会得到一个"全错但仍然看起来像数字"的结果 —— 实测会从 0.3% 变成 24.4%，
      而且绝不报错。用序列字符串当键就没有这个问题（数据集已按序列去重，序列唯一）。
    """
    idx2seq = {}                       # s{i} -> 序列
    name = None
    with open(allfasta) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                name = int(line[1:].lstrip("s"))
            elif name is not None:
                idx2seq[name] = line
                name = None

    seq2split = {}
    for sp in ("train", "valid", "test"):
        for s in parts[sp]["sequence"]:
            seq2split[s] = sp

    best = {}                          # 序列 -> (同一性, 覆盖率)，只记「命中 train」的
    with open(tsv) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            try:
                qi, si = int(f[0].lstrip("s")), int(f[1].lstrip("s"))
            except ValueError:
                continue
            sq, ss = idx2seq.get(qi), idx2seq.get(si)
            if sq is None or ss is None:
                continue
            if seq2split.get(sq) not in ("valid", "test"):
                continue           # 只关心 test/valid 能不能找到 train 里的同源物
            if seq2split.get(ss) != "train":
                continue           # 命中必须落在 train 里才算泄漏
            pid, aln, ql, sl = float(f[2]), int(f[3]), int(f[4]), int(f[5])
            cov = min(aln / max(ql, 1), aln / max(sl, 1))
            cur = best.get(sq)
            if cur is None or (pid, cov) > cur:
                best[sq] = (pid, cov)

    return _tally_by_seq(best, parts)


def _tally_by_seq(best, parts):
    """汇总：键是序列本身（all-vs-all 路径用）。"""
    result = {}
    for sp in ("valid", "test"):
        seqs = list(parts[sp]["sequence"])
        result[sp] = {"n": len(seqs)}
        for th in THRESHOLDS:
            result[sp][f"ge{th}"] = sum(
                1 for s in seqs if s in best and best[s][0] >= th and best[s][1] >= 0.5)
        w = [(best[s], s) for s in seqs if s in best]
        result[sp]["worst"] = max(w, key=lambda x: x[0])[0] if w else (0, 0)
        result[sp]["n_any_hit"] = len(w)
    return result


def _tally(best, parts):
    """汇总：键是 `{split}_{行号}`（单独 blast 路径用）。"""
    result = {}
    for sp in ("valid", "test"):
        n = len(parts[sp])
        ids = [f"{sp}_{i}" for i in parts[sp].index]
        result[sp] = {"n": n}
        for th in THRESHOLDS:
            hit = sum(1 for q in ids
                      if q in best and best[q][0] >= th and best[q][1] >= 0.5)
            result[sp][f"ge{th}"] = hit
        worst = max((best[q] for q in ids if q in best), default=(0, 0))
        result[sp]["worst"] = worst
        result[sp]["n_any_hit"] = sum(1 for q in ids if q in best)
    return result


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-root", default=os.path.join(here, "..", "data", "splits"))
    ap.add_argument("--split-dir", default=None, help="只查这一套")
    ap.add_argument("--work", default=os.path.join(here, "..", "data", "blast_work"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--allvall", default=None,
                    help="★ 推荐：复用 make_splits 的 all-vs-all 命中表（如 "
                         "data/blast_work/allvall_25_0.5.tsv）来量化泄漏，"
                         "避免与小库比对因 E-value 缩放产生判据不一致。给了它就不再单独 blastp。")
    ap.add_argument("--allfasta", default=None,
                    help="all-vs-all 用的序列文件（默认取 --allvall 同目录下的 all.fasta）")
    a = ap.parse_args()

    if a.allvall and not a.allfasta:
        a.allfasta = os.path.join(os.path.dirname(os.path.abspath(a.allvall)), "all.fasta")
    if a.allvall and not os.path.exists(a.allfasta):
        raise SystemExit(f"找不到 {a.allfasta} —— 它是 make_splits 写 all-vs-all 时生成的，"
                         f"用 --allfasta 指定")

    dirs = ([a.split_dir] if a.split_dir
            else [os.path.join(a.splits_root, m) for m in ("random", "homology")])

    print("=" * 88)
    print("泄漏量化：test / valid 能在 train 里找到同源物的比例")
    print("判据：同一性 ≥ 阈值  且  覆盖率 ≥ 50%")
    if a.allvall:
        print(f"证据来源：all-vs-all 命中表 {a.allvall}（与划分逻辑同源，口径必然一致）")
    else:
        print("证据来源：单独 blastp（库 = 仅 train）")
        print("  ⚠️ 与划分用的 all-vs-all（库 = 全量）不是同一份证据；")
        print("     E-value 随库大小缩放，边缘命中会出现判据不一致（见 quantify_from_allvall 注释）。")
    print("=" * 88)

    summary = []
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        if not os.path.isdir(d):
            print(f"\n[跳过] {d} 不存在")
            continue
        print(f"\n【{name}】{d}")
        if a.allvall:
            parts = read_split(d)
            res = quantify_from_allvall(a.allvall, parts, a.allfasta)
        else:
            tsv, parts = blast_against_train(d, os.path.join(a.work, name), a.threads)
            res = quantify(tsv, parts, d)
        for sp in ("valid", "test"):
            r = res[sp]
            print(f"  {sp:6s} n={r['n']:5d}  "
                  + "  ".join(f">={th}%: {r[f'ge{th}']:4d} ({r[f'ge{th}']/r['n']*100:5.1f}%)"
                              for th in THRESHOLDS))
            print(f"         有任意命中的 {r['n_any_hit']}/{r['n']}；"
                  f"最严重一例 同一性 {r['worst'][0]:.1f}% 覆盖 {r['worst'][1]*100:.0f}%")
            summary.append(dict(split=name, part=sp, n=r["n"],
                                **{f"ge{th}_pct": round(r[f"ge{th}"] / r["n"] * 100, 2)
                                   for th in THRESHOLDS},
                                worst_pident=round(r["worst"][0], 1)))

    df = pd.DataFrame(summary)
    out = os.path.join(a.splits_root, "leakage_summary.csv")
    os.makedirs(a.splits_root, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n汇总写出：{out}")

    # ---- 判读 ----
    if len(df):
        print("\n" + "-" * 88)
        print("怎么读这张表：")
        print("  · homology 划分下 test 的 >=25% 比例 ≈ 0  → 合格")
        print("  · random  划分下的比例（通常几十 %）与 homology 的**差值** = 泄漏的规模")
        print("  · 后续报数时，**主口径用 homology**；random 的结果当「文献可比口径」另列")
        print("-" * 88)
    print("CHECK_LEAKAGE_DONE")


if __name__ == "__main__":
    main()
