"""构建热稳定性 ΔΔG 数据集（MegaScale 单点突变 + 官方 25% 聚类划分 + 去泄漏）。

★ 本脚本严格照 ProStab 官方实现（`prostab/datamodules/datasets/megascale.py`）的口径，
  每一步都写清"官方怎么做的"和"我们为什么这么做"。**不自己发明过滤规则** ——
  一旦口径和文献不同，后面所有分数就没法和已发表数字对话。

官方口径（逐条对应）
  1. 主表 = HF `RosettaCommons/MegaScale` 的 `dataset2` + `dataset3` 合并
     （官方代码里叫 `Tsuboyama2023_Dataset2_Dataset3_20230416.csv`，CSV 本体不在 GitHub 上）。
  2. 剔除 `ddG_ML == '-'` 的行（作者标注为不可靠）。
  3. 剔除 `mut_type` 含 `ins` / `del` / `:` 的行 —— **只留单点替换**。
     多突变/插入删除的点突变模型处理不了，混进来只会制造噪声。
  4. 去泄漏：读 `mmseq_mut_search_0.25.m8`，取**第 2 列**（0 基索引 1）当行号，
     把训练集与验证集里的这些行**删掉**（测试集不删）。
     含义：这些训练行与测试集某条序列在 25% 同一性下互相命中 → 属于同源泄漏。
  5. 划分：`mega_splits.pkl` 给的 train/val/test 是**按野生型蛋白**做的 25% 聚类划分。
     ⇒ **同一蛋白的所有突变必然落在同一侧** —— 这正是不泄漏的关键，
        也是它比 `Dataset-Meta-scale` 那个 `stage` 列强的地方
        （那个 stage 列实测把同一蛋白的突变拆到 train/test，是泄漏源）。
  6. 标签 = **`-ddG_ML`**（官方代码里 `ddG = -torch.tensor([float(mut_seq.ddG_ML)])`）。
     负号不能丢：符号约定不同，报出来的相关方向会反。

★ 需要联网吗？不需要。本脚本只读已经下好的本地文件。

用法
    python src/build_dataset.py --inspect     # 先看数据长什么样（不写任何文件）
    python src/build_dataset.py               # 正式构建
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"          # ★ 官方 21 字母表，顺序来自 ProStab/SPURS


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ 读主表
def load_main(root):
    raw = os.path.join(root, "data", "raw")
    files = ["ds2_a.parquet", "ds2_b.parquet", "ds3.parquet"]
    frames = []
    for f in files:
        fp = os.path.join(raw, f)
        if not os.path.exists(fp):
            raise SystemExit(f"缺文件 {fp}（先跑 download_data.sh）")
        d = pd.read_parquet(fp)
        log(f"  {f:16s} {len(d):>9,} 行  列={list(d.columns)}")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    log(f"  合并后 {len(df):,} 行")
    return df


def inspect(root):
    df = load_main(root)
    log("\n=== 列与取值 ===")
    for c in df.columns:
        v = df[c]
        if v.dtype == object and v.nunique() < 12:
            log(f"  {c:20s} 唯一值 {sorted(set(map(str, v.dropna().unique())))}")
        else:
            log(f"  {c:20s} dtype={v.dtype}  nunique={v.nunique():,}  "
                f"样例={str(v.iloc[0])[:70]}")
    log("\n=== mut_type 形态抽样 ===")
    log(df["mut_type"].value_counts().head(10).to_string())
    log("\n=== aa_seq 长度分布 ===")
    L = df["aa_seq"].str.len()
    log(f"  n={len(L):,}  min={L.min()}  p50={L.median():.0f}  max={L.max()}")
    log("\n=== ddG_ML ===")
    log(f"  等于 '-' 的行数：{(df['ddG_ML'].astype(str) == '-').sum():,}")
    log("\n=== WT_name 样例 ===")
    log(df["WT_name"].head(5).tolist())

    ps = os.path.join(root, "data", "prostab")
    pk = os.path.join(ps, "mega_splits.pkl")
    if os.path.exists(pk):
        with open(pk, "rb") as fh:
            splits = pickle.load(fh)
        log(f"\n=== mega_splits.pkl  keys={list(splits)[:24]}")
        for k in list(splits)[:24]:
            v = splits[k]
            log(f"  {k:14s} {len(v):>6d} 项  样例={list(v)[:2]}")
    m8 = os.path.join(ps, "mmseq_mut_search_0.25.m8")
    if os.path.exists(m8):
        with open(m8, "r", encoding="utf-8", errors="replace") as fh:
            lines = [next(fh) for _ in range(3)]
        log(f"\n=== mmseq m8 前 3 行（制表符分列）===")
        for ln in lines:
            log("  " + ln.rstrip()[:160])


# ------------------------------------------------------------------ 构建
def build(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    funnel = []
    df = load_main(root)
    funnel.append(("数据源合并（dataset2 + dataset3）", len(df)))

    # --- 官方第 2、3 步：剔不可靠 + 只留单点替换 ---
    n = len(df)
    df = df.loc[df["ddG_ML"].astype(str) != "-", :].reset_index(drop=True)
    funnel.append(("剔 ddG_ML='-'", len(df)))
    mt = df["mut_type"].astype(str)
    df = df.loc[~mt.str.contains("ins") & ~mt.str.contains("del")
               & ~mt.str.contains(":"), :].reset_index(drop=True)
    funnel.append(("剔 ins/del/多突变（只留单点）", len(df)))

    # --- 官方第 4 步：去泄漏（★ 索引必须与官方一致：在上一步的 df 上取 index）---
    m8 = os.path.join(root, "data", "prostab", "mmseq_mut_search_0.25.m8")
    leak_idx = []
    if os.path.exists(m8):
        with open(m8, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split("\t")
                if len(parts) > 1 and parts[1].strip().lstrip("-").isdigit():
                    leak_idx.append(int(parts[1]))
    leak_idx = sorted(set(leak_idx))
    log(f"去泄漏表命中行号 {len(leak_idx):,} 个（最大 {max(leak_idx) if leak_idx else '-'}），"
        f"当前 df 长度 {len(df):,}")
    assert not leak_idx or max(leak_idx) < len(df), \
        ("mmseq 表的行号超出当前 df 范围 —— 说明上游过滤口径和官方**不一致**，"
         "必须回去对齐，不能先改了再往下跑（否则去泄漏删错行、且不会报错）")
    is_leak = df.index.isin(leak_idx)
    funnel.append(("其中被 mmseq 标为同源泄漏的行", int(is_leak.sum())))

    # --- 官方第 5 步：官方 25% 聚类划分（按野生型蛋白）---
    with open(os.path.join(root, "data", "prostab", "mega_splits.pkl"), "rb") as fh:
        splits = pickle.load(fh)
    name2split = {}
    for sp, tag in (("train", "train"), ("val", "valid"), ("test", "test")):
        for w in splits[sp]:
            name2split[str(w)] = tag
    log(f"官方划分覆盖 {len(name2split):,} 个野生型（train/val/test）")
    # ★ 官方是 df.query('WT_name == @wt_name')，即**字符串完全相同**才算命中。
    #   所以直接映射即可，但要把"没被划到的蛋白"显式报出来（不能静默丢）。
    _names_in_df = set(df["WT_name"].astype(str))
    log(f"  df 里的蛋白 {len(_names_in_df):,} 个，"
        f"其中不在官方划分里的 {len(_names_in_df - set(name2split)):,} 个")

    wt_name = df["WT_name"].astype(str)
    # ★ 列名不能以下划线开头：pandas 的 itertuples 会把"以下划线开头"的列名
    #   改写成位置名（_1/_2...），命名元组里就取不到 r._split 了（实测 AttributeError）。
    df = df.assign(split_tag=wt_name.map(name2split))
    outside = df["split_tag"].isna().sum()
    funnel.append(("落在官方 train/val/test 之外的突变行", int(outside)))
    # ★ 测试集不删泄漏行（官方就是这样），train/val 删
    keep = (df["split_tag"].notna()) & (~(is_leak & df["split_tag"].isin(["train", "valid"])))
    df = df.loc[keep, :].reset_index(drop=True)
    funnel.append(("去泄漏 + 限官方划分后剩余", len(df)))

    # --- 野生型序列查找表：★ 严格照官方，在**去泄漏之后**的 df 上取 ---
    #   官方 megascale.py L114-127：先删 m8 泄漏行，再 query mut_type=='wt'；
    #   **取不到 wt 行的蛋白记进 removed_wt_names 并整个从该 split 移除**。
    #   我一开始想改成"在过滤前建表"（更宽容、少丢蛋白），但那样会让 train/valid
    #   的样本与官方口径不一致 -> 分数没法和已发表数字对话。
    #   所以照官方做，但把"因此被移除的蛋白"显式报数（官方也只是 log，不是静默）。
    wt_rows = df.loc[df["mut_type"].astype(str) == "wt", :]
    wt_seq_of = {}
    for w, g in wt_rows.groupby("WT_name"):
        wt_seq_of[str(w)] = str(g["aa_seq"].iloc[0])
    _mut_names = set(df.loc[df["mut_type"].astype(str) != "wt", "WT_name"].astype(str))
    missing_wt = sorted(_mut_names - set(wt_seq_of))
    log(f"野生型序列表：{len(wt_seq_of):,} 个（去泄漏后建立，与官方一致）")
    if missing_wt:
        log(f"  ★ 有突变行但缺 wt 行的蛋白 {len(missing_wt)} 个"
            f"（它们是被 m8 命中而删掉 wt 行的；官方同样会移除）"
            f"，例：{missing_wt[:4]}")
    funnel.append(("缺 wt 行而被移除的蛋白数", len(missing_wt)))

    # --- 解析突变 ---
    mut = df.loc[df["mut_type"].astype(str) != "wt", :].copy()
    rec = []
    bad = Counter()
    for r in mut.itertuples(index=False):
        w = str(r.WT_name)
        seq = wt_seq_of.get(w)
        if seq is None:
            bad["该蛋白没有 wt 行"] += 1
            continue
        mts = str(r.mut_type)
        # 形态：<wt><pos><mut>，如 S11A / A123C
        if len(mts) < 3 or not mts[1:-1].isdigit():
            bad["mut_type 形态异常"] += 1
            continue
        wta, pos1, mta = mts[0], int(mts[1:-1]), mts[-1]
        pos0 = pos1 - 1
        if not (0 <= pos0 < len(seq)):
            bad["位置越界"] += 1
            continue
        if seq[pos0] != wta:
            bad["wt 序列与 mut_type 不符"] += 1
            continue
        if str(r.aa_seq)[pos0] != mta or len(str(r.aa_seq)) != len(seq):
            bad["突变体序列与 mut_type 不符"] += 1
            continue
        if wta not in ALPHABET or mta not in ALPHABET:
            bad["非标准氨基酸"] += 1
            continue
        rec.append(dict(split=r.split_tag, wt_name=w, mut_type=mts, pos0=pos0,
                        wt_aa=wta, mut_aa=mta, seq_len=len(seq),
                        wt_seq=seq, mut_seq=str(r.aa_seq),
                        label=-float(r.ddG_ML), ddG_ML=float(r.ddG_ML),
                        dG_ML=(float(r.dG_ML) if str(r.dG_ML) not in ("-", "nan")
                               else np.nan)))
    log(f"解析成功 {len(rec):,} 条；丢弃明细 {dict(bad)}")
    funnel.append(("解析成功的单点突变", len(rec)))
    if not rec:
        raise SystemExit("一条都没解析出来，先跑 --inspect 看数据形态")

    m = pd.DataFrame(rec)
    # 同一条突变（同蛋白 + 同 mut_type）若被重复测量，官方是逐行保留的；
    # 这里保留全部行，但在 uid 里带上序号，避免后面按 uid 合并时静默丢行。
    m["uid"] = [f"{r.wt_name}|{r.mut_type}|{i}" for i, r in enumerate(m.itertuples())]
    m = m.sort_values(["split", "wt_name", "pos0"]).reset_index(drop=True)

    for sp in ("train", "valid", "test"):
        g = m.loc[m["split"] == sp]
        log(f"  {sp:6s} 突变 {len(g):>8,} 条  蛋白 {g['wt_name'].nunique():>5,} 个")

    # --- 待嵌入序列去重 ---
    seqs = {}
    for r in m.itertuples():
        seqs.setdefault(r.wt_seq, {"kind": "wt", "n_ref": 0})
        seqs[r.wt_seq]["n_ref"] += 1
        seqs.setdefault(r.mut_seq, {"kind": "mut", "n_ref": 0})
        seqs[r.mut_seq]["n_ref"] += 1
    sid_of = {s: f"s{i:07d}" for i, s in enumerate(sorted(seqs))}
    seq_df = pd.DataFrame([{"seq_id": sid_of[s], "seq": s, "kind": v["kind"],
                            "n_ref": v["n_ref"], "seq_len": len(s)}
                           for s, v in seqs.items()])
    n_wt_uniq = int((seq_df["kind"] == "wt").sum())
    log(f"\n待嵌入：唯一序列 {len(seq_df):,} 条（其中野生型 {n_wt_uniq:,} 条）"
        f"  总残基 {int(seq_df['seq_len'].sum()):,}")
    log("  ★ h_wt 只需按蛋白数跑一次（几百条），成本全在 h_mut")

    m["wt_seq_id"] = m["wt_seq"].map(sid_of)
    m["mut_seq_id"] = m["mut_seq"].map(sid_of)

    cols = ["uid", "split", "wt_name", "mut_type", "pos0", "wt_aa", "mut_aa",
            "seq_len", "label", "ddG_ML", "dG_ML", "wt_seq_id", "mut_seq_id"]
    m[cols].to_csv(os.path.join(out_dir, "mut.csv"), index=False)
    seq_df.to_csv(os.path.join(out_dir, "seqs.csv"), index=False)

    stats = {
        "funnel": [{"step": k, "n": int(v)} for k, v in funnel],
        "n_mut": int(len(m)),
        "n_wt_proteins": int(m["wt_name"].nunique()),
        "n_seq_to_embed": int(len(seq_df)),
        "n_wt_seq_unique": n_wt_uniq,
        "total_residues": int(seq_df["seq_len"].sum()),
        "by_split": {sp: int((m["split"] == sp).sum())
                     for sp in ("train", "valid", "test")},
        "label_mean": float(m["label"].mean()),
        "label_std": float(m["label"].std()),
        "dropped": {k: int(v) for k, v in bad.items()},
        "alphabet": ALPHABET,
        "note": "label = -ddG_ML（官方符号约定，负号不能丢）",
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    log("\n=== 漏斗 ===")
    for k, v in funnel:
        log(f"  {k:42s} {v:>10,}")
    log(f"\n标签：均值 {stats['label_mean']:+.3f}  标准差 {stats['label_std']:.3f}")
    log(f"产物：{out_dir}/mut.csv  {out_dir}/seqs.csv  {out_dir}/stats.json")
    log("BUILD_DONE")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=root)
    ap.add_argument("--out", default=None)
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "data")
    log("=" * 88)
    log("热稳定性 ΔΔG 数据集构建（MegaScale / 官方 25% 聚类划分 / 去泄漏）")
    log("=" * 88)
    if a.inspect:
        inspect(a.root)
        return
    build(a.root, out)


if __name__ == "__main__":
    main()
