"""把外部溶解度数据集整理成规范 csv，供 eval_cross_dataset.py 使用。

为什么要单独一步：不同来源的数据集标签形式差别很大（fasta 头里的 0/1、
csv 里的 true/false、甚至只是"可溶/不可溶"中文），如果直接在评估脚本里解析，
每换一个数据集就要改主流程。这里统一成一张表：

    seq_id, sequence, label, seq_len, source

其中 label 统一成 **1 = 可溶 / 0 = 不可溶**（二分类口径）。
如果原数据集是连续标签（0~100 的溶解度百分比），还可以带一列 `value`，
评估脚本会自动切成"回归口径"而不是"分类口径"。

★ 为什么必须去重 + 查标签冲突
  同一序列出现两次、却给了相反的标签 —— 这在新手拼数据集时极常见。
  留着它会让 AUC 的上限低于 1，而且**不报错**。这里直接拦掉。

支持的输入
  1) fasta，标签在头部：DeepSoluE 的 `>97522|1|training`（|1| 可溶 / |0| 不可溶）
  2) csv/tsv，自动识别列名（sequence/seq/序列；label/target/tag）

用法
    python prep_cross_dataset.py --in ../DeepSoluE/testing.fasta \
        --name deepsolue_test --out ../data/cross/deepsolue_test.csv
"""
import argparse
import os
import re
import sys

import pandas as pd

SEQ_COL_CANDS = ("sequence", "seq", "sequences", "序列", "protein", "amino_acid")
LABEL_COL_CANDS = ("label", "target", "tag", "y", "class", "solubility_label")
VALUE_COL_CANDS = ("value", "solubility", "solubility_pct", "score", "ddg", "y_value")


def read_fasta(path):
    """读 fasta，返回 [(header, seq), ...]。"""
    out = []
    name, buf = None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out.append((name, "".join(buf)))
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def label_from_header(header):
    """从 fasta 头里挖出 0/1 标签。返回 (seq_id, label)。

    已知格式：
      DeepSoluE  97522|1|training        → id=97522, label=1
      DeepSoluE  97522|0|training        → id=97522, label=0
      其他常见   sp|P12345|NAME_soluble  → 找 soluble/insoluble 关键词
    """
    parts = header.split("|")
    for i, p in enumerate(parts):
        if p.strip() in ("0", "1"):
            sid = parts[i - 1] if i > 0 else header.split()[0]
            return sid.strip(), int(p)
    low = header.lower()
    if "_insoluble" in low or "insoluble" in low:
        return header.split()[0], 0
    if "_soluble" in low or "soluble" in low:
        return header.split()[0], 1
    return header.split()[0], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="输入 fasta 或 csv/tsv")
    ap.add_argument("--name", required=True, help="数据集名，写进 source 列")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=20, help="短于此长度的丢掉")
    ap.add_argument("--max-len", type=int, default=2046, help="ESM-C 位置上限，超长会被截断")
    a = ap.parse_args()

    ext = os.path.splitext(a.inp)[1].lower()
    print("=" * 84)
    print(f"整理外部数据集 {a.name}   输入 {a.inp}")
    print("=" * 84)

    if ext in (".fa", ".fasta", ".faa", ".fna"):
        rows = read_fasta(a.inp)
        print(f"  读到 fasta 记录 {len(rows)} 条")
        recs = []
        n_nolabel = 0
        for h, s in rows:
            sid, lab = label_from_header(h)
            if lab is None:
                n_nolabel += 1
                continue
            recs.append(dict(seq_id=sid, sequence=s, label=lab))
        if n_nolabel:
            print(f"  ⚠️ {n_nolabel} 条头部里找不到 0/1 标签，已跳过")
        df = pd.DataFrame(recs)
    else:
        sep = "\t" if ext in (".tsv", ".tab") else ","
        df = pd.read_csv(a.inp, sep=sep)
        print(f"  读到表 {df.shape}，列：{list(df.columns)}")
        low = {c.lower(): c for c in df.columns}
        sc = next((low[c] for c in SEQ_COL_CANDS if c in low), None)
        lc = next((low[c] for c in LABEL_COL_CANDS if c in low), None)
        vc = next((low[c] for c in VALUE_COL_CANDS if c in low), None)
        if sc is None:
            raise SystemExit(f"找不到序列列。候选：{SEQ_COL_CANDS}；实际：{list(df.columns)}")
        out = pd.DataFrame({"sequence": df[sc].astype(str)})
        out["seq_id"] = df.index.astype(str)
        if lc is not None:
            out["label"] = df[lc]
        if vc is not None:
            out["value"] = df[vc]
        if lc is None and vc is None:
            raise SystemExit("既没有标签列也没有数值列 —— 至少需要一个")
        df = out

    if df.empty:
        raise SystemExit("解析后一条都没有 —— 检查输入格式")

    # ---- 标签归一化到 0/1 ----
    if "label" in df.columns:
        def norm(x):
            if isinstance(x, str):
                s = x.strip().lower()
                if s in ("1", "true", "yes", "soluble", "sol", "可溶", "positive"):
                    return 1
                if s in ("0", "false", "no", "insoluble", "insol", "不可溶", "negative"):
                    return 0
                try:
                    return int(float(s))
                except ValueError:
                    return None
            return x
        df["label"] = df["label"].map(norm)
        n_bad = int(df["label"].isna().sum())
        if n_bad:
            print(f"  ⚠️ {n_bad} 条标签无法识别，已丢掉")
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)

    # ---- 清洗序列 ----
    df["sequence"] = (df["sequence"].astype(str)
                      .str.strip()
                      .str.replace(r"\s+", "", regex=True)
                      .str.upper())
    n0 = len(df)
    bad = ~df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYXBZUO]+", na=False)
    if bad.any():
        print(f"  ⚠️ {int(bad.sum())} 条含非法氨基酸字符，已丢掉"
              f"（例：{df.loc[bad, 'sequence'].iloc[0][:40]!r}）")
        df = df[~bad].copy()
    df["seq_len"] = df["sequence"].str.len()
    too_short = df["seq_len"] < a.min_len
    too_long = df["seq_len"] > a.max_len
    if too_short.any() or too_long.any():
        print(f"  ⚠️ 长度越界丢掉：<{a.min_len} 共 {int(too_short.sum())} 条、"
              f">{a.max_len} 共 {int(too_long.sum())} 条（后者会被 ESM-C 截断）")
        df = df[~(too_short | too_long)].copy()

    # ---- 序列去重 + 标签冲突检查（★ 不报错但会静默压低 AUC 上限）----
    dup_n = int(df["sequence"].duplicated().sum())
    grp = df.groupby("sequence")["label"].nunique() if "label" in df.columns else None
    conflict = int((grp > 1).sum()) if grp is not None else 0
    if dup_n:
        print(f"  {dup_n} 条序列重复 → 去重（保留第一条）")
        df = df.drop_duplicates("sequence", keep="first").copy()
    if conflict:
        print(f"  ⛔ {conflict} 条序列有**冲突标签**（同序列既有 0 又有 1）—— 全部丢掉。")
        print("     留着会让 AUC 的理论上限低于 1，而且不报错。")
        conf_seqs = set(grp[grp > 1].index)
        df = df[~df["sequence"].isin(conf_seqs)].copy()

    df["source"] = a.name
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cols = [c for c in ("seq_id", "sequence", "label", "value", "seq_len", "source")
            if c in df.columns]
    df = df[cols].reset_index(drop=True)
    df.to_csv(a.out, index=False, encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"写出 {a.out}   原始 {n0} 条 → 最终 {len(df)} 条")
    if "label" in df.columns:
        vc = df["label"].value_counts().to_dict()
        print(f"  标签分布：{vc}"
              f"（可溶 {vc.get(1, 0)} / 不可溶 {vc.get(0, 0)}，"
              f"正例比 {vc.get(1, 0) / max(len(df), 1) * 100:.1f}%）")
    print(f"  序列长度：min {df.seq_len.min()} / 中位 {int(df.seq_len.median())} / "
          f"max {df.seq_len.max()}")
    print("PREP_CROSS_DONE")


if __name__ == "__main__":
    main()
