"""数据准备：把公开的 eSOL 溶解度数据整理成可直接训练的回归数据集。

★ 数据来源（全部实测可达，2026-09-24 核实）
  ① 标签：eSOL 官方归档
     https://dbarchive.biosciencedbc.jp/data/esol/LATEST/esol.zip   (193 KB)
     → esol.csv，4132 行，其中 "Solubility (%)" 非空 3173 条（连续值，0~147%）
     这是 Niwa et al. 2009 PNAS 的原始测量：E. coli 蛋白用 PURE 无细胞体系表达，
     溶解度 = 上清中的蛋白量 / 总蛋白量（%）。
     ⚠️ 该归档地址在部分网络环境不可达 → 在**有外网的机器**上跑本脚本，
        产物 data/esol_reg.csv 再上传服务器。
     许可：CC BY-SA 2.1 Japan，署名格式见文末 ATTRIBUTION。

  ② 序列：UniProt E. coli K-12 参考蛋白组（reviewed）
     https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000000625
       &format=tsv&fields=accession,gene_oln,gene_primary,gene_synonym,length,sequence
     → 4403 条，含「位点名(b 号) / 主基因名 / 别名」三种索引，共 2 万余键
     esol.csv 只给基因名、不给序列，必须靠它映射。
     本脚本实测映射成功率 3167/3173 = 99.8%。

用法：
    python fetch_data.py                       # 默认写到 ../data
    python fetch_data.py --data-dir ../data --no-download   # 复用已下载的原始文件
"""
import argparse
import io
import os
import re
import sys
import time
import zipfile

import pandas as pd
import requests

ESOL_URL = "https://dbarchive.biosciencedbc.jp/data/esol/LATEST/esol.zip"
UNIPROT_URL = ("https://rest.uniprot.org/uniprotkb/stream"
               "?query=proteome:UP000000625"
               "&format=tsv"
               "&fields=accession,gene_oln,gene_primary,gene_synonym,length,sequence")

ATTRIBUTION = ('eSOL © Hideki Taguchi (Tokyo Institute of Technology) '
               'licensed under CC Attribution-Share Alike 2.1 Japan')

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def log(msg):
    print(msg, flush=True)


def download(url, path, timeout=180, retries=3):
    """带重试的下载；已存在且非空则跳过。"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        log(f"  [复用] {os.path.basename(path)}  ({os.path.getsize(path):,} 字节)")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for k in range(1, retries + 1):
        try:
            log(f"  [下载] {os.path.basename(path)}  (第 {k} 次) ...")
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            with open(path, "wb") as fh:
                fh.write(r.content)
            log(f"  [完成] {os.path.getsize(path):,} 字节")
            return path
        except Exception as e:               # noqa: BLE001
            log(f"  [失败] {type(e).__name__}: {e}")
            if k == retries:
                raise
            time.sleep(3 * k)
    return path


# ----------------------------------------------------------------------------
# 1. 标签：eSOL
# ----------------------------------------------------------------------------
def load_esol(raw_dir):
    """读 esol.csv，只保留有连续溶解度标签的行。"""
    zpath = download(ESOL_URL, os.path.join(raw_dir, "esol.zip"))
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        assert "esol.csv" in names, f"zip 内容异常：{names}"
        raw = z.read("esol.csv")
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    log(f"  esol.csv: {len(df)} 行 × {len(df.columns)} 列")

    df["solubility_pct"] = pd.to_numeric(df["Solubility (%)"], errors="coerce")
    n_all, n_lab = len(df), int(df["solubility_pct"].notna().sum())
    log(f"  有溶解度标签: {n_lab} / {n_all}")
    return df[df["solubility_pct"].notna()].copy()


# ----------------------------------------------------------------------------
# 2. 序列：UniProt
# ----------------------------------------------------------------------------
def load_uniprot_index(raw_dir):
    """建基因名 → 序列 的索引（含位点名 / 主名 / 别名）。"""
    tpath = download(UNIPROT_URL, os.path.join(raw_dir, "ecoli_k12_uniprot.tsv"))
    up = pd.read_csv(tpath, sep="\t", dtype=str).fillna("")
    log(f"  UniProt: {len(up)} 条")

    idx = {}
    for _, r in up.iterrows():
        seq = r["Sequence"]
        for col in ("Gene Names (ordered locus)", "Gene Names (primary)",
                    "Gene Names (synonym)"):
            for tok in re.split(r"[;\s]+", r[col]):
                tok = tok.strip()
                if tok:
                    idx.setdefault(tok, seq)
        idx.setdefault(r["Entry"], seq)
    log(f"  索引键数: {len(idx)}")
    return idx


def map_sequence(row, idx, cols):
    """按优先级拿基因名去索引里找序列。"""
    cands = []
    for c in cols:
        v = str(row.get(c, "")).strip()
        if v and v.lower() != "nan":
            cands.append(v)
    syn = str(row.get("Synonyms of locus names K-12", ""))
    if syn and syn.lower() != "nan":
        cands += [t.strip() for t in re.split(r"[;,\s]+", syn) if t.strip()]
    for t in cands:
        if t in idx:
            return idx[t], t
    return None, None


# ----------------------------------------------------------------------------
# 3. 清洗
# ----------------------------------------------------------------------------
def clean(df):
    """去重 / 范围截断 / 序列合法性检查。返回 (清洗后 df, 统计 dict)。"""
    st = {"n_in": len(df)}

    # --- 序列基本合法性 ---
    df["seq_len"] = df["sequence"].str.len()
    bad_aa = ~df["sequence"].str.fullmatch(r"[A-Z]+")
    st["n_bad_chars"] = int(bad_aa.sum())
    if st["n_bad_chars"]:
        log(f"  [警告] {st['n_bad_chars']} 条序列含非字母字符，已剔除")
        df = df[~bad_aa]
    nonstd = df["sequence"].apply(lambda s: sorted(set(s) - STANDARD_AA))
    st["n_nonstd_aa"] = int((nonstd.str.len() > 0).sum())
    if st["n_nonstd_aa"]:
        ex = nonstd[nonstd.str.len() > 0].head(3).tolist()
        log(f"  [提示] {st['n_nonstd_aa']} 条含非标准氨基酸，例：{ex}")

    # --- 标签范围：原始最大 147%，按"可溶比例"的物理含义截断到 100 ---
    over = int((df["solubility_pct"] > 100).sum())
    st["n_over_100"] = over
    df["solubility_raw"] = df["solubility_pct"]
    df["solubility_pct"] = df["solubility_pct"].clip(0, 100)
    log(f"  >100% 的 {over} 条已截断到 100（原始值保留在 solubility_raw）")

    # --- 同序列去重：同一序列被测多次 → 取均值；同时用它估计"测量噪声" ---
    # ★ 必须把两种"重复"分开，否则会把旁系同源误当成实验误差：
    #   (a) 同一基因名出现多次  → 真正的重复测量，其差值 = 测量噪声下界
    #   (b) 不同基因名但序列相同 → 旁系同源/重复条目，序列相同但测量对象不同
    g = df.groupby("sequence")["solubility_pct"]
    dup = g.size()
    dup = dup[dup > 1]
    st["n_dup_groups"] = int(len(dup))
    st["n_dup_rows"] = int(dup.sum())

    rep_spreads, para_groups = [], []
    for s in dup.index:
        rows = df[df["sequence"] == s]
        sp = float(rows["solubility_pct"].max() - rows["solubility_pct"].min())
        genes = sorted(set(rows["Gene name K-12"].astype(str)))
        if len(genes) == 1:                      # (a) 同基因重复测量
            rep_spreads.append((sp, genes[0], rows["solubility_pct"].tolist()))
        else:                                    # (b) 旁系同源
            para_groups.append((sp, genes, rows["solubility_pct"].tolist()))

    st["n_replicate_groups"] = len(rep_spreads)
    st["n_paralog_groups"] = len(para_groups)
    st["replicate_details"] = sorted(rep_spreads, reverse=True)
    st["paralog_details"] = sorted(para_groups, reverse=True)

    if rep_spreads:
        sps = [x[0] for x in rep_spreads]
        st["noise_median"] = float(pd.Series(sps).median())
        st["noise_max"] = float(max(sps))
        log(f"  真重复测量（同基因多次）：{len(rep_spreads)} 组，"
            f"组内极差 中位 {st['noise_median']:.1f} 最大 {st['noise_max']:.1f}")
        for sp, gn, vals in st["replicate_details"]:
            log(f"      {gn:10s} 测得 {vals}  → 极差 {sp:.0f}")
    else:
        st["noise_median"] = st["noise_max"] = float("nan")

    if para_groups:
        log(f"  旁系同源（不同基因、同序列）：{len(para_groups)} 组，已合并，"
            f"例：{para_groups[0][1]} 测得 {para_groups[0][2]}")

    agg = df.groupby("sequence").agg(
        solubility_pct=("solubility_pct", "mean"),
        solubility_raw=("solubility_raw", "mean"),
        n_measure=("solubility_pct", "size"),
        gene=("Gene name K-12", "first"),
        esol_jw=("JW_ID", "first"),
        esol_bnum=("B number", "first"),
    ).reset_index()
    st["n_out"] = len(agg)
    log(f"  去重后：{len(agg)} 条")
    return agg, st


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data-dir", default=os.path.join(here, "..", "data"))
    ap.add_argument("--no-download", action="store_true",
                    help="只用 raw/ 下已存在的原始文件")
    a = ap.parse_args()

    data_dir = os.path.abspath(a.data_dir)
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    log("=" * 78)
    log("eSOL 回归数据集构建")
    log("=" * 78)

    log("\n[1/4] 拉取标签（eSOL 官方归档）")
    esol = load_esol(raw_dir)

    log("\n[2/4] 拉取序列（UniProt E. coli K-12 蛋白组）")
    idx = load_uniprot_index(raw_dir)

    log("\n[3/4] 基因名 → 序列 映射")
    cols = ["B number", "JW_ID", "Locus name K-12", "Gene name K-12"]
    got = esol.apply(lambda r: map_sequence(r, idx, cols), axis=1)
    esol["sequence"] = [g[0] for g in got]
    esol["matched_key"] = [g[1] for g in got]
    hit = esol["sequence"].notna()
    log(f"  映射成功 {int(hit.sum())} / {len(esol)} = {100 * hit.mean():.1f}%")
    miss = esol.loc[~hit, "Gene name K-12"].tolist()
    if miss:
        log(f"  未命中 {len(miss)} 条（原样记录，不静默丢弃）：{miss[:12]}")

    log("\n[4/4] 清洗")
    clean_in = esol[hit].copy()
    df, st = clean(clean_in)

    # --- 落盘 ---
    out_csv = os.path.join(data_dir, "esol_reg.csv")
    df["solubility_frac"] = df["solubility_pct"] / 100.0
    df["seq_len"] = df["sequence"].str.len()
    df.to_csv(out_csv, index=False)

    rej = esol[~hit][["JW_ID", "Gene name K-12", "Locus name K-12", "solubility_pct"]].copy()
    rej["reject_reason"] = "gene_name_not_in_uniprot"
    rej.to_csv(os.path.join(data_dir, "rejected.csv"), index=False)

    # --- 数据卡 ---
    v = df["solubility_pct"]
    card = f"""# 数据卡：eSOL 溶解度回归集

生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}
生成脚本：`src/fetch_data.py`

## 来源
| 部分 | 来源 | 许可 |
|---|---|---|
| 溶解度标签 | eSOL 官方归档 `dbarchive.biosciencedbc.jp/data/esol/LATEST/esol.zip` | CC BY-SA 2.1 JP |
| 蛋白序列 | UniProt E. coli K-12 参考蛋白组 `UP000000625`（reviewed，4403 条） | CC BY 4.0 |

署名（再分发时**必须**保留）：
> {ATTRIBUTION}

## 条数流水账
| 步骤 | 条数 |
|---|---|
| esol.csv 原始行数 | 4132 |
| 其中 `Solubility (%)` 非空 | 3173 |
| 基因名 → UniProt 序列 映射成功 | {int(hit.sum())} |
| 去重后（按序列合并多次测量） | {st['n_out']} |

未命中基因名 {len(miss)} 条 → `rejected.csv`（原因：UniProt 里查不到该基因名）

## 标签分布（单位：%，已截断到 [0,100]）
| 统计 | 值 |
|---|---|
| 均值 | {v.mean():.2f} |
| 标准差 | {v.std():.2f} |
| 中位 | {v.median():.1f} |
| 最小 / 最大 | {v.min():.1f} / {v.max():.1f} |
| 唯一取值个数 | {v.nunique()} |

原始 >100% 的条目数：{st['n_over_100']}（SDS-PAGE 定量的固有误差，已截断）

## ★ 测量噪声下界（重要）
同一序列被测多次，分两类：

### (a) 真重复测量（同一基因名出现多次）—— 这才是实验误差
{st['n_replicate_groups']} 组，组内极差 **中位 {st.get('noise_median', float('nan')):.1f}　最大 {st.get('noise_max', float('nan')):.1f}**

{chr(10).join(f"| `{gn}` | {vals} | {sp:.0f} |" for sp, gn, vals in st.get('replicate_details', [])) or '（无）'}

### (b) 旁系同源（不同基因名、序列却完全相同）
{st['n_paralog_groups']} 组，已合并（序列相同 → 特征相同，无法区分，只能取均值）

{chr(10).join(f"| {'/'.join(gs)} | {vals} | {sp:.0f} |" for sp, gs, vals in st.get('paralog_details', [])) or '（无）'}

## ★★ 这条对模型的含义（必须写进报告）
**同一个蛋白、同样的实验，重复测量之间就能差 {st.get('noise_median', float('nan')):.0f} 个百分点（最大 {st.get('noise_max', float('nan')):.0f}）。**

推论：
1. **RMSE / MAE 存在无法突破的下限** —— 模型再准也压不到这个噪声以下。
   报 RMSE 时必须对着这个数量级解读，否则会把"实验噪声"误判成"模型不行"。
2. **主指标应该用 Spearman（排序相关）** —— 工程上要的是"哪几个突变/蛋白更可能可溶"
   的排序，而排序对这种随机噪声不敏感。
3. 这也解释了为什么文献在 eSOL 上做回归，R² 只报到 0.41 左右
   （Han et al. 2019, Bioinformatics 35:4640）。

## 序列
| 统计 | 值 |
|---|---|
| 长度 最小/中位/最大 | {df['seq_len'].min()} / {df['seq_len'].median():.0f} / {df['seq_len'].max()} |
| 含非标准氨基酸的条数 | {st['n_nonstd_aa']} |

## 列说明
| 列 | 含义 |
|---|---|
| `sequence` | 蛋白氨基酸序列 |
| `solubility_pct` | 溶解度百分比（主标签，0~100） |
| `solubility_frac` | 同上，归一化到 0~1 |
| `solubility_raw` | 截断前的原始值（可 >100） |
| `n_measure` | 该序列被独立测量的次数 |
| `gene` / `esol_jw` / `esol_bnum` | 溯源用标识 |
| `matched_key` | 靠哪个别名匹配上的（便于核查） |

## ★ 已知局限（必须写进报告）
1. **只有一个物种**（E. coli K-12，3167 个蛋白）。这些蛋白彼此同源程度高，
   **随机划分会有严重同源泄漏** → 必须做同源感知划分（见 `make_splits.py`）。
2. **标签是"体外无细胞体系的溶解度"**，不等于"在你自己宿主里表达时的可溶性"。
   换宿主、换温度、换标签都会变。
3. 测量方式是 SDS-PAGE 定量，有 ±若干 % 的噪声（见上）。
4. 3,167 条属于小数据，**必须用交叉验证 + 多种子集成**报数。
"""
    with open(os.path.join(data_dir, "dataset_card.md"), "w", encoding="utf-8") as fh:
        fh.write(card)

    log(f"\n写出 {out_csv}")
    log(f"写出 {os.path.join(data_dir, 'dataset_card.md')}")
    log(f"写出 {os.path.join(data_dir, 'rejected.csv')}")
    log("\n最终数据集：%d 条，标签 %s" % (len(df), f"{v.min():.0f}~{v.max():.0f}%"))
    log("FETCH_DONE")


if __name__ == "__main__":
    main()
