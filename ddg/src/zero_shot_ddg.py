"""热稳定性 ΔΔG · **零样本（zero-shot）ESM 基线**：不做任何训练，直接用预训练
掩码语言模型的"掩码边际"打分。

★ 它回答的是一个跟主结果完全不同、但面试一定会问的问题
    "你这个 0.65 的 Spearman，是**训练/fine-tune 带来的**，还是**预训练模型本来就会**？"
    做法：把野生型序列的突变位点遮住（<mask>），跑一次前向，取该位上
    `logit(突变氨基酸) − logit(野生型氨基酸)` 当"这个突变有多好"的分数。
    这个数字**完全不含任何标签信息**，是真正的零样本。

★ 口径（与主结果严格对齐，否则不可比）
    - 只在**同一批测试集**（mega_splits 25% 划分的 test，56,344 条）上算；
    - 同一套标签 `label = -ddG_ML`；
    - 同一批蛋白（28 个 DMS 深度扫描蛋白，44~68 aa）；
    - 唯一 (蛋白, 位点) 只有约 1,562 个 ⇒ 只需约 1,562 次前向，成本极低。

★ 方向必须验证，不能想当然
    z = logit(mut) − logit(wt)   **越大 = 模型越偏好突变体 = 越稳定**
    label = -ddG_ML              **越小 = 越稳定**（本项目约定：label<0 = 提高稳定性）
    ⇒ 预期 Spearman(z, label) < 0。脚本会把两个方向都打出来，
      若 |rho| 很小或符号不对，就是位置索引对错了 —— 那时不许出图，必须先查。

★ 索引对齐（本项目最容易错的地方）
    ESMC 的 token 序列是 [<cls>, aa_1, ..., aa_L, <eos>] ⇒
    序列下标 pos0（0-based）对应 token 下标 pos0 + 1。
    脚本用"**遮蔽后模型能否猜回野生型氨基酸**"来实证这一点：
    猜回率高 ⇒ 索引对；猜回率 ~5%（=1/20 随机）⇒ 索引错。

用法
    python src/zero_shot_ddg.py --enc esmc600m --probe        # 只探针，不跑全量
    python src/zero_shot_ddg.py --enc esmc600m                # 全量，存 csv
"""
import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# ★ 不写死缓存路径：优先用环境变量 HF_HOME，否则落到用户级默认目录。
os.environ["HF_HOME"] = os.environ.get("HF_HOME") or os.path.expanduser(
    "~/.cache/huggingface")

import numpy as np
import pandas as pd
import torch

# ★ 权重目录从环境变量取；不硬编码任何机器上的绝对路径。
_HF = os.environ["HF_HOME"]
REPO_OF = {"esmc300m": os.environ.get("ESMC_REPO_300M") or os.path.join(_HF, "ESMC-300M"),
           "esmc600m": os.environ.get("ESMC_REPO_600M") or os.path.join(_HF, "ESMC-600M"),
           "esmc6b":   os.environ.get("ESMC_REPO_6B")   or os.path.join(_HF, "ESMC-6B")}
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def load_mlm(repo, device="cuda"):
    """★ 必须用 AutoModelForMaskedLM —— AutoModel 没有 lm_head，拿不到 logits。"""
    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    print(f"[环境] transformers {transformers.__version__} | 权重 {repo}", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForMaskedLM.from_pretrained(repo, dtype=torch.float32)
    model = model.eval().to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[模型] AutoModelForMaskedLM 参数 {n/1e9:.3f}B  device={device}", flush=True)
    return model, tok


def check_tok(tok):
    print(f"[tokenizer] 类={type(tok).__name__}  mask={tok.mask_token!r} "
          f"(id={tok.mask_token_id})  cls={tok.cls_token!r}  eos={tok.eos_token!r}  "
          f"pad={tok.pad_token!r}  vocab={tok.vocab_size}", flush=True)
    ids = {}
    for aa in AA20:
        for form in (aa, f"<{aa}>", f" {aa}"):
            i = tok.convert_tokens_to_ids(form)
            if i is not None and i != tok.unk_token_id:
                ids[aa] = (form, i)
                break
    miss = [a for a in AA20 if a not in ids]
    print(f"[tokenizer] 20 种氨基酸可解析 {len(ids)}/20" +
          (f"   ★ 缺 {miss}" if miss else ""), flush=True)
    print("[tokenizer] 例：" + "  ".join(f"{a}->{ids[a][0]}({ids[a][1]})"
                                        for a in "ACDEFG" if a in ids), flush=True)
    if tok.mask_token_id is None:
        raise SystemExit("★ tokenizer 没有 mask token，掩码边际法不成立，需要换做法")
    return {a: ids[a][1] for a in ids}


def encode(aa_id, seq):
    return np.array([aa_id.get(c, -1) for c in seq], dtype=np.int64)


@torch.no_grad()
def logits_masked(model, tok, seqs, positions, device, batch=64):
    """对每个 (seq, position) 做一次"该位遮住"的前向，返回该位 20 维 logits。"""
    out = []
    order = np.argsort([len(s) for s in seqs])          # 按长度批，减少 padding
    seqs = [seqs[i] for i in order]
    positions = [positions[i] for i in order]
    for b in range(0, len(seqs), batch):
        ss = seqs[b:b + batch]
        ps = positions[b:b + batch]
        masked = [s[:p] + tok.mask_token + s[p + 1:] for s, p in zip(ss, ps)]
        enc = tok(masked, return_tensors="pt", padding=True, add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        logit = model(**enc).logits.float()             # (B, T, V)
        for j, p in enumerate(ps):
            out.append(logit[j, p + 1].cpu().numpy())   # ★ +1：跳过 <cls>
    res = [None] * len(order)
    for k, i in enumerate(order):
        res[i] = out[k]
    return np.stack(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="esmc600m")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--data-dir", default=None, help="含 mut.csv / seqs.csv（默认脚本上一级 data/）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只取前 N 个位点（冒烟）")
    ap.add_argument("--probe", action="store_true", help="只做校验与探针，不写结果")
    a = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    ddir = a.data_dir or os.path.join(ROOT, "data")
    repo = a.repo or REPO_OF[a.enc]
    out = a.out or os.path.join(ROOT, "_server_results", "zeroshot", f"zeroshot_{a.enc}.csv")

    t0 = time.time()
    print("=" * 100)
    print(f"ΔΔG · 零样本 masked-marginal 基线   编码器 {a.enc}")
    print("=" * 100)

    mut = pd.read_csv(os.path.join(ddir, "mut.csv"))
    seqs = pd.read_csv(os.path.join(ddir, "seqs.csv"))
    te = mut[mut["split"] == "test"].reset_index(drop=True)
    print(f"[数据] 测试集 {len(te):,} 条 / {te['wt_name'].nunique()} 个蛋白")

    smap = dict(zip(seqs.loc[seqs["kind"] == "wt", "seq_id"],
                    seqs.loc[seqs["kind"] == "wt", "seq"]))
    print(f"[数据] 野生型序列 {len(smap)} 条（kind=='wt'）")

    miss_seq = ~te["wt_seq_id"].isin(smap)
    if miss_seq.any():
        print(f"  ★ {int(miss_seq.sum())} 条找不到野生型序列，排除")
        te = te[~miss_seq].reset_index(drop=True)

    # ---- 位置/残基一致性：这是索引口径的第一道体检 ----
    wtseq = te["wt_seq_id"].map(smap).to_numpy()
    pos0 = te["pos0"].to_numpy(np.int64)
    wa = te["wt_aa"].astype(str).to_numpy()
    ma = te["mut_aa"].astype(str).to_numpy()
    bad_range = (pos0 < 0) | (pos0 >= np.array([len(s) for s in wtseq]))
    obs = np.array([s[p] if 0 <= p < len(s) else "?" for s, p in zip(wtseq, pos0)])
    n_match = int((obs == wa).sum())
    print(f"[体检] pos0 越界 {int(bad_range.sum())} 条；"
          f"野生型序列在该位的残基与 wt_aa 一致 {n_match:,}/{len(te):,} "
          f"({n_match/len(te)*100:.2f}%)")
    if n_match < len(te) * 0.999:
        raise SystemExit(f"★ 只有 {n_match/len(te)*100:.2f}% 对得上 ⇒ pos0 的基准"
                         f"（0-based？指向什么？）与本脚本假设不符，先查清再跑。")
    bad_aa = ~np.isin(ma, list(AA20))
    if bad_aa.any():
        print(f"  ★ 突变氨基酸里有 {int(bad_aa.sum())} 条非 20 标准氨基酸，排除")
        keep = ~bad_aa
        te, wtseq, pos0, wa, ma = (te[keep].reset_index(drop=True), wtseq[keep],
                                   pos0[keep], wa[keep], ma[keep])

    # ---- 唯一 (蛋白序列, 位点) ----
    key = pd.DataFrame({"sid": te["wt_seq_id"].to_numpy(),
                        "p": pos0}).drop_duplicates().reset_index(drop=True)
    print(f"[任务] 唯一 (蛋白, 位点) = {len(key):,} 个"
          f"（{len(te):,} 条突变里平均每位点 {len(te)/len(key):.1f} 条替换）")
    if a.limit:
        key = key.head(a.limit).reset_index(drop=True)
        print(f"  --limit {a.limit} ⇒ 只跑 {len(key)} 个位点（冒烟）")
    key["seq"] = key["sid"].map(smap)
    key["wt"] = [s[p] for s, p in zip(key["seq"], key["p"])]

    model, tok = load_mlm(repo, a.device)
    aa_id = check_tok(tok)

    # ---- 探针：遮蔽后能否猜回野生型（证明 +1 的索引对了）----
    nprobe = min(64, len(key))
    lp = logits_masked(model, tok, key["seq"].tolist()[:nprobe],
                       key["p"].tolist()[:nprobe], a.device, a.batch)
    cols = np.array([aa_id.get(c, -1) for c in AA20])
    hit = 0
    for i in range(nprobe):
        v = lp[i][cols]
        if AA20[int(np.argmax(v))] == key["wt"].iloc[i]:
            hit += 1
    print(f"[探针] 遮蔽后 argmax 猜回野生型：{hit}/{nprobe} = {hit/nprobe*100:.1f}%"
          f"（随机 = 5.0%）⇒ " +
          ("索引对齐正确" if hit / nprobe > 0.25 else "★ 索引疑似错位，先停下查"))
    if hit / nprobe <= 0.25:
        raise SystemExit("★ 探针没通过，不出结果")
    if a.probe:
        print("PROBE_ONLY_DONE")
        return

    # ---- 全量 ----
    LP = logits_masked(model, tok, key["seq"].tolist(), key["p"].tolist(),
                       a.device, a.batch)
    pos_of = {(r.sid, r.p): i for i, r in enumerate(key.itertuples())}
    idx = np.array([pos_of[(s, p)] for s, p in zip(te["wt_seq_id"], pos0)])
    l_wt = LP[idx, np.array([aa_id[c] for c in wa])]
    l_mu = LP[idx, np.array([aa_id[c] for c in ma])]
    z = l_mu - l_wt
    print(f"[打分] z = logit(mut) − logit(wt)：均值 {z.mean():+.4f}  标准差 {z.std():.4f}  "
          f"分位 5/50/95% = {np.percentile(z,5):+.3f} / {np.percentile(z,50):+.3f} / "
          f"{np.percentile(z,95):+.3f}")

    # ---- 方向实证（不靠推理）----
    from scipy.stats import spearmanr
    lab = te["label"].to_numpy(float)
    r_z, p_z = spearmanr(z, lab)
    print(f"[方向] Spearman(z, label) = {r_z:+.4f} (p={p_z:.2e})")
    print(f"       预期为**负**（z 越大越稳定，label 越小越稳定）⇒ "
          + ("符号正确 ✓" if r_z < 0 else "★ 符号为正，需求确认口径"))
    print(f"       与我们头部的口径对齐（越大=越不稳定）时：zhat = -z，"
          f"Spearman(zhat, label) = {-r_z:+.4f}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"uid": te["uid"], "z": z, "label": lab,
                  "wt_name": te["wt_name"], "pos0": pos0,
                  "wt_aa": wa, "mut_aa": ma}).to_csv(out, index=False)
    print(f"[产物] {out}（{len(te):,} 行）  用时 {time.time()-t0:.0f}s")
    print("ZEROSHOT_DONE")


if __name__ == "__main__":
    main()
