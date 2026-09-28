"""跨数据集验证 —— 回答"换一个实验室的数据集，模型还剩多少"。

★ 为什么这是整个项目最该补的一个实验
   在 eSOL 上把 Spearman 做到 0.70，只证明了"在 eSOL 这个口径下排得对"。
   它没能回答审稿人/面试官一定会问的那句话：
       「这到底是学到了溶解度的规律，还是只学到了 eSOL 的标注口径？」

   做法：拿一个**训练时完全没见过的外部数据集**，直接用 eSOL 训好的模型去预测。
   本项目用 DeepSoluE 的独立测试集（二分类标签：可溶 1 / 不可溶 0）。

★ 两步走，缺一不可
   ① **先量同源重叠**。两个数据集如果来自同一物种/同一批文献，很可能有大量近亲。
      有近亲就有泄漏，跨数据集的分数也会虚高 —— 和训练集内的泄漏是同一个道理。
      这里用 blastp 把外部序列打回 eSOL 的 **train**（注意：不是全量！），
      报 25%/30%/40%/50% 四档重叠率。
   ② **分层报指标**。把外部测试集切成「有同源近亲」与「无同源近亲」两半，
      分别报 AUC。**只有"无近亲"那一半才是真正的泛化能力**。
      两半的差值 = "同源泄漏在跨数据集场景下虚高了多少"。

★ 回归模型怎么评二分类数据集
   我们的模型输出连续的溶解度（0~1）。外部数据集是二分类。
   直接把预测值当**打分**算 AUC（ROC 曲线下面积）——
   即"可溶蛋白的预测值是否普遍高于不可溶蛋白"。
   这也正好对应我们主张的"工程上要的是排序"。
   为了可比，同时报两条基线：
     · 尺子：用 eSOL train 特征**同超参**重拟合的 Ridge（冻结嵌入线性探针）
     · 最便宜的先验：只用序列长度（越长越可能不可溶）
   只有同时超过这两条线，跨数据集的分数才有意义。

★ AUC 自己算，不依赖 sklearn
   AUC 等价于 Mann-Whitney U 统计量：AUC = (正例秩和 − n⁺(n⁺+1)/2) / (n⁺·n⁻)。
   并列值取平均秩。这样在任何"借来的环境"里都能跑，不受包缺失影响。

用法
    python eval_cross_dataset.py --csv data/cross/deepsolue_test.csv \
        --ckpt runs/mlp_esmc600m_hand_homology/seed0/best.ckpt \
        --repo "$ESMC_REPO_600M" --out runs/_cross/deepsolue_test
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# ★ 不写死缓存路径：优先用环境变量 HF_HOME，否则落到用户级默认目录。
os.environ["HF_HOME"] = os.environ.get("HF_HOME") or os.path.expanduser(
    "~/.cache/huggingface")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import load_esmc, pool_batch                  # noqa: E402
from handcrafted import handcrafted_features                        # noqa: E402
from model import Normalizer, SolubilityHead                        # noqa: E402

THRESHOLDS = (25, 30, 40, 50)

SUBSET_CN = {"all": "全部外部测试集", "homolog": "有同源近亲的子集",
             "clean": "无同源近亲的子集（真泛化）"}


# ---------------------------------------------------------------- AUC
def auc_score(y, s):
    """ROC-AUC，用 Mann-Whitney U 的秩公式算（并列取平均秩）。

    为什么不用 sklearn：这个函数 5 行就够，而且借来的环境常常没有 sklearn。
    公式：AUC = (正例的秩和 − n⁺(n⁺+1)/2) / (n⁺ · n⁻)
      · 直觉：AUC = 随机抽一个正例、一个负例，正例打分更高的概率
      · 分子就是从正例秩和里去掉"正例之间互相比较"贡献的那部分
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ss = s[order]
    i = 0
    while i < len(ss):                      # 并列值给平均秩
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def boot_ci(y, s, n_boot=2000, seed=0, alpha=0.05):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = auc_score(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(lo), float(hi)


def paired_auc(y, sA, sB, n_boot=2000, seed=0, alpha=0.05):
    """配对 bootstrap：ΔAUC = AUC(A) − AUC(B) 的置信区间与 p 值。

    ★ 为什么必须配对而不是比两个独立 CI
      两个模型吃的是**同一批**外部测试样本 —— 难样本上一起错、易样本上一起对，
      误差高度相关。分别做 bootstrap 会把这个相关性丢掉，置信区间偏宽、检验功效偏低，
      于是"真差异"很容易被误判成"不显著"。
      正确做法：在**同一批重采样索引**上同时算两个 AUC 再相减，重复 n_boot 次。

    p 值用"差值的 bootstrap 分布跨过 0 的比例"（双侧，取两边较小者乘 2）。
    """
    y = np.asarray(y).astype(int)
    sA, sB = np.asarray(sA, float), np.asarray(sB, float)
    d = auc_score(y, sA) - auc_score(y, sB)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auc_score(y[idx], sA[idx]) - auc_score(y[idx], sB[idx]))
    if not vals:
        return d, float("nan"), float("nan"), float("nan")
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    p = 2.0 * min(float((vals <= 0).mean()), float((vals >= 0).mean()))
    return float(d), float(lo), float(hi), float(min(p, 1.0))


# ---------------------------------------------------------------- BLAST 重叠
def overlap_with_train(ext_df, train_csv, work, threads):
    """把外部序列 blast 回 eSOL 的 **train**，返回每个外部序列的最佳命中。

    ★ 只对 train 建库（不是全量）—— 因为要回答的是"这个外部序列在我们的
      训练数据里有没有近亲"。有 = 泄漏，没有 = 真泛化。
    """
    os.makedirs(work, exist_ok=True)
    db_fa = os.path.join(work, "train.fasta")
    q_fa = os.path.join(work, "query.fasta")
    tsv = os.path.join(work, "vs_train.tsv")

    tr = pd.read_csv(train_csv)
    with open(db_fa, "w") as fh:
        for i, s in enumerate(tr["sequence"]):
            fh.write(f">t{i}\n{s}\n")
    with open(q_fa, "w") as fh:
        for i, s in enumerate(ext_df["sequence"]):
            fh.write(f">q{i}\n{s}\n")

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
        print(f"  blastp 用时 {time.time() - t0:.0f}s")

    best = {}
    with open(tsv) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            pid, aln, ql, sl = float(f[2]), int(f[3]), int(f[4]), int(f[5])
            cov = min(aln / max(ql, 1), aln / max(sl, 1))
            cur = best.get(f[0])
            if cur is None or (pid, cov) > cur:
                best[f[0]] = (pid, cov)
    return best


# ---------------------------------------------------------------- 主流程
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="prep_cross_dataset.py 的产物（--reuse 时可省）")
    ap.add_argument("--ckpt", default=None,
                    help="train.py 训好的 best.ckpt（--reuse 时可省）")
    ap.add_argument("--train-csv", default=None,
                    help="用于查同源重叠的训练集（默认 data/splits/homology/train.csv）")
    ap.add_argument("--feat-train", default=None,
                    help="eSOL 训练集特征目录（用于重拟合尺子基线）；默认自动推断")
    ap.add_argument("--repo", default=os.environ.get("ESMC_REPO_600M")
                    or os.path.join(os.environ["HF_HOME"], "ESMC-600M"),
                    help="编码器权重目录。默认取 $ESMC_REPO_600M，否则 $HF_HOME/ESMC-600M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-blast", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="复用 --out 下已跑出的 predictions.csv，只重算统计量"
                         "（跳过 BLAST 与 ESMC 特征提取，秒级完成）")
    a = ap.parse_args()

    train_csv = a.train_csv or os.path.join(root, "data", "splits", "homology", "train.csv")
    feat_train = a.feat_train or os.path.join(root, "features", "esmc600m_mean__homology")
    work = a.work or os.path.join(root, "data", "blast_work", "cross")
    os.makedirs(a.out, exist_ok=True)

    if not a.reuse and not a.csv:
        raise SystemExit("完整模式必须给 --csv；只想重算统计量请加 --reuse")
    if not a.reuse and not a.ckpt:
        raise SystemExit("完整模式必须给 --ckpt；只想重算统计量请加 --reuse")

    if a.reuse:
        ext = None
        name = os.path.basename(os.path.dirname(os.path.join(a.out, "")))
    else:
        ext = pd.read_csv(a.csv)
        name = (str(ext["source"].iloc[0]) if len(ext)
                else os.path.basename(a.csv))
    print("=" * 92)
    print(f"跨数据集验证：{name}   n={'待读盘' if ext is None else len(ext)}")
    print(f"  模型 {a.ckpt or '（复用 predictions.csv）'}")
    print(f"  训练集基准 {train_csv}")
    if a.reuse:
        print("  模式 --reuse：跳过 BLAST / 特征提取 / 重训，只重算统计量")
    print("=" * 92)

    # ---------- ① 同源重叠 ----------
    prev_res = {}
    if a.reuse:
        pth = os.path.join(a.out, "predictions.csv")
        if not os.path.exists(pth):
            raise SystemExit(f"--reuse 需要 {pth} 存在；请先完整跑一次")
        prev = pd.read_csv(pth)
        need = [c for c in ("best_pident", "best_cov", "pred_mlp", "pred_ridge")
                if c not in prev.columns]
        if need:
            raise SystemExit(f"--reuse 读到的 predictions.csv 缺列：{need}")
        mp = os.path.join(a.out, "metrics.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as fh:
                prev_res = json.load(fh)
        print(f"  【复用模式】读 {pth}（{len(prev)} 条），"
              f"跳过 BLAST 与特征提取")
        ext = prev
        if len(prev) and "source" in prev.columns:
            name = str(prev["source"].iloc[0])
        best = {f"q{i}": (float(p), float(c))
                for i, (p, c) in enumerate(zip(prev["best_pident"],
                                               prev["best_cov"]))}
    else:
        best = ({} if a.skip_blast
                else overlap_with_train(ext, train_csv, work, a.threads))

    ov = {}
    if best:
        ext = ext.copy()
        ext["best_pident"] = [best.get(f"q{i}", (0.0, 0.0))[0] for i in range(len(ext))]
        ext["best_cov"] = [best.get(f"q{i}", (0.0, 0.0))[1] for i in range(len(ext))]
        ext["any_hit"] = ext["best_pident"] > 0
        for th in THRESHOLDS:
            ext[f"homolog_ge{th}"] = (ext["best_pident"] >= th) & (ext["best_cov"] >= 0.5)
            ov[f"ge{th}_pct"] = round(float(ext[f"homolog_ge{th}"].mean() * 100), 2)
        print("\n  【同源重叠】外部测试集 vs eSOL 训练集（判据：同一性≥阈值 且 覆盖率≥50%）")
        for th in THRESHOLDS:
            print(f"    ≥{th}%: {ov[f'ge{th}_pct']:5.1f}%  "
                  f"({int(ext[f'homolog_ge{th}'].sum())} 条)")
        print(f"    有任意命中 {int(ext['any_hit'].sum())} 条；"
              f"最严重一例 同一性 {ext['best_pident'].max():.1f}%")

    # ---------- ②③④ 预测与对照（复用模式直接读盘） ----------
    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    if a.reuse:
        pred = ext["pred_mlp"].to_numpy(float)
        pred_ridge = ext["pred_ridge"].to_numpy(float)
        al = float(prev_res.get("ridge_alpha", 1.0))
        ck = {"epoch": int(prev_res.get("ckpt_epoch", -1)),
              "valid_spearman": float(prev_res.get("ckpt_valid_spearman", np.nan))}
        needs_hand = None
        print("\n  【模型预测】直接取自 predictions.csv（pred_mlp / pred_ridge）")
        print(f"  ckpt 存于第 {int(ck['epoch'])} 轮"
              f"（当时 valid_spearman {float(ck['valid_spearman']):.4f}）")
        print(f"  尺子基线：Ridge(alpha={al:g})，读数同样取自 predictions.csv")
    else:
        print("\n  【提取 ESMC-600M 特征】")
        model, tok, mode = load_esmc(a.repo, device)
        seqs = ext["sequence"].tolist()
        emb = {}
        t0 = time.time()
        for i in range(0, len(seqs), a.batch):
            chunk = seqs[i:i + a.batch]
            v, _ = pool_batch(model, tok, chunk, "mean", device, torch.bfloat16)
            v = v.cpu().numpy().astype(np.float32)
            for s, row in zip(chunk, v):
                emb[s] = row
            if (i + a.batch) % (a.batch * 20) == 0 or i + a.batch >= len(seqs):
                el = time.time() - t0
                print(f"    {min(i + a.batch, len(seqs))}/{len(seqs)}  {el:.0f}s  "
                      f"({el / max(i + a.batch, 1) * 1000:.0f} ms/条)", flush=True)
        X = np.stack([emb[s] for s in seqs])
        print(f"  特征矩阵 {X.shape}")

        # ---------- ③ 用 eSOL 训好的模型预测 ----------
        dev = torch.device(device)
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
        needs_hand = int(ck["d_in"]) != X.shape[1]
        if needs_hand:
            H = handcrafted_features(seqs)
            Xin = np.concatenate([X, H], 1)
        else:
            Xin = X
        assert Xin.shape[1] == int(ck["d_in"]), \
            f"输入维度 {Xin.shape[1]} 与 ckpt 记录的 {int(ck['d_in'])} 不符"
        net = SolubilityHead(int(ck["d_in"]), hidden=tuple(ck["hidden"]),
                             dropout=float(ck["dropout"]), out_act=str(ck["out_act"]))
        net.load_state_dict(ck["model"])
        net = net.eval().to(dev)
        norm = Normalizer(np.asarray(ck["norm_mu"]), np.asarray(ck["norm_sigma"]))
        with torch.no_grad():
            p = net(torch.from_numpy(norm.transform(Xin).astype(np.float32)).to(dev))
            pred = p.cpu().numpy().reshape(-1)
        pred = np.clip(pred, 0.0, 1.0)
        print(f"  模型：{needs_hand and 'ESMC+手工特征' or '仅 ESMC 嵌入'}  "
              f"头部 {net.n_params():,} 参数  ckpt 存于第 {int(ck['epoch'])} 轮"
              f"（当时 valid_spearman {float(ck['valid_spearman']):.4f}）")

        # ---------- ④ 对照基线：尺子（同超参重拟合） ----------
        ztr = np.load(os.path.join(feat_train, "train.npz"))
        Xtr = ztr["X"].astype(np.float32)
        ytr = ztr["y"].astype(np.float32)
        try:
            bdf = pd.read_csv(os.path.join(
                root, "runs",
                f"baselines_esmc600m_"
                f"{os.path.basename(os.path.dirname(train_csv))}",
                "baselines.csv"))
            al = float(bdf.set_index("baseline").loc["esmc_ridge", "alpha"])
        except Exception:                                    # noqa: BLE001
            al = 1.0
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=al).fit(Xtr, ytr)                # ★ 超参与主实验一致
        pred_ridge = np.clip(ridge.predict(X), 0.0, 1.0)
        print(f"  尺子基线：ESMC 嵌入 + Ridge(alpha={al:g})，同超参重拟合于 eSOL train")

    pred_len = -ext["seq_len"].to_numpy(float)               # 越长越可能不可溶 → 打分取负

    # ---------- ⑤ 分层报指标 ----------
    res = {"dataset": name, "n_total": int(len(ext)), "overlap": ov,
           "ckpt": a.ckpt, "ckpt_epoch": int(ck["epoch"]),
           "ckpt_valid_spearman": float(ck["valid_spearman"]),
           "ridge_alpha": al, "n_boot": a.n_boot}
    y = ext["label"].to_numpy(int)
    res["label_pos_frac"] = round(float(y.mean()), 4)

    models = {"ours_mlp": pred, "ruler_ridge": pred_ridge, "length_only": pred_len}
    if best:
        models_hom = {k: v[ext["homolog_ge25"].to_numpy(bool)] for k, v in models.items()}
        models_clean = {k: v[~ext["homolog_ge25"].to_numpy(bool)] for k, v in models.items()}
        y_hom = y[ext["homolog_ge25"].to_numpy(bool)]
        y_clean = y[~ext["homolog_ge25"].to_numpy(bool)]
    else:
        models_hom = models_clean = {}
        y_hom = y_clean = np.array([])

    def block(title, yy, mm, key):
        if len(yy) == 0 or len(np.unique(yy)) < 2:
            return {}
        out = {}
        print(f"\n  【{title}】n={len(yy)}"
              f"（可溶 {int((yy == 1).sum())} / 不可溶 {int((yy == 0).sum())}）")
        for k, pp in mm.items():
            auc = auc_score(yy, pp)
            lo, hi = boot_ci(yy, pp, a.n_boot)
            out[k] = {"auc": round(auc, 4), "auc_lo": round(lo, 4),
                      "auc_hi": round(hi, 4), "n": int(len(yy))}
            print(f"    {k:14s} AUC {auc:.4f}  [{lo:.4f}, {hi:.4f}]")
        res[key] = out
        return out

    block("全部外部测试集", y, models, "all")
    if best:
        block("仅「有同源近亲」的子集（≥25% 泄漏）", y_hom, models_hom, "homolog")
        block("★ 仅「无同源近亲」的子集（真泛化）", y_clean, models_clean, "clean")

    # ---------- ⑥ 配对检验 ----------
    # ★ 判"两个模型谁更强"只能看这里。看两个独立 95% CI 是否重叠是错的推理：
    #   两个模型吃的是同一批样本，误差高度相关，独立 CI 会高估不确定性。
    PAIRS = [("ours_mlp", "ruler_ridge"), ("ours_mlp", "length_only"),
             ("ruler_ridge", "length_only")]
    res["paired"] = {}
    for skey, yy, mm in (("all", y, models), ("homolog", y_hom, models_hom),
                         ("clean", y_clean, models_clean)):
        if len(yy) == 0 or len(np.unique(yy)) < 2 or not mm:
            continue
        rows = {}
        print(f"\n  【配对检验 · {SUBSET_CN.get(skey, skey)}】n={len(yy)}"
              f"（同一批重采样索引上做差）")
        for ma, mb in PAIRS:
            if ma not in mm or mb not in mm:
                continue
            dlt, lo, hi, p = paired_auc(yy, mm[ma], mm[mb], a.n_boot)
            sig = bool(np.isfinite(p) and p < 0.05)
            rows[f"{ma}_minus_{mb}"] = {
                "delta": round(dlt, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                "p": round(p, 4), "n": int(len(yy)), "significant": sig}
            print(f"    ΔAUC({ma} − {mb}) = {dlt:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                  f"p={p:.3f}  {'★显著' if sig else '不显著'}")
        res["paired"][skey] = rows

    # ---------- 落盘 ----------
    ext_out = ext.copy()
    ext_out["pred_mlp"] = pred
    ext_out["pred_ridge"] = pred_ridge
    ext_out.to_csv(os.path.join(a.out, "predictions.csv"), index=False,
                   encoding="utf-8")
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    # ---------- 判读 ----------
    print("\n" + "=" * 92)
    if "all" in res and "clean" in res:
        d_all = res["all"]["ours_mlp"]["auc"]
        d_cl = res["clean"]["ours_mlp"]["auc"]
        r_cl = res["clean"]["ruler_ridge"]["auc"]
        print(f"★ 泛化落差：全部子集 {d_all:.4f} → 无近亲子集 {d_cl:.4f}"
              f"（差 {d_all - d_cl:+.4f}）")
        print("  这个差就是「跨数据集里同源泄漏虚高了多少」的直接估计。")
        if np.isfinite(d_cl) and np.isfinite(r_cl):
            print(f"★ 无近亲子集上：我们的 MLP {d_cl:.4f} vs 尺子 {r_cl:.4f}"
                  f"（差 {d_cl - r_cl:+.4f}）")
            print("  只有这一档才是「这个模型能不能用在新蛋白上」的答案。")
    pc = res.get("paired", {}).get("clean", {}).get("ours_mlp_minus_ruler_ridge")
    if pc:
        verdict = ("显著落后" if pc["significant"] and pc["delta"] < 0
                   else "显著领先" if pc["significant"] else "无显著差异")
        print(f"★ 配对检验（无近亲子集，MLP vs 尺子）：ΔAUC {pc['delta']:+.4f} "
              f"[{pc['lo']:+.4f}, {pc['hi']:+.4f}]  p={pc['p']:.3f}  → {verdict}")
        print("  结论以这一行为准：两个独立 CI 是否重叠不能用来判显著性。")
    print(f"\n产物：{a.out}/predictions.csv、metrics.json")
    print("CROSS_EVAL_DONE")


if __name__ == "__main__":
    main()
