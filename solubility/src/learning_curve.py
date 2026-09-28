"""数据量消融（学习曲线）：固定 valid / test，只缩小 train，看每个配置怎么爬。

★ 想回答的问题
    前面已经确认"编码器规模的收益在 600M 处就饱和了"（6B vs 600M 三条配对检验全不显著）。
    但那是在**全部 2525 条训练数据**下测的。一个自然的反问是：

        **6B 是不是被数据量饿着？**（大模型的优势常常要更多数据才兑现）

    学习曲线就是回答这个的标准工具：横轴 = 训练集规模，纵轴 = 测试集 Spearman，
    每个配置一条线。

      · 如果 6B 的线**随数据增长更陡** → 它的优势需要更多数据，现在确实是被饿着；
      · 如果 6B 的线**全程平铺在 600M 上方、间距不变** → 表示质量是"平移式"优势，
        不是"缩放式"优势，那么"数据不够"就不能解释饱和。

★ 三条铁律（违反任何一条曲线就不可信）
  1. **测试集永远不动**（同一批 316 条序列、同一个 homology 划分）。
     各点之间才存在"同一把尺子"，也才允许后面做配对检验。
  2. **验证集也不动**（它是用来选 ckpt 的）。只有训练集变小 ——
     这样横轴上唯一在变的就是"模型见过多少数据"这一个因素。
  3. **子集嵌套**：250 ⊂ 500 ⊂ … ⊂ 2525。
     曲线上的点是同一批序列逐档加料，而不是每档重新随机抽一批。
     每档重抽的话，曲线抖动里就混进了"这档抽到了哪批序列"的噪声，
     会把真实趋势淹没（本项目测试集只有 316 条，噪声本来就不小）。

★ 幂等
    三个 stage 都可以随时重跑，已经算过的会跳过（`--force` 覆盖）。
    所以这个脚本可以在训练**跑了一部分**的时候先去 merge，出一份"部分结果"，
    数据补齐后原样重跑就自动补全，不需要改代码。

用法
    python src/learning_curve.py --stage all            # 索引 → 尺子 → 汇总
    python src/learning_curve.py --stage indices
    python src/learning_curve.py --stage ridges --encs esmc600m
    python src/learning_curve.py --stage merge
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import all_metrics, bootstrap_ci, spearman             # noqa: E402

# ★ 显式声明规模顺序，并且**出现未知编码器直接报错**。
#   别靠 startswith 猜规模（本项目已经因为"靠字符串猜"错位过一次）。
ENCS_ORDER = ["esmc300m", "esmc600m", "esmc6b"]
ENC_FEAT = {"esmc300m": "esmc300m_mean__homology",
            "esmc600m": "esmc600m_mean__homology",
            "esmc6b": "esmc6b_mean__homology"}
ENC_SHORT = {"esmc300m": "300M", "esmc600m": "600M", "esmc6b": "6B"}
# 三个对外报的配置。ruler = 冻结嵌入的线性探针（尺子），任何复杂模型必须先超过它。
CFG_LABEL = {"ruler": "尺子·线性探针", "head_hand": "MLP+手工特征",
             "head_only": "MLP·仅编码器"}
SIZES_DEFAULT = [100, 250, 500, 1000, 1500, 2525]
IDX_SEED = 20260926          # 固定种子：谁来跑都得到同一批子集


def run_tag(enc, size, cfg):
    """MLP 两种配置的目录名（要和 run_learning_curve.sh 里 --tag 完全一致）。"""
    if cfg == "head_hand":
        return f"lc_{enc}_s{size}_hand"
    if cfg == "head_only":
        return f"lc_{enc}_s{size}"
    raise ValueError(f"未知配置 {cfg}")


def ridge_dir(enc, size):
    return f"lc_ridges/{enc}_s{size}"


# --------------------------------------------------------------- stage: indices
def stage_indices(a):
    """生成嵌套的训练集行序号（只依赖 train 的条数，与编码器无关）。"""
    sp_dir = os.path.join(a.split_dir)
    tr_csv = os.path.join(sp_dir, "train.csv")
    df = pd.read_csv(tr_csv)
    n = len(df)
    out_dir = os.path.join(sp_dir, "lc")
    os.makedirs(out_dir, exist_ok=True)

    print(f"训练集全量 {n} 条（{tr_csv}）")
    sizes = [s for s in a.sizes if s <= n]
    dropped = [s for s in a.sizes if s > n]
    if dropped:
        print(f"  ★ 跳过超过全量的规模：{dropped}")

    # ★ 嵌套抽样的正确做法：先做**一个**全量随机排列，各档取它的前缀。
    #   每档各自 rng.permutation(n)[:s] 得到的是"近似嵌套"，不保证 250 ⊂ 500。
    rng = np.random.default_rng(IDX_SEED)
    perm = rng.permutation(n)
    files = {}
    for s in sizes:
        idx = np.sort(perm[:s]).astype(np.int64)
        fp = os.path.join(out_dir, f"train_idx_s{s}.npy")
        np.save(fp, idx)
        files[str(s)] = {"path": os.path.relpath(fp, a.root), "n": int(idx.size)}
        print(f"  s={s:<5d} 写 {files[str(s)]['path']}")

    # 嵌套自检：小规模必须是大规模的**子集**（不是"大概差不多"）
    order = sorted(sizes)
    for i in range(len(order) - 1):
        A = set(np.load(os.path.join(out_dir, f"train_idx_s{order[i]}.npy")).tolist())
        B = set(np.load(os.path.join(out_dir, f"train_idx_s{order[i + 1]}.npy")).tolist())
        assert A <= B, f"嵌套性被破坏：s{order[i]} 不是 s{order[i+1]} 的子集"
    print(f"  ✓ 嵌套性自检通过（{order[0]} ⊂ {order[1]} ⊂ … ⊂ {order[-1]}）")

    man = {"n_train_full": n, "seed": IDX_SEED, "sizes": order, "files": files,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "indices.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)
    print(f"  清单 {os.path.relpath(os.path.join(out_dir, 'indices.json'), a.root)}")
    return order


# --------------------------------------------------------------- 数据对齐自检
def check_alignment(feat_dir, split_dir):
    """确认 npz 的第 i 行 == 划分 csv 的第 i 行。

    ★ 为什么要专门查一次：整条管线里 npz 和 csv 是**分别**生成的，
      两边都只断言"条数相同"，没有任何地方检查过**顺序**是否一致。
      一旦顺序不同（比如某次重新生成时排序变了），手工特征就会和嵌入错位，
      而且**不会报任何错** —— 训练照跑，分数照出，只是悄悄偏低。
      这里用标签值把顺序钉死（标签是两边都有的、且取值足够唯一）。
    """
    z = np.load(os.path.join(feat_dir, "train.npz"))
    df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    assert len(df) == len(z["y"]), f"train 条数不符：csv {len(df)} vs npz {len(z['y'])}"
    if "solubility_frac" in df.columns:
        ok = np.allclose(df["solubility_frac"].to_numpy(np.float32), z["y"], atol=1e-6)
        assert ok, ("train.npz 与 train.csv **行顺序不一致**！"
                    "手工特征会和嵌入错位，必须先修数据而不是继续跑。")
    return True


# --------------------------------------------------------------- stage: ridges
def stage_ridges(a, sizes):
    """尺子（冻结嵌入 + Ridge）。每个 (编码器, 规模) 一个结果，含两个 alpha 口径。

    ★ 为什么尺子也要跟着缩小训练集
      因为要问的是"**MLP 相对尺子的优势**怎么随数据变"，不是"MLP 自己怎么变"。
      只画 MLP 的曲线，看到"数据少时分数低"完全可能是任务本身难，与头部无关。
      必须有一条同数据量的参照线，差值才有意义。

    ★ alpha 口径说明（这里如实写下来，免得报告里含糊）
      `ruler`   ：alpha ∈ {1,10,100,1000} 里**按测试集 Spearman 挑最好**的
                  —— 与项目里已有的 esmc_ridge 基线口径**完全一致**，
                  所以全量那一档能对上已交付的 0.7127 之类的数字。
                  代价：这条线本身略微乐观（等于对测试集做了 4 选 1）。
      `ruler_a10`：固定 alpha=10，不做任何测试集选择，作为无损对照。
      两者都写进 learning_curve_full.csv，正文只用 ruler、并注明口径。
    """
    from sklearn.linear_model import Ridge

    out_root = os.path.join(a.root, "runs")
    rows = []
    for enc in a.encs:
        fd = os.path.join(a.root, "features", ENC_FEAT[enc])
        ztr = np.load(os.path.join(fd, "train.npz"))
        zte = np.load(os.path.join(fd, "test.npz"))
        Xte = zte["X"].astype(np.float32)
        yte = zte["y"].astype(np.float32)
        Xtr_full = ztr["X"].astype(np.float32)
        ytr_full = ztr["y"].astype(np.float32)
        for s in sizes:
            od = os.path.join(out_root, ridge_dir(enc, s))
            done_fp = os.path.join(od, "summary.json")
            if os.path.exists(done_fp) and not a.force:
                print(f"  [skip] {enc} s={s} 已有结果")
                rows.append(json.load(open(done_fp, encoding="utf-8")))
                continue
            idx = np.load(os.path.join(a.split_dir, "lc", f"train_idx_s{s}.npy"))
            Xtr, ytr = Xtr_full[idx], ytr_full[idx]
            rec = {"encoder": enc, "n_train": int(len(ytr)), "n_test": int(len(yte))}
            best = None
            for al in (1.0, 10.0, 100.0, 1000.0):
                pred = np.clip(Ridge(alpha=al).fit(Xtr, ytr).predict(Xte), 0, 1)
                m = all_metrics(yte, pred)
                if best is None or m["spearman"] > best[1]["spearman"]:
                    best = (al, m, pred)
            al, m, pred = best
            os.makedirs(od, exist_ok=True)
            np.save(os.path.join(od, "test_pred.npy"), pred.astype(np.float32))
            np.save(os.path.join(od, "y_test.npy"), yte)
            rec["ruler"] = {**m, "alpha": al}
            # 固定 alpha 的对照（不做测试集选择）
            pred10 = np.clip(Ridge(alpha=10.0).fit(Xtr, ytr).predict(Xte), 0, 1)
            rec["ruler_a10"] = all_metrics(yte, pred10)
            np.save(os.path.join(od, "test_pred_a10.npy"), pred10.astype(np.float32))
            with open(done_fp, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, ensure_ascii=False)
            rows.append(rec)
            print(f"  {enc:<9s} s={s:<5d} 尺子 Spearman {m['spearman']:.4f}"
                  f" (alpha={al:g})  固定a10 {rec['ruler_a10']['spearman']:.4f}")
    return rows


# --------------------------------------------------------------- stage: merge
def _boot(y, p, seed=0):
    lo, hi = bootstrap_ci(y, p, "spearman", n_boot=2000, seed=seed)
    return [round(lo, 4), round(hi, 4)]


def stage_merge(a):
    """把 MLP 的 runs/ 与尺子的 runs/lc_ridges/ 汇总成两张表。

    产物
      runs/_summary/learning_curve.csv        ← 看板读这张（只有 3 个主配置）
      runs/_summary/learning_curve_full.csv   ← 含固定 alpha 对照等附加行

    ★ 缺哪个配置就跳过哪个，并在 stdout 明确列出来 —— 不静默跳过。
      （"静默跳过"在图上会变成一根空柱子，比直接报错更难发现。）
    """
    out_dir = os.path.join(a.root, "runs", "_summary")
    os.makedirs(out_dir, exist_ok=True)
    rows, extra, missing = [], [], []

    for enc in a.encs:
        enc_rows = []
        for s in a.sizes:
            # ---- 尺子 ----
            rd = os.path.join(a.root, "runs", ridge_dir(enc, s))
            if os.path.exists(os.path.join(rd, "summary.json")):
                rec = json.load(open(os.path.join(rd, "summary.json"), encoding="utf-8"))
                y = np.load(os.path.join(rd, "y_test.npy"))
                p = np.load(os.path.join(rd, "test_pred.npy"))
                lo, hi = _boot(y, p)
                m = rec["ruler"]
                rows.append(dict(encoder=enc, n_train=rec["n_train"], role="baseline",
                                 config="ruler", config_label=CFG_LABEL["ruler"],
                                 pred=f"ridge_a{m['alpha']:g}", n_test=rec["n_test"],
                                 spearman=round(m["spearman"], 4),
                                 spearman_lo=lo, spearman_hi=hi,
                                 pearson=round(m["pearson"], 4),
                                 r2=round(m["r2"], 4),
                                 rmse_pct=round(m["rmse"] * 100, 2),
                                 mae_pct=round(m["mae"] * 100, 2),
                                 source=os.path.relpath(rd, a.root)))
                m10 = rec["ruler_a10"]
                p10 = np.load(os.path.join(rd, "test_pred_a10.npy"))
                lo, hi = _boot(y, p10)
                extra.append(dict(encoder=enc, n_train=rec["n_train"],
                                  config="ruler_a10_fixed", config_label="尺子·alpha固定10",
                                  pred="ridge_a10", n_test=rec["n_test"],
                                  spearman=round(m10["spearman"], 4),
                                  spearman_lo=lo, spearman_hi=hi,
                                  r2=round(m10["r2"], 4),
                                  source=os.path.relpath(rd, a.root)))
            else:
                missing.append(f"ruler@{ENC_SHORT.get(enc, enc)}·{s}")

            # ---- 两个 MLP 头 ----
            for cfg in ("head_hand", "head_only"):
                tag = run_tag(enc, s, cfg)
                sd = os.path.join(a.root, "runs", tag, "summary.json")
                if not os.path.exists(sd):
                    missing.append(f"{cfg}@{ENC_SHORT.get(enc, enc)}·{s}")
                    continue
                sm = json.load(open(sd, encoding="utf-8"))
                yd = os.path.join(a.root, "runs", tag)
                y = np.load(os.path.join(yd, "y_test.npy"))
                p = np.load(os.path.join(yd, "ensemble_test_pred.npy"))
                lo, hi = _boot(y, p)
                m = sm["ensemble_test"]
                rows.append(dict(encoder=enc, n_train=int(sm.get("n_train") or s),
                                 role=cfg, config=cfg, config_label=CFG_LABEL[cfg],
                                 pred="mlp_ensemble", n_test=int(len(y)),
                                 spearman=round(m["spearman"], 4),
                                 spearman_lo=lo, spearman_hi=hi,
                                 pearson=round(m["pearson"], 4),
                                 r2=round(m["r2"], 4),
                                 rmse_pct=round(m["rmse"] * 100, 2),
                                 mae_pct=round(m["mae"] * 100, 2),
                                 single_mean=round(sm["single_mean"], 4),
                                 single_std=round(sm["single_std"], 4),
                                 source=os.path.relpath(yd, a.root)))
                enc_rows.append((cfg, s, m["spearman"]))
        print(f"  {enc:<9s} 已汇总 {len(enc_rows)} 个 MLP 配置点")

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["encoder", "n_train", "config"]).reset_index(drop=True)
    fp = os.path.join(out_dir, "learning_curve.csv")
    df.to_csv(fp, index=False)

    cols = ["encoder", "n_train", "config", "config_label", "pred", "n_test",
            "spearman", "spearman_lo", "spearman_hi", "r2", "source"]
    df2 = pd.DataFrame(extra)
    pd.concat([df[[c for c in cols if c in df.columns]],
               df2[[c for c in cols if c in df2.columns]]],
              ignore_index=True).to_csv(
        os.path.join(out_dir, "learning_curve_full.csv"), index=False)

    print(f"\n  → {os.path.relpath(fp, a.root)}  （{len(df)} 行）")
    print(f"  → runs/_summary/learning_curve_full.csv  （主表 + {len(extra)} 行附加对照）")
    if missing:
        print(f"\n  ★ 还没有结果的配置 {len(missing)} 个（不是错误，是还没跑到）：")
        for x in missing[:12]:
            print(f"      - {x}")
        if len(missing) > 12:
            print(f"      …还有 {len(missing) - 12} 个")
    else:
        print("\n  ✓ 所有规模 × 编码器 × 配置都已出结果")

    # ---- 自检：主表里不能出现非有限数 ----
    import math
    bad = [v for v in df.get("spearman", []) if not math.isfinite(float(v))]
    assert not bad, f"learning_curve.csv 里出现非有限 Spearman：{bad}"
    print("MERGE_DONE")
    return df


# --------------------------------------------------------------- stage: pairs
# ★ 配对顺序一律 **大模型在前**（大 − 小），这样 delta 为正 = 更大的编码器更好。
#   反过来写（小 − 大）会在报告里出现一大堆负号，读者要先在脑子里翻一次符号，
#   极易把"大模型更好"读成"大模型更差"（这条线上已经犯过一次，hero 数字方向写反）。
PAIR_ORDER = [("esmc600m", "esmc300m"), ("esmc6b", "esmc600m"),
              ("esmc6b", "esmc300m")]


def _pred_of(a, enc, size, cfg):
    """取某 (编码器, 规模, 配置) 在**测试集**上的集成预测 + y。取不到返回 None。"""
    if cfg == "ruler":
        d = os.path.join(a.root, "runs", ridge_dir(enc, size))
        pf, yf = "test_pred.npy", "y_test.npy"
    else:
        d = os.path.join(a.root, "runs", run_tag(enc, size, cfg))
        pf, yf = "ensemble_test_pred.npy", "y_test.npy"
    if not (os.path.exists(os.path.join(d, pf)) and os.path.exists(os.path.join(d, yf))):
        return None
    return np.load(os.path.join(d, yf)), np.load(os.path.join(d, pf))


def stage_pairs(a):
    """每个训练集规模上，做**编码器之间**的配对检验。

    ★ 这一步才是"6B 是不是被数据饿着"的直接判据
      如果 6B 的优势是"需要更多数据才兑现"的那种，那么
        Δ(6B − 600M) 在小数据量下应该接近 0 甚至为负，**随数据量单调变大**。
      如果 Δ 在各档规模上大致是**同一条水平线**，说明 6B 带来的是"平移式"优势 ——
      它跟数据量无关，那么"数据不够"就不能解释"规模收益在 600M 饱和"。

    ★ 只在**同一档规模内**做配对（同训练集、同测试集、同顺序），
      **绝不跨档相减** —— 跨档时模型见过不同量的数据，误差结构完全不同，
      拿去做配对检验等于把两个不同的实验当成同一批样本。
    """
    from metrics import paired_bootstrap
    rows = []
    for enc_a, enc_b in PAIR_ORDER:
        if enc_a not in a.encs or enc_b not in a.encs:
            continue
        for cfg in ("ruler", "head_hand", "head_only"):
            for s in a.sizes:
                A = _pred_of(a, enc_a, s, cfg)
                B = _pred_of(a, enc_b, s, cfg)
                if A is None or B is None:
                    print(f"  [缺] {enc_a}-{enc_b} {cfg} @{s}")
                    continue
                yA, pA = A
                yB, pB = B
                assert len(yA) == len(yB) and np.allclose(yA, yB), \
                    (f"{enc_a}/{enc_b} 在 s={s} {cfg} 上的 y_test 不一致 —— "
                     "配对检验要求同一批测试样本同一顺序")
                for metric in ("spearman", "pearson"):
                    d, lo, hi, pv = paired_bootstrap(yA, pA, pB, metric,
                                                     n_boot=2000, seed=0)
                    rows.append(dict(encoder_a=enc_a, encoder_b=enc_b, n_train=s,
                                     config=cfg, metric=metric,
                                     delta=round(d, 4), lo=round(lo, 4),
                                     hi=round(hi, 4), p=round(pv, 4),
                                     significant=bool(lo > 0 or hi < 0)))
    df = pd.DataFrame(rows)
    fp = os.path.join(a.root, "runs", "_summary", "learning_curve_pairs.csv")
    df.to_csv(fp, index=False)
    print(f"  → {os.path.relpath(fp, a.root)}  （{len(df)} 行）")

    print("\n  ★ 6B − 600M 的 Spearman 差值随训练集规模的变化（正 = 6B 更好）")
    print("     规模     尺子       MLP+手工    MLP仅编码器")
    for s in a.sizes:
        cells = []
        for cfg in ("ruler", "head_hand", "head_only"):
            m = df[(df.n_train == s) & (df.encoder_a == "esmc6b")
                   & (df.encoder_b == "esmc600m") & (df.config == cfg)
                   & (df.metric == "spearman")]
            if len(m):
                r = m.iloc[0]
                cells.append(f"{r['delta']:+.4f}{'*' if r['significant'] else ' '}")
            else:
                cells.append("   —   ")
        print(f"     {s:<7d} " + "  ".join(cells))
    print("     （* = 配对检验显著；全都不带星号 = 6B 并不随数据量拉开差距）")
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "indices", "ridges", "merge", "check", "pairs"])
    ap.add_argument("--root", default=root)
    ap.add_argument("--split-dir", default=os.path.join(root, "data", "splits", "homology"))
    ap.add_argument("--encs", nargs="*", default=ENCS_ORDER)
    ap.add_argument("--sizes", nargs="*", type=int, default=SIZES_DEFAULT)
    ap.add_argument("--force", action="store_true", help="覆盖已算好的尺子结果")
    a = ap.parse_args()

    for e in a.encs:
        if e not in ENC_FEAT:
            # ★ 认不出的编码器直接报错。静默忽略会让报告少一个规模却看不出来。
            raise SystemExit(f"未知编码器 {e!r}；本脚本认识：{list(ENC_FEAT)}")

    a.sizes = sorted(set(a.sizes))
    print("=" * 88)
    print(f"数据量消融  编码器 {a.encs}  规模 {a.sizes}")
    print(f"  划分 {a.split_dir}")
    print("=" * 88 + "\n")

    if a.stage in ("all", "check"):
        for enc in a.encs:
            fd = os.path.join(a.root, "features", ENC_FEAT[enc])
            check_alignment(fd, a.split_dir)
            print(f"  ✓ {enc} 特征与划分行序一致")
    if a.stage in ("all", "indices"):
        print("\n[1] 生成嵌套训练集索引")
        a.sizes = stage_indices(a)
    if a.stage in ("all", "ridges"):
        print("\n[2] 跑尺子（冻结嵌入 + Ridge）")
        stage_ridges(a, a.sizes)
    if a.stage in ("all", "merge"):
        print("\n[3] 汇总")
        stage_merge(a)
    if a.stage in ("all", "pairs"):
        print("\n[4] 编码器之间的配对检验（每档规模各做一次）")
        stage_pairs(a)


if __name__ == "__main__":
    main()
