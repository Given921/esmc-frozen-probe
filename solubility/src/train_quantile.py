"""分位数回归头 —— 让模型不只给一个数，而是给一个**区间**。

★ 为什么要另外做这个（而不是只做普通回归）
    普通回归输出一个点估计（比如"溶解度 62%"），但工程上真正要的是
    "大概 45%~78%，中位 62%" —— 有了区间才知道这个预测敢不敢信。
    这就是分位数回归（quantile regression）：不预测条件均值，而是直接预测
    条件分位点 P10 / P50 / P90。

    与"训完模型再看误差分布"的根本区别：
      后者假设所有样本的误差分布**一样宽**；
      分位数回归允许**逐样本**给出不同的宽度 —— 难预测的蛋白自动给宽区间。

★ 损失函数：pinball（弹球损失 / 分位损失）
    L_q(y, p) = max( q·(y−p),  (q−1)·(y−p) )
    q=0.5 时退化成 MAE 的一半（等价于预测中位数）；
    q=0.9 时"低估"被罚 9 倍、"高估"只罚 1 倍 → 逼着模型往高处报，正好卡在第 90 百分位。
    把若干个 q 的损失加起来一起训，一个网络同时输出所有分位点。

★ 分位点不会交叉
    网络输出 K 个数，先做「第 1 个自由 + 后面每个 = 前一个 + softplus(增量)」，
    在实数轴上单调不减，再套 sigmoid → [0,1] 内依然单调不减。
    所以 P10 ≤ P50 ≤ P90 是**结构保证**的，不需要额外惩罚项。

★ 评价
    · 点预测指标（Spearman/R²/RMSE）**用 P50 算** —— 与其他实验完全可比，
      evaluate.py 会把它当成 ensemble_test_pred.npy 一起收进总表。
    · 额外报两个只有分位数回归才有的指标：
        coverage（覆盖率）  ：真实值落在 [P10,P90] 里的比例，理想的 80% 预测区间应 ≈ 80%
        width_pct（区间宽度）：平均区间宽度，单位百分点。越窄越好（前提是 coverage 够）。

用法：
    python train_quantile.py --feat features/esmc600m_mean__homology \\
        --split-dir data/splits/homology --tag quantile_esmc600m_homology \\
        --hand --quantiles 0.1,0.5,0.9 --seeds 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handcrafted import handcrafted_features                     # noqa: E402
from metrics import all_metrics, spearman                         # noqa: E402
from model import Normalizer, SolubilityHead                      # noqa: E402

PATIENCE = 30


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def pinball(pred, target, q):
    """pred (B,K)  target (B,)  q (K,) —— 返回标量损失。"""
    e = target.unsqueeze(1) - pred                     # (B,K)  正 = 低估
    return torch.maximum(q * e, (q - 1.0) * e).mean()


def load_split(feat_dir, split, split_dir, use_hand):
    z = np.load(os.path.join(feat_dir, f"{split}.npz"))
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.float32)
    if use_hand:
        df = pd.read_csv(os.path.join(split_dir, f"{split}.csv"))
        assert len(df) == len(y), f"{split} 特征与划分文件条数不符"
        X = np.concatenate([X, handcrafted_features(df["sequence"].tolist())], 1)
    return X, y


def coverage_width(y, Q, lo_i, hi_i):
    """覆盖率与平均区间宽度（单位：百分点）。"""
    lo, hi = Q[:, lo_i], Q[:, hi_i]
    cov = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo) * 100)
    return cov, width


def train_one(seed, Xtr, ytr, Xva, yva, Xte, yte, out_dir, args, device, qs):
    set_seed(seed)
    norm = Normalizer.fit(Xtr)
    Xtr_n, Xva_n, Xte_n = norm.transform(Xtr), norm.transform(Xva), norm.transform(Xte)

    t_tr = torch.from_numpy(Xtr_n).to(device)
    t_ytr = torch.from_numpy(ytr).to(device)
    t_va = torch.from_numpy(Xva_n).to(device)
    t_te = torch.from_numpy(Xte_n).to(device)
    tq = torch.from_numpy(qs.astype(np.float32)).to(device)

    mid = int(np.argmin(np.abs(qs - 0.5)))            # 中位数那一列 = 点预测

    model = SolubilityHead(Xtr.shape[1], hidden=tuple(args.hidden), dropout=args.dropout,
                           out_act="sigmoid", out_dim=len(qs)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    n = len(Xtr)
    steps = max(1, n // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.15)

    seed_dir = os.path.join(out_dir, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    ckpt_path = os.path.join(seed_dir, "best.ckpt")

    best, best_ep, wait, hist = -np.inf, -1, 0, []

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for k in range(steps):
            idx = perm[k * args.batch:(k + 1) * args.batch]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            out = model(t_tr[idx])
            loss = pinball(out, t_ytr[idx], tq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)

        model.eval()
        with torch.no_grad():
            pv = model(t_va).cpu().numpy()
        # ★ 与 train.py 同一口径：用**中位数预测**的验证集 Spearman 选 ckpt
        v_metric = spearman(yva, pv[:, mid])
        hist.append(dict(epoch=ep, train_pinball=tot / n, valid_spearman=v_metric))

        if v_metric > best:
            best, best_ep, wait = v_metric, ep, 0
            torch.save({"model": model.state_dict(),
                        "norm_mu": torch.as_tensor(norm.mu),
                        "norm_sigma": torch.as_tensor(norm.sigma),
                        "d_in": int(Xtr.shape[1]), "hidden": tuple(args.hidden),
                        "dropout": float(args.dropout), "out_act": "sigmoid",
                        "out_dim": int(len(qs)),
                        "quantiles": torch.as_tensor(qs.astype(np.float32)),
                        "epoch": int(ep), "valid_spearman": float(v_metric),
                        "seed": int(seed)}, ckpt_path)
        else:
            wait += 1

        if ep % args.log_every == 0 or ep == 1:
            print(f"      ep{ep:4d}  pinball {tot/n:.5f}  valid_spearman(P50) {v_metric:.4f}"
                  f"  (best {best:.4f} @ep{best_ep}, 已等 {wait})", flush=True)
        if wait >= PATIENCE:
            print(f"      早停于 ep{ep}（{PATIENCE} 轮无进步），回滚到 best.ckpt ep{best_ep}",
                  flush=True)
            break

    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        Qv = model(t_va).cpu().numpy()
        Qt = model(t_te).cpu().numpy()

    pd.DataFrame(hist).to_csv(os.path.join(seed_dir, "history.csv"), index=False)
    lo_i, hi_i = 0, len(qs) - 1
    cov_v, w_v = coverage_width(yva, Qv, lo_i, hi_i)
    cov_t, w_t = coverage_width(yte, Qt, lo_i, hi_i)
    info = dict(seed=seed, best_epoch=best_ep, best_valid_spearman=best,
                n_params=model.n_params(),
                valid_coverage=cov_v, valid_width_pct=w_v,
                test_coverage=cov_t, test_width_pct=w_t,
                **{f"valid_{k}": v for k, v in all_metrics(yva, Qv[:, mid]).items()},
                **{f"test_{k}": v for k, v in all_metrics(yte, Qt[:, mid]).items()})
    return Qv, Qt, info, mid


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--split-dir", default=os.path.join(root, "data", "splits", "homology"))
    ap.add_argument("--out-root", default=os.path.join(root, "runs"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hand", action="store_true")
    ap.add_argument("--quantiles", default="0.1,0.5,0.9",
                    help="逗号分隔，必须包含 0.5（那一列当点预测）")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, nargs=2, default=[512, 128])
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    qs = np.array(sorted(float(x) for x in a.quantiles.split(",")), dtype="float64")
    assert (np.diff(qs) > 0).all(), "分位点必须严格递增"
    assert np.any(np.abs(qs - 0.5) < 1e-9), "分位点里必须有 0.5（否则没有点预测）"

    device = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(a.out_root, a.tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 92)
    print(f"分位数回归训练 {a.tag}   设备 {device}   种子 {a.seeds} 个")
    print(f"  分位点 {qs.tolist()}   （点预测取 P{int(round(qs[int(np.argmin(np.abs(qs-0.5)))]*100))}）")
    print(f"  目标区间 P{int(qs[0]*100)}~P{int(qs[-1]*100)}"
          f"（理想覆盖率 {int((qs[-1]-qs[0])*100)}%）")
    print(f"  特征 {a.feat}{'  +手工特征' if a.hand else ''}")
    print(f"  输出 {out_dir}")
    print("=" * 92)

    Xtr, ytr = load_split(a.feat, "train", a.split_dir, a.hand)
    Xva, yva = load_split(a.feat, "valid", a.split_dir, a.hand)
    Xte, yte = load_split(a.feat, "test", a.split_dir, a.hand)
    print(f"\n数据：train {Xtr.shape}  valid {Xva.shape}  test {Xte.shape}\n")

    Qvs, Qts, infos = [], [], []
    for s in range(a.seeds):
        print(f"  --- 种子 {s} ---")
        Qv, Qt, info, mid = train_one(s, Xtr, ytr, Xva, yva, Xte, yte,
                                      out_dir, a, device, qs)
        Qvs.append(Qv)
        Qts.append(Qt)
        infos.append(info)
        print(f"      best.ckpt @ep{info['best_epoch']}  "
              f"valid P50 Spearman {info['best_valid_spearman']:.4f}  "
              f"| test P50 Spearman {info['test_spearman']:.4f}  R² {info['test_r2']:.4f}"
              f"  | 区间覆盖率 {info['test_coverage']*100:.1f}%"
              f"  平均宽度 {info['test_width_pct']:.1f} 点")
        np.save(os.path.join(out_dir, f"seed{s}", "valid_pred.npy"), Qv[:, mid])
        np.save(os.path.join(out_dir, f"seed{s}", "test_pred.npy"), Qt[:, mid])
        np.save(os.path.join(out_dir, f"seed{s}", "valid_quantiles.npy"), Qv)
        np.save(os.path.join(out_dir, f"seed{s}", "test_quantiles.npy"), Qt)

    Qi_v, Qi_t = np.mean(Qvs, 0), np.mean(Qts, 0)
    ens_v, ens_t = Qi_v[:, mid], Qi_t[:, mid]
    np.save(os.path.join(out_dir, "ensemble_valid_pred.npy"), ens_v)
    np.save(os.path.join(out_dir, "ensemble_test_pred.npy"), ens_t)
    np.save(os.path.join(out_dir, "ensemble_valid_quantiles.npy"), Qi_v)
    np.save(os.path.join(out_dir, "ensemble_test_quantiles.npy"), Qi_t)
    np.save(os.path.join(out_dir, "y_valid.npy"), yva)
    np.save(os.path.join(out_dir, "y_test.npy"), yte)

    single = [i["test_spearman"] for i in infos]
    lo_i, hi_i = 0, len(qs) - 1
    cov_v, w_v = coverage_width(yva, Qi_v, lo_i, hi_i)
    cov_t, w_t = coverage_width(yte, Qi_t, lo_i, hi_i)
    summary = {
        "tag": a.tag, "feat": a.feat, "hand": a.hand, "out_act": "sigmoid",
        "kind": "quantile", "quantiles": qs.tolist(),
        "split_dir": a.split_dir, "seeds": a.seeds,
        "n_params_head": infos[0]["n_params"],
        "single_test_spearman": single,
        "single_mean": float(np.mean(single)), "single_std": float(np.std(single)),
        "single_min": float(np.min(single)), "single_max": float(np.max(single)),
        "ensemble_valid": all_metrics(yva, ens_v),
        "ensemble_test": all_metrics(yte, ens_t),
        "ensemble_valid_coverage": cov_v, "ensemble_valid_width_pct": w_v,
        "ensemble_test_coverage": cov_t, "ensemble_test_width_pct": w_t,
        "target_coverage": float(qs[-1] - qs[0]),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    pd.DataFrame(infos).to_csv(os.path.join(out_dir, "per_seed.csv"), index=False)

    print("\n" + "=" * 92)
    print(f"【{a.tag}】单种子 test P50 Spearman {['%.4f' % s for s in single]}")
    print(f"  均值 {summary['single_mean']:.4f}  标准差 {summary['single_std']:.4f}")
    print(f"\n【{a.tag}】集成后 test 指标（P50 当点预测）")
    for k, v in summary["ensemble_test"].items():
        print(f"    {k:9s} {v:.4f}")
    print(f"\n【{a.tag}】预测区间（集成后）")
    print(f"    valid 覆盖率 {cov_v*100:.1f}%  平均宽度 {w_v:.1f} 点")
    print(f"    test  覆盖率 {cov_t*100:.1f}%  平均宽度 {w_t:.1f} 点"
          f"   （目标 {summary['target_coverage']*100:.0f}%）")
    if cov_t < summary["target_coverage"] - 0.10:
        print("    ⚠️ 覆盖率明显低于目标 → 区间偏窄（过自信），说模型还没校准好。")
    elif cov_t > summary["target_coverage"] + 0.10:
        print("    ⚠️ 覆盖率明显高于目标 → 区间偏宽（欠自信），可以试着收窄。")
    else:
        print("    ✓ 覆盖率接近目标，区间尺度合理。")
    print(f"\n产物：{out_dir}")
    print("TRAIN_Q_DONE")


if __name__ == "__main__":
    main()
