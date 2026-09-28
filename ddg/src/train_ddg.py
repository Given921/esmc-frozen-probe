"""热稳定性 ΔΔG · 训练轻量头部（一个变体一个模型，多种子集成）。

★ 与溶解度那条线保持完全相同的纪律
  · 标准化统计量**只用训练集**算；
  · 用**验证集**选 ckpt（绝不用测试集选）；早停后回滚到 best；
  · 多种子取平均作为对外报的数，同时留每个种子的分数看方差；
  · 测试集只在最后评估一次。

★ 输出激活用 linear
    溶解度线的标签被限制在 [0,1]（用 sigmoid 更自然）；
    ΔΔG 是**无界**的量（稳定化突变可为负、去稳定为正，单位 kcal/mol），
    所以用 linear，不加任何挤压。

★ 为什么要逐变体单独训练、而不是"一个模型多输出"
    因为要比较的是"**同样的训练流程、同样的超参**下，换掉输入特征会怎样"。
    多输出模型会让几个变体共享隐层，比较就变成了"联合训练 vs 单独训练"，
    多了一个变量，差值不再干净。

用法
    python src/train_ddg.py --enc esmc600m --variant A_hwt_hmut
    python src/train_ddg.py --enc esmc600m --variant B_hwt --seeds 3
清单跑法见 run_ddg_600m.sh / run_ddg_all.sh
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import all_metrics, spearman                         # noqa: E402
from model import Normalizer, SolubilityHead                      # noqa: E402

PATIENCE = 30


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_one(seed, d, args, device, out_dir):
    """训练一个种子。返回 dict（含验证/测试预测与指标）。

    ★ out_dir 必须**显式传参**。原版在函数体里写 `args.out_dir`，但命令行里根本没有
      `--out-dir` 这个参数（out_dir 是 main() 里的局部变量）→ 四个变体一进函数就
      `AttributeError: 'Namespace' object has no attribute 'out_dir'`，秒崩。
      这个 bug 的危害不只在于训练失败，更在于 run_ddg.sh 的训练循环不做 exit（见该文件注释），
      于是**四个变体全崩，脚本照样打印 DDG_ALL_DONE** —— 完成标记成了假的。
    """
    set_seed(seed)
    Xtr, ytr = d["Xtr"], d["ytr"]
    Xva, yva = d["Xva"], d["yva"]
    Xte, yte = d["Xte"], d["yte"]
    norm = Normalizer.fit(Xtr)                       # ★ 只用训练集
    t_tr = torch.from_numpy(norm.transform(Xtr)).to(device)
    t_va = torch.from_numpy(norm.transform(Xva)).to(device)
    t_te = torch.from_numpy(norm.transform(Xte)).to(device)
    t_ytr = torch.from_numpy(ytr).to(device)

    model = SolubilityHead(Xtr.shape[1], hidden=tuple(args.hidden),
                           dropout=args.dropout, out_act="linear").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = nn.MSELoss()
    n = len(Xtr)
    steps = max(1, n // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.15)

    seed_dir = os.path.join(out_dir, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    ckpt = os.path.join(seed_dir, "best.ckpt")
    prog = open(os.path.join(seed_dir, "progress.jsonl"), "w",
                encoding="utf-8", buffering=1)
    prog.write(json.dumps({"epoch": 0, "epochs": args.epochs, "seed": int(seed),
                           "started": True, "ts": time.time()},
                          ensure_ascii=False) + "\n")

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
            loss = lossf(out, t_ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            pv = model(t_va).cpu().numpy()
        v = spearman(yva, pv)
        hist.append(dict(epoch=ep, train_loss=tot / n, valid_spearman=v))
        if v > best:
            best, best_ep, wait = v, ep, 0
            torch.save({"model": model.state_dict(),
                        "norm_mu": torch.as_tensor(norm.mu),
                        "norm_sigma": torch.as_tensor(norm.sigma),
                        "d_in": int(Xtr.shape[1]), "hidden": tuple(args.hidden),
                        "dropout": float(args.dropout), "out_act": "linear",
                        "epoch": int(ep), "valid_spearman": float(v),
                        "seed": int(seed)}, ckpt)
        else:
            wait += 1
        prog.write(json.dumps({"epoch": int(ep), "epochs": int(args.epochs),
                               "seed": int(seed), "train_loss": round(float(tot / n), 6),
                               "valid_spearman": round(float(v), 6),
                               "best": round(float(best), 6), "best_epoch": int(best_ep),
                               "wait": int(wait), "ts": time.time()},
                              ensure_ascii=False) + "\n")
        if ep % args.log_every == 0 or ep == 1:
            print(f"      ep{ep:4d} loss {tot/n:.4f} valid_rho {v:.4f}"
                  f" (best {best:.4f} @ep{best_ep}, 已等 {wait})", flush=True)
        if wait >= PATIENCE:
            print(f"      早停于 ep{ep}，回滚 best.ckpt ep{best_ep}", flush=True)
            break

    ck = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        pv = model(t_va).cpu().numpy()
        pt = model(t_te).cpu().numpy()
    import pandas as pd
    pd.DataFrame(hist).to_csv(os.path.join(seed_dir, "history.csv"), index=False)
    np.save(os.path.join(seed_dir, "valid_pred.npy"), pv)
    np.save(os.path.join(seed_dir, "test_pred.npy"), pt)
    prog.write(json.dumps({"done": True, "epoch": int(hist[-1]["epoch"]),
                           "best": round(float(best), 6), "ts": time.time()},
                          ensure_ascii=False) + "\n")
    prog.close()
    return {"seed": seed, "best_epoch": best_ep, "best_valid_spearman": float(best),
            "valid_pred": pv, "test_pred": pt,
            **{f"valid_{k}": v for k, v in all_metrics(yva, pv).items()},
            **{f"test_{k}": v for k, v in all_metrics(yte, pt).items()}}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="esmc600m")
    ap.add_argument("--variant", required=True,
                    help="D_hand / B_hwt / C_hmut / A_hwt_hmut")
    ap.add_argument("--asm-dir", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, nargs=2, default=[512, 128])
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    asm = a.asm_dir or os.path.join(root, "assembled", a.enc)
    out_root = a.out_root or os.path.join(root, "runs")
    tag = a.tag or f"ddg_{a.enc}_{a.variant}"
    out_dir = os.path.join(out_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    device = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    z = {sp: np.load(os.path.join(asm, f"{sp}.npz")) for sp in ("train", "valid", "test")}
    key = f"X_{a.variant}"
    for sp, zz in z.items():
        assert key in zz, f"{asm}/{sp}.npz 里没有 {key}（可选：{list(zz)}）"
    d = {"Xtr": z["train"][key].astype(np.float32), "ytr": z["train"]["y"].astype(np.float32),
         "Xva": z["valid"][key].astype(np.float32), "yva": z["valid"]["y"].astype(np.float32),
         "Xte": z["test"][key].astype(np.float32), "yte": z["test"]["y"].astype(np.float32)}
    print("=" * 88)
    print(f"ΔΔG 训练 {tag}   设备 {device}  种子 {a.seeds} 个")
    print(f"  变体 {a.variant}  输入 {d['Xtr'].shape[1]} 维")
    print(f"  train {d['Xtr'].shape}  valid {d['Xva'].shape}  test {d['Xte'].shape}")
    print(f"  标签 train {d['ytr'].mean():+.3f}±{d['ytr'].std():.3f}  "
          f"test {d['yte'].mean():+.3f}±{d['yte'].std():.3f}")
    print("=" * 88, flush=True)

    val_preds, test_preds, infos = [], [], []
    for s in range(a.seeds):
        print(f"  --- 种子 {s} ---", flush=True)
        r = train_one(s, d, a, device, out_dir)
        val_preds.append(r.pop("valid_pred"))
        test_preds.append(r.pop("test_pred"))
        infos.append(r)
        print(f"      best@ep{r['best_epoch']}  valid_rho {r['best_valid_spearman']:.4f}"
              f"  | test rho {r['test_spearman']:.4f}  R² {r['test_r2']:+.4f}"
              f"  RMSE {r['test_rmse']:.3f} kcal/mol", flush=True)

    ens_v = np.mean(val_preds, 0)
    ens_t = np.mean(test_preds, 0)
    np.save(os.path.join(out_dir, "ensemble_valid_pred.npy"), ens_v)
    np.save(os.path.join(out_dir, "ensemble_test_pred.npy"), ens_t)
    np.save(os.path.join(out_dir, "y_valid.npy"), d["yva"])
    np.save(os.path.join(out_dir, "y_test.npy"), d["yte"])
    single = [i["test_spearman"] for i in infos]
    summary = {"tag": tag, "enc": a.enc, "variant": a.variant, "asm_dir": asm,
               "n_train": int(len(d["ytr"])), "n_valid": int(len(d["yva"])),
               "n_test": int(len(d["yte"])), "d_in": int(d["Xtr"].shape[1]),
               "seeds": a.seeds, "hidden": list(a.hidden), "dropout": a.dropout,
               "lr": a.lr, "wd": a.wd, "epochs": a.epochs, "batch": a.batch,
               "single_test_spearman": single,
               "single_mean": float(np.mean(single)), "single_std": float(np.std(single)),
               "ensemble_valid": all_metrics(d["yva"], ens_v),
               "ensemble_test": all_metrics(d["yte"], ens_t)}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    import pandas as pd
    pd.DataFrame(infos).to_csv(os.path.join(out_dir, "per_seed.csv"), index=False)

    print("\n" + "=" * 88)
    print(f"【{tag}】单种子 test Spearman {['%.4f' % s for s in single]}")
    print(f"  均值 {summary['single_mean']:.4f}  标准差 {summary['single_std']:.4f}")
    print(f"【{tag}】集成后 test 指标")
    for k, v in summary["ensemble_test"].items():
        print(f"    {k:9s} {v:.4f}")
    print("TRAIN_DDG_DONE")


if __name__ == "__main__":
    main()
