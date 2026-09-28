"""第 4 步：训练轻量头部网络。**验证集选 ckpt** 这一步就在这里，且是硬要求。

★ 什么是 ckpt（checkpoint，检查点）—— 这个项目里最需要说清的一件事
    训练一个神经网络，不是"跑完就有模型"，而是一轮一轮（epoch）地更新参数。
    每训练一轮，参数就变一次，**每一轮其实都是一个不同的模型**。
    ckpt 就是"把某一轮结束时的全部参数（和优化器状态）存到磁盘上的那个文件"。

    为什么要存：
      ① 训练会跑很多轮，最后一轮往往不是最好的（小模型上尤其明显，几百轮后开始过拟合）
      ② 断电 / 被抢占 / 显存炸了，可以从最近一个 ckpt 接着跑
      ③ 想比较"第 50 轮 vs 第 200 轮"，有 ckpt 才能回看

★ 怎么选 "最好的那一个" —— 这是能不能信的关键
    绝不能用测试集来选。用测试集选 = 你已经"偷看"了测试集，
    报出来的分数是乐观偏高的，这个错误在文献里非常常见。

    正确做法（本脚本的做法）：
        每训练一轮 →
          在 **验证集** 上算主指标（本项目 = Spearman 秩相关）→
          如果比"历史最好"还好，就把当前参数存成 best.ckpt →
        训练结束后，**用 best.ckpt 去测试集上评估一次**（只评估一次！）

    验证集（valid）: 训练过程中用来做选择、调超参、早停
    测试集（test）  : **只在最后用一次**，报告性能。训练期间绝不碰。

★ 早停（early stopping）
    连续 PATIENCE 轮验证指标都没进步就停下，并回滚到 best.ckpt。
    省时间，也防止过拟合。

★ 多种子 + 集成
    单个种子的结果波动可达 0.03 以上（本任务实测），只看一次不可信。
    默认跑 3 个种子，**把 3 个种子的预测取平均**再报数（这就是"集成"）。
    同时把每个种子的单独结果也存下来，方便看方差。

用法：
    python train.py --feat ../features/esmc600m_mean --tag esmc600m_mean
    python train.py --feat ../features/esmc600m_mean --hand --tag esmc600m_mean+hand
    python train.py --feat ../features/esmc600m_mean --out-act linear --tag linear实验
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
from metrics import all_metrics, spearman                        # noqa: E402
from model import Normalizer, SolubilityHead                     # noqa: E402

PATIENCE = 30


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_split(feat_dir, split, split_dir, use_hand, index=None):
    """读一个划分的特征 + 标签。

    ★ index 参数（可选）：只取这些行序号。
      用途是**数据量消融**（学习曲线）：固定 valid/test，只让 train 变小。
        ① 子集必须**只用行号**表达，不重新提特征 —— 否则横轴变的不只是"数据量"，
           还混进了"重新算了一遍特征"这个变量；
        ② 同一个行号表可以喂给 300M/600M/6B 三个编码器 ——
           特征维度不同但**行的顺序是同一批序列**，所以三个编码器的曲线是配对的；
        ③ 默认 None = 用全部行，**行为与加这个参数之前逐字节一致**，
           已有实验的数字不会因此改变（可比性优先）。
    """
    z = np.load(os.path.join(feat_dir, f"{split}.npz"))
    y_all = z["y"].astype(np.float32)
    X = z["X"].astype(np.float32)
    y = y_all
    if index is not None:
        X = X[index]
        y = y_all[index]
    if use_hand:
        df = pd.read_csv(os.path.join(split_dir, f"{split}.csv"))
        # ★ 这个断言要拿**原始**长度比，不能用子采样后的 y：
        #   否则一旦索引文件和划分文件来自不同版本的划分，断言照样通过，
        #   但手工特征会和嵌入张冠李戴（静默错位，最难查）。
        assert len(df) == len(y_all), f"{split} 特征与划分文件条数不符"
        if index is not None:
            df = df.iloc[index].reset_index(drop=True)
        H = handcrafted_features(df["sequence"].tolist())
        X = np.concatenate([X, H], 1)
    return X, y


def train_one(seed, Xtr, ytr, Xva, yva, Xte, yte, out_dir, args, device):
    """训练一个种子。返回 (valid_pred, test_pred, history, best_info)。"""
    set_seed(seed)
    norm = Normalizer.fit(Xtr)                       # ★ 统计量只用训练集
    Xtr_n, Xva_n, Xte_n = (norm.transform(Xtr), norm.transform(Xva), norm.transform(Xte))

    t_tr = torch.from_numpy(Xtr_n).to(device)
    t_ytr = torch.from_numpy(ytr).to(device)
    t_va = torch.from_numpy(Xva_n).to(device)
    t_te = torch.from_numpy(Xte_n).to(device)

    model = SolubilityHead(Xtr.shape[1], hidden=tuple(args.hidden),
                           dropout=args.dropout, out_act=args.out_act).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)     # ★ 只用优化器的 L2，损失里不再加
    lossf = nn.MSELoss()

    n = len(Xtr)
    steps = max(1, n // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.15)

    seed_dir = os.path.join(out_dir, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    ckpt_path = os.path.join(seed_dir, "best.ckpt")
    # ★★ 逐轮落盘的进度文件 —— 让看板能"实时"看到训练走到哪。
    #   为什么不能只靠 history.csv：那个文件是在训练**循环结束之后**才写的，
    #   训练进行中它根本不存在（实测确认），看板只能盯着日志文本猜。
    #   这里每轮 append 一行 JSON，代价可以忽略，换来秒级精度的实时进度 + 实时曲线。
    #   注意用 "w" 打开：同一个 seed 重跑时要覆盖旧进度，否则曲线会接在上一轮的后面。
    prog_path = os.path.join(seed_dir, "progress.jsonl")
    prog_fh = open(prog_path, "w", encoding="utf-8", buffering=1)
    n_planned = int(args.epochs)
    # ★ 先写一条"启动"记录（epoch=0）。
    #   不写的话，一个刚起来、还没跑完第一轮的种子在看板上**完全不可见**
    #   （空文件会被直接跳过）——于是"任务到底起没起来"没法确认。
    #   万一它在第一轮就被杀，这条记录也能让它显示成"已中断"而不是凭空消失。
    prog_fh.write(json.dumps({"epoch": 0, "epochs": n_planned, "seed": int(seed),
                              "train_loss": None, "valid_spearman": None,
                              "best": None, "best_epoch": None, "wait": 0,
                              "started": True, "ts": time.time()},
                             ensure_ascii=False) + "\n")

    best = -np.inf
    best_ep = -1
    wait = 0
    hist = []

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
        if args.out_act == "linear":
            pv_eval = np.clip(pv, 0.0, 1.0)
        else:
            pv_eval = pv
        # ★★ 用**验证集**的主指标选 ckpt
        v_metric = spearman(yva, pv_eval)
        hist.append(dict(epoch=ep, train_loss=tot / n, valid_spearman=v_metric))

        if v_metric > best:
            best, best_ep, wait = v_metric, ep, 0
            # ★ 存成"只有张量和基本类型"的字典。
            #   为什么不能直接存 numpy：torch ≥ 2.6 起 torch.load 的 weights_only 默认为 True，
            #   而 numpy 数组反序列化需要 numpy.core.multiarray._reconstruct ——
            #   它不在白名单里，于是 torch.load 会抛 UnpicklingError。
            #   把 mu/sigma 转成 tensor、把指标转成 python float，就能在 weights_only=True 下安全加载。
            torch.save({"model": model.state_dict(),
                        "norm_mu": torch.as_tensor(norm.mu),
                        "norm_sigma": torch.as_tensor(norm.sigma),
                        "d_in": int(Xtr.shape[1]), "hidden": tuple(args.hidden),
                        "dropout": float(args.dropout), "out_act": str(args.out_act),
                        "epoch": int(ep), "valid_spearman": float(v_metric),
                        "seed": int(seed)}, ckpt_path)
        else:
            wait += 1

        # 逐轮写进度（含时间戳，方便看板判断"是不是卡住了"）
        prog_fh.write(json.dumps({
            "epoch": int(ep), "epochs": n_planned, "seed": int(seed),
            "train_loss": round(float(tot / n), 6),
            "valid_spearman": round(float(v_metric), 6),
            "best": round(float(best), 6), "best_epoch": int(best_ep),
            "wait": int(wait), "ts": time.time(),
        }, ensure_ascii=False) + "\n")

        if ep % args.log_every == 0 or ep == 1:
            print(f"      ep{ep:4d}  loss {tot/n:.5f}  valid_spearman {v_metric:.4f}"
                  f"  (best {best:.4f} @ep{best_ep}, 已等 {wait})", flush=True)
        if wait >= PATIENCE:
            print(f"      早停于 ep{ep}（{PATIENCE} 轮无进步），回滚到 best.ckpt ep{best_ep}",
                  flush=True)
            break

    # ---- 只加载 best.ckpt 去预测（验证集与测试集各一次）----
    # weights_only=True 是 torch ≥ 2.6 的默认值；上面的 torch.save 已保证 ckpt 里只有
    # 张量 + 基本类型，所以这里能安全加载（不需要为了兼容去关掉它）。
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        pv_best = model(t_va).cpu().numpy()
        pt_best = model(t_te).cpu().numpy()
    if args.out_act == "linear":
        pv_best, pt_best = np.clip(pv_best, 0, 1), np.clip(pt_best, 0, 1)

    pd.DataFrame(hist).to_csv(os.path.join(seed_dir, "history.csv"), index=False)
    # ★ 收尾标记：写一行 done 再关闭。
    #   看板据此区分「正常跑完」与「中途被杀」—— 后者不会有这一行，
    #   于是看板能明确报警"进程不在了但没跑完"，而不是傻等。
    prog_fh.write(json.dumps({"done": True, "epoch": int(hist[-1]["epoch"]),
                              "best": round(float(best), 6),
                              "best_epoch": int(best_ep),
                              "ts": time.time()}, ensure_ascii=False) + "\n")
    prog_fh.close()
    info = dict(seed=seed, best_epoch=best_ep, best_valid_spearman=best,
                n_params=model.n_params(),
                **{f"valid_{k}": v for k, v in all_metrics(yva, pv_best).items()},
                **{f"test_{k}": v for k, v in all_metrics(yte, pt_best).items()})
    return pv_best, pt_best, info


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True, help="特征目录（extract_features.py 的产物）")
    ap.add_argument("--split-dir", default=os.path.join(root, "data", "splits", "homology"))
    ap.add_argument("--out-root", default=os.path.join(root, "runs"))
    ap.add_argument("--tag", required=True, help="本次实验名，决定输出目录")
    ap.add_argument("--hand", action="store_true", help="拼上手工生物物理特征")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, nargs=2, default=[512, 128])
    ap.add_argument("--out-act", choices=["sigmoid", "linear"], default="sigmoid")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--train-index", default=None,
                    help="训练集只取这些行序号（.npy int 数组）；默认 None=全部。"
                         "数据量消融用。★ 只作用于 train，valid/test 不受影响。")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    device = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(a.out_root, a.tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 88)
    print(f"训练 {a.tag}   设备 {device}   种子 {a.seeds} 个")
    print(f"  特征 {a.feat}{'  +手工特征' if a.hand else ''}")
    print(f"  划分 {a.split_dir}")
    print(f"  输出 {out_dir}")
    # ---- 可选的训练集子采样（数据量消融）----
    tr_idx = None
    if a.train_index:
        tr_idx = np.load(a.train_index)
        assert tr_idx.ndim == 1 and tr_idx.dtype.kind in "iu", \
            f"训练集索引必须是 1 维整数数组，实际 shape={tr_idx.shape} dtype={tr_idx.dtype}"
        n_all = int(np.load(os.path.join(a.feat, "train.npz"))["y"].shape[0])
        assert tr_idx.size > 0, "训练集索引是空的"
        assert int(tr_idx.min()) >= 0 and int(tr_idx.max()) < n_all, \
            f"训练集索引越界（[0,{n_all})），最大 {int(tr_idx.max())}"
        assert len(set(tr_idx.tolist())) == tr_idx.size, "训练集索引里有重复行号"
        print(f"  ★ 训练集子采样：{tr_idx.size} / {n_all} 条"
              f"  （索引 {a.train_index}）")
    print("=" * 88)

    Xtr, ytr = load_split(a.feat, "train", a.split_dir, a.hand, tr_idx)
    Xva, yva = load_split(a.feat, "valid", a.split_dir, a.hand)
    Xte, yte = load_split(a.feat, "test", a.split_dir, a.hand)
    print(f"\n数据：train {Xtr.shape}  valid {Xva.shape}  test {Xte.shape}")
    print(f"      标签(0~1) train 均值 {ytr.mean():.3f}  std {ytr.std():.3f}\n")

    val_preds, test_preds, infos = [], [], []
    for s in range(a.seeds):
        print(f"  --- 种子 {s} ---")
        pv, pt, info = train_one(s, Xtr, ytr, Xva, yva, Xte, yte, out_dir, a, device)
        val_preds.append(pv)
        test_preds.append(pt)
        infos.append(info)
        print(f"      best.ckpt @ep{info['best_epoch']}  "
              f"valid Spearman {info['best_valid_spearman']:.4f}  "
              f"| test Spearman {info['test_spearman']:.4f}  "
              f"R² {info['test_r2']:.4f}  RMSE {info['test_rmse']*100:.2f}%")
        np.save(os.path.join(out_dir, f"seed{s}", "valid_pred.npy"), pv)
        np.save(os.path.join(out_dir, f"seed{s}", "test_pred.npy"), pt)

    # ---- 多种子集成（平均预测）----
    ens_v = np.mean(val_preds, 0)
    ens_t = np.mean(test_preds, 0)
    np.save(os.path.join(out_dir, "ensemble_valid_pred.npy"), ens_v)
    np.save(os.path.join(out_dir, "ensemble_test_pred.npy"), ens_t)
    np.save(os.path.join(out_dir, "y_valid.npy"), yva)
    np.save(os.path.join(out_dir, "y_test.npy"), yte)

    single = [i["test_spearman"] for i in infos]
    summary = {
        "tag": a.tag, "feat": a.feat, "hand": a.hand, "out_act": a.out_act,
        "split_dir": a.split_dir, "seeds": a.seeds,
        # ★ 记下训练集到底用了多少条 —— 学习曲线要靠这个字段认横轴，
        #   没有它就只能靠 tag 里的字符串猜（本项目已经在别处踩过这个坑）。
        "n_train": int(len(ytr)), "train_index": a.train_index,
        "n_valid": int(len(yva)), "n_test": int(len(yte)),
        "n_params_head": infos[0]["n_params"],
        "single_test_spearman": single,
        "single_mean": float(np.mean(single)), "single_std": float(np.std(single)),
        "single_min": float(np.min(single)), "single_max": float(np.max(single)),
        "ensemble_valid": all_metrics(yva, ens_v),
        "ensemble_test": all_metrics(yte, ens_t),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    pd.DataFrame(infos).to_csv(os.path.join(out_dir, "per_seed.csv"), index=False)

    print("\n" + "=" * 88)
    print(f"【{a.tag}】多种子结果（单个种子 test Spearman）")
    print(f"  {['%.4f' % s for s in single]}")
    print(f"  均值 {summary['single_mean']:.4f}  标准差 {summary['single_std']:.4f}  "
          f"极差 {summary['single_max']-summary['single_min']:.4f}")
    print(f"  ★ 单种子波动 {summary['single_std']:.4f}"
          + ("  —— 波动很大，只看一次结果不可信，必须报集成" if summary['single_std'] > 0.01 else ""))
    print(f"\n【{a.tag}】集成后的 test 指标（这才是对外报的数）")
    for k, v in summary["ensemble_test"].items():
        print(f"    {k:9s} {v:.4f}")
    print(f"\n产物：{out_dir}")
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
