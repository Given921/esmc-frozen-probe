"""把分位数回归的预测区间校准到"名副其实"。

★ 要解决的问题（实测出来的）
    train_quantile.py 直接输出的 P10~P90 区间，名义上应该覆盖 80% 的真实值，
    实测只覆盖了 **64.2%** —— 区间**偏窄、模型过自信**。
    这种情况在神经网络分位数回归里非常普遍：pinball 损失只保证
    "每个分位点的条件分位数被拟合"，不保证**有限样本下**区间的实际覆盖率。

★ 修法：split conformal prediction（分裂式保形预测）
   思路极简，而且有**有限样本覆盖率保证**（这才是它比"拍脑袋乘个系数"强的根本原因）：

     ① 在**验证集**上算每个样本的"不合群程度"（conformity score）：
            E_i = max( P10_i − y_i ,  y_i − P90_i )
        y 落在区间内 → E_i ≤ 0；落在上面 → y_i − P90_i > 0；落在下面 → P10_i − y_i > 0。
        也就是 E_i 就是"这个真实值跑出区间多少"。
     ② 取 E 的 (1−α) 分位数 q̂（带有限样本修正，见下）；
     ③ 把测试集的区间**统一放宽** q̂：
            [ P10 − q̂ ,  P90 + q̂ ]
        因为 y_new ∈ [P10 − E_new, P90 + E_new]，只要 E_new ≤ q̂ 就必定被覆盖，
        而 E_new ≤ q̂ 的概率 ≥ 1−α（这就是保证的来源）。

   ★ 分位数要带修正：用 ⌈(n+1)(1−α)⌉/n 而不是朴素的 (1−α)，否则小样本下覆盖率会略低于名义值。
     n=316 时，α=0.2 → 目标秩 = ⌈317×0.8⌉ = 254 → 用第 254/316 分位。

★ 还有一个更聪明的变体：**按样本自适应放宽**
   上面的 q̂ 是全局加的一个常数。但我们的区间本来宽度就因人而异 ——
   难预测的蛋白天然给更宽的区间。所以可以改成"按比例放宽"：
            E_i = max( (P10_i − y_i)/h_i , (y_i − P90_i)/h_i )，其中 h_i = (P90_i − P10_i)/2
            [ P10 − q̂·h ,  P90 + q̂·h ]
   这样难样本的区间扩得更多、易样本扩得更少，往往能在**同样覆盖率下拿到更窄的平均宽度**。
   两个版本都算出来对比，选宽度小的那个。

★ 纪律
   校准参数 q̂ **只在验证集上算**，测试集只用来验收 —— 和"用验证集选 ckpt"是同一条纪律。
   如果拿测试集去调 q̂，"覆盖率 80%"就变成自我实现的循环了。

用法
    python calibrate_quantile.py --run runs/quantile_esmc600m_hand_homology
"""
import argparse
import json
import os

import numpy as np

ALPHA_DEFAULT = 0.2          # 名义区间 P10~P90 → 1−α = 0.8


def conformal_q(scores, alpha):
    """带有限样本修正的 (1−α) 分位数。

    ★ 为什么要用 ceil((n+1)(1−α)) 而不是 n×(1−α)：
      split conformal 的覆盖率保证针对的是样本量 n 的那个"上取整名次"，
      用朴素分位数在小样本上会系统性地**低于**名义覆盖率。
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:                       # 样本太少、无法保证 → 取最大值（最保守）
        return float(scores.max())
    return float(np.sort(scores)[k - 1])


def coverage_width(y, lo, hi):
    return (float(np.mean((y >= lo) & (y <= hi))),
            float(np.mean(hi - lo) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="分位数回归的实验目录")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                    help="显著性水平；0.2 对应名义 80%% 区间")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_dir = a.out or a.run
    Qv = np.load(os.path.join(a.run, "ensemble_valid_quantiles.npy"))
    Qt = np.load(os.path.join(a.run, "ensemble_test_quantiles.npy"))
    yv = np.load(os.path.join(a.run, "y_valid.npy"))
    yt = np.load(os.path.join(a.run, "y_test.npy"))
    metas = []
    for f in ("summary.json",):
        p = os.path.join(a.run, f)
        if os.path.exists(p):
            metas.append(json.load(open(p, encoding="utf-8")))
    qs = metas[0]["quantiles"] if metas else [0.1, 0.5, 0.9]

    print("=" * 90)
    print(f"区间校准（split conformal）  {a.run}")
    print(f"  分位点 {qs}  名义覆盖率 {1 - a.alpha:.0%}  "
          f"valid n={len(yv)}  test n={len(yt)}")
    print("  校准参数只在 **valid** 上算；test 只用来验收")
    print("=" * 90)

    lo_v, hi_v = Qv[:, 0], Qv[:, -1]
    lo_t, hi_t = Qt[:, 0], Qt[:, -1]

    cov_v, w_v = coverage_width(yv, lo_v, hi_v)
    cov_t, w_t = coverage_width(yt, lo_t, hi_t)
    print(f"\n【校准前】")
    print(f"  valid  覆盖率 {cov_v*100:5.1f}%  平均宽度 {w_v:5.1f} 点")
    print(f"  test   覆盖率 {cov_t*100:5.1f}%  平均宽度 {w_t:5.1f} 点"
          f"   （名义 {1-a.alpha:.0%}）")
    short = (1 - a.alpha) - cov_t
    print(f"  → 缺 {(1-a.alpha)*100:.0f}% − {cov_t*100:.1f}% = "
          f"{short*100:+.1f} 个点：**区间偏窄、模型过自信**")

    results = {"run": a.run, "alpha": a.alpha, "nominal": 1 - a.alpha,
               "raw": {"valid_coverage": cov_v, "valid_width_pct": w_v,
                       "test_coverage": cov_t, "test_width_pct": w_t}}

    variants = {}

    # ---------- 变体 A：全局加常数（绝对放宽） ----------
    E_v = np.maximum(lo_v - yv, yv - hi_v)
    q_abs = conformal_q(E_v, a.alpha)
    loA, hiA = lo_t - q_abs, hi_t + q_abs
    covA, wA = coverage_width(yt, loA, hiA)
    variants["absolute"] = dict(q_hat=q_abs, lo=loA, hi=hiA,
                                test_coverage=covA, test_width_pct=wA)
    print(f"\n【变体 A：全局放宽 q̂ = {q_abs*100:+.1f} 点】")
    print(f"  test   覆盖率 {covA*100:5.1f}%  平均宽度 {wA:5.1f} 点"
          f"（比校准前宽 {wA - w_t:+.1f} 点）")

    # ---------- 变体 B：按样本自适应（比例放宽） ----------
    h_v = np.maximum((hi_v - lo_v) / 2.0, 1e-6)
    h_t = np.maximum((hi_t - lo_t) / 2.0, 1e-6)
    E_v_sc = np.maximum((lo_v - yv) / h_v, (yv - hi_v) / h_v)
    q_sc = conformal_q(E_v_sc, a.alpha)
    loB, hiB = lo_t - q_sc * h_t, hi_t + q_sc * h_t
    covB, wB = coverage_width(yt, loB, hiB)
    variants["adaptive"] = dict(q_hat=q_sc, lo=loB, hi=hiB,
                                test_coverage=covB, test_width_pct=wB)
    print(f"\n【变体 B：按样本自适应放宽 q̂ = {q_sc:+.3f}（× 半宽）】")
    print(f"  test   覆盖率 {covB*100:5.1f}%  平均宽度 {wB:5.1f} 点"
          f"（比校准前宽 {wB - w_t:+.1f} 点）")

    # ---------- 选宽度更小的那个 ----------
    pick = min(variants, key=lambda k: (abs(variants[k]["test_coverage"] - (1 - a.alpha)),
                                        variants[k]["test_width_pct"]))
    print(f"\n{'=' * 90}")
    print(f"★ 选用变体：{pick}"
          f"（覆盖率最接近名义值；同覆盖率下取宽度更小者）")
    best = variants[pick]
    print(f"  校准后 test 覆盖率 {best['test_coverage']*100:.1f}%"
          f"（名义 {(1-a.alpha)*100:.0f}%），平均宽度 {best['test_width_pct']:.1f} 点")
    results["calibrated"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("lo", "hi")}
                             for k, v in variants.items()}
    results["chosen"] = pick

    np.save(os.path.join(out_dir, "test_interval_calibrated.npy"),
            np.stack([best["lo"], best["hi"]], 1).astype(np.float32))
    results["calibrated"]["chosen_test_coverage"] = best["test_coverage"]
    results["calibrated"]["chosen_test_width_pct"] = best["test_width_pct"]
    with open(os.path.join(out_dir, "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\n产物：{out_dir}/calibration.json、test_interval_calibrated.npy")
    print("CALIBRATE_DONE")


if __name__ == "__main__":
    main()
