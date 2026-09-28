"""评估指标与显著性检验 —— 所有报数都从这里出，保证口径统一。

★ 本文件是从 `solubility-regression/src/metrics.py` **原样复制**过来的。
  为什么复制而不是跨项目 import：两个项目各自要能独立跑、独立打包交付，
  跨项目 import 会让"热稳定性项目"依赖另一个项目的目录结构。
  代价是两份代码要同步 —— 所以**指标定义与配对检验的做法必须一字不改地保持一致**，
  否则两条实验线的数字没法互相印证。改这里之前先去看那边有没有一起改。

指标
    spearman  主指标。秩相关，只看"排序对不对"。
              ★ 为什么它是主指标：工程上要的是"从几千个候选里挑出最可能可溶的那几十个去做实验"，
                这是排序问题，不是数值拟合问题。
              ★ 而且本数据的标签噪声很大（同一蛋白重复测量可差 37~67 个百分点，见数据卡），
                RMSE/MAE 会被噪声淹没，秩相关对这种随机噪声不敏感。
    pearson   线性相关，文献惯例，便于和已发表数字对比
    rmse/mae  绝对误差（单位：% 溶解度），但要对着噪声下界解读
    r2        决定系数，**和 Han et al. 2019 的 0.4115 对齐用的就是它**

★ 显著性检验：配对 bootstrap（不是看两个 CI 是否重叠）
    "两个独立估计的 95% CI 有重叠 ⇒ 差异不显著"是**错误推理**。
    两个模型吃的是同一批测试样本，难样本上会一起错、易样本上会一起对，
    误差是相关的 → Var(A−B) 远小于 Var(A)+Var(B)。
    用各自 CI 去比会丢掉相关性信息，结论过于保守。
    正确做法是在**同一批重采样索引**上同时算两个模型再相减，看差的分布跨不跨 0。
"""
import numpy as np


# --------------------------------------------------------------------------- 指标
def _rank(a):
    """平均秩（处理并列）—— 自己实现避免依赖 scipy 版本差异。

    ★★ 2026-09-27 改成**向量化**：原实现用纯 Python 的 while 循环给并列段取平均秩，
       每次算 Spearman 都要走一遍 len(a) 次 Python 循环（而且一次调用走两遍：y 和 p）。
       5.6 万条测试样本时，单次配对 bootstrap（2000 轮）实测要 ~100 秒，
       600M 的变体对照因此跑了 27 分钟 —— 而这个文件被两条实验线共用，代价是持续的。
    ★ 改动经 `tmp/verify_rank_fast.py` 逐位对拍验证：8 组合成并列形态 + 真实 600M
       四变体的 y/预测（各 5.6 万条）+ Spearman 与配对 bootstrap 同种子 60 轮，
       **全部 np.array_equal 为 True**，即"只加速、不改数"。
    ★ 等价性依据：并列段在排序后占下标 [i..j]，原实现赋 (i+1 + j+1)/2；
       令组大小 k = j-i+1，则 (2i+k+1)/2 = i + (k+1)/2，与这里的 start + (k+1)/2 相同。
       对整数做 /2 在二进制里精确 ⇒ 位级一致（含 k 为奇数时的 .5）。
       单元素组 k=1 得到 start+1，正好是原本的序数秩，不会被改写。
    """
    a = np.asarray(a, float)
    n = len(a)
    if n == 0:
        return np.empty(0, float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    if n == 1:
        r = np.ones(1, float)
        return r
    # 排序后相邻不等的位置 = 新并列段的起点
    newgrp = np.empty(n, bool)
    newgrp[0] = True
    np.not_equal(sa[1:], sa[:-1], out=newgrp[1:])
    starts = np.flatnonzero(newgrp)
    cnt = np.diff(np.append(starts, n))                    # 每段大小
    s_b = np.repeat(starts, cnt).astype(float)
    k_b = np.repeat(cnt, cnt).astype(float)
    r = np.empty(n, float)
    r[order] = s_b + (k_b + 1) / 2.0
    return r


def spearman(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3:
        return float("nan")
    ry, rp = _rank(y), _rank(p)
    ry, rp = ry - ry.mean(), rp - rp.mean()
    d = np.sqrt((ry ** 2).sum() * (rp ** 2).sum())
    return float((ry * rp).sum() / d) if d > 0 else float("nan")


def pearson(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    y, p = y - y.mean(), p - p.mean()
    d = np.sqrt((y ** 2).sum() * (p ** 2).sum())
    return float((y * p).sum() / d) if d > 0 else float("nan")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


METRICS = {"spearman": spearman, "pearson": pearson,
           "rmse": rmse, "mae": mae, "r2": r2}
HIGHER_BETTER = {"spearman": True, "pearson": True, "rmse": False,
                 "mae": False, "r2": True}


def all_metrics(y, p):
    return {k: f(y, p) for k, f in METRICS.items()}


# ------------------------------------------------------------------- bootstrap CI
def bootstrap_ci(y, p, metric="spearman", n_boot=2000, seed=0, alpha=0.05):
    """独立 bootstrap 的 95% CI（**只用于报告区间，不用于判显著性**）。"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    f = METRICS[metric]
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = f(y[idx], p[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(y, pA, pB, metric="spearman", n_boot=2000, seed=0):
    """配对 bootstrap：返回 (差值, 95%CI, p值)。判据是 **CI 是否跨 0**。

    两个模型必须在**同一批测试样本**上产生预测（同一 test split、同一顺序）。

    ★ delta 用**全样本上的点估计**，不用 bootstrap 分布的均值。
      原因：bootstrap 均值有 O(1/n) 的偏差，会让这里报的 Δ 与结果表里
      「两个 Spearman 直接相减」得到的 Δ 对不上（本项目实测差 0.0003）。
      报告里同一件事出现两个数 = 审稿人眼里的 bug。CI 与 p 值仍来自 bootstrap。
    """
    y = np.asarray(y, float)
    pA, pB = np.asarray(pA, float), np.asarray(pB, float)
    assert len(y) == len(pA) == len(pB), "两个模型必须在同一批样本上评估"
    f = METRICS[metric]
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = f(y[idx], pA[idx]) - f(y[idx], pB[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    frac_neg = float((d <= 0).mean())
    frac_pos = float((d >= 0).mean())
    pv = 2.0 * min(frac_neg, frac_pos)
    delta = float(f(y, pA) - f(y, pB))                       # ★ 点估计，非 d.mean()
    return delta, float(lo), float(hi), float(min(pv, 1.0))


def fmt_paired(name, diff, lo, hi, pv, metric="spearman", higher_better=True):
    sig = "显著" if (lo > 0 or hi < 0) else "不显著"
    return (f"{name:52s} Δ{metric}={diff:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
            f"p={pv:.3f}  → {sig}")
