"""热稳定性 ΔΔG · 变体对照（配对 bootstrap），并给看板写一份状态。

★ 核心交付就是这个脚本的输出
    变体 A(h_wt + h_mut) − 变体 B(h_wt) 的差值
    = **"多花几卡·天把突变体序列也跑一遍编码器，到底买回几个点"**
    文献里少见直接给这个数的，所以我们把它作为一个明确的交付结论。

★ 判显著性只认配对检验
    四个变体吃的是**同一批测试样本**（同一划分、同一顺序，脚本里会断言），
    误差相关 ⇒ "两个独立 95% CI 重叠所以不显著"是错误推理（见 metrics.py 注释）。
    这里全部走 paired_bootstrap，并且 delta 用全样本的点估计（不是 bootstrap 均值，
    避免 O(1/n) 偏差导致报告里同一个 Δ 出现两个数）。

用法
    python src/compare_ddg.py --enc esmc600m
    python src/compare_ddg.py --encs esmc300m esmc600m esmc6b
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import all_metrics, bootstrap_ci, paired_bootstrap   # noqa: E402

VARIANTS = ["D_hand", "B_hwt", "C_hmut", "A_hwt_hmut"]
VSHORT = {"D_hand": "D 无编码器", "B_hwt": "B h_wt", "C_hmut": "C h_mut",
          "A_hwt_hmut": "A h_wt+h_mut"}
# 要报的配对（有向：前 − 后）
PAIRS = [("A_hwt_hmut", "B_hwt"),     # ★ 核心：h_mut 的增量
         ("A_hwt_hmut", "C_hmut"),    # 有 h_wt 上下文 vs 没有
         ("B_hwt", "D_hand"),         # h_wt 的增量
         ("C_hmut", "D_hand"),        # h_mut 单独的增量
         ("A_hwt_hmut", "D_hand")]    # 全部编码器信息 vs 一点不用


def load_variants(root, enc, variants):
    """读四个变体的集成测试预测。**断言 y_test 完全一致**（配对检验的前提）。"""
    out, yref = {}, None
    for v in variants:
        d = os.path.join(root, "runs", f"ddg_{enc}_{v}")
        fp = os.path.join(d, "ensemble_test_pred.npy")
        if not (os.path.exists(fp) and os.path.exists(os.path.join(d, "summary.json"))):
            print(f"  [缺] {enc} · {v}（还没跑）")
            continue
        y = np.load(os.path.join(d, "y_test.npy"))
        p = np.load(fp)
        if yref is None:
            yref = y
        else:
            assert len(y) == len(yref) and np.allclose(y, yref), \
                (f"{enc} 的变体 {v} 用的 y_test 与其他变体不一致 —— "
                 f"配对检验要求同一批样本同一顺序，这必须先查清再比")
        out[v] = p
    return out, yref


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--encs", nargs="*", default=["esmc600m"])
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--n-boot", type=int, default=2000)
    # ★ --require-all：只要有任何一个（编码器, 变体）没跑出结果，就**非 0 退出**。
    #   全链路自动化里必须打开它。原因见文件末尾那段注释：本脚本的完成标记
    #   COMPARE_DDG_DONE 原来是**无条件打印**的，"0 行结果"也照打，
    #   于是下游拿它当判据必然假通过。
    ap.add_argument("--require-all", action="store_true",
                    help="缺任何（编码器, 变体）组合就算失败；全链路自动化用")
    # ★ --res：结果根目录（其下有 runs/）。
    #   脚本与结果同根时 ⇒ 默认值就是"脚本上一级"，行为不变；
    #   若结果被单独搬到了别处 ⇒ 用 --res 指向它即可，
    #   不必把数据搬来搬去（实测踩过：直接跑 → 读不到任何变体、
    #   把汇总表覆盖成 0 行）。
    ap.add_argument("--res", default=None,
                    help="结果根目录（含 runs/）；默认 = 脚本上一级（服务器布局）")
    # ★ --force-write-partial：结果不全时仍然写 csv **仅用于排查**。
    #   默认**不写** —— 见文件末尾那段"覆盖即毁数据"的教训。
    ap.add_argument("--force-write-partial", action="store_true",
                    help="结果不全时也写 csv（只用于排查，绝不要当交付）")
    a = ap.parse_args()

    print("=" * 100)
    print(f"ΔΔG 变体对照   编码器 {a.encs}   配对 bootstrap {a.n_boot} 次")
    print("=" * 100, flush=True)
    res_root = a.res or root
    print(f"  结果根目录：{res_root}", flush=True)

    rows, pairs, enc_best, missing = [], [], {}, []
    for enc in a.encs:
        P, y = load_variants(res_root, enc, a.variants)
        # ★ 在 continue 之前就要记账，否则"一个变体都没跑"的编码器会被漏掉
        missing += [(enc, v) for v in a.variants if v not in P]
        if not P:
            print(f"  {enc}: 一个变体都没跑，跳过")
            continue
        n = len(y)
        print(f"\n--- {enc}（测试 {n:,} 条，标签 {y.mean():+.3f}±{y.std():.3f}）---")
        for v, p in P.items():
            m = all_metrics(y, p)
            lo, hi = bootstrap_ci(y, p, "spearman", n_boot=a.n_boot, seed=0)
            rows.append(dict(enc=enc, variant=v, variant_label=VSHORT.get(v, v),
                             n_test=n, spearman=round(m["spearman"], 4),
                             spearman_lo=round(lo, 4), spearman_hi=round(hi, 4),
                             pearson=round(m["pearson"], 4), r2=round(m["r2"], 4),
                             rmse=round(m["rmse"], 4), mae=round(m["mae"], 4)))
            print(f"  {VSHORT.get(v, v):16s} Spearman {m['spearman']:.4f} "
                  f"[{lo:.4f},{hi:.4f}]  Pearson {m['pearson']:.4f}  "
                  f"RMSE {m['rmse']:.3f} kcal/mol")
        # ---- 配对检验 ----
        for A, B in PAIRS:
            if A not in P or B not in P:
                continue
            for metric in ("spearman", "pearson"):
                d, lo, hi, pv = paired_bootstrap(y, P[A], P[B], metric,
                                                 n_boot=a.n_boot, seed=0)
                sig = bool(lo > 0 or hi < 0)
                pairs.append(dict(enc=enc, A=A, B=B, metric=metric,
                                  delta=round(d, 4), lo=round(lo, 4),
                                  hi=round(hi, 4), p=round(pv, 4),
                                  significant=sig))
                if metric == "spearman":
                    tag = "★显著" if sig else "不显著"
                    print(f"    {VSHORT.get(A, A)} − {VSHORT.get(B, B):16s} "
                          f"Δspearman {d:+.4f}  [{lo:+.4f},{hi:+.4f}]  "
                          f"p={pv:.3f}  → {tag}")
        if rows:
            best = max([r for r in rows if r["enc"] == enc], key=lambda r: r["spearman"])
            enc_best[enc] = {"variant": best["variant"], "spearman": best["spearman"]}

    out_dir = os.path.join(res_root, "runs", "_summary")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    dp = pd.DataFrame(pairs)

    # ★★ 结果不全时**拒绝覆盖**已存在的汇总表（2026-09-27 实测事故）
    #   原来的行为：把 0 行/不完整的结果直接写进 ddg_compare.csv，
    #   只在最后打印一句"已被覆盖，只能用于排查"。结果就是
    #   **三档汇总表被两档覆盖 → 6B 那一行没了**，而且旧文件无法恢复
    #   （本次幸好本地 runs/ 还在，才得以重算；若当时是唯一副本就永久丢失）。
    #   "警告 + 照写" 等于没有保护：出了事才看到警告，数据已经没了。
    #   ⇒ 默认改为**不写**；确实要写残缺结果排查，用 --force-write-partial 显式打开。
    cmp_fp = os.path.join(out_dir, "ddg_compare.csv")
    pair_fp = os.path.join(out_dir, "ddg_pairs.csv")
    if missing and not a.force_write_partial:
        _old = [f for f in (cmp_fp, pair_fp) if os.path.exists(f)]
        print(f"\n★ 有 {len(missing)} 个（编码器, 变体）没有结果：{missing}")
        print(f"  ⇒ 本次**不写**汇总表（保护已有产物{'：' + '、'.join(os.path.basename(f) for f in _old) if _old else ''}）。")
        print("     要写残缺结果排查，加 --force-write-partial。")
        print("COMPARE_DDG_INCOMPLETE")
        if a.require_all:
            print("  --require-all：按失败退出（退出码 1）")
        sys.exit(1)

    df.to_csv(cmp_fp, index=False)
    dp.to_csv(pair_fp, index=False)

    # ---- 自检：不能有非有限数 ----
    import math
    bad = [x for x in list(df.get("spearman", [])) + list(dp.get("delta", []))
           if not math.isfinite(float(x))]
    assert not bad, f"对照表里出现非有限值：{bad}，中止"

    # ---- 给看板写状态（看板不需要读 csv，只要一句话 + 进度）----
    core = [p for p in pairs if p["metric"] == "spearman"
            and p["A"] == "A_hwt_hmut" and p["B"] == "B_hwt"]
    note = ""
    if core:
        c = core[0]
        note = (f"核心结论 h_mut 增量 Δspearman {c['delta']:+.4f}"
                f"（p={c['p']:.3f}，{'显著' if c['significant'] else '不显著'}）")
    st = {"updated": time.time(), "task": "ΔΔG 变体对照",
          "stage": "done" if not missing else "incomplete",
          "done": len(df), "total": len(df), "unit": "个配置",
          "pct": 100.0 if not missing else 0.0,
          "eta_sec": 0, "notes": note or "（还没有完整结果）",
          "encoders": enc_best}
    with open(os.path.join(res_root, "status.json"), "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=2, ensure_ascii=False)

    print(f"\n产物：runs/_summary/ddg_compare.csv（{len(df)} 行）"
          f"  runs/_summary/ddg_pairs.csv（{len(dp)} 行）")
    if note:
        print("  " + note)
    # ★ 完成标记必须"真的代表全成功"。
    #   原来是无条件打印 COMPARE_DDG_DONE 的：实测 logs/compare_esmc600m.log
    #   里明明写着"一个变体都没跑，跳过"、产物 0 行，末尾却照样有那行标记。
    #   下游如果拿它当判据就会一路假通过（这是本项目踩的第三次"假标记"：
    #   第一次 run_ddg.sh 的训练段、第二次同文件的 compare 段、第三次这里）。
    #   ★ 2026-09-27 补：又发现同源的第二层问题 —— 标记改对了，但**文件照样被写**。
    #     现在"缺结果"已在上面提前 sys.exit(1)，走到这里必定是完整的；
    #     若人为加了 --force-write-partial，标记也绝不打印成完成态。
    if missing:
        print(f"\n★ 有 {len(missing)} 个（编码器, 变体）没有结果：{missing}")
        print("  （--force-write-partial：csv 已写，但只能用于排查，绝不要当交付）")
        print("COMPARE_DDG_INCOMPLETE")
        sys.exit(1)
    print("COMPARE_DDG_DONE")


if __name__ == "__main__":
    main()
