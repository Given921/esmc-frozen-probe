"""把复现产物与 `*/expected/key_metrics.csv` 逐条比对，带**分级容差**，并检查趋势。

用法：
    python tools/verify.py --line solubility --pred <你的产物根目录>
    python tools/verify.py --line ddg        --pred <你的产物根目录>

★ 设计要点（为什么不是简单对数字）
  1. **容差分级**：与编码器无关的量容差为 0（必须逐位相同），训练出来的头放宽。
     用一个容差判所有锚点是错的（见 EXPECTED 里的 tol 列）。
  2. **趋势优先于绝对值**：即使绝对值有偏差，只要**三档的排序趋势**一致，
     结论就仍然成立 —— 而这通常是复现方真正关心的。
  3. **找不到要报出来**，绝不静默跳过（静默跳过 = 你以为全过了）。
"""
import argparse
import csv
import io
import os
import sys

# 与 make_expected.py 保持一致的三档顺序（★ 显式声明，不靠名字猜）
ENCS = ["esmc300m", "esmc600m", "esmc6b"]


# ---------------------------------------------------------------- 查文件
def find_file(root, name):
    """在 root 下递归找文件名 == name 的文件，返回**恰好一个**（0 或多个都报错）。"""
    hits = []
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if f == name:
                hits.append(os.path.join(dp, f))
    if len(hits) == 0:
        raise SystemExit(f"★ 在 {root} 下找不到 {name}")
    if len(hits) > 1:
        raise SystemExit(f"★ 在 {root} 下找到 {len(hits)} 个 {name}，请清理后重试：\n  "
                         + "\n  ".join(hits))
    return hits[0]


def rd(path):
    with io.open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick(rows, **conds):
    """精确匹配取**恰好一行**；不猜名字。"""
    hit = [r for r in rows if all(r.get(k) == v for k, v in conds.items())]
    return hit[0] if len(hit) == 1 else None


# ---------------------------------------------------------------- 定位产物值
def locate_sol(exp_rows, leak_rows, enc, config, metric):
    """返回 (实得值, 找不到的原因)。"""
    if config == "leak_homology_test" or config == "leak_random_test":
        split = "homology" if "homology" in config else "random"
        if metric != "ge25_pct":
            return None, f"未知 metric {metric}"
        r = pick(leak_rows, split=split, part="test")
        return (float(r["ge25_pct"]), None) if r else (None, f"leakage_summary 里没有 {split}/test")

    # 与编码器无关的基线与尺子：都在 baselines_ 里，按 pred 区分
    if config in ("length_ridge", "aac_ridge", "aac_rf", "esmc_ridge"):
        r = pick(exp_rows, experiment=f"baselines_{enc}_homology", pred=config)
        if not r:
            return None, f"没有 baselines_{enc}_homology / {config}（是否跑的不是 homology 划分？）"
        return (float(r["spearman"]), None)

    tag = {"mlp_hand": f"mlp_{enc}_hand_homology",
           "mlp_only": f"mlp_{enc}_homology"}.get(config)
    if tag is None:
        return None, f"未知 config {config}"
    r = pick(exp_rows, experiment=tag, pred="ensemble")
    if not r:
        return None, f"没有 {tag} / ensemble（这一档可能没跑）"
    return (float(r["spearman"]), None)


def locate_ddg(comp_rows, pair_rows, enc, config, metric):
    if config == "A_minus_B":
        r = pick(pair_rows, enc=enc, A="A_hwt_hmut", B="B_hwt", metric="spearman")
        if not r:
            return None, f"ddg_pairs 里没有 {enc} 的 A_hwt_hmut − B_hwt"
        key = "delta" if metric == "delta" else "p"
        return (float(r[key]), None)
    if config in ("D_hand", "B_hwt", "C_hmut", "A_hwt_hmut"):
        r = pick(comp_rows, enc=enc, variant=config)
        if not r:
            return None, f"ddg_compare 里没有 {enc} / {config}"
        return (float(r["spearman"]), None)
    return None, f"未知 config {config}"


# ---------------------------------------------------------------- 趋势检查
def check_trend(label, series, expect_rising=True):
    """series: [(enc, value)]。返回 (是否通过, 说明)。"""
    vals = [v for _e, v in series]
    if any(v is None for v in vals):
        return False, "有缺失，无法判断"
    rising = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    txt = " < ".join(f"{e}={v:.4f}" for e, v in series)
    if expect_rising and not rising:
        return False, f"{txt}  ★ 不是单调递增 —— 趋势与预期相反"
    return True, txt


# ---------------------------------------------------------------- 主流程
def run(line, pred_root, expected_path, summary_dir=None):
    exp = rd(expected_path)
    print(f"\n{'=' * 78}")
    print(f"{line} 线 · 产物目录: {pred_root}")
    if summary_dir:
        print(f"{' ' * len(line)}   汇总目录: {summary_dir}")
    print(f"{'=' * 78}\n")

    # ★ 找到多个同名产物时**不静默挑一个** —— 明确报错并提示用 --summary-dir 指定。
    def locate(name):
        if summary_dir:
            p = os.path.join(summary_dir, name)
            if not os.path.exists(p):
                raise SystemExit(f"★ {summary_dir} 下没有 {name}")
            return p
        return find_file(pred_root, name)

    if line == "solubility":
        exp_rows = rd(locate("all_experiments.csv"))
        leak_rows = rd(locate("leakage_summary.csv"))
        loc = lambda enc, cfg, m: locate_sol(exp_rows, leak_rows, enc, cfg, m)
    else:
        comp_rows = rd(locate("ddg_compare.csv"))
        pair_rows = rd(locate("ddg_pairs.csv"))
        loc = lambda enc, cfg, m: locate_ddg(comp_rows, pair_rows, enc, cfg, m)

    n_ok = n_fail = n_miss = 0
    got = {}                       # (config, enc) -> 实得值，用于趋势检查

    for r in exp:
        enc, cfg, metric = r["enc"], r["config"], r["metric"]
        want, tol = float(r["expected"]), float(r["tol"])
        val, why = loc(enc, cfg, metric)

        tag = f"{cfg:<18} {enc:<9} {metric:<9}"
        if val is None:
            n_miss += 1
            print(f"  [缺失] {tag} 期望 {want:<10.6f}  ← {why}")
            continue
        got[(cfg, enc)] = val
        diff = abs(val - want)
        if diff <= tol:
            n_ok += 1
            print(f"  [ OK ] {tag} 期望 {want:<10.6f} 实得 {val:<10.6f} "
                  f"差 {diff:.6f}  (容差 {tol:.6f})")
        else:
            n_fail += 1
            print(f"  [FAIL] {tag} 期望 {want:<10.6f} 实得 {val:<10.6f} "
                  f"差 {diff:.6f}  (容差 {tol:.6f})  ← 超出容差")

    # ---------------- 趋势检查（比绝对值更重要）
    print(f"\n{'-' * 78}\n趋势检查（★ 绝对值有偏差时，趋势对 ⇒ 结论仍成立）\n{'-' * 78}")
    trend_cfgs = ["mlp_hand"] if line == "solubility" else ["A_hwt_hmut"]
    for cfg in trend_cfgs:
        series = [(e, got.get((cfg, e))) for e in ENCS]
        if all(v is not None for _e, v in series):
            ok, txt = check_trend(cfg, series)
            print(f"  [{' OK ' if ok else 'FAIL'}] {cfg:<18} {txt}")
            if not ok:
                n_fail += 1

    # 与编码器无关的量：三档必须逐位相同
    indep = ["length_ridge", "aac_ridge", "aac_rf"] if line == "solubility" else ["D_hand"]
    for cfg in indep:
        vals = [got.get((cfg, e)) for e in ENCS]
        if any(v is None for v in vals):
            continue
        if len(set(round(v, 9) for v in vals)) == 1:
            print(f"  [ OK ] {cfg:<18} 三档逐位相同 = {vals[0]:.6f}  ← 管线正确性证据")
        else:
            print(f"  [FAIL] {cfg:<18} 三档**不同** {vals}  ★ 数据或划分被串了")
            n_fail += 1

    print(f"\n{'=' * 78}")
    print(f"汇总: {n_ok} OK / {n_fail} FAIL / {n_miss} 缺失   （共 {len(exp)} 条锚点）")
    print(f"{'=' * 78}")
    return n_fail, n_miss


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--line", required=True, choices=["solubility", "ddg"])
    ap.add_argument("--pred", required=True, help="你跑出来的产物根目录")
    ap.add_argument("--summary-dir", default=None,
                    help="汇总目录（当 --pred 下有多份同名产物时，用它明确指定；"
                         "例：<proj>/runs/_summary）")
    ap.add_argument("--expected", default=None, help="默认用仓库里的 expected/key_metrics.csv")
    a = ap.parse_args()

    exp_path = a.expected or os.path.join(here, "..", a.line, "expected", "key_metrics.csv")
    if not os.path.exists(exp_path):
        raise SystemExit(f"★ expected 快照不存在：{exp_path}")

    n_fail, n_miss = run(a.line, a.pred, exp_path, a.summary_dir)
    print()
    if n_fail or n_miss:
        print(f"VERIFY_FAIL  {n_fail} 条超差 / {n_miss} 条缺失")
        sys.exit(1)
    print("VERIFY_OK  全部锚点在容差内")


if __name__ == "__main__":
    main()
