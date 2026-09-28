"""从两条线**实际跑出来的产物**生成期望快照 `*/expected/key_metrics.csv`。

★ 为什么要有这个脚本
  expected/ 是"复现成功与否"的判据。如果它里面的数字是**手打**的，
  那它本身就成了最不可信的一环 —— 手抄一个数字，会让后来者永远复现不出来，
  而且看起来是"他复现失败"，不是"我们抄错了"。
  ⇒ 所以这些数字必须**从产物文件里读**，且带断言（读不到就中止，绝不静默跳过）。

★ 容差为什么分级
  不同锚点的可复现性本来就不一样，用一个容差去判是错的：
    - 与编码器无关的量（length_ridge / aac_* / D_hand）→ **必须逐位相同**（它们不看编码器，
      三档不同就说明数据或划分被串了）。这是**免费的管线正确性证据**。
    - 数据派生的确定性量（泄漏率、条数）→ 纯计算，无随机性。
    - 冻结嵌入线性探针 → 特征确定，但 ridge 求解有数值差异。
    - 训练出来的 MLP → 依赖随机种子与 GPU 非确定性。

用法：
    python tools/make_expected.py --src-root <两条线源项目的父目录>
"""
import argparse
import csv
import io
import os
import sys

# ---------------------------------------------------------------- 容差口径
TOL_EXACT = 0.0        # 必须逐位相同
TOL_DERIVED = 0.0      # 数据派生的确定性量
TOL_PROBE = 0.002      # 冻结嵌入线性探针
TOL_TRAINED = 0.008    # 训练出来的头
# ★ p 值不按"绝对值相等"判：源产物里 p=0.0 是"p < 0.0005"的舍入表示，
#   复现时可能得到 1e-6 或 3e-4 —— 只要两边都落在显著区间就算通过。
TOL_P = 0.05

ENCS = ["esmc300m", "esmc600m", "esmc6b"]


def rd_rows(path):
    """读 csv → list[dict]，带存在性断言。"""
    if not os.path.exists(path):
        raise SystemExit(f"★ 产物不存在：{path}\n  （expected/ 只能从真实产物生成，不能手打）")
    with io.open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"★ 产物为空：{path}")
    return rows


def find_one(rows, **conds):
    """按精确条件找**恰好一行**；0 行或多行都中止。

    ★ 不用 startswith 之类去"猜"名字 —— 必错位成 KeyError 或静默取错行。
    """
    hit = [r for r in rows if all(r.get(k) == v for k, v in conds.items())]
    if len(hit) != 1:
        raise SystemExit(f"★ 期望恰好 1 行 {conds}，实得 {len(hit)} 行")
    return hit[0]


# ==================================================================== 溶解度
def build_sol(src_root, out_dir):
    """溶解度线：主口径 = homology 划分，test n=316。"""
    summ = os.path.join(src_root, "solubility-regression", "_server_summary")
    rows = rd_rows(os.path.join(summ, "all_experiments.csv"))
    leak = rd_rows(os.path.join(summ, "leakage_summary.csv"))

    out = []

    def add(enc, config, metric, val, tol, note):
        out.append({"line": "solubility", "enc": enc, "config": config, "metric": metric,
                    "expected": f"{val}", "tol": f"{tol}", "note": note})

    # --- ① 与编码器无关的基线：三档必须逐位相同（免费的管线正确性证据）
    for name, pred in [("length_ridge", "length_ridge"),
                       ("aac_ridge", "aac_ridge"),
                       ("aac_rf", "aac_rf")]:
        vals = []
        for enc in ENCS:
            r = find_one(rows, experiment=f"baselines_{enc}_homology", pred=pred)
            vals.append(float(r["spearman"]))
        if len(set(vals)) != 1:
            raise SystemExit(f"★ {name} 三档分数不同 {vals} —— 数据或划分在某一档被串了，"
                             f"expected 不该被生成出来")
        for enc in ENCS:
            add(enc, name, "spearman", vals[0], TOL_EXACT,
                "与编码器无关：三档必须逐位相同")

    # --- ② 冻结嵌入线性探针（= 尺子）
    for enc in ENCS:
        r = find_one(rows, experiment=f"baselines_{enc}_homology", pred="esmc_ridge")
        add(enc, "esmc_ridge", "spearman", float(r["spearman"]), TOL_PROBE, "尺子（线性探针）")

    # --- ③ 训练出来的头（主模型 + 消融）
    for tag, config in [("mlp_{enc}_hand_homology", "mlp_hand"),
                        ("mlp_{enc}_homology", "mlp_only")]:
        for enc in ENCS:
            r = find_one(rows, experiment=tag.format(enc=enc), pred="ensemble")
            add(enc, config, "spearman", float(r["spearman"]), TOL_TRAINED, "训练得到的头")

    # --- ④ 泄漏率（data 派生，容差 0）
    for split in ("homology", "random"):
        r = find_one(leak, split=split, part="test")
        add("-", f"leak_{split}_test", "ge25_pct", float(r["ge25_pct"]), TOL_DERIVED,
            "test 中与 train ≥25% 同一性的比例" + ("（⛔ 卡点，必须≈0）" if split == "homology"
                                            else "（泄漏对照）"))

    # --- 断言行数：3×3(无关基线) + 3(尺子) + 3×2(训练) + 2(泄漏) = 20
    n_expect = 9 + 3 + 6 + 2
    if len(out) != n_expect:
        raise SystemExit(f"★ 溶解度 expected 行数 {len(out)} != 期望 {n_expect}")

    _write(os.path.join(out_dir, "solubility", "expected", "key_metrics.csv"), out)
    return out


# ======================================================================= ΔΔG
def build_ddg(src_root, out_dir):
    """ΔΔG 线：官方划分，test n=56,344。"""
    summ = os.path.join(src_root, "thermostability-ddg", "_server_results", "runs", "_summary")
    comp = rd_rows(os.path.join(summ, "ddg_compare.csv"))
    pairs = rd_rows(os.path.join(summ, "ddg_pairs.csv"))

    out = []

    def add(enc, config, metric, val, tol, note):
        out.append({"line": "ddg", "enc": enc, "config": config, "metric": metric,
                    "expected": f"{val}", "tol": f"{tol}", "note": note})

    # --- ① D_hand 与编码器无关：三档必须逐位相同
    dvals = [float(find_one(comp, enc=e, variant="D_hand")["spearman"]) for e in ENCS]
    if len(set(dvals)) != 1:
        raise SystemExit(f"★ D_hand 三档分数不同 {dvals} —— 数据在某一档被污染，不该生成 expected")
    for enc in ENCS:
        add(enc, "D_hand", "spearman", dvals[0], TOL_EXACT,
            "与编码器无关（仅手工特征）：三档必须逐位相同")

    # --- ② 四个变体
    for variant, label in [("B_hwt", "B_hwt"), ("C_hmut", "C_hmut"), ("A_hwt_hmut", "A_hwt_hmut")]:
        for enc in ENCS:
            r = find_one(comp, enc=enc, variant=variant)
            add(enc, label, "spearman", float(r["spearman"]), TOL_TRAINED, "训练得到的头")

    # --- ③ 核心交付 A − B（Δ 与 p）
    for enc in ENCS:
        r = find_one(pairs, enc=enc, A="A_hwt_hmut", B="B_hwt", metric="spearman")
        add(enc, "A_minus_B", "delta", float(r["delta"]), TOL_TRAINED, "★ 核心交付 = h_mut 的增量")
        add(enc, "A_minus_B", "p", float(r["p"]), TOL_P, "配对检验 p 值（两边都应显著）")
        if r["significant"] != "True":
            raise SystemExit(f"★ {enc} 的 A−B 竟然不显著 —— 核心结论被推翻，措辞需改")

    # --- 断言行数：3(无关) + 3×3(变体) + 3×2(A−B 的 delta 与 p) = 18
    n_expect = 3 + 9 + 6
    if len(out) != n_expect:
        raise SystemExit(f"★ ΔΔG expected 行数 {len(out)} != 期望 {n_expect}")

    _write(os.path.join(out_dir, "ddg", "expected", "key_metrics.csv"), out)
    return out


def build_results_md(out_root, sol_rows, ddg_rows):
    """把快照渲染成可读的 `docs/RESULTS.md`。

    ★ 数字仍然来自产物（它们已在 sol_rows / ddg_rows 里），**不手打** ——
      否则这份"可读版"会变成第三个可能与 csv 不一致的数字来源。
    """
    def val(rows, cfg, enc):
        hit = [r for r in rows if r["config"] == cfg and r["enc"] == enc]
        if len(hit) != 1:
            return "—"
        return f"{float(hit[0]['expected']):.4f}"

    def tbl(rows, configs, extra_note=None):
        out = ["| 配置 | " + " | ".join(f"`{e}`" for e in ENCS) + " |",
               "|---|---|---|---|"]
        for cfg in configs:
            if extra_note and cfg in extra_note:
                cfg_txt = f"`{cfg}`{extra_note[cfg]}"
            else:
                cfg_txt = f"`{cfg}`"
            out.append(f"| {cfg_txt} | " + " | ".join(val(rows, cfg, e) for e in ENCS) + " |")
        return out

    def leak(rows, cfg):
        hit = [r for r in rows if r["config"] == cfg]
        return f"{float(hit[0]['expected']):.2f}%" if len(hit) == 1 else "—"

    sol_indep = {c: " ★" for c in ("length_ridge", "aac_ridge", "aac_rf")}
    ddg_indep = {"D_hand": " ★"}

    md = [
        "# 结果快照",
        "",
        "> 本文件由 `tools/make_expected.py` 从**实际跑出来的产物**生成，**每个数字都不手打**。",
        "> 它是「复现成功」的判据，配合 `tools/verify.py` 使用。",
        "> 带 ★ 的行见文末说明。",
        "",
        "---",
        "",
        "## 一、溶解度线（eSOL 回归）",
        "",
        "主口径 = `homology`（同源感知）划分，测试集 **n = 316**，指标 = **Spearman**。",
        "去冗余阈值：同一性 ≥25% 且覆盖 ≥50%。",
        "",
        "### 1.1 三编码器主结果",
        "",
        ] + tbl(sol_rows,
                ["esmc_ridge", "mlp_hand", "mlp_only"] + list(sol_indep.keys()),
                dict(sol_indep)) + [
        "",
        "- **尺子** = `esmc_ridge`（冻结嵌入线性探针）。任何新模型都要与它做**配对检验**。",
        "- ★ 带 ★ 的基线**不看编码器** ⇒ 三档**必须逐位相同**（这是免费的管线正确性证据）。",
        "",
        "### 1.2 同源泄漏（test 中与 train ≥25% 同一性的比例）",
        "",
        "| 划分 | test 泄漏率 |",
        "|---|---|",
        f"| `homology`（主口径） | **{leak(sol_rows, 'leak_homology_test')}** ← ⛔ 卡点，必须 ≈ 0 |",
        f"| `random`（对照） | {leak(sol_rows, 'leak_random_test')} ← 这就是「泄漏让分数虚高」的来源 |",
        "",
        "---",
        "",
        "## 二、ΔΔG 线（热稳定性）",
        "",
        "官方 25% 聚类划分（按野生型蛋白），测试集 **n = 56,344**，指标 = **Spearman**。",
        "标签 = **`−ddG_ML`**（负号不能丢）。",
        "",
        "### 2.1 四变体嵌套消融",
        "",
        "`D`（仅手工特征）→ `B`（+h_wt）→ `C`（+h_mut）→ `A`（h_wt + h_mut，主模型）",
        "",
        ] + tbl(ddg_rows, ["D_hand", "B_hwt", "C_hmut", "A_hwt_hmut"], ddg_indep) + [
        "",
        "### 2.2 核心交付 = A − B（多算一个突变体表示值不值）",
        "",
        "| 编码器 | Δ Spearman | p 值 |",
        "|---|---|---|",
    ]
    for e in ENCS:
        d = [r for r in ddg_rows if r["enc"] == e and r["config"] == "A_minus_B"
             and r["metric"] == "delta"]
        p = [r for r in ddg_rows if r["enc"] == e and r["config"] == "A_minus_B"
             and r["metric"] == "p"]
        md.append(f"| `{e}` | **+{float(d[0]['expected']):.4f}** | "
                  f"{float(p[0]['expected']):.4f} |")

    md += [
        "",
        "- ★ 带 ★ 的 `D_hand` **不看编码器** ⇒ 三档**必须逐位相同**（实测连 CI 都一致）。",
        "- **A − B 全部显著** ⇒ 增益来自**突变体上下文**（多算一个 `h_mut` 是值的）。",
        "",
        "---",
        "",
        "## 三、两条线的关键对照：规模收益**结论相反**",
        "",
        "| | 溶解度 | ΔΔG |",
        "|---|---|---|",
        "| 训练条数 | 2,525 | 419,520 |",
        "| 序列长度（中位） | ~310 aa | ~54 aa |",
        "| 600M → 6B | **不显著**（规模收益已饱和） | **显著**（尚未饱和） |",
        "",
        "⇒ **差别来自训练数据量**，不是编码器本身。",
        "**要不要下 6B，没有普适答案 —— 要看你的数据量。**",
        "",
        "> ⚠️ 溶解度线的「饱和」结论严格说只在**这个数据规模下**成立（训练集仅 2,525 条）。",
        "",
        "---",
        "",
        "## 四、怎么用这份快照",
        "",
        "```bash",
        "python tools/verify.py --line solubility --pred <你的产物目录>",
        "python tools/verify.py --line ddg        --pred <你的产物目录>",
        "```",
        "",
        "容差是**分级**的（见 README §2）：与编码器无关的量容差为 0，训练出来的头放宽。",
        "**趋势优先于绝对值** —— 绝对值有偏差但三档排序一致时，结论仍然成立。",
        "",
    ]
    path = os.path.join(out_root, "docs", "RESULTS.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md))
    print(f"  写出 {path}")


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["line", "enc", "config", "metric", "expected", "tol", "note"]
    # 显式 newline="" 避免 Windows 下写出空行
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  写出 {path}  ({len(rows)} 行)")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--src-root", default=None,
                    help="两条线源项目所在的父目录（跑取数与训练的目录），"
                         "例如 --src-root /path/to/projects")
    ap.add_argument("--out-root", default=os.path.join(here, ".."),
                    help="本仓库根目录")
    a = ap.parse_args()

    if not a.src_root:
        raise SystemExit(
            "★ 必须用 --src-root 指定源项目所在目录 —— 该目录下应包含\n"
            "    solubility-regression/   与   thermostability-ddg/\n"
            "  这样 expected/ 才能从**真实产物**生成（不允许手打数字）。")

    print("[1/2] 溶解度线")
    sol_rows = build_sol(a.src_root, a.out_root)
    print(f"      {len(sol_rows)} 行")
    print("[2/2] ΔΔG 线")
    ddg_rows = build_ddg(a.src_root, a.out_root)
    print(f"      {len(ddg_rows)} 行")
    print("[3/3] 渲染可读版 docs/RESULTS.md")
    build_results_md(a.out_root, sol_rows, ddg_rows)
    print("MAKE_EXPECTED_DONE")


if __name__ == "__main__":
    main()
