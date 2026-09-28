#!/usr/bin/env bash
# =============================================================================
# 溶解度回归模型 —— 一键训练流水线
#
# 用法（在 GPU 机器上，项目根目录下）：
#     bash run_all.sh                 # 跑完整流程（推荐先看一遍下面的阶段说明）
#     bash run_all.sh --to 2          # 只跑到第 2 阶段
#     bash run_all.sh --only 3        # 只跑第 3 阶段
#     bash run_all.sh --smoke         # 冒烟：每步只用少量数据，2 分钟内跑通
#
# 阶段
#     0  环境检查（GPU / 依赖 / 权重）
#     1  数据划分（random + homology 两套）
#     2  泄漏量化（test/valid 与 train 的同源重叠）
#     ⛔ 卡点：homology 划分下 test 的 ≥25% 重叠必须 ≈ 0，否则不许往下走
#     3  特征提取（ESMC-600M，冻结；这是唯一较耗 GPU 的一步）
#     4  基线体系（6 条，含冻结嵌入线性探针）
#     5  训练轻量头部（多种子 + 验证集选 ckpt）
#     6  汇总评估（配对检验 + 泄漏代价）
#
# 预计耗时（3157 条序列，单张 A100）
#     阶段 1  BLAST all-vs-all  约 3~8 分钟（CPU 密集）
#     阶段 2  BLAST vs train    约 1~2 分钟
#     阶段 3  ESMC-600M 前向    约 10~15 分钟（批量 + bf16）
#     阶段 4/5/6                各 1~5 分钟（CPU，特征已冻结）
#     合计约 30 分钟，GPU 占用 ≈ 0.01 卡·天
# =============================================================================
set -euo pipefail

# ---------- 配置 ----------
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$PROJ/.." && pwd)
# shellcheck source=../tools/lib.sh
. "$ROOT/tools/lib.sh"
load_config "$ROOT"                       # 不写死任何机器路径：见 tools/config.env.example
SPLIT=homology                 # 主口径。想跑文献可比口径就改成 random
POOL=mean
SEEDS=${SEEDS:-3}
THREADS=${THREADS:-32}
# -----------------------------------------

PY=$(resolve_py "$VENV") || exit 1
ESMC_REPO=${ESMC_REPO:-${ESMC_REPO_600M:-$HF_HOME/ESMC-600M}}
export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

ONLY=""; TO=""; SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --to)   TO="$2";   shift 2 ;;
    --smoke) SMOKE=1;  shift ;;
    --split) SPLIT="$2"; shift 2 ;;
    *) echo "未知参数 $1"; exit 1 ;;
  esac
done

want() {                    # want <阶段号>
  [[ -n "$TO" ]]   && [[ "$1" -gt "$TO" ]]   && return 1
  [[ -n "$ONLY" ]] && [[ "$1" != "$ONLY" ]]  && return 1
  return 0
}
# hdr() 已由 tools/lib.sh 提供（带阶段号只在这里用，故这里覆盖一版）
hdr() { echo; echo "=================================================================="; \
        echo "  阶段 $1  $2"; echo "=================================================================="; }

LIMIT=""; SFX=""
# ⚠️ 注意别写成 "${SMOKE:+_smoke}"：SMOKE=0 也是"已设置且非空"，那个写法**永远**会加上 _smoke，
#    导致正式跑出来的实验名里也带 _smoke（会让报告的表格看起来像冒烟结果）。
[[ $SMOKE -eq 1 ]] && { LIMIT="--limit 200"; SFX="_smoke"; }

cd "$PROJ"

# ---------------------------------------------------------------- 0 环境
if want 0; then
  hdr 0 "环境检查"
  echo "项目目录   : $PROJ"
  echo "虚拟环境   : $VENV"
  echo "ESMC 权重  : $ESMC_REPO"
  echo "HF 端点    : $HF_ENDPOINT"
  echo "HF 缓存    : $HF_HOME"
  [[ -x "$PY" ]] || { echo "❌ 找不到解释器 $PY —— 见 tools/config.env.example"; exit 1; }
  "$PY" - <<'PYEOF'
import os, sys, shutil
import torch, numpy, pandas
print(f"  torch        {torch.__version__}  cuda可用={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU          {torch.cuda.get_device_name(0)}  "
          f"空闲显存 {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
try:
    import transformers; print(f"  transformers {transformers.__version__}")
except Exception as e: print(f"  transformers 缺失: {e}")
try:
    import sklearn; print(f"  scikit-learn {sklearn.__version__}")
except Exception:
    # 只有阶段 4（基线体系）用到 sklearn，它是在 baselines.py 里**延迟 import** 的。
    # 所以这里不能硬退出 —— 否则阶段 1/2/3 明明不需要它却被一起卡住。
    print("  scikit-learn 缺失（⚠️ 只影响阶段 4 的 Ridge / RandomForest 基线）")
    print("    有网时：pip install scikit-learn")
    print("    无网时：下 wheel → 上传 → pip install --no-index --find-links=<目录> scikit-learn")
for tool in ("makeblastdb", "blastp"):
    print(f"  {tool:12s} {shutil.which(tool) or '❌ 缺失（阶段1/2 需要 BLAST+）'}")
PYEOF
  # ★★ 权重必须**硬失败**，不能只打一句“不存在”就往下走。
  #    理由（PITFALLS.md #2）：本项目踩过"完成标记是假的"——崩了照样打印 ALL_DONE，
  #    看门狗按成功返回，整条链"跑完"却一个模型都没训出来。
  #    环境检查不通过就应该以非 0 退出码结束，让调用方**必须**看见。
  if ! require_weights "$ESMC_REPO" ; then
    echo
    echo "❌ 环境检查未通过：ESMC 权重不可用（$ESMC_REPO）"
    echo "   下载方式见 DATA.md §3，例如："
    echo "     HF_ENDPOINT=$HF_ENDPOINT huggingface-cli download biohub/ESMC-600M \\"
    echo "         --local-dir \"$ESMC_REPO\""
    exit 1
  fi
  echo "  ✅ 环境检查通过"
fi

# ---------------------------------------------------------------- 1 划分
if want 1; then
  hdr 1 "数据划分（random + homology）"
  "$PY" src/make_splits.py --data data/esol_reg.csv \
        --out-root data/splits --work data/blast_work \
        --id 25 --cov 0.5 --threads "$THREADS" --seed 42
fi

# ---------------------------------------------------------------- 2 泄漏
if want 2; then
  hdr 2 "泄漏量化"
  # ★★ 卡点必须是**硬断言**，不能只是打印一句话。
  #    理由：这是整条链的前提 —— homology 划分若还有同源重叠，后面所有分数都不必看。
  #    只打印的话，人一忙就跳过去了，然后拿着虚高的分数做结论。
  LEAK_CSV="runs/_summary/leakage_summary.csv"
  "$PY" src/check_leakage.py --splits-root data/splits \
        --work data/blast_work --threads "$THREADS"
  "$PY" - "$LEAK_CSV" <<'PYEOF'
import csv, io, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    raise SystemExit(f"★ 卡点：泄漏汇总 {p} 不存在")
rows = list(csv.DictReader(io.open(p, encoding="utf-8")))
hit = [r for r in rows if r["split"] == "homology" and r["part"] == "test"]
if len(hit) != 1:
    raise SystemExit(f"★ 卡点：{p} 里 homology/test 应恰好 1 行，实得 {len(hit)}")
ge25 = float(hit[0]["ge25_pct"])
print(f"  homology/test 与 train ≥25% 同一性的比例 = {ge25:.2f}%")
if ge25 > 1.0:
    raise SystemExit(
        f"★ 卡点未通过：homology 划分下 test 泄漏还有 {ge25:.2f}%（应 ≈ 0）。\n"
        f"  不要继续 —— 先回头查划分逻辑。这个前提破了，后面所有分数都不必看。")
print("  ✅ 卡点通过（homology 划分干净）")
PYEOF
fi

# ---------------------------------------------------------------- 3 特征
if want 3; then
  hdr 3 "特征提取（ESMC-600M，冻结编码器）"
  FEAT="features/esmc600m_${POOL}__${SPLIT}"
  # ★ 权重完整性两级校验（见 tools/lib.sh 的 require_weights / PITFALLS.md #3）
  require_weights "$ESMC_REPO" || exit 1
  "$PY" src/extract_features.py \
        --split-dir "data/splits/$SPLIT" \
        --out "$FEAT" \
        --repo "$ESMC_REPO" --pool "$POOL" --batch 32 $LIMIT
  # ★ 产物级自检：断言维度与数值健全（NaN/Inf 会让后面所有指标失去意义）
  "$PY" - "$FEAT" <<'PYEOF'
import glob, os, sys
import numpy as np
feat = sys.argv[1]
files = sorted(glob.glob(os.path.join(feat, "*.npz")))
if not files:
    raise SystemExit(f"★ 特征产物为空：{feat} 下没有 .npz（上游可能静默失败）")
bad = []
for f in files:
    z = np.load(f)
    for k in z.files:
        a = z[k]
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            bad.append(f"{os.path.basename(f)}:{k}")
if bad:
    raise SystemExit(f"★ 特征里含 NaN/Inf：{bad[:5]}（共 {len(bad)} 处）")
print(f"  特征自检通过：{len(files)} 个文件，无 NaN/Inf")
PYEOF
fi

# ---------------------------------------------------------------- 4 基线
if want 4; then
  hdr 4 "基线体系"
  FEAT="features/esmc600m_${POOL}__${SPLIT}"
  "$PY" src/baselines.py --feat "$FEAT" --split-dir "data/splits/$SPLIT" \
        --tag "baselines_esmc600m_${SPLIT}${SFX}"
fi

# ---------------------------------------------------------------- 5 训练
if want 5; then
  hdr 5 "训练轻量头部（多种子 + 验证集选 ckpt）"
  FEAT="features/esmc600m_${POOL}__${SPLIT}"
  # 主实验：编码器向量 + 手工特征
  "$PY" src/train.py --feat "$FEAT" --split-dir "data/splits/$SPLIT" \
        --tag "mlp_esmc600m_hand_${SPLIT}${SFX}" --hand \
        --seeds "$SEEDS" --epochs 300 --out-act sigmoid
  # 消融：只用编码器向量，不加手工特征
  "$PY" src/train.py --feat "$FEAT" --split-dir "data/splits/$SPLIT" \
        --tag "mlp_esmc600m_${SPLIT}${SFX}" \
        --seeds "$SEEDS" --epochs 300 --out-act sigmoid
  # 消融：输出层改成线性（对照 sigmoid）
  "$PY" src/train.py --feat "$FEAT" --split-dir "data/splits/$SPLIT" \
        --tag "mlp_esmc600m_hand_linear_${SPLIT}${SFX}" --hand \
        --seeds "$SEEDS" --epochs 300 --out-act linear
fi

# ---------------------------------------------------------------- 6 汇总
if want 6; then
  hdr 6 "汇总评估（配对检验 + 泄漏代价）"
  "$PY" src/evaluate.py --runs runs --ref esmc_ridge
  # ★ 产物级自检：**判据只认产物**，不认日志里的完成标记（见 PITFALLS.md #2）。
  "$PY" - <<'PYEOF'
import csv, io, os, sys
def rows(p):
    if not os.path.exists(p):
        raise SystemExit(f"★ 汇总产物缺失：{p}")
    r = list(csv.DictReader(io.open(p, encoding="utf-8")))
    if not r:
        raise SystemExit(f"★ 汇总产物为空：{p}（可能一个实验都没跑成）")
    return r

ae = rows("runs/_summary/all_experiments.csv")
pv = rows("runs/_summary/paired_vs_ref.csv")
print(f"  all_experiments.csv  {len(ae)} 行")
print(f"  paired_vs_ref.csv    {len(pv)} 行")

# 尺子必须在场（没有它就无法做配对检验，本仓库的核心口径）
if not any(r.get("pred") == "esmc_ridge" for r in ae):
    raise SystemExit("★ 总表里没有 esmc_ridge（尺子）—— 后续无法做配对检验，口径不成立")
# 配对表必须真的判过显著/不显著
if not any(str(r.get("significant", "")).lower() in ("true", "false") for r in pv):
    raise SystemExit("★ 配对表里没有 significant 列的有效值 —— 配对检验没真正跑")
print("  ✅ 汇总产物自检通过")
PYEOF
fi

echo
echo "=================================================================="
# ★★ 完成标记必须"真的代表成功"（PITFALLS.md #2）。
#    原来这里**无条件**打印"全部完成 / ALL_DONE" —— 于是 `--only 0`（只做环境检查）
#    也会打印 ALL_DONE；环境检查失败时也一样（只要没走到 exit）。
#    看门狗/下游拿它当判据就会一路假通过。改成：只有**默认全套跑完**才算 ALL_DONE。
if [ -z "$ONLY" ] && { [ -z "$TO" ] || [ "$TO" -ge 6 ]; }; then
  echo "  全部完成（阶段 0~6）"
  echo "=================================================================="
  echo "看结果："
  echo "  runs/_summary/all_experiments.csv   总表（含 CI）"
  echo "  runs/_summary/paired_vs_ref.csv     配对检验"
  echo "  runs/_summary/leakage_cost.csv      泄漏代价（两套划分的差）"
  echo "  runs/<tag>/seed0/best.ckpt          选出来的最佳权重"
  echo
  echo "★ 与期望快照比对（推荐每次跑完都做）："
  echo "    $PY \"$ROOT/tools/verify.py\" --line solubility \\"
  echo "        --pred \"$PROJ\" --summary-dir runs/_summary"
  echo
  echo "★ 想量化「同源泄漏让分数虚高多少」，再跑一遍 random 划分后对比："
  echo "    bash run_all.sh --split random"
  echo "    $PY src/evaluate.py"
  echo
  echo "ALL_DONE"
else
  echo "  指定阶段已完成（ONLY=${ONLY:-无} TO=${TO:-无}）"
  echo "=================================================================="
  echo
  echo "PARTIAL_DONE  ★ 这不是 ALL_DONE —— 只跑了指定阶段，别当成全套成功"
fi
