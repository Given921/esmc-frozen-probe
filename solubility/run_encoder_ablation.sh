#!/usr/bin/env bash
# =============================================================================
# 编码器规模消融 —— ESMC-300M / 600M / 6B
#
# 为什么要做这个
#   600M 那边的结论是「MLP 未显著超过线性探针（p=0.238）」，
#   说明分数几乎全部来自"预训练表示"，不来自头部设计。
#   那么换一个更小的编码器，这条规律还成立吗？
#     ① 300M ≈ 600M  → 规模不敏感 ⇒ 不值得花大代价下 6B
#     ② 300M << 600M → 规模确实有用 ⇒ 6B 值得下
#   这是"用 20 分钟换几小时"的决策实验。
#
# ★ 安全红线：本脚本绝不覆盖 600M 的任何产物
#      本脚本产物 → features/<TAGNAME>_mean__<split>、runs/*_<TAGNAME>_*
#      600M 产物  → features/esmc600m_mean__<split>、runs/*_esmc600m_*
#   阶段 0 有断言，TAGNAME/特征目录里若出现 600m 直接中止。
#
# ★ 公平对照：阶段 3 的头部配置（--hand / --epochs 300 / --out-act sigmoid
#   / --seeds 3）与 600M 主实验**逐字一致**，唯一变量就是编码器。
#   加载精度也刻意保持 F32（见 src/extract_features.py）——前两轮都是 F32，
#   为了可比性不能为省显存改成 bf16（计算仍走 bf16 autocast，不受影响）。
#
# 用法（服务器项目根目录下）：
#     bash run_encoder_ablation.sh            # 全跑（默认 300M）
#     bash run_encoder_ablation.sh --only 1   # 只跑第 1 阶段
#     bash run_encoder_ablation.sh --split random
#
# 跑 6B（分片权重，批大小调小）：
#     REPO="$ESMC_REPO_6B" TAGNAME=esmc6b BATCH=8 \
#       bash run_encoder_ablation.sh
# =============================================================================
set -euo pipefail

PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$PROJ/.." && pwd)
# shellcheck source=../tools/lib.sh
. "$ROOT/tools/lib.sh"
load_config "$ROOT"                       # 不写死任何机器路径
REPO=${REPO:-${ESMC_REPO_300M:-$HF_HOME/ESMC-300M}}
TAGNAME=${TAGNAME:-esmc300m}
SPLIT=homology
POOL=mean
SEEDS=${SEEDS:-3}
THREADS=${THREADS:-32}
# ★ 2560 维 / 80 层的 6B 用 32 会白吃显存；批大小不影响结论，只影响速度与显存
BATCH=${BATCH:-32}
# ★ 权重完整性下限（字节）。6B 分片合计约 23.66 GB，300M 约 1.33 GB，
#   统一按"所有权重文件之和"判，阈值取 1 GB 足以挡住半截文件。
WSZ_MIN=${WSZ_MIN:-1000000000}

PY=$(resolve_py "$VENV") || exit 1
export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)  ONLY="$2";  shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    *) echo "未知参数 $1"; exit 1 ;;
  esac
done
want() { [[ -n "$ONLY" ]] && [[ "$1" != "$ONLY" ]] && return 1; return 0; }
hdr()  { echo; echo "=================================================================="; \
         echo "  阶段 $1  $2"; \
         echo "=================================================================="; }

FEAT="features/${TAGNAME}_${POOL}__${SPLIT}"
cd "$PROJ"

# ---------------------------------------------------------------- 0 自检
if want 0; then
  hdr 0 "环境与安全自检"
  echo "项目     : $PROJ"
  echo "编码器   : $REPO"
  echo "特征目录 : $FEAT"
  [[ -x "$PY" ]] || { echo "找不到解释器 $PY"; exit 1; }
  [[ -f "$REPO/config.json" ]] || { echo "配置缺失: $REPO/config.json"; exit 1; }
  # ★ 权重可能是单文件，也可能是分片（6B = 6 个 model-0000X-of-0000Y.safetensors）。
  #   原来只判 model.safetensors，换 6B 会直接报"权重缺失"而中止。
  WSUM=0
  WN=0
  if [[ -f "$REPO/model.safetensors" ]]; then
    WSUM=$(stat -c%s "$REPO/model.safetensors"); WN=1
    WSINGLE="$REPO/model.safetensors"
  else
    for f in "$REPO"/model-*-of-*.safetensors; do
      [[ -f "$f" ]] || continue
      WSUM=$(( WSUM + $(stat -c%s "$f") )); WN=$(( WN + 1 ))
    done
    WSINGLE=""
  fi
  if [[ "$WN" -eq 0 ]]; then
    echo "权重缺失：$REPO 下既没有 model.safetensors，也没有 model-*-of-*.safetensors"; exit 1
  fi
  # ★ 只判"文件存在"是不够的：传输中的文件同样存在，只是大小不对。
  #   不加这道大小下限，就会拿半截权重去跑，报错还很难看出是权重的问题。
  if [[ "$WSUM" -lt "$WSZ_MIN" ]]; then
    echo "权重疑似不完整：合计 $WSUM 字节（应 >= $WSZ_MIN）—— 可能还在传输中"; exit 1
  fi
  echo "  权重     : $WN 个文件合计 $(numfmt --to=iec --suffix=B "$WSUM" 2>/dev/null || echo "$WSUM 字节")"
  # ★ 分片格式必须核对 index.json —— 缺一个分片时 transformers 的报错非常难懂
  if [[ -f "$REPO/model.safetensors.index.json" ]]; then
    "$PY" - "$REPO" <<'PYEOF'
import json, os, sys
repo = sys.argv[1]
idx = json.load(open(os.path.join(repo, "model.safetensors.index.json")))
need = sorted(set(idx["weight_map"].values()))
miss = [f for f in need if not os.path.exists(os.path.join(repo, f))]
print(f"  index.json: 需 {len(need)} 个分片，缺失 {len(miss)} 个")
if miss:
    print("  !! 缺失分片:", miss)
    sys.exit(1)
big = idx.get("metadata", {}).get("total_size")
if big:
    print(f"  权重张量总字节: {big/2**30:.2f} GiB")
PYEOF
  fi
  echo "  向量维度 : $("$PY" -c "import json;print(json.load(open('$REPO/config.json'))['hidden_size'])")"
  # ★ 红线：产物路径不许出现 600m
  case "$FEAT" in
    *600m*) echo "红线拦截：特征目录名含 600m，会覆盖既有产物，已中止"; exit 1 ;;
  esac
  case "$TAGNAME" in
    *600m*) echo "红线拦截：TAGNAME 含 600m，会覆盖既有实验，已中止"; exit 1 ;;
  esac
  echo "  ✅ 安全自检通过（不会触碰 600m 的任何产物）"
  "$PY" - <<'PYEOF'
import torch, transformers
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  tf {transformers.__version__}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU {torch.cuda.get_device_name(0)}  空闲 {free/1e9:.1f}/{total/1e9:.1f} GB")
PYEOF
fi

# ---------------------------------------------------------------- 1 特征
if want 1; then
  hdr 1 "特征提取（${TAGNAME}，冻结编码器）"
  "$PY" src/extract_features.py \
        --split-dir "data/splits/$SPLIT" \
        --out "$FEAT" \
        --repo "$REPO" --pool "$POOL" --batch "$BATCH"
fi

# ---------------------------------------------------------------- 2 基线
if want 2; then
  hdr 2 "基线体系（★ 含冻结嵌入线性探针 = 尺子）"
  "$PY" src/baselines.py --feat "$FEAT" --split-dir "data/splits/$SPLIT" \
        --tag "baselines_${TAGNAME}_${SPLIT}"
fi

# ---------------------------------------------------------------- 3 训练
if want 3; then
  hdr 3 "训练轻量头部（配置与 600M 严格一致，唯一变量=编码器）"
  "$PY" src/train.py --feat "$FEAT" --split-dir "data/splits/$SPLIT" \
        --tag "mlp_${TAGNAME}_hand_${SPLIT}" --hand \
        --seeds "$SEEDS" --epochs 300 --out-act sigmoid
  "$PY" src/train.py --feat "$FEAT" --split-dir "data/splits/$SPLIT" \
        --tag "mlp_${TAGNAME}_${SPLIT}" \
        --seeds "$SEEDS" --epochs 300 --out-act sigmoid
fi

# ---------------------------------------------------------------- 4 汇总
if want 4; then
  hdr 4 "汇总评估"
  "$PY" src/evaluate.py --runs runs --ref esmc_ridge
  echo
  echo "看 runs/_summary/all_experiments.csv，把 esmc300m 与 esmc600m 的同名配置并排比。"
fi

echo
echo "ABLATION_DONE"
