#!/usr/bin/env bash
# =============================================================================
# 热稳定性 ΔΔG · 全链路（默认 600M；同一个脚本换参数就能跑 6B）
#
#   [1] 提特征：把「去重后的待嵌入序列」跑一遍编码器（h_wt 与 h_mut 一起提）
#   [2] 拼装：四个变体 D / B / C / A
#   [3] 训练：四个变体各 3 个种子（同一套超参，唯一变量是输入特征）
#   [4] 对照：配对 bootstrap，核心输出 = A − B（多算 h_mut 值不值）
#
# 用法（服务器）
#   bash run_ddg.sh                 # 600M（推荐先跑这条，约 1 小时）
#   bash run_ddg.sh esmc6b 16       # 6B，批调小（约 0.5~1 卡·天）
# 取消
#   tmux kill-session -t '=ddg'
#
# ★ 为什么默认只跑 600M
#   6B 的成本高一个量级，而**管线对不对**用 600M 几分钟就能验完。
#   先用便宜的规模把链路走通并交付一个完整结果，再决定要不要上 6B ——
#   反过来（先上 6B 再发现口径有问题）会浪费一整天。
# ★ 为什么用 --batch 参数化
#   序列很短（中位 ~56 aa），批越大越划算；但 6B 显存吃紧，必须能调小。
# =============================================================================
set -uo pipefail

PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$PROJ/.." && pwd)
# shellcheck source=../tools/lib.sh
. "$ROOT/tools/lib.sh"
load_config "$ROOT"                       # 不写死任何机器路径
cd "$PROJ" || exit 1

ENC="${1:-esmc600m}"
BATCH="${2:-64}"
GPU="${GPU:-0}"
PY=$(resolve_py "$VENV") || { echo "DDG_ABORT 见 tools/config.env.example"; exit 1; }

export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

mkdir -p logs runs assembled features

# ★ 断言：数据必须先由 build_dataset.py 生成好（它只读本地文件，不需要联网）。
for f in data/mut.csv data/seqs.csv; do
  require_nonempty "$f" "ΔΔG 输入数据" || { echo "DDG_ABORT 先跑 python src/build_dataset.py"; exit 1; }
done

# ★ 断言：权重完整性（两级校验，见 tools/lib.sh / PITFALLS.md #3）
case "$ENC" in
  esmc300m) WREPO=${ESMC_REPO_300M:-$HF_HOME/ESMC-300M} ;;
  esmc600m) WREPO=${ESMC_REPO_600M:-$HF_HOME/ESMC-600M} ;;
  esmc6b)   WREPO=${ESMC_REPO_6B:-$HF_HOME/ESMC-6B} ;;
  *) echo "DDG_ABORT 未知编码器 $ENC（可选 esmc300m / esmc600m / esmc6b）"; exit 1 ;;
esac
require_weights "$WREPO" || { echo "DDG_ABORT 权重不完整：$WREPO"; exit 1; }

echo "=== ΔΔG 全链路  enc=$ENC  batch=$BATCH  gpu=$GPU  start=$(date '+%F %T') ==="
$PY -c "
import pandas as pd
m = pd.read_csv('data/mut.csv'); s = pd.read_csv('data/seqs.csv')
print(f'  突变 {len(m):,} 条  待嵌入唯一序列 {len(s):,} 条  总残基 {int(s.seq_len.sum()):,}')
print('  各划分：' + '  '.join(f'{k}={v:,}' for k, v in m.split.value_counts().items()))
"

echo
echo "[1] 特征提取"
# ★ 已完成就跳过：整条链跑完一次后重跑时，没必要把 27 万条序列再算一遍。
#   而且 extract 脚本在"已完成"时走的是空循环分支，manifest 里的 ms/条 会算成 0，
#   把一份本来正确的交付产物写坏 —— 所以在**外层**跳过，比在里面省事且安全。
SKIP_EXTRACT=0
if [ -f "features/$ENC/state.json" ]; then
  SKIP_EXTRACT=$($PY -c "
import json
s = json.load(open('features/$ENC/state.json', encoding='utf-8'))
print(1 if (s.get('enc')=='$ENC' and s.get('done') and s.get('n')
            and s['done'] >= s['n']) else 0)
" 2>/dev/null || echo 0)
fi
if [ "$SKIP_EXTRACT" = "1" ]; then
  echo "  已有完整特征（features/$ENC/state.json 显示已完成），跳过"
else
  $PY src/extract_ddg_features.py --enc "$ENC" --batch "$BATCH" > "logs/extract_$ENC.log" 2>&1
  rc=$?; echo "  extract exit=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "DDG_ABORT 特征提取失败，见 logs/extract_$ENC.log"
    tail -25 "logs/extract_$ENC.log"
    exit 1
  fi
  tail -3 "logs/extract_$ENC.log"
fi

echo
echo "[2] 拼装四个变体"
$PY src/assemble_ddg.py --enc "$ENC" > "logs/assemble_$ENC.log" 2>&1
rc=$?; echo "  assemble exit=$rc"
if [ "$rc" -ne 0 ]; then
  echo "DDG_ABORT 拼装失败"; tail -25 "logs/assemble_$ENC.log"; exit 1
fi
tail -6 "logs/assemble_$ENC.log"

echo
echo "[3] 训练四个变体（各 3 种子）"
# 顺序：先跑便宜的（D 无编码器 → B 只有 h_wt），再跑贵的（C、A）。
# 这样万一时间不够，也已经有了"下限"和"h_wt 基线"，能说出部分结论。
# ★★ 每个变体都要记账，**不能失败还往下走**。
#    实测踩过：train_ddg.py 里一个 AttributeError 让四个变体在 1 秒内全崩，
#    而这个循环不判失败、compare 又只对存在的变体出结果，
#    于是脚本**照样打印 DDG_ALL_DONE** —— 完成标记成了假的，看门狗按"成功"返回，
#    整条链"30 分钟跑完"却一个模型都没训出来。完成标记必须是"真的全成功"才有意义。
TRAIN_FAIL=""
for V in D_hand B_hwt C_hmut A_hwt_hmut; do
  echo "  --- $V  $(date '+%T') ---"
  $PY src/train_ddg.py --enc "$ENC" --variant "$V" --seeds 3 --epochs 150 \
      --batch 256 > "logs/train_${ENC}_${V}.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] || ! grep -q "TRAIN_DDG_DONE" "logs/train_${ENC}_${V}.log"; then
    echo "  $V exit=$rc  ★ 失败"
    tail -12 "logs/train_${ENC}_${V}.log"
    TRAIN_FAIL="$TRAIN_FAIL $V"
  else
    echo "  $V exit=$rc  $(grep -m1 '集成后的 test 指标' -A6 "logs/train_${ENC}_${V}.log" | tr '\n' ' ')"
  fi
done
if [ -n "$TRAIN_FAIL" ]; then
  echo
  echo "DDG_ABORT 以下变体训练失败：$TRAIN_FAIL（详见 logs/train_${ENC}_*.log）"
  exit 1
fi

echo
echo "[4] 变体配对对照"
# ★★ 这一段原来有两个漏洞，合起来就是"假成功"：
#    ① 完全**没判 compare 的退出码**；
#    ② compare_ddg.py 的完成标记 COMPARE_DDG_DONE 是**无条件打印**的
#       （实测：logs/compare_esmc600m.log 里写着"一个变体都没跑，跳过"、
#        产物 0 行，末尾却照样有那行标记）。
#    ⇒ compare 崩了也照样往下走到 DDG_ALL_DONE，完成标记又成了假的。
#       这是本项目第三次踩"假标记"（前两次：训练段、同文件的 compare 段）。
#    现在加 --require-all（缺任何变体即以退出码 1 结束），并**同时**看退出码与标记。
$PY src/compare_ddg.py --encs "$ENC" --require-all > "logs/compare_$ENC.log" 2>&1
rc=$?
echo "  compare exit=$rc"
tail -25 "logs/compare_$ENC.log"
if [ "$rc" -ne 0 ] || ! grep -q "COMPARE_DDG_DONE" "logs/compare_$ENC.log"; then
  echo
  echo "DDG_ABORT 变体对照失败（见 logs/compare_$ENC.log）"
  exit 1
fi

echo
echo "end=$(date '+%F %T')"
echo "DDG_ALL_DONE $ENC"
