#!/usr/bin/env bash
# =============================================================================
# 热稳定性 ΔΔG 项目 · 数据下载（在**有外网**的机器上跑；产物可再传到 GPU 机器）
#
# 来源
#   ① HF 镜像 `RosettaCommons/MegaScale`（Rocklin 实验室发布，CC-BY-4.0，无门禁）
#      主表按 dataset2 / dataset3 分目录放，都是 parquet
#   ② GitHub `xtanh/ProStab` 的派生文件（raw 直链，**别整仓 clone** —— 仓库里两千多个文件
#      绝大多数是 PDB，clone 要几十 GB）
#
# ★ 路径注意：ProStab 的派生文件带 `data/dataset/` 两级前缀。
#   我第一版笔记里漏了这两级，结果四个文件全 404，一度以为数据源没了。
#   教训：**404 的第一嫌疑是"路径记错了"，不是"文件被删了"**（体积对得上就更不能下结论）。
#
# ★ 每条 curl 都带停滞检测：
#   `--speed-limit 20480 --speed-time 30` = 30 秒内平均速度低于 20 KB/s 就断开重试。
#   hf-mirror 会**间歇性返回 HTTP 200 但 body 是 0 字节**，只靠 --max-time 会白等到天亮。
#   支持 range，所以 `-C -` 能逐字节精确续传。
# =============================================================================
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RAW="$ROOT/data/raw"
PS="$ROOT/data/prostab"
mkdir -p "$RAW" "$PS"

HF=https://hf-mirror.com/datasets/RosettaCommons/MegaScale/resolve/main
GH=https://raw.githubusercontent.com/xtanh/ProStab/main

dl() {  # dl <url> <本地路径> <期望字节数，0=不校验>
  local url="$1" out="$2" want="${3:-0}"
  if [ -f "$out" ] && [ "$want" != "0" ]; then
    local have; have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$have" = "$want" ]; then echo "[skip] 已完成 $(basename "$out") $have B"; return 0; fi
  fi
  for i in 1 2 3 4 5 6; do
    curl -L -C - --retry 3 --retry-delay 2 --max-time 3600 \
         --speed-limit 20480 --speed-time 30 \
         -o "$out" "$url" && break
    echo "  重试 $i ($(basename "$out"))"
    sleep 3
  done
  local have; have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$want" != "0" ] && [ "$have" != "$want" ]; then
    echo "[!!] 大小不符 $(basename "$out")：期望 $want 实得 $have"
    return 1
  fi
  echo "[ok] $(basename "$out") $have B"
}

echo "=== ① MegaScale 主表（parquet） ==="
# ★ 并行下载：hf-mirror 是**按连接限速**的（单连接实测 0.2~3.4 MB/s 波动，
#   3~4 路并行合计能到 7 MB/s 以上）。所以 4 个文件一起拉，别排队。
dl "$HF/dataset2/data/train-00000-of-00002.parquet" "$RAW/ds2_a.parquet"  148383792 &
dl "$HF/dataset2/data/train-00001-of-00002.parquet" "$RAW/ds2_b.parquet"  143104796 &
dl "$HF/dataset3/data/train-00000-of-00001.parquet" "$RAW/ds3.parquet"    233585731 &
dl "$HF/README.md"                                  "$RAW/MegaScale_README.md" 0 &
wait

echo
echo "=== ② ProStab 派生文件 ==="
dl "$GH/data/dataset/geostab_data/megascale.fasta"        "$PS/megascale.fasta"  24783344 &
dl "$GH/data/dataset/megascale/mega_splits.pkl"           "$PS/mega_splits.pkl"  16046 &
dl "$GH/data/dataset/megascale/mmseq_mut_search_0.25.m8"  "$PS/mmseq_mut_search_0.25.m8" 40313395 &
dl "$GH/data/dataset/S669/s669_clean_dir.csv"             "$PS/s669_clean_dir.csv" 0 &
dl "$GH/prostab/datamodules/datasets/megascale.py"        "$PS/prostab_megascale_dataset.py" 0 &
wait

echo
echo "=== 实际大小 ==="
ls -la "$RAW" "$PS"
echo "DL_ALL_DONE"
