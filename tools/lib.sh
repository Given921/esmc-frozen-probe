#!/usr/bin/env bash
# =============================================================================
# 共享库：被 susceptibility/ 与 ddg/ 的入口脚本 source。
#
# 抽出来的理由：入口脚本有 3~4 个，而"加载配置 + 定位解释器 + 路径校验"这三件事
# 每份都要做。复制 4 份的话，改一处就会漏掉另外三处（本项目踩过同类问题）。
#
# 用法（在入口脚本开头）：
#     ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
#     . "$ROOT/tools/lib.sh"
#     load_config "$ROOT"
#     PY=$(resolve_py "$VENV") || exit 1
# =============================================================================

# ---------------------------------------------------------------- 加载配置
# 优先级：① 环境里已有的值  ② tools/config.env  ③ 本库给的默认值
# ★ 不写死任何机器路径 —— 换机器只需改 tools/config.env 这一个文件。
load_config() {
  local root="$1"
  if [ -f "$root/tools/config.env" ]; then
    # shellcheck disable=SC1091
    . "$root/tools/config.env"
  fi
  # ★ HF 缓存：没设就落到用户级默认目录（不放任何人的绝对路径）
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
}

# ---------------------------------------------------------------- 定位解释器
# ★★ 这是实测踩到的一个真 bug：
#   venv 里 python 的位置**跨平台不同** ——
#     Linux / macOS : <venv>/bin/python
#     Windows       : <venv>/Scripts/python.exe
#   原来写死 bin/python，在 Windows 上直接报 "No such file or directory"，
#   而且信息完全没提示"是路径风格问题"，很难查。
resolve_py() {
  local venv="$1"
  if [ -z "$venv" ]; then
    echo "❌ 未设置 VENV。先 cp tools/config.env.example tools/config.env 并填好路径，" >&2
    echo "   或用 VENV=/path/to/venv 前缀临时指定。" >&2
    return 1
  fi
  if   [ -x "$venv/bin/python" ];         then echo "$venv/bin/python"
  elif [ -x "$venv/Scripts/python.exe" ]; then echo "$venv/Scripts/python.exe"
  else
    echo "❌ $venv 下找不到解释器（试过 bin/python 与 Scripts/python.exe）。" >&2
    echo "   VENV 应指向虚拟环境的**根目录**。" >&2
    return 1
  fi
}

# ---------------------------------------------------------------- 断言：文件非空
# ★ 判据只认产物：**存在 ≠ 完整**（传输中的文件同样存在，只是大小不对）
require_nonempty() {
  local path="$1" what="${2:-产物}"
  if [ ! -e "$path" ]; then echo "★ 缺 $what：$path" >&2; return 1; fi
  if [ -f "$path" ] && [ ! -s "$path" ]; then
    echo "★ $what 是空文件：$path（可能还没传完，或上游静默失败）" >&2; return 1
  fi
  return 0
}

# ---------------------------------------------------------------- 断言：权重完整性
# 两级校验：① 分片齐全（读 index.json）② 大小下限（挡半截文件）
# 详见 PITFALLS.md #3
require_weights() {
  local repo="$1" min="${2:-1000000000}"
  "$PY" - "$repo" "$min" <<'PYEOF'
import glob, json, os, sys
repo, minb = sys.argv[1], int(sys.argv[2])
if not os.path.exists(os.path.join(repo, "config.json")):
    raise SystemExit(f"★ 权重目录缺 config.json：{repo}")
single = os.path.join(repo, "model.safetensors")
shards = sorted(glob.glob(os.path.join(repo, "model-*-of-*.safetensors")))
if os.path.exists(single):
    total, n = os.path.getsize(single), 1
elif shards:
    total, n = sum(os.path.getsize(f) for f in shards), len(shards)
else:
    raise SystemExit(f"★ 权重缺失：{repo} 下既无 model.safetensors，也无分片")
if total < minb:
    raise SystemExit(f"★ 权重疑似不完整：合计 {total:,} 字节（应 >= {minb:,}）"
                     f" —— 可能还在传输中")
idx = os.path.join(repo, "model.safetensors.index.json")
if os.path.exists(idx):
    need = sorted(set(json.load(open(idx, encoding="utf-8"))["weight_map"].values()))
    miss = [f for f in need if not os.path.exists(os.path.join(repo, f))]
    if miss:
        raise SystemExit(f"★ index.json 列了 {len(need)} 个分片，缺 {len(miss)} 个：{miss}")
    print(f"  分片自检通过：{len(need)} 个分片齐全")
print(f"  权重 {n} 个文件，合计 {total/2**30:.2f} GiB")
PYEOF
}

# ---------------------------------------------------------------- 断言：CSV 非空且有列
require_csv() {
  local path="$1" what="${2:-产物}" min_rows="${3:-1}"
  "$PY" - "$path" "$what" "$min_rows" <<'PYEOF'
import csv, io, os, sys
path, what, minrow = sys.argv[1], sys.argv[2], int(sys.argv[3])
if not os.path.exists(path):
    raise SystemExit(f"★ 缺 {what}：{path}")
rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
if len(rows) < minrow:
    raise SystemExit(f"★ {what} 只有 {len(rows)} 行（应 >= {minrow}）：{path}"
                     f"\n  —— 很可能上游静默失败了，注意别拿空产物继续往下算。")
print(f"  {what}: {len(rows)} 行  OK")
PYEOF
}

# ---------------------------------------------------------------- 小工具
hdr() { echo; echo "=================================================================="; \
        echo "  $1"; echo "=================================================================="; }

die() { echo "★ $*" >&2; exit 1; }
