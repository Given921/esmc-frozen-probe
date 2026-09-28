"""第 3 步：用 ESMC-600M 提蛋白质级特征向量。

★ 路线（服务器已实测确认，2026-09-20）
  transformers **5.x 已原生内置 esmc** → 直接走标准 HF 接口：
      tok   = AutoTokenizer.from_pretrained(<本地权重目录>)
      model = AutoModel.from_pretrained(<本地权重目录>, dtype=torch.float32)
  · AutoModel 返回 last_hidden_state（首选）
  · 若只能用 AutoModelForMaskedLM，则 MaskedLMOutput **没有** last_hidden_state，
    要传 output_hidden_states=True 再取 hidden_states[-1]（这是个坑，已兼容处理）
  · tokenizer 会自动加 <cls> 开头、<eos> 结尾 → 取残基嵌入时必须剥掉首尾各一个
  · 权重目录必须是 HF 原生格式的 biohub/ESMC-600M（不是 esmc-600m-2024-12，
    后者 config.json 是空 {}）

★ "冻结编码器 + 轻头部"的算力账
  3157 条序列，600M 模型：串行 fp32 实测 893 ms/条 → 约 47 分钟；
  批量(32) + bf16 后约 10~15 分钟。**训练阶段 GPU 占用 = 0**（特征算一次存盘）。
  这是"轻量化"的含义：可训练参数只有头部那几百万，编码器完全冻结。

★ 池化方式
  蛋白质级任务要把「每残基向量」压成一个向量。三种都实现，可用 --pool 切：
    mean（默认，ESM-C 推荐）  /  cls（用 <cls> 位置的向量）  /  max（逐维取最大）
  池化方式是**消融项**，不要只报一种。

★ 落盘
  原子写（先写 .tmp 再 os.replace），写完真 np.load 复检形状与条数。

用法：
    python extract_features.py --split-dir ../data/splits/homology --out ../features/esmc600m
    python extract_features.py --split-dir ../data/splits/homology --out ../features/esmc600m_cls --pool cls
"""
import argparse
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# ★ 不写死缓存路径：优先用环境变量 HF_HOME，否则落到用户级默认目录。
#   （原实现在这里硬编码了某台机器的缓存路径 —— 换机器会静默用错目录，且会泄露主机信息。）
os.environ["HF_HOME"] = os.environ.get("HF_HOME") or os.path.expanduser(
    "~/.cache/huggingface")

import numpy as np
import pandas as pd
import torch

EMB_DIM_600M = 1152
MAX_LEN = 2046          # ESM-C max_position_embeddings=2048，留首尾两个特殊符


def load_esmc(repo, device="cuda"):
    """返回 (model, tokenizer, mode)。"""
    import transformers
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    print(f"[环境] transformers {transformers.__version__} | 权重 {repo}", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    model, mode = None, None
    try:
        model = AutoModel.from_pretrained(repo, dtype=torch.float32)
        mode = "base"
    except Exception as e:                                     # noqa: BLE001
        print(f"[提示] AutoModel 不可用（{type(e).__name__}），改用 AutoModelForMaskedLM")
        model = AutoModelForMaskedLM.from_pretrained(repo, dtype=torch.float32)
        mode = "mlm"
    model = model.eval().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[模型] 载入完成 mode={mode} 参数 {n_par/1e6:.0f}M "
          f"显存 {torch.cuda.memory_allocated()/1e9:.2f} GB" if device == "cuda"
          else f"[模型] 载入完成 mode={mode} 参数 {n_par/1e6:.0f}M")
    return model, tok, mode


def unwrap_hidden(out, model, mode):
    """拿到 last_hidden_state。"""
    v = getattr(out, "last_hidden_state", None)
    if v is not None:
        return v
    if mode == "mlm":
        hs = getattr(out, "hidden_states", None)
        if hs is None:
            raise RuntimeError("MaskedLMOutput 没有 hidden_states —— "
                               "需要对模型传 output_hidden_states=True")
        return hs[-1]
    raise RuntimeError("拿不到 last_hidden_state")


@torch.no_grad()
def pool_batch(model, tok, seqs, pool, device, amp_dtype):
    """一批序列 → (B, D) 蛋白质级向量。"""
    enc = tok(list(seqs), return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_LEN + 2)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda" and amp_dtype is not None)):
        if getattr(model, "config", None) is not None:
            try:
                out = model(**enc, output_hidden_states=(True if pool == "residue_cls" else False))
            except TypeError:
                out = model(**enc)
        else:
            out = model(**enc)
    if isinstance(out, tuple):
        out = out[0]
    h = unwrap_hidden(out, model, "base" if hasattr(out, "last_hidden_state") else "mlm")
    h = h.float()
    mask = enc["attention_mask"].unsqueeze(-1)              # (B, L, 1)
    n_special = 2                                            # <cls> ... <eos>
    lengths = enc["attention_mask"].sum(1)
    keep = mask.clone()
    # 去掉首尾特殊符
    ar = torch.arange(keep.shape[1], device=device).unsqueeze(0)
    keep = keep * ((ar >= 1) & (ar < (lengths.unsqueeze(1) - 1))).unsqueeze(-1)
    if pool == "mean":
        v = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
    elif pool == "cls":
        v = h[:, 0, :]
    elif pool == "max":
        v = (h.masked_fill(keep == 0, -1e9)).max(1).values
    else:
        raise ValueError(f"未知池化 {pool}")
    return v, int(lengths.max())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default=os.path.join(here, "..", "data", "splits", "homology"))
    ap.add_argument("--out", default=None, help="默认 <repo>/features/esmc600m_<pool>")
    ap.add_argument("--repo", default=os.environ.get("ESMC_REPO_600M")
                    or os.path.join(os.environ["HF_HOME"], "ESMC-600M"),
                    help="编码器权重目录（含 config.json + 权重）。"
                         "默认取 $ESMC_REPO_600M，否则 $HF_HOME/ESMC-600M")
    ap.add_argument("--pool", choices=["mean", "cls", "max"], default="mean")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true", help="关掉 bf16（默认开，实测快 6 倍）")
    ap.add_argument("--limit", type=int, default=0, help="只提前 N 条（冒烟用）")
    ap.add_argument("--tag", default=None, help="给产物加后缀，便于多套实验并存")
    a = ap.parse_args()

    root = os.path.abspath(os.path.join(here, ".."))
    out_dir = a.out or os.path.join(root, "features", f"esmc600m_{a.pool}"
                                     + (f"_{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)

    amp_dtype = None if a.fp32 else torch.bfloat16
    dtype_name = "fp32" if amp_dtype is None else "bf16"
    print("=" * 84)
    # ★ 这里必须打印**实际**权重目录名。写死成 "ESMC-600M" 时，
    #   换 300M/6B 跑出来的日志会冒充 600M，多编码器对比时极易看错。
    print(f"ESMC 特征提取 [{os.path.basename(os.path.normpath(a.repo))}]   "
          f"池化={a.pool}  精度={dtype_name}  批={a.batch}")
    print(f"  输入 {a.split_dir}\n  输出 {out_dir}")
    print("=" * 84)

    model, tok, mode = load_esmc(a.repo, a.device)

    # ---- 先把三个 split 的序列全读进来，按序列去重（同序列只算一次）----
    frames = {}
    for sp in ("train", "valid", "test"):
        p = os.path.join(a.split_dir, f"{sp}.csv")
        df = pd.read_csv(p)
        if a.limit:
            df = df.head(a.limit).copy()
        frames[sp] = df
        print(f"  [{sp}] {len(df)} 条")

    uniq = list(dict.fromkeys(
        s for df in frames.values() for s in df["sequence"].tolist()))
    print(f"  去重后唯一序列 {len(uniq)} 条（省算力）")

    # ---- 批量前向 ----
    emb = {}
    t0 = time.time()
    done = 0
    for i in range(0, len(uniq), a.batch):
        chunk = uniq[i:i + a.batch]
        v, maxlen = pool_batch(model, tok, chunk, a.pool, a.device, amp_dtype)
        v = v.cpu().numpy().astype(np.float32)
        for s, row in zip(chunk, v):
            emb[s] = row
        done += len(chunk)
        if done % (a.batch * 10) == 0 or done == len(uniq):
            el = time.time() - t0
            print(f"    {done}/{len(uniq)}  {el:.0f}s  "
                  f"({el/done*1000:.0f} ms/条, 本批最长 {maxlen})  "
                  f"按此速度全部需 {el/done*len(uniq)/60:.1f} 分钟", flush=True)

    # ---- 落盘（原子写）----
    manifest = []
    for sp, df in frames.items():
        X = np.stack([emb[s] for s in df["sequence"]])
        y = df["solubility_frac"].to_numpy(np.float32)
        y_pct = df["solubility_pct"].to_numpy(np.float32)
        tmp = os.path.join(out_dir, f"{sp}.tmp.npz")
        final = os.path.join(out_dir, f"{sp}.npz")
        np.savez_compressed(tmp, X=X, y=y, y_pct=y_pct,
                            gene=df["gene"].astype(str).to_numpy(),
                            seq_len=df["seq_len"].to_numpy(np.int32))
        os.replace(tmp, final)
        # 真读一遍复检（不能只看文件在不在）
        z = np.load(final)
        assert z["X"].shape == (len(df), X.shape[1]), f"{sp} 形状不符"
        assert np.allclose(z["y"], y), f"{sp} 标签不符"
        manifest.append(dict(split=sp, n=len(df), dim=X.shape[1],
                             pool=a.pool, dtype=dtype_name,
                             file=os.path.basename(final),
                             # ★ 注意：mean/std 是**标签 y** 的统计量，不是特征的。
                             #   列名容易误读（与 dim/pool/dtype 并列时几乎必然被读成特征统计量）。
                             #   实测踩过一次：300M 与 600M 的这两个数完全相同，我一度怀疑
                             #   "两个编码器算出了同一批特征"。实际上同一划分下 y 不变，
                             #   所以这两列**本来就该完全相同**，相同才是正常的。
                             #   （真要核对特征是否不同，比的是 X，不是这里。）
                             mean=float(np.mean(y)), std=float(np.std(y))))
        print(f"  [写出+复检OK] {sp}.npz  X={X.shape}  y={y.shape}")

    pd.DataFrame(manifest).to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    with open(os.path.join(out_dir, "environment.txt"), "w") as fh:
        import transformers
        fh.write(f"torch=={torch.__version__}\n")
        fh.write(f"transformers=={transformers.__version__}\n")
        fh.write(f"numpy=={np.__version__}\n")
        fh.write(f"encoder={a.repo}\npool={a.pool}\nprecision={dtype_name}\n")
    print(f"\n共 {time.time() - t0:.0f}s。产物在 {out_dir}")
    print("EXTRACT_DONE")


if __name__ == "__main__":
    main()
