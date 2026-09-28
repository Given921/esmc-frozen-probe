"""热稳定性 ΔΔG · 用冻结的 ESMC 编码器给"待嵌入序列表"提向量（可断点续传）。

★ 和溶解度那条线的三个区别（都是这个任务特有的）
  1. **要提的序列不是"数据集的样本"，而是"去重后的序列"**。
     27 万条突变里，野生型只对应几百条唯一序列（同一蛋白的全部突变复用同一份 h_wt）；
     突变体序列才是大头。所以输入是 `data/seqs.csv`（seq_id, seq），
     输出是"seq_id → 向量"的一张表，再由 assemble 脚本按 mutation 拼装。
  2. **序列很短**（MegaScale 中位 ~56 aa，而溶解度线中位 ~310 aa）。
     短序列时**单条调度开销占比变大**，"按残基数外推时间"会严重低估 → 日志按**条数**
     报速度和 ETA，不按残基。
  3. **必须能续传**。27 万条不是几分钟的事，任何一次断线/被抢占都不该从头再来。
     做法：预分配一个 `.npy` 内存映射文件，逐批写入，并把"已完成到第几条"存进
     `state.json`。重跑时接着写。**按 seq_id 排序处理**，所以"已完成条数"就是断点。

★ GPU 显存：6B 全精度载入约 26 GB（A100 80GB 够）。批太小会浪费，太大在长序列上会 OOM；
  序列短（≤ 几百 aa），批 64 通常没问题，但要能调。

用法
    python src/extract_ddg_features.py --enc esmc600m
    python src/extract_ddg_features.py --enc esmc600m --limit 64      # 冒烟
    python src/extract_ddg_features.py --enc esmc6b --batch 16
"""
import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# ★ 不写死缓存路径：优先用环境变量 HF_HOME，否则落到用户级默认目录。
os.environ["HF_HOME"] = os.environ.get("HF_HOME") or os.path.expanduser(
    "~/.cache/huggingface")

import numpy as np
import pandas as pd
import torch

# ★ 权重目录从环境变量取（ESMC_REPO_300M / _600M / _6B），
#   没设就落到 $HF_HOME/<模型名>。**不硬编码任何机器上的绝对路径**。
_HF = os.environ["HF_HOME"]
REPO_OF = {"esmc300m": os.environ.get("ESMC_REPO_300M") or os.path.join(_HF, "ESMC-300M"),
           "esmc600m": os.environ.get("ESMC_REPO_600M") or os.path.join(_HF, "ESMC-600M"),
           "esmc6b":   os.environ.get("ESMC_REPO_6B")   or os.path.join(_HF, "ESMC-6B")}
DIM_HINT = {"esmc300m": 960, "esmc600m": 1152, "esmc6b": 2560}
MAX_LEN = 2046


def load_esmc(repo, device="cuda"):
    import transformers
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
    print(f"[环境] transformers {transformers.__version__} | 权重 {repo}", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    try:
        model, mode = AutoModel.from_pretrained(repo, dtype=torch.float32), "base"
    except Exception as e:                                     # noqa: BLE001
        print(f"[提示] AutoModel 不可用（{type(e).__name__}），改用 AutoModelForMaskedLM")
        model = AutoModelForMaskedLM.from_pretrained(repo, dtype=torch.float32)
        mode = "mlm"
    model = model.eval().to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[模型] mode={mode} 参数 {n/1e9:.3f}B", flush=True)
    return model, tok, mode


@torch.no_grad()
def pool_batch(model, tok, seqs, device, amp_dtype):
    """一批序列 → (B, D) 均值池化向量（剥掉 <cls>/<eos>）。"""
    enc = tok(list(seqs), return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_LEN + 2)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.autocast("cuda", dtype=amp_dtype,
                        enabled=(device == "cuda" and amp_dtype is not None)):
        out = model(**enc)
    if isinstance(out, tuple):
        out = out[0]
    h = getattr(out, "last_hidden_state", None)
    if h is None:
        hs = getattr(out, "hidden_states", None)
        if hs is None:
            raise RuntimeError("拿不到 last_hidden_state / hidden_states")
        h = hs[-1]
    h = h.float()
    mask = enc["attention_mask"].unsqueeze(-1)
    lengths = enc["attention_mask"].sum(1)
    ar = torch.arange(mask.shape[1], device=device).unsqueeze(0)
    keep = mask * ((ar >= 1) & (ar < (lengths.unsqueeze(1) - 1))).unsqueeze(-1)
    v = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
    return v, int(lengths.max())


def write_progress(fp, rec):
    """给看板用的进度文件（追加一行 JSON）。

    ★ 追加而不是覆盖：看板只需要最后一行，但保留历史就能事后算"哪一段变慢了"。
      用 buffering=1 保证每行立刻落盘（看板 20 秒轮询一次，缓冲住就白轮询）。
    """
    try:
        with open(fp, "a", encoding="utf-8", buffering=1) as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:                                          # noqa: BLE001
        pass


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="esmc600m", choices=list(REPO_OF))
    ap.add_argument("--repo", default=None)
    ap.add_argument("--seqs", default=os.path.join(root, "data", "seqs.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    repo = a.repo or REPO_OF[a.enc]
    out = a.out or os.path.join(root, "features", a.enc)
    os.makedirs(out, exist_ok=True)
    prog_fp = os.path.join(root, "progress.jsonl")

    df = pd.read_csv(a.seqs)
    df = df.sort_values("seq_id").reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit).copy()
    N = len(df)
    seqs = df["seq"].tolist()
    amp = None if a.fp32 else torch.bfloat16
    print("=" * 88)
    print(f"ΔΔG 特征提取  enc={a.enc}  pool=mean  精度={'fp32' if a.fp32 else 'bf16'}  批={a.batch}")
    print(f"  待嵌入 {N:,} 条序列（总残基 {int(df['seq_len'].sum()):,}，"
          f"中位长度 {int(df['seq_len'].median())}）")
    print(f"  输出 {out}")
    print("=" * 88, flush=True)

    model, tok, mode = load_esmc(repo, a.device)

    # ---- 探测维度（用第一条），然后预分配可续写的 npy ----
    probe, _ = pool_batch(model, tok, seqs[:1], a.device, amp)
    D = int(probe.shape[1])
    del probe
    if DIM_HINT[a.enc] != D:
        print(f"  ★ 注意：实测维度 {D} 与预期 {DIM_HINT[a.enc]} 不同，以实测为准")
    emb_path = os.path.join(out, "emb.npy")
    state_path = os.path.join(out, "state.json")
    start = 0
    if os.path.exists(emb_path) and os.path.exists(state_path):
        try:
            st = json.load(open(state_path, encoding="utf-8"))
            # ★ 续传前必须核对形状和"这次的数据"是不是同一批：
            #   换了 seqs.csv（条数或顺序变了）却接着旧的写，会得到一张
            #   行号与 seq_id 错位的表 —— 不报错，但下游全错。
            if st.get("n") == N and st.get("dim") == D and st.get("enc") == a.enc:
                start = int(st.get("done", 0))
                print(f"  ★ 发现已有进度，从第 {start:,} 条续写")
            else:
                print(f"  ★ 已有 emb.npy 与本次不匹配"
                      f"（旧 n={st.get('n')} dim={st.get('dim')} enc={st.get('enc')}），"
                      f"从头重算")
        except Exception:                                      # noqa: BLE001
            pass

    mm = np.lib.format.open_memmap(emb_path, mode=("r+" if start else "w+"),
                                   dtype=np.float32, shape=(N, D))
    t0 = time.time()
    done = start
    maxlen_seen = 0
    for i in range(start, N, a.batch):
        chunk = seqs[i:i + a.batch]
        v, mx = pool_batch(model, tok, chunk, a.device, amp)
        mm[i:i + len(chunk)] = v.cpu().numpy().astype(np.float32)
        done = i + len(chunk)
        maxlen_seen = max(maxlen_seen, mx)
        if (done // a.batch) % 5 == 0 or done >= N:
            mm.flush()
            el = time.time() - t0
            rate = (done - start) / max(el, 1e-9)          # 条/秒
            # ★ 单位：rate 是"条/秒"，要得到"毫秒/条"必须 1000/rate。
            #   第一版写成 rate*1000，看板上直接显示成 144378 ms/条（真值 6.9），差 1000 倍。
            #   同一份 manifest.json 里的那处算的是 seconds/seq*1000，是对的 ——
            #   同一个文件里两处口径不一致，正是这种错最容易漏掉的地方。
            ms_per = 1000.0 / max(rate, 1e-9)
            eta = (N - done) / max(rate, 1e-9)
            print(f"    {done:,}/{N:,}  {el:.0f}s  ({ms_per:.2f} ms/条, "
                  f"本批最长 {mx})  剩余约 {eta/60:.1f} 分钟", flush=True)
            write_progress(prog_fp, {
                "updated": time.time(), "enc": a.enc,
                "task": f"ΔΔG 特征提取 · {a.enc}（{D} 维）",
                "stage": "h_mut/h_wt 向量", "done": done, "total": N,
                "unit": "条", "pct": round(done / N * 100, 1),
                "eta_sec": round(eta), "ms_per_seq": round(ms_per, 2),
                "notes": f"精度 {'fp32' if a.fp32 else 'bf16'} · 批 {a.batch}"})
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump({"enc": a.enc, "n": N, "dim": D, "done": done,
                           "repo": repo, "batch": a.batch,
                           "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                          fh, ensure_ascii=False, indent=2)

    mm.flush()
    del mm
    df[["seq_id"]].to_csv(os.path.join(out, "seq_order.csv"), index=False)
    man = {"enc": a.enc, "repo": repo, "dim": D, "n": N, "batch": a.batch,
           "precision": "fp32" if a.fp32 else "bf16", "pool": "mean",
           "total_residues": int(df["seq_len"].sum()),
           "elapsed_sec": round(time.time() - t0),
           "ms_per_seq": round((time.time() - t0) / max(N - start, 1) * 1000, 2)}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)

    # ---- 复检：真读一遍 + 查坏值 ----
    z = np.load(emb_path, mmap_mode="r")
    assert z.shape == (N, D), f"形状不符 {z.shape} != {(N, D)}"
    sample = np.asarray(z[:: max(1, N // 500)])                  # 抽 ~500 行
    n_bad = int(np.sum(~np.isfinite(sample)))
    assert n_bad == 0, f"抽样里出现 {n_bad} 个非有限值，特征有问题"
    norm = float(np.linalg.norm(sample, axis=1).mean())
    print(f"\n复检 OK：形状 {z.shape}  抽样范数均值 {norm:.2f}  用时 {man['elapsed_sec']}s")
    write_progress(prog_fp, {"updated": time.time(), "enc": a.enc,
                             "task": f"ΔΔG 特征提取 · {a.enc} 完成",
                             "stage": "done", "done": N, "total": N, "unit": "条",
                             "pct": 100.0, "eta_sec": 0,
                             "notes": f"{man['ms_per_seq']} ms/条，{D} 维"})
    print("EXTRACT_DDG_DONE")


if __name__ == "__main__":
    main()
