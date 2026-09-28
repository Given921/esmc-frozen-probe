"""把一份已算好的特征目录"重索引"到另一套划分上 —— 不重跑编码器。

★ 为什么可以这么做
  ESMC 是**冻结**编码器、逐条前向、没有随机性：同一条序列 → 永远同一个向量。
  而划分只是把**同一批 3157 条序列**重新分组（random / homology 都是它的一个划分）。
  ⇒ 特征只需要算一次；换划分时把"行"按序列名重新排列即可。

  这与 ESM-GraphSol 项目里的 `reindex_split.py` 是同一个套路：
  **换划分不重算嵌入**，省掉重复的 GPU 时间，同时避免"重算一遍数字对不上"的风险。

★ 安全性（不是"信任"，是"校验"）
  1. 目标划分的每一条序列都必须在源特征里找得到，否则直接报错退出；
  2. 逐条断言 **标签一致**（同一序列的溶解度标签必须相同）；
  3. 落盘走原子写（.tmp + os.replace），写完真 np.load 复检形状与标签。

用法：
    python reindex_features.py \
        --src features/esmc600m_mean__homology --src-split-dir ../data/splits/homology \
        --dst-split-dir ../data/splits/random --out features/esmc600m_mean__random
"""
import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd

SPLITS = ("train", "valid", "test")


def load_src(src_dir, src_split_dir):
    """返回 {序列: (向量, frac 标签, pct 标签)}，并做自洽性检查。

    ★★ 性能坑（实测踩过，务必别这么写）：npz 是**惰性**的，
        `z["X"][i]` 每次取一行都会把**整块** X 重新解压一遍 —— 循环 2525 行就是 O(n²)，
        实测单行 86 ms（10.8 MB 的 train.npz）→ 一轮下来 200 多秒，而且看起来像"卡死"。
        正确做法：先 `X = z["X"]` 取到内存里的 ndarray，再 X[i]（此时是 0 开销的切片）。
    """
    table = {}
    for sp in SPLITS:
        z = np.load(os.path.join(src_dir, f"{sp}.npz"))
        X = z["X"]                      # ★ 一次性读出（见上面的性能坑）
        Y = z["y"]
        P = z["y_pct"]
        df = pd.read_csv(os.path.join(src_split_dir, f"{sp}.csv"))
        assert len(df) == X.shape[0], f"{sp}: csv {len(df)} 行 vs npz {X.shape[0]} 行"
        for i, s in enumerate(df["sequence"].tolist()):
            y = float(Y[i])
            if s in table:
                assert np.isclose(table[s][1], y), "同一序列在源特征里标签不一致"
                continue
            table[s] = (X[i], y, float(P[i]))
        z.close()
    print(f"  源特征载入：{len(table)} 条唯一序列（来自 {src_dir}）")
    return table


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="已算好的特征目录（源）")
    ap.add_argument("--src-split-dir", required=True)
    ap.add_argument("--dst-split-dir", required=True, help="目标划分目录（只用来取分组）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src_dir = a.src if os.path.isabs(a.src) else os.path.join(root, a.src)
    src_split = a.src_split_dir if os.path.isabs(a.src_split_dir) else os.path.join(root, a.src_split_dir)
    dst_split = a.dst_split_dir if os.path.isabs(a.dst_split_dir) else os.path.join(root, a.dst_split_dir)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(root, a.out)

    if os.path.abspath(src_dir) == os.path.abspath(out_dir):
        raise SystemExit("源目录与输出目录相同 —— 不需要重索引")

    os.makedirs(out_dir, exist_ok=True)
    print("=" * 84)
    print("特征重索引（不重跑编码器）")
    print(f"  源   {src_dir}\n  目标 {out_dir}\n  划分 {dst_split}")
    print("=" * 84)

    table = load_src(src_dir, src_split)
    t0 = time.time()

    manifest = []
    for sp in SPLITS:
        df = pd.read_csv(os.path.join(dst_split, f"{sp}.csv"))
        missing = [s for s in df["sequence"] if s not in table]
        if missing:
            raise SystemExit(f"{sp}: 有 {len(missing)} 条序列在源特征里找不到，"
                             f"例如 {missing[0][:40]}… —— 源特征不完整")
        X = np.stack([table[s][0] for s in df["sequence"]]).astype(np.float32)
        y = df["solubility_frac"].to_numpy(np.float32)
        y_pct = df["solubility_pct"].to_numpy(np.float32)
        # 断言：从源特征带过来的标签与目标 csv 的标签一致
        y_src = np.array([table[s][1] for s in df["sequence"]], np.float32)
        assert np.allclose(y_src, y, atol=1e-5), f"{sp}: 标签对不上"

        tmp = os.path.join(out_dir, f"{sp}.tmp.npz")
        final = os.path.join(out_dir, f"{sp}.npz")
        np.savez_compressed(tmp, X=X, y=y, y_pct=y_pct,
                            gene=df["gene"].astype(str).to_numpy(),
                            seq_len=df["seq_len"].to_numpy(np.int32))
        os.replace(tmp, final)
        z = np.load(final)
        assert z["X"].shape == X.shape and np.allclose(z["y"], y), f"{sp} 复检失败"
        manifest.append(dict(split=sp, n=len(df), dim=X.shape[1],
                             pool="mean", dtype="reindex_from_" + os.path.basename(src_dir),
                             file=os.path.basename(final),
                             mean=float(np.mean(y)), std=float(np.std(y))))
        print(f"  [写出+复检OK] {sp}.npz  X={X.shape}  标签差异 0")

    pd.DataFrame(manifest).to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    for f in ("environment.txt",):
        p = os.path.join(src_dir, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out_dir, f))
    with open(os.path.join(out_dir, "REINDEX_NOTE.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"本目录的特征向量不是重新前向算的，而是从 {os.path.basename(src_dir)} 重索引而来。\n"
                 f"理由：ESMC 冻结编码器确定性输出，同序列 → 同向量；划分只改变分组。\n"
                 f"已校验：序列全覆盖、逐条标签一致、形状复检通过。\n")
    print(f"共 {time.time() - t0:.1f}s（重索引部分）")
    print("REINDEX_DONE")


if __name__ == "__main__":
    main()
