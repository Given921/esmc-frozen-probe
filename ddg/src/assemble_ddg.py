"""热稳定性 ΔΔG · 把「序列向量」拼装成「每个突变的特征矩阵」（四个变体一起出）。

★ 为什么要拼四个变体，而不是只拼一个
    本项目要回答的核心问题不是"能跑多准"，而是**"多算 h_mut 到底值不值"**。
    要回答它，必须把"多算的那部分"单独变成一条消融：

      变体 D  floor   ：完全不用编码器。只用 [wt氨基酸 one-hot, mut氨基酸 one-hot, 相对位置, log长度]
                        → 回答"这个任务到底有多难"（氨基酸种类 + 位置就能解释多少）
      变体 B  h_wt    ：D + h_wt（**野生型**的序列向量）
                        → 一条野生型向量就能拿到多少？这是 ThermoMPNN 那类"只看野生型结构"
                          方法的序列版。★ 关键：h_wt 只需按**蛋白**跑一次（几百条），
                          成本几乎为零。
      变体 C  h_mut   ：D + h_mut（**突变体全长**的序列向量）
                        → 不给野生型上下文，光看突变体序列本身。
      变体 A  h_wt+h_mut：D + h_wt + h_mut
                        → ★ **核心交付：A − B 就是"多花几卡·天算 h_mut 买回了几个点"**。

    ★★ 硬约束：A 必须**恰好等于** B 再拼上 h_mut。
       所以这里四个变体共用同一段 hand 特征、同一个拼接顺序，
       绝不出现"A 少了一项、B 多了一项"这种偷偷换了两个变量的事
       （那样差值就什么都说明不了）。

★ 数据划分：不做任何重新划分。用的是 ProStab 官方的 25% 聚类划分（按野生型蛋白），
  并且 train/valid 已经按官方做法剔掉了 mmseq 命中的同源泄漏行。
  这里只按 split 列切，不做二次加工。

用法
    python src/assemble_ddg.py --enc esmc600m
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
VARIANTS = ["D_hand", "B_hwt", "C_hmut", "A_hwt_hmut"]
VARIANT_DESC = {
    "D_hand": "无编码器：氨基酸 one-hot + 位置",
    "B_hwt": "h_wt（野生型向量）+ 位置",
    "C_hmut": "h_mut（突变体向量）+ 位置",
    "A_hwt_hmut": "h_wt + h_mut + 位置  ★核心",
}


def onehot(aa_arr, dim=21):
    """氨基酸 → one-hot。★ 不认识的字符要**报错**，不能静默给全零行
    （全零行 = 一个合法但无意义的输入，模型照样能训，错误却不会被发现）。"""
    idx = np.array([ALPHABET.find(s) for s in aa_arr], dtype=np.int64)
    bad = idx < 0
    if bad.any():
        raise ValueError(f"出现了 21 字母表之外的残基：{sorted(set(np.asarray(aa_arr)[bad]))}")
    out = np.zeros((len(idx), dim), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="esmc600m")
    ap.add_argument("--feat-dir", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data_dir = a.data_dir or os.path.join(root, "data")
    fd = a.feat_dir or os.path.join(root, "features", a.enc)
    out = a.out or os.path.join(root, "assembled", a.enc)
    os.makedirs(out, exist_ok=True)

    mut = pd.read_csv(os.path.join(data_dir, "mut.csv"))
    order = pd.read_csv(os.path.join(fd, "seq_order.csv"))
    emb = np.load(os.path.join(fd, "emb.npy"), mmap_mode="r")
    man = json.load(open(os.path.join(fd, "manifest.json"), encoding="utf-8"))
    print("=" * 88)
    print(f"拼装 ΔΔG 特征  enc={a.enc}  向量维度 {emb.shape[1]}  嵌入条数 {emb.shape[0]:,}")
    print(f"  突变 {len(mut):,} 条   输出 {out}")
    print("=" * 88, flush=True)

    # ★ seq_id → 行号 的映射只能来自 seq_order.csv（提取脚本真正写盘的顺序），
    #   不能靠"seq_id 是顺序编号所以行号=编号"来假设 —— 那种假设一旦不成立，
    #   就是整张表错位（不报错，分数照样出，只是全错）。这里显式断言钉死。
    sid2row = {s: i for i, s in enumerate(order["seq_id"].astype(str))}
    assert len(sid2row) == emb.shape[0], "seq_order.csv 与 emb.npy 条数不一致"
    first_ids = sorted(sid2row, key=lambda s: sid2row[s])
    assert first_ids == sorted(sid2row), \
        "seq_order.csv 的行顺序与 seq_id 排序不一致 —— 提取脚本的排序假设不成立，停下来查"

    def rows_of(ids):
        idx = np.array([sid2row[str(x)] for x in ids], dtype=np.int64)
        return np.asarray(emb[idx], dtype=np.float32)

    # ---- 与编码器无关的那部分（所有变体共用，保证 A−B 只差 h_mut）----
    pos = mut["pos0"].to_numpy(np.int64)
    L = mut["seq_len"].to_numpy(np.float32)
    rel = np.where(L > 1, pos / np.maximum(L - 1, 1), 0.0).astype(np.float32)
    hand = np.concatenate([
        onehot(mut["wt_aa"].astype(str).to_numpy()),
        onehot(mut["mut_aa"].astype(str).to_numpy()),
        rel.reshape(-1, 1),
        np.log(L).reshape(-1, 1).astype(np.float32),
    ], axis=1).astype(np.float32)
    print(f"  hand 部分（不含编码器）{hand.shape[1]} 维：2×21 one-hot + 相对位置 + log长度")

    print("  正在取 h_wt 向量 …")
    X_wt = rows_of(mut["wt_seq_id"].astype(str).tolist())
    print(f"  正在取 h_mut 向量（{len(mut):,} 条）…")
    X_mut = rows_of(mut["mut_seq_id"].astype(str).tolist())

    mats = {
        "D_hand": hand,
        "B_hwt": np.concatenate([hand, X_wt], 1),
        "C_hmut": np.concatenate([hand, X_mut], 1),
        "A_hwt_hmut": np.concatenate([hand, X_wt, X_mut], 1),
    }
    # ★ 自检：A 必须恰好 = B + h_mut（列数上就能验，这里再验一次数值前缀）
    assert mats["A_hwt_hmut"].shape[1] == mats["B_hwt"].shape[1] + X_wt.shape[1]
    assert np.array_equal(mats["A_hwt_hmut"][:, :mats["B_hwt"].shape[1]],
                          mats["B_hwt"]), "A 的前缀不等于 B —— 变体构造不一致"
    for k, v in mats.items():
        assert v.shape[0] == len(mut), f"{k} 行数不符"
        assert np.isfinite(v).all(), f"{k} 含非有限值"
    print("  ✓ 变体自检通过：A 的前缀 == B，且 A 恰好比 B 多 h_mut 那一段")

    # ---- 按 split 落盘 ----
    summary = []
    for sp in ("train", "valid", "test"):
        m = (mut["split"] == sp).to_numpy()
        d = {k: v[m] for k, v in mats.items()}
        y = mut.loc[m, "label"].to_numpy(np.float32)
        uid = mut.loc[m, "uid"].astype(str).to_numpy()
        tmp = os.path.join(out, f"{sp}.tmp.npz")
        final = os.path.join(out, f"{sp}.npz")
        np.savez_compressed(tmp, y=y, uid=uid,
                            pos0=pos[m], wt_aa=mut.loc[m, "wt_aa"].astype(str).to_numpy(),
                            mut_aa=mut.loc[m, "mut_aa"].astype(str).to_numpy(),
                            wt_name=mut.loc[m, "wt_name"].astype(str).to_numpy(),
                            **{f"X_{k}": v for k, v in d.items()})
        os.replace(tmp, final)
        z = np.load(final)
        assert len(z["y"]) == int(m.sum()), f"{sp} 落盘行数不符"
        assert z["y"].std() > 0, f"{sp} 标签是常数（划分或取数出错了）"
        summary.append({"split": sp, "n": int(m.sum()), "n_prot": int(mut.loc[m, "wt_name"].nunique()),
                        "y_mean": round(float(y.mean()), 4), "y_std": round(float(y.std()), 4),
                        "dims": {k: int(v.shape[1]) for k, v in d.items()}})
        print(f"  [{sp:5s}] {int(m.sum()):>7,} 条  蛋白 {summary[-1]['n_prot']:>4} 个  "
              f"标签 {y.mean():+.3f}±{y.std():.3f}  "
              + "  ".join(f"{k}={v.shape[1]}" for k, v in d.items()))

    with open(os.path.join(out, "variants.json"), "w", encoding="utf-8") as fh:
        json.dump({"enc": a.enc, "variants": VARIANTS, "desc": VARIANT_DESC,
                   "alphabet": ALPHABET, "splits": summary,
                   "feature_layout": ["2×21 one-hot(wt,mu)", "rel_pos", "log_len"]
                                     + ["[h_wt 若该变体含]", "[h_mut 若该变体含]"],
                   "note": "hand 段在拼接里排最前，编码器向量在后；A−B 恰等于 h_mut 段"},
                  fh, indent=2, ensure_ascii=False)
    print(f"\n产物：{out}/{{train,valid,test}}.npz + variants.json")
    print("ASSEMBLE_DONE")


if __name__ == "__main__":
    main()
