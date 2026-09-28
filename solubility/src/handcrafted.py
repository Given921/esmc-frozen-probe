"""手工生物物理特征 —— 便宜、可解释，而且是很强的基线。

为什么要有它
  1. 文献里 eSOL 上的经典做法就是「氨基酸组成 + 分类/回归器」
     （Han et al. 2019 Bioinformatics，SVM + 氨基酸组成，R² = 0.4115）。
     所以这组特征本身就是一条**必须跑的基线**。
  2. 它是"模型打不过就说明没学到真东西"的下限。
  3. 加到编码器向量旁边，成本几乎为零，还能提升可解释性。

单个特征（共 20 + 12 = 32 维）
    ── 氨基酸组成 20 维：每种残基的占比
    ── 疏水性/电荷等汇总量：
        mean_kd      平均 Kyte-Doolittle 疏水指数
        frac_hydro   疏水残基(AVILMFWY)占比
        frac_charged 带电残基(DEKR)占比
        net_charge_pH7  近似净电荷 (K+R) - (D+E)，忽略 His
        frac_aromatic 芳香残基(FWY)占比
        frac_pro_G   脯氨酸+甘氨酸占比（柔性/无序相关）
        frac_cys     半胱氨酸占比（二硫键/聚集相关）
        frac_neg / frac_pos  酸性 / 碱性残基占比
        gravy        Kyte-Doolittle 平均亲水性（= mean_kd，冗余但便于对照）
        aliphatic_index  脂肪族指数
        disorder_prone   无序倾向残基(SQETKP)占比
    ── pI 需要外部工具，先不算（避免引入依赖），用净电荷代替。

注意：这里是**代理量**，不是严格的物理计算。目的是给模型一个便宜的物理先验，
并且让"氨基酸组成基线"这件事有一份可直接复用的实现。

用法：
    from handcrafted import handcrafted_features
    H = handcrafted_features(list_of_sequences)     # (N, 32) float32
"""
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle 疏水指数（1982）
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}

HYDROPHOBIC = set("AVILMFWY")
CHARGED = set("DEKR")
NEG = set("DE")
POS = set("KR")
AROMATIC = set("FWY")
FLEX = set("PG")
DISORDER_PRONE = set("SQETKP")

# 脂肪族指数用的三个残基
AI_AA = {"A": 100.0, "V": 100.0, "I": 100.0}
AI_COEF_ALANINE = 0.0    # 简化处理，见下

FEATURE_NAMES = ([f"frac_{a}" for a in AA] +
                 ["mean_kd", "frac_hydro", "frac_charged", "net_charge_pH7",
                  "frac_aromatic", "frac_pro_G", "frac_cys", "frac_neg",
                  "frac_pos", "gravy", "aliphatic_index", "disorder_prone"])


def _one(seq):
    L = max(len(seq), 1)
    s = seq.upper()
    cnt = {a: 0 for a in AA}
    other = 0
    for ch in s:
        if ch in cnt:
            cnt[ch] += 1
        else:
            other += 1
    frac = np.array([cnt[a] / L for a in AA], dtype=np.float32)

    mean_kd = float(np.mean([KD.get(ch, 0.0) for ch in s])) if s else 0.0
    gravy = float(sum(KD.get(ch, 0.0) for ch in s) / L)
    frac_hydro = sum(1 for ch in s if ch in HYDROPHOBIC) / L
    frac_charged = sum(1 for ch in s if ch in CHARGED) / L
    net = (cnt["K"] + cnt["R"] - cnt["D"] - cnt["E"]) / L
    frac_arom = sum(1 for ch in s if ch in AROMATIC) / L
    frac_flex = sum(1 for ch in s if ch in FLEX) / L
    frac_cys = cnt["C"] / L
    frac_neg = (cnt["D"] + cnt["E"]) / L
    frac_pos = (cnt["K"] + cnt["R"]) / L
    # 脂肪族指数 = 100*(A + 2.9*V + 3.9*(I+L)) / L （Ikai 1980）
    ai = 100.0 * (cnt["A"] + 2.9 * cnt["V"] + 3.9 * (cnt["I"] + cnt["L"])) / L
    disorder = sum(1 for ch in s if ch in DISORDER_PRONE) / L

    extra = np.array([mean_kd, frac_hydro, frac_charged, net, frac_arom,
                      frac_flex, frac_cys, frac_neg, frac_pos, gravy, ai,
                      disorder], dtype=np.float32)
    return np.concatenate([frac, extra])


def handcrafted_features(seqs):
    return np.stack([_one(s) for s in seqs]).astype(np.float32)


def n_features():
    return len(FEATURE_NAMES)
