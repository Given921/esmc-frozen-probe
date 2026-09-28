"""轻量头部网络 —— 本项目的"轻量化"指的**就是这一块**。

设计
    输入：冻结的编码器向量（ESMC-600M 均值池化，1152 维）
          + 可选的手工生物物理特征（见 handcrafted.py）
      ↓
    Linear(d_in → 512) → LayerNorm → GELU → Dropout
    Linear(512 → 128)  → LayerNorm → GELU → Dropout
    Linear(128 → 1)    → 输出激活
      ↓
    输出：溶解度（默认 sigmoid → 落在 [0,1]，对应 solubility_frac）

参数量（d_in=1152 时）
    1152*512 + 512*128 + 128*1 + 偏置 ≈ 66 万
    → 远小于 500 万，CPU 上几分钟训完，可离线部署。
    编码器是 6 亿参数但**完全冻结**，不参与训练。

★ 输出激活怎么选（这是本任务的一个关键设计点）
    标签 solubility_frac ∈ [0,1] 是**有界**的，两种做法都合理：
      sigmoid + MSE  ：输出天然落在 [0,1]，不会越界；缺点是靠近 0/1 处梯度小
      linear  + MSE  ：收敛快，但会给到 <0 或 >1 的预测；评估时需 clip
    → 两种都跑，当**消融项**比。默认 sigmoid。

★ 归一化
    标准化统计量（均值/方差）**只能用训练集算**。用全量算 = 信息泄漏。
    这个模块接收已经标准化好的输入，标准化在 train.py 里按 split 分别做。
"""
import torch
import torch.nn as nn


class SolubilityHead(nn.Module):
    def __init__(self, d_in, hidden=(512, 128), dropout=0.2,
                 out_act="sigmoid", use_layernorm=True, out_dim=1):
        """out_dim=1  → 普通回归（一个数）
        out_dim=K  → 分位数回归（K 个分位点，见 forward 里的单调构造）
        """
        super().__init__()
        self.out_act = out_act
        self.out_dim = int(out_dim)
        layers = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, self.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)                              # (B, out_dim)
        if self.out_dim > 1:
            # ★ 分位点必须**单调不减**（P10 ≤ P50 ≤ P90），否则区间没意义。
            #   做法：第 1 个自由输出当基准，后面每个 = 前一个 + softplus(增量)，
            #   在实数轴上是单调不减的；再套 sigmoid（单调函数）→ [0,1] 内仍然单调不减。
            #   这样无需在损失里额外加"防止交叉"的惩罚项，训练更稳。
            base = z[:, :1]
            inc = torch.nn.functional.softplus(z[:, 1:])
            z = torch.cat([base, base + torch.cumsum(inc, dim=1)], dim=1)
        if self.out_act == "sigmoid":
            z = torch.sigmoid(z)
        # ★★ 单输出必须压成 (B,)。
        #    否则返回 (B,1)，而 nn.MSELoss()(out(B,1), target(B,)) 会**广播**成 (B,B)，
        #    损失照样下降、不报任何错 —— 是最隐蔽的一类静默 bug。
        return z.squeeze(-1) if self.out_dim == 1 else z

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class Normalizer:
    """按列标准化。用给定的 mu/sigma（**必须来自训练集**）。"""

    def __init__(self, mu, sigma):
        import numpy as np
        self.mu = np.asarray(mu, dtype="float32")
        self.sigma = np.asarray(sigma, dtype="float32")

    @classmethod
    def fit(cls, X):
        import numpy as np
        mu = X.mean(0)
        sigma = X.std(0)
        sigma[sigma < 1e-6] = 1.0
        return cls(mu, sigma)

    def transform(self, X):
        return (X - self.mu) / self.sigma

    @property
    def dim(self):
        return len(self.mu)
