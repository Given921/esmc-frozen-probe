# 第三方材料声明

> **本仓库的代码**采用 MIT 许可（见 `LICENSE`）。
> 但本仓库**引用的数据集与模型权重各有其原许可**，不受 MIT 影响。
> 本文件逐条列出，来源与获取方式的细节见 `DATA.md`。
>
> ★ 为什么 `LICENSE` 里不写这些：GitHub 靠匹配标准文本自动识别许可。
> 在标准 MIT 文本前后插入额外内容会导致识别失败（实测会显示 `NOASSERTION`）。
> 所以说明放在本文件，`LICENSE` 保持纯标准文本。

## 数据

| 对象 | 许可 | 能否再分发 | 本仓库处理 |
|---|---|---|---|
| **eSOL 溶解度数据** | `CC BY-SA 2.1 Japan` | 可以，但**是传染性许可**（需署名 + 以相同许可发布） | 只放获取脚本，**不打包本体**（避免整个仓库被拖进 CC BY-SA） |
| **UniProt 序列**（E. coli K-12，`UP000000625`） | `CC BY 4.0` | 可以（需署名） | 只放获取脚本 |
| **MegaScale**（ΔΔG 主表，Tsuboyama et al. 2023） | `CC BY 4.0`，`gated=False` | 可以（**需署名**） | 只放获取脚本 |
| **ProStab 派生文件** | ⚠️ **该仓库未提供 LICENSE 文件** | **默认保留所有权利 ⇒ 不应再分发** | **只给获取方式**，不打包派生文件 |

### 必须保留的署名

> eSOL © Hideki Taguchi (Tokyo Institute of Technology)
> licensed under CC Attribution-Share Alike 2.1 Japan

MegaScale 与 UniProt 的署名按各自原仓库要求执行（见 `DATA.md` 的引用一节）。

## 模型权重

| 模型 | 发布方 | 门禁 | HF 标注许可 |
|---|---|---|---|
| ESMC-300M / 600M / 6B | Chan Zuckerberg Biohub（`biohub/ESMC-*`） | 无 | `mit` + `other` |

**关于那个 `other`**：它通过 `license_link` 指向
`https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md`，
经查该文件列的是**第三方依赖库**（flash-attn / PyTorch / xformers / jaxtyping /
einops / omegaconf / attrs / scipy / lightning）的许可，
**不是模型权重本身的额外使用限制**。

> ⚠️ 模型权属与许可在历史上发生过变更（ESMC 由 EvolutionaryScale 转至 CZ Biohub，
> 许可亦随之变化）。**使用前请以 HuggingFace 页面当时标注的许可为准自行复核。**

## 本仓库不含数据与权重本体

原因见 `DATA.md`：许可限制、体积过大（6B 权重 23.66 GiB），
以及避免仓库随数据一起腐烂。本仓库提供的是**获取脚本 + 完整性校验**。

## 引用

如果本仓库的代码对你有帮助，请优先引用下列**数据与模型的原始文献** ——
它们才是结论的来源：

- 溶解度数据：Niwa et al. 2009, PNAS（eSOL）
- 序列数据：UniProt，E. coli K-12 参考蛋白组 `UP000000625`
- 热稳定性数据：Tsuboyama et al. 2023（MegaScale）
- ΔΔG 处理口径：ProStab
- 编码器：ESMC（ESM Cambrian），Chan Zuckerberg Biohub
