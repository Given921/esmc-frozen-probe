# DATA · 数据与权重的来源、许可与获取

> **本仓库不包含任何数据本体与权重本体。** 只包含获取方式与校验。
> 理由有两类：**许可**（有些不许再分发）与**体积**（6B 权重 23.66 GiB）。
>
> ★ 下表"能否再分发"是**依据许可文字的判断，不构成法律意见**。
> 如果你要商用或再分发，请自己复核原始许可。

---

## 1. 溶解度线

### 1.1 eSOL 溶解度标签

| 项 | 内容 |
|---|---|
| URL | `https://dbarchive.biosciencedbc.jp/data/esol/LATEST/esol.zip` |
| 体积 | 193 KB（`esol.csv` 4,132 行） |
| 许可 | **`CC BY-SA 2.1 Japan`** |
| 署名要求 | `eSOL © Hideki Taguchi (Tokyo Institute of Technology) licensed under CC Attribution-Share Alike 2.1 Japan` |
| 能否再分发 | ⚠️ **是传染性许可** —— 允许再分发，但**必须署名 + 以相同许可发布**。本仓库选择**不打包**，只给获取脚本，避免把整个仓库拖进 CC BY-SA |
| 原始文献 | Niwa et al. 2009, PNAS（E. coli 蛋白用 PURE 无细胞体系表达，溶解度 = 上清蛋白量 / 总蛋白量） |

> ⚠️ **该归档地址在部分网络环境不可达**（实测差异见 `PITFALLS.md` #4）。
> ⇒ 取数在**有外网的机器**上跑，产物 `esol_reg.csv` 再传给 GPU 机器。

### 1.2 UniProt 序列

| 项 | 内容 |
|---|---|
| URL | `https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000000625&format=tsv&fields=accession,gene_oln,gene_primary,gene_synonym,length,sequence` |
| 条数 | 4,403（E. coli K-12 参考蛋白组，reviewed） |
| 许可 | `CC BY 4.0` |
| 能否再分发 | ✅ 可以（署名即可）。但同样选择只给获取脚本 |

**为什么必须用它**：eSOL **只给基因名、不给序列**。要靠 UniProt 的
「位点名（b 号）/ 主基因名 / 别名」三种索引做映射。实测成功率 **3,167 / 3,173 = 99.8%**。

---

## 2. ΔΔG 线

### 2.1 MegaScale 主表

| 项 | 内容 |
|---|---|
| 仓库 | `RosettaCommons/MegaScale`（HF datasets） |
| 门禁 | **`gated = False`**（无需申请） |
| 许可 | **`cc-by-4.0`** |
| 文件 | `dataset2/data/train-00000-of-00002.parquet` 148,383,792 B<br>`dataset2/data/train-00001-of-00002.parquet` 143,104,796 B<br>`dataset3/data/train-00000-of-00001.parquet` 233,585,731 B |
| 能否再分发 | ✅ 可以（**必须署名**）。本仓库仍只给获取脚本 |
| 原始文献 | Tsuboyama et al. 2023（MegaScale） |

> ★ 官方代码里主表叫 `Tsuboyama2023_Dataset2_Dataset3_20230416.csv`，
> **但 CSV 本体不在 GitHub 上** —— 要从上面的 parquet 自行合并（见 `ddg/src/build_dataset.py`）。

### 2.2 ProStab 派生文件

| 项 | 内容 |
|---|---|
| 仓库 | GitHub `xtanh/ProStab` |
| 许可 | 🔴 **该仓库没有 LICENSE 文件**（实测 `LICENSE` / `LICENSE.md` / `LICENSE.txt` 全部 404） |
| 判定 | 无声明许可 ⇒ **默认保留所有权利** ⇒ **本仓库不放这些文件，只给获取方式** |
| 需要的文件 | `data/dataset/geostab_data/megascale.fasta` 24,783,344 B<br>`data/dataset/megascale/mega_splits.pkl` 16,046 B<br>`data/dataset/megascale/mmseq_mut_search_0.25.m8` 40,313,395 B<br>`data/dataset/S669/s669_clean_dir.csv`<br>`prostab/datamodules/datasets/megascale.py`（用来核对口径） |

> ★ **别整仓 clone** —— 该仓库里两千多个文件绝大多数是 PDB，clone 要几十 GB。
> 用 raw 直链按需取（见 `ddg/download_data.sh`）。
>
> ★ 路径**必须带 `data/dataset/` 两级前缀**。少了两级 → 四个文件全 404，
> **容易误判成"数据源被下架了"**（见 `PITFALLS.md` #12）。

### 2.3 口径以官方实现为准

`prostab/datamodules/datasets/megascale.py` 是**口径的最终依据**。
`ddg/src/build_dataset.py` 的文档字符串逐条对应它的做法。
**不要自己发明过滤规则** —— 口径一旦不同，后面所有分数都没法和已发表数字对话。

---

## 3. 编码器权重（ESMC）

| 模型 | HF 仓库 | 门禁 | HF 标注许可 | 体积 |
|---|---|---|---|---|
| ESMC-300M | `biohub/ESMC-300M` | 无 | `mit` + `other` | ~1.33 GB |
| ESMC-600M | `biohub/ESMC-600M` | 无 | `mit` + `other` | 2.2 GB |
| ESMC-6B | `biohub/ESMC-6B` | 无 | `mit` + `other` | **23.66 GiB（6 分片）** |

**许可说明**：HF 上标注 `mit` + `other`；`other` 指向
`https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md`，
该文件列的是**第三方依赖库**（flash-attn / PyTorch / xformers / jaxtyping / einops /
omegaconf / attrs / scipy / lightning）的许可，**不是模型权重本身的额外限制**。

> ★ 模型权重的许可在历史上变动过（ESMC 由 EvolutionaryScale 转到 **Chan Zuckerberg Biohub**）。
> **使用前请以 HF 页面当时的标注为准自行复核。**

### 获取

```bash
# 推荐：用 hf-mirror（HF 主站在部分网络不稳）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=<你的缓存目录>

# 例：拉 6B（6 个分片，合计 25,408,281,233 字节）
huggingface-cli download biohub/ESMC-6B --local-dir "$HF_HOME/ESMC-6B"
```

★ **别下 `esmc-300m-2024-12`** —— 那是空 config 的 `.pth` 版本，加载会失败。
用 `biohub/ESMC-300M`（HF 原生格式）。

### ★ 权重校验必须两级（见 `PITFALLS.md` #3）

```bash
# ① 结构级：解析 safetensors 头，核对 8 + 头长 + max(data_offsets) == 文件大小
# ② 全量 md5 两端对拍
```

只做 ① 会漏"大小正常、结构自洽、中间字节被改坏"那种 ——
它会产出**看着正常、实则错误**的特征，事后极难发现（本项目实测比对过 6 个分片的 md5）。

### 已实测的关键参数（6B）

| 项 | 值 |
|---|---|
| 分片数 × 合计字节 | 6 × 25,408,281,233 B = 23.66 GiB |
| 隐藏维 / 层数 / 头数 | 2560 / 80 / 40 |
| 参数量 | **63.52 亿**（加载实测 6345M） |
| 加载显存 | 25.84 GB |
| `trust_remote_code` | **不需要**（transformers ≥ 5.17 原生支持 ESMC） |
| 预期警告 | `lm_head.*` 报 UNEXPECTED **属正常** |

---

## 4. 全部下载都要带体积校验

这不是洁癖 —— 镜像会**间歇性返回 HTTP 200 但 body 是 0 字节**（见 `PITFALLS.md` #4）。
`ddg/download_data.sh` 里每条 `curl` 都写了**期望字节数**，大小不符即报 `[!!]` 并返回 1。

```bash
curl -L -C - --retry 3 --retry-delay 2 --max-time 3600 \
     --speed-limit 20480 --speed-time 30 \
     -o "$out" "$url"
```

★ **`--speed-limit` / `--speed-time` 不能省** —— 只靠 `--max-time` 会白等到超时。

## 5. 下载后的完整性期望（用于自检）

跑完两条线的取数后，下面这些数字应当对得上；对不上就**先查取数，别往下走**：

| 量 | 期望值 |
|---|---|
| `esol.csv` 原始行数 | 4,132 |
| 其中有溶解度标签 | 3,173 |
| 基因名 → UniProt 映射成功 | 3,167（99.8%） |
| 去重后（溶解度训练集） | **3,157** |
| ΔΔG 合并后 → 单点突变漏斗 | **1,384,137 → 529,740** |
| ΔΔG 蛋白数 | **298** |
| ΔΔG 待嵌入唯一序列 | 271,526 条（14.43 M 残基） |
| ΔΔG 划分 | train 419,520 / valid 53,876 / test **56,344**（28 蛋白） |
