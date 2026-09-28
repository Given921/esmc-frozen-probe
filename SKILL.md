---
name: esmc-frozen-probe
description: 用 ESMC 冻结编码器 + 轻量头部复现蛋白性质预测（eSOL 溶解度回归 / 热稳定性 ΔΔG 回归）的配方。当需要「复现这套流程」「跑一遍溶解度或 ΔΔG 的模型」「验证某个复现结果对不对」「判断某个新表示 / 更大编码器在这两个任务上值不值」「审一个有泄漏嫌疑的蛋白性质模型」时使用。含：不可改口径清单（附理由）、产物级自检断言、带分级容差的结果比对、14 条静默失败坑，以及两条线结论相反（规模收益是否饱和）的原因。
---

# ESMC 冻结编码器 + 轻头 · 复现配方

## 这个 skill 解决什么

两条任务线，方法相同：**冻结一个蛋白语言模型当特征提取器，只训练一个小头部**。

| 线 | 任务 | 数据规模 | 主指标 | 关键产物 |
|---|---|---|---|---|
| `solubility/` | eSOL 溶解度回归 | 3,157 条 / test 316 | Spearman | `runs/_summary/*.csv` |
| `ddg/` | 热稳定性 ΔΔG | 529,740 条 / test 56,344 | Spearman | `runs/_summary/ddg_*.csv` |

**本配方最容易做错的不是代码，是口径。** 所以顺序是：先读口径 → 再冒烟 → 再全跑 → 最后带容差比对。

---

## 0. 开工前必做三件事（不要跳）

1. **读 `PROTOCOL.md`** —— 27 条不可改的口径，每条附理由。改任何一条都会**静默**让结果不可比。
2. **读 `PITFALLS.md`** —— 14 条静默失败。**判据只认产物，不认日志里的完成标记。**
3. **确认数据的条数对得上**（`DATA.md` §5）。对不上**先查取数，别往下走** ——
   后面所有分数都建立在这个前提上。

---

## 1. 两条线各自怎么跑

### 1.1 溶解度线

```bash
# 取数（有外网的机器；不需要 GPU）
python solubility/src/fetch_data.py          # → data/esol_reg.csv（3,157 条）

# GPU 机器
bash solubility/run_all.sh --smoke           # ① 先冒烟：--limit 200，2 分钟
bash solubility/run_all.sh                   # ② 正式：单张 A100 约 30 分钟
bash solubility/run_encoder_ablation.sh      # ③ 规模消融（300M；换 REPO/TAGNAME 跑 600M/6B）
```

**每阶段的预期产物与断言**：

| 阶段 | 产物 | 必须成立的断言 |
|---|---|---|
| 1 划分 | `data/splits/{random,homology}/` | — |
| 2 泄漏 | `runs/_summary/leakage_summary.csv` | ⛔ **homology 行 test 的 `ge25_pct` 必须 ≈ 0**（实测 0.00）；random 行 test ≈ 31.01 |
| 3 特征 | `features/<enc>_mean__<split>/` | 维度 = 该编码器的 `hidden_size`（300M/600M → 1152，6B → 2560）；无 NaN/Inf |
| 4 基线 | `runs/baselines_<enc>_<split>/` | `length_ridge` / `aac_ridge` / `aac_rf` **三档必须逐位相同** |
| 5 训练 | `runs/mlp_<enc>*_<split>/seed*/best.ckpt` | 3 个种子都有产物 |
| 6 汇总 | `runs/_summary/{all_experiments,paired_vs_ref}.csv` | 行数 = 配置数；配对表里显著条目数不为 0 |

### 1.2 ΔΔG 线

```bash
bash ddg/download_data.sh                    # 取 MegaScale + ProStab 派生文件
python ddg/src/build_dataset.py --inspect    # ★ 先看数据长什么样（不写文件）
python ddg/src/build_dataset.py              # → data/mut.csv + data/seqs.csv
# 把这两个 csv 传到 GPU 机器，然后：
bash ddg/run_ddg.sh                          # 600M，约 1 小时
bash ddg/run_ddg.sh esmc6b 16                # 6B（batch 调小）
```

**断言**：

| 阶段 | 断言 |
|---|---|
| 数据构建 | 漏斗 **1,384,137 → 529,740**；蛋白数 **298**；test **56,344** |
| 特征 | 唯一序列 **271,526** 条；维度 2560（6B） |
| 拼装 | ★ **`np.array_equal(A[:, :B.shape[1]], B)`** —— A 必须**恰好**等于 B 再拼 `h_mut` |
| 训练 | 四个变体都要有产物；**入口脚本会判退出码 + 日志标记，缺一即 `exit 1`** |
| 对照 | `D_hand` **三档逐位相同**（实测 0.4710，连 CI 一致） |

---

## 2. ★ 如何判断"复现成功"

**不要用"跑完没报错"判断。** 用带**分级容差**的产物比对：

```bash
python tools/verify.py --line solubility --pred <产物目录>
python tools/verify.py --line ddg        --pred <产物目录>
```

| 锚点类别 | 容差 | 判据 |
|---|---|---|
| **与编码器无关的量**（`length_ridge` / `aac_*`；ΔΔG 的 `D_hand`） | **0** | 三档必须**逐位相同** —— 不同就说明数据或划分串了 |
| 数据派生确定性量（泄漏率、条数、划分大小） | 0 | 纯计算，无随机性 |
| 冻结嵌入线性探针（`esmc_ridge` / `esmc_hand_ridge`） | ±0.002 | 特征确定，ridge 求解有数值差异 |
| 训练出来的 MLP | ±0.008 ~ ±0.010 | 依赖随机种子与 GPU 非确定性 |

### 判"复现失败"之前，先查两件事

1. **与编码器无关的基线是否逐位相同？** 不同 → **数据 / 划分问题，与模型无关**，
   别去调模型超参。
2. **三档之间的趋势是否一致？** 例如溶解度线应满足
   "6B 只显著优于 300M、与 600M **不**显著"。
   - 趋势对、绝对值在小范围 → **正常波动**，不算失败。
   - 趋势反 → 才是真问题。

---

## 3. 要改东西时怎么改（决策规则）

| 你想做的事 | 先看这里 | 注意 |
|---|---|---|
| 换个编码器 / 加个新模型 | 先跑 `run_encoder_ablation.sh` | 产物路径**不许**覆盖既有编码器的目录（脚本有红线断言） |
| 想声称"我的模型更好" | `PROTOCOL.md` 规则 9 | **必须和冻结嵌入线性探针（尺子）做配对检验**，不是和随机/长度基线比 |
| 想知道"更大模型值不值" | 两条线**结论相反** | 溶解度线：2,525 条训练数据 → **600M 处已饱和**；ΔΔG 线：419,520 条 → **未饱和**。差别在数据量，不是编码器 |
| 想知道"泄漏让分数虚高多少" | 跑 `--split random` 对照 | 泄漏率**必须用 BLAST 量**，k-mer 法不可用 |
| 判两个配置谁更好 | `PROTOCOL.md` 规则 11 | **只认配对检验**；不能用两个独立 95% CI 是否重叠来判断 |
| 改任何口径 | `PROTOCOL.md` 末尾"改口径的规矩" | 成组改 + 同步更新 `expected/` + commit message 写明"口径变更" |

---

## 4. 失败排查表

| 现象 | 先查 | 详见 |
|---|---|---|
| 脚本打印 `DONE` 但产物是空的 | **判据只认产物** —— 完成标记可能是假的 | `PITFALLS.md` #2 |
| 泄漏率明显不为 0（homology 划分） | 划分逻辑；是否误用了 k-mer 法 | `PROTOCOL.md` 规则 7 |
| 与编码器无关的基线三档分数不同 | 数据 / 划分被串了（**不是模型问题**） | `PROTOCOL.md` 规则 10 |
| 下载卡住 / 产出 0 字节文件 | 镜像 200 但 body 0；缺 `--speed-limit` | `PITFALLS.md` #4 |
| 权重加载报缺失 | 分片不全；是否下了 `esmc-300m-2024-12`（空 config 的 `.pth`） | `DATA.md` §3 |
| 传文件静默挂住 | paramiko channel 无超时 | `PITFALLS.md` #6 |
| 远端命令行为诡异、不报错 | Git Bash 改写了 `/home/...` 路径 | `PITFALLS.md` #5 |
| 关联两个 csv 结果完全错但不报错 | **不能用行号当键** | `PITFALLS.md` #9 |
| 报告里两张图一模一样 | 绘图变量名传错（两张图同 md5） | `PITFALLS.md` #11 |
| 删数据后条数对不上 | 删除范围：ΔΔG 只删 train/valid，**test 不删** | `PROTOCOL.md` 规则 16 |

---

## 5. 红线（违反会静默毁数据 / 得出错误结论）

1. **不要用 shell 重定向改既有文件**（`cat >>` / `echo >>` / `sed -i`）——
   会**从偏移 0 覆盖**且**文件大小不变**，`ls` 看不出。用 Write/Edit 工具。
2. **不要改 `PROTOCOL.md` 里的口径而不同步 `expected/`** —— 会让后来者以为复现失败。
3. **不要用 RMSE / MAE 当溶解度线的主指标** —— 实验噪声下界就有 37 个百分点（`PROTOCOL.md` 规则 5）。
4. **不要丢 ΔΔG 标签的负号**（`−ddG_ML`）—— 分数照样能算，但**结论全反**。
5. **不要在小数据上宣称"大模型没用"** —— 溶解度线的"饱和"结论严格说只在**该数据规模下**成立。
6. **不要拿大基线的分数去比小配置** —— 会凭空造出优势（本项目踩过）。
7. **不要把数据本体与权重本体提交进仓库** —— 见 `DATA.md` 的许可表。
