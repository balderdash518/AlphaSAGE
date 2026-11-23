# AlphaSAGE 复现：与作者设定的偏离

论文：AlphaSAGE（ICLR 2026，arXiv 2509.25055v3）。代码：上游 `BerkinChen/AlphaSAGE`，偏离以 commit `517467a` 为基准。
目的：在统一协议下得到可横向比较的 baseline。论文写明的设定照用；论文没写的，用仓库默认值；没有为结果调过任何超参数。

## 复现命令

```bash
# 挖掘：instrument ∈ {csi300, csi500, csi1000}，seed ∈ {0, 1, 2}
.venv\Scripts\python.exe train_gfn.py --instrument csi300 --seed 0 --snapshot cn_2026.11b --split C --pool_capacity 50 --n_episodes 10000 --log_freq 500 --update_freq 64 --encoder_type gnn --entropy_coef 0.01 --entropy_temperature 1.0 --mask_dropout_prob 1.0 --ssl_weight 1.0 --nov_weight 0.3 --weight_decay_type linear --final_weight_ratio 0.0

# 论文原生评测（动态组合）
.venv\Scripts\python.exe run_adaptive_combination.py --expressions_file <run_dir>/pool_9999.json --instruments csi300 --seed 0 --snapshot cn_2026.11b --split C --threshold_ric 0.015 --threshold_ricir 0.15 --chunk_size 400 --window inf --n_factors 20 --cuda 0
```

需要设置 `ALPHADATA_ROOT`（本机为 `E:\yifei\Alphadata`）。每次运行在 `registry`（默认 `$ALPHASAGE_REGISTRY`，没有设置时为 `out/registry.csv`）里追加一行。

## 偏离清单

| # | 项目 | 原来 | 现在 | 原因 |
|---|---|---|---|---|
| 1 | 数据源 | `src/data_collection/fetch_baostock_data.py` 从 Baostock 抓取：前复权（`adjustflag=2`），再按抓取当天的因子归一；股票池文件从本地 Qlib 数据复制 | AlphaData 冻结快照 `cn_2026.11b` 的 `qlib-last` 导出（前复权，锚定快照最后一天 2026-10-23），用 `--snapshot` 选择 | 统一协议：所有方法读同一份冻结快照，运行中不联网下载。复权方式和原脚本相同，只是锚点日期从作者的抓取日变成了快照最后一天。两种做法下，训练期价格都包含之后的除权信息，原实现也是这样 |
| 2 | 时间划分 | 论文（A 股）：训练 2010–2020 / 验证 2021 / 测试 2022–2024。代码 `train_gfn.py` 写死训练 2010-01-01–2016-12-31（即论文的美股划分）；`run_adaptive_combination.py` 按 `--train_end_year` 逐年推算 | 快照 `splits.yaml` 的 C 划分：训练 2010-01-01–2020-12-31 / 验证 2021-01-01–2022-06-30 / 测试 2022-07-01–2025-06-30，两个脚本都通过 `--split C` 读取 | 统一协议。训练期与论文的 A 股设定一致；验证期和测试期按统一协议 |
| 3 | 挖掘时的监控窗口 | `GFNLogger` 每隔 `log_freq` 用 2018-01-01–2020-12-31 的数据计算因子池 IC，只写进 TensorBoard 和控制台，不参与训练 | 不变。在 C 划分下这段时间属于训练期；加了断言，窗口超出训练期时直接报错 | 保持论文逻辑，同时保证挖掘阶段不读验证期和测试期 |
| 4 | 标签跨越边界 | `StockData` 会读到结束日之后 30 个交易日，用来算 20 日远期收益标签 | 不变：训练期最后约 20 个交易日的标签会用到 2021 年 1 月的价格 | 原实现如此；AlphaForge 的复现也照原样处理 |
| 5 | CSI1000 早期没有成分股 | 论文写训练从 2010 年开始，但 CSI1000 指数 2014-10 才发布，原始数据和本快照的成分股都从 2014-10-31 开始 | 按原样跑 2010-01-01 起。代码会自动跳过没有成分股的日子，再扣掉 100 天的回看，有效训练期从 2015-03-31 开始 | 数据本身如此，原实现行为相同 |
| 6 | 种子 | 主表（表 1）是单次运行；附录 E.7 在 CSI300 上用 5 个种子做稳健性检验 | 每个股票池 3 个种子（0、1、2），报均值和标准差 | 统一协议：所有方法用同一组种子 |
| 7 | Episode 数 | 论文表 4 写"10000/20000"，没说明哪个数据集用哪个；README（CSI300）用 10000，`run.sh`（S&P500）用 20000 | A 股统一用 10000 | 以 README 为准 |
| 8 | 组合时选用的因子数 `n_factors` | 论文没写；README 20，`eval.sh` 50，argparse 默认 10 | 20 | README 是作者写明的复现命令 |
| 9 | `--cuda` | README 写 2 | 0 | 本机只有一块 GPU；这个参数会被写进 `CUDA_VISIBLE_DEVICES` |
| 10 | 运行环境 | Linux，Python 3.11，torch 2.4.0+cu121（`pyproject.toml` 写死 Linux 安装包），含 tensorflow | Windows 11，Python 3.11，torch 2.7.1+cu128，torchgfn 1.2.1，torch-geometric 2.8.1，torchtyping 0.1.5，gym 0.26.2；去掉 tensorflow | RTX 5070 Ti 至少要 cu128；torchtyping 0.1.4 在 torch ≥ 2.3 下 import 失败；gym 被 `src/fqf_iqn_qrdqn/env.py` 用到；代码没有 import tensorflow |
| 11 | 回看天数不足 | 训练起点之前不满 100 个交易日时，下标变成负数，会绕到日历末尾，读错数据且不报错 | `StockData` 直接报错 | 修 bug；输入合法时行为不变 |
| 12 | 组合时的 inf 值 | 某个因子的值溢出成 inf 时，最小二乘的权重全部变成 NaN，预测随之失效 | `run_adaptive_combination.py` 在计算前把 NaN 和 ±inf 置为 0 | 修 bug，与 AlphaForge 的复现一致 |
| 13 | 运行记录 | 只定期保存因子池快照（`pool_*.json`）和模型 | 另外在运行目录写 `config.json`、`meta.json`（代码 commit、快照哈希、评估公式数、无效样本数、墙钟 / CPU / GPU 时间）和 `candidates.csv`（每个候选公式的训练期 IC、与池中因子的最大相关、是否入库及原因），并在 registry 里追加一行 | 统一协议的记录要求；只记录，不影响挖掘 |
| 14 | 论文原生评测 | — | 不改：每天取前 20% 等权，收益为 20 日远期收益 ÷ 20，信号当天收盘价成交，无交易成本，不处理涨跌停和停牌，不剔除 ST；IC 带符号；年化收益 = 日均 × 天数 / 3（3 是写死的年数：C 划分的测试期正好 3 年，所以测试期数值正确；验证期只有 1.5 年，脚本输出的验证期年化收益只有实际的一半）；夏普比率无风险利率取 0；最大回撤按累加收益算 | 保持论文实现；统一评测另用统一评测器计算 |
| 15 | 可重复性 | — | torch-geometric 的部分 GPU 算子不确定，同一种子不保证逐位相同 | 只做记录 |
