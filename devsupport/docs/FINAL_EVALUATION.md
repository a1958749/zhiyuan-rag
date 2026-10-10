# 最终验证执行清单

状态：DEV 控制四组各10题已完成，DEV 正式选型为 H0；Full=H0、FullMinusHybridBranch=B0，实验 Q+gate 改名 QG。原 v1 lock 已在 TEST 前生成，五组本地各150题完成后因 Unicode JSONL 读取问题停止。完整 TEST 尚未完成。修正版 v2 已于 2026-10-10 07:10:38 UTC 冻结并实际启动；五种本地方法及原生 U 各完成150条、零运行错误，剩余阶段串行执行。v2 仅修正解析源码，保留原锁/750条结果，不改选择、参数或数据，也不把两次运行当独立重复。原 v1 未启动付费 TEST，因此本次没有重放原付费 TEST。实时进度以 controller 状态为准，不由本文预先宣布完成。

当前配置为 `configs/evaluation.final.v2.json`，与 v1 配置字节相同；锁为 `reports/final-evaluation-lock-v2.json`，SHA256 为 `5b0e89be88826853c5cbe1808d85798a8369791d9d757773c5cc18bd334e4674`。实际启动证明为 `reports/final-controller-launch-verification-20261010-v2.json`。源代码修正提交为 fa9a185。原750条的修正后离线上下文审计也已完整通过，见 `reports/test-initial-framing-context-v1/summary.json`；这不是另一个独立测试样本。

## DEV 选择的前置证据

当前四组 `H0 / Q / G / Full` 必须各有相同顺序的全部 12 条记录，失败保留，且没有未决调用。完成对应 AI 评分和 10 道 DEV 控制题的四组对照后，联合记录：含失败效用、回答覆盖、误拒答、无支持控制题回答率、澄清决策、分阶段耗时、实际请求与 token 使用。控制标签未经人工审核，AI 分数不称准确率。生成失败不能充当知识不足拒答；回退和已修复的中间错误单列。

选型使用已有完整 DEV53 检索/装箱和 DEV12 生成证据，不通过只挑成功子集或单个好案例保留组件。费用/延迟增加而质量未建立增益时，默认减少复杂度。门控需要同时检查错误回答与误拒答，不能因拒答更多就认定有效。明确记录样本小、单次生成、方法顺序及网关波动的限制；不能将差值归因于已隔离的纯算法因果效果。

## 最终配置与消融

保留每个组件的选择理由和被拒绝方案的负结果，然后创建新的最终 evaluation 配置。数据、embedding、回答路由/提示词、上下文协议与计费保护不因方法改变。冻结同一配置中的全部对照、最终方法和消融参数，不能看 TEST 后更改。

| 最终保留内容 | 必须对应的 Full-minus 条件 |
|---|---|
| BM25 + Dense + RRF | 去掉 BM25 的 Dense 条件；Dense/RRF 默认参数固定 |
| 标识符词法处理 | 替换为普通词法的 Hybrid，其他组件相同 |
| 重排 | 同一基础候选池和后续流程，仅关闭重排 |
| 语境前缀 | 同一 embedding 与流程，改用 raw 索引 |
| 按需复检 | 关闭分解、保留第一轮流程和最终 gate |
| 证据门控 | 仅关闭最终 gate，保留检索和复检 |

正式预先选择 H0，仅保留普通 BM25 + RRF 增量。标识符分词、两种重排、语境前缀、复检与 gate 已实现并实验，但未进入默认方案。最终生成队列固定为 `B0 / H1 / H0`；Full 复用 H0，FullMinusHybridBranch 复用 B0，不再次生成等价答案后声称独立重复。产品控制对照保留 `H0 / G`，明确 gate 是实验功能，不是默认 H0 已有的行为。

AI judge 使用完整同题三方法队列、固定8192输出 token、explicit_coverage协议和零重试；先离线预检，不截断证据或丢弃条件以通过预算。TEST开始后不增加方法或重新选型，也不把不同方法集合的独立评分拼成一个公平总榜。没有选中重排，不虚构“选定重排版”。

## 顺序与对应产物

| 顺序 | 阶段 | 必须完整的范围 | 输出证据 |
|---|---|---|---|
| 1 | DEV 选择 | 上述 DEV53、DEV12、DEV10 | 决策表与依据路径/hash，保留负结果 |
| 2 | 最终冻结 | 所有选定/对照/消融配置与源码/数据/model manifests | 新的 immutable lock，记录冻结时间 |
| 3 | 本地检索 | U、B0、B1、H0、H1、C 各 TEST150 | 完整有序 JSONL/manifest，原生 U 使用实际 Python3.12 配置 |
| 4 | 本地重排 | R1/R2 各 TEST150 | 同一完整 H1 50项候选缓存、输入/输出池一致 |
| 5 | 复检及最终/消融检索 | Q与QAlways各TEST150，以及需要复检的最终与消融各TEST150 | 兑现所有检索实验完整问题范围；retrieve-only、触发/扩展/回退及成本，该阶段付费 |
| 6 | 回答生成 | 冻结的 TEST 前30题，各生成对照/消融 | 所有尝试保留，包括错误；不挑成功题补齐 |
| 7 | 质量与控制 | 同题 AI 评价；控制 TEST30 | AI 分数与条件分母、40题中独立留出30题的决策代理结果 |
| 8 | 总结 | 所有完整比较与逐题失败 | 配对区间、上下文/答案/拒答/运行指标、限制 |
| 9 | UI/视频/交付 | 最终选定流程和完整成绩 | 实际 UI parity、五分钟演示、脱敏 archive/receipt/解包验证 |

基础检索包括已验证的全部方案，不能只跑 H0/最终方案就声称完成了原批准范围。U 的实际配置和候选截断保持披露；不能叫原厂默认 Granite。R1/R2 的负 DEV 结果不等于可以省略承诺的留出检索对照。真实 TEST150 与 TEST30 输出分开分析，不混合 cohort。

Q/QAlways即使不进入默认产品，也要按冻结参数运行完整TEST150检索，满足原批准的“检索实验全部问题都跑”；它们的TEST结果仅验证预先选择的实验，不用于重新选型。最终控制TEST30至少保留相同底座的无gate/gate对照；若最终保留额外组件，再加入相应必要条件。最终生成保留B0、H1、DEV选定方案及所有保留组件的minus条件，等价条件明确复用。

### 跨运行目录执行约定

DEV选择已完成，本地依赖中断停止修复已合入139cb99。最初 v1 配置和锁均已生成，随后仅为 JSONL 解析故障创建 corrective v2 配置/锁和新目录。下述 v1 命令保留为历史执行示例，不得继续执行、覆盖或作为当前恢复入口。当前串行执行器为已捕获源码 `reports/jsonl-framing-recovery-v2-sources/final-workflow-controller-20261010-v2.py`，实时状态为 `reports/final-workflow-controller-20261010-v2.json`。每一实际阶段均核对 v2 contract；数据和源码改变会使锁校验失败。

原生U单独使用现有Python3.12，其他本地组使用core-repro；不能将单独目录改写成另一个runner的输出。现有context审计已经支持相同合同、相同完整cohort且方法名不重叠的多个目录。

```powershell
# 第一步：零付费本地检索；只有最终锁已生成且DEV控制进程已终止后执行。
& 'F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe' -X utf8 devsupport/scripts/evaluate_pipeline.py --config devsupport/configs/evaluation.final.json --split test --lock devsupport/reports/final-evaluation-lock-v1.json --phase retrieval --profiles B0 B1 H0 H1 C --output devsupport/reports/test-local-v1
& 'F:/DevSupportCopilot/.venv/Scripts/python.exe' -X utf8 devsupport/scripts/evaluate_pipeline.py --config devsupport/configs/evaluation.final.json --split test --lock devsupport/reports/final-evaluation-lock-v1.json --phase retrieval --profiles U --output devsupport/reports/test-native-u-v1

# 第二步：复用上述完整H1池，保持成员/顺序身份检查。
& 'F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe' -X utf8 devsupport/scripts/evaluate_pipeline.py --config devsupport/configs/evaluation.final.json --split test --lock devsupport/reports/final-evaluation-lock-v1.json --phase rerank --profiles R1 R2 --pools devsupport/reports/test-local-v1/H1.jsonl --output devsupport/reports/test-rerank-v1

# 第三步：一次只有一个付费队列；不能与控制/生成/UI调用并行。
& 'F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe' -X utf8 devsupport/scripts/evaluate_pipeline.py --config devsupport/configs/evaluation.final.json --split test --lock devsupport/reports/final-evaluation-lock-v1.json --phase retrieve-only --profiles Q QAlways --output devsupport/reports/test-adaptive-v1

# 每组150条完整、无未决调用后，合并分析并保留全部失败。
& 'F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe' -X utf8 devsupport/analysis/evaluation_context.py --split test --lock devsupport/reports/final-evaluation-lock-v1.json --runs devsupport/reports/test-local-v1 devsupport/reports/test-native-u-v1 devsupport/reports/test-rerank-v1 devsupport/reports/test-adaptive-v1 --output devsupport/reports/test-context-v1
```

生成及控制命令中的方法列表须由完整DEV选择文件得出，再与最终lock核对；不能预先将实验候选Full当作最终方案。每个阶段先查健康和原进程终态；非零退出、记录不足或inflight都停止自动链，不能用report的部分输出当完整验收。TEST150检索与固定前30题生成的context报告分别创建；原生U只参与检索对照，不额外生成一份等价dense回答。

现有执行入口：`scripts/freeze_evaluation.py`，`scripts/evaluate_pipeline.py`（`--split test --lock ...`），`scripts/report_evaluation.py`，`analysis/evaluation_context.py`，`analysis/answer_judge.py` / `answer_quality.py`，`analysis/control_decisions.py`，`analysis/provider_accounting.py`。后者只分析完整队列，区分阶段调用、已记录传输、未知尝试和响应 token；部分维度与重试不标为完整计费 token，费用为配置费率估计或不可用，不由共享余额差反推逐方法费用。真实 TEST 必须提供对应 lock，发布使用新路径。命令的实际配置和输出路径必须从最终 lock 核对，不直接执行 example 配置或在已有目录覆盖结果。

## 运行与交付约束

一次只运行一个付费任务，沿用同一 pacing/预算账本；每次恢复复检健康和新鲜零付费检索。探针不算问答样本；已尝试的失败题不自动重放，未决调用须先查明状态。供应商错误停止后续调用并保存检查点，保守等待不声称已恢复；不自动充值、扩充授权或偷偷改路由。

最终 UI config 需对应 DEV 选择。核对最终 gate/adaptive/retrieval/reranker/上下文参数、实际证据与引用，不将评测 tracer 与产品 wrapper 的代码 hash 冒称完全相同。演示使用 DEV 或预先指定的新问题，不能从 TEST 挑好看的成功题。保留知识快照、真实等待/故障及适用限制。

最终 archive 使用 explicit allowlist，包含源代码、锁定信息、数据获取流程、完整实验/消融和可分享报告、文档及最终视频。排除凭据、环境密文、账户截图、余额文件、原始错误正文、模型权重、索引和数据库；核验源内容、媒体画面、ZIP 与实际解包。已有打包演练和 MVP 视频不能替代最终产物。人工评分/标签核验按用户要求暂缓，实际数量保持 0。

原始逐题 JSONL 可能内嵌 `balance_observations`，生成校验失败 trace 可能保留 `response_text`；不能因文件名没有 balance 或 error 就直接公开。保留原始完整记录私有且不改写，交付中的逐题材料使用明确白名单派生视图，记录来源文件 SHA256、原始 record_sha256、派生规则和去除字段。公开全部题 ID、方法、排名/证据 ID、原始答案、决策、指标、分阶段运行诊断及用量状态，去除账户观察和校验失败的原始供应商文本。共享视图不冒充原始完整合同记录，也不能用于原 runner 的 `--resume`。汇总和配对统计始终从完整原始 cohort 计算，不能因公开派生过程丢掉困难题。最终扫描针对选中的实际字节及两项内存凭据进行；媒体另作目视复核。

真实DEV导出发现公开数据问题78549270本身含疑似密钥模式，不能确认其有效性。显式query-only遮盖仅替换原问题/检索/复检查询中的该模式，逐字段记录路径、原文hash与替换数；原始query/record hash、答案、证据、指标不变，整题保留。默认导出仍严格拒绝，已知凭据必须在遮盖前拒绝；答案和证据中的模式不允许借此改写。已导出 `reports/public-dev-completed-v1/summary.json` 的21份完整DEV方法文件、703条记录，三条记录共有六个字段遮盖。该视图不是原始输入复现或最终TEST证据。
