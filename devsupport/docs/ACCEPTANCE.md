# 对照批准方案的交付检查表

更新：2026-10-10。项目仍在进行，本表不构成整体验收。依据用户确认的完整 DevSupport Copilot 方案建立，工程实现、开发集实验和最终留出验证分别核验；用户要求暂缓人工审核，实际人工数量保持为 0。

最新状态：DEV 四组各10道控制题已全部完成，正式预先选择 H0，Full 为 H0 的别名、保留组件的 minus 为 B0；gate 与复检保留为实验功能。首次最终锁在 TEST 前生成。五组本地 TEST 各150条已完成、零运行错误，但外层读取被文档中的 U+2028 触发 JSONL 解析失败；原750条、顺序与 digest 已核验并保留。修正17处物理行读取，核心833项及独立833项通过。corrective v2 已冻结并实际执行，八种本地方法（B0/B1/H0/H1/C/U/R1/R2）各150条完整、零运行错误和零 fallback；固定TEST30生成正在执行，其余阶段仍在执行。原750条也已通过修正后的离线上下文审计。不能将两次运行当独立重复或称已完成最终 TEST。原因和恢复计划见 `../../docs/superpowers/plans/2026-10-10-jsonl-unicode-framing-recovery.md`，实际启动证明见 `reports/final-controller-launch-verification-20261010-v2.json`。

| 批准要求 | 当前证据 | 剩余验收 |
|---|---|---|
| 复用成熟 DocsGPT 的聊天、UI、上传和引用 | 原生 v2 回答/两个快照证据 `reports/ui/native-v2-answer-success-20261009.json`，原生上传/检索截图；上游固定提交见 PROVENANCE | 最终选定H0在UI中复验；Full=H0是正式等价别名，旧UI Full不是该别名 |
| 固定知识快照，不能包装成最新版 | 界面、回答和引用展示 LangChain 2024-10；引用展示保存的文档/源码 | 最终录屏保持快照声明 |
| 完整公开语料、稳定官方 ID、原文不改写 | `reports/data-manifest.json`：49,514 原始行、49,505 唯一 ID，重复/conflict 审计和 official_last 行选择 | 最终交付保留数据获取、归属与校验说明 |
| 公开来源 URL、path、byte offsets 可追溯 | 统一 Evidence 对象及原生快照引用 | 在最终 UI 再抽查文档/源码引用 |
| 答案、nuggets、qrels 只用于离线评价 | 在线 pipeline 输入字符串；数据/流程边界测试和源码记录 | 最终源码合同与运行样本对应 |
| 分离 Python/GPU/原生依赖，不重置既有数据库 | core-repro、原生 Python 3.12、WSL GPU 环境及固定依赖；恢复只启动原容器 | 最终启动说明与环境 manifest 校验 |
| Dense、BM25-only、普通 Hybrid 受控基线 | DEV53及corrective TEST150四条件完整；`reports/public-complete-local-test-v2/summary.json`，排名和装箱口径分开 | 最终完整报告继续保留各对照 |
| 上游原生 U 真实行为单列 | `reports/native-u/dev.jsonl`；实际 ClassicRAG，共享 Qwen/45,000 token 候选限制 | 保留与原厂默认配置的差别 |
| 标识符保真 Hybrid 与普通 Hybrid 分开比较 | H0/H1 DEV53、完整/拆分符号分词与 RRF；贡献不能与增加 BM25 混同 | 依据 DEV 决定保留，冻结后验证 |
| 两种现代重排在相同 50 项池比较 | DEV53 R1/R2、真实 GPU 推理及候选成员检查 | 负结果保留；不默认增加无净收益重排 |
| listwise 输入顺序敏感性 | 5 问×3 顺序记录及审计；不是 15 个独立问题 | 最终报告保留小样本边界 |
| never/always/selective 有界复检对照 | H0/Q/QAlways 各 DEV53；`reports/dev-adaptive-h0-v2-audit/audit.json` | DEV12独立回答观察已完成；Q/QAlways各TEST150仍待完成 |
| 原问题保留、最多两子查询/一轮、失败回退 | 严格标识符校验、bounded anchoring、容量预检、运行 trace 与回归 | 最终配置与 UI 行为一致 |
| 证据门控独立于再次检索 | 三类实现；四组DEV12、同题AI评分及四组DEV10控制均完整；gate未选入默认H0 | H0/G的TEST30控制对照待完成；实现完成不能当作门控有效 |
| 元数据语境前缀不冒充全文 LLM Contextual Retrieval | C 全量索引、DEV53、原文和索引表示分离 | 已完成限时独立实验，最终是否保留以 DEV 质量/成本决定 |
| ModernColBERT 挑战实验 | 原方案明确仅主线完成且瓶颈支持时考虑、最多四小时 | 条件式可选项，不是当前承诺或已实现功能 |
| DEV53/TEST150 本地划分，近重复分组隔离 | `reports/split.json`与冻结身份；八种本地TEST150已完整，不是官方split | 不得根据TEST结果改选或调参；披露corrective解析故障重跑 |
| 同一模型路由、提示词、预算与 seeds | DEV选择已预先完成；v1及corrective v2锁和源码/参数差异证据完整，v2参数未变 | 剩余队列仍须对应v2合同；网关底层身份未验证须披露 |
| 固定上下文和输出预算 | 4,000 cl100k 代理 tokens、整块装箱、2,048 输出上限 | 与未知生成 tokenizer 的差别保留；旧字节协议结果不混算 |
| 候选 Recall@50、排序指标、nugget Coverage | 八种本地TEST150完整；H0−B0 Coverage@10 +8.59pp，配对区间[+3.16,+13.53]pp | Q/QAlways各TEST150及全体最终报告待完成 |
| 固定预算下 Context Precision/Recall 和证据覆盖 | `reports/context-quality-dev-v1/`，371 条 DEV；新增现代格式完整审计和 DEV12 零付费复算 `reports/dev-context-recovery-v3-audit-v2/`，34 专项/161 相关回归通过，两个独立 AI 代码审查发现已修复 | 八种本地TEST150同预算装箱分析已完整；Q/QAlways及生成记录的最终全体审计待完成，未标注不直接等同不相关 |
| 正确性、覆盖、引用支持、Faithfulness | 历史 DEV12 与新决策DEV12均完整AI评分；新 `dev-decision-ai-quality-v4/summary.json`：12问、29回答+1非错误拒答，18次失败计入效用分母 | 最终 TEST30 AI 评价；不混合历史两队列，AI 不等于人工或独立底层模型 |
| 产品控制集 40 题，DEV10/TEST30 | 冻结40题；四种方法DEV10全部完成，完整报告与严格复审保留 | H0/G两组TEST30仍待完成；标签为AI编写、人工未核验的代理标签 |
| 真正人工盲评30题与标签核验 | 用户明确暂缓，实际均为 0 | 不阻塞当前交付，保留未完成人工状态，不生成虚构评分 |
| 运行错误、错答、误拒答、回答覆盖同时报告 | outcome/stage failures 与分母审计实现；旧运行故障保留 | 完整控制决策报告与答案评价尚待完成 |
| 分阶段耗时、p50/p95、费用/重试/降级/截断 | DEV traces及新增完整cohort accounting工具，95项相关/独立回归通过；新DEV12有93次阶段调用、105次已记录传输，G另有2次未知传输计数，费用不可用 | 最终TEST/控制报告；不得把阶段调用当精确付费次数或部分usage当全量计费tokens；提前失败的低耗时不是加速 |
| Final Full-minus-retained-component 消融 | 最终预先选择H0；Full=H0、FullMinusHybridBranch=B0，检索各TEST150完整 | 同题TEST30生成/AI评分待完成；等价条件复用，不当独立重复 |
| 来源变化、无效引用、模型/索引不匹配等异常 | 针对真实 ID、池成员、调用边界、引用合同与失败分母的回归；当前核心 833 项通过，4 项既有依赖弃用警告 | 修正版最终配置后按变更范围验证，不以单元测试替代业务效果 |
| PRD、架构、决策、失败、运行与复现 | 本目录 PRD/ARCHITECTURE/EXPERIMENTS/FAILURES/RUNBOOK/PROVENANCE | 最终成绩和选择更新，检查过时占位 |
| 五分钟真实演示 | 317.9 秒 MVP 视频、175 实时样本、中文导出，采样空隙公开；无新增生成或上传 | 最终选定流程、门控与 TEST/消融结论视频仍需补录 |
| 简历/面试说明，不能编造收益 | INTERVIEW已加入完整本地TEST检索证据、区间、corrective口径；不称答案准确率 | 剩余生成/控制/复检完成后再更新最终版本 |
| 可分发脱敏交付包 | 历史源码扫描无当前凭据明文；新离线 exporter 经独立复核、23 项边界测试，实际演练包 2,909 个选中文件经两项凭据字面扫描、ZIP 校验和新目录解包/逐文件哈希复检；精确匹配上游 dummy fixture | 演练包不含最终 TEST/消融/视频；新增当前基底2990文件/134,986,411字节实际预检、两凭据匹配为0；最终allowlist、archive与验收仍待完成，字节扫描不等于媒体像素扫描 |
| 全部逐题证据可分享，原始记录不改写 | 新白名单派生工具经15项测试与独立复核；`reports/public-dev-decision-generation-v2/summary.json`：DEV12x4全部48条保留、原文件/派生文件哈希复检、两项内存凭据扫描 | 最终所有完整队列用同一规则派生；汇总仍从私有完整原记录计算，派生视图不能用于resume |

当前 H0/Q/G/Full 的旧批次 `dev-decision-generation-v1` 含本地服务中断造成的无效结果，保留为故障记录，不用于新批次的配对统计。新 `dev-decision-generation-v2` 在 12 题零付费检索全部通过后独立运行所有四组；旧失败不覆盖、不选择性替换，新的失败也计入分母。具体检查点随 JSONL、manifest 与 controller 的实际状态更新。

原始质量目标（相对 Dense 的 Coverage@10 +5pp、相对普通 Hybrid 的 alpha-nDCG@10 +0.03）属于待验证假设，不能改写为保证。实验覆盖一个方向与证明它有效是两个不同验收项；没有净收益的组件应保留实验记录并减少默认复杂度。

2026-10-09 21:24 的离线交付演练证据为 `reports/handoff-packaging-rehearsal-20261009-2122.json` 和 `reports/handoff-rehearsal-extraction-20261009-2124.json`。两份上游前端 `.env` 配置未进入包，其他受跟踪文件通过显式路径选择；最终安装须按 RUNBOOK 配置本地 API，不沿用上游云端地址。演练包含已有 MVP 视频，未冒充最终演示。质量审计/exporter 与相关工具合计 128 项回归通过，42 个当前生成运行源码合同保持不变。

2026-10-10 08:29：原生U及查询模式遮盖扩展完成后，核心800项通过，4项既有依赖弃用警告；public/export相关44项及两次只读独立复核通过。实际21份完整DEV方法文件703条派生记录、3条记录6个明确遮盖查询字段、两项内存凭据扫描与原始/派生hash复检见 `reports/public-dev-completed-v1/summary.json`。这不代表703个独立问题、TEST成绩或最终交付。

2026-10-10 后续复验：现代原生 runner 的固定 DEV12 已完成，零运行错误、零付费生成，全部十二题排序与历史原生结果一致。对应十二条现代格式公开派生记录保留完整 native trace，原始/派生哈希均复检；这是兼容性诊断，不增加独立评测样本数。Python 3.12 与 3.10 的浮点求和产生三行、八项 nDCG 微小差异；离线审计仍严格验证原 record digest 与所有非 metric 字段，仅对有限非布尔数值采用公开的 1e-12 相对/绝对容差，逐项披露差异，不改写原记录。历史 DEV12 四组共48条上下文、方法成绩与配对区间保持一致。核心811项通过，独立审查无剩余 P1/P2，见 `reports/native-cross-python-context-verification-20261010-v1.json`。原生环境仅补三个缺失包、其余版本未改，见 `reports/native-evaluation-dependencies-verification-20261010-v1.json`。TEST、最终 UI/视频与交付包仍待完成。
