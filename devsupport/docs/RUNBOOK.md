# 运行与排障手册

2026-10-10 原生 U 批量评测环境补充：原有 Python 3.12 能运行 DocsGPT 和旧的原生检索脚本，但缺少新统一 runner 所需的 pyarrow、BM25S 和 SciPy。首次真实 DEV 预检在读取数据时得到 `ModuleNotFoundError: pyarrow`，尚未发出 embedding 或生成调用。已在原 `.venv` 仅新增 pyarrow 24.0.0、bm25s 0.2.14、scipy 1.18.1；前后分发版本逐项比较，原已安装包没有升级、降级或删除。NumPy 2.5.3、FAISS 1.15.1 保持原生版本，core-repro 的对应版本为 2.2.6、1.13.0；原生 U 不冒称与受控组完全相同的 Python 数值运行环境。

本机使用已有 `F:/DevSupportCopilotWork/tools/uv.exe pip install --python F:/DevSupportCopilot/.venv/Scripts/python.exe --index-url https://pypi.org/simple --no-deps --only-binary :all: pyarrow==24.0.0 bm25s==0.2.14 scipy==1.18.1` 添加缺失包。SciPy 版本来自已冻结上游 uv.lock，并兼容现有 NumPy；此命令针对该记录环境，不替代新机器上的依赖完整性检查。未重建环境、停止数据库或更换模型。后续统一原生 runner 的完整 DEV 预检记录单独保存，不冒充 TEST 或新增独立质量样本。

统一原生 runner 的固定 DEV12 已实际完成，零运行错误、零付费生成，全部十二题排序与历史原生 U 一致。环境分发版本前后复检见 `reports/native-evaluation-dependencies-verification-20261010-v1.json`。混合 Python 3.12/3.10 的离线上下文审计核验原记录 digest 和所有非 metric 字段，并逐项披露有限 metric 在 1e-12 相对/绝对容差内的浮点差异；不修改原始指标。真实 DEV12 与历史四组上下文复验见 `reports/native-cross-python-context-verification-20261010-v1.json`。此兼容性验证不能代替最终 TEST。

更新：2026-10-10。仓库 `F:/DevSupportCopilot`，任务环境 `F:/DevSupportCopilotWork`。本手册按源码入口和实际执行记录核对；进程及健康以每次新鲜检查为准。先检查已有进程，避免重复启动。

可选 C 全量索引现已 complete_verified（49,505 ID），DEV53 检索零错误。构建期间 GPU 服务曾暂停以避免重复载入；启动前检查 8093 实际状态，不能假定当前已运行或重复启动。索引完成不代表正式生成/holdout 完成。

当前共享上下文协议为4,000 `cl100k_base_proxy_v1`代理tokens，校验上游打包tokenizer的SHA256，不代表未知网关生成模型的精确token数。旧4,000 UTF-8字节报告仅作被取代的诊断，不混合汇总。DEV所有选型对照、四组控制与八种本地TEST150及上下文审计已完成；最终TEST30生成/控制、Q/QAlways TEST150、完整汇总及最终版视频仍在进行。真人盲评按用户要求暂缓，实际0。详情见`EXPERIMENTS.md`。

当前最终默认是H0，Full仅是评测合同中H0的别名；服务的旧`--enable-full`开关会启用历史Q+gate实验流程，与选定H0不等价，因此仍不启用该开关。最初PID3308/session86337与v3续跑均已异常退出；当前续跑入口为v4，状态为`reports/final-workflow-resume-20261010-v4.json`，最近确认的运行会话为26952、PID32496。合同SHA仍为`5b0e89be88826853c5cbe1808d85798a8369791d9d757773c5cc18bd334e4674`。进程、阶段和健康每次以新鲜检查为准，不凭这些历史标识重新启动。禁止在付费队列活跃时并行提交UI问题或网关探针。

普通下载者的入口是根目录`GETTING_STARTED.md`，不必照搬本机F盘路径或先重建FreshStack。Windows首次启动使用三个PowerShell终端：`uv run python -m docsgpt dev`启动API/worker；在`frontend`执行`npm run dev`；另开终端`uv run python -m docsgpt beat`。当前CLI的`--ui`在Python子进程中使用`npm`，本机真实探针得到`FileNotFoundError`，而`npm.cmd --version`成功。此处采用独立启动前端的方式，未修改正在评测的原生Python源码，也未宣称整套新机器安装已通过。

## 环境与启动顺序

| 组件 | 环境/端口 |
|---|---|
| GPU embedding / rerank | WSL `Ubuntu2204-F`，`/opt/devsupport-runtime/venv-models/bin/python`，8093 |
| 共享 pipeline | `F:/DevSupportCopilotWork/.venv-core/Scripts/python.exe`，8094 |
| 原生 API / worker / beat | `F:/DevSupportCopilot/.venv/Scripts/python.exe`，Python 3.12，API 7091 |
| PostgreSQL / Redis | 现有 `docsgpt-oss-postgres-1` / `docsgpt-oss-redis-1`，5433 / 6380 |

新离线实验使用独立的 `F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe`。其 58 个包版本已核对，环境复现步骤见 `../environments/README.md`。上表和 `start.ps1` 默认仍为现有 UI 服务使用的原 core 环境；历史生成记录不改写为使用了新环境。以下离线命令采用 core-repro，实际 GPU 运算仍由 WSL 服务完成。

2026-10-09 15 时恢复检查：WSL 和原有数据库/Redis 容器此前已停止；现已启动原容器，GPU 8093、pipeline 8094、原生页面 7091 均返回 HTTP 200，API/worker/beat 已恢复。此次 pipeline 显式使用 core-repro 与 `--enable-contextual`，未启用 Full。旧 DPAPI 密文在当前 Windows 用户环境无法解密，用户重新提供原路由凭据后已重新加密保存并验证生产读取；没有把密钥写入文档或仓库。单次有界网关探测成功后，以原合同续跑 QAlways，不重放已有失败行。当前进程/健康仍须在执行时重新核对，恢复检查不是完整产品验收。

在 PowerShell 进入仓库：

```powershell
Set-Location F:/DevSupportCopilot
$env:PYTHONPATH = (Join-Path (Get-Location) 'devsupport/src')
Get-NetTCPConnection -State Listen -LocalPort 7091,8093,8094,5433,6380 -ErrorAction SilentlyContinue
wsl -d Ubuntu2204-F -u root -- docker ps -a --filter name=docsgpt-oss
```

`PYTHONPATH` 只设置当前 PowerShell 会话的源码导入路径；干净 core 环境没有安装本项目包，`download_data.py` 等入口依赖这个路径。重新打开终端后须再次设置，不要求向全局 Python 安装项目。

仅当已存在的容器停止时启动；不运行 down、rm、volume prune，也不重建数据环境：

```powershell
wsl -d Ubuntu2204-F -u root -- docker start docsgpt-oss-postgres-1 docsgpt-oss-redis-1
```

模型服务独立终端（需要时保持 WSL 会话存活）：

```powershell
wsl -d Ubuntu2204-F -- /opt/devsupport-runtime/venv-models/bin/python /mnt/f/DevSupportCopilot/devsupport/scripts/serve_gpu.py --unload-before-rerank
```

pipeline使用现有安全进程记录脚本；**core Python不加`-X utf8`**，以便与启动/停止脚本的进程身份匹配。最终选择为H0，不传EnableFull；历史实验Full不是最终别名H0。

```powershell
./devsupport/scripts/start.ps1
```

原生 tokenizer 缓存准备只从本地已冻结 tokenizer 复制并校验 SHA，不重新下载模型。内部 worker/API 密钥准备脚本是幂等本地配置步骤，不显示或提交凭证：

```powershell
./devsupport/scripts/prepare_native_secrets.ps1
& F:/DevSupportCopilot/.venv/Scripts/python.exe -X utf8 devsupport/scripts/prepare_native_tokenizer.py
```

分别在三个终端运行原生角色；原生 Python 使用 `-X utf8`：

```powershell
& F:/DevSupportCopilot/.venv/Scripts/python.exe -X utf8 devsupport/scripts/serve_native.py --role api
& F:/DevSupportCopilot/.venv/Scripts/python.exe -X utf8 devsupport/scripts/serve_native.py --role worker
& F:/DevSupportCopilot/.venv/Scripts/python.exe -X utf8 devsupport/scripts/serve_native.py --role beat
```

上述三行是三个独立前台进程，不应在同一个终端等待上一行结束后才启动下一行。原生 launcher 配置离线 HF tokenizer 缓存并读取已授权的本机凭证；不要打印 `.env`、`.secrets` 或把密钥放在命令行。若自己通过 Start-Process 后台运行，指定 `-WindowStyle Hidden`，记录 PID、路径及启动时间后再管理进程。

2026-10-09 20 时重新续跑时发现旧服务已退出，WSL 显示 stopped。只核对了源码与余额就续跑，导致 19 条新增 `ConnectError`，没有记录到模型调用；随后中止批次并恢复原服务。**每次恢复、切换执行环境或开始新的批次，都必须重新检查 8093/8094 的实际健康和进程，并通过一组零付费真实检索。历史健康截图、余额可用和状态文件不能替代当前检查。** 新批次在 12 道 DEV 题全部召回 50 项、零错误后才启动生成。旧故障与中断标记保留于 `reports/dev-decision-generation-v1/local-outage-20261009-2017.json`；不续跑、覆盖或混入新配对批次。

## 检查与停止

浏览器打开 `http://127.0.0.1:7091`。健康检查不等于回答质量验收：

```powershell
Invoke-RestMethod http://127.0.0.1:8094/health
```

先核对选定来源，检索并打开快照引用；生成问题只有在网关可用且预算已核对时执行。上传需同时检查 UI 状态和 worker 处理结果，不能把“上传已提交”视为索引成功。

停止 pipeline 使用其进程身份校验：

```powershell
./devsupport/scripts/stop.ps1
```

原生三个前台进程和 GPU 前台进程分别 Ctrl+C。对既有后台进程先核对 PID、可执行路径与命令行，不能按所有 Python 进程批量终止。保留数据库、Redis、语料、索引和模型缓存。

## 数据、索引和 DEV 复现

下列是按需重建/复现命令，会下载数据或占用 GPU；不要为启动服务例行重建。已有产物优先用 manifest 校验。下载脚本维护完整快照和审计报告：

```powershell
& F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe devsupport/scripts/download_data.py --data devsupport/data --reports devsupport/reports
wsl -d Ubuntu2204-F -- /opt/devsupport-runtime/venv-models/bin/python /mnt/f/DevSupportCopilot/devsupport/scripts/build_index.py
wsl -d Ubuntu2204-F -- /opt/devsupport-runtime/venv-models/bin/python /mnt/f/DevSupportCopilot/devsupport/scripts/evaluate_retrieval.py --split dev --variants B0 B1 H0 H1 --run-name retrieval-reproduction
```

统一 runner 参数以 `evaluate_pipeline.py --help` 为准，阶段包括 retrieval / rerank / retrieve-only / pipeline；后两类可能因配置启用 Q/G 而付费，不能凭阶段名认为免费。示例配置是模板，执行前核对所有路径、预算、模型身份和选定组件。当前不要执行 holdout；须完成 DEV 选择并由 `freeze_evaluation.py --config ... --output ...` 生成不可变锁后，再按已批准范围运行。不得以看过 TEST 结果后调整配置的方式保留“未触碰测试集”说法。

分阶段 TEST 评估中，`retrieval`、`rerank`、`retrieve-only` 按冻结顺序覆盖全部 150 个 ID；只有生成回答的 `pipeline` 使用前 30 个。所有 TEST 阶段仍须验证同一冻结锁及 generation cohort 绑定；DEV 使用配置的 `dev_ids`，`--controls` 则使用冻结 controls 文件中对应 split 的 ID。`retrieve-only` 不生成回答或运行决策 gate，但自适应查询分解仍可能调用付费模型，须沿用获准预算与记账。其结果仅支持检索范围的结论，不构成人工评审或回答质量证据。

预算协议可用缓存排名单独复现，不进行模型/API 调用；输出选择未存在目录，保留已生成的 `reports/packing-budget-comparison`：

```powershell
& F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe devsupport/scripts/audit_context_budget.py --rankings devsupport/reports/retrieval-v1/dev/B0.jsonl devsupport/reports/retrieval-v1/dev/H0.jsonl devsupport/reports/retrieval-v1/dev/H1.jsonl devsupport/reports/contextual-v1/dev/C.jsonl --output devsupport/reports/packing-budget-reproduction
```

脚本要求每个输入包含完整有序 DEV53，输出记录排名、拆分、语料和脚本哈希。新旧计数器差值只解释预算协议影响，不作方法或回答质量收益。

## 两类额度与续跑

用户已于 2026-10-09 明确取消 Codex 账户已用达到 71% 时停止工作的规则，见仓库 `AGENTS.md`。后续不再因该阈值暂停项目；第三方 API 的既有预算记账与服务限额仍适用，不自动充值。

付费 runner 从配置的 `budget_start` 读取授权，现有路径为 `devsupport/reports/budget-current-gateway.json`。文件中的 `starting_quota`、`allowance_quota`、`reserve_quota` 均为网关原始 quota 数值，要求有限、非负；allowance 必须大于 0 且不超过 starting，reserve 必须大于 0。当前记账换算为 500,000 quota / 平台额度单位，不能直接当人民币或美元。此文件不是凭据；凭据仍由已忽略的 DPAPI 文件读取。

不要为恢复调用重写授权或删除其 `.accounting.json` 记账文件。账本累计共享密钥余额下降，人工充值也不会自动扩大获准 allowance。余额不可读、账本损坏或额度到线时会阻止调用。现有保护是调用前观察检查，不能保证在供应商统计延迟、并发或其他客户端使用同一密钥时形成精确金额上限；评测、UI 和 probe 不并发付费。UI 共用预算保护已在完整 DEV53 复检批次结束后以 `f1ca3fe` 合入并重启部署；所有 UI profile 共用同一 guard/provider 和服务内锁。缺失或损坏的授权/账本使启动失败，构造不检查余额或发起模型调用。该锁不构成跨进程调用额度的原子预留，仍禁止与付费评测并发。

一次 `run` 的 JSONL、manifest、摘要和源码哈希共同定义续跑点。只有进程已确认结束、没有 `.inflight.json` 未决调用，且代码、配置与数据仍匹配原合同，才能显式使用 `--resume`。该选项保留已落盘的成功和失败行，从下一条未尝试问题继续；不会替换失败样本。源码改变时不能强改 manifest 来续跑，应保留旧实验并另立协议。

最新恢复检查点：2026-10-10 02:42（北京时间）。原批次 H0/Q/G/Full 已记录 12/12/10/0，共34次尝试；H0/Q/G 分别有1/4/6条运行错误，全部保留，不重放。02:21 与02:30的末题记录只有 gate 的通用 provider_failure，无法确定是否实际发出模型请求或具体 HTTP 原因；不能沿用此前 HTTP502 的解释。02:35的一次64-token恢复探针在 GuardedProvider 的预算检查处抛出 BudgetStopped，provider_record为空、观察列表为空，因此尚未调用底层模型。02:38的零模型预算恢复检查明确失败为balance_unavailable，该controller已停止；不能宣称它恢复了生成。随后一次只读usage GET为HTTP200，返回数值额度，证据为 `reports/budget-read-diagnostic-20261010-0242.json`，没有模型调用、请求头或正文保存。间歇性读取失败的网络/供应商根因未建立，额度充足不代表下次读取或模型调用一定成功。02:41的新controller `reports/decision-dev-controller-20261010-v11.json` 核验健康、原合同和34行前缀，完成DEV12本地检索后，用原runner和原预算账本续跑未尝试题；不再发额外付费探针或冗余的额度GET，实际每次调用仍须通过生产guard自己的新鲜余额检查，没有缓存替代、禁用保护或扩充allowance。固定前缀见 `dev-decision-generation-v2/stop-20261010-0236.json`。运行阶段重新读取controller/JSONL，不并行付费UI/模型任务。当前四组、AI评价和控制对照未完成，预算读取通过不是模型质量证明。

恢复前历史：2026-10-09 20:59:49 门控 v2 因 HTTP403/code1007 停在 H0/Q/G/Full=12/2/0/0；21:30 探针仍为 HTTP403。22:30 的第二个探针则是约 90 秒 `network_error`，没有新 HTTP403 响应，进程已退出、余额核对未变。随后无凭据网络诊断确认 TLS1.3 和无认证 models 路径的 HTTP401，仅证明可达，不证明模型恢复。所有旧探针、失败和前缀检查点保留，包括 `stop-20261009-2132.json`、`stop-20261010-0112.json` 与 `gateway-network-diagnostics-20261009-2236.json`。

历史停止检查点：`dev-adaptive-h0-v2` 曾在 2026-10-09 05:59 因 HTTP 403 / code 1007 停止，当时 H0 53/53、Q 33/53、QAlways 0/53。响应没有 Retry-After；本地保守等待到 06:29，不代表供应商承诺恢复。之后已多次在核对原合同后续跑，15 时恢复见前文；以下是条件式续跑示例，不能对已有活动进程重复执行。71% Codex 停止条件现已取消，原合同、供应商冷却与 API 预算约束仍适用。

```powershell
& F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe -u devsupport/scripts/evaluate_pipeline.py --config devsupport/configs/evaluation.adaptive.dev.json --phase retrieve-only --profiles H0 Q QAlways --output devsupport/reports/dev-adaptive-h0-v2 --resume
```

完整三组各 53 题后才能运行 `devsupport/analysis/audit_adaptive_effects.py`。该工具会拒绝部分批次；未完成时使用 `report_evaluation.py` 输出注明尝试数的部分报告，不能将 H0 全 53 题与 Q 部分题的均值当配对增益。

## 已知故障与恢复条件

| 现象 | 依据与处理 |
|---|---|
| 原全局 Torch 无 sm_120 kernel | 复用已验证 WSL CUDA 环境；不升级全局 Python/Torch |
| 上传等待 HF 网络 | 本地 tokenizer SHA 校验缓存；launcher 使用 HF_HUB_OFFLINE=1 |
| 上传 worker 回调 401 | 曾缺 INTERNAL_KEY；prepare_native_secrets + launcher 修正，重启后 22:20:09 实际上传/入库/检索成功 |
| gate 512 输出截断 | 旧 smoke 失败；2048 smoke 后恢复 probe 需要 2 次调用修复引用，保留调用成本 |
| 网关 403 / rate_limit_error | 历次冷却后可恢复但仍反复出现；最近 2026-10-09 05:59 为 code 1007、无 Retry-After。保留失败行，按上文冷却与合同检查后续跑，不并发轰击或自动充值 |
| 引用有显示但没有有效回答 | 只证明检索/来源通路，按 provider trace 区分截断、身份不匹配与其他错误 |

HTTP 失败记录及对应事件的 `http_error` 只保留 0..1,000,000 的数字 `provider_error_code`、白名单 `provider_error_type` 和解析后的 `retry_after_seconds`，不保存原始响应体、headers 或错误消息。错误体最多读取 16 KiB，达到上限时保守丢弃其字段；`Retry-After` 支持整数秒数和完整 IMF-fixdate（如 `Tue, 14 Nov 2023 22:15:20 GMT`），要求正确星期与 GMT，不接受尾随内容、其他时区或旧式日期；按注入时钟计算，过去日期记为 0，超过 86400 秒或无法解析则不记录。有效等待秒数大于 0 时立即返回失败并禁止该调用的自动 transport 重试，由控制端决定等待；不会按 header 自动睡眠或恢复批次。缺失字段不表示等待为 0；403 仍不自动重试，无有效 header 时原有有界重试、pacing 与记账规则不变。

同一 provider 实例会保留最晚冷却截止时间；到期前的后续调用（包括自适应 fallback 后的 gate/生成阶段）在读取凭据或访问 transport 前返回 `provider_cooldown`，仅附剩余秒数和零尝试记录。到期后只有显式新调用才能继续。冷却状态仅存在于该实例内，不跨进程或重启持久化；控制端重启后仍须依据已保存的 trace 遵守等待时间。

共享 pacing 目前默认至少 65 秒间隔，不能作为供应商限额已确认的依据。批次工作前确认获准预算与当前限制，继续记录请求/返回模型名、实际费用与失败数。当前 v2 已通过真实 H0 UI 回答与快照引用工程复验，见 `reports/ui/native-v2-answer-success-20261009.json`；一次逻辑生成经历两次网络尝试、端到端 125.97 秒，保留 text_content 理由错误。工程通路与答案质量分别验收。日志外发前只保留已脱敏诊断。
