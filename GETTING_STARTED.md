# 第一次使用知源

知源是一个支持自定义文档的个人 RAG 知识助手。AI 开发文档只是演示材料，你可以换成自己的笔记、手册或项目文档。产品介绍、使用方法和能力限制由固定文本提供，不依赖模型生成。

下载的是**源码**，不是自带模型、数据库和 API 密钥的桌面安装包。以下先启动普通文档问答；复现 FreshStack 优化实验另见[实验运行手册](devsupport/docs/RUNBOOK.md)。两条路径不需要同时完成。

## 已经能打开网页？直接从这里开始

1. 打开主页，点击 **管理 / 上传知识库**。也可在 **设置 → 知识库** 打开上传入口。
2. 第一次先用小 TXT 或 Markdown 文件。可以在网页 **使用指南** 下载示例文件，也可使用仓库内的 [示例 TXT](frontend/public/examples/zhiyuan-first-document.txt)。不要先用扫描 PDF 测试最小流程。
3. 上传原始文件，保留默认分块参数，等待处理完成或可用。系统自动解析、切分、生成 embedding 并建立索引；上传成功通知不等于已经可检索。
4. 回到聊天页，通过输入框旁的知识来源选择器，**选中刚上传的知识库**。
5. 提问：“星河项目由谁负责演示，什么时候开始？请引用原文。”检查答案是否含“小林”和“周五下午三点”，并打开引用核验。这样才完成最小的文档问答验证。

界面默认简体中文；在 **设置 → 常规设置 → 语言** 切换 English。主页和侧栏帮助菜单都能打开固定的 **使用指南**，不必询问 AI 如何使用产品。

## 从源码启动普通文档问答

需要 Python 3.12、Node.js 22、uv 和 Docker（或已有 PostgreSQL、Redis）。以下命令在**解压后的仓库根目录**执行。源码启动流程按仓库 CLI 与配置核对；本次未在全新机器上再次执行整套安装，现有 Windows + WSL 运行记录见 RUNBOOK。

### 1. 安装依赖

```powershell
uv sync
Set-Location frontend
npm ci --include=dev
Set-Location ..
```

macOS / Linux 将 `Set-Location` 改为 `cd`。

### 2. 准备数据库与配置

新环境可使用开发 Compose（只启动数据库和 Redis，不替换本项目 UI）：

```powershell
docker compose -f deployment/docker-compose-dev.yaml up -d
Copy-Item .env-template .env
```

**已有 `.env` 时不要覆盖**。macOS / Linux 对应复制命令是 `cp .env-template .env`。已有数据库、Redis 可以直接复用，把连接地址改成自己的；不要重复占用端口。

打开 `.env`，配置以下项。尖括号内容需要自己填写，不能原样运行：

```dotenv
POSTGRES_URI=postgresql://docsgpt:docsgpt@localhost:5432/docsgpt
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
LLM_PROVIDER=openai
LLM_NAME=<你的模型服务支持的模型名>
API_KEY=<你自己的模型服务密钥>
INTERNAL_KEY=<生成的一串随机值，API和worker使用同一个值>
```

如使用 OpenAI 兼容服务，再配置 `OPENAI_BASE_URL=<服务商提供的接口地址>`；模型名、接口地址和密钥必须来自同一家服务。未设置兼容接口时，云模型名还须存在于 `docsgpt/core/models/` 对应的 YAML 注册表，或自己的模型配置中，不能随便填写。内部密钥可用 `uv run python -c "import secrets; print(secrets.token_hex(32))"` 在本机生成，填入 `.env`。不要提交 `.env` 或密钥。

普通启动会按原生配置准备 embedding 模型，首次下载需要网络和时间。没有自己的回答模型时，可以临时使用下一步的 `--mock-llm` 检查界面连通性；mock 输出不能作为真实文档问答或效果验证。

### 3. 配置前端的本地 API 地址

在仓库根目录执行（已有 `frontend/.env.local` 时不要覆盖，打开后核对地址）：

```powershell
Copy-Item frontend/.env.example frontend/.env.local
```

macOS / Linux 对应 `cp frontend/.env.example frontend/.env.local`。模板指向 `http://localhost:7091`；自定义API端口时同时修改 `VITE_API_HOST` 与 `VITE_BASE_URL`。这个步骤不能省略：源码开发页面不会自动从后端 `.env` 获取API地址，未配置时部分请求会使用上游云端地址或空地址。前端配置只存公开地址和功能开关，**不要放模型密钥**。

### 4. 启动服务与页面

```powershell
uv run python -m docsgpt dev --ui
```

保持这个终端运行，打开 **http://localhost:5173**。API 默认在 **http://localhost:7091**；本地 Vite 页面是 5173。普通知识查询需要 worker，不能只开一个前端页面。

Windows 再在第二个终端、同一仓库根目录启动调度器：

```powershell
uv run python -m docsgpt beat
```

无付费生成的界面连通性检查用 `uv run python -m docsgpt dev --ui --mock-llm`；完成后切回真实配置，才能检验回答质量。修改前端 `.env.local` 后重启开发命令，页面才会读取新地址。

不要用上游已构建的远程镜像替代本仓库源码 UI，否则会看不到知源的名称、语言选择和使用指南。

## 常见问题

| 现象 | 先检查 |
|---|---|
| 上传了却答不出 | 处理是否完成、聊天中是否选中正确来源、原文是否含答案 |
| 文档入库失败 | worker 是否运行、INTERNAL_KEY 是否一致、文件是否能提取文字 |
| 模型连接失败 | API 地址、模型名、密钥、额度和服务商状态 |
| 页面连接不到本地 API | 是否复制了frontend/.env.example、两个API地址与实际端口是否一致、修改后是否重启 |
| 知识库为空 | 自己的文件、数据库、索引没有随源码打包，需要重新添加 |
| 想更换知识主题 | 新建来源并上传即可；不需要重新训练大模型 |
| 想改分块方式 | 来源设置中修改；重新索引后生效 |
| 找不到中文 / English 切换 | 设置 → 常规设置 → 语言 |

## 使用边界

- 模型回答可能出错；引用也需要核验，不能仅凭出现引用就认为结论正确。
- 使用云端回答模型时，问题和检索到的文档片段会发送给配置的服务；远程 embedding 也会发送文本。本地保存文档不代表所有处理均离线。
- AI 开发演示资料是 2024-10 快照；检索实验数字不代表最新文档支持，也不代表任意新知识库都有同等提升。
- 普通知识库上传和固定 FreshStack 优化实验是不同的数据入口；普通启动不需要先下载整个评测集。
- 当前版本不会自动执行修复命令。完整 TEST 回答、复检和门控评测未完成；人工验证仍为 0。

## English quick guide

This download contains source code, not a standalone desktop installer. Install Python 3.12, Node.js 22, uv and Docker, then follow the dependency, database and `.env` steps above with your own model credentials. Copy **frontend/.env.example** to **frontend/.env.local** without overwriting an existing configuration; the frontend needs both API URLs to point to your local backend. Never put model keys in frontend variables. Start `uv run python -m docsgpt dev --ui`, open **http://localhost:5173**, and keep the worker running. On Windows, run `uv run python -m docsgpt beat` in a second terminal.

In **Settings → General → Language**, choose **English**. Open **User guide**, download the sample TXT, upload it in **Settings → Knowledge**, wait for indexing, then select that source in the chat input. Ask who owns the Star River demo and when it starts; check for **Lin**, **Friday at 3 pm** and a citation you can open. Product help is fixed text and does not call an AI model.
