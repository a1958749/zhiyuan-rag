# 来源、版本与交付边界

以下记录对应本地已下载文件及其 manifest，供复现和展示时核对。模型权重、完整语料、账号凭证与运行数据库不放入源码交付包。

| 组件 | 固定来源 | 本地声明的许可证 | 核对位置 |
|---|---|---|---|
| DocsGPT | [arc53/DocsGPT，43bc3df](https://github.com/arc53/DocsGPT/tree/43bc3dfd0e0fca2246e669e34aadabe03fa70a1d) | MIT | 根目录 LICENSE、UPSTREAM_PROVENANCE.json |
| FreshStack corpus | [corpus-oct-2024，069f66d](https://huggingface.co/datasets/freshstack/corpus-oct-2024/tree/069f66dc323e163b48b10d08408d282733d4393b) | CC-BY-SA-4.0 | reports/data-manifest.json |
| FreshStack queries | [queries-oct-2024，023ac3a](https://huggingface.co/datasets/freshstack/queries-oct-2024/tree/023ac3a14caf9d6ebb13d01adcfb8fa05e5a9630) | CC-BY-SA-4.0 | reports/data-manifest.json |
| Embedding | [Qwen3-Embedding-0.6B，97b0c61](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) | Apache-2.0 | model-cache/embedding/README.md、artifact-manifest.json |
| Pairwise reranker | [Qwen3-Reranker-0.6B，e61197e](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/e61197ed45024b0ed8a2d74b80b4d909f1255473) | Apache-2.0 | model-cache/pairwise/README.md、artifact-manifest.json |
| Listwise reranker | [jina-reranker-v3.5，e8a93f3](https://huggingface.co/jinaai/jina-reranker-v3.5/tree/e8a93f33f0b22108f8c2364f8484ce3422552fbc) | CC-BY-NC-4.0 | model-cache/listwise/README.md、artifact-manifest.json |

DocsGPT 通过官方 commit archive 导入。本地首提交 `164c9037d62cb200ae0193ef349d9956de7ff3b7` 是导入提交，不能写成上游 GitHub 提交。原压缩包 SHA256 为 `b73a27379f51d252dc3ef38ce621fe49b9e31a14d99e57418d8965a5eada0eed`。

数据的原始行、重复 ID 审计和选中行均保留；报告中的原文、引用和题目仍应保留来源归属，不能统一改称项目原创内容。Jina 模型卡明确标注非商业许可证，并提供商业使用咨询链接；项目中保留其独立实验身份，不把它描述为与 MIT 主仓库具有相同许可证。

生成回答使用用户授权网关的固定路由。请求名 `deepseek-v4-pro` 与返回名 `claude-opus-4-8` 已记录，底层模型身份未验证；这部分没有可审计的本地模型权重版本，复现报告须保留该限制。

最终交付按清单选择源码、文档、数据/模型 manifest、实验汇总和逐题结果。排除 `.env`、`.secrets`、账户截图、完整运行日志、数据库、模型权重、索引及历史原始网关探测响应。不会将这些排除项直接打包进求职展示材料。

## 索引资源与成本观测

2026-10-10 对现有两个完整索引进行零模型调用检查，逐份绑定 manifest SHA256，并统计 manifest 列出的十五个实际文件。没有重建索引，资源报告不替代文件内容完整性核验。

| 索引 | 官方唯一 ID | Embedding 记录耗时 | 索引文件大小 | 向量化输入被截断的块 |
|---|---:|---:|---:|---:|
| raw | 49,505 | 2,038.15 秒（33.97 分钟） | 861,643,715 bytes（0.8025 GiB） | 5,995 |
| contextual | 49,505 | 1,850.85 秒（30.85 分钟） | 875,491,461 bytes（0.8154 GiB） | 7,327 |

两份均使用 CUDA、bfloat16、1,024 维向量及 2,048 输入-token 上限，记录的 OOM 重试数为零。截断影响索引表示，原始证据文本没有被替换。元数据前缀增加截断块数量，因此不能仅报告其覆盖收益而忽略表示损失。

耗时只覆盖记录的 embedding 阶段，不含未记录的数据准备、词法索引、下载或完整构建墙钟时间；两个独立运行的耗时差异不能声称前缀让计算更快。文件大小不含权重、manifest 本身或另存的原生 U 索引。Embedding 使用本地 GPU，无付费 embedding API 调用；本地电费、硬件摊销和完整索引构建货币成本未测，保持 null，不能写作零成本。依据为 `reports/index-resource-observation-20261010-v1.json`。
