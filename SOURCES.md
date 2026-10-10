# 来源与许可证

本项目基于 [DocsGPT](https://github.com/arc53/DocsGPT/tree/43bc3dfd0e0fca2246e669e34aadabe03fa70a1d) 二次开发。聊天、原生文档上传、会话、基础引用和其他上游能力来自 DocsGPT。上游 MIT 许可证保留在根目录 [LICENSE](LICENSE)，固定来源记录见 [UPSTREAM_PROVENANCE.json](UPSTREAM_PROVENANCE.json)。

技术文档检索组件可使用以下独立资源；源码不分发这些模型权重或完整数据。部署者需要遵守对应许可证和模型服务条款。

| 可选资源 | 来源 | 许可证 |
|---|---|---|
| 技术文档语料 | [FreshStack corpus](https://huggingface.co/datasets/freshstack/corpus-oct-2024)、[queries](https://huggingface.co/datasets/freshstack/queries-oct-2024) | CC-BY-SA-4.0 |
| Embedding | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Apache-2.0 |
| 逐对重排 | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Apache-2.0 |
| Listwise 重排 | [Jina Reranker v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) | CC-BY-NC-4.0，商业使用需另行核对许可 |

主仓库的 MIT 许可证不改变独立资源的许可证。产品界面中的问答内容也不应替代原始资料和服务商条款。
