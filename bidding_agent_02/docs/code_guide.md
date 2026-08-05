# 当前 Python 文件功能

## 在线主流程

| 文件 | 功能 |
|---|---|
| main.py | 接收问题，调用 Planner、混合检索、上下文构建和答案生成。 |
| planning/retrieval_planner.py | 生成多分类语义任务、子分类提示和白名单元数据条件；显式联网或 URL 请求会强制联网。 |
| retrieval/retrieval_executor.py | 调度五类本地检索、RRF、Reranker 和按需联网。 |
| retrieval/category_vector_search.py | 每个分类独立执行 Dense 与 BM25 召回，并进行分类内 RRF。 |
| retrieval/metadata_filter.py | 把字段、操作符和值安全编译为 Milvus 元数据表达式，拒绝原始表达式。 |
| retrieval/web_search.py | 调用 Tavily；用户指定域名时优先使用该域名，不再套用政府网站限制。 |
| ranking/result_fusion.py | 跨语义任务融合、去重，不直接比较 Dense 与 BM25 原始分数。 |
| ranking/reranker.py | 使用 DeepSeek 统一重排；失败时保留 RRF 顺序并记录 warning。 |
| generation/context_builder.py | 组织分类证据、过滤条件和来源。 |
| generation/answer_generator.py | 只依据检索证据生成答案。 |

## 公共模块

| 文件 | 功能 |
|---|---|
| common/milvus_config.py | 六分类、五类本地数据库路径、阈值、Top-K 和 RRF 配置。 |
| common/embedding.py | 延迟加载 BGE Embedding，生成 Dense 向量。 |
| common/llm_client.py | 统一封装 DeepSeek Function Calling 和回答请求。 |
| common/retrieval_models.py | 定义语义任务、元数据条件、候选证据和检索结果协议。 |

## 离线构建

| 文件 | 功能 |
|---|---|
| builders/build_vectors.py | 从三类业务 CSV 和 laws/policy TXT 构建 Dense + BM25 新 Schema；news 只保留目录。 |

## 历史文件

以下文件为旧 MySQL/双 Collection 流程，主入口已经不再导入：

- retrieval/mysql_query.py
- retrieval/business_vector_search.py
- retrieval/knowledge_vector_search.py
- builders/business_vector_builder.py
- builders/build_vector.py

这些文件暂时保留，避免直接删除历史代码；不要再作为当前主流程运行。

## 测试

| 文件 | 功能 |
|---|---|
| tests/test_hybrid_retrieval.py | 验证元数据白名单、防注入、Dense/BM25 双路名次、显式域名联网。 |
| tests/test_six_category_architecture.py | 验证六分类、news 联网边界、物理数据库隔离和 RRF 去重。 |
