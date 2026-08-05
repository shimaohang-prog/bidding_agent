# 招投标机器人 Dense + BM25 混合检索说明

该方案已经接入当前项目主流程。现有 data、.env 和旧 milvus_db 保持原样；新 Schema 需要使用 builders/build_vectors.py 重新构建五类独立数据库。

## 新增功能

1. Dense 稠密向量召回：理解同义表达和描述性需求。
2. Milvus BM25：使用 jieba 中文分词召回名称、代码和关键词。
3. Milvus 元数据过滤：地区、分类、完整名称、代码、金额和日期先缩小候选域。
4. 分类内 RRF：融合 Dense 与 BM25 名次，不混合两种不可直接比较的原始分数。
5. 跨任务 RRF：合并多个语义主题、分片和分类，并统一去重。
6. DeepSeek Reranker：对最终有限候选做相关性与精确值核对。
7. 诊断输出：分别记录 Dense、BM25、过滤任务、分类命中和联网结果数量。
8. 指定网站搜索：问题中出现 URL/域名时，Tavily 强制使用该域名，
   不再被政府网站优先规则覆盖。

## 安全边界

- Planner 不能生成原始 Milvus filter。
- 所有过滤字段和操作符由 retrieval/metadata_filter.py 本地白名单控制。
- 字符串值使用 JSON 规则转义，不能注入额外 Milvus 表达式。
- 系统仍不生成 SQL，也不执行 MySQL 查询。
- news 仍只联网搜索，不创建本地向量库。

## 关键配置

- 元数据过滤字段由 retrieval/metadata_filter.py 白名单控制。
- 金额范围过滤使用 CSV 中的原始数值单位。
- Dense 分类阈值位于 common/milvus_config.py。
- HYBRID_RECALL_MULTIPLIER 默认扩大 4 倍候选后再融合。
- product 可以选择 major_category 作为子分类字段。
- Reranker 当前使用 DeepSeek；失败时保留 RRF 顺序。

## 新库构建示例

当前项目已存在 data，直接执行：

    python -m builders.build_vectors --categories enterprise tender product laws policy

如果后续确认 product 需要按一级分类拆子库，再单独执行：

    python -m builders.build_vectors --categories product --subcategory-field product=major_category --rebuild

BM25 Schema 与旧 Collection 不兼容，旧 bidding_agent.db 会保留不用；五类新库写入 milvus_db/<category>/。已有新分类库需要覆盖时才增加 --rebuild。

## 验证

    python -m unittest discover -s tests -v
    python main.py --show-plan "查询安徽省网络安全项目和相关政策"

第二条只有在完成新库构建并配置 DeepSeek/Tavily 后，才属于真实端到端验证。
