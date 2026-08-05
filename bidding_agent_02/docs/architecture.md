# 六分类混合检索架构

## 实施边界

当前主流程已经取消 MySQL 查询，并切换到五类本地独立向量库。真实 data、
.env 和旧 milvus_db/bidding_agent.db 保持不变。因为 BM25 Function、稀疏
向量字段和中文 analyzer 必须在 Collection 创建时定义，需要使用
builders/build_vectors.py 构建新库，不能直接给旧 Collection 补字段。

## 六个大分类

enterprise、tender、product、laws、policy 分别使用物理隔离的 Milvus Lite
数据库。news 保留同级目录，但当前只调用联网搜索。

每个本地分类数据库包含：

| 字段 | 作用 |
|---|---|
| dense_vector | BGE 稠密向量，负责语义相似召回 |
| searchable_text | Dense 与 BM25 共用的原始检索文本 |
| sparse_vector | Milvus BM25 Function 自动生成的稀疏向量 |
| metadata | 完整业务行或文档元数据，用于安全过滤和回答 |
| category / subcategory | 大分类和可选子分类 |
| source_id / source / title | 去重和来源追踪 |
| content | 最终交给 Reranker 和答案模型的证据正文 |

## 在线检索流程

1. Planner 把问题拆成多个语义任务。
2. Planner 只能输出白名单字段、eq/in/gte/lte 和值，不能输出 Milvus 表达式。
3. 本地 metadata_filter.py 校验字段并安全转义，生成 Milvus 过滤表达式。
4. 同一过滤条件同时作用于 Dense 和 BM25。
5. Dense 使用 COSINE，先按大分类独立阈值过滤。
6. BM25 使用 jieba 中文分词，负责完整名称、代码和关键词召回。
7. 两路只使用名次做分类内 RRF，不直接比较 COSINE 与 BM25 原始分数。
8. 多语义任务再次按名次融合并按 category + source_id 去重。
9. DeepSeek Reranker 使用统一标准判断证据是否直接支持问题。
10. news、显式联网、时效问题、本地无结果或重排相关性不足时调用 Tavily。
11. 用户给出 URL/域名时，Tavily 使用该域名作为 include_domains，
    跳过政府网站优先限制。

## 元数据过滤白名单

| 分类 | 主要字段 |
|---|---|
| enterprise | enterprise_name、uscc、corporation、province、city、district、industry、enterprise_type、status、event_time、registered_capital_amount |
| tender | tender_title、project_type、source_name、province、city、town、purchasing_staff、bid_company、event_time、bid_date、bid_amount |
| product | title、major_category、middle_category、supplier_name、currency、province、city、event_time、amount |
| laws / policy | title、subcategory、source、updated_at |

金额字段在建库时转换为数值，支持 gte/lte。其他完整 CSV 字段仍保留在
metadata 中用于回答，但没有进入过滤白名单的字段不能由 Planner 直接过滤。

## 三层重排

- 第一层：每个分类内部对 Dense 和 BM25 使用 RRF。
- 第二层：对多个语义任务、分类分片和重复记录再次使用 RRF。
- 第三层：对有限候选使用 DeepSeek Reranker，核对名称、代码、金额和日期。

如果 DeepSeek Reranker 不可用，系统保留第二层 RRF 顺序，并在 warnings 中
明确记录降级，不会把检索分数当作业务事实。

## 子分类

未拆子分类时使用 main.db。启用子分类后，每个子类写入
大分类/subcategories/独立数据库。没有子分类提示时检索全部已有子库；有明确
提示时只打开对应子库。大分类阈值、BM25、元数据过滤和 Reranker 规则保持
一致。
