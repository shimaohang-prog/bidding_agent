# -*- coding: utf-8 -*-
"""把分类向量和联网候选整理成可追溯证据。"""

import json
import os
from collections import defaultdict

from common.retrieval_models import Candidate, RetrievalResult


CATEGORY_NAMES = {
    "enterprise": "企业信息",
    "tender": "招投标项目",
    "product": "产品信息",
    "laws": "法律法规",
    "policy": "政策文件",
    "news": "行业资讯",
    "web": "联网资料",
    "private": "私有上传文件",
}
CATEGORY_ORDER = tuple(CATEGORY_NAMES)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _source_line(candidate: Candidate) -> str:
    if candidate.source_type == "web":
        return f"URL：{candidate.metadata.get('url', candidate.source_id)}"
    if candidate.source_type == "private_document":
        return f"私有文件：{candidate.metadata.get('original_name', candidate.title)}"
    source = candidate.metadata.get("source") or candidate.title
    sub = f"；子分类：{candidate.subcategory}" if candidate.subcategory else ""
    return f"向量库：{candidate.category}{sub}；来源：{source}"


def _candidate_block(
    evidence_id: str,
    candidate: Candidate,
    max_item_chars: int,
) -> str:
    lines = [
        f"[{evidence_id}]",
        f"数据类型：{candidate.source_type}",
        f"分类：{candidate.category}",
        f"标题：{candidate.title or '无'}",
        _source_line(candidate),
    ]
    published = candidate.metadata.get("published_date")
    updated = candidate.metadata.get("updated_at")
    if published:
        lines.append(f"发布时间：{published}")
    if updated:
        lines.append(f"数据更新时间：{updated}")
    lines.extend(["内容：", candidate.content[:max_item_chars]])
    return "\n".join(lines)


def build_answer_context(
    question: str,
    retrieval_result: RetrievalResult,
) -> str:
    context, _ = build_answer_context_with_citations(question, retrieval_result)
    return context


def build_answer_context_with_citations(
    question: str,
    retrieval_result: RetrievalResult,
) -> tuple[str, list[dict]]:
    max_total_chars = max(4000, _int_env("MAX_CONTEXT_CHARS", 28000))
    max_item_chars = max(500, _int_env("MAX_CONTEXT_ITEM_CHARS", 4000))
    max_per_category = max(
        1, _int_env("MAX_CONTEXT_PER_CATEGORY", 5)
    )

    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in retrieval_result.candidates:
        if len(groups[candidate.category]) < max_per_category:
            groups[candidate.category].append(candidate)

    plan_summary = {
        "semantic_queries": [
            {
                "query": task.query,
                "categories": task.categories,
                "subcategory_hints": task.subcategory_hints,
                "metadata_filters": {
                    category: [
                        condition.model_dump()
                        for condition in conditions
                    ]
                    for category, conditions
                    in task.metadata_filters.items()
                },
            }
            for task in retrieval_result.plan.semantic_queries
        ],
        "requires_web_search": retrieval_result.plan.requires_web_search,
        "requires_fresh_data": retrieval_result.plan.requires_fresh_data,
        "answer_focus": retrieval_result.plan.answer_focus,
    }
    parts = [
        "【用户问题】",
        question,
        "",
        "【已校验检索计划】",
        json.dumps(plan_summary, ensure_ascii=False),
        "",
        "【证据说明】",
        "本地证据先经过 Milvus 元数据过滤，再由 Dense 与 BM25 双路召回、"
        "RRF 融合和统一 Reranker。enterprise、tender、product 的完整 "
        "CSV 行数据随向量保存并直接返回；系统没有执行 MySQL 查询。"
        "laws、policy 返回文档片段。"
        "news 只使用带 URL 的联网结果。",
    ]
    evidence_number = 0
    citations: list[dict] = []
    current_chars = sum(len(item) for item in parts)
    for category in CATEGORY_ORDER:
        if not groups[category]:
            continue
        header = f"\n【{CATEGORY_NAMES[category]}】"
        if current_chars + len(header) > max_total_chars:
            break
        parts.append(header)
        current_chars += len(header)
        for candidate in groups[category]:
            evidence_number += 1
            block = _candidate_block(
                f"E{evidence_number}",
                candidate,
                max_item_chars,
            )
            if current_chars + len(block) > max_total_chars:
                break
            parts.append(block)
            current_chars += len(block)
            citations.append(
                {
                    "evidence_id": f"E{evidence_number}",
                    "source_type": candidate.source_type,
                    "category": candidate.category,
                    "title": candidate.title,
                    "source_id": candidate.source_id,
                    "url": candidate.metadata.get("url"),
                    "metadata": {
                        key: value
                        for key, value in candidate.metadata.items()
                        if key not in {"db_path", "filter_expression"}
                    },
                }
            )

    if evidence_number == 0:
        parts.extend(
            [
                "",
                "【检索结论】",
                "当前没有取得可用于回答的可靠证据，不得补写业务事实。",
            ]
        )
    if retrieval_result.warnings:
        parts.extend(["", "【检索提示】"])
        parts.extend(f"- {item}" for item in retrieval_result.warnings)
    return "\n\n".join(parts), citations
