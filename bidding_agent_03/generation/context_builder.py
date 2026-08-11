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
DIVERSITY_CATEGORIES = frozenset({"laws", "policy", "private"})


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


def _document_key(candidate: Candidate) -> str:
    """Return a document-level key rather than a chunk-level source ID."""
    payload = candidate.metadata.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    if candidate.source_type == "private_document":
        file_id = candidate.metadata.get("file_id")
        if file_id:
            return f"private:{file_id}"
        try:
            file_id, _ = candidate.source_id.rsplit(":", 1)
            return f"private:{file_id}"
        except ValueError:
            return candidate.identity_key

    source = candidate.metadata.get("source") or payload.get("source")
    if source:
        return f"{candidate.category}:{source}"
    try:
        source, _ = candidate.source_id.rsplit("#", 1)
        return f"{candidate.category}:{source}"
    except ValueError:
        return candidate.identity_key


def _select_diverse_candidates(
    candidates: list[Candidate],
    *,
    max_items: int,
    max_per_source: int,
) -> list[Candidate]:
    """Prefer source diversity, then backfill in original rerank order."""
    selected: list[Candidate] = []
    deferred: list[Candidate] = []
    source_counts: dict[str, int] = defaultdict(int)

    for candidate in candidates:
        if candidate.category not in DIVERSITY_CATEGORIES:
            selected.append(candidate)
        else:
            source_key = _document_key(candidate)
            if (
                candidate.exact_title_match
                or source_counts[source_key] < max_per_source
            ):
                selected.append(candidate)
                source_counts[source_key] += 1
            else:
                deferred.append(candidate)
        if len(selected) >= max_items:
            return selected[:max_items]

    for candidate in deferred:
        selected.append(candidate)
        if len(selected) >= max_items:
            break
    return selected


def _chunk_location(candidate: Candidate) -> tuple[str, int] | None:
    """Return the document key and chunk number when both are available."""
    payload = candidate.metadata.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    source = (
        candidate.metadata.get("file_id")
        or candidate.metadata.get("source")
        or payload.get("source")
    )
    chunk_id = payload.get("chunk_id")
    if chunk_id is None:
        separator = (
            ":" if candidate.source_type == "private_document" else "#"
        )
        try:
            parsed_source, parsed_chunk_id = candidate.source_id.rsplit(
                separator,
                1,
            )
            chunk_id = int(parsed_chunk_id)
            source = source or parsed_source
        except (TypeError, ValueError):
            return None

    try:
        chunk_number = int(chunk_id)
    except (TypeError, ValueError):
        return None
    if not source:
        return None
    return str(source), chunk_number


def _trim_pair_overlap(
    reference: str,
    current: str,
    *,
    min_overlap: int = 40,
    max_overlap: int = 400,
) -> str:
    """Trim an exact shared boundary without semantic text deletion."""
    limit = min(len(reference), len(current), max_overlap)
    for length in range(limit, min_overlap - 1, -1):
        if reference[-length:] == current[:length]:
            return current[length:].lstrip()
        if current[-length:] == reference[:length]:
            return current[:-length].rstrip()
    return current


def _trim_adjacent_chunk_overlap(
    candidate: Candidate,
    selected: list[Candidate],
) -> str:
    """Trim overlap only for adjacent chunks from the same document."""
    location = _chunk_location(candidate)
    if location is None:
        return candidate.content

    source, chunk_number = location
    content = candidate.content
    for previous in selected:
        if previous.category != candidate.category:
            continue
        previous_location = _chunk_location(previous)
        if previous_location is None:
            continue
        previous_source, previous_chunk_number = previous_location
        if (
            previous_source == source
            and abs(previous_chunk_number - chunk_number) == 1
        ):
            content = _trim_pair_overlap(previous.content, content)
    return content


def _candidate_block(
    evidence_id: str,
    candidate: Candidate,
    max_item_chars: int,
    *,
    display_content: str | None = None,
) -> str:
    content = (
        candidate.content if display_content is None else display_content
    )
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
    lines.extend(["内容：", content[:max_item_chars]])
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
    max_per_source = max(1, _int_env("MAX_CONTEXT_PER_SOURCE", 2))

    ranked_groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in retrieval_result.candidates:
        ranked_groups[candidate.category].append(candidate)
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for category, candidates in ranked_groups.items():
        groups[category] = _select_diverse_candidates(
            candidates,
            max_items=max_per_category,
            max_per_source=max_per_source,
        )

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
    selected_candidates: list[Candidate] = []
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
            display_content = _trim_adjacent_chunk_overlap(
                candidate,
                selected_candidates,
            )
            block = _candidate_block(
                f"E{evidence_number}",
                candidate,
                max_item_chars,
                display_content=display_content,
            )
            if current_chars + len(block) > max_total_chars:
                break
            parts.append(block)
            current_chars += len(block)
            selected_candidates.append(candidate)
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
