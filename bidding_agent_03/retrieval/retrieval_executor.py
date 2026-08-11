# -*- coding: utf-8 -*-
"""执行六分类向量检索、统一重排和按需联网。"""

from typing import Any

from common.milvus_config import (
    FINAL_CANDIDATE_LIMIT,
    RERANK_CANDIDATE_LIMIT,
    RERANK_WEB_THRESHOLD,
    VECTOR_CATEGORIES,
)
from common.retrieval_models import RetrievalPlan, RetrievalResult
from ranking.reranker import rerank_candidates
from ranking.result_fusion import (
    normalise_vector_results,
    normalise_web_results,
    reciprocal_rank_fusion,
)
from retrieval.category_vector_search import (
    category_vector_search_multi,
    exact_tender_title_search,
)
from retrieval.web_search import web_search


def _execute_semantic_tasks(
    plan: RetrievalPlan,
    warnings: list[str],
) -> list[dict[str, Any]]:
    hits_with_context: list[dict[str, Any]] = []
    for query_index, task in enumerate(plan.semantic_queries):
        vector_categories = [
            category
            for category in task.categories
            if category in VECTOR_CATEGORIES
        ]
        if not vector_categories:
            continue
        try:
            hits, search_warnings = category_vector_search_multi(
                task.query,
                categories=vector_categories,
                subcategory_hints=task.subcategory_hints,
                metadata_filters=task.metadata_filters,
                top_k_per_category=task.top_k_per_category,
            )
            warnings.extend(search_warnings)
            hits_with_context.extend(
                {
                    "query_index": query_index,
                    "semantic_query": task.query,
                    "hit": hit,
                }
                for hit in hits
            )
        except Exception as exc:
            warnings.append(f"语义任务 {query_index} 检索失败：{exc}")
    return hits_with_context


def should_use_web_search(
    plan: RetrievalPlan,
    candidates,
) -> tuple[bool, str]:
    if plan.requires_web_search:
        return True, "用户明确要求联网或指定网站搜索"
    news_requested = any(
        "news" in task.categories for task in plan.semantic_queries
    )
    if plan.requires_fresh_data or news_requested:
        return True, "问题要求最新信息或包含 news 分类"
    if not candidates:
        return True, "五类本地向量库均无有效候选"
    judged_scores = [
        item.rerank_score
        for item in candidates
        if item.rerank_score is not None
    ]
    if judged_scores and max(judged_scores) < RERANK_WEB_THRESHOLD:
        return True, "统一重排后的最高相关性不足"
    return False, "本地分类向量证据达到阈值"


def execute_retrieval_plan(
    question: str,
    plan: RetrievalPlan,
) -> RetrievalResult:
    warnings: list[str] = []
    if plan.planner_warning:
        warnings.append(f"Planner 已降级：{plan.planner_warning}")

    # The CLI entry point uses this synchronous executor rather than the web
    # service. Run the same planner-independent exact-title lookup first so a
    # complete tender title cannot be lost to semantic query rewriting.
    exact_hits, exact_warnings = exact_tender_title_search(question)
    warnings.extend(exact_warnings)
    exact_hits_with_context = [
        {
            "query_index": -1,
            "semantic_query": question,
            "hit": hit,
        }
        for hit in exact_hits
    ]
    raw_vector_hits = _execute_semantic_tasks(plan, warnings)
    local_candidates = normalise_vector_results(
        [*exact_hits_with_context, *raw_vector_hits]
    )
    fused = reciprocal_rank_fusion(
        local_candidates,
        limit=RERANK_CANDIDATE_LIMIT,
    )
    ranked, rerank_warning = rerank_candidates(
        question,
        fused,
        top_n=FINAL_CANDIDATE_LIMIT,
    )
    if rerank_warning:
        warnings.append(
            f"本地候选 LLM 重排失败，保留 RRF 顺序：{rerank_warning}"
        )

    use_web, web_reason = should_use_web_search(plan, ranked)
    web_hits: list[dict[str, Any]] = []
    if use_web:
        categories = list(
            dict.fromkeys(
                category
                for task in plan.semantic_queries
                for category in task.categories
            )
        )
        try:
            web_hits = web_search(
                question,
                categories=categories,
                max_results=6,
            )
        except Exception as exc:
            warnings.append(f"Tavily 联网搜索失败：{exc}")

    if web_hits:
        fused_with_web = reciprocal_rank_fusion(
            [*ranked, *normalise_web_results(web_hits)],
            limit=RERANK_CANDIDATE_LIMIT,
        )
        ranked, second_warning = rerank_candidates(
            question,
            fused_with_web,
            top_n=FINAL_CANDIDATE_LIMIT,
        )
        if second_warning:
            warnings.append(
                f"联网候选重排失败，保留 RRF 顺序：{second_warning}"
            )

    per_category: dict[str, int] = {}
    dense_hit_count = 0
    bm25_hit_count = 0
    filtered_task_count = 0
    for item in raw_vector_hits:
        category = item["hit"]["category"]
        per_category[category] = per_category.get(category, 0) + 1
        routes = item["hit"].get("route_positions", {})
        dense_hit_count += int(
            any(name.startswith("dense:") for name in routes)
        )
        bm25_hit_count += int(
            any(name.startswith("bm25:") for name in routes)
        )
    filtered_task_count = sum(
        1
        for task in plan.semantic_queries
        if task.metadata_filters
    )

    return RetrievalResult(
        plan=plan,
        candidates=ranked,
        used_web_search=bool(web_hits),
        warnings=warnings,
        diagnostics={
            "mysql_query_count": 0,
            "vector_hit_count": len(raw_vector_hits),
            "exact_title_hit_count": len(exact_hits),
            "dense_hit_count": dense_hit_count,
            "bm25_hit_count": bm25_hit_count,
            "metadata_filtered_task_count": filtered_task_count,
            "vector_hits_by_category": per_category,
            "web_result_count": len(web_hits),
            "web_decision_reason": web_reason,
        },
    )
