"""异步复用现有 Planner、Dense+BM25、RRF 与 Reranker 领域规则。"""

from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

import anyio

from common.milvus_config import FINAL_CANDIDATE_LIMIT, RERANK_CANDIDATE_LIMIT, VECTOR_CATEGORIES
from common.embedding import encode_query
from common.retrieval_models import RetrievalPlan, RetrievalResult, SemanticQueryTask
from planning.retrieval_planner import create_retrieval_plan
from ranking.reranker import rerank_candidates
from ranking.result_fusion import normalise_vector_results, normalise_web_results, reciprocal_rank_fusion
from retrieval.category_vector_search import (
    category_vector_search_multi,
    exact_tender_title_search,
)
from retrieval.retrieval_executor import should_use_web_search

from backend.services.private_documents import PrivateDocumentStore
from backend.services.tavily_client import AsyncTavilyClient


StatusCallback = Callable[[str], Awaitable[None]]


class AsyncRAGService:
    def __init__(self, *, limiter: anyio.CapacityLimiter, tavily: AsyncTavilyClient, private_store: PrivateDocumentStore) -> None:
        self.limiter = limiter
        self.tavily = tavily
        self.private_store = private_store

    @staticmethod
    def _deduplicate_tasks(plan: RetrievalPlan) -> list[tuple[int, SemanticQueryTask]]:
        merged: dict[str, tuple[int, SemanticQueryTask]] = {}
        for index, task in enumerate(plan.semantic_queries):
            key = task.query.casefold()
            if key not in merged:
                merged[key] = (index, task.model_copy(deep=True))
                continue
            _, current = merged[key]
            current.categories = list(dict.fromkeys([*current.categories, *task.categories]))
            current.top_k_per_category = max(current.top_k_per_category, task.top_k_per_category)
            for category, labels in task.subcategory_hints.items():
                current.subcategory_hints[category] = list(dict.fromkeys([*current.subcategory_hints.get(category, []), *labels]))
            for category, conditions in task.metadata_filters.items():
                current.metadata_filters[category] = [*current.metadata_filters.get(category, []), *conditions]
        return list(merged.values())

    async def _local_hits(self, plan: RetrievalPlan) -> tuple[list[dict[str, Any]], list[str]]:
        hits: list[dict[str, Any]] = []
        warnings: list[str] = []
        lock = anyio.Lock()
        # 单个复杂问题最多占两个同步检索槽，避免它把全局线程额度
        # 全部占满，使其他用户或会话只能等该问题检索结束。
        request_search_slots = anyio.CapacityLimiter(2)

        async def run(index: int, task: SemanticQueryTask) -> None:
            categories = [item for item in task.categories if item in VECTOR_CATEGORIES]
            if not categories:
                return
            # 相同 semantic_query 只编码一次；各分类使用同一向量并发查询。
            query_vector = await anyio.to_thread.run_sync(
                partial(encode_query, task.query), limiter=self.limiter
            )
            result: list[dict[str, Any]] = []
            task_warnings: list[str] = []
            category_lock = anyio.Lock()

            async def run_category(category: str) -> None:
                async with request_search_slots:
                    category_result, category_warnings = await anyio.to_thread.run_sync(
                        partial(
                            category_vector_search_multi, task.query, categories=[category],
                            subcategory_hints=task.subcategory_hints, metadata_filters=task.metadata_filters,
                            top_k_per_category=task.top_k_per_category, query_vector=query_vector,
                        ), limiter=self.limiter,
                    )
                async with category_lock:
                    result.extend(category_result)
                    task_warnings.extend(category_warnings)

            async with anyio.create_task_group() as category_group:
                for category in categories:
                    category_group.start_soon(run_category, category)
            async with lock:
                warnings.extend(task_warnings)
                hits.extend({"query_index": index, "semantic_query": task.query, "hit": item} for item in result)

        async with anyio.create_task_group() as group:
            for index, task in self._deduplicate_tasks(plan):
                group.start_soon(run, index, task)
        return hits, warnings

    async def retrieve(
        self, question: str, *, user_id: str, conversation_id: str,
        file_ids: list[str], status: StatusCallback,
    ) -> RetrievalResult:
        plan = await anyio.to_thread.run_sync(partial(create_retrieval_plan, question), limiter=self.limiter)
        warnings = [f"Planner 已降级：{plan.planner_warning}"] if plan.planner_warning else []
        categories = list(dict.fromkeys(category for task in plan.semantic_queries for category in task.categories))
        explicit_web = plan.requires_web_search or plan.requires_fresh_data or "news" in categories
        local_result: tuple[list[dict[str, Any]], list[str]] = ([], [])
        web_hits: list[dict[str, Any]] = []
        private_hits: list = []

        await status("retrieving")

        # Run the deterministic exact-title lookup first and close its Milvus
        # client before ordinary parallel retrieval starts. This both protects
        # complete-title hits from planner drift and avoids lock contention.
        exact_hits, exact_warnings = await anyio.to_thread.run_sync(
            partial(exact_tender_title_search, question),
            limiter=self.limiter,
        )
        warnings.extend(exact_warnings)
        exact_candidates = normalise_vector_results(
            {
                "query_index": -1,
                "semantic_query": question,
                "hit": hit,
            }
            for hit in exact_hits
        )

        async def local() -> None:
            nonlocal local_result
            local_result = await self._local_hits(plan)

        async def private() -> None:
            nonlocal private_hits
            private_hits = await self.private_store.search(question, user_id, conversation_id, file_ids)

        async def web() -> None:
            nonlocal web_hits
            try:
                web_hits = await self.tavily.search(question, categories)
            except Exception as exc:
                warnings.append(f"Tavily 联网搜索失败：{exc}")

        async with anyio.create_task_group() as group:
            group.start_soon(local)
            group.start_soon(private)
            if explicit_web:
                group.start_soon(web)

        raw_hits, local_warnings = local_result
        warnings.extend(local_warnings)
        await status("reranking")
        candidates = [
            *exact_candidates,
            *normalise_vector_results(raw_hits),
            *private_hits,
        ]
        fused = reciprocal_rank_fusion(candidates, limit=RERANK_CANDIDATE_LIMIT)
        ranked, rerank_warning = await anyio.to_thread.run_sync(
            partial(rerank_candidates, question, fused, top_n=FINAL_CANDIDATE_LIMIT), limiter=self.limiter
        )
        if rerank_warning:
            warnings.append(f"本地候选 LLM 重排失败，保留 RRF 顺序：{rerank_warning}")

        use_web, reason = should_use_web_search(plan, ranked)
        if use_web and not explicit_web:
            await web()
        if web_hits:
            combined = reciprocal_rank_fusion([*ranked, *normalise_web_results(web_hits)], limit=RERANK_CANDIDATE_LIMIT)
            ranked, second_warning = await anyio.to_thread.run_sync(
                partial(rerank_candidates, question, combined, top_n=FINAL_CANDIDATE_LIMIT), limiter=self.limiter
            )
            if second_warning:
                warnings.append(f"联网候选重排失败，保留 RRF 顺序：{second_warning}")

        return RetrievalResult(
            plan=plan, candidates=ranked, used_web_search=bool(web_hits), warnings=warnings,
            diagnostics={
                "mysql_query_count": 0, "vector_hit_count": len(raw_hits),
                "exact_title_hit_count": len(exact_candidates),
                "private_hit_count": len(private_hits), "web_result_count": len(web_hits),
                "web_decision_reason": reason,
            },
        )
