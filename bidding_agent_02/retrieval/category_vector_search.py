# -*- coding: utf-8 -*-
"""Dense + BM25 双路召回、元数据过滤和分类内 RRF。"""

from pathlib import Path
from typing import Any, Iterable

from common.embedding import encode_query
from common.milvus_config import (
    CATEGORY_THRESHOLDS,
    COLLECTION_NAME,
    HYBRID_RECALL_MULTIPLIER,
    HYBRID_RRF_K,
    TOP_K_PER_CATEGORY,
    VECTOR_CATEGORIES,
    close_milvus_client,
    get_milvus_client,
    iter_existing_shards,
)
from common.retrieval_models import MetadataCondition
from retrieval.metadata_filter import compile_metadata_filter


OUTPUT_FIELDS = [
    "category",
    "subcategory",
    "source_id",
    "title",
    "content",
    "source",
    "metadata",
    "updated_at",
]


def _clean_categories(categories: Iterable[str] | None) -> list[str]:
    values = list(categories or VECTOR_CATEGORIES)
    invalid = [item for item in values if item not in VECTOR_CATEGORIES]
    if invalid:
        raise ValueError(f"以下分类没有本地向量库：{invalid}")
    return list(dict.fromkeys(values))


def _search_one_shard(
    db_path: Path,
    semantic_query: str,
    query_vector: list[float],
    limit: int,
    filter_expression: str,
) -> dict[str, list[dict[str, Any]]]:
    """对同一分片分别执行 Dense 与 BM25，保留两路独立排名。"""
    client = get_milvus_client(db_path)
    try:
        if not client.has_collection(collection_name=COLLECTION_NAME):
            return {"dense": [], "bm25": []}
        client.load_collection(collection_name=COLLECTION_NAME)
        common = {
            "collection_name": COLLECTION_NAME,
            "limit": limit,
            "filter": filter_expression,
            "output_fields": OUTPUT_FIELDS,
        }
        dense_result = client.search(
            **common,
            data=[query_vector],
            anns_field="dense_vector",
            search_params={
                "metric_type": "COSINE",
                "params": {},
            },
        )
        bm25_result = client.search(
            **common,
            data=[semantic_query],
            anns_field="sparse_vector",
            search_params={
                "metric_type": "BM25",
                "params": {},
            },
        )
        return {
            "dense": (
                dense_result[0]
                if dense_result and dense_result[0]
                else []
            ),
            "bm25": (
                bm25_result[0]
                if bm25_result and bm25_result[0]
                else []
            ),
        }
    finally:
        close_milvus_client(client)


def _base_hit(
    category: str,
    hinted_subcategory: str | None,
    db_path: Path,
    item: dict[str, Any],
) -> dict[str, Any]:
    entity = item.get("entity", {}) or {}
    stored_subcategory = str(
        entity.get("subcategory", "") or ""
    ).strip()
    metadata = entity.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    return {
        "category": category,
        "subcategory": (
            stored_subcategory or hinted_subcategory or None
        ),
        "source_id": str(entity.get("source_id", item.get("id", ""))),
        "title": str(entity.get("title", "") or ""),
        "content": str(entity.get("content", "") or ""),
        "source": str(entity.get("source", "") or ""),
        "payload": metadata,
        "updated_at": entity.get("updated_at"),
        "threshold": CATEGORY_THRESHOLDS[category],
        "db_path": str(db_path),
        "route_positions": {},
        "dense_score": None,
        "bm25_score": None,
    }


def _merge_route_hits(
    category: str,
    hinted_subcategory: str | None,
    db_path: Path,
    route: str,
    raw_hits: list[dict[str, Any]],
    merged: dict[str, dict[str, Any]],
) -> None:
    accepted_rank = 0
    for item in raw_hits:
        score = float(item.get("distance", item.get("score", 0.0)))
        if (
            route == "dense"
            and score < CATEGORY_THRESHOLDS[category]
        ):
            continue
        accepted_rank += 1
        base = _base_hit(
            category,
            hinted_subcategory,
            db_path,
            item,
        )
        identity = f"{category}:{base['source_id']}"
        hit = merged.setdefault(identity, base)
        route_name = f"{route}:{db_path.name}"
        previous = hit["route_positions"].get(route_name)
        hit["route_positions"][route_name] = (
            accepted_rank
            if previous is None
            else min(previous, accepted_rank)
        )
        score_field = f"{route}_score"
        current_score = hit.get(score_field)
        if current_score is None or score > current_score:
            hit[score_field] = score


def category_vector_search_multi(
    semantic_query: str,
    categories: Iterable[str] | None = None,
    subcategory_hints: dict[str, list[str]] | None = None,
    metadata_filters: (
        dict[str, list[MetadataCondition]] | None
    ) = None,
    top_k_per_category: int = TOP_K_PER_CATEGORY,
) -> tuple[list[dict[str, Any]], list[str]]:
    semantic_query = " ".join((semantic_query or "").split())
    if not semantic_query:
        return [], []

    clean_categories = _clean_categories(categories)
    hints = subcategory_hints or {}
    filters = metadata_filters or {}
    top_k = max(1, min(int(top_k_per_category), 50))
    recall_k = min(top_k * HYBRID_RECALL_MULTIPLIER, 200)
    query_vector = encode_query(semantic_query)
    all_hits: list[dict[str, Any]] = []
    warnings: list[str] = []

    for category in clean_categories:
        try:
            filter_expression = compile_metadata_filter(
                category,
                filters.get(category),
            )
        except Exception as exc:
            warnings.append(f"{category} 元数据过滤条件无效：{exc}")
            continue

        shards = iter_existing_shards(category, hints.get(category))
        if not shards:
            warnings.append(f"{category} 尚未构建向量数据库")
            continue

        merged: dict[str, dict[str, Any]] = {}
        for hinted_subcategory, db_path in shards:
            try:
                routes = _search_one_shard(
                    db_path,
                    semantic_query,
                    query_vector,
                    recall_k,
                    filter_expression,
                )
            except Exception as exc:
                warnings.append(
                    f"{category} 分片 {db_path.name} 检索失败：{exc}"
                )
                continue
            for route in ("dense", "bm25"):
                _merge_route_hits(
                    category,
                    hinted_subcategory,
                    db_path,
                    route,
                    routes[route],
                    merged,
                )

        category_hits = list(merged.values())
        for hit in category_hits:
            hit["hybrid_score"] = sum(
                1.0 / (HYBRID_RRF_K + rank)
                for rank in hit["route_positions"].values()
                if rank > 0
            )
            hit["score"] = hit["hybrid_score"]
        category_hits.sort(
            key=lambda item: (
                item["hybrid_score"],
                item["dense_score"] or float("-inf"),
            ),
            reverse=True,
        )
        for rank, hit in enumerate(category_hits[:top_k], start=1):
            hit["rank"] = rank
            hit["filter_expression"] = filter_expression
            all_hits.append(hit)

    return all_hits, warnings

