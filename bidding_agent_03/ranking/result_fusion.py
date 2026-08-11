# -*- coding: utf-8 -*-
"""统一六分类向量与联网结果，并用 RRF 融合名次。"""

import hashlib
import json
import unicodedata
from typing import Any, Iterable

from common.retrieval_models import Candidate


RRF_K = 60


def normalise_vector_results(
    hits: Iterable[dict[str, Any]],
) -> list[Candidate]:
    output: list[Candidate] = []
    for item in hits:
        hit = item["hit"]
        category = hit["category"]
        route_positions = hit.get("route_positions") or {
            "hybrid": int(hit["rank"])
        }
        rank_positions = {
            (
                f"semantic:{item['query_index']}:{category}:"
                f"{route_name}"
            ): int(route_rank)
            for route_name, route_rank in route_positions.items()
        }
        payload = hit.get("payload") or {}
        content = hit.get("content") or json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
        output.append(
            Candidate(
                source_type="category_vector",
                category=category,
                subcategory=hit.get("subcategory"),
                source_id=hit["source_id"],
                title=hit.get("title", ""),
                content=content,
                original_score=float(hit["score"]),
                retrieval_lists=list(rank_positions),
                rank_positions=rank_positions,
                metadata={
                    "source": hit.get("source"),
                    "payload": payload,
                    "updated_at": hit.get("updated_at"),
                    "vector_threshold": hit["threshold"],
                    "dense_score": hit.get("dense_score"),
                    "bm25_score": hit.get("bm25_score"),
                    "hybrid_score": hit.get("hybrid_score"),
                    "filter_expression": hit.get("filter_expression", ""),
                    "semantic_query": item["semantic_query"],
                    "db_path": hit.get("db_path"),
                },
                exact_title_match=bool(hit.get("exact_title_match")),
            )
        )
    return output


def normalise_web_results(
    hits: Iterable[dict[str, Any]],
) -> list[Candidate]:
    output: list[Candidate] = []
    for item in hits:
        raw_score = item.get("score")
        output.append(
            Candidate(
                source_type="web",
                category="web",
                source_id=item["url"],
                title=item.get("title", ""),
                content=item.get("content", ""),
                original_score=(
                    float(raw_score) if raw_score is not None else None
                ),
                retrieval_lists=["web:tavily"],
                rank_positions={"web:tavily": int(item["rank"])},
                metadata={
                    "url": item["url"],
                    "published_date": item.get("published_date"),
                },
            )
        )
    return output


def _content_fingerprint(content: str) -> str | None:
    """Return a conservative fingerprint for non-empty candidate content."""
    normalized = " ".join(
        unicodedata.normalize("NFKC", content or "").split()
    )
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _source_reference(candidate: Candidate) -> dict[str, Any]:
    """Keep traceability when identical content has multiple source IDs."""
    return {
        "source_type": candidate.source_type,
        "category": candidate.category,
        "source_id": candidate.source_id,
        "title": candidate.title,
        "source": candidate.metadata.get("source"),
        "url": candidate.metadata.get("url"),
    }


def _merge_rank_evidence(
    current: Candidate,
    candidate: Candidate,
) -> None:
    current.exact_title_match = (
        current.exact_title_match or candidate.exact_title_match
    )
    if (
        current.original_score is None
        or (
            candidate.original_score is not None
            and candidate.original_score > current.original_score
        )
    ):
        current.original_score = candidate.original_score
    current.retrieval_lists = list(
        dict.fromkeys(current.retrieval_lists + candidate.retrieval_lists)
    )
    for name, rank in candidate.rank_positions.items():
        previous = current.rank_positions.get(name)
        current.rank_positions[name] = (
            rank if previous is None else min(previous, rank)
        )


def _deduplicate_same_content(
    candidates: Iterable[Candidate],
) -> list[Candidate]:
    """Merge identical non-empty content inside the same category."""
    merged: dict[tuple[str, str], Candidate] = {}
    output: list[Candidate] = []
    for candidate in candidates:
        fingerprint = _content_fingerprint(candidate.content)
        if fingerprint is None:
            output.append(candidate.model_copy(deep=True))
            continue

        key = (candidate.category, fingerprint)
        current = merged.get(key)
        if current is None:
            current = candidate.model_copy(deep=True)
            merged[key] = current
            output.append(current)
            continue

        _merge_rank_evidence(current, candidate)
        references = current.metadata.get("duplicate_sources")
        if not isinstance(references, list):
            references = []
        for reference in (
            _source_reference(current),
            _source_reference(candidate),
        ):
            if reference not in references:
                references.append(reference)
        current.metadata["duplicate_sources"] = references

    return output


def deduplicate_candidates(
    candidates: Iterable[Candidate],
) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for candidate in candidates:
        key = candidate.identity_key
        current = merged.get(key)
        if current is None:
            merged[key] = candidate.model_copy(deep=True)
            continue
        current.exact_title_match = (
            current.exact_title_match or candidate.exact_title_match
        )
        if (
            current.original_score is None
            or (
                candidate.original_score is not None
                and candidate.original_score > current.original_score
            )
        ):
            current.original_score = candidate.original_score
            current.content = candidate.content
            current.metadata.update(candidate.metadata)
        current.retrieval_lists = list(
            dict.fromkeys(current.retrieval_lists + candidate.retrieval_lists)
        )
        for name, rank in candidate.rank_positions.items():
            previous = current.rank_positions.get(name)
            current.rank_positions[name] = (
                rank if previous is None else min(previous, rank)
            )
    return _deduplicate_same_content(merged.values())


def reciprocal_rank_fusion(
    candidates: Iterable[Candidate],
    limit: int = 100,
) -> list[Candidate]:
    unique = deduplicate_candidates(candidates)
    for candidate in unique:
        candidate.fusion_score = sum(
            1.0 / (RRF_K + rank)
            for rank in candidate.rank_positions.values()
            if rank > 0
        )
    unique.sort(
        key=lambda item: (
            item.exact_title_match,
            item.fusion_score,
        ),
        reverse=True,
    )
    return unique[: max(1, min(int(limit), 200))]
