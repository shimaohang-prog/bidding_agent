# -*- coding: utf-8 -*-
"""用同一个 LLM 相关性标准重排不同来源的候选证据。"""

import os
from typing import Any, Iterable

from common.llm_client import call_forced_tool
from common.retrieval_models import Candidate


def _tool(candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "submit_relevance_assessments",
            "description": "提交每条候选证据与用户问题的相关性判断。",
            "parameters": {
                "type": "object",
                "properties": {
                    "assessments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {
                                    "type": "string",
                                    "enum": candidate_ids,
                                },
                                "score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "reason": {"type": "string"},
                            },
                            "required": ["candidate_id", "score", "reason"],
                        },
                    }
                },
                "required": ["assessments"],
            },
        },
    }


def rerank_candidates(
    question: str,
    candidates: Iterable[Candidate],
    top_n: int = 20,
) -> tuple[list[Candidate], str | None]:
    items = [item.model_copy(deep=True) for item in candidates]
    if not items:
        return [], None

    mode = os.getenv("RERANK_MODE", "llm").strip().lower()
    if mode in {"off", "none", "rrf"}:
        return items[:top_n], None

    ids = [item.identity_key for item in items]
    evidence = "\n\n".join(
        (
            f"candidate_id={item.identity_key}\n"
            f"来源={item.source_type}/{item.category}\n"
            f"完整标题精确命中={'是' if item.exact_title_match else '否'}\n"
            f"子分类={item.subcategory or '无'}\n"
            f"标题={item.title}\n"
            f"内容={item.content[:1600]}"
        )
        for item in items
    )
    try:
        raw = call_forced_tool(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是统一相关性重排器。只判断证据是否能直接帮助回答"
                        "用户问题，不因来源类别预先加分，不把 Dense、BM25、"
                        "RRF 或相似度分数当事实。问题包含完整名称、代码、"
                        "金额或日期时，应核对证据中是否出现一致值。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：\n{question}\n\n候选证据：\n{evidence}"
                    ),
                },
            ],
            tool=_tool(ids),
            model=os.getenv("RERANK_MODEL", "").strip() or None,
            max_tokens=3500,
        )
        assessments = {
            item["candidate_id"]: item
            for item in raw.get("assessments", [])
            if item.get("candidate_id") in ids
        }
        for candidate in items:
            assessment = assessments.get(candidate.identity_key)
            if assessment:
                candidate.rerank_score = max(
                    0.0,
                    min(float(assessment.get("score", 0.0)), 1.0),
                )
                candidate.relevance_reason = str(
                    assessment.get("reason", "")
                )[:300]
        items.sort(
            key=lambda item: (
                item.exact_title_match,
                item.rerank_score is not None,
                item.rerank_score or 0.0,
                item.fusion_score,
            ),
            reverse=True,
        )
        return items[: max(1, min(int(top_n), 50))], None
    except Exception as exc:
        items.sort(
            key=lambda item: (
                item.exact_title_match,
                item.fusion_score,
            ),
            reverse=True,
        )
        return items[: max(1, min(int(top_n), 50))], str(exc)
