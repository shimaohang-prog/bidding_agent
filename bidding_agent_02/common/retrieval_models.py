# -*- coding: utf-8 -*-
"""六分类向量检索共用的数据协议。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.milvus_config import ALL_CATEGORIES, VECTOR_CATEGORIES


Category = Literal[
    "enterprise",
    "tender",
    "product",
    "laws",
    "policy",
    "news",
]
CandidateCategory = Literal[
    "enterprise",
    "tender",
    "product",
    "laws",
    "policy",
    "news",
    "web",
]
SourceType = Literal["category_vector", "web"]


class MetadataCondition(BaseModel):
    """由本地白名单编译为 Milvus 表达式，禁止接收原始表达式。"""

    model_config = ConfigDict(extra="ignore")

    field: str = Field(min_length=1, max_length=80)
    operator: Literal["eq", "in", "gte", "lte"] = "eq"
    value: str | int | float | list[str | int | float]


class SemanticQueryTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1, max_length=500)
    categories: list[Category] = Field(
        default_factory=lambda: list(VECTOR_CATEGORIES)
    )
    subcategory_hints: dict[str, list[str]] = Field(default_factory=dict)
    metadata_filters: dict[str, list[MetadataCondition]] = Field(
        default_factory=dict
    )
    top_k_per_category: int = Field(default=8, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        clean = " ".join(value.split())
        if not clean:
            raise ValueError("语义检索内容不能为空")
        return clean

    @field_validator("categories")
    @classmethod
    def unique_categories(cls, value: list[str]) -> list[str]:
        valid = [item for item in dict.fromkeys(value) if item in ALL_CATEGORIES]
        return valid or list(VECTOR_CATEGORIES)

    @field_validator("subcategory_hints")
    @classmethod
    def clean_subcategory_hints(
        cls,
        value: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        for category, labels in value.items():
            if category not in ALL_CATEGORIES:
                continue
            clean = [
                " ".join(str(label).split())
                for label in labels
                if str(label).strip()
            ]
            if clean:
                output[category] = list(dict.fromkeys(clean))
        return output

    @field_validator("metadata_filters", mode="before")
    @classmethod
    def normalise_empty_metadata_filters(cls, value):
        if value is None or value == []:
            return {}
        return value

    @field_validator("metadata_filters")
    @classmethod
    def clean_metadata_filters(
        cls,
        value: dict[str, list[MetadataCondition]],
    ) -> dict[str, list[MetadataCondition]]:
        return {
            category: conditions
            for category, conditions in value.items()
            if category in ALL_CATEGORIES and conditions
        }


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    semantic_queries: list[SemanticQueryTask] = Field(
        default_factory=list,
        max_length=6,
    )
    requires_web_search: bool = False
    requires_fresh_data: bool = False
    answer_focus: list[str] = Field(default_factory=list, max_length=10)
    planner_source: str = "llm_function_call"
    planner_warning: str | None = None

    @field_validator("answer_focus")
    @classmethod
    def clean_focus(cls, value: list[str]) -> list[str]:
        return [
            text[:120]
            for text in dict.fromkeys(
                " ".join(str(item).split())
                for item in value
                if str(item).strip()
            )
        ]


class Candidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_type: SourceType
    category: CandidateCategory
    source_id: str = Field(min_length=1, max_length=500)
    title: str = ""
    content: str = ""
    subcategory: str | None = None
    original_score: float | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = Field(default=None, ge=0.0, le=1.0)
    relevance_reason: str | None = None
    retrieval_lists: list[str] = Field(default_factory=list)
    rank_positions: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def identity_key(self) -> str:
        return f"{self.category}:{self.source_id}"


class RetrievalResult(BaseModel):
    plan: RetrievalPlan
    candidates: list[Candidate] = Field(default_factory=list)
    used_web_search: bool = False
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
