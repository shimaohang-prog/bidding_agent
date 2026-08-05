# -*- coding: utf-8 -*-
"""把结构化过滤条件安全编译为 Milvus 标量过滤表达式。"""

import json
from typing import Any, Iterable

from common.retrieval_models import MetadataCondition


ALLOWED_FILTER_FIELDS: dict[str, set[str]] = {
    "enterprise": {
        "enterprise_name",
        "uscc",
        "corporation",
        "province",
        "city",
        "district",
        "industry",
        "enterprise_type",
        "status",
        "event_time",
        "registered_capital_amount",
        "subcategory",
        "source",
    },
    "tender": {
        "tender_title",
        "project_type",
        "source_name",
        "province",
        "city",
        "town",
        "purchasing_staff",
        "bid_company",
        "event_time",
        "bid_date",
        "bid_amount",
        "subcategory",
        "source",
    },
    "product": {
        "title",
        "major_category",
        "middle_category",
        "supplier_name",
        "currency",
        "province",
        "city",
        "event_time",
        "amount",
        "subcategory",
        "source",
    },
    "laws": {"title", "subcategory", "source", "updated_at"},
    "policy": {"title", "subcategory", "source", "updated_at"},
}

NUMERIC_FIELDS = {
    "registered_capital_amount",
    "bid_amount",
    "amount",
}
SCALAR_FIELDS = {"title", "subcategory", "source", "updated_at"}


def _field_expression(field: str) -> str:
    if field in SCALAR_FIELDS:
        return field
    return f'metadata["{field}"]'


def _number(value: Any) -> int | float:
    if isinstance(value, bool):
        raise ValueError("数值过滤不接受布尔值")
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("数值过滤值不能为空")
    number = float(text)
    return int(number) if number.is_integer() else number


def _literal(field: str, value: Any) -> str:
    if field in NUMERIC_FIELDS:
        return str(_number(value))
    return json.dumps(str(value), ensure_ascii=False)


def compile_metadata_filter(
    category: str,
    conditions: Iterable[MetadataCondition | dict[str, Any]] | None,
) -> str:
    """只允许白名单字段和四种操作符，不拼接模型给出的原始表达式。"""
    allowed = ALLOWED_FILTER_FIELDS.get(category)
    if allowed is None:
        return ""

    clauses: list[str] = []
    for raw in conditions or []:
        condition = (
            raw
            if isinstance(raw, MetadataCondition)
            else MetadataCondition.model_validate(raw)
        )
        field = condition.field
        if field not in allowed:
            raise ValueError(
                f"{category} 不允许过滤字段 {field}；"
                f"允许字段：{sorted(allowed)}"
            )

        expression = _field_expression(field)
        operator = condition.operator
        value = condition.value
        if operator == "in":
            if not isinstance(value, list) or not value:
                raise ValueError(f"{field} 的 in 操作必须提供非空数组")
            literals = ", ".join(_literal(field, item) for item in value)
            clauses.append(f"{expression} in [{literals}]")
            continue
        if isinstance(value, list):
            raise ValueError(f"{field} 的 {operator} 操作不接受数组")
        symbol = {
            "eq": "==",
            "gte": ">=",
            "lte": "<=",
        }[operator]
        clauses.append(f"{expression} {symbol} {_literal(field, value)}")
    return " and ".join(clauses)

