# -*- coding: utf-8 -*-
"""把问题规划为一个或多个六分类语义检索任务。"""

import os
import re
from datetime import date
from typing import Any

from common.llm_client import call_forced_tool
from common.milvus_config import ALL_CATEGORIES, VECTOR_CATEGORIES
from common.retrieval_models import (
    RetrievalPlan,
    SemanticQueryTask,
)


PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_vector_retrieval_plan",
        "description": "提交招投标六分类语义检索计划。",
        "parameters": {
            "type": "object",
            "properties": {
                "semantic_queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "categories": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": list(ALL_CATEGORIES),
                                },
                            },
                            "subcategory_hints": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "metadata_filters": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "field": {"type": "string"},
                                            "operator": {
                                                "type": "string",
                                                "enum": [
                                                    "eq",
                                                    "in",
                                                    "gte",
                                                    "lte",
                                                ],
                                            },
                                            "value": {},
                                        },
                                        "required": [
                                            "field",
                                            "operator",
                                            "value",
                                        ],
                                    },
                                },
                            },
                            "top_k_per_category": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 50,
                            },
                        },
                        "required": [
                            "query",
                            "categories",
                            "subcategory_hints",
                            "metadata_filters",
                            "top_k_per_category",
                        ],
                    },
                },
                "requires_web_search": {"type": "boolean"},
                "requires_fresh_data": {"type": "boolean"},
                "answer_focus": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "semantic_queries",
                "requires_web_search",
                "requires_fresh_data",
                "answer_focus",
            ],
        },
    },
}


def _messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是招投标检索规划器。只生成语义检索计划，"
                "不要生成 SQL，也不要生成 MySQL 精确查询。"
            ),
        },
        {
            "role": "user",
            "content": f"""
当前日期：{date.today().isoformat()}

请把用户问题拆成一个或多个语义主题，并选择所有真正相关的分类：
- enterprise：企业、工商、供应商主体与能力；
- tender：招标、采购、中标、项目公告；
- product：产品、设备、软件、货物与报价；
- laws：法律、行政法规、司法解释；
- policy：部门规章、规范性文件、地方政策；
- news：最新动态和新闻，news 只联网搜索，不查本地向量。

规则：
1. 多意图问题可以选择多个分类，不要强制唯一分类。
2. 不要为了扩大召回而无条件选择全部分类。
3. 用户明确要求“联网搜索、上网搜索、搜索指定网站”或给出 URL 时，
   设置 requires_web_search=true。明确要求最新、近期、当前或新闻时，
   包含 news，并设置 requires_fresh_data=true。
4. metadata_filters 没有条件时必须返回空对象 {{}}，不能返回数组 []。
5. 只有用户明确说出已存在的子分类名称时，才填写
   subcategory_hints；不确定时留空，由系统检索整个大分类。
6. 明确出现的地区、类别、完整名称、统一信用代码、金额或日期条件，
   写入对应分类的 metadata_filters，同时仍保留在 query 中供
   Dense 和 BM25 召回。不得返回原始 Milvus 表达式。
7. metadata_filters 只使用 eq、in、gte、lte。本地代码会再次执行
   分类字段白名单验证并安全编译。

允许过滤字段：
- enterprise：enterprise_name, uscc, corporation, province, city,
  district, industry, enterprise_type, status, event_time,
  registered_capital_amount, subcategory
- tender：tender_title, project_type, source_name, province, city, town,
  purchasing_staff, bid_company, event_time, bid_date, bid_amount,
  subcategory
- product：title, major_category, middle_category, supplier_name,
  currency, province, city, event_time, amount, subcategory
- laws/policy：title, subcategory, source, updated_at

用户问题：
{question}
""".strip(),
        },
    ]


CATEGORY_RULES = {
    "enterprise": r"企业|公司|供应商|工商|法人|信用代码|经营范围|资质",
    "tender": r"招标|投标|中标|采购|项目|公告|评标|成交",
    "product": r"产品|设备|软件|货物|材料|型号|报价|单价|价格|包子机|食品机械",
    "laws": r"法律|法规|法条|条例|司法解释|违法|责任|处罚",
    "policy": r"政策|办法|规定|通知|意见|细则|规范|标准",
    "news": r"新闻|动态|最新|近期|今天|当前|截至目前|最近",
}


def _rules_fallback(question: str, warning: str) -> RetrievalPlan:
    categories = [
        category
        for category, pattern in CATEGORY_RULES.items()
        if re.search(pattern, question)
    ]
    requires_fresh = "news" in categories
    requires_web = bool(
        re.search(
            r"联网搜索|上网搜索|网络搜索|搜索网站|"
            r"https?://|www\.",
            question,
            re.IGNORECASE,
        )
    )
    if not categories:
        categories = list(VECTOR_CATEGORIES)
    metadata_filters: dict[str, list[dict[str, Any]]] = {}
    province = next(
        (
            item
            for item in (
                "北京市", "天津市", "上海市", "重庆市", "河北省", "山西省",
                "辽宁省", "吉林省", "黑龙江省", "江苏省", "浙江省", "安徽省",
                "福建省", "江西省", "山东省", "河南省", "湖北省", "湖南省",
                "广东省", "海南省", "四川省", "贵州省", "云南省", "陕西省",
                "甘肃省", "青海省", "内蒙古自治区", "广西壮族自治区",
                "西藏自治区", "宁夏回族自治区", "新疆维吾尔自治区",
            )
            if item in question
        ),
        None,
    )
    if province:
        for category in categories:
            if category in {"enterprise", "tender", "product"}:
                metadata_filters.setdefault(category, []).append(
                    {
                        "field": "province",
                        "operator": "eq",
                        "value": province,
                    }
                )
    uscc = re.search(
        r"(?<![0-9A-Z])([0-9A-HJ-NPQRTUWXY]{18})(?![0-9A-Z])",
        question.upper(),
    )
    if uscc and "enterprise" in categories:
        metadata_filters.setdefault("enterprise", []).append(
            {
                "field": "uscc",
                "operator": "eq",
                "value": uscc.group(1),
            }
        )
    return RetrievalPlan(
        semantic_queries=[
            SemanticQueryTask(
                query=question,
                categories=categories,
                metadata_filters=metadata_filters,
            )
        ],
        requires_web_search=requires_web,
        requires_fresh_data=requires_fresh,
        answer_focus=[question],
        planner_source="rules_fallback",
        planner_warning=warning,
    )


def _normalise(raw: dict[str, Any], question: str) -> RetrievalPlan:
    plan = RetrievalPlan.model_validate(raw)
    if not plan.semantic_queries:
        plan.semantic_queries = [
            SemanticQueryTask(
                query=question,
                categories=list(VECTOR_CATEGORIES),
            )
        ]
    if any("news" in task.categories for task in plan.semantic_queries):
        plan.requires_fresh_data = True
    rule_categories = [
        category
        for category, pattern in CATEGORY_RULES.items()
        if re.search(pattern, question)
    ]
    if len(rule_categories) == 1:
        for task in plan.semantic_queries:
            if set(task.categories) == set(VECTOR_CATEGORIES):
                task.categories = rule_categories
    if re.search(
        r"联网搜索|上网搜索|网络搜索|搜索网站|https?://|www\.",
        question,
        re.IGNORECASE,
    ):
        plan.requires_web_search = True
    return plan


def create_retrieval_plan(question: str) -> RetrievalPlan:
    question = " ".join((question or "").split())
    if not question:
        raise ValueError("用户问题不能为空")
    try:
        raw = call_forced_tool(
            messages=_messages(question),
            tool=PLAN_TOOL,
            model=os.getenv("PLANNER_MODEL", "").strip() or None,
            max_tokens=3000,
        )
        return _normalise(raw, question)
    except Exception as exc:
        return _rules_fallback(question, str(exc))
