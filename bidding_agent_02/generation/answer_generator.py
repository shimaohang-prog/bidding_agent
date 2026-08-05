# -*- coding: utf-8 -*-
"""仅依据已检索证据生成最终答案。"""

import os

from common.llm_client import chat_completion, message_content


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是专业的招投标智能助手。必须以提供的证据为边界，"
                "不得编造企业、项目、金额、日期、政策条文或来源。"
            ),
        },
        {
            "role": "user",
            "content": f"""
请回答用户问题。

要求：
1. 综合 enterprise、tender、product、laws、policy、news 多类证据，
   不要把问题强行归入单一分类。
2. 结构化业务事实以分类向量命中后返回的完整 CSV 行数据为准；
   系统不执行 MySQL 查询。
3. 不要把向量相似度、RRF 分数、重排分数当作业务事实。
4. 法律政策说明文件来源；新闻和联网信息说明日期与 URL。
5. 重要结论在句末标注证据编号，例如 [E1]。
6. 证据冲突时明确列出冲突；证据不足时直接说明无法确认。
7. 不引用上下文中没有出现的事实。

用户问题：
{question}

证据上下文：
{context}
""".strip(),
        },
    ]


def generate_answer(question: str, context: str) -> str:
    body = chat_completion(
        build_messages(question, context),
        model=os.getenv("ANSWER_MODEL", "").strip() or None,
        temperature=0.1,
        max_tokens=3500,
    )
    return message_content(body)

