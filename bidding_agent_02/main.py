# -*- coding: utf-8 -*-
"""招投标六分类 Dense + BM25 混合检索问答系统入口。"""

import argparse
import json
import sys
from typing import Any

from generation.answer_generator import generate_answer
from generation.context_builder import build_answer_context
from planning.retrieval_planner import create_retrieval_plan
from retrieval.retrieval_executor import execute_retrieval_plan


def answer_question(question: str) -> dict[str, Any]:
    question = " ".join((question or "").split())
    if not question:
        raise ValueError("问题不能为空")
    plan = create_retrieval_plan(question)
    retrieval_result = execute_retrieval_plan(question, plan)
    context = build_answer_context(question, retrieval_result)
    answer = generate_answer(question, context)
    return {
        "answer": answer,
        "plan": plan,
        "retrieval_result": retrieval_result,
        "context": context,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="六分类 Dense、BM25、元数据过滤和重排问答机器人"
    )
    parser.add_argument("question", nargs="?", help="用户问题；留空则交互输入")
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="输出通过本地校验后的检索计划和检索统计",
    )
    args = parser.parse_args()
    question = args.question or input("请输入问题：").strip()
    result = answer_question(question)

    if args.show_plan:
        print("\n【检索计划】")
        print(
            json.dumps(
                result["plan"].model_dump(),
                ensure_ascii=False,
                default=str,
                indent=2,
            )
        )
        print("\n【检索统计】")
        print(
            json.dumps(
                result["retrieval_result"].diagnostics,
                ensure_ascii=False,
                default=str,
                indent=2,
            )
        )
        if result["retrieval_result"].warnings:
            print("\n【检索提示】")
            for warning in result["retrieval_result"].warnings:
                print(f"- {warning}")

    print("\n" + "=" * 60)
    print("智能回答")
    print("=" * 60)
    print(result["answer"])


if __name__ == "__main__":
    main()

