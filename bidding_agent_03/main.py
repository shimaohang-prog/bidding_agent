# -*- coding: utf-8 -*-
"""招投标六分类 Dense + BM25 混合检索问答系统入口。"""

import argparse
import json
import sys
from typing import Any

import anyio

from backend.core.config import get_settings
from backend.services.deepseek_stream import DeepSeekStreamClient
from backend.services.private_documents import PrivateDocumentStore
from backend.services.qa_pipeline import (
    build_generation_messages,
    create_upstream_http_client,
    retrieve_qa_context,
)
from backend.services.rag_service import AsyncRAGService
from backend.services.tavily_client import AsyncTavilyClient


CLI_USER_ID = "00000000-0000-0000-0000-000000000001"
CLI_CONVERSATION_ID = "00000000-0000-0000-0000-000000000002"


async def answer_question_async(question: str) -> dict[str, Any]:
    """Run the same no-history/no-file pipeline as a fresh web conversation."""
    question = " ".join((question or "").split())
    if not question:
        raise ValueError("问题不能为空")

    settings = get_settings()
    http = create_upstream_http_client()
    limiter = anyio.CapacityLimiter(settings.milvus_thread_limit)
    private_store = PrivateDocumentStore(
        settings.private_milvus_uri,
        settings.private_collection,
        limiter,
    )
    rag = AsyncRAGService(
        limiter=limiter,
        tavily=AsyncTavilyClient(
            http,
            url=settings.tavily_url,
            api_key=settings.tavily_api_key.get_secret_value(),
        ),
        private_store=private_store,
    )
    deepseek = DeepSeekStreamClient(
        http,
        url=settings.deepseek_url,
        api_key=settings.deepseek_api_key.get_secret_value(),
        model=settings.answer_model,
        temperature=settings.answer_temperature,
    )

    stages: list[str] = []

    async def status(stage: str) -> None:
        stages.append(stage)

    try:
        retrieval_result, context, citations = await retrieve_qa_context(
            rag,
            question,
            user_id=CLI_USER_ID,
            conversation_id=CLI_CONVERSATION_ID,
            file_ids=[],
            status=status,
        )
        messages = build_generation_messages(question, context)
        answer_parts: list[str] = []
        usage: dict = {}
        async for chunk in deepseek.stream(messages):
            if chunk.content:
                answer_parts.append(chunk.content)
            if chunk.usage:
                usage = chunk.usage
        answer = "".join(answer_parts).strip()
        if not answer:
            raise RuntimeError("模型未返回可用答案")
        return {
            "answer": answer,
            "plan": retrieval_result.plan,
            "retrieval_result": retrieval_result,
            "context": context,
            "citations": citations,
            "usage": usage,
            "stages": stages,
        }
    finally:
        await http.aclose()


def answer_question(question: str) -> dict[str, Any]:
    return anyio.run(answer_question_async, question)


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
