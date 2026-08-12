# -*- coding: utf-8 -*-
"""招投标六分类 Dense + BM25 混合检索问答系统入口。"""

import argparse
import asyncio
import getpass
import json
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
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
from backend.services.remote_cli import (
    answer_questions_via_server,
    backend_is_available,
)
from backend.services.tavily_client import AsyncTavilyClient


CLI_USER_ID = "00000000-0000-0000-0000-000000000001"
CLI_CONVERSATION_ID = "00000000-0000-0000-0000-000000000002"
TokenCallback = Callable[[str], Awaitable[None]]
BatchTokenCallback = Callable[[int, str, str], Awaitable[None]]


class AsyncQuestionRuntime:
    """可被一个或多个并发 CLI 问题共享的异步运行时。"""

    def __init__(self) -> None:
        self.http = None
        self.rag = None
        self.deepseek = None

    async def __aenter__(self) -> "AsyncQuestionRuntime":
        settings = get_settings()
        self.http = create_upstream_http_client()
        limiter = anyio.CapacityLimiter(settings.milvus_thread_limit)
        private_store = PrivateDocumentStore(
            settings.private_milvus_uri,
            settings.private_collection,
            limiter,
        )
        self.rag = AsyncRAGService(
            limiter=limiter,
            tavily=AsyncTavilyClient(
                self.http,
                url=settings.tavily_url,
                api_key=settings.tavily_api_key.get_secret_value(),
            ),
            private_store=private_store,
        )
        self.deepseek = DeepSeekStreamClient(
            self.http,
            url=settings.deepseek_url,
            api_key=settings.deepseek_api_key.get_secret_value(),
            model=settings.answer_model,
            temperature=settings.answer_temperature,
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self.http is not None:
            await self.http.aclose()

    async def answer(
        self,
        question: str,
        *,
        on_token: TokenCallback | None = None,
    ) -> dict[str, Any]:
        question = " ".join((question or "").split())
        if not question:
            raise ValueError("问题不能为空")
        if self.rag is None or self.deepseek is None:
            raise RuntimeError("异步问答运行时尚未启动")

        stages: list[str] = []

        async def status(stage: str) -> None:
            stages.append(stage)

        retrieval_result, context, citations = await retrieve_qa_context(
            self.rag,
            question,
            user_id=CLI_USER_ID,
            conversation_id=CLI_CONVERSATION_ID,
            file_ids=[],
            status=status,
        )
        messages = build_generation_messages(question, context)
        answer_parts: list[str] = []
        usage: dict = {}
        async for chunk in self.deepseek.stream(messages):
            if chunk.content:
                answer_parts.append(chunk.content)
                if on_token is not None:
                    await on_token(chunk.content)
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


async def answer_question_async(
    question: str,
    *,
    on_token: TokenCallback | None = None,
) -> dict[str, Any]:
    """运行单个异步问题；回调存在时把模型 token 实时交给调用方。"""
    async with AsyncQuestionRuntime() as runtime:
        return await runtime.answer(question, on_token=on_token)


async def answer_questions_async(
    questions: Sequence[str],
    *,
    concurrency: int = 4,
    on_token: BatchTokenCallback | None = None,
) -> list[dict[str, Any]]:
    """在同一运行时中并发执行多个问题，并保持返回顺序与输入一致。"""
    clean_questions = [" ".join((item or "").split()) for item in questions]
    if not clean_questions or any(not item for item in clean_questions):
        raise ValueError("问题不能为空")
    if concurrency < 1:
        raise ValueError("并发数必须大于 0")

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any] | None] = [None] * len(clean_questions)
    batch_started = time.perf_counter()
    async with AsyncQuestionRuntime() as runtime:
        async def run_one(index: int, question: str) -> None:
            async with semaphore:
                started = time.perf_counter()
                first_token: float | None = None

                async def emit(content: str) -> None:
                    nonlocal first_token
                    if first_token is None:
                        first_token = time.perf_counter()
                    if on_token is not None:
                        await on_token(index, question, content)

                result = await runtime.answer(
                    question,
                    on_token=emit if on_token is not None else None,
                )
                finished = time.perf_counter()
                result["timing"] = {
                    "started_offset_ms": round(
                        (started - batch_started) * 1000
                    ),
                    "first_token_ms": round(
                        ((first_token or finished) - started) * 1000
                    ),
                    "duration_ms": round((finished - started) * 1000),
                }
                results[index] = result

        await asyncio.gather(*(
            run_one(index, question)
            for index, question in enumerate(clean_questions)
        ))
    return [item for item in results if item is not None]


def answer_question(question: str) -> dict[str, Any]:
    return anyio.run(answer_question_async, question)


def print_result_details(result: dict[str, Any], *, show_plan: bool) -> None:
    if show_plan:
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


async def run_cli_async(args: argparse.Namespace) -> None:
    questions = list(args.question) or [input("请输入问题：").strip()]
    output_lock = asyncio.Lock()
    batch_clock = time.perf_counter()

    server_url = args.server_url
    if not args.local and not server_url:
        candidate = "http://127.0.0.1:8000"
        if await backend_is_available(candidate):
            server_url = candidate
            print(
                "检测到网页后端正在运行。为避免跨进程争抢 Milvus Lite，"
                "CLI 已自动切换为服务端异步模式。"
            )

    if server_url:
        username = args.username or input("Web 用户名：").strip()
        password = getpass.getpass("Web 密码：")
        if not username or not password:
            raise ValueError("服务端模式必须提供 Web 用户名和密码")

        if len(questions) == 1 and not args.no_stream:
            print("\n" + "=" * 60)
            print("智能回答（服务端流式）")
            print("=" * 60)

        async def emit_server(index: int, question: str, content: str) -> None:
            async with output_lock:
                if len(questions) == 1:
                    print(content, end="", flush=True)
                else:
                    print(json.dumps({
                        "type": "token",
                        "question_index": index + 1,
                        "elapsed_ms": round(
                            (time.perf_counter() - batch_clock) * 1000
                        ),
                        "content": content,
                    }, ensure_ascii=False), flush=True)

        results = await answer_questions_via_server(
            questions,
            server_url=server_url,
            origin=args.origin,
            username=username,
            password=password,
            concurrency=min(args.concurrency, len(questions)),
            timeout=args.timeout,
            on_token=None if args.no_stream else emit_server,
        )
        if len(questions) == 1 and not args.no_stream:
            print()
        for index, result in enumerate(results, 1):
            if len(questions) > 1 or args.no_stream:
                print(
                    f"\n{'=' * 60}\n问题 {index}："
                    f"{questions[index - 1]}\n{'=' * 60}"
                )
                print(result["answer"])
            print(json.dumps({
                "type": "timing",
                "question_index": index,
                **result["timing"],
            }, ensure_ascii=False))
        if args.show_plan:
            print("服务端模式不回传内部检索计划；可在后端日志中查看阶段。")
        return

    if len(questions) == 1:
        if not args.no_stream:
            print("\n" + "=" * 60)
            print("智能回答（流式）")
            print("=" * 60)

        async def emit(content: str) -> None:
            print(content, end="", flush=True)

        result = await answer_question_async(
            questions[0],
            on_token=None if args.no_stream else emit,
        )
        if not args.no_stream:
            print()
        print_result_details(result, show_plan=args.show_plan)
        if args.no_stream:
            print("\n" + "=" * 60)
            print("智能回答")
            print("=" * 60)
            print(result["answer"])
        return

    async def emit_batch(index: int, question: str, content: str) -> None:
        async with output_lock:
            print(json.dumps({
                "type": "token", "question_index": index + 1,
                "elapsed_ms": round(
                    (time.perf_counter() - batch_clock) * 1000
                ),
                "content": content,
            }, ensure_ascii=False), flush=True)

    results = await answer_questions_async(
        questions,
        concurrency=min(args.concurrency, len(questions)),
        on_token=None if args.no_stream else emit_batch,
    )
    for index, result in enumerate(results, 1):
        print(f"\n{'=' * 60}\n问题 {index}：{questions[index - 1]}\n{'=' * 60}")
        print(result["answer"])
        print(json.dumps({
            "type": "timing", "question_index": index,
            **result["timing"],
        }, ensure_ascii=False))
        print_result_details(result, show_plan=args.show_plan)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="六分类 Dense、BM25、元数据过滤和重排问答机器人"
    )
    parser.add_argument("question", nargs="*", help="一个或多个用户问题；留空则交互输入一个问题")
    parser.add_argument("--concurrency", type=int, default=4, choices=range(1, 17), metavar="1-16", help="多个问题的最大并发数（默认 4）")
    parser.add_argument("--no-stream", action="store_true", help="不实时打印模型 token，只输出完整答案")
    parser.add_argument("--server-url", help="复用正在运行的 Web 后端，例如 http://127.0.0.1:8000")
    parser.add_argument("--username", help="服务端模式的 Web 用户名；省略时安全提示输入")
    parser.add_argument("--origin", default="http://127.0.0.1:5173", help="服务端允许的 WebSocket Origin")
    parser.add_argument("--timeout", type=float, default=180.0, help="服务端单次事件等待超时秒数")
    parser.add_argument("--local", action="store_true", help="即使检测到 Web 后端也强制本地运行；Lite 数据库可能被锁")
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="输出通过本地校验后的检索计划和检索统计",
    )
    args = parser.parse_args()
    anyio.run(run_cli_async, args)


if __name__ == "__main__":
    main()
