"""轻量 WebSocket 压测；Cookie 只从环境读取且永不打印。"""

import argparse
import asyncio
import json
import os
import statistics
import time
from uuid import uuid4

import websockets


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * quantile))]


async def one(args, semaphore):
    async with semaphore:
        request_id = str(uuid4())
        started = time.perf_counter()
        first_token = None
        cookie = os.environ.get("BIDDING_COOKIE", "")
        if not cookie:
            raise RuntimeError("请通过 BIDDING_COOKIE 环境变量提供短期登录 Cookie")
        async with websockets.connect(args.url, origin=args.origin, additional_headers={"Cookie": cookie}) as socket:
            await socket.send(json.dumps({
                "type": "ask", "request_id": request_id, "conversation_id": args.conversation_id,
                "client_message_id": str(uuid4()), "question": args.question, "file_ids": [],
            }, ensure_ascii=False))
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), timeout=args.timeout))
                if event["type"] == "token" and first_token is None:
                    first_token = time.perf_counter()
                if event["type"] == "done":
                    finished = time.perf_counter()
                    return ((first_token or finished) - started, finished - started, None)
                if event["type"] in {"error", "cancelled"}:
                    return (0.0, time.perf_counter() - started, event.get("payload", {}).get("error_code", event["type"]))


async def run(args):
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(one(args, semaphore) for _ in range(args.requests)), return_exceptions=True)
    ttft, total, errors = [], [], []
    for item in results:
        if isinstance(item, Exception):
            errors.append(type(item).__name__)
        else:
            first, duration, error = item
            total.append(duration * 1000)
            if first:
                ttft.append(first * 1000)
            if error:
                errors.append(error)
    print(json.dumps({
        "concurrency": args.concurrency, "requests": args.requests,
        "ttft_ms": {"p50": percentile(ttft, .5), "p95": percentile(ttft, .95), "p99": percentile(ttft, .99)},
        "total_ms": {"p50": percentile(total, .5), "p95": percentile(total, .95), "p99": percentile(total, .99)},
        "error_rate": len(errors) / args.requests, "errors": dict((code, errors.count(code)) for code in sorted(set(errors))),
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--question", default="请依据证据说明招标文件审查的主要注意事项")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
