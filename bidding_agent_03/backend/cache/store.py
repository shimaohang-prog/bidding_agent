from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any, Protocol


class CacheStore(Protocol):
    async def get_context(self, user_id: str, conversation_id: str) -> list[dict[str, str]] | None: ...
    async def set_context(self, user_id: str, conversation_id: str, messages: list[dict[str, str]]) -> None: ...
    async def append_event(self, request_id: str, event: dict[str, Any]) -> dict[str, Any]: ...
    async def replay(self, request_id: str, last_seq: int) -> list[dict[str, Any]]: ...
    async def set_cancelled(self, request_id: str) -> None: ...
    async def is_cancelled(self, request_id: str) -> bool: ...
    async def acquire_generation(self, user_id: str, limit: int) -> bool: ...
    async def release_generation(self, user_id: str) -> None: ...
    async def allow_request(self, user_id: str, limit: int) -> bool: ...
    async def delete_conversation(self, user_id: str, conversation_id: str) -> None: ...
    async def enqueue_file(self, payload: dict[str, str]) -> None: ...


class RedisCacheStore:
    def __init__(self, redis: Any, *, context_ttl: int, stream_ttl: int, stream_max_events: int, file_stream: str) -> None:
        self.redis = redis
        self.context_ttl = context_ttl
        self.stream_ttl = stream_ttl
        self.stream_max_events = stream_max_events
        self.file_stream = file_stream

    @staticmethod
    def _context_key(user_id: str, conversation_id: str) -> str:
        return f"ctx:{user_id}:{conversation_id}"

    async def get_context(self, user_id: str, conversation_id: str) -> list[dict[str, str]] | None:
        raw = await self.redis.get(self._context_key(user_id, conversation_id))
        return json.loads(raw) if raw else None

    async def set_context(self, user_id: str, conversation_id: str, messages: list[dict[str, str]]) -> None:
        await self.redis.set(self._context_key(user_id, conversation_id), json.dumps(messages, ensure_ascii=False), ex=self.context_ttl)

    async def append_event(self, request_id: str, event: dict[str, Any]) -> dict[str, Any]:
        seq_key = f"stream-seq:{request_id}"
        stream_key = f"stream:{request_id}"
        seq = int(await self.redis.incr(seq_key))
        body = {**event, "seq": seq}
        if body.get("type") == "done":
            body.setdefault("payload", {})["final_seq"] = seq
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.xadd(stream_key, {"seq": str(seq), "payload": json.dumps(body, ensure_ascii=False)}, maxlen=self.stream_max_events, approximate=True)
            pipe.expire(stream_key, self.stream_ttl)
            pipe.expire(seq_key, self.stream_ttl)
            await pipe.execute()
        return body

    async def replay(self, request_id: str, last_seq: int) -> list[dict[str, Any]]:
        rows = await self.redis.xrange(f"stream:{request_id}", min="-", max="+")
        output = []
        for _, values in rows:
            seq = int(values.get("seq", 0))
            if seq > last_seq:
                output.append(json.loads(values["payload"]))
        return output

    async def set_cancelled(self, request_id: str) -> None:
        await self.redis.set(f"cancel:{request_id}", "1", ex=self.stream_ttl)

    async def is_cancelled(self, request_id: str) -> bool:
        return bool(await self.redis.exists(f"cancel:{request_id}"))

    async def acquire_generation(self, user_id: str, limit: int) -> bool:
        key = f"active:{user_id}"
        count = int(await self.redis.incr(key))
        await self.redis.expire(key, self.stream_ttl)
        if count > limit:
            await self.redis.decr(key)
            return False
        return True

    async def release_generation(self, user_id: str) -> None:
        key = f"active:{user_id}"
        if int(await self.redis.get(key) or 0) > 0:
            await self.redis.decr(key)

    async def allow_request(self, user_id: str, limit: int) -> bool:
        key = f"rate:{user_id}:{int(time.time() // 60)}"
        count = int(await self.redis.incr(key))
        if count == 1:
            await self.redis.expire(key, 65)
        return count <= limit

    async def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        await self.redis.delete(self._context_key(user_id, conversation_id))

    async def enqueue_file(self, payload: dict[str, str]) -> None:
        await self.redis.xadd(self.file_stream, {"payload": json.dumps(payload, ensure_ascii=False)})


class MemoryCacheStore:
    """只供单元测试使用，不用于可扩展部署。"""

    def __init__(self) -> None:
        self.contexts: dict[tuple[str, str], list[dict[str, str]]] = {}
        self.events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.cancelled: set[str] = set()
        self.active: dict[str, int] = defaultdict(int)
        self.rates: dict[tuple[str, int], int] = defaultdict(int)
        self.file_jobs: list[dict[str, str]] = []
        self._lock = asyncio.Lock()

    async def get_context(self, user_id: str, conversation_id: str) -> list[dict[str, str]] | None:
        return self.contexts.get((user_id, conversation_id))

    async def set_context(self, user_id: str, conversation_id: str, messages: list[dict[str, str]]) -> None:
        self.contexts[(user_id, conversation_id)] = messages

    async def append_event(self, request_id: str, event: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            body = {**event, "seq": len(self.events[request_id]) + 1}
            if body.get("type") == "done":
                body.setdefault("payload", {})["final_seq"] = body["seq"]
            self.events[request_id].append(body)
            return body

    async def replay(self, request_id: str, last_seq: int) -> list[dict[str, Any]]:
        return [item for item in self.events[request_id] if item["seq"] > last_seq]

    async def set_cancelled(self, request_id: str) -> None:
        self.cancelled.add(request_id)

    async def is_cancelled(self, request_id: str) -> bool:
        return request_id in self.cancelled

    async def acquire_generation(self, user_id: str, limit: int) -> bool:
        if self.active[user_id] >= limit:
            return False
        self.active[user_id] += 1
        return True

    async def release_generation(self, user_id: str) -> None:
        self.active[user_id] = max(0, self.active[user_id] - 1)

    async def allow_request(self, user_id: str, limit: int) -> bool:
        key = (user_id, int(time.time() // 60))
        self.rates[key] += 1
        return self.rates[key] <= limit

    async def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        self.contexts.pop((user_id, conversation_id), None)

    async def enqueue_file(self, payload: dict[str, str]) -> None:
        self.file_jobs.append(payload)
