"""共享 httpx 客户端上的 DeepSeek SSE，只暴露最终 content。"""

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx


class UpstreamStreamError(RuntimeError):
    pass


@dataclass(slots=True)
class StreamChunk:
    content: str = ""
    usage: dict[str, Any] | None = None


class DeepSeekStreamClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.url = url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    async def stream(self, messages: list[dict[str, Any]]) -> AsyncIterator[StreamChunk]:
        if not self.api_key:
            raise UpstreamStreamError("未配置 DeepSeek API Key")
        payload = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 3500, "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            async with self.client.stream("POST", self.url, headers=headers, json=payload) as response:
                if response.status_code == 429:
                    raise UpstreamStreamError("DeepSeek 限流")
                if response.status_code >= 500:
                    raise UpstreamStreamError("DeepSeek 服务暂时不可用")
                if response.status_code >= 400:
                    raise UpstreamStreamError(f"DeepSeek 请求失败 HTTP {response.status_code}")
                saw_data = False
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        body = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise UpstreamStreamError("DeepSeek 流包含无效 JSON") from exc
                    saw_data = True
                    if body.get("usage"):
                        yield StreamChunk(usage=body["usage"])
                    choices = body.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    # reasoning_content、工具参数和系统提示词永不转发。
                    if content:
                        yield StreamChunk(content=str(content))
                if not saw_data:
                    raise UpstreamStreamError("DeepSeek 返回空流")
        except httpx.TimeoutException as exc:
            raise UpstreamStreamError("DeepSeek 流响应超时") from exc
