import httpx
import pytest

from backend.services.deepseek_stream import DeepSeekStreamClient, UpstreamStreamError


async def collect(response: httpx.Response):
    async def handler(_: httpx.Request) -> httpx.Response:
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        stream = DeepSeekStreamClient(client, url="https://deepseek.test/chat", api_key="test-key", model="test")
        return [item async for item in stream.stream([{"role": "user", "content": "q"}])]


@pytest.mark.asyncio
async def test_stream_forwards_only_content_and_usage():
    body = (
        'data: {"choices":[{"delta":{"reasoning_content":"internal","content":"你"}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"secret":"x"}],"content":"好"}}]}\n\n'
        'data: {"choices":[],"usage":{"total_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    )
    chunks = await collect(httpx.Response(200, text=body))
    assert "".join(item.content for item in chunks) == "你好"
    assert chunks[-1].usage == {"total_tokens": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_stream_maps_retryable_http_errors(status):
    with pytest.raises(UpstreamStreamError):
        await collect(httpx.Response(status, text="internal stack must not leak"))


@pytest.mark.asyncio
async def test_stream_rejects_invalid_json_and_empty_response():
    with pytest.raises(UpstreamStreamError, match="无效 JSON"):
        await collect(httpx.Response(200, text="data: {broken}\n\n"))
    with pytest.raises(UpstreamStreamError, match="空流"):
        await collect(httpx.Response(200, text=""))


@pytest.mark.asyncio
async def test_stream_maps_timeout():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        stream = DeepSeekStreamClient(client, url="https://deepseek.test/chat", api_key="test-key", model="test")
        with pytest.raises(UpstreamStreamError, match="超时"):
            _ = [item async for item in stream.stream([{"role": "user", "content": "q"}])]
