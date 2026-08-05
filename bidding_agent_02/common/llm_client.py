# -*- coding: utf-8 -*-
"""统一封装 DeepSeek 普通对话和 Function Calling。"""

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_URL = os.getenv(
    "DEEPSEEK_URL",
    "https://api.deepseek.com/chat/completions",
).strip()
DEFAULT_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-v4-flash",
).strip()


class DeepSeekClientError(RuntimeError):
    """请求参数、模型能力或认证等不可重试的 4xx 错误。"""


def _api_key() -> str:
    value = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not value:
        raise ValueError("未配置 DEEPSEEK_API_KEY")
    return value


def _request(payload: dict[str, Any]) -> dict[str, Any]:
    """只重试限流和服务端错误，不重试参数错误。"""
    api_key = _api_key()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(
                DEEPSEEK_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(10, 90),
            )
            if (
                400 <= response.status_code < 500
                and response.status_code != 429
            ):
                detail = response.text.strip()
                if len(detail) > 1200:
                    detail = detail[:1200] + "..."
                raise DeepSeekClientError(
                    f"DeepSeek HTTP {response.status_code}："
                    f"{detail or '服务端未返回错误详情'}"
                )
            if response.status_code == 429 or response.status_code >= 500:
                raise RuntimeError(
                    f"DeepSeek 暂时不可用，HTTP {response.status_code}"
                )
            response.raise_for_status()
            body = response.json()
            if not body.get("choices"):
                raise RuntimeError("DeepSeek 未返回 choices")
            return body
        except DeepSeekClientError:
            # 参数错误、模型不支持或认证失败时，重复请求没有意义。
            raise
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"DeepSeek 请求失败：{last_error}") from last_error


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 3000,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    thinking: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if thinking is not None:
        payload["thinking"] = {
            "type": "enabled" if thinking else "disabled"
        }
    return _request(payload)


def call_forced_tool(
    *,
    messages: list[dict[str, Any]],
    tool: dict[str, Any],
    model: str | None = None,
    max_tokens: int = 3000,
) -> dict[str, Any]:
    """强制模型调用一个函数，并返回已经解析的 arguments。"""
    function_name = tool["function"]["name"]
    body = chat_completion(
        messages,
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        tools=[tool],
        # 当前只提供一个工具，required 与指定函数等价，并且对
        # DeepSeek V4 的兼容性更稳定。
        tool_choice="required",
        # 检索规划和相关性评分需要结构化、稳定的工具参数，
        # 不需要默认开启的思考模式。
        thinking=False,
    )
    message = body["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        raise ValueError(f"模型没有调用 {function_name}")
    selected = next(
        (
            item
            for item in tool_calls
            if item.get("function", {}).get("name") == function_name
        ),
        tool_calls[0],
    )
    arguments = selected.get("function", {}).get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Function Calling arguments 不是 JSON 对象")
    return arguments


def message_content(body: dict[str, Any]) -> str:
    content = body["choices"][0]["message"].get("content")
    if not content or not str(content).strip():
        raise ValueError("模型没有返回文本内容")
    return str(content).strip()
