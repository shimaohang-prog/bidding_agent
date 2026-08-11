# -*- coding: utf-8 -*-
"""本地证据不足或问题要求时效性时调用 Tavily。"""

import os
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common.milvus_config import ALL_CATEGORIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

TAVILY_URL = "https://api.tavily.com/search"

# 招投标、政策法规类问题优先搜索这些官方站点。
# gov.cn 可以覆盖大量中央和地方政府网站。
GOV_SITES = [
    "gov.cn",
    "ndrc.gov.cn",
    "mof.gov.cn",
    "ccgp.gov.cn",
    "ggzy.gov.cn",
    "cebpubservice.com",
]

CATEGORY_SUFFIX = {
    "enterprise": "中国 企业 工商信息 供应商信息",
    "tender": "中国 招标 采购 中标 公告",
    "product": "中国 产品 软件 设备 供应商",
    "laws": "中国 招标投标 政府采购 法律法规",
    "policy": "中国 招标投标 政府采购 政策文件",
    "news": "中国 招投标 政府采购 最新新闻 动态",
}

URL_PATTERN = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+",
    re.IGNORECASE,
)
DOMAIN_PATTERN = re.compile(
    r"(?<![@\w.-])(?:www\.)?"
    r"(?:[a-z0-9-]+\.)+[a-z]{2,}"
    r"(?![\w.-])",
    re.IGNORECASE,
)
SEARCH_COMMAND_PATTERN = re.compile(
    r"基于|请|给我|联网搜索|上网搜索|网络搜索|"
    r"搜索网站|再给我答案|给我答案|告诉我",
    re.IGNORECASE,
)


def _create_session() -> requests.Session:
    """创建带自动重试机制的 HTTP 会话。"""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


_SESSION = _create_session()


def _api_key() -> str:
    """读取 Tavily API Key。"""
    value = os.getenv("TAVILY_API_KEY", "").strip()

    if not value:
        raise ValueError(
            "未配置 TAVILY_API_KEY，请在项目 .env 文件中添加：\n"
            "TAVILY_API_KEY=tvly-你的API_KEY"
        )

    return value


def _clean_categories(
    categories: Iterable[str] | None,
) -> list[str]:
    """清理并去重分类参数。"""
    values = list(categories or ALL_CATEGORIES)

    return list(
        dict.fromkeys(
            item
            for item in values
            if item in ALL_CATEGORIES
        )
    )


def extract_requested_domains(query: str) -> list[str]:
    """提取用户明确给出的 URL/域名，作为 Tavily 强制域名范围。"""
    domains: list[str] = []
    for raw_url in URL_PATTERN.findall(query or ""):
        hostname = (urlparse(raw_url).hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        if hostname:
            domains.append(hostname)
    for hostname in DOMAIN_PATTERN.findall(query or ""):
        clean = hostname.lower()
        if clean.startswith("www."):
            clean = clean[4:]
        domains.append(clean)
    return list(dict.fromkeys(domains))


def _remove_explicit_urls(query: str) -> str:
    text = URL_PATTERN.sub(" ", query or "")
    text = DOMAIN_PATTERN.sub(" ", text)
    return " ".join(text.split())


def _clean_requested_site_query(query: str) -> str:
    text = _remove_explicit_urls(query)
    text = SEARCH_COMMAND_PATTERN.sub(" ", text)
    return " ".join(text.split()).strip(" ?？,，。")


def _url_matches_domains(url: str, domains: list[str]) -> bool:
    if not domains:
        return True
    hostname = (urlparse(url).hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in domains
    )


def build_search_query(
    query: str,
    categories: Iterable[str] | None,
) -> str:
    """构建更适合搜索中文网页的查询语句。"""
    clean_categories = _clean_categories(categories)

    suffixes = [
        CATEGORY_SUFFIX[item]
        for item in clean_categories
        if item in CATEGORY_SUFFIX
    ]

    return " ".join(
        item.strip()
        for item in [query, *suffixes]
        if item and item.strip()
    )


def _get_error_message(response: requests.Response) -> str:
    """从 Tavily 错误响应中提取可读信息。"""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500].strip()

    if isinstance(body, dict):
        return str(
            body.get("detail")
            or body.get("message")
            or body.get("error")
            or body
        )

    return str(body)


def _request(
    query: str,
    max_results: int,
    include_domains: list[str] | None = None,
) -> dict[str, Any]:
    """请求 Tavily Search API。"""
    api_key = _api_key()

    payload: dict[str, Any] = {
        "query": query,

        # country 参数只支持 general，因此固定使用 general。
        "topic": "general",

        # 优先返回中国地区网页。
        "country": "china",

        # advanced 精度更高，但每次通常消耗更多 Tavily credits。
        "search_depth": "advanced",
        "chunks_per_source": 3,

        "max_results": max(1, min(int(max_results), 10)),
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "auto_parameters": False,
    }

    if include_domains:
        payload["include_domains"] = include_domains

    try:
        response = _SESSION.post(
            TAVILY_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=(10, 45),
        )

        if not response.ok:
            error_message = _get_error_message(response)

            if response.status_code == 401:
                raise RuntimeError(
                    "Tavily API Key 无效或已失效，请检查 "
                    "TAVILY_API_KEY。"
                )

            if response.status_code == 429:
                raise RuntimeError(
                    "Tavily 请求过于频繁或额度不足，请稍后重试。"
                )

            raise RuntimeError(
                f"Tavily 请求失败，状态码：{response.status_code}，"
                f"错误信息：{error_message}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Tavily 返回的内容不是有效 JSON。"
            ) from exc

        if not isinstance(body, dict):
            raise RuntimeError(
                f"Tavily 返回格式异常：{type(body).__name__}"
            )

        return body

    except requests.exceptions.ConnectTimeout as exc:
        raise RuntimeError(
            "连接 Tavily API 超时。请检查当前网络是否能够访问 "
            "https://api.tavily.com。"
        ) from exc

    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            "Tavily 搜索响应超时，请稍后重试。"
        ) from exc

    except requests.exceptions.ProxyError as exc:
        raise RuntimeError(
            "代理连接失败。请检查系统代理、VPN 或 requests 代理配置。"
        ) from exc

    except requests.exceptions.SSLError as exc:
        raise RuntimeError(
            "Tavily HTTPS 证书验证失败，请检查系统时间、证书或代理软件。"
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "无法连接 Tavily API。可能是 DNS、公司防火墙、"
            "跨境网络或代理配置问题。"
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Tavily 网络请求异常：{exc}"
        ) from exc


def web_search(
    query: str,
    categories: Iterable[str] | None = None,
    max_results: int = 6,
) -> list[dict[str, Any]]:
    """返回结构化搜索结果，保留 URL 和发布时间供最终回答引用。"""
    query = (query or "").strip()

    if not query:
        return []

    clean_categories = _clean_categories(categories)
    requested_domains = extract_requested_domains(query)
    clean_query = (
        _clean_requested_site_query(query)
        if requested_domains
        else _remove_explicit_urls(query)
    ) or query
    search_query = (
        clean_query
        if requested_domains
        else build_search_query(clean_query, clean_categories)
    )

    # 政策、法规、新闻、招投标类问题优先搜索政府及官方平台。
    prefer_gov = not requested_domains and bool(
        set(clean_categories)
        & {"laws", "policy", "news", "tender"}
    )

    body = _request(
        query=search_query,
        max_results=max_results,
        include_domains=(
            requested_domains
            if requested_domains
            else GOV_SITES if prefer_gov else None
        ),
    )

    # 官方网站没有结果时，自动取消域名限制，再搜索一次中文网页。
    if prefer_gov and not body.get("results"):
        body = _request(
            query=search_query,
            max_results=max_results,
            include_domains=None,
        )

    output: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for rank, item in enumerate(
        body.get("results", []),
        start=1,
    ):
        if not isinstance(item, dict):
            continue

        url = str(item.get("url", "")).strip()

        if (
            not url
            or url in seen_urls
            or not _url_matches_domains(url, requested_domains)
        ):
            continue

        seen_urls.add(url)

        output.append(
            {
                "category": "web",
                "title": str(
                    item.get("title", "")
                ).strip(),
                "content": str(
                    item.get("content", "")
                ).strip(),
                "url": url,
                "published_date": item.get(
                    "published_date"
                ),
                "score": item.get("score"),
                "rank": rank,
            }
        )

    return output
