"""异步 Tavily 客户端，保留指定域名请求与 hostname 二次过滤。"""

from typing import Any, Iterable

import httpx

from retrieval.web_search import (
    GOV_SITES, _clean_requested_site_query, _remove_explicit_urls,
    _url_matches_domains, build_search_query, extract_requested_domains,
)


class AsyncTavilyClient:
    def __init__(self, client: httpx.AsyncClient, *, url: str, api_key: str) -> None:
        self.client = client
        self.url = url
        self.api_key = api_key

    async def _request(self, query: str, max_results: int, domains: list[str] | None) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("未配置 Tavily API Key")
        payload: dict[str, Any] = {
            "api_key": self.api_key, "query": query, "search_depth": "advanced",
            "max_results": max_results, "include_answer": False, "include_raw_content": False,
        }
        if domains:
            payload["include_domains"] = domains
        response = await self.client.post(self.url, json=payload)
        if response.status_code == 429:
            raise RuntimeError("Tavily 限流或额度不足")
        if response.status_code >= 400:
            raise RuntimeError(f"Tavily 请求失败 HTTP {response.status_code}")
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("Tavily 返回格式异常")
        return body

    async def search(self, query: str, categories: Iterable[str], max_results: int = 6) -> list[dict[str, Any]]:
        requested = extract_requested_domains(query)
        clean = (_clean_requested_site_query(query) if requested else _remove_explicit_urls(query)) or query
        search_query = clean if requested else build_search_query(clean, categories)
        prefer_gov = not requested and bool(set(categories) & {"laws", "policy", "news", "tender"})
        body = await self._request(search_query, max_results, requested or (GOV_SITES if prefer_gov else None))
        if prefer_gov and not body.get("results"):
            body = await self._request(search_query, max_results, None)
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, item in enumerate(body.get("results", []), 1):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            if not url or url in seen or not _url_matches_domains(url, requested):
                continue
            seen.add(url)
            output.append({
                "category": "web", "title": str(item.get("title", "")).strip(),
                "content": str(item.get("content", "")).strip(), "url": url,
                "published_date": item.get("published_date"), "score": item.get("score"), "rank": rank,
            })
        return output
