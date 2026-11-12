"""src/web_fallback.py - Live Web Search Fallback Engine (Tavily primary, DuckDuckGo zero-key fallback)."""

import os
from typing import List, Optional

from src.schemas import RetrievedChunk


class WebFallbackEngine:
    """Executes real-time web search when internal cross-encoder confidence drops below threshold."""

    def __init__(self, tavily_api_key: Optional[str] = None) -> None:
        self._tavily_key = (
            tavily_api_key if tavily_api_key is not None else os.getenv("TAVILY_API_KEY", "")
        ).strip()

    def search(self, query: str, max_results: int = 3) -> List[RetrievedChunk]:
        """Fetches top external web snippets formatted as normalized RetrievedChunk objects."""
        cleaned = (query or "").strip()
        if not cleaned:
            return []

        if self._tavily_key:
            tavily_chunks = self._search_tavily(cleaned, max_results=max_results)
            if tavily_chunks:
                return tavily_chunks

        return self._search_ddg(cleaned, max_results=max_results)

    def _search_tavily(self, query: str, max_results: int) -> List[RetrievedChunk]:
        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]

            client = TavilyClient(api_key=self._tavily_key)
            response = client.search(query=query, max_results=max_results, search_depth="basic")
            results = response.get("results", [])
            chunks: List[RetrievedChunk] = []
            for idx, item in enumerate(results[:max_results], start=1):
                url = (item.get("url") or "").strip()
                snippet = (item.get("content") or "").strip()
                title = (item.get("title") or url).strip()
                if url and snippet:
                    chunks.append(
                        RetrievedChunk(
                            chunk_id=url,
                            content=snippet,
                            score=round(max(0.50, 0.90 - (idx - 1) * 0.08), 4),
                            source="web",
                            title=title,
                            reranked_rank=idx,
                        )
                    )
            return chunks
        except Exception:
            return []

    def _search_ddg(self, query: str, max_results: int) -> List[RetrievedChunk]:
        try:
            try:
                from ddgs import DDGS  # type: ignore[import-untyped]
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore[import-untyped]

            chunks: List[RetrievedChunk] = []
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))

            for idx, item in enumerate(raw_results[:max_results], start=1):
                url = (item.get("href") or item.get("url") or "").strip()
                snippet = (item.get("body") or item.get("snippet") or "").strip()
                title = (item.get("title") or url).strip()
                if url and snippet:
                    chunks.append(
                        RetrievedChunk(
                            chunk_id=url,
                            content=snippet,
                            score=round(max(0.50, 0.88 - (idx - 1) * 0.08), 4),
                            source="web",
                            title=title,
                            reranked_rank=idx,
                        )
                    )
            return chunks
        except Exception:
            return []
