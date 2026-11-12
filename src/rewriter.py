"""src/rewriter.py - Contextual Query Rewriter with fast-path bypass and fail-open resilience."""

import os
import re
from typing import Dict, List, Optional
from openai import OpenAI

_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Detect anaphoric pronouns, temporal back-references, or conversational filler that benefit from LLM rewriting
_CONTEXT_DEPENDENT_PATTERN = re.compile(
    r"(\b(it|its|they|them|their|this|that|these|those|he|she|former|latter|above|previous|earlier|last month)\b"
    r"|^(hi|hello|hey|can you|could you|please|help me|tell me about)\b)",
    flags=re.IGNORECASE,
)

_REWRITER_SYSTEM_PROMPT = """You are an enterprise technical search query optimizer.
Rewrite the user's latest query into a self-contained, keyword-rich technical search query.
Resolve all pronouns using the conversation history, strip filler, preserve exact technical flags/versions, and output ONLY the rewritten query."""


class QueryRewriter:
    """Rewrites conversational or underspecified queries into standalone search queries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 3.5,
        client: Optional[OpenAI] = None,
    ) -> None:
        raw_key = (
            api_key
            if api_key is not None
            else (os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "")
        ).strip()
        self._api_key = raw_key
        self._timeout = timeout_seconds

        is_gemini_key = raw_key.startswith(("AQ.", "AIza"))
        self._model = model or ("gemini-3.5-flash-lite" if is_gemini_key else "gpt-4o-mini")

        if client is not None:
            self._client = client
        elif raw_key:
            self._client = (
                OpenAI(api_key=raw_key, base_url=_GEMINI_OPENAI_BASE_URL, timeout=self._timeout)
                if is_gemini_key
                else OpenAI(api_key=raw_key, timeout=self._timeout)
            )
        else:
            self._client = None

    def rewrite(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Rewrites a user query using optional conversation context.
        Fast-paths standalone single-turn queries in <1ms and fails open on LLM timeout.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return ""

        if self._client is None:
            return cleaned_query

        has_history = bool(chat_history and any(t.get("content", "").strip() for t in chat_history))
        if not has_history and not _CONTEXT_DEPENDENT_PATTERN.search(cleaned_query):
            return cleaned_query

        messages: List[Dict[str, str]] = [{"role": "system", "content": _REWRITER_SYSTEM_PROMPT}]

        if has_history and chat_history is not None:
            recent_turns = chat_history[-4:]
            history_lines = [
                f"{turn.get('role', 'user').upper()}: {turn.get('content', '').strip()}"
                for turn in recent_turns
                if turn.get("content", "").strip()
            ]
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Conversation Context:\n"
                        + "\n".join(history_lines)
                        + f"\n\nLatest User Query to Rewrite:\n{cleaned_query}"
                    ),
                }
            )
        else:
            messages.append({"role": "user", "content": cleaned_query})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=80,
            )
            rewritten = (response.choices[0].message.content or "").strip().strip('"').strip("'")
            return rewritten if rewritten else cleaned_query
        except Exception:
            return cleaned_query
