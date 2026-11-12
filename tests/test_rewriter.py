"""tests/test_rewriter.py - Unit tests for contextual pronoun resolution & fail-open behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.rewriter import QueryRewriter


def test_conversational_pronoun_resolution():
    """Test 1: Rewrites 'What are its system requirements?' with 'Kubernetes Worker Nodes' context."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Kubernetes Worker Nodes minimum system requirements CPU RAM storage"
                )
            )
        ]
    )

    rewriter = QueryRewriter(api_key="sk-test-key", client=mock_client)
    history = [
        {"role": "user", "content": "Tell me about Kubernetes Worker Nodes in our EKS clusters."},
        {"role": "assistant", "content": "Kubernetes Worker Nodes run managed workloads on EKS 1.30."},
    ]

    rewritten = rewriter.rewrite("What are its system requirements?", chat_history=history)

    assert "Kubernetes Worker Nodes" in rewritten
    assert "system requirements" in rewritten.lower()


def test_rewriter_fails_open_on_timeout_and_empty_input():
    """Verifies empty queries return '' and LLM exceptions fail open to the raw query."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = TimeoutError("Upstream LLM timed out")

    rewriter = QueryRewriter(api_key="sk-test-key", client=mock_client)

    assert rewriter.rewrite("   ") == ""
    raw = "How do I provision an isolated VPC in staging?"
    assert rewriter.rewrite(raw) == raw
