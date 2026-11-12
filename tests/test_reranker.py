"""tests/test_reranker.py - Unit tests for Sigmoid logit normalization and confidence gating."""

from unittest.mock import MagicMock

from src.reranker import CrossEncoderReranker, sigmoid
from src.schemas import RetrievedChunk


def test_sigmoid_logit_normalization():
    """Test 2: Positive, zero, and negative raw logits map strictly into [0.0, 1.0]."""
    sample_logits = [-100.0, -8.5, -1.2, 0.0, 0.25, 4.7, 100.0]
    normalized = [sigmoid(x) for x in sample_logits]

    for score in normalized:
        assert 0.0 <= score <= 1.0

    # Verify strict monotonicity and midpoint calibration
    assert sigmoid(0.0) == 0.5
    assert sigmoid(5.0) > 0.99
    assert sigmoid(-5.0) < 0.01


def test_threshold_gating_logic():
    """Test 3: Top score 0.42 triggers LOW_CONFIDENCE, while 0.81 passes HIGH_CONFIDENCE."""
    mock_model = MagicMock()
    reranker = CrossEncoderReranker(threshold=0.55, cross_encoder=mock_model)

    low_conf_chunks = [
        RetrievedChunk(chunk_id="doc#1", content="Irrelevant passage", score=0.42, source="internal_doc"),
        RetrievedChunk(chunk_id="doc#2", content="Another weak match", score=0.31, source="internal_doc"),
    ]
    status_low, score_low = reranker.evaluate_gate(low_conf_chunks)
    assert status_low == "LOW_CONFIDENCE"
    assert score_low == 0.42

    high_conf_chunks = [
        RetrievedChunk(chunk_id="doc#1", content="Staging VPC CIDR 10.20.0.0/16", score=0.81, source="internal_doc"),
        RetrievedChunk(chunk_id="doc#2", content="Subnets across 3 AZs", score=0.64, source="internal_doc"),
    ]
    status_high, score_high = reranker.evaluate_gate(high_conf_chunks)
    assert status_high == "HIGH_CONFIDENCE"
    assert score_high == 0.81
