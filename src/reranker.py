"""src/reranker.py - Stage 2 Cross-Encoder Re-Ranker with Sigmoid Calibration & Confidence Gate."""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import math
from typing import List, Literal, Optional, Tuple

from sentence_transformers import CrossEncoder

from src.schemas import RetrievedChunk

DEFAULT_CONFIDENCE_THRESHOLD = 0.55


def sigmoid(logit: float) -> float:
    """Numerically stable sigmoid normalization mapping raw logits (-inf, +inf) to [0.0, 1.0]."""
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


class CrossEncoderReranker:
    """Re-scores Stage 1 bi-encoder candidates with a Cross-Encoder and enforces confidence gating."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        cross_encoder: Optional[CrossEncoder] = None,
    ) -> None:
        self.threshold = threshold
        if cross_encoder is not None:
            self._model = cross_encoder
        else:
            # Load from local cache first to eliminate startup Hugging Face Hub network round-trips
            try:
                self._model = CrossEncoder(model_name, local_files_only=True)
            except Exception:
                self._model = CrossEncoder(model_name)

    def evaluate_gate(
        self, scored_chunks: List[RetrievedChunk]
    ) -> Tuple[Literal["HIGH_CONFIDENCE", "LOW_CONFIDENCE"], float]:
        """Evaluates whether the top normalized confidence score meets the empirical threshold."""
        if not scored_chunks:
            return "LOW_CONFIDENCE", 0.0
        top_score = max(chunk.score for chunk in scored_chunks)
        if top_score >= self.threshold:
            return "HIGH_CONFIDENCE", top_score
        return "LOW_CONFIDENCE", top_score

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5,
    ) -> Tuple[List[RetrievedChunk], Literal["HIGH_CONFIDENCE", "LOW_CONFIDENCE"], float]:
        """Scores (query, candidate.content) pairs, normalizes logits via sigmoid, and evaluates the gate."""
        if not candidates or not query.strip():
            return [], "LOW_CONFIDENCE", 0.0

        pairs = [(query, cand.content) for cand in candidates]
        raw_logits = self._model.predict(pairs)

        scored_items: List[Tuple[float, float, RetrievedChunk]] = []
        for cand, logit_val in zip(candidates, raw_logits):
            raw_float = float(logit_val)
            prob_score = round(sigmoid(raw_float), 4)
            scored_items.append((prob_score, raw_float, cand))

        scored_items.sort(key=lambda item: item[0], reverse=True)

        reranked: List[RetrievedChunk] = []
        for new_rank, (prob_score, raw_float, cand) in enumerate(scored_items[:top_k], start=1):
            reranked.append(
                RetrievedChunk(
                    chunk_id=cand.chunk_id,
                    content=cand.content,
                    score=prob_score,
                    source=cand.source,
                    title=cand.title,
                    initial_rank=cand.initial_rank,
                    reranked_rank=new_rank,
                    raw_logit=round(raw_float, 4),
                )
            )

        status, top_confidence = self.evaluate_gate(reranked)
        return reranked, status, top_confidence
