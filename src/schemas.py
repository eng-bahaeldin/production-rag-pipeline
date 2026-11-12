"""src/schemas.py - Standardized Pydantic v2 schemas for the Two-Stage RAG Pipeline."""

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """Normalized document or web snippet passed to the grounded generator."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., min_length=1, description="Unique internal chunk ID or external web URL")
    content: str = Field(..., min_length=1, description="Retrieved passage or sanitized web snippet")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized relevance/confidence score in [0.0, 1.0]")
    source: Literal["internal_doc", "web"] = Field(..., description="Origin of the retrieved evidence")
    title: Optional[str] = Field(default=None, description="Document filename or web page title")
    initial_rank: Optional[int] = Field(default=None, description="1-based rank in Stage 1 bi-encoder retrieval")
    reranked_rank: Optional[int] = Field(default=None, description="1-based rank after Stage 2 cross-encoder")
    raw_logit: Optional[float] = Field(default=None, description="Unnormalized cross-encoder logit")


class RAGResponse(BaseModel):
    """Final synthesized pipeline output with routing telemetry and citations."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(..., description="Grounded synthesis with inline bracketed citations [1], [2]")
    citations: List[str] = Field(default_factory=list, description="Ordered list of chunk IDs or web URLs")
    route_taken: Literal["INTERNAL_RAG", "WEB_FALLBACK"] = Field(..., description="Execution path chosen by gate")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Top sigmoid-normalized cross-encoder score")
    transformed_query: str = Field(..., description="Standalone search query emitted by the Query Rewriter")
    stage1_candidates: List[RetrievedChunk] = Field(default_factory=list, description="Top K1 bi-encoder pool")
    stage2_reranked: List[RetrievedChunk] = Field(default_factory=list, description="Top K2 cross-encoder results")
    evidence_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Final chunks used for synthesis")
