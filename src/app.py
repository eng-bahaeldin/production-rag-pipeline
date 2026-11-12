"""src/app.py - Interactive Observability UI for the Two-Stage RAG Pipeline."""

from concurrent.futures import ThreadPoolExecutor
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, List
from dotenv import load_dotenv
import streamlit as st

from src.generator import GroundedGenerator
from src.reranker import CrossEncoderReranker, DEFAULT_CONFIDENCE_THRESHOLD
from src.retriever import BiEncoderRetriever
from src.rewriter import QueryRewriter
from src.schemas import RAGResponse
from src.web_fallback import WebFallbackEngine

load_dotenv(".env", override=True)


def _sync_streamlit_secrets() -> None:
    """Mirrors Streamlit Community Cloud secrets into os.environ before pipeline init."""
    try:
        for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY"):
            if key in st.secrets:
                secret_val = str(st.secrets[key]).strip()
                if secret_val and not os.getenv(key):
                    os.environ[key] = secret_val
    except Exception:
        pass


_sync_streamlit_secrets()

st.set_page_config(
    page_title="Production Two-Stage RAG Pipeline",
    layout="wide",
)


def _detect_active_backends() -> tuple[str, str]:
    """Identifies active LLM and web search providers from environment/secrets."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    if openai_key.startswith("sk-"):
        llm_label = "OpenAI (`gpt-4o-mini`)"
    elif openai_key.startswith(("AQ.", "AIza")) or gemini_key:
        llm_label = "Google Gemini (`gemini-3.5-flash-lite`)"
    else:
        llm_label = "Unconfigured (Add API Key in Secrets)"

    search_label = "Tavily Search API" if tavily_key else "DuckDuckGo Search (Zero-Key)"
    return llm_label, search_label


@st.cache_resource(show_spinner="Initializing bi-encoder & cross-encoder models...")
def load_pipeline_components() -> tuple[
    QueryRewriter, BiEncoderRetriever, CrossEncoderReranker, WebFallbackEngine, GroundedGenerator
]:
    persist_path = "/tmp/.chroma_db" if sys.platform != "win32" else ".chroma_db"

    def _init_retriever() -> BiEncoderRetriever:
        ret = BiEncoderRetriever(docs_dir="docs", persist_dir=persist_path)
        ret.ingest_documents()
        return ret

    def _init_reranker() -> CrossEncoderReranker:
        return CrossEncoderReranker(threshold=DEFAULT_CONFIDENCE_THRESHOLD)

    # Load both Hugging Face models concurrently to halve cold-start time
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_ret = pool.submit(_init_retriever)
        fut_rer = pool.submit(_init_reranker)
        retriever = fut_ret.result()
        reranker = fut_rer.result()

    rewriter = QueryRewriter()
    web_fallback = WebFallbackEngine()
    generator = GroundedGenerator()
    return rewriter, retriever, reranker, web_fallback, generator


rewriter, retriever, reranker, web_fallback, generator = load_pipeline_components()

if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []
if "last_response" not in st.session_state:
    st.session_state.last_response: RAGResponse | None = None

st.title("Production Two-Stage RAG Pipeline & Adaptive Web Fallback")
st.caption(
    "Query Transformation -> Stage 1 Bi-Encoder (K1=20) -> Stage 2 Cross-Encoder (K2=5) -> Sigmoid Gate (0.55) -> Grounded Synthesis"
)

with st.sidebar:
    st.header("Gate Configuration")
    llm_backend, search_backend = _detect_active_backends()
    st.caption(f"**LLM Engine:** {llm_backend}\n\n**Fallback Search:** {search_backend}")

    threshold = st.slider(
        "Confidence Gate Cutoff (Sigmoid)",
        min_value=0.10,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
    )
    reranker.threshold = threshold

    multi_turn_enabled = st.toggle("Enable Multi-Turn Conversational Context", value=True)
    if st.button("Clear Conversation History"):
        st.session_state.chat_history = []
        st.session_state.last_response = None
        st.rerun()

    st.divider()
    st.header("Pipeline Telemetry")

    last_resp = st.session_state.last_response
    if last_resp is not None:
        if last_resp.route_taken == "INTERNAL_RAG":
            st.success("🟢 **INTERNAL_RAG** (High Confidence)")
        else:
            st.warning("🟠 **WEB_FALLBACK** (Low Confidence Triggered)")

        st.metric(
            label=f"Top Cross-Encoder Confidence (Cutoff: {threshold:.2f})",
            value=f"{last_resp.confidence_score:.4f}",
            delta=f"{last_resp.confidence_score - threshold:+.4f} vs threshold",
        )
        st.progress(min(max(last_resp.confidence_score, 0.0), 1.0))

        st.markdown("**Transformed Search Query:**")
        st.code(last_resp.transformed_query, language="text")
    else:
        st.info("Submit a query to inspect live routing telemetry.")

st.markdown("**Quick Demo Scenarios:**")
col1, col2, col3 = st.columns(3)
preset_query = ""
with col1:
    if st.button("1. Core Internal (Staging VPC)", use_container_width=True):
        preset_query = "How do I provision an isolated VPC in our staging environment?"
with col2:
    if st.button("2. Conversational Follow-up (Terraform)", use_container_width=True):
        st.session_state.chat_history = [
            {
                "role": "user",
                "content": "How do I provision an isolated VPC in our staging environment?",
            },
            {
                "role": "assistant",
                "content": "Use the internal golden module terraform-aws-vpc-enterprise v5.8.0+ across three AZs.",
            },
        ]
        preset_query = "Can I do that with the older terraform modules we used last month?"
with col3:
    if st.button("3. Out-of-Scope (Live AWS Outage)", use_container_width=True):
        preset_query = "What was the root cause of the major AWS us-east-1 outage announced yesterday?"

with st.form("rag_query_form", clear_on_submit=False):
    user_query = st.text_input(
        "Enter your engineering or infrastructure question:",
        value=preset_query,
        placeholder="e.g., What are the minimum system requirements for Kubernetes Worker Nodes?",
    )
    submitted = st.form_submit_button("Execute Two-Stage RAG Pipeline", type="primary")

if submitted and user_query.strip():
    with st.spinner("Running Query Rewriter -> Bi-Encoder -> Cross-Encoder -> Confidence Gate..."):
        active_history = st.session_state.chat_history if multi_turn_enabled else []
        transformed_query = rewriter.rewrite(user_query, chat_history=active_history)

        stage1_candidates = retriever.retrieve(transformed_query, top_k=20)
        stage2_reranked, gate_status, top_confidence = reranker.rerank(
            transformed_query, stage1_candidates, top_k=5
        )

        if gate_status == "HIGH_CONFIDENCE":
            route_taken = "INTERNAL_RAG"
            evidence_chunks = stage2_reranked[:3]
        else:
            route_taken = "WEB_FALLBACK"
            evidence_chunks = web_fallback.search(transformed_query, max_results=3)

        try:
            rag_response = generator.generate(
                original_query=user_query,
                transformed_query=transformed_query,
                chunks=evidence_chunks,
                route_taken=route_taken,
                confidence_score=top_confidence,
                stage1_candidates=stage1_candidates,
                stage2_reranked=stage2_reranked,
            )
        except Exception as exc:
            st.error(f"Upstream LLM synthesis failed after retries: {exc}")
            st.stop()

        st.session_state.last_response = rag_response
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": rag_response.answer})
        st.rerun()

resp = st.session_state.last_response
if resp is not None:
    st.subheader("Synthesized Grounded Answer")
    st.markdown(resp.answer)

    # Read cached evidence_chunks directly from RAGResponse (eliminates duplicate web search)
    if resp.citations:
        st.markdown("**Source Citations (Click to inspect raw evidence):**")
        active_evidence = resp.evidence_chunks or resp.stage2_reranked[:3]
        cite_cols = st.columns(min(len(resp.citations), 3))
        for idx, cite_id in enumerate(resp.citations):
            col = cite_cols[idx % len(cite_cols)]
            with col:
                with st.popover(f"[{idx + 1}] {cite_id[:42]}..."):
                    st.markdown(f"**Source Identifier:** `{cite_id}`")
                    matched = next((c for c in active_evidence if c.chunk_id == cite_id), None)
                    if matched is not None:
                        st.metric("Relevance Score", f"{matched.score:.4f}")
                        st.write(matched.content)
                    if cite_id.startswith("http"):
                        st.markdown(f"[Open External Source URL]({cite_id})")

    st.divider()
    st.subheader("Two-Stage Retrieval & Re-Ranking Visualizer")

    st.markdown("#### Stage 2: Cross-Encoder Re-Ranked Top 5 Candidates")
    if resp.stage2_reranked:
        stage2_rows = []
        for item in resp.stage2_reranked:
            init_r = item.initial_rank or 0
            new_r = item.reranked_rank or 0
            shift = init_r - new_r
            shift_label = f"▲ +{shift}" if shift > 0 else (f"▼ {shift}" if shift < 0 else "— 0")
            stage2_rows.append(
                {
                    "New Rank": new_r,
                    "Stage 1 Rank": init_r,
                    "Rank Shift": shift_label,
                    "Chunk ID": item.chunk_id,
                    "Raw Logit": item.raw_logit,
                    "Sigmoid Score": item.score,
                    "Preview": item.content[:140].replace("\n", " ") + "...",
                }
            )
        st.dataframe(stage2_rows, use_container_width=True, hide_index=True)

    with st.expander(f"Stage 1: Bi-Encoder Candidate Pool (Top {len(resp.stage1_candidates)} Retrieved)"):
        stage1_rows = [
            {
                "Stage 1 Rank": c.initial_rank,
                "Chunk ID": c.chunk_id,
                "Cosine Similarity": c.score,
                "Preview": c.content[:150].replace("\n", " ") + "...",
            }
            for c in resp.stage1_candidates
        ]
        st.dataframe(stage1_rows, use_container_width=True, hide_index=True)
