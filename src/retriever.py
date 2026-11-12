"""src/retriever.py - Stage 1 Bi-Encoder Dense Retriever with SHA-256 deduplication."""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

from src.schemas import RetrievedChunk


def chunk_markdown_text(text: str, chunk_size: int = 750, overlap: int = 120) -> List[str]:
    """Splits markdown documentation into overlapping, sentence-aware passages."""
    cleaned = text.strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
            else:
                step = max(chunk_size - overlap, 100)
                for start in range(0, len(para), step):
                    piece = para[start : start + chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                current = ""

    if current:
        chunks.append(current)

    return chunks


class BiEncoderRetriever:
    """ChromaDB bi-encoder retriever with content-hash deduplication and offline-first model loading."""

    def __init__(
        self,
        docs_dir: str = "docs",
        persist_dir: str = ".chroma_db",
        collection_name: str = "enterprise_cloud_docs",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        in_memory: bool = False,
    ) -> None:
        self._docs_path = Path(docs_dir)
        self._persist_path = Path(persist_dir)
        self._manifest_path = self._persist_path / "ingestion_manifest.json"
        self._collection_name = collection_name

        # Load from local Hugging Face cache first to skip remote HTTP HEAD latency
        try:
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name,
                local_files_only=True,
            )
        except Exception:
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name,
            )

        if in_memory:
            self._client = chromadb.EphemeralClient()
        else:
            self._persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_path))

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._ef,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )
        self._in_memory = in_memory

    def _compute_docs_manifest(self) -> Dict[str, str]:
        """Computes SHA-256 hashes for every markdown file in docs_dir."""
        manifest: Dict[str, str] = {}
        if not self._docs_path.exists():
            return manifest
        for md_file in sorted(self._docs_path.glob("*.md")):
            digest = hashlib.sha256(md_file.read_bytes()).hexdigest()
            manifest[md_file.name] = digest
        return manifest

    def ingest_documents(self, force_reindex: bool = False) -> int:
        """Indexes markdown files into ChromaDB only when new or modified (via SHA-256 check)."""
        current_manifest = self._compute_docs_manifest()
        if not current_manifest:
            return self._collection.count()

        if not self._in_memory and not force_reindex and self._manifest_path.exists():
            try:
                saved_manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                if saved_manifest == current_manifest and self._collection.count() > 0:
                    return self._collection.count()
            except Exception:
                pass

        if self._collection.count() > 0:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=self._ef,  # type: ignore[arg-type]
                metadata={"hnsw:space": "cosine"},
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []

        for filename in sorted(current_manifest.keys()):
            file_path = self._docs_path / filename
            raw_text = file_path.read_text(encoding="utf-8")
            chunks = chunk_markdown_text(raw_text)
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{filename}#chunk-{idx + 1}"
                ids.append(chunk_id)
                documents.append(chunk_text)
                metadatas.append({"title": filename, "chunk_index": str(idx + 1)})

        if ids:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)  # type: ignore[arg-type]
            if not self._in_memory:
                self._manifest_path.write_text(
                    json.dumps(current_manifest, indent=2), encoding="utf-8"
                )

        return self._collection.count()

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        """Retrieves top K1 candidate chunks via dense bi-encoder cosine similarity."""
        total_docs = self._collection.count()
        if total_docs == 0 or not query.strip():
            return []

        n_results = min(top_k, total_docs)
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]

        candidates: List[RetrievedChunk] = []
        for rank_idx, (chunk_id, doc_text, meta, dist) in enumerate(
            zip(ids_list, docs_list, metas_list, distances_list), start=1
        ):
            similarity = max(0.0, min(1.0, round(1.0 - (float(dist) / 2.0), 4)))
            title = meta.get("title") if isinstance(meta, dict) else None
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=doc_text,
                    score=similarity,
                    source="internal_doc",
                    title=title,
                    initial_rank=rank_idx,
                )
            )

        return candidates
