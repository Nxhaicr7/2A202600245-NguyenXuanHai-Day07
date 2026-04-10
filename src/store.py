from __future__ import annotations

import heapq
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)

        record_id = str(self._next_index)
        self._next_index += 1

        return {
            "id": record_id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not records:
            return []

        query_embedding = self._embedding_fn(query)
        dot = _dot
        heappush = heapq.heappush
        heapreplace = heapq.heapreplace

        heap: list[tuple[float, int, dict[str, Any]]] = []
        for idx, record in enumerate(records):
            embedding = record.get("embedding")
            if embedding is None:
                continue
            score = dot(query_embedding, embedding)
            if len(heap) < top_k:
                heappush(heap, (score, idx, record))
            elif score > heap[0][0]:
                heapreplace(heap, (score, idx, record))

        if not heap:
            return []

        heap.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict[str, Any]] = []
        for score, _idx, record in heap:
            results.append(
                {
                    "id": record.get("id"),
                    "content": record.get("content"),
                    "metadata": record.get("metadata"),
                    "score": score,
                }
            )
        return results

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        if self._use_chroma and self._collection is not None:
            records = [self._make_record(doc) for doc in docs]
            self._collection.add(
                ids=[r["id"] for r in records],
                documents=[r["content"] for r in records],
                embeddings=[r["embedding"] for r in records],
                metadatas=[r["metadata"] for r in records],
            )
            return

        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances", "ids"],
            )
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            results: list[dict[str, Any]] = []
            for rid, content, meta, dist in zip(ids, docs, metas, dists):
                results.append({"id": rid, "content": content, "metadata": meta, "score": -float(dist)})
            return results

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            try:
                return int(self._collection.count())
            except Exception:
                return 0
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter,
                include=["documents", "metadatas", "distances", "ids"],
            )
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            results: list[dict[str, Any]] = []
            for rid, content, meta, dist in zip(ids, docs, metas, dists):
                results.append({"id": rid, "content": content, "metadata": meta, "score": -float(dist)})
            return results

        mf_items = tuple(metadata_filter.items())
        filtered: list[dict[str, Any]] = []
        for record in self._store:
            meta = record.get("metadata") or {}
            if all(meta.get(k) == v for k, v in mf_items):
                filtered.append(record)
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            before = self.get_collection_size()
            try:
                self._collection.delete(where={"doc_id": doc_id})
            except Exception:
                return False
            after = self.get_collection_size()
            return after < before

        before = len(self._store)
        if before == 0:
            return False

        kept: list[dict[str, Any]] = []
        for record in self._store:
            meta = record.get("metadata") or {}
            if meta.get("doc_id") != doc_id:
                kept.append(record)
        self._store = kept
        return len(self._store) < before
