from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def _embedder_from_env():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    if provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    return _mock_embed


def _quality_label(avg_top1_score: float) -> str:
    if avg_top1_score >= 0.30:
        return "Good"
    if avg_top1_score >= 0.15:
        return "OK"
    return "Bad"


def print_chunk_report_md(doc_path: str, chunk_size: int = 200) -> int:
    """
    Print a Markdown table (per strategy):
    | Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
    """
    path = Path(doc_path)
    if not path.exists() or not path.is_file():
        print(f"Missing file: {doc_path}")
        return 1

    text = path.read_text(encoding="utf-8")
    embedder = _embedder_from_env()

    try:
        from src.chunking import ParentChildChunker  # type: ignore
    except Exception:
        ParentChildChunker = None  # type: ignore

    strategies = {
        "fixed_size": lambda t: FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(t),
        "by_sentences": lambda t: SentenceChunker(max_sentences_per_chunk=3).chunk(t),
        "recursive": lambda t: RecursiveChunker(chunk_size=chunk_size).chunk(t),
    }
    if ParentChildChunker is not None:
        strategies["parent_child"] = lambda t: ParentChildChunker().chunk(t)

    # Small automatic proxy for "retrieval quality": average top-1 score over a few generic queries.
    queries = [
        "Summarize the key information.",
        "What is this document about?",
        "Give 3 important points.",
    ]

    print("| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |")
    print("|---|---|---:|---:|---|")

    for name, chunk_fn in strategies.items():
        chunks = chunk_fn(text)

        # Normalize to list[str] (ParentChildChunker returns list[dict]).
        if chunks and isinstance(chunks[0], dict):
            child_texts = [c.get("child_content", "") for c in chunks]  # type: ignore[union-attr]
            chunk_texts = [c for c in child_texts if c]
        else:
            chunk_texts = [c for c in chunks if c]  # type: ignore[assignment]

        count = len(chunk_texts)
        avg_len = (sum(len(c) for c in chunk_texts) / count) if count else 0.0

        store = EmbeddingStore(collection_name=f"report_{name}", embedding_fn=embedder)
        store.add_documents(
            [
                Document(
                    id=f"{path.stem}_{name}_{i}",
                    content=chunk,
                    metadata={"source": str(path), "strategy": name},
                )
                for i, chunk in enumerate(chunk_texts)
            ]
        )

        top1_scores: list[float] = []
        for q in queries:
            res = store.search(q, top_k=1)
            if res:
                top1_scores.append(float(res[0].get("score", 0.0)))
        avg_top1 = (sum(top1_scores) / len(top1_scores)) if top1_scores else 0.0
        quality = f"{_quality_label(avg_top1)} (avg_top1={avg_top1:.3f})"

        print(f"| `{doc_path}` | {name} | {count} | {avg_len:.1f} | {quality} |")

    return 0


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    embedder = _embedder_from_env()

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--chunk-report":
        return print_chunk_report_md(args[1])

    if len(args) >= 2 and args[0] == "--parent-child":
        doc_path = args[1]
        question = " ".join(args[2:]).strip() if len(args) > 2 else "What is this document about?"
        top_k = 3
        for i, a in enumerate(args):
            if a == "--top-k" and i + 1 < len(args):
                try:
                    top_k = int(args[i + 1])
                except Exception:
                    top_k = 3

        path = Path(doc_path)
        if not path.exists() or not path.is_file():
            print(f"Missing file: {doc_path}")
            return 1

        from src.chunking import ParentChildChunker

        text = path.read_text(encoding="utf-8")
        chunks = ParentChildChunker().chunk(text)
        child_texts = [c.get("child_content", "") for c in chunks if c.get("child_content")]

        embedder = _embedder_from_env()
        store = EmbeddingStore(collection_name="parent_child_store", embedding_fn=embedder)
        store.add_documents(
            [
                Document(
                    id=f"{path.stem}_pc_{i}",
                    content=child,
                    metadata={"source": str(path), "strategy": "parent_child"},
                )
                for i, child in enumerate(child_texts)
            ]
        )

        print(f"Stored {store.get_collection_size()} child chunks from {doc_path}")
        print(f"Query: {question}")
        results = store.search(question, top_k=top_k)
        for idx, r in enumerate(results, start=1):
            print(f"{idx}. score={r['score']:.3f} source={r['metadata'].get('source')}")
            print(f"   child preview: {r['content'][:200].replace(chr(10), ' ')}...")
        return 0

    question = " ".join(args).strip() if args else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
