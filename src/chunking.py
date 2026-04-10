from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        sentences = [s.strip() for s in re.split(r"[.!?]+(?=\s)", text) if s.strip()]
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]
        if not self.separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        return self._split(text, list(self.separators))

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        text = current_text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        if not remaining_separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        if separator not in text:
            return self._split(text, next_separators)

        parts = [p for p in text.split(separator) if p]

        chunks: list[str] = []
        buffer = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # If a single part is too large, recurse on it.
            if len(part) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split(part, next_separators))
                continue

            candidate = part if not buffer else f"{buffer}{separator}{part}"
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = part

        if buffer:
            chunks.append(buffer)

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float] | str, vec_b: list[float] | str) -> float:
    """
    Compute cosine similarity between two vectors.

    Convenience: if both inputs are strings, they are embedded with the lab's
    deterministic mock embedder before computing cosine similarity.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if isinstance(vec_a, str) and isinstance(vec_b, str):
        # Use a lightweight token-hashing embedding for string inputs so this
        # helper can be used without heavy model dependencies.
        import hashlib

        dim = 512
        tokens_a = re.findall(r"[A-Za-z0-9]+", vec_a.lower())
        tokens_b = re.findall(r"[A-Za-z0-9]+", vec_b.lower())

        def _hash_embed(tokens: list[str]) -> list[float]:
            vec = [0.0] * dim
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            return [v / norm for v in vec]

        vec_a = _hash_embed(tokens_a)
        vec_b = _hash_embed(tokens_b)
    elif isinstance(vec_a, str) or isinstance(vec_b, str):
        raise TypeError("compute_similarity expects both inputs to be vectors or both to be strings")

    dot_product = _dot(vec_a, vec_b)
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text)
        sentence_chunks = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        recursive_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def _stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count else 0.0
            return {"count": count, "avg_length": avg_length, "chunks": chunks}

        return {
            "fixed_size": _stats(fixed_chunks),
            "by_sentences": _stats(sentence_chunks),
            "recursive": _stats(recursive_chunks),
        }
    
class ParentChildChunker:
    """
    Chiến lược Parent-Child Chunking:
    1. Chia tài liệu thành các khối lớn (Parent) để giữ ngữ cảnh rộng.
    2. Chia mỗi khối Parent thành các khối nhỏ hơn (Child) để thực hiện embedding/retrieval.
    
    Kết quả trả về thường là một danh sách các dictionary chứa thông tin liên kết 
    giữa Child và Parent.
    """

    def __init__(
        self, 
        parent_size: int = 1500, 
        child_size: int = 400, 
        child_overlap: int = 50
    ) -> None:
        self.parent_size = parent_size
        self.child_size = child_size
        # Sử dụng RecursiveChunker cho Parent để đảm bảo không cắt ngang đoạn văn
        self.parent_splitter = RecursiveChunker(chunk_size=parent_size)
        # Sử dụng FixedSizeChunker cho Child để tối ưu hóa việc tìm kiếm vector
        self.child_splitter = FixedSizeChunker(chunk_size=child_size, overlap=child_overlap)

    def chunk(self, text: str) -> list[dict]:
        if not text:
            return []

        # Bước 1: Tạo các khối Parent
        parent_chunks = self.parent_splitter.chunk(text)
        
        hierarchical_results = []

        for i, parent_text in enumerate(parent_chunks):
            # Bước 2: Với mỗi Parent, tạo các khối Child
            child_chunks = self.child_splitter.chunk(parent_text)
            
            for child_text in child_chunks:
                hierarchical_results.append({
                    "metadata": {
                        "parent_id": i,
                        "parent_length": len(parent_text)
                    },
                    "parent_content": parent_text,  # Dùng để cung cấp cho LLM
                    "child_content": child_text     # Dùng để thực hiện embedding
                })
        
        return hierarchical_results
