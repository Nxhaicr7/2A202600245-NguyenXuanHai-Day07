from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved_records = self.store.search(question, top_k=top_k)

        # 2. Xây dựng ngữ cảnh (context) từ các bản ghi lấy được
        # Ở bước trước trong hàm _make_record, chúng ta đã lưu text vào key "content"
        contexts = [record["content"] for record in retrieved_records]
        context_string = "\n\n---\n\n".join(contexts)

        prompt = f"""Vui lòng trả lời câu hỏi dựa trên các thông tin được cung cấp dưới đây. Nếu thông tin không có trong ngữ cảnh, hãy nói là bạn không biết, đừng tự bịa ra câu trả lời.

        [Ngữ cảnh bắt đầu]
        {context_string}
        [Ngữ cảnh kết thúc]

        Câu hỏi của người dùng: {question}

        Câu trả lời:"""

        return self.llm_fn(prompt)
