# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Xuân Hải
**Nhóm:** D1
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
Nó cho thấy hai vector hướng về cùng một phía trong không gian đa chiều, ám chỉ sự tương đồng lớn về mặt ngữ nghĩa giữa hai đoạn văn bản, bất kể độ dài của chúng khác nhau thế nào.

**Ví dụ HIGH similarity:**
- Sentence A: Hôm nay tôi đi đá bóng
- Sentence B: Hôm nay tôi đá banh
- Tại sao tương đồng: 2 từ bóng với banh đều có nghĩa tương đương nhau, ý nghĩa 2 câu cũng tương đồng 

**Ví dụ LOW similarity:**
- Sentence A: Hôm nay tôi đi học
- Sentence B: Thời tiết Hà Nội nắng nhẹ, không mưa
- Tại sao khác: Vì 2 câu không liên quan đến nhau, nghĩa cũng khác hẳn nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
Vì Cosine similarity tập trung vào góc giữa hai vector thay vì khoảng cách tuyệt đối, nên ít bị ảnh hưởng bởi độ dài văn bản. Một đoạn ngắn và một đoạn dài nhưng cùng nội dung có thể vẫn có cosine similarity cao.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Gọi L là tổng số ký tự, S là kích thước chunk, O là độ chồng lấp.
Bước nhảy (Step) giữa các chunk là: S - O = 500 - 50 = 450.
Số lượng chunk (N) được tính theo công thức: N = ceil((L - O)/(S-O)) = ceil((10000-50)/(500-50)) = ceil(22.11)
> *Đáp án:* 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
Số lượng chunk sẽ tăng lên (trong trường hợp này là 25 chunks) vì bước nhảy giữa các chunk ngắn lại. Tăng overlap giúp tránh cắt đôi ý quan trọng ở ranh giới chunk, giữ ngữ cảnh tốt hơn cho retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Information Systems Management (IS governance/urbanization/alignment)

**Tại sao nhóm chọn domain này?**
Nhóm chọn domain này vì tài liệu có cấu trúc rõ ràng theo phần/chương/mục, rất phù hợp để thử nhiều chiến lược chunking. Các câu hỏi dạng “ai/khi nào/framework nào” cũng giúp đánh giá retrieval khá khách quan (chunk có chứa đúng fact hay không).

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | book.md | Tài liệu nhóm (export từ sách) | 503401 | `{"category": "information-systems", "source": "book.md"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | string | "information-systems" | Giúp lọc tài liệu theo domain khi hệ thống có nhiều nguồn khác nhau. |
| source | string | "book.md" | Cho phép truy xuất nguồn gốc của chunk, giúp xác minh thông tin và cung cấp thêm ngữ cảnh cho người dùng. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| book.md | FixedSizeChunker (`fixed_size`) | 2518 | 199.9 | Trung bình (cắt theo ký tự) |
| book.md | SentenceChunker (`by_sentences`) | 1160 | 428.2 | Tốt (giữ ranh giới câu) |
| book.md | RecursiveChunker (`recursive`) | 3484 | 142.9 | Khá (ưu tiên theo separator) |

### Strategy Của Tôi

**Loại:** Parent-Child Chunking

**Mô tả cách hoạt động:**
Chiến lược này chia tài liệu thành các khối lớn (Parent) để giữ ngữ cảnh rộng, sau đó chia mỗi Parent thành các khối nhỏ hơn (Child) để làm embedding/retrieval. Khi truy vấn, hệ thống tìm kiếm trên Child nhưng có thể “map” ngược lên Parent để cung cấp ngữ cảnh đầy đủ hơn cho phần trả lời.

**Tại sao tôi chọn strategy này cho domain nhóm?**
Với tài liệu dạng sách có nhiều chương/mục, Parent-Child giúp giảm tình trạng “chunk quá nhỏ thiếu ngữ cảnh” hoặc “chunk quá lớn khó match”. Child vừa đủ nhỏ để match query theo từ khóa/khái niệm, còn Parent giúp giữ phần giải thích liền mạch khi tổng hợp câu trả lời.

**Code snippet (nếu custom):**
```python
class ParentChildChunker:
    """
    Chiến lược Parent-Child: Chia Parent theo cấu trúc (Recursive) 
    và Child theo kích thước cố định (Fixed).
    """
    def __init__(self, parent_size: int = 1500, child_size: int = 400, child_overlap: int = 50):
        self.parent_splitter = RecursiveChunker(chunk_size=parent_size)
        self.child_splitter = FixedSizeChunker(chunk_size=child_size, overlap=child_overlap)

    def chunk(self, text: str) -> list[dict]:
        if not text: return []
        
        parents = self.parent_splitter.chunk(text)
        results = []

        for i, p_text in enumerate(parents):
            children = self.child_splitter.chunk(p_text)
            for c_text in children:
                results.append({
                    "child_content": c_text,
                    "parent_content": p_text,
                    "metadata": {"parent_id": i}
                })
        return results
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
|---|---|---:|---:|---|
| `docs/book.md` | fixed_size | 2518 | 199.9 | Good (avg_top1=0.430) |
| `docs/book.md` | by_sentences | 1160 | 428.2 | Good (avg_top1=0.398) |
| `docs/book.md` | recursive | 3484 | 142.9 | Good (avg_top1=0.394) |
| `docs/book.md` | parent_child | 1593 | 351.2 | Good (avg_top1=0.429) |


### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Semantic Chunking | 9.5 | Giữ trọn vẹn ngữ cảnh của từng mục, truy xuất chính xác. | Các chunk có thể rất lớn, không phù hợp với các mô hình có giới hạn context nhỏ. |
| Lê Minh Hoàng | SoftwareEngineeringChunker (Custom RecursiveTrunker) | 9 | Bảo tồn hoàn hảo cấu trúc tài liệu kỹ thuật nhờ ngắt theo Header; Giữ được mối liên kết logic. | Kích thước chunk trung bình lớn, gây tốn context window của mô hình. |
| Nguyễn Xuân Hải | Parent-Child Chunking | 8 | Child nhỏ giúp tìm kiếm vector đúng mục tiêu, ít nhiễu | Parent lớn có thể làm tăng context/cost |
| Nguyễn Đăng Hải | DocumentStructureChunker | 6.3 | Giữ ngữ cảnh theo heading/list/table; grounding tốt cho tài liệu dài | Phức tạp hơn và tốn xử lý hơn; lợi thế giảm khi dữ liệu ít cấu trúc |
| Thái Minh Kiên | Agentic Chunking | 8 | chunk giữ được ý nghĩa trọn vẹn, retrieval chính xác hơn, ít trả về nửa vời, Không cần một rule cố định cho mọi loại dữ liệu | Với dataset lớn cost sẽ tăng mạnh, chậm hơn pipeline thường, không ổn định tuyệt đối |
| Trần Trung Hậu | Token-Based Chunking (chia theo token) | 8 | Kiểm soát chính xác giới hạn đầu vào (context window) và chi phí API | Cắt máy móc, dễ làm đứt gãy ngữ nghĩa giữa chừng |
| Tạ Bảo Ngọc | Sliding Window + Overlap | 7/10 | Giữ vẹn câu/khối logic, tối ưu length | bị trùng dữ liệu -> tăng số chunk |

**Strategy nào tốt nhất cho domain này? Tại sao?**
Trong thử nghiệm với `book.md`, Parent-Child hoạt động ổn vì vừa giữ được ngữ cảnh theo “khối lớn” vừa cho phép retrieval match tốt theo “khối nhỏ”. Khi cần trả lời dạng fact (tên tác giả, năm xuất bản, framework), việc tách child hợp lý giúp top-k chứa chunk có đúng thông tin thường xuyên hơn.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
Hàm tách câu bằng regex `[.!?]+(?=\\s)` rồi nhóm theo `max_sentences_per_chunk`. Sau khi split, mình `strip()` và lọc bỏ chuỗi rỗng để tránh tạo chunk rỗng, giúp chunk ổn định hơn khi chạy trên nhiều loại văn bản.

**`RecursiveChunker.chunk` / `_split`** — approach:
Thuật toán thử tách theo danh sách separator (ưu tiên `\\n\\n`, rồi `\\n`, rồi `. `, rồi ` `). Nếu một phần vẫn quá dài thì đệ quy với separator “nhỏ hơn”; nếu hết separator thì fallback cắt theo `chunk_size` để luôn trả về kết quả và tránh treo.

### EmbeddingStore

**`add_documents` + `search`** — approach:
Mỗi document được embed một lần khi add và lưu kèm `metadata`. Khi search, mình embed query một lần và tính dot-product với toàn bộ embedding đã lưu, sau đó dùng min-heap để lấy `top_k` nhanh hơn (`O(n log k)`) thay vì sort toàn bộ.

**`search_with_filter` + `delete_document`** — approach:
`search_with_filter` lọc theo `metadata_filter` trước, rồi mới similarity-search trên tập ứng viên nhỏ hơn để nhanh hơn và chính xác hơn. `delete_document` loại bỏ toàn bộ record có `metadata['doc_id']` khớp và trả về `True/False` tùy có xóa được record nào hay không.

### KnowledgeBaseAgent

**`answer`** — approach:
Agent dùng RAG: lấy top-k chunks từ store, ghép thành một đoạn context, rồi build prompt gồm (1) hướng dẫn “chỉ trả lời theo ngữ cảnh”, (2) context, và (3) câu hỏi. Cách này giúp giảm hallucination vì câu trả lời bị “ràng buộc” bởi nội dung đã retrieve.

### Test Results

```
# Paste output of: pytest tests/ -v
(.venv) nxhai@nxhai:~/AI_thucchien/Day-07-Lab-Data-Foundations$ pytest tests/ -v
===================================================== test session starts ======================================================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0 -- /home/nxhai/AI_thucchien/Day-07-Lab-Data-Foundations/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/nxhai/AI_thucchien
configfile: pyproject.toml
collected 42 items                                                                                                             

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                    [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                             [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                      [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                       [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                            [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                            [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                  [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                   [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                 [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                   [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                              [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                          [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                           [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                               [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                         [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                               [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                     [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                       [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                             [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                  [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                    [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                        [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                     [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                              [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                             [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                        [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                    [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                               [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                   [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                         [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                   [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                              [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                             [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                 [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                            [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                     [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED           [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED               [100%]

====================================================== 42 passed in 0.03s ======================================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The cat sits on the mat | A cat is sitting on a mat | high | 0.354 | Đúng |
| 2 | I love pizza | The capital of France is Paris. | low | 0.000 | Đúng |
| 3 | Machine learning uses data to train models. | ML trains models using datasets | high | 0.169 | Đúng |
| 4 | He went to the bank to deposit money. | The river bank was covered in grass | low | 0.239 | Sai |
| 5 | Python is a programming language | Python is a type of snake | low | 0.548 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
Kết quả bất ngờ nhất là cặp 4 và 5: dù nghĩa khác nhau, similarity vẫn khá cao vì hai câu chia sẻ từ khóa giống nhau (bank, Python). Điều này cho thấy nếu embedding/đại diện dựa nhiều vào từ khóa (bag-of-words), nó sẽ khó phân biệt đồng âm khác nghĩa; cần embedding “ngữ cảnh” tốt hơn để tách biệt các trường hợp này.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Who is the series editor? | Jean-Charles Pomerol. |
| 2 | When was the book first published? | First published 2019 in Great Britain and the United States by ISTE Ltd and John Wiley & Sons, Inc. |
| 3 | List the authors of this work. | Daniel Alban; Philippe Eynaud; Julien Malaurent; Jean Loup Richet; Claudio Vitari. |
| 4 | What is the Library of Congress Control Number? | 2018967671 |
| 5 | Name two IS governance benchmarks mentioned. | COBIT, ITIL (also ValIT/RiskIT/GTAG/ISO/IEC are mentioned). |


### Kết Quả Của Tôi

| # | Query                                         | Top-1 Retrieved Chunk (tóm tắt)                                      | Score | Relevant? | Agent Answer (tóm tắt)                                                                              |
|---|-----------------------------------------------|----------------------------------------------------------------------|-------|-----------|-----------------------------------------------------------------------------------------------------|
| 1 | Who is the series editor? | Information Systems Management Series Editor Jean-Charles Pomerol... | 0.376 | Yes | Jean-Charles Pomerol |
| 2 | When was the book first published? | Information Systems Management Series Editor Jean-Charles Pomerol... | 0.370 | Yes | First published 2019 in Great Britain and the United States by ISTE Ltd and John Wiley & Sons, Inc. |
| 3 | List the authors of this work. | First Edition. Daniel Alban, Philippe Eynaud, Julien Malaurent... | 0.679 | Yes | First Edition. Daniel Alban, Philippe Eynaud, Julien Malaurent, Jean Loup Richet and Claudio Vitari. |
| 4 | What is the Library of Congress Control Number? | © ISTE Ltd 2019. The rights of Daniel Alban, Philippe Eynaud... | 0.326 | Yes | Library of Congress Control Number: 2018967671 |
| 5 | Name two IS governance benchmarks mentioned. | Infrastructure Library (ITIL), PMBOK and International Org... | 0.422 | Yes | COBIT, ValIT, RiskIT, ITIL, ISO |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
Mình học được rằng metadata filter (ví dụ theo `department`, `lang`, hoặc `source`) có thể cải thiện precision rất rõ nếu thiết kế schema ngay từ đầu. Ngoài ra, bạn ấy nhấn mạnh việc chọn `chunk_size/overlap` theo cấu trúc tài liệu (đoạn, tiêu đề) sẽ giúp chunk “đọc được” và retrieval ổn định hơn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
Nhóm khác demo cách benchmark retrieval bằng 5 câu hỏi cố định + gold answer, rồi ghi lại top‑3 chunks để kiểm tra thủ công “relevant hay không”. Cách làm này giúp phát hiện failure cases nhanh hơn so với chỉ nhìn score, và dễ so sánh giữa các chunking strategy.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
Mình sẽ chuẩn hóa dữ liệu đầu vào hơn (loại bỏ phần mục lục/lặp header/footer, giữ lại các section có nội dung) trước khi chunking để giảm nhiễu. Mình cũng sẽ thiết kế metadata schema sớm hơn và tạo bộ benchmark queries bám sát tài liệu ngay từ đầu để đo retrieval quality khách quan hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 4/ 5 |
| Document selection | Nhóm | 10/ 10 |
| Chunking strategy | Nhóm | 13/ 15 |
| My approach | Cá nhân | 8/ 10 |
| Similarity predictions | Cá nhân | 5/ 5 |
| Results | Cá nhân | 8/ 10 |
| Core implementation (tests) | Cá nhân | 25/ 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | 78**/ 90** |
