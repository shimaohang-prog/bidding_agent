from common.retrieval_models import Candidate, RetrievalPlan, RetrievalResult
from generation.context_builder import build_answer_context_with_citations


def _result(candidates: list[Candidate]) -> RetrievalResult:
    return RetrievalResult(
        plan=RetrievalPlan(),
        candidates=candidates,
    )


def _document_candidate(
    source: str,
    chunk_id: int,
    content: str,
    *,
    category: str = "laws",
) -> Candidate:
    return Candidate(
        source_type="category_vector",
        category=category,
        source_id=f"{source}#{chunk_id}",
        title=source,
        content=content,
        metadata={
            "source": source,
            "payload": {"source": source, "chunk_id": chunk_id},
        },
    )


def test_context_prefers_different_document_sources(monkeypatch):
    monkeypatch.setenv("MAX_CONTEXT_PER_CATEGORY", "3")
    monkeypatch.setenv("MAX_CONTEXT_PER_SOURCE", "1")
    candidates = [
        _document_candidate("a.txt", 1, "来源A第一条"),
        _document_candidate("a.txt", 2, "来源A第二条"),
        _document_candidate("b.txt", 1, "来源B第一条"),
        _document_candidate("c.txt", 1, "来源C第一条"),
    ]

    context, citations = build_answer_context_with_citations(
        "测试来源多样性",
        _result(candidates),
    )

    assert [item["source_id"] for item in citations] == [
        "a.txt#1",
        "b.txt#1",
        "c.txt#1",
    ]
    assert "来源A第二条" not in context


def test_context_backfills_when_only_one_document_is_available(monkeypatch):
    monkeypatch.setenv("MAX_CONTEXT_PER_CATEGORY", "3")
    monkeypatch.setenv("MAX_CONTEXT_PER_SOURCE", "1")
    candidates = [
        _document_candidate("a.txt", index, f"来源A第{index}条")
        for index in range(1, 4)
    ]

    _, citations = build_answer_context_with_citations(
        "测试同来源回填",
        _result(candidates),
    )

    assert len(citations) == 3


def test_context_trims_exact_overlap_between_adjacent_chunks():
    overlap = "这是用于验证相邻切片边界重叠的文本。" * 4
    candidates = [
        _document_candidate("law.txt", 1, f"第一段独有内容。{overlap}"),
        _document_candidate("law.txt", 2, f"{overlap}第二段独有内容。"),
    ]

    context, citations = build_answer_context_with_citations(
        "测试重叠裁剪",
        _result(candidates),
    )

    assert len(citations) == 2
    assert context.count(overlap) == 1
    assert "第一段独有内容" in context
    assert "第二段独有内容" in context


def test_context_keeps_similar_but_distinct_legal_clauses():
    candidates = [
        _document_candidate(
            "law.txt",
            1,
            "投标保证金不得超过项目估算价的百分之二。",
        ),
        _document_candidate(
            "law.txt",
            2,
            "履约保证金不得超过合同金额的百分之十。",
        ),
    ]

    context, citations = build_answer_context_with_citations(
        "测试相似条款",
        _result(candidates),
    )

    assert len(citations) == 2
    assert "投标保证金" in context
    assert "履约保证金" in context
