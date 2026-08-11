from pathlib import Path
from unittest.mock import MagicMock, patch

from common.retrieval_models import Candidate
from ranking.reranker import rerank_candidates
from ranking.result_fusion import normalise_vector_results, reciprocal_rank_fusion
from retrieval.category_vector_search import (
    _exact_tender_title_candidates,
    exact_tender_title_search,
)
from retrieval.retrieval_executor import execute_retrieval_plan
from common.retrieval_models import RetrievalPlan, SemanticQueryTask


TITLE = (
    "安徽芜湖三山经济开发区龙湖街道社区卫生服务中心"
    "关于硒鼓的网上超市采购项目成交公告"
)


def test_extracts_complete_title_from_plain_and_quoted_question():
    assert _exact_tender_title_candidates(TITLE) == [TITLE]
    assert _exact_tender_title_candidates(f"请核实“{TITLE}”") == [TITLE]


def test_exact_title_search_uses_scalar_title_and_marks_hit():
    client = MagicMock()
    client.has_collection.return_value = True
    client.query.return_value = [
        {
            "id": 2,
            "category": "tender",
            "subcategory": "",
            "source_id": "2",
            "title": TITLE,
            "content": '{"id":"2"}',
            "source": "data/csv/tender.csv",
            "metadata": {"id": "2", "tender_title": TITLE},
            "updated_at": "",
        }
    ]

    with (
        patch(
            "retrieval.category_vector_search.iter_existing_shards",
            return_value=[(None, Path("tender/main.db"))],
        ),
        patch(
            "retrieval.category_vector_search.get_milvus_client",
            return_value=client,
        ),
        patch("retrieval.category_vector_search.close_milvus_client"),
    ):
        hits, warnings = exact_tender_title_search(TITLE)

    assert warnings == []
    assert len(hits) == 1
    assert hits[0]["source_id"] == "2"
    assert hits[0]["exact_title_match"] is True
    assert client.query.call_args.kwargs["filter"].startswith("title == ")

    candidates = normalise_vector_results(
        [
            {
                "query_index": -1,
                "semantic_query": TITLE,
                "hit": hits[0],
            }
        ]
    )
    assert candidates[0].exact_title_match is True


def test_rrf_limit_never_drops_exact_title_match():
    exact = Candidate(
        source_type="category_vector",
        category="tender",
        source_id="2",
        title=TITLE,
        exact_title_match=True,
        retrieval_lists=["exact-title"],
        rank_positions={"exact-title": 100},
    )
    generic = Candidate(
        source_type="category_vector",
        category="tender",
        source_id="other",
        title="其他硒鼓成交公告",
        retrieval_lists=["dense", "bm25"],
        rank_positions={"dense": 1, "bm25": 1},
    )

    result = reciprocal_rank_fusion([generic, exact], limit=1)

    assert [item.source_id for item in result] == ["2"]


@patch("ranking.reranker.call_forced_tool")
def test_llm_reranker_cannot_demote_exact_title_below_generic(mock_tool):
    exact = Candidate(
        source_type="category_vector",
        category="tender",
        source_id="2",
        title=TITLE,
        exact_title_match=True,
        fusion_score=0.01,
    )
    generic = Candidate(
        source_type="category_vector",
        category="tender",
        source_id="other",
        title="其他硒鼓成交公告",
        fusion_score=0.10,
    )
    mock_tool.return_value = {
        "assessments": [
            {
                "candidate_id": exact.identity_key,
                "score": 0.01,
                "reason": "模型误判",
            },
            {
                "candidate_id": generic.identity_key,
                "score": 1.0,
                "reason": "泛化结果",
            },
        ]
    }

    result, warning = rerank_candidates(TITLE, [generic, exact], top_n=1)

    assert warning is None
    assert [item.source_id for item in result] == ["2"]


def test_sync_cli_executor_injects_exact_hit_before_bad_semantic_plan():
    exact_hit = {
        "category": "tender",
        "subcategory": None,
        "source_id": "2",
        "title": TITLE,
        "content": '{"id":"2"}',
        "source": "data/csv/tender.csv",
        "payload": {"id": "2", "tender_title": TITLE},
        "updated_at": "",
        "score": 1.0,
        "threshold": 0.65,
        "hybrid_score": 1.0,
        "exact_title_match": True,
        "route_positions": {"exact_title:main.db": 1},
        "filter_expression": f'title == "{TITLE}"',
        "db_path": "tender/main.db",
    }
    generic_hit = {
        **exact_hit,
        "source_id": "other",
        "title": "其他硒鼓成交公告",
        "payload": {"id": "other"},
        "exact_title_match": False,
        "route_positions": {"dense:main.db": 1, "bm25:main.db": 1},
    }
    plan = RetrievalPlan(
        semantic_queries=[
            SemanticQueryTask(
                query="硒鼓成交公告",
                categories=["tender", "product"],
            )
        ]
    )

    with (
        patch(
            "retrieval.retrieval_executor.exact_tender_title_search",
            return_value=([exact_hit], []),
        ),
        patch(
            "retrieval.retrieval_executor._execute_semantic_tasks",
            return_value=[
                {
                    "query_index": 0,
                    "semantic_query": "硒鼓成交公告",
                    "hit": generic_hit,
                }
            ],
        ),
        patch(
            "retrieval.retrieval_executor.rerank_candidates",
            side_effect=lambda _question, items, top_n: (list(items)[:top_n], None),
        ),
    ):
        result = execute_retrieval_plan(TITLE, plan)

    assert result.diagnostics["exact_title_hit_count"] == 1
    assert result.candidates[0].source_id == "2"
    assert result.candidates[0].exact_title_match is True
