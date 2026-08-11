import unittest

from common.retrieval_models import Candidate, RetrievalPlan, SemanticQueryTask
from ranking.result_fusion import reciprocal_rank_fusion


class RetrievalModelTests(unittest.TestCase):
    def test_plan_has_only_semantic_tasks(self):
        plan = RetrievalPlan(
            semantic_queries=[
                SemanticQueryTask(
                    query="网络安全项目和相关政策",
                    categories=["tender", "policy"],
                )
            ]
        )
        self.assertNotIn("exact_queries", plan.model_dump())

    def test_rrf_merges_dense_and_bm25_for_same_record(self):
        dense = Candidate(
            source_type="category_vector",
            category="tender",
            source_id="12",
            content='{"id": "12"}',
            retrieval_lists=["semantic:0:tender:dense:main.db"],
            rank_positions={"semantic:0:tender:dense:main.db": 2},
        )
        bm25 = Candidate(
            source_type="category_vector",
            category="tender",
            source_id="12",
            content='{"id": "12"}',
            retrieval_lists=["semantic:0:tender:bm25:main.db"],
            rank_positions={"semantic:0:tender:bm25:main.db": 1},
        )
        result = reciprocal_rank_fusion([dense, bm25])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].rank_positions), 2)

    def test_rrf_uses_rank_not_raw_scores(self):
        first = Candidate(
            source_type="category_vector",
            category="laws",
            source_id="a#1",
            original_score=0.2,
            retrieval_lists=["semantic:0:laws:dense:main.db"],
            rank_positions={"semantic:0:laws:dense:main.db": 1},
        )
        second = Candidate(
            source_type="category_vector",
            category="policy",
            source_id="b#1",
            original_score=99.0,
            retrieval_lists=["semantic:0:policy:bm25:main.db"],
            rank_positions={"semantic:0:policy:bm25:main.db": 3},
        )
        result = reciprocal_rank_fusion([first, second])
        self.assertEqual(result[0].source_id, "a#1")


if __name__ == "__main__":
    unittest.main()
