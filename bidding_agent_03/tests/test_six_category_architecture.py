import unittest
from unittest.mock import patch

from builders.build_vectors import _payload_json, _semantic_text
from common.milvus_config import (
    ALL_CATEGORIES,
    CATEGORY_SPECS,
    VECTOR_CATEGORIES,
    category_db_path,
)
from common.retrieval_models import Candidate, RetrievalPlan, SemanticQueryTask
from planning.retrieval_planner import create_retrieval_plan
from ranking.result_fusion import reciprocal_rank_fusion
from retrieval.retrieval_executor import should_use_web_search


class SixCategoryArchitectureTests(unittest.TestCase):
    def test_six_categories_and_news_web_only(self):
        self.assertEqual(
            ALL_CATEGORIES,
            (
                "enterprise",
                "tender",
                "product",
                "laws",
                "policy",
                "news",
            ),
        )
        self.assertNotIn("news", VECTOR_CATEGORIES)
        self.assertFalse(CATEGORY_SPECS["news"].vector_enabled)
        with self.assertRaises(ValueError):
            category_db_path("news")

    def test_categories_use_different_physical_db_paths(self):
        paths = {
            category_db_path(category)
            for category in VECTOR_CATEGORIES
        }
        self.assertEqual(len(paths), len(VECTOR_CATEGORIES))

    def test_subcategory_uses_child_database(self):
        parent = category_db_path("product")
        child = category_db_path("product", "办公设备")
        self.assertNotEqual(parent, child)
        self.assertEqual(child.parent.name, "subcategories")

    def test_plan_contains_no_exact_query_contract(self):
        plan = RetrievalPlan(
            semantic_queries=[
                SemanticQueryTask(
                    query="安徽网络安全项目和政策",
                    categories=["tender", "policy"],
                )
            ]
        )
        self.assertNotIn("exact_queries", plan.model_dump())

    @patch(
        "planning.retrieval_planner.call_forced_tool",
        side_effect=RuntimeError("offline"),
    )
    def test_rule_fallback_keeps_multi_category(self, _mock):
        plan = create_retrieval_plan("查询网络安全招标项目和相关政策")
        categories = plan.semantic_queries[0].categories
        self.assertIn("tender", categories)
        self.assertIn("policy", categories)

    def test_news_forces_web(self):
        plan = RetrievalPlan(
            semantic_queries=[
                SemanticQueryTask(
                    query="最新招投标新闻",
                    categories=["news"],
                )
            ]
        )
        use_web, _reason = should_use_web_search(plan, [])
        self.assertTrue(use_web)

    def test_rrf_deduplicates_same_vector_record(self):
        first = Candidate(
            source_type="category_vector",
            category="tender",
            source_id="12",
            content='{"id": "12"}',
            retrieval_lists=["semantic:0:tender"],
            rank_positions={"semantic:0:tender": 1},
        )
        second = Candidate(
            source_type="category_vector",
            category="tender",
            source_id="12",
            content='{"id": "12"}',
            retrieval_lists=["semantic:1:tender"],
            rank_positions={"semantic:1:tender": 2},
        )
        result = reciprocal_rank_fusion([first, second])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].rank_positions), 2)

    def test_structured_vector_keeps_identifiers_in_semantic_text(self):
        text = _semantic_text(
            "enterprise",
            {
                "enterprise_name": "测试公司",
                "uscc": "91340000TEST000001",
                "industry": "软件",
            },
        )
        self.assertIn("测试公司", text)
        self.assertIn("91340000TEST000001", text)

    def test_payload_keeps_complete_nonempty_row(self):
        payload = {
            "id": "1",
            "enterprise_name": "测试公司",
            "registered_capital_amount": "5000000.00",
        }
        encoded = _payload_json(payload)
        self.assertIn("registered_capital_amount", encoded)
        self.assertIn("5000000.00", encoded)


if __name__ == "__main__":
    unittest.main()

