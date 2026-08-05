import json
import unittest
from unittest.mock import patch

from builders.build_vectors import _typed_metadata
from common.retrieval_models import MetadataCondition, SemanticQueryTask
from planning.retrieval_planner import create_retrieval_plan
from ranking.result_fusion import normalise_vector_results
from retrieval.metadata_filter import compile_metadata_filter
from retrieval.web_search import extract_requested_domains, web_search


class MetadataFilterTests(unittest.TestCase):
    def test_empty_filter_array_is_normalised_for_deepseek_compatibility(self):
        task = SemanticQueryTask.model_validate(
            {
                "query": "包子机价格",
                "categories": ["product"],
                "metadata_filters": [],
            }
        )
        self.assertEqual(task.metadata_filters, {})

    def test_compiles_json_string_and_numeric_range(self):
        expression = compile_metadata_filter(
            "tender",
            [
                MetadataCondition(
                    field="province",
                    operator="eq",
                    value="安徽省",
                ),
                MetadataCondition(
                    field="bid_amount",
                    operator="gte",
                    value=5000000,
                ),
            ],
        )
        self.assertIn('metadata["province"] == "安徽省"', expression)
        self.assertIn('metadata["bid_amount"] >= 5000000', expression)

    def test_rejects_non_whitelisted_field(self):
        with self.assertRaises(ValueError):
            compile_metadata_filter(
                "tender",
                [
                    MetadataCondition(
                        field='province"] == "安徽省" or id',
                        operator="gte",
                        value=0,
                    )
                ],
            )

    def test_filter_value_is_json_escaped(self):
        value = '安徽省" or id > 0 or province == "'
        expression = compile_metadata_filter(
            "enterprise",
            [
                MetadataCondition(
                    field="province",
                    operator="eq",
                    value=value,
                )
            ],
        )
        self.assertEqual(
            expression,
            f'metadata["province"] == '
            f"{json.dumps(value, ensure_ascii=False)}",
        )

    def test_in_requires_nonempty_list(self):
        with self.assertRaises(ValueError):
            compile_metadata_filter(
                "product",
                [
                    MetadataCondition(
                        field="city",
                        operator="in",
                        value=[],
                    )
                ],
            )

    def test_builder_types_amount_metadata(self):
        metadata = _typed_metadata(
            {
                "id": "1",
                "bid_amount": "5,000,000.00",
                "province": "安徽省",
            }
        )
        self.assertEqual(metadata["bid_amount"], 5000000.0)
        self.assertEqual(metadata["province"], "安徽省")

    @patch(
        "planning.retrieval_planner.call_forced_tool",
        side_effect=RuntimeError("offline"),
    )
    def test_fallback_extracts_province_filter(self, _mock):
        plan = create_retrieval_plan("查询安徽省网络安全招标项目")
        filters = plan.semantic_queries[0].metadata_filters
        self.assertEqual(
            filters["tender"][0].model_dump(),
            {
                "field": "province",
                "operator": "eq",
                "value": "安徽省",
            },
        )

    @patch(
        "planning.retrieval_planner.call_forced_tool",
        side_effect=RuntimeError("offline"),
    )
    def test_explicit_url_forces_web_search(self, _mock):
        plan = create_retrieval_plan(
            "基于 https://b2b.baidu.com 联网搜索包子机价格"
        )
        self.assertTrue(plan.requires_web_search)


class HybridFusionTests(unittest.TestCase):
    def test_dense_and_bm25_keep_separate_rrf_lists(self):
        candidates = normalise_vector_results(
            [
                {
                    "query_index": 0,
                    "semantic_query": "网络安全招标",
                    "hit": {
                        "category": "tender",
                        "subcategory": None,
                        "source_id": "12",
                        "title": "网络安全项目",
                        "content": '{"id":"12"}',
                        "source": "data/csv/tender.csv",
                        "payload": {"id": "12"},
                        "updated_at": "",
                        "score": 0.03,
                        "threshold": 0.65,
                        "dense_score": 0.82,
                        "bm25_score": 4.5,
                        "hybrid_score": 0.03,
                        "route_positions": {
                            "dense:main.db": 2,
                            "bm25:main.db": 1,
                        },
                        "filter_expression": (
                            'metadata["province"] == "安徽省"'
                        ),
                        "db_path": "tender/main.db",
                    },
                }
            ]
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0].rank_positions), 2)
        self.assertTrue(
            any("dense:" in key for key in candidates[0].rank_positions)
        )
        self.assertTrue(
            any("bm25:" in key for key in candidates[0].rank_positions)
        )


class WebDomainRoutingTests(unittest.TestCase):
    def test_extracts_explicit_baidu_domain(self):
        self.assertEqual(
            extract_requested_domains(
                "基于 https://b2b.baidu.com 联网搜索包子机价格"
            ),
            ["b2b.baidu.com"],
        )

    @patch("retrieval.web_search._request")
    def test_explicit_domain_overrides_government_domains(self, request):
        request.return_value = {
            "results": [
                {
                    "title": "无关政府网页",
                    "content": "无关内容",
                    "url": "https://www.gov.cn/example",
                },
                {
                    "title": "包子机",
                    "content": "参考价格",
                    "url": "https://b2b.baidu.com/example",
                }
            ]
        }
        results = web_search(
            "基于 https://b2b.baidu.com 联网搜索包子机价格",
            categories=[
                "product",
                "tender",
                "laws",
                "policy",
                "news",
            ],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(
            request.call_args.kwargs["include_domains"],
            ["b2b.baidu.com"],
        )
        self.assertEqual(
            request.call_args.kwargs["query"],
            "包子机价格",
        )
        self.assertEqual(request.call_count, 1)
        self.assertEqual(
            results[0]["url"],
            "https://b2b.baidu.com/example",
        )


if __name__ == "__main__":
    unittest.main()
