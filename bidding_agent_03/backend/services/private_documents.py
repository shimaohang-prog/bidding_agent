"""私有文件向量层；所有查询和删除都强制所有权过滤。"""

import json
from functools import partial
from pathlib import Path
from typing import Any

import anyio

from common.embedding import encode_query, encode_texts
from common.milvus_config import close_milvus_client
from common.retrieval_models import Candidate


class PrivateDocumentStore:
    def __init__(self, uri: str, collection: str, limiter: anyio.CapacityLimiter) -> None:
        self.uri = str(Path(uri).resolve()) if "://" not in uri else uri
        self.collection = collection
        self.limiter = limiter

    def _client(self):
        from pymilvus import MilvusClient
        if "://" not in self.uri:
            Path(self.uri).parent.mkdir(parents=True, exist_ok=True)
        return MilvusClient(uri=self.uri)

    @staticmethod
    def _filter(user_id: str, conversation_id: str, file_ids: list[str] | None = None) -> str:
        parts = [
            f"user_id == {json.dumps(user_id)}",
            f"conversation_id == {json.dumps(conversation_id)}",
        ]
        if file_ids:
            parts.append(f"file_id in {json.dumps(file_ids)}")
        return " and ".join(parts)

    def _search_sync(self, query: str, user_id: str, conversation_id: str, file_ids: list[str]) -> list[Candidate]:
        if not file_ids:
            return []
        client = self._client()
        try:
            if not client.has_collection(self.collection):
                return []
            common = {
                "collection_name": self.collection,
                "limit": 8,
                "filter": self._filter(user_id, conversation_id, file_ids),
                "output_fields": ["file_id", "chunk_id", "original_name", "content"],
            }
            dense = client.search(
                **common, data=[encode_query(query)], anns_field="dense_vector",
                search_params={"metric_type": "COSINE", "params": {}},
            )
            sparse = client.search(
                **common, data=[query], anns_field="sparse_vector",
                search_params={"metric_type": "BM25", "params": {}},
            )
            merged: dict[str, Candidate] = {}
            for route, result in (("private:dense", dense), ("private:bm25", sparse)):
                for rank, raw in enumerate((result[0] if result else []), 1):
                    entity = raw.get("entity", {})
                    source_id = f"{entity.get('file_id')}:{entity.get('chunk_id')}"
                    item = merged.get(source_id)
                    if item is None:
                        item = Candidate(
                            source_type="private_document", category="private", source_id=source_id,
                            title=str(entity.get("original_name", "")), content=str(entity.get("content", "")),
                            metadata={"file_id": str(entity.get("file_id", "")), "original_name": str(entity.get("original_name", ""))},
                        )
                        merged[source_id] = item
                    item.retrieval_lists.append(route)
                    item.rank_positions[route] = rank
            for item in merged.values():
                item.fusion_score = sum(1 / (60 + rank) for rank in item.rank_positions.values())
            return sorted(merged.values(), key=lambda item: item.fusion_score, reverse=True)[:8]
        finally:
            close_milvus_client(client)

    async def search(self, query: str, user_id: str, conversation_id: str, file_ids: list[str]) -> list[Candidate]:
        return await anyio.to_thread.run_sync(
            partial(self._search_sync, query, user_id, conversation_id, file_ids), limiter=self.limiter
        )

    def _delete_sync(self, user_id: str, conversation_id: str, file_id: str) -> None:
        client = self._client()
        try:
            if client.has_collection(self.collection):
                client.delete(collection_name=self.collection, filter=self._filter(user_id, conversation_id, [file_id]))
        finally:
            close_milvus_client(client)

    async def delete_file(self, user_id: str, conversation_id: str, file_id: str) -> None:
        await anyio.to_thread.run_sync(
            partial(self._delete_sync, user_id, conversation_id, file_id), limiter=self.limiter
        )

    def ensure_collection(self, dimension: int) -> None:
        from pymilvus import DataType, Function, FunctionType
        client = self._client()
        try:
            if client.has_collection(self.collection):
                return
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dimension)
            schema.add_field("searchable_text", DataType.VARCHAR, max_length=65535, enable_analyzer=True, enable_match=True, analyzer_params={"tokenizer": "jieba", "filter": ["removepunct", "lowercase"]})
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("user_id", DataType.VARCHAR, max_length=36)
            schema.add_field("conversation_id", DataType.VARCHAR, max_length=36)
            schema.add_field("file_id", DataType.VARCHAR, max_length=36)
            schema.add_field("chunk_id", DataType.INT64)
            schema.add_field("original_name", DataType.VARCHAR, max_length=255)
            schema.add_field("content", DataType.VARCHAR, max_length=65535)
            schema.add_function(Function(name="private_bm25", function_type=FunctionType.BM25, input_field_names=["searchable_text"], output_field_names=["sparse_vector"]))
            indexes = client.prepare_index_params()
            indexes.add_index("dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
            indexes.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
            client.create_collection(collection_name=self.collection, schema=schema, index_params=indexes)
        finally:
            close_milvus_client(client)

    def index_chunks(self, *, user_id: str, conversation_id: str, file_id: str, original_name: str, chunks: list[str]) -> int:
        if not chunks:
            return 0
        vectors = encode_texts(chunks)
        self.ensure_collection(int(vectors.shape[1]))
        rows = [
            {
                "id": f"{file_id}:{index}", "dense_vector": vector.tolist(), "searchable_text": content,
                "user_id": user_id, "conversation_id": conversation_id, "file_id": file_id,
                "chunk_id": index, "original_name": original_name, "content": content,
            }
            for index, (content, vector) in enumerate(zip(chunks, vectors), 1)
        ]
        client = self._client()
        try:
            client.upsert(collection_name=self.collection, data=rows)
        finally:
            close_milvus_client(client)
        return len(rows)
