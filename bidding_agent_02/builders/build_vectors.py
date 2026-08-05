# -*- coding: utf-8 -*-
"""从 CSV/TXT 构建物理隔离的分类和子分类向量库。"""

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from common.embedding import encode_texts
from common.milvus_config import (
    ALL_CATEGORIES,
    BUSINESS_CATEGORIES,
    COLLECTION_NAME,
    PROJECT_ROOT,
    VECTOR_CATEGORIES,
    VECTOR_DB_ROOT,
    category_db_dir,
    category_db_path,
    close_milvus_client,
    get_category_spec,
    get_milvus_client,
    subcategory_key,
)


CSV_DIR = PROJECT_ROOT / "data" / "csv"
TEXT_DIRS = {
    "laws": PROJECT_ROOT / "data" / "laws",
    "policy": PROJECT_ROOT / "data" / "policy",
    "news": PROJECT_ROOT / "data" / "news",
}
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")
TXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")
DEFAULT_BATCH_SIZE = 256
MAX_PAYLOAD_CHARS = 60_000
MAX_SEARCHABLE_TEXT_CHARS = 65_000
NUMERIC_METADATA_FIELDS = {
    "registered_capital_amount",
    "bid_amount",
    "amount",
}

SEMANTIC_FIELDS = {
    "enterprise": (
        ("enterprise_name", "企业名称"),
        ("uscc", "统一社会信用代码"),
        ("corporation", "法定代表人"),
        ("industry", "所属行业"),
        ("enterprise_type", "企业类型"),
        ("status", "经营状态"),
        ("province", "省份"),
        ("city", "城市"),
        ("district", "区县"),
        ("location", "地址"),
        ("content", "经营与企业介绍"),
    ),
    "tender": (
        ("tender_title", "项目名称"),
        ("project_type", "项目类型"),
        ("source_name", "信息来源"),
        ("purchasing_staff", "采购人"),
        ("bid_company", "中标企业"),
        ("province", "省份"),
        ("city", "城市"),
        ("town", "区县"),
        ("bid_amount", "中标金额"),
        ("bid_date", "中标日期"),
        ("content", "项目内容"),
    ),
    "product": (
        ("title", "产品名称"),
        ("major_category", "一级分类"),
        ("middle_category", "二级分类"),
        ("supplier_name", "供应商"),
        ("amount", "金额"),
        ("currency", "币种"),
        ("province", "省份"),
        ("city", "城市"),
        ("supplier_address", "供应商地址"),
        ("content", "产品介绍"),
    ),
}
TITLE_FIELDS = {
    "enterprise": "enterprise_name",
    "tender": "tender_title",
    "product": "title",
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _detect_encoding(path: Path, candidates: Iterable[str]) -> str:
    errors: list[str] = []
    for encoding in candidates:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                while handle.read(1024 * 1024):
                    pass
            return encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"无法识别文件编码：{path}；{errors}")


def _stable_int64(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:15], 16)


def _payload_json(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > MAX_PAYLOAD_CHARS:
        raise ValueError(
            f"单条完整 payload 超过 {MAX_PAYLOAD_CHARS} 字符，"
            "为避免静默丢失业务事实，本次拒绝写入"
        )
    return text


def _typed_metadata(payload: dict[str, str]) -> dict[str, Any]:
    """保留完整元数据，并把允许范围过滤的金额字段转换为数值。"""
    output: dict[str, Any] = dict(payload)
    for field in NUMERIC_METADATA_FIELDS:
        value = output.get(field)
        if value in {None, ""}:
            continue
        try:
            output[field] = float(str(value).replace(",", ""))
        except ValueError:
            output.pop(field, None)
    return output


def _searchable_text(value: str) -> str:
    text = str(value or "").strip()
    if len(text) > MAX_SEARCHABLE_TEXT_CHARS:
        raise ValueError(
            f"BM25 文本超过 {MAX_SEARCHABLE_TEXT_CHARS} 字符，"
            "请先按逻辑边界切片"
        )
    return text


def _csv_files(category: str) -> list[Path]:
    output: set[Path] = set()
    for pattern in (
        f"{category}.csv",
        f"{category}_*.csv",
        f"{category}-*.csv",
    ):
        output.update(path.resolve() for path in CSV_DIR.glob(pattern))
    category_dir = CSV_DIR / category
    if category_dir.is_dir():
        output.update(path.resolve() for path in category_dir.rglob("*.csv"))
    return sorted(path for path in output if path.is_file())


def _file_subcategory(path: Path, category: str) -> str | None:
    category_dir = (CSV_DIR / category).resolve()
    try:
        relative = path.resolve().relative_to(category_dir)
    except ValueError:
        return None
    return relative.parts[0] if len(relative.parts) > 1 else None


def _semantic_text(category: str, row: dict[str, Any]) -> str:
    parts = []
    for field, label in SEMANTIC_FIELDS[category]:
        value = _clean(row.get(field))
        if value:
            parts.append(f"{label}：{value}")
    return "\n".join(parts)


def load_structured_records(
    category: str,
    subcategory_field: str | None = None,
) -> list[dict[str, Any]]:
    files = _csv_files(category)
    if not files:
        return []
    records: dict[str, dict[str, Any]] = {}
    for path in files:
        encoding = _detect_encoding(path, CSV_ENCODINGS)
        source = path.relative_to(PROJECT_ROOT).as_posix()
        directory_subcategory = _file_subcategory(path, category)
        with path.open("r", encoding=encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            if "id" not in (reader.fieldnames or []):
                raise ValueError(f"CSV 缺少 id 字段：{path}")
            if (
                subcategory_field
                and subcategory_field not in (reader.fieldnames or [])
            ):
                raise ValueError(
                    f"{path.name} 不包含子分类字段 {subcategory_field}"
                )
            for row in reader:
                source_id = _clean(row.get("id"))
                if not source_id:
                    raise ValueError(f"CSV 存在空 id：{path}")
                semantic_text = _semantic_text(category, row)
                if not semantic_text:
                    continue
                subcategory = (
                    _clean(row.get(subcategory_field))
                    if subcategory_field
                    else directory_subcategory
                ) or None
                payload = {
                    key: _clean(value)
                    for key, value in row.items()
                    if _clean(value)
                }
                payload_text = _payload_json(payload)
                key = f"{category}:{source_id}"
                records[key] = {
                    "id": _stable_int64(key),
                    "category": category,
                    "subcategory": subcategory or "",
                    "source_id": source_id,
                    "title": _clean(row.get(TITLE_FIELDS[category])),
                    "content": payload_text,
                    "source": source,
                    "metadata": _typed_metadata(payload),
                    "updated_at": _clean(row.get("updated_at")),
                    "searchable_text": _searchable_text(semantic_text),
                }
    return list(records.values())


def _split_general_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 120,
) -> list[str]:
    paragraphs = [
        item.strip()
        for item in re.split(r"\n\s*\n", text)
        if item.strip()
    ]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + chunk_size])
                start += max(1, chunk_size - overlap)
            continue
        proposed = f"{current}\n\n{paragraph}".strip()
        if current and len(proposed) > chunk_size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}".strip()
        else:
            current = proposed
    if current:
        chunks.append(current)
    return chunks


def _split_document(path: Path, text: str) -> list[str]:
    if "问" in path.stem:
        starts = list(
            re.finditer(
                r"(?m)^(?=(?:问题\s*)?Q?\d{1,5}\s*[.、：:])",
                text,
            )
        )
        if len(starts) >= 5:
            blocks = []
            for index, match in enumerate(starts):
                end = (
                    starts[index + 1].start()
                    if index + 1 < len(starts)
                    else len(text)
                )
                block = text[match.start() : end].strip()
                if block:
                    blocks.append(block)
            return blocks
    return _split_general_text(text)


def load_document_records(category: str) -> list[dict[str, Any]]:
    root = TEXT_DIRS[category]
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.txt")):
        encoding = _detect_encoding(path, TXT_ENCODINGS)
        text = path.read_text(encoding=encoding).strip()
        if not text:
            continue
        relative = path.relative_to(root)
        subcategory = relative.parts[0] if len(relative.parts) > 1 else None
        source = path.relative_to(PROJECT_ROOT).as_posix()
        for index, content in enumerate(
            _split_document(path, text),
            start=1,
        ):
            source_id = f"{source}#{index}"
            payload = {
                "source": source,
                "chunk_id": index,
                "category": category,
                "subcategory": subcategory,
            }
            records.append(
                {
                    "id": _stable_int64(f"{category}:{source_id}"),
                    "category": category,
                    "subcategory": subcategory or "",
                    "source_id": source_id,
                    "title": path.stem,
                    "content": content,
                    "source": source,
                    "metadata": payload,
                    "updated_at": "",
                    "searchable_text": _searchable_text(content),
                }
            )
    return records


def _batches(
    records: list[dict[str, Any]],
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _assert_safe_db_path(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(VECTOR_DB_ROOT.resolve()):
        raise ValueError(f"拒绝操作向量库根目录外的文件：{resolved}")
    if resolved.suffix.lower() != ".db":
        raise ValueError(f"拒绝操作非 .db 文件：{resolved}")


def _prepare_new_db(path: Path, rebuild: bool) -> None:
    _assert_safe_db_path(path)
    if path.exists() and not rebuild:
        raise FileExistsError(
            f"向量库已存在：{path}；确认后使用 --rebuild 重建"
        )
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _remove_stale_category_dbs(category: str) -> None:
    """全量重建时清理已经不再出现的旧子分类分片。"""
    root = category_db_dir(category)
    if not root.is_dir():
        return
    for path in root.glob("*.db"):
        _assert_safe_db_path(path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    sub_dir = root / "subcategories"
    for path in sub_dir.glob("*.db") if sub_dir.is_dir() else []:
        _assert_safe_db_path(path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    manifest = root / "manifest.json"
    if manifest.is_file():
        manifest.unlink()


def _create_collection(path: Path, dimension: int) -> None:
    from pymilvus import DataType, Function, FunctionType

    client = get_milvus_client(path, create_parent=True)
    try:
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            "id",
            DataType.INT64,
            is_primary=True,
        )
        schema.add_field(
            "dense_vector",
            DataType.FLOAT_VECTOR,
            dim=dimension,
        )
        schema.add_field(
            "searchable_text",
            DataType.VARCHAR,
            max_length=65_535,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params={
                "tokenizer": "jieba",
                "filter": ["removepunct", "lowercase"],
            },
        )
        schema.add_field(
            "sparse_vector",
            DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field("category", DataType.VARCHAR, max_length=32)
        schema.add_field("subcategory", DataType.VARCHAR, max_length=512)
        schema.add_field("source_id", DataType.VARCHAR, max_length=1024)
        schema.add_field("title", DataType.VARCHAR, max_length=4096)
        schema.add_field("content", DataType.VARCHAR, max_length=65_535)
        schema.add_field("source", DataType.VARCHAR, max_length=4096)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("updated_at", DataType.VARCHAR, max_length=128)
        schema.add_function(
            Function(
                name="searchable_text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["searchable_text"],
                output_field_names=["sparse_vector"],
            )
        )
        index_params = client.prepare_index_params()
        index_params.add_index(
            "dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            "sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
    finally:
        close_milvus_client(client)


def build_shard(
    category: str,
    subcategory: str | None,
    records: list[dict[str, Any]],
    batch_size: int,
    rebuild: bool,
) -> int:
    if not records:
        return 0
    path = category_db_path(category, subcategory)
    _prepare_new_db(path, rebuild)

    first_vectors = encode_texts(
        [item["searchable_text"] for item in records[:1]]
    )
    _create_collection(path, int(first_vectors.shape[1]))

    inserted = 0
    for batch in _batches(records, batch_size):
        vectors = encode_texts([item["searchable_text"] for item in batch])
        milvus_rows = []
        for item, vector in zip(batch, vectors):
            row = dict(item)
            row["dense_vector"] = vector.tolist()
            milvus_rows.append(row)

        client = get_milvus_client(path)
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                data=milvus_rows,
            )
        finally:
            close_milvus_client(client)
        inserted += len(batch)
        print(
            f"{category}/{subcategory or 'main'}："
            f"{inserted}/{len(records)}"
        )

    final_client = get_milvus_client(path)
    try:
        final_client.load_collection(collection_name=COLLECTION_NAME)
    finally:
        close_milvus_client(final_client)
    return inserted


def _write_manifest(
    category: str,
    groups: dict[str | None, list[dict[str, Any]]],
) -> None:
    path = category_db_dir(category) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "category": category,
        "vector_enabled": True,
        "shards": [
            {
                "subcategory": label,
                "key": subcategory_key(label) if label else "main",
                "records": len(records),
            }
            for label, records in groups.items()
        ],
    }
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_category(
    category: str,
    *,
    subcategory_field: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rebuild: bool = False,
) -> dict[str, int]:
    spec = get_category_spec(category)
    if not spec.vector_enabled:
        TEXT_DIRS["news"].mkdir(parents=True, exist_ok=True)
        print("news 为联网分类：已保留 data/news，不创建向量数据库。")
        return {}

    if category in BUSINESS_CATEGORIES:
        records = load_structured_records(category, subcategory_field)
    else:
        if subcategory_field:
            raise ValueError("laws/policy 通过一级子目录划分子分类")
        records = load_document_records(category)
    if not records:
        print(f"{category} 没有可构建数据，跳过。")
        return {}

    if rebuild:
        _remove_stale_category_dbs(category)

    groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["subcategory"] or None].append(record)

    stats: dict[str, int] = {}
    for subcategory, shard_records in groups.items():
        stats[subcategory or "main"] = build_shard(
            category,
            subcategory,
            shard_records,
            max(1, min(int(batch_size), 2000)),
            rebuild,
        )
    _write_manifest(category, groups)
    return stats


def _parse_subcategory_fields(
    items: list[str],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("子分类字段格式必须是 category=field")
        category, field = (part.strip() for part in item.split("=", 1))
        if category not in BUSINESS_CATEGORIES or not field:
            raise ValueError(f"无效的子分类字段配置：{item}")
        output[category] = field
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建六分类物理隔离向量库（news 仅预留目录）"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=ALL_CATEGORIES,
        default=list(ALL_CATEGORIES),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--subcategory-field",
        action="append",
        default=[],
        metavar="CATEGORY=FIELD",
        help="可选：例如 product=major_category",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="允许覆盖审阅版目录中已存在的分类数据库",
    )
    args = parser.parse_args()
    fields = _parse_subcategory_fields(args.subcategory_field)
    for category in args.categories:
        stats = build_category(
            category,
            subcategory_field=fields.get(category),
            batch_size=args.batch_size,
            rebuild=args.rebuild,
        )
        print(f"{category}：{stats}")


if __name__ == "__main__":
    main()

