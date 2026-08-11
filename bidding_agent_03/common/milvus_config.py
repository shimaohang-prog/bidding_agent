# -*- coding: utf-8 -*-
"""六分类、物理隔离的 Milvus Lite 配置。"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

from dotenv import load_dotenv


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

VECTOR_DB_ROOT: Final[Path] = Path(
    os.getenv(
        "VECTOR_DB_ROOT",
        str(PROJECT_ROOT / "milvus_db"),
    )
).expanduser().resolve()

EMBEDDING_MODEL: Final[str] = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-base-zh-v1.5",
).strip()
COLLECTION_NAME: Final[str] = "records"


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class CategorySpec:
    key: str
    label: str
    source_kind: str
    vector_enabled: bool
    threshold: float


CATEGORY_SPECS: Final[dict[str, CategorySpec]] = {
    "enterprise": CategorySpec(
        "enterprise",
        "企业信息",
        "structured_csv",
        True,
        _float_env("ENTERPRISE_THRESHOLD", 0.68),
    ),
    "tender": CategorySpec(
        "tender",
        "招投标项目",
        "structured_csv",
        True,
        _float_env("TENDER_THRESHOLD", 0.65),
    ),
    "product": CategorySpec(
        "product",
        "产品信息",
        "structured_csv",
        True,
        _float_env("PRODUCT_THRESHOLD", 0.66),
    ),
    "laws": CategorySpec(
        "laws",
        "法律法规",
        "text_documents",
        True,
        _float_env("LAWS_THRESHOLD", 0.62),
    ),
    "policy": CategorySpec(
        "policy",
        "政策文件",
        "text_documents",
        True,
        _float_env("POLICY_THRESHOLD", 0.62),
    ),
    "news": CategorySpec(
        "news",
        "行业资讯",
        "web_only",
        False,
        _float_env("NEWS_THRESHOLD", 0.60),
    ),
}

ALL_CATEGORIES: Final[tuple[str, ...]] = tuple(CATEGORY_SPECS)
VECTOR_CATEGORIES: Final[tuple[str, ...]] = tuple(
    key for key, spec in CATEGORY_SPECS.items() if spec.vector_enabled
)
BUSINESS_CATEGORIES: Final[tuple[str, ...]] = (
    "enterprise",
    "tender",
    "product",
)
KNOWLEDGE_CATEGORIES: Final[tuple[str, ...]] = ("laws", "policy")

CATEGORY_THRESHOLDS: Final[dict[str, float]] = {
    key: spec.threshold for key, spec in CATEGORY_SPECS.items()
}
TOP_K_PER_CATEGORY: Final[int] = max(
    1, min(_int_env("TOP_K_PER_CATEGORY", 8), 50)
)
HYBRID_RECALL_MULTIPLIER: Final[int] = max(
    1, min(_int_env("HYBRID_RECALL_MULTIPLIER", 4), 10)
)
HYBRID_RRF_K: Final[int] = max(
    1, min(_int_env("HYBRID_RRF_K", 60), 200)
)
RERANK_CANDIDATE_LIMIT: Final[int] = max(
    1, min(_int_env("RERANK_CANDIDATE_LIMIT", 30), 100)
)
FINAL_CANDIDATE_LIMIT: Final[int] = max(
    1, min(_int_env("FINAL_CANDIDATE_LIMIT", 20), 50)
)
RERANK_WEB_THRESHOLD: Final[float] = _float_env(
    "RERANK_WEB_THRESHOLD", 0.55
)


def get_category_spec(category: str) -> CategorySpec:
    try:
        return CATEGORY_SPECS[category]
    except KeyError as exc:
        raise ValueError(f"不支持的分类：{category}") from exc


def subcategory_key(value: str) -> str:
    clean = " ".join(str(value or "").split())
    if not clean:
        raise ValueError("子分类名称不能为空")
    return hashlib.sha1(clean.encode("utf-8")).hexdigest()[:16]


def category_db_dir(category: str) -> Path:
    get_category_spec(category)
    return VECTOR_DB_ROOT / category


def category_db_path(
    category: str,
    subcategory: str | None = None,
) -> Path:
    spec = get_category_spec(category)
    if not spec.vector_enabled:
        raise ValueError(f"{category} 当前仅联网检索，不创建向量数据库")
    if subcategory:
        return (
            category_db_dir(category)
            / "subcategories"
            / f"{subcategory_key(subcategory)}.db"
        )
    return category_db_dir(category) / "main.db"


def iter_existing_shards(
    category: str,
    subcategories: Iterable[str] | None = None,
) -> list[tuple[str | None, Path]]:
    """返回一个大分类下已有的主库和子分类库。"""
    spec = get_category_spec(category)
    if not spec.vector_enabled:
        return []

    root = category_db_dir(category)
    output: list[tuple[str | None, Path]] = []
    main_path = root / "main.db"
    if main_path.exists():
        output.append((None, main_path))

    hints = [
        " ".join(str(item).split())
        for item in (subcategories or [])
        if str(item).strip()
    ]
    sub_dir = root / "subcategories"
    if hints:
        for label in dict.fromkeys(hints):
            path = sub_dir / f"{subcategory_key(label)}.db"
            if path.exists():
                output.append((label, path))
    elif sub_dir.is_dir():
        output.extend(
            (None, path)
            for path in sorted(sub_dir.glob("*.db"))
            if path.exists()
        )
    return output


def get_milvus_client(db_path: Path, *, create_parent: bool = False):
    """为一个分类分片建立短生命周期的 Milvus Lite 连接。"""
    from pymilvus import MilvusClient

    path = Path(db_path).expanduser().resolve()
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(uri=str(path))


def close_milvus_client(client) -> None:
    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        close()

