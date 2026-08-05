# -*- coding: utf-8 -*-
"""全项目共享的 Embedding 模型，避免重复加载和不必要的联网检查。"""

import os
from threading import Lock
from typing import Any, Optional, Sequence

import numpy as np

from common.milvus_config import EMBEDDING_MODEL


EMBEDDING_LOCAL_FILES_ONLY = os.getenv(
    "EMBEDDING_LOCAL_FILES_ONLY",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}

# 必须在导入 sentence-transformers 前设置，避免 transformers/huggingface_hub
# 启动时访问 Hugging Face 检查模型和 adapter_config.json。
if EMBEDDING_LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_model: Optional[Any] = None
_model_lock = Lock()


def get_embedding_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(
                        "未安装 sentence-transformers，"
                        "请先执行 pip install -r requirements.txt"
                    ) from exc
                mode = (
                    "本地离线模式"
                    if EMBEDDING_LOCAL_FILES_ONLY
                    else "联网模式"
                )
                print(f"正在加载 Embedding 模型：{EMBEDDING_MODEL}（{mode}）")
                try:
                    _model = SentenceTransformer(
                        EMBEDDING_MODEL,
                        local_files_only=EMBEDDING_LOCAL_FILES_ONLY,
                    )
                except Exception as exc:
                    if EMBEDDING_LOCAL_FILES_ONLY:
                        raise RuntimeError(
                            "本地 Embedding 模型加载失败。请确认 "
                            f"{EMBEDDING_MODEL} 已下载到当前用户的 "
                            "Hugging Face 缓存；或临时设置 "
                            "EMBEDDING_LOCAL_FILES_ONLY=false 后联网下载。"
                        ) from exc
                    raise
    return _model


def encode_texts(texts: Sequence[str]) -> np.ndarray:
    clean_texts = [str(item).strip() for item in texts]
    if not clean_texts or any(not item for item in clean_texts):
        raise ValueError("待编码文本不能为空")
    return get_embedding_model().encode(
        clean_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def encode_query(query: str) -> list[float]:
    query = (query or "").strip()
    if not query:
        raise ValueError("语义检索内容不能为空")
    return encode_texts([query])[0].tolist()

