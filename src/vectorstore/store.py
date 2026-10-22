"""Chroma 컬렉션 생성과 고정 ID 색인. 도메인에 묶이지 않은 코드만 둔다. (F2002, NF2005)"""

from collections.abc import Sequence
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config import get_env, get_settings


def get_embeddings() -> Embeddings:
    """GMS(OpenAI 호환) 임베딩. 키와 주소는 .env 에서 읽는다. (NF2003)"""
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings["llm"]["embedding_model"],
        api_key=get_env("OPENAI_API_KEY"),
        base_url=get_env("OPENAI_BASE_URL", required=False),
        # 토큰 배열 대신 원문 문자열을 보낸다. 프록시(GMS)에서 토큰 배열 입력이 막힐 수 있다.
        check_embedding_ctx_length=False,
        # 일시 오류(502 등)는 OpenAI 클라이언트가 간격을 늘려 다시 시도한다
        max_retries=settings["llm"].get("max_retries", 2),
        request_timeout=settings["llm"].get("timeout_sec"),
    )


def get_store(collection: str, embeddings: Embeddings | None = None, persist_dir=None) -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(persist_dir or settings["paths"]["chroma_dir"]),
        collection_metadata={"hnsw:space": "cosine"},
    )


def count_collection(collection: str, persist_dir=None) -> int:
    """임베딩 없이 저장된 청크 수만 센다. 화면 사이드바 표시용. 컬렉션이 없으면 0."""
    path = Path(persist_dir or get_settings()["paths"]["chroma_dir"])
    if not path.exists():
        return 0
    client = chromadb.PersistentClient(path=str(path))
    try:
        return client.get_collection(collection).count()
    except Exception:
        return 0


def upsert_documents(
    store: Chroma,
    docs: Sequence[Document],
    ids: Sequence[str],
    stale_where: dict | None = None,
    batch_size: int = 100,
) -> dict:
    """고정 ID로 덮어쓴다(upsert). 다시 실행해도 같은 ID는 늘지 않는다.

    stale_where 를 주면 그 조건에 걸리는 기존 청크 중 이번 ID 목록에 없는 것을 지운다.
    (원본에서 사라진 상품 정리용. 이번에 색인하지 않는 유형은 건드리지 않도록 조건으로 범위를 좁힌다.)
    """
    if len(set(ids)) != len(ids):
        raise ValueError("색인할 ID가 중복됩니다.")
    before = store._collection.count()
    for start in range(0, len(docs), batch_size):
        store.add_documents(list(docs[start : start + batch_size]), ids=list(ids[start : start + batch_size]))

    removed = 0
    if stale_where is not None:
        existing = store.get(where=stale_where, include=[])["ids"]
        stale = sorted(set(existing) - set(ids))
        if stale:
            store.delete(ids=stale)
            removed = len(stale)
    return {"before": before, "upserted": len(ids), "removed": removed, "after": store._collection.count()}
