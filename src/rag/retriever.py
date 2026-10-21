"""두 컬렉션에서 근거를 찾아 병합하고 중복을 없앤다. (F2006)

- 상품을 고른 경우: 그 상품의 loan_meta 청크를 항상 근거 1번으로 넣는다.
  상품을 고르지 않은 경우: loan_meta 에서 질문과 비슷한 상품 meta_top_k 개를 찾는다.
- loan_docs 는 top_k 개를 찾는다. 상품을 고른 경우 공통 문서(common)와 그 상품 문서만 본다.
- 검색 방식(similarity/mmr)과 개수는 settings.yaml 의 retrieval 에서 바꾼다.
- 같은 청크 ID, 또는 내용이 같은 청크(여러 용어사전에 같은 설명)는 한 번만 남긴다.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

DOC_TYPE_LABELS = {
    "meta": "금감원 상품정보",
    "wiki": "위키백과",
    "glossary_hf": "주택금융공사 단어사전",
    "glossary_epost": "우체국예금보험 금융용어사전",
    "glossary_epost_chatbot": "우체국 챗봇 금융용어 사전",
    "glossary_bok": "한국은행 경제금융용어 800선",
}


def _search(store: Chroma, query: str, k: int, cfg: dict, where: dict | None) -> list[Document]:
    if k <= 0:
        return []
    if cfg.get("search_type") == "mmr":
        return store.max_marginal_relevance_search(
            query, k=k, fetch_k=max(cfg.get("mmr_fetch_k", 20), k), lambda_mult=cfg.get("mmr_lambda", 0.5), filter=where
        )
    return store.similarity_search(query, k=k, filter=where)


def _evidence(doc: Document, chunk_id: str, collection: str) -> dict:
    meta = doc.metadata or {}
    doc_type = "meta" if collection == "meta" else meta.get("doc_type", "")
    return {
        "id": chunk_id,
        "collection": collection,
        "doc_type": doc_type,
        "label": DOC_TYPE_LABELS.get(doc_type, doc_type),
        "title": meta.get("title", ""),
        "source": meta.get("source", ""),
        "source_url": meta.get("source_url", ""),
        "source_file": meta.get("source_file", ""),
        "page": meta.get("page"),
        "collected_at": meta.get("collected_at", ""),
        "text": doc.page_content,
    }


def retrieve(
    question: str,
    meta_store: Chroma,
    doc_store: Chroma,
    cfg: dict,
    product_id: str | None = None,
) -> list[dict]:
    """번호(n)가 붙은 근거 목록. 메타 근거가 앞, 외부 문서가 뒤."""
    found: list[dict] = []
    if product_id:
        got = meta_store.get(ids=[product_id])
        if got["ids"]:
            doc = Document(page_content=got["documents"][0], metadata=got["metadatas"][0])
            found.append(_evidence(doc, product_id, "meta"))
        where = {"product_id": {"$in": ["common", product_id]}}
    else:
        for doc in _search(meta_store, question, cfg.get("meta_top_k", 3), cfg, None):
            found.append(_evidence(doc, doc.id, "meta"))
        where = None

    for doc in _search(doc_store, question, cfg["top_k"], cfg, where):
        found.append(_evidence(doc, doc.id, "docs"))

    seen_ids, seen_text, merged = set(), set(), []
    for ev in found:
        text_key = " ".join(ev["text"].split())
        if ev["id"] in seen_ids or text_key in seen_text:
            continue
        seen_ids.add(ev["id"])
        seen_text.add(text_key)
        merged.append(ev)
    for n, ev in enumerate(merged, start=1):
        ev["n"] = n
    return merged


def format_evidence(evidence: list[dict]) -> str:
    """프롬프트3에 넣을 근거 블록"""
    return "\n\n".join(f"[근거 {ev['n']}] ({ev['label']} | {ev['title']})\n{ev['text']}" for ev in evidence)
