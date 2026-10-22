"""두 컬렉션에서 근거를 찾아 병합하고 중복을 없앤다. (F2006)

- 상품을 고른 경우: 그 상품의 loan_meta 청크를 항상 근거 1번으로 넣는다.
  상품을 고르지 않은 경우: loan_meta 에서 질문과 비슷한 상품 meta_top_k 개를 찾는다.
- loan_docs 는 top_k 개를 찾는다. 상품을 고른 경우 공통 문서(common)와 그 상품 문서만 본다.
- 관련도(코사인 유사도)가 min_relevance 미만이거나 1등보다 relative_margin 이상 낮은 근거는 버린다.
  상품을 고르지 않았을 때 loan_meta 는 meta_min_relevance 를 넘는 상품만 넣는다.
- 질문의 용어(LTV, 담보인정비율 등)와 제목이 같은 용어사전 항목은 유사도와 상관없이 keyword_top_k 개까지 먼저 넣는다.
- 검색 방식(similarity/mmr)과 개수는 settings.yaml 의 retrieval 에서 바꾼다.
- 같은 청크 ID, 또는 내용이 같은 청크(여러 용어사전에 같은 설명)는 한 번만 남긴다.
"""

import re

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


def _scored(store: Chroma, query: str, k: int, cfg: dict, where: dict | None, min_score: float) -> list[tuple]:
    """(문서, 관련도) 목록. 관련도(코사인 유사도) min_score 미만, 그리고 1등보다 relative_margin 이상 낮은 것은 뺀다."""
    if k <= 0:
        return []
    fetch_k = max(cfg.get("mmr_fetch_k", 20), k) if cfg.get("search_type") == "mmr" else k * 2
    # 컬렉션은 cosine 거리(hnsw:space=cosine)라 관련도 = 1 - 거리
    scored = [(d, 1 - dist) for d, dist in store.similarity_search_with_score(query, k=fetch_k, filter=where)]
    if not scored:
        return []
    best = scored[0][1]
    margin = cfg.get("relative_margin")
    scored = [(d, sc) for d, sc in scored if sc >= min_score and (margin is None or sc >= best - margin)]
    if cfg.get("search_type") == "mmr" and scored:
        keep = {d.id: sc for d, sc in scored}
        picked = store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=cfg.get("mmr_lambda", 0.5), filter=where
        )
        return [(d, keep[d.id]) for d in picked if d.id in keep]
    return scored[:k]


# 질문 끝에 붙는 조사·어미. 용어 정확 일치 검색 전에 떼어 낸다.
_SUFFIXES = ("이란", "란", "이야", "야", "가", "이", "은", "는", "을", "를", "의", "에", "로", "으로", "와", "과")
_STOP = {"이란", "이야", "이랑", "뭐야", "무엇", "뭔가요", "뭐예요", "알려줘", "설명해줘", "어떻게", "얼마", "얼마야", "무슨", "뜻"}


def question_terms(question: str) -> list[str]:
    """질문에서 용어 후보: 영문 약어(LTV, DSR)와 조사를 뗀 2글자 이상 한글 낱말"""
    terms = []
    for word in re.findall(r"[A-Za-z]{2,}|[가-힣A-Za-z0-9]+", question):
        if re.fullmatch(r"[A-Za-z]{2,}", word):
            terms.append(word.upper())
            continue
        for suf in _SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 2:
                word = word[: -len(suf)]
                break
        if len(word) >= 2 and word not in _STOP:
            terms.append(word)
    return list(dict.fromkeys(terms))


def _keyword_hits(store: Chroma, question: str, where: dict | None, limit: int) -> list[Document]:
    """용어사전에서 제목이 질문 용어와 같거나, 본문 첫 줄에 '(약어)'가 있는 항목"""
    hits: list[Document] = []
    for term in question_terms(question):
        conds = [{"title": term}, {"doc_type": {"$ne": "wiki"}}]
        if where:
            conds.append(where)
        got = store.get(where={"$and": conds}, limit=limit)
        if not got["ids"] and re.fullmatch(r"[A-Z]{2,}", term):
            got = store.get(where=where, where_document={"$contains": f"({term})"}, limit=limit)
        for i, doc_id in enumerate(got["ids"]):
            hits.append(Document(id=doc_id, page_content=got["documents"][i], metadata=got["metadatas"][i]))
        if len(hits) >= limit:
            break
    return hits[:limit]


def _evidence(doc: Document, chunk_id: str, collection: str, score: float | None = None, how: str = "") -> dict:
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
        "score": round(score, 3) if score is not None else None,
        "how": how,
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
        # 상품을 고르지 않은 용어 질문에서 상관없는 상품이 끼지 않도록 메타는 더 높은 관련도를 요구한다.
        for doc, sc in _scored(meta_store, question, cfg.get("meta_top_k", 3), cfg, None, cfg.get("meta_min_relevance", 0)):
            found.append(_evidence(doc, doc.id, "meta", sc, "유사도"))
        where = None

    for doc in _keyword_hits(doc_store, question, where, cfg.get("keyword_top_k", 2)):
        found.append(_evidence(doc, doc.id, "docs", None, "용어 일치"))
    for doc, sc in _scored(doc_store, question, cfg["top_k"], cfg, where, cfg.get("min_relevance", 0)):
        found.append(_evidence(doc, doc.id, "docs", sc, "유사도"))

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
