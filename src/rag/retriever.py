"""두 컬렉션에서 근거를 찾아 병합하고 중복을 없앤다. (F2006)

- 상품을 고른 경우: 그 상품의 loan_meta 청크를 항상 근거 1번으로 넣는다.
  상품을 고르지 않은 경우: loan_meta 에서 질문과 비슷한 상품 meta_top_k 개를 찾는다.
- loan_docs 는 top_k 개를 찾는다. 상품을 고른 경우 공통 문서(common)와 그 상품 문서만 본다.
- 관련도(코사인 유사도)가 min_relevance 미만이거나 1등보다 relative_margin 이상 낮은 근거는 버린다.
  상품을 고르지 않았을 때 loan_meta 는 meta_min_relevance 를 넘는 상품만 넣는다.
- 질문의 용어(LTV, 담보인정비율 등)와 제목이 같은 용어사전 항목은 유사도와 상관없이 keyword_top_k 개까지 먼저 넣는다.
  약어는 항목 첫 줄(용어 줄)에 '(LTV)'가 있는 용어 1개 = 문서 1개 사전(SHORT_GLOSSARIES)만 본다.
  800선 PDF는 한 쪽에 용어가 여러 개라 본문 어딘가에 약어가 있다고 그 쪽이 답이 되지는 않는다.
- 검색 방식(similarity/mmr)과 개수는 settings.yaml 의 retrieval 에서 바꾼다.
- 같은 청크 ID, 또는 내용이 같은 청크(여러 용어사전에 같은 설명)는 한 번만 남긴다.
"""

import re

import numpy as np
from langchain_chroma import Chroma
from langchain_chroma.vectorstores import maximal_marginal_relevance
from langchain_core.documents import Document

DOC_TYPE_LABELS = {
    "meta": "금감원 상품정보",
    "wiki": "위키백과",
    "glossary_hf": "주택금융공사 단어사전",
    "glossary_epost": "우체국예금보험 금융용어사전",
    "glossary_epost_chatbot": "우체국 챗봇 금융용어 사전",
    "glossary_bok": "한국은행 경제금융용어 800선",
}


def _scored(store: Chroma, query_vec: list[float], k: int, cfg: dict, where: dict | None, min_score: float) -> list[tuple]:
    """(문서, 관련도) 목록. 관련도(코사인 유사도) min_score 미만, 그리고 1등보다 relative_margin 이상 낮은 것은 뺀다.

    질문 임베딩(query_vec)은 retrieve 에서 한 번만 만들어 두 컬렉션·MMR 이 같이 쓴다. (임베딩 API 호출 수 줄이기)
    MMR 은 문턱을 통과한 후보 안에서만 다양성을 따져 k 개를 고른다. (문턱 밖 문서를 골랐다가 버려 근거가 줄지 않게)
    """
    if k <= 0:
        return []
    mmr = cfg.get("search_type") == "mmr"
    fetch_k = max(cfg.get("mmr_fetch_k", 20), k) if mmr else k * 2
    got = store._collection.query(
        query_embeddings=[query_vec],
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "distances"] + (["embeddings"] if mmr else []),
    )
    rows = [
        # 컬렉션은 cosine 거리(hnsw:space=cosine)라 관련도 = 1 - 거리
        (Document(id=i, page_content=t, metadata=m or {}), 1 - dist, n)
        for n, (i, t, m, dist) in enumerate(
            zip(got["ids"][0], got["documents"][0], got["metadatas"][0], got["distances"][0], strict=True)
        )
    ]
    if not rows:
        return []
    best = rows[0][1]
    margin = cfg.get("relative_margin")
    rows = [r for r in rows if r[1] >= min_score and (margin is None or r[1] >= best - margin)]
    if mmr and rows:
        vectors = [got["embeddings"][0][n] for _, _, n in rows]
        picked = maximal_marginal_relevance(
            np.array(query_vec, dtype=np.float32), vectors, lambda_mult=cfg.get("mmr_lambda", 0.5), k=k
        )
        return [(rows[i][0], rows[i][1]) for i in picked]
    return [(d, sc) for d, sc, _ in rows[:k]]


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


# 용어 1개 = 문서 1개인 사전. 약어 일치는 이 사전들의 첫 줄에서만 찾는다.
SHORT_GLOSSARIES = ["glossary_hf", "glossary_epost", "glossary_epost_chatbot"]


def _first_line_docs(store: Chroma, needle: str, where: dict | None) -> list[Document]:
    """단문 사전에서 첫 줄(용어 줄)에 needle 이 있는 항목"""
    short = {"doc_type": {"$in": SHORT_GLOSSARIES}}
    # 본문에만 나오는 항목도 걸리므로 넉넉히 받아 첫 줄로 거른다
    got = store.get(where={"$and": [short, where]} if where else short, where_document={"$contains": needle}, limit=100)
    return [
        Document(id=i, page_content=t, metadata=m)
        for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])
        if needle in t.split("\n", 1)[0]
    ]


def _bigram_dice(a: str, b: str) -> float:
    ga = {a[i : i + 2] for i in range(len(a) - 1)}
    gb = {b[i : i + 2] for i in range(len(b) - 1)}
    return 2 * len(ga & gb) / (len(ga) + len(gb)) if ga and gb else 0.0


# 제목이 질문 용어와 비슷하다고 볼 2글자 단위 겹침 정도 (원리금균등상환 ~ 원리금균등분할상환 = 0.71)
TITLE_SIMILARITY = 0.6


def keyword_hits(store: Chroma, question: str, where: dict | None, limit: int) -> list[Document]:
    """용어사전에서 질문 용어와 맞는 항목.

    1) 제목이 용어와 같은 항목
    2) 없으면: 약어(LTV)는 첫 줄에 '(LTV)'가 있는 항목,
       4글자 이상 한글 용어는 제목이 비슷한 항목(예: 원리금균등상환 -> 원리금균등분할상환). 앞 3글자로 후보를 찾고
       2글자 단위 겹침이 TITLE_SIMILARITY 이상인 것만, 비슷한 순서로. 짧은 낱말(금리)은 너무 많이 걸려 2) 를 하지 않는다.
    """
    hits: list[Document] = []
    for term in question_terms(question):
        conds = [{"title": term}, {"doc_type": {"$ne": "wiki"}}]
        if where:
            conds.append(where)
        got = store.get(where={"$and": conds}, limit=limit)
        docs = [Document(id=i, page_content=t, metadata=m) for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])]
        if not docs and re.fullmatch(r"[A-Z]{2,}", term):
            docs = sorted(_first_line_docs(store, f"({term})", where), key=lambda d: len(d.metadata.get("title", "")))
        elif not docs and len(term) >= 4:
            scored = [
                (_bigram_dice(term, d.metadata.get("title", "").replace(" ", "")), d)
                for d in _first_line_docs(store, term[:3], where)
            ]
            docs = [d for sc, d in sorted(scored, key=lambda x: -x[0]) if sc >= TITLE_SIMILARITY]
        hits.extend(docs[:limit])
        if len(hits) >= limit:
            break
    return hits[:limit]


def make_evidence(doc: Document, chunk_id: str, collection: str, score: float | None = None, how: str = "") -> dict:
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
    query_vec = doc_store.embeddings.embed_query(question)
    if product_id:
        got = meta_store.get(ids=[product_id])
        if got["ids"]:
            doc = Document(page_content=got["documents"][0], metadata=got["metadatas"][0])
            found.append(make_evidence(doc, product_id, "meta"))
        where = {"product_id": {"$in": ["common", product_id]}}
    else:
        # 상품을 고르지 않은 용어 질문에서 상관없는 상품이 끼지 않도록 메타는 더 높은 관련도를 요구한다.
        for doc, sc in _scored(meta_store, query_vec, cfg.get("meta_top_k", 3), cfg, None, cfg.get("meta_min_relevance", 0)):
            found.append(make_evidence(doc, doc.id, "meta", sc, "유사도"))
        where = None

    for doc in keyword_hits(doc_store, question, where, cfg.get("keyword_top_k", 2)):
        found.append(make_evidence(doc, doc.id, "docs", None, "용어 일치"))
    for doc, sc in _scored(doc_store, query_vec, cfg["top_k"], cfg, where, cfg.get("min_relevance", 0)):
        found.append(make_evidence(doc, doc.id, "docs", sc, "유사도"))

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
