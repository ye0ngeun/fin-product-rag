"""질문과 대출 상품으로 위키백과 문서를 골라 수집한다. (F2003, F2004, NF2002)

흐름: 프롬프트1 검색어 추출 -> 검색어별 후보 모으기 -> 프롬프트2 후보 선별 -> 본문 수집
- 후보: 검색어와 제목이 같은 문서(넘겨주기 포함)를 먼저 찾고, 전문 검색 결과 중 제목·요약에
  검색어 글자가 들어간 것만 더한다. (검색 잡음 제거)
- 제목이 검색어와 같은 문서는 그 개념 자체의 문서라 선별 결과와 상관없이 남긴다.
  상품의 대출 유형(예: 주택담보대출)은 검색어에 항상 넣으므로 그 문서도 항상 남는다.
- 이미 수집한 검색어는 캐시(data/processed/wiki_cache.json)를 보고 다시 검색·수집하지 않는다.
  선별 규칙이나 프롬프트를 바꾸면 CACHE_VERSION 을 올려 검색어 캐시를 새로 쌓는다.
- 이미 받은 문서(pageid)는 다른 검색어에서 다시 나와도 본문을 다시 받지 않는다.

위키백과에는 상품별 문서가 없어서 위키 문서는 모든 상품이 함께 쓰는 공통 문서(product_id=common)로 둔다.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from src.data_prep.wiki_client import LICENSE, WikiClient
from src.rag.llm import parse_json_list
from src.rag.prompts import CANDIDATE_SELECTION_PROMPT, QUERY_EXTRACTION_PROMPT

CACHE_VERSION = 2


class WikiCache:
    """검색어 -> 선별된 문서, pageid -> 수집 여부를 파일에 저장한다."""

    def __init__(self, path: Path):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        # 버전이 다르면 검색어 캐시만 버린다. 이미 받은 문서 목록(pages)은 그대로 둔다.
        self.stale = bool(data) and data.get("version") != CACHE_VERSION
        self.queries: dict[str, dict] = {} if self.stale else data.get("queries", {})
        self.pages: dict[str, dict] = data.get("pages", {})

    @staticmethod
    def key(query: str) -> str:
        return " ".join(query.split()).lower()

    def has_query(self, query: str) -> bool:
        return self.key(query) in self.queries

    def add_query(self, query: str, titles: list[str]) -> None:
        self.queries[self.key(query)] = {"query": query, "titles": titles, "at": datetime.now(timezone.utc).isoformat()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"version": CACHE_VERSION, "queries": self.queries, "pages": self.pages}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )


def extract_queries(llm: BaseChatModel, question: str, product_text: str, own_topic: str, max_queries: int) -> list[str]:
    """프롬프트1. 상품 자신의 대출 유형을 맨 앞에 둔다."""
    reply = llm.invoke(
        QUERY_EXTRACTION_PROMPT.format_messages(question=question, product=product_text, max_queries=max_queries)
    )
    queries = [own_topic] + [str(q).strip() for q in parse_json_list(reply.content) if str(q).strip()]
    seen, result = set(), []
    for q in queries:
        if WikiCache.key(q) not in seen:
            seen.add(WikiCache.key(q))
            result.append(q)
    return result[: max(max_queries, 1)]


def _bigrams(text: str) -> set[str]:
    t = "".join(text.split()).lower()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def gather_candidates(wiki: WikiClient, query: str, limit: int) -> list[dict]:
    """제목이 같은 문서를 먼저 두고, 검색 결과 중 검색어 글자(2글자 단위)가 제목·요약에 있는 것만 더한다."""
    exact = wiki.lookup_title(query)
    candidates = [exact] if exact else []
    grams = _bigrams(query)
    for c in wiki.search(query, limit):
        if exact and c["pageid"] == exact["pageid"]:
            continue
        if grams and not grams & _bigrams(c["title"] + c["snippet"]):
            continue
        candidates.append({**c, "exact": False})
    return candidates


def select_candidates(
    llm: BaseChatModel,
    question: str,
    product_text: str,
    query: str,
    candidates: list[dict],
    max_docs: int,
    own_topic: str,
) -> list[dict]:
    """프롬프트2. 제목이 검색어·대출 유형과 같은 문서는 항상 남기고, 나머지는 모델이 고른다."""
    if not candidates:
        return []
    keys = {own_topic.replace(" ", ""), query.replace(" ", "")}
    keep = [i for i, c in enumerate(candidates) if c.get("exact") or c["title"].replace(" ", "") in keys]
    listing = "\n".join(
        f"{i}. {c['title']}{' [검색어와 제목 일치]' if i in keep else ''} - {c['snippet']}" for i, c in enumerate(candidates)
    )
    reply = llm.invoke(
        CANDIDATE_SELECTION_PROMPT.format_messages(
            question=question, product=product_text, query=query, candidates=listing, max_docs=max_docs
        )
    )
    picked = [i for i in parse_json_list(reply.content) if isinstance(i, int) and 0 <= i < len(candidates)]
    order = keep + [i for i in picked if i not in keep]
    return [candidates[i] for i in order[: max(max_docs, len(keep))]]


def collect_wiki(
    question: str,
    product_text: str,
    own_topic: str,
    llm: BaseChatModel,
    wiki: WikiClient,
    cache: WikiCache,
    cfg: dict,
) -> tuple[list[dict], dict]:
    """새로 받은 위키 문서 목록과 확인용 로그(검색어, 캐시 여부, 선별 결과)를 돌려준다.

    캐시는 색인까지 끝난 뒤 호출하는 쪽에서 cache.save() 로 저장한다.
    """
    queries = extract_queries(llm, question, product_text, own_topic, cfg["max_queries"])
    log = {"queries": queries, "steps": []}
    docs = []
    for query in queries:
        if cache.has_query(query):
            log["steps"].append({"query": query, "cached": True, "titles": cache.queries[cache.key(query)]["titles"]})
            continue
        candidates = gather_candidates(wiki, query, cfg["search_limit"])
        selected = select_candidates(llm, question, product_text, query, candidates, cfg["max_docs_per_query"], own_topic)
        titles = []
        for c in selected:
            titles.append(c["title"])
            if str(c["pageid"]) in cache.pages:
                continue
            page = wiki.fetch_page(c["pageid"])
            if not page or len(page["text"]) < cfg.get("min_chars", 200):
                continue
            cache.pages[str(page["pageid"])] = {"title": page["title"], "url": page["url"]}
            docs.append({**page, "query": query})
        cache.add_query(query, titles)
        log["steps"].append(
            {
                "query": query,
                "cached": False,
                "candidates": [c["title"] + (" (제목 일치)" if c.get("exact") else "") for c in candidates],
                "selected": titles,
            }
        )
    return docs, log


def wiki_to_records(pages: list[dict]) -> list[dict]:
    """색인용 레코드: doc_id, text, metadata (청킹 전)"""
    return [
        {
            "doc_id": f"wiki-{p['pageid']}",
            "text": f"{p['title']}\n{p['text']}",
            "metadata": {
                "product_id": "common",
                "doc_type": "wiki",
                "title": p["title"],
                "source": "한국어 위키백과",
                "source_url": p["url"],
                "license": LICENSE,
                "query": p.get("query", ""),
            },
        }
        for p in pages
    ]
