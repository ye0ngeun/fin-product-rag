"""화면이 쓰는 RAG 기능 묶음: 길잡이 대화(F2009), 상품 비교 요약(F2010), 조건 탐색(F2011).

화면(app.py, src/ui)은 이 클래스만 부르고, 검색·생성·수집 모듈을 직접 알지 않는다. (NF2005)
- 길잡이: 상품을 고른 질문이면 먼저 위키 문서를 모은 뒤(캐시에 있으면 건너뜀) 답변 그래프를 돌린다.
  진행 상황은 on_status(문구) 로 알린다. 위키 호출이 실패해도 이미 색인된 문서로 답한다.
- 비교 요약: 같은 답변 그래프에 '고른 상품들의 메타 + 비교 용어의 용어사전' 근거를 넣어 차이만 요약한다.
"""

from collections.abc import Callable

import requests
from langchain_core.documents import Document

from src.config import ROOT_DIR, get_settings
from src.data_prep.doc_collector import WikiCache, collect_wiki, wiki_to_records
from src.data_prep.fss_loader import LOAN_TYPE_NAMES
from src.rag.graph import build_graph
from src.rag.retriever import keyword_hits, make_evidence, retrieve
from src.services.product_search import SearchResult, search_products
from src.vectorstore.index_docs import index_records
from src.vectorstore.store import get_embeddings, get_store

COMPARE_QUESTION = (
    "고른 상품들의 금리 범위, 금리 유형, 상환 방식, 대출 한도, 중도상환수수료, 연체 이자율의 차이를 상품별로 정리해줘. "
    "어느 상품이 유리한지는 말하지 마."
)
# 비교표에 나오는 용어. 용어사전 근거로 같이 넣는다.
COMPARE_TERMS = "고정금리 변동금리 중도상환수수료 분할상환 만기일시상환"

SEARCH_TYPES = ("similarity", "mmr")
STEP_LABELS = {"retrieve": "근거를 찾았어요. 답변 작성 중…", "generate": "답변을 다 썼어요."}


def _noop(_msg: str) -> None:
    pass


class Guide:
    def __init__(self, llm=None, embeddings=None, persist_dir=None, retrieval_cfg=None, wiki=None):
        settings = get_settings()
        self.settings = settings
        self.cfg = retrieval_cfg or settings["retrieval"]
        self.embeddings = embeddings or get_embeddings()
        self.persist_dir = persist_dir
        self.meta_store = get_store(settings["vectorstore"]["meta_collection"], self.embeddings, persist_dir)
        self.doc_store = get_store(settings["vectorstore"]["doc_collection"], self.embeddings, persist_dir)
        if llm is None:
            from src.rag.llm import get_chat_model

            llm = get_chat_model()
        self.llm = llm
        self._wiki = wiki
        self.graph = build_graph(self._retriever(self.cfg), llm)
        # 길잡이에서 검색 방식을 바꿔 결과를 비교할 수 있게 방식별 그래프를 둔다 (F2014)
        self.graphs = {
            kind: build_graph(self._retriever({**self.cfg, "search_type": kind}), llm) for kind in SEARCH_TYPES
        }

    # --- 공통 ---
    def _retriever(self, cfg: dict):
        def retrieve_fn(question, product_id):
            return retrieve(question, self.meta_store, self.doc_store, cfg, product_id=product_id)

        return retrieve_fn

    def products(self) -> list[dict]:
        """loan_meta 의 모든 상품 메타데이터 (상품 고르기·비교표용)"""
        got = self.meta_store.get(include=["metadatas", "documents"])
        return [{**m, "text": t} for m, t in zip(got["metadatas"], got["documents"])]

    def _run(self, state: dict, on_status: Callable[[str], None], graph=None) -> dict:
        """그래프를 노드 단위로 돌리며 진행 상황을 알린다."""
        on_status("근거를 찾는 중…")
        result = dict(state)
        for update in (graph or self.graph).stream(state, stream_mode="updates"):
            for node, values in update.items():
                result.update(values or {})
                on_status(STEP_LABELS.get(node, node))
        return result

    # --- 길잡이 (F2009) ---
    def collect_wiki_for(self, product_id: str, question: str, on_status: Callable[[str], None] = _noop) -> dict:
        """선택 상품과 질문으로 위키 문서를 모아 색인한다. 결과: {titles, new, error}"""
        got = self.meta_store.get(ids=[product_id])
        if not got["ids"]:
            return {"titles": [], "new": [], "error": ""}
        meta, text = got["metadatas"][0], got["documents"][0]
        try:
            if self._wiki is None:
                from src.data_prep.wiki_client import WikiClient

                self._wiki = WikiClient()
            cache = WikiCache(ROOT_DIR / self.settings["wiki"]["cache_file"])
            on_status("위키백과 검색어를 정하는 중…")
            pages, log = collect_wiki(
                question, text, LOAN_TYPE_NAMES[meta["loan_type"]], self.llm, self._wiki, cache, self.settings["wiki"]
            )
            if pages:
                on_status(f"위키백과 문서 {len(pages)}건 색인 중…")
                index_records(
                    wiki_to_records(pages), self.settings, False, embeddings=self.embeddings, persist_dir=self.persist_dir
                )
            cache.save()
        except (RuntimeError, requests.RequestException):  # 위키 호출 실패는 화면에 오류 추적 없이 안내만 한다 (NF2004)
            return {"titles": [], "new": [], "error": "외부 문서를 가져오지 못해 이미 색인된 문서로만 답합니다."}
        titles = list(dict.fromkeys(t for step in log["steps"] for t in step.get("titles", step.get("selected", []))))
        return {"titles": titles, "new": [p["title"] for p in pages], "error": ""}

    def ask(
        self,
        question: str,
        product_id: str | None = None,
        history: list[dict] | None = None,
        use_wiki: bool = True,
        on_status: Callable[[str], None] = _noop,
        search_type: str | None = None,
    ) -> dict:
        """search_type: similarity / mmr. 없으면 settings.yaml 의 retrieval.search_type"""
        wiki = {"titles": [], "new": [], "error": ""}
        if product_id and use_wiki:
            wiki = self.collect_wiki_for(product_id, question, on_status)
        state = {"question": question, "product_id": product_id, "history": history or []}
        result = self._run(state, on_status, self.graphs.get(search_type))
        return {**result, "wiki": wiki}

    # --- 비교 요약 (F2010) ---
    def _compare_evidence(self, product_ids: list[str]) -> list[dict]:
        found = []
        got = self.meta_store.get(ids=product_ids)
        by_id = dict(zip(got["ids"], zip(got["documents"], got["metadatas"])))
        for pid in product_ids:
            if pid in by_id:
                text, meta = by_id[pid]
                found.append(make_evidence(Document(page_content=text, metadata=meta), pid, "meta"))
        for doc in keyword_hits(self.doc_store, COMPARE_TERMS, {"product_id": "common"}, limit=4):
            found.append(make_evidence(doc, doc.id, "docs", None, "용어 일치"))
        for n, ev in enumerate(found, start=1):
            ev["n"] = n
        return found

    def compare(self, product_ids: list[str], on_status: Callable[[str], None] = _noop) -> dict:
        graph = build_graph(lambda q, pid: self._compare_evidence(product_ids), self.llm)
        on_status("고른 상품의 공시 정보를 모으는 중…")
        result = graph.invoke({"question": COMPARE_QUESTION, "product_id": None})
        on_status("차이 요약을 다 썼어요.")
        return result

    # --- 조건 탐색 (F2011) ---
    def search(self, condition: str, on_status: Callable[[str], None] = _noop) -> SearchResult:
        on_status("조건을 해석하는 중…")
        return search_products(condition, self.meta_store, self.llm)
