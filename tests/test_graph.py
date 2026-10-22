"""F2006~F2008: 두 컬렉션 검색·병합, retrieve/generate 그래프, 인용된 근거만 출처로 표시 (가짜 LLM·임베딩)"""

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config import get_settings
from src.rag.ask import make_app
from src.rag.graph import NO_CITATION_NOTICE, NOT_FOUND, build_graph, cited_numbers, source_line
from src.rag.retriever import question_terms
from src.vectorstore.index_docs import index_glossaries
from src.vectorstore.index_meta import index_meta
from src.vectorstore.store import get_store

FIXTURES = Path(__file__).parent / "fixtures"
PID = "mortgage:0010001:1054"


@pytest.fixture
def chroma_dir(tmp_path):
    emb = DeterministicFakeEmbedding(size=16)
    index_meta(["mortgage", "rent_house", "credit"], sample=0, embeddings=emb, persist_dir=tmp_path, raw_dir=FIXTURES)
    index_glossaries(docs_dir=FIXTURES / "docs", embeddings=emb, persist_dir=tmp_path)
    return tmp_path


# 가짜 임베딩은 점수가 무작위라 관련도 문턱은 끈다 (문턱 동작은 test_relevance_threshold 에서 따로 본다)
NO_THRESHOLD = {**get_settings()["retrieval"], "min_relevance": -1, "meta_min_relevance": -1, "relative_margin": None}


def _app(chroma_dir, replies, cfg=None):
    return make_app(
        llm=FakeListChatModel(responses=replies),
        embeddings=DeterministicFakeEmbedding(size=16),
        persist_dir=chroma_dir,
        retrieval_cfg=cfg or NO_THRESHOLD,
    )


def test_graph_has_exactly_retrieve_and_generate(chroma_dir):
    nodes = set(_app(chroma_dir, ["x"]).get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"retrieve", "generate"}


def test_selected_product_meta_is_first_evidence_and_no_duplicates(chroma_dir):
    result = _app(chroma_dir, ["중도상환수수료는 0.71% 기준입니다 [근거 1]."]).invoke(
        {"question": "중도상환수수료?", "product_id": PID}
    )
    ev = result["evidence"]
    assert ev[0]["id"] == PID and ev[0]["label"] == "금감원 상품정보"
    assert [e["n"] for e in ev] == list(range(1, len(ev) + 1))
    assert len({e["id"] for e in ev}) == len(ev)
    # 다른 상품의 메타는 섞이지 않고, 외부 문서는 공통 문서만
    assert all(e["collection"] == "docs" for e in ev[1:])
    assert [s["id"] for s in result["sources"]] == [PID]
    assert result["notice"] == ""


def test_only_cited_evidence_becomes_sources(chroma_dir):
    result = _app(chroma_dir, ["LTV는 담보 대비 비율입니다 [근거 2]. 상품 한도는 LTV 70% [근거 1, 9]."]).invoke(
        {"question": "LTV?", "product_id": PID}
    )
    assert [s["n"] for s in result["sources"]] == [2, 1]  # 없는 번호 9 는 버림
    assert len(result["evidence"]) > 2


def test_no_citation_notice(chroma_dir):
    result = _app(chroma_dir, [NOT_FOUND]).invoke({"question": "오늘 날씨?", "product_id": PID})
    assert result["sources"] == [] and result["notice"] == NO_CITATION_NOTICE


def test_without_product_searches_meta_collection(chroma_dir):
    result = _app(chroma_dir, ["[근거 1]"]).invoke({"question": "전세자금대출", "product_id": None})
    metas = [e for e in result["evidence"] if e["collection"] == "meta"]
    assert 1 <= len(metas) <= 3


def test_empty_evidence_short_circuits():
    app = build_graph(lambda q, p: [], FakeListChatModel(responses=["쓰이면 안 됨"]))
    result = app.invoke({"question": "q"})
    assert result["answer"] == NOT_FOUND and result["notice"] == NO_CITATION_NOTICE


def test_cited_numbers_and_source_line():
    assert cited_numbers("a [근거 3][근거 1] b [근거 1, 4] [근거2]") == [3, 1, 4, 2]
    pdf = {"n": 2, "label": "한국은행 경제금융용어 800선", "title": "800선 12쪽", "source_file": "800.pdf", "page": 12}
    assert source_line(pdf).endswith("800.pdf 12쪽")
    wiki = {"n": 1, "label": "위키백과", "title": "이자율", "source_url": "https://ko.wikipedia.org/wiki/이자율"}
    assert "https://ko.wikipedia.org" in source_line(wiki)


def test_relevance_threshold_drops_low_scores(chroma_dir):
    strict = {**NO_THRESHOLD, "min_relevance": 1.01, "meta_min_relevance": 1.01, "keyword_top_k": 0}
    result = _app(chroma_dir, ["x"], strict).invoke({"question": "아무 말", "product_id": None})
    assert result["evidence"] == [] and result["answer"] == NOT_FOUND


def test_keyword_hit_finds_glossary_by_acronym_and_title(chroma_dir):
    cfg = {**NO_THRESHOLD, "min_relevance": 1.01, "meta_min_relevance": 1.01}  # 유사도 검색은 막고 용어 일치만
    ev = _app(chroma_dir, ["x"], cfg).invoke({"question": "LTV가 뭐야?", "product_id": None})["evidence"]
    assert [e["title"] for e in ev] == ["담보인정비율"] and ev[0]["how"] == "용어 일치"
    ev = _app(chroma_dir, ["x"], cfg).invoke({"question": "고정금리란?", "product_id": None})["evidence"]
    assert [e["title"] for e in ev] == ["고정금리"]


def test_acronym_hit_skips_pdf_pages_and_body_mentions(chroma_dir):
    store = get_store(get_settings()["vectorstore"]["doc_collection"], DeterministicFakeEmbedding(size=16), chroma_dir)
    store.add_texts(
        ["노동비용\n단위노동비용 ... 주택담보대출비율(LTV) 규제", "총부채상환비율\n소득 대비 상환액. 담보인정비율(LTV)과 함께 쓴다."],
        metadatas=[
            {"product_id": "common", "doc_type": "glossary_bok", "title": "800선 108쪽"},
            {"product_id": "common", "doc_type": "glossary_hf", "title": "총부채상환비율"},
        ],
        ids=["common:bok800-p108:0", "common:hf-X:0"],
    )
    cfg = {**NO_THRESHOLD, "min_relevance": 1.01, "meta_min_relevance": 1.01}
    ev = _app(chroma_dir, ["x"], cfg).invoke({"question": "LTV가 뭐야?", "product_id": None})["evidence"]
    assert [e["title"] for e in ev] == ["담보인정비율"]


def test_question_terms():
    assert question_terms("LTV가 뭐야?") == ["LTV"]
    assert question_terms("중도상환수수료는 어떻게 계산돼?") == ["중도상환수수료", "계산돼"]
    assert question_terms("dsr이란") == ["DSR"]
