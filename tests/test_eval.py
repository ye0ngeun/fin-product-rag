"""F2014 유사도/MMR 비교, F2015 RAG 전후 비교 (가짜 LLM·임베딩)"""

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config import get_settings
from src.eval.rag_compare import expected_values, judge, run_item, to_markdown
from src.eval.search_compare import compare_search, diversity, doc_key, load_questions
from src.rag.ask import make_app
from src.vectorstore.index_docs import index_glossaries, index_records
from src.vectorstore.index_meta import index_meta
from src.vectorstore.store import get_store

FIXTURES = Path(__file__).parent / "fixtures"
PID = "mortgage:0010001:1054"
NO_THRESHOLD = {**get_settings()["retrieval"], "min_relevance": -1, "meta_min_relevance": -1, "relative_margin": None}


@pytest.fixture
def stores(tmp_path):
    emb = DeterministicFakeEmbedding(size=16)
    index_meta(["mortgage", "rent_house", "credit"], sample=0, embeddings=emb, persist_dir=tmp_path, raw_dir=FIXTURES)
    index_glossaries(docs_dir=FIXTURES / "docs", embeddings=emb, persist_dir=tmp_path)
    # 한 위키 문서가 여러 청크로 나뉜 상황 (MMR 비교용)
    page = {"doc_id": "wiki-1", "text": "담보 인정 비율\n" + "LTV 설명 문단. " * 300, "metadata": {"product_id": "common", "doc_type": "wiki", "title": "담보 인정 비율"}}
    index_records([page], get_settings(), False, embeddings=emb, persist_dir=tmp_path)
    vs = get_settings()["vectorstore"]
    return tmp_path, get_store(vs["meta_collection"], emb, tmp_path), get_store(vs["doc_collection"], emb, tmp_path)


def test_questions_file_is_valid():
    items = load_questions()
    assert len(items) >= 5 and len({q["id"] for q in items}) == len(items)
    assert {q["kind"] for q in items} <= {"fact", "concept", "unknown"}
    assert all(q.get("check") for q in items if q["kind"] == "fact")


def test_doc_key_and_diversity():
    evs = [
        {"id": "common:wiki-1:0", "collection": "docs", "doc_type": "wiki"},
        {"id": "common:wiki-1:2", "collection": "docs", "doc_type": "wiki"},
        {"id": PID, "collection": "meta", "doc_type": "meta"},
    ]
    assert doc_key(evs[0]) == "common:wiki-1" and doc_key(evs[2]) == PID
    assert diversity(evs) == {"근거 수": 3, "서로 다른 문서": 2, "같은 문서 중복": 1, "출처 종류": 2}


def test_compare_search_runs_both_methods(stores):
    _, meta_store, doc_store = stores
    r = compare_search("LTV가 뭐야?", meta_store, doc_store, NO_THRESHOLD, product_id=PID)
    assert set(r["runs"]) == {"similarity", "mmr"}
    for kind in ("similarity", "mmr"):
        assert r["runs"][kind][0]["id"] == PID  # 고른 상품은 두 방식 모두 근거 1번
        assert r["stats"][kind]["근거 수"] == len(r["runs"][kind])
    assert set(r["only_similarity"]).isdisjoint(r["only_mmr"])


def test_expected_values_come_from_disclosure(stores):
    _, meta_store, _ = stores
    assert expected_values({"product_id": PID, "check": {"meta": "rate_min"}}, meta_store) == ["3.52"]
    assert expected_values({"product_id": PID, "check": {"field": "중도상환수수료"}}, meta_store) == ["0.71", "0.95"]
    assert expected_values({"product_id": "없는:상품", "check": {"meta": "rate_min"}}, meta_store) == []


def test_judge():
    assert judge({"kind": "fact"}, "최저 3.52%입니다", ["3.52"]) == "맞음"
    assert judge({"kind": "fact"}, "보통 3% 내외", ["3.52"]) == "틀림"
    assert judge({"kind": "fact"}, "x", []) == "판단 불가"
    assert judge({"kind": "concept", "keywords": ["담보", "비율"]}, "담보 가치 대비 대출 비율", []) == "맞음"
    assert judge({"kind": "unknown"}, "제공된 문서에서 확인되지 않습니다.", []) == "맞음"
    assert judge({"kind": "unknown"}, "다음 달에는 0.25%p 오를 것입니다.", []) == "틀림"


def test_run_item_and_report(stores):
    tmp, meta_store, _ = stores
    llm = FakeListChatModel(responses=["보통 3% 정도입니다.", "최저 3.52%입니다 [근거 1]."])
    app = make_app(llm=llm, embeddings=DeterministicFakeEmbedding(size=16), persist_dir=tmp, retrieval_cfg=NO_THRESHOLD)
    item = {"id": "q2", "question": "우리아파트론 최저 금리?", "product_id": PID, "kind": "fact", "check": {"meta": "rate_min"}}
    row = run_item(item, llm, app, meta_store)
    assert row["baseline_judge"] == "틀림" and row["rag_judge"] == "맞음"
    assert len(row["sources"]) == 1 and "금감원 상품정보" in row["sources"][0]
    md = to_markdown([row], "test-model")
    assert "| 자동 점검 맞음 | 0/1 | 1/1 |" in md and "| 근거 제시(출처 있는 답변) | 0/1 | 1/1 |" in md
    assert "> 최저 3.52%입니다 [근거 1]." in md
