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
    assert "참고(한계)" in to_markdown([{**row, "note": "근거 문서 없음"}], "m") and "참고(한계)" not in md


def test_retrieve_embeds_question_once_even_with_mmr(stores):
    """두 컬렉션 검색과 MMR 이 질문 임베딩 하나를 같이 쓴다 (GMS 호출 수·일시 오류 기회 줄이기)"""
    from src.rag.retriever import retrieve

    class Counting(DeterministicFakeEmbedding):
        calls: int = 0

        def embed_query(self, text):
            self.calls += 1
            return super().embed_query(text)

    tmp, _, _ = stores
    emb = Counting(size=16)
    vs = get_settings()["vectorstore"]
    meta, docs = get_store(vs["meta_collection"], emb, tmp), get_store(vs["doc_collection"], emb, tmp)
    retrieve("LTV가 뭐야?", meta, docs, {**NO_THRESHOLD, "search_type": "mmr"})
    assert emb.calls == 1


def test_mmr_picks_k_among_candidates_that_pass_threshold(stores):
    from src.rag.retriever import _scored

    tmp, _, doc_store = stores
    vec = doc_store.embeddings.embed_query("LTV 설명")
    cfg = {**NO_THRESHOLD, "search_type": "mmr", "mmr_fetch_k": 20, "mmr_lambda": 0.7}
    got = _scored(doc_store, vec, 3, cfg, None, -1)
    assert len(got) == 3 and len({d.id for d, _ in got}) == 3
    # 문턱을 세게 걸면 통과한 후보 안에서만 고른다
    sim = _scored(doc_store, vec, 10, {**cfg, "search_type": "similarity"}, None, -1)
    cut = sim[1][1]
    got = _scored(doc_store, vec, 5, cfg, None, cut)
    assert {d.id for d, _ in got} <= {d.id for d, sc in sim if sc >= cut}


def test_keyword_hit_finds_similar_title(stores):
    from src.rag.retriever import keyword_hits

    _, _, doc_store = stores
    doc_store.add_texts(
        ["원리금균등분할상환\n매달 원금과 이자를 합친 같은 금액을 갚는 방식", "원리금\n원금과 이자"],
        metadatas=[
            {"product_id": "common", "doc_type": "glossary_hf", "title": "원리금균등분할상환"},
            {"product_id": "common", "doc_type": "glossary_hf", "title": "원리금"},
        ],
        ids=["common:hf-A:0", "common:hf-B:0"],
    )
    hits = keyword_hits(doc_store, "원리금균등상환이 뭐야?", None, 2)
    assert [d.metadata["title"] for d in hits] == ["원리금균등분할상환"]
    assert keyword_hits(doc_store, "금리가 뭐야?", None, 2) == []  # 짧은 낱말은 비슷한 제목 찾기를 하지 않는다
