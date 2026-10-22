"""F2009~F2012: 상환액 계산, 불빛 상태, 조건 탐색, 비교표, 길잡이 대화 (가짜 LLM·임베딩)"""

import json
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config import get_settings
from src.services.compare import compare_rows, text_fields
from src.services.guide import Guide
from src.services.loan_calc import BLINK, FAR, WARM, Stats, light_status, methods_for, month_steps, repayment
from src.services.product_search import search_products
from src.vectorstore.index_docs import index_glossaries
from src.vectorstore.index_meta import index_meta
from src.vectorstore.store import get_store

FIXTURES = Path(__file__).parent / "fixtures"
PID = "mortgage:0010001:1054"
NO_THRESHOLD = {**get_settings()["retrieval"], "min_relevance": -1, "meta_min_relevance": -1, "relative_margin": None}


@pytest.fixture
def chroma_dir(tmp_path):
    emb = DeterministicFakeEmbedding(size=16)
    index_meta(["mortgage", "rent_house", "credit"], sample=0, embeddings=emb, persist_dir=tmp_path, raw_dir=FIXTURES)
    index_glossaries(docs_dir=FIXTURES / "docs", embeddings=emb, persist_dir=tmp_path)
    return tmp_path


def _meta_store(chroma_dir):
    return get_store(get_settings()["vectorstore"]["meta_collection"], DeterministicFakeEmbedding(size=16), chroma_dir)


def _guide(chroma_dir, replies):
    return Guide(
        llm=FakeListChatModel(responses=replies),
        embeddings=DeterministicFakeEmbedding(size=16),
        persist_dir=chroma_dir,
        retrieval_cfg=NO_THRESHOLD,
    )


# --- 계산 ---
def test_repayment_matches_known_values():
    # 3억, 연 4%, 30년 원리금균등: 매월 약 143.22만원
    r = repayment(30000, 4.0, 360, "원리금균등")
    assert round(r.first_month, 2) == 143.22 and r.first_month == r.last_month
    r = repayment(30000, 4.0, 360, "원금균등")
    assert round(r.first_month, 2) == 183.33 and round(r.last_month, 2) == 83.61
    assert round(r.total_interest) == 18050  # P*r*(n+1)/2
    r = repayment(30000, 4.0, 360, "만기일시")
    assert r.first_month == pytest.approx(100) and r.last_month == pytest.approx(30100)
    assert repayment(12000, 0, 12, "원리금균등").first_month == 1000


def test_methods_follow_disclosed_repayment_types():
    assert methods_for("분할상환방식") == ["원리금균등", "원금균등"]
    assert methods_for("만기일시상환방식") == ["만기일시"]
    assert methods_for(None) == ["원리금균등", "원금균등", "만기일시"]


def test_light_status_and_month_steps():
    stats = Stats(cash=30000, emergency=1000, income=400, living=200, existing_debt=0, margin_goal=50)
    assert light_status(59000, stats, loan=30000, new_payment=100) == WARM  # 자기자금 29000 + 대출 30000
    assert light_status(70000, stats, loan=30000, new_payment=100) == FAR
    assert light_status(50000, stats, loan=30000, new_payment=180) == BLINK  # 남는 돈 20 < 목표 50
    steps = month_steps(stats, 143.2)
    assert steps[0] == ("월급", 400) and steps[-1][0] == "남은 돈"
    assert steps[-1][1] == pytest.approx(400 - 200 - 0 - 143.2)


# --- 조건 탐색 (F2011) ---
def test_search_keeps_only_products_matching_conditions(chroma_dir):
    cond = {"loan_type": "mortgage", "rate_max": 4.0, "rate_type": "고정", "fin_grp": "은행"}
    llm = FakeListChatModel(responses=[json.dumps(cond, ensure_ascii=False)])
    result = search_products("고정금리 4% 이하 은행 주담대", _meta_store(chroma_dir), llm)
    assert [c["product_id"] for c in result.cards] == [PID]
    assert "고정금리" in result.cards[0]["matched"] and result.message == ""
    assert [n for _, n in result.funnel][0] == 5


def test_search_reports_which_condition_emptied_the_list(chroma_dir):
    llm = FakeListChatModel(responses=['{"loan_type": "rent_house", "rate_max": 2.0}'])
    result = search_products("2% 이하 전세", _meta_store(chroma_dir), llm)
    assert result.cards == [] and "최저 금리 2% 이하" in result.message


def test_search_llm_cannot_add_products_outside_filters(chroma_dir):
    # 모델이 후보 밖 번호나 이상한 값을 줘도 결과는 필터를 통과한 후보 중에서만 나온다
    replies = ['{"loan_type": "credit", "query": "마이너스한도"}', '[{"no": 0, "reason": "마이너스"}, {"no": 7}, "x"]']
    result = search_products("마이너스통장", _meta_store(chroma_dir), FakeListChatModel(responses=replies))
    assert len(result.cards) == 1 and result.cards[0]["metadata"]["loan_type"] == "credit"


# --- 비교 (F2010) ---
def test_compare_rows_copy_disclosed_values(chroma_dir):
    guide = _guide(chroma_dir, ["x"])
    by_id = {p["product_id"]: p for p in guide.products()}
    rows = compare_rows([by_id[PID], by_id["mortgage:0013175:TEST-M2"]])
    assert rows[0]["금리 범위"] == "3.52% ~ 5.25%" and rows[0]["공시월"] == "202610"
    assert rows[1]["금리 범위"] == "공시 없음 ~ 공시 없음"
    assert text_fields("대출 한도: LTV 70%\n기타: x")["한도"] == "LTV 70%"


def test_compare_summary_cites_only_chosen_products(chroma_dir):
    ids = [PID, "mortgage:0013175:TEST-M2"]
    result = _guide(chroma_dir, ["우리아파트론은 3.52~5.25% [근거 1]."]).compare(ids)
    assert [e["id"] for e in result["evidence"][:2]] == ids
    assert all(e["collection"] == "docs" for e in result["evidence"][2:])
    assert [s["id"] for s in result["sources"]] == [PID]


# --- 길잡이 (F2009, F2012) ---
def test_guide_ask_reports_progress_and_passes_history(chroma_dir):
    guide = _guide(chroma_dir, ["중도상환수수료는 공시를 보세요 [근거 1]."])
    seen = []
    history = [{"role": "user", "content": "이 상품 금리는?"}, {"role": "assistant", "content": "3.52% [근거 1]"}]
    result = guide.ask("그럼 수수료는?", PID, history, use_wiki=False, on_status=seen.append)
    assert result["evidence"][0]["id"] == PID and [s["id"] for s in result["sources"]] == [PID]
    assert seen[0] == "근거를 찾는 중…" and len(seen) >= 3
    assert result["wiki"]["titles"] == []


def test_guide_wiki_failure_falls_back_to_indexed_docs(chroma_dir):
    class BrokenWiki:
        def lookup_title(self, title):
            raise RuntimeError("위키 요청 실패")

        search = fetch_page = lookup_title

    guide = _guide(chroma_dir, ['["주택담보대출"]', "답 [근거 1]"])
    guide._wiki = BrokenWiki()
    result = guide.ask("LTV가 뭐야?", PID, use_wiki=True)
    assert "외부 문서를 가져오지 못해" in result["wiki"]["error"]
    assert result["answer"] == "답 [근거 1]"
