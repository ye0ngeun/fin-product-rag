"""F2009~F2012 화면 확인: 키·색인 안내, 화면 4개가 오류 없이 그려지는지, 면책 문구 (Streamlit AppTest)

LLM·임베딩 호출이 필요한 버튼(찾기, 차이 요약, 질문)은 누르지 않는다. 그 동작은 test_services 에서 본다.
"""

import csv
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest

from src.config import NOTICES, ROOT_DIR, get_settings
from src.vectorstore.index_docs import index_glossaries
from src.vectorstore.index_meta import index_meta

FIXTURES = Path(__file__).parent / "fixtures"
APP = str(ROOT_DIR / "app.py")
# 첫 AppTest 실행이 무거운 import(chromadb·langchain)까지 떠안으면 새 가상환경(맥 첫 실행 등)에서 시간 제한을 넘는다.
# 화면이 쓰는 모듈을 여기서 미리 불러 두고, 제한도 넉넉히 둔다.
import src.services.guide  # noqa: E402,F401
import src.ui.chat, src.ui.explore, src.ui.plan, src.ui.sidebar  # noqa: E401,E402,F401

TIMEOUT = 60


@pytest.fixture
def env(tmp_path, monkeypatch):
    """settings 의 chroma_dir·raw_dir 를 임시 폴더로 바꾼다 (get_settings 는 캐시된 dict 를 돌려준다)."""
    paths = get_settings()["paths"]
    old = dict(paths)
    paths["chroma_dir"] = tmp_path / "chroma"
    paths["raw_dir"] = tmp_path / "raw"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield tmp_path
    paths.update(old)


def _index(tmp_path):
    emb = DeterministicFakeEmbedding(size=16)
    persist = tmp_path / "chroma"
    index_meta(["mortgage", "rent_house", "credit"], sample=0, embeddings=emb, persist_dir=persist, raw_dir=FIXTURES)
    index_glossaries(docs_dir=FIXTURES / "docs", embeddings=emb, persist_dir=persist)
    molit = tmp_path / "raw" / "molit"
    molit.mkdir(parents=True)
    with open(molit / "apt_trade.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["gu_nm", "umd_nm", "apt_nm", "area_pyeong", "deal_date", "price_manwon", "direction"])
        w.writeheader()
        w.writerow({"gu_nm": "노원구", "umd_nm": "상계동", "apt_nm": "가", "area_pyeong": 25, "deal_date": "2026-09-01", "price_manwon": 50000, "direction": "북"})
        w.writerow({"gu_nm": "강남구", "umd_nm": "대치동", "apt_nm": "나", "area_pyeong": 34, "deal_date": "2026-09-02", "price_manwon": 300000, "direction": "남"})


def _texts(at) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.error]
    return "\n".join(str(p) for p in parts)


def test_missing_key_and_index_show_guidance(env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    errors = "\n".join(e.value for e in at.error)
    assert "OPENAI_API_KEY" in errors and "python -m src.vectorstore.index_meta" in errors
    assert NOTICES["disclaimer"] in _texts(at)
    assert not at.exception


def test_every_page_renders_with_disclaimers(env):
    _index(env)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    for page in ["대출 계획", "상품 찾기·비교", "한 달 살아보기", "길잡이"]:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, (page, at.exception)
        text = _texts(at)
        assert NOTICES["disclaimer"] in text and NOTICES["fss_source"] in text


def test_plan_shows_lights_and_month_uses_same_plan(env):
    _index(env)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert any("첫 달 상환액" in m.label for m in at.metric)
    assert "불빛" in _texts(at)
    at.session_state["page"] = "한 달 살아보기"
    at.run()
    assert "남은 돈" in _texts(at)


def test_chat_history_is_kept_per_product(env):
    _index(env)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state["histories"] = {
        "mortgage:0010001:1054": [{"role": "user", "content": "우리아파트론 질문"}],
        "_general": [{"role": "user", "content": "일반 질문"}],
    }
    at.session_state["page"] = "길잡이"
    at.run()
    assert "일반 질문" in _texts(at) and "우리아파트론 질문" not in _texts(at)
    at.session_state["pending_pick"] = "mortgage:0010001:1054"
    at.run()
    assert "우리아파트론 질문" in _texts(at) and "일반 질문" not in _texts(at)
