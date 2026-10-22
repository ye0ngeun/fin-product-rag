"""화면 공통: 세션 상태, 상품 표시 이름, 근거 펼치기, 안내 문구. (F2009, F2012)"""

import streamlit as st

from src.config import NOTICES
from src.rag.graph import NO_CITATION_NOTICE, source_line

PAGES = ["대출 계획", "상품 찾기·비교", "한 달 살아보기", "길잡이"]
GENERAL = "_general"  # 상품을 고르지 않은 대화의 기록 키

CALC_NOTICE = (
    "계산 값은 공시 금리로 단순 계산한 참고치입니다. 실제 금리·한도는 개인 신용도와 심사에 따라 다르고, "
    "취득세·중개보수와 제도상 LTV·DSR 한도는 반영하지 않았습니다."
)


def init_state() -> None:
    defaults = {
        "page": PAGES[0],
        "selected": None,  # 고른 상품 ID
        "histories": {},  # {상품 ID 또는 GENERAL: [메시지]}
        "compare_ids": [],
        "search_result": None,
        "compare_summary": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go(page: str, product_id: str | None = None) -> None:
    """버튼 콜백: 화면 이동(필요하면 상품 선택까지)"""
    if product_id is not None:
        st.session_state.pending_pick = product_id  # 사이드바가 다음 실행 때 반영
    st.session_state.page = page


def rate_text(meta: dict) -> str:
    lo, hi = meta.get("rate_min"), meta.get("rate_max")
    return f"{lo:.2f}~{hi:.2f}%" if lo is not None and hi is not None else "금리 공시 없음"


def product_label(meta: dict) -> str:
    return f"{meta.get('title', '')} · {meta.get('loan_type_nm', '')} · {rate_text(meta)}"


def history_key(product_id: str | None) -> str:
    return product_id or GENERAL


def show_answer(result: dict) -> None:
    """답변 + 인용된 근거(펼치기) + 안내. 인용되지 않은 근거는 출처에 넣지 않는다. (F2008)"""
    st.markdown(result.get("answer", ""))
    sources = result.get("sources") or []
    if sources:
        with st.expander(f"근거 {len(sources)}개 펼치기"):
            for ev in sources:
                st.markdown(f"**{source_line(ev)}**")
                st.caption(ev["text"][:400] + ("…" if len(ev["text"]) > 400 else ""))
    if result.get("notice") or not sources:
        st.caption(result.get("notice") or NO_CITATION_NOTICE)


def disclaimers(compact: bool = False) -> None:
    st.caption(NOTICES["disclaimer"])
    if not compact:
        st.caption(NOTICES["fss_source"])


def friendly_error(what: str) -> None:
    """오류 추적·키를 화면에 내지 않는다. (NF2004)"""
    st.error(f"{what} 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요. 계속되면 .env 설정과 색인 상태를 확인하세요.")
