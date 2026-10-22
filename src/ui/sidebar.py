"""사이드바: 시작 스탯, 상품 고르기, 색인 현황, 면책. (F2009, F2012)"""

import streamlit as st

from src.data_prep.fss_loader import LOAN_TYPE_NAMES
from src.services.loan_calc import Stats
from src.ui.common import disclaimers, product_label


def stats_inputs() -> Stats:
    st.subheader("나의 시작 스탯")
    st.caption("단위: 만원")
    d = Stats()
    cash = st.number_input("현금자산", 0, 500000, int(d.cash), step=500)
    emergency = st.number_input("비상금 (집값에 쓰지 않을 돈)", 0, 100000, int(d.emergency), step=100)
    income = st.number_input("월 소득 (세후)", 0, 10000, int(d.income), step=10)
    living = st.number_input("월 생활비", 0, 10000, int(d.living), step=10)
    existing = st.number_input("기존 대출 월 상환액", 0, 10000, int(d.existing_debt), step=10)
    goal = st.number_input("월 여유 목표", 0, 10000, int(d.margin_goal), step=10)
    return Stats(cash, emergency, income, living, existing, goal)


def product_picker(products: list[dict]) -> str | None:
    """대출 유형 -> 상품. 고른 상품은 st.session_state.selected 에 둔다."""
    st.subheader("상품 고르기")
    by_id = {p["product_id"]: p for p in products}
    types = list(LOAN_TYPE_NAMES)
    pending = st.session_state.pop("pending_pick", None)  # 상품 카드에서 고른 상품
    if pending in by_id:
        st.session_state.loan_type_pick = by_id[pending]["loan_type"]
        st.session_state.product_pick = pending
    loan_type = st.selectbox("대출 유형", types, key="loan_type_pick", format_func=LOAN_TYPE_NAMES.get)
    ids = [None] + sorted(
        (p["product_id"] for p in products if p["loan_type"] == loan_type), key=lambda pid: by_id[pid].get("title", "")
    )
    if st.session_state.get("product_pick") not in ids:
        st.session_state.product_pick = None
    pick = st.selectbox(
        "상품",
        ids,
        key="product_pick",
        format_func=lambda pid: "고르지 않음 (일반 질문)" if pid is None else product_label(by_id[pid]),
    )
    st.session_state.selected = pick
    return pick


def index_status(meta_count: int, docs_count: int) -> None:
    st.subheader("색인 현황")
    c1, c2 = st.columns(2)
    c1.metric("대출 상품", f"{meta_count}건")
    c2.metric("외부 문서 청크", f"{docs_count}건")


def footer() -> None:
    st.divider()
    disclaimers()
