"""《불빛 하나》 Streamlit 화면. 실행: streamlit run app.py

화면 구성 (기획안 2.6):
- 사이드바: 시작 스탯, 상품 고르기, 색인 현황, 면책
- 대출 계획 / 상품 찾기·비교 / 한 달 살아보기 / 길잡이
화면은 src/ui, 기능은 src/services 에 둔다. (NF2005)
"""

import os

import streamlit as st

from src.config import ROOT_DIR, get_settings
from src.ui.chat import render_chat
from src.ui.common import PAGES, disclaimers, init_state
from src.ui.explore import render_compare, render_search
from src.ui.plan import render_month, render_plan
from src.ui.sidebar import footer, index_status, product_picker, stats_inputs
from src.vectorstore.store import count_collection

st.set_page_config(page_title="불빛 하나", page_icon="🏠", layout="wide")
settings = get_settings()
init_state()


@st.cache_resource(show_spinner="검색 준비 중…")
def get_guide():
    from src.services.guide import Guide

    return Guide()


@st.cache_data(show_spinner=False)
def load_products(meta_count: int) -> list[dict]:
    """meta_count 가 바뀌면(재색인) 다시 읽는다."""
    return get_guide().products()


meta_count = count_collection(settings["vectorstore"]["meta_collection"])
docs_count = count_collection(settings["vectorstore"]["doc_collection"])

st.title("🏠 불빛 하나")
st.caption("대출 계획을 바꾸면 서울의 불빛과 한 달 생활이 함께 바뀌어요. 낯선 말은 길잡이에게 근거와 함께 물어보세요.")

# 실행 전 점검 (F2012): 키가 없거나 색인이 비면 안내하고 멈춘다.
problems = []
if not os.getenv("OPENAI_API_KEY"):
    problems.append(".env 에 OPENAI_API_KEY(GMS 키)가 없습니다. `.env.example` 을 복사해 키를 넣고 다시 실행하세요.")
if meta_count == 0:
    problems.append("대출 상품 색인이 없습니다. `python -m src.vectorstore.index_meta` 를 먼저 실행하세요.")
if problems:
    with st.sidebar:
        index_status(meta_count, docs_count)
        footer()
    for p in problems:
        st.error(p)
    st.caption(f"프로젝트 폴더: {ROOT_DIR.name}")
    disclaimers()
    st.stop()
if docs_count == 0:
    st.warning(
        "외부 문서 색인이 없어 길잡이가 용어를 설명하지 못할 수 있어요. "
        "`python -m src.vectorstore.index_docs --glossary` 를 실행하세요."
    )

try:
    guide = get_guide()
    products = load_products(meta_count)
except Exception:
    st.error("검색 준비 중 문제가 생겼어요. .env 설정(OPENAI_API_KEY, OPENAI_BASE_URL)을 확인하세요.")
    st.stop()
by_id = {p["product_id"]: p for p in products}

with st.sidebar:
    stats = stats_inputs()
    st.divider()
    selected = product_picker(products)
    st.divider()
    index_status(meta_count, docs_count)
    footer()

product = by_id.get(selected)
page = st.segmented_control("화면", PAGES, key="page", label_visibility="collapsed") or PAGES[0]

if page == PAGES[0]:
    render_plan(product, stats)
elif page == PAGES[1]:
    render_search(guide)
    st.divider()
    render_compare(guide, products, stats)
elif page == PAGES[2]:
    render_month(product, stats)
else:
    render_chat(guide, product)

st.divider()
disclaimers()
