"""Streamlit 진입점. 실행: streamlit run app.py"""

import streamlit as st

from src.config import NOTICES, get_settings
from src.vectorstore.store import count_collection

st.set_page_config(page_title="대출 상품 RAG", page_icon="💰")

# 현재 색인 건수 (F2002)
settings = get_settings()
with st.sidebar:
    st.subheader("색인 현황")
    st.metric("대출 상품 (loan_meta)", f"{count_collection(settings['vectorstore']['meta_collection'])}건")

st.title("대출 상품 RAG")
st.info("아직 구현 전입니다.")
st.caption(NOTICES["disclaimer"])
st.caption(NOTICES["fss_source"])
