"""Streamlit 진입점. 실행: streamlit run app.py"""

import streamlit as st

from src.config import NOTICES

st.set_page_config(page_title="대출 상품 RAG", page_icon="💰")
st.title("대출 상품 RAG")
st.info("아직 구현 전입니다.")
st.caption(NOTICES["disclaimer"])
st.caption(NOTICES["fss_source"])
