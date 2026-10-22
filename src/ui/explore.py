"""상품 찾기(F2011)와 비교(F2010) 화면 (장면 5)."""

import streamlit as st

from src.services.compare import compare_rows, plan_for
from src.services.loan_calc import Stats
from src.ui.common import CALC_NOTICE, PAGES, friendly_error, go, rate_text, show_answer

EXAMPLES = ["아파트 담보로 고정금리 4% 이하인 은행 주담대", "전세자금대출 중에 분할상환 되는 저축은행 상품", "마이너스통장 되는 신용대출"]
MAX_COMPARE = 3


def _add_compare(pid: str) -> None:
    ids = st.session_state.compare_ids
    if pid not in ids:
        ids.append(pid)
    st.session_state.compare_ids = ids[-MAX_COMPARE:]


def render_search(guide) -> None:
    st.subheader("조건으로 상품 찾기")
    condition = st.text_input("원하는 조건을 말로 적어 주세요", placeholder=EXAMPLES[0])
    st.caption("예: " + " / ".join(EXAMPLES))
    if st.button("찾기", type="primary", disabled=not condition):
        with st.status("상품을 찾는 중…", expanded=True) as status:
            try:
                st.session_state.search_result = guide.search(condition, on_status=status.write)
                status.update(label="찾기 끝", state="complete", expanded=False)
            except Exception:
                status.update(label="찾기 실패", state="error")
                friendly_error("상품 찾기")
                return

    result = st.session_state.search_result
    if result is None:
        return
    cond = {k: v for k, v in result.conditions.items() if v is not None}
    st.caption("해석한 조건: " + (", ".join(f"{k}={v}" for k, v in cond.items()) or "없음"))
    st.caption("후보가 줄어든 과정: " + " → ".join(f"{d} {n}건" for d, n in result.funnel))
    if result.message:
        st.warning(result.message)
    for card in result.cards:
        meta = card["metadata"]
        with st.container(border=True):
            st.markdown(f"**{card['title']}** · {meta.get('loan_type_nm', '')} · {meta.get('fin_grp_nm', '')}")
            st.caption(f"금리 {rate_text(meta)} · 평균 {meta.get('rate_avg', '-')}% · 공시월 {meta.get('dcls_month', '-')}")
            if card["matched"]:
                st.markdown(" ".join(f"`{m}`" for m in card["matched"]))
            if card["reason"]:
                st.caption(f"모델 판단: {card['reason']}")
            b1, b2 = st.columns(2)
            b1.button("이 상품으로 길잡이에게 묻기", key=f"ask-{card['product_id']}", on_click=go, args=(PAGES[3], card["product_id"]))
            b2.button("비교에 담기", key=f"cmp-{card['product_id']}", on_click=_add_compare, args=(card["product_id"],))


def render_compare(guide, products: list[dict], stats: Stats) -> None:
    st.subheader("상품 비교")
    by_id = {p["product_id"]: p for p in products}
    st.session_state.compare_ids = [pid for pid in st.session_state.compare_ids if pid in by_id]
    ids = st.multiselect(
        f"비교할 상품 (최대 {MAX_COMPARE}개)",
        sorted(by_id, key=lambda pid: by_id[pid].get("title", "")),
        default=st.session_state.compare_ids,
        max_selections=MAX_COMPARE,
        format_func=lambda pid: f"{by_id[pid].get('title', '')} · {by_id[pid].get('loan_type_nm', '')}",
    )
    if ids != st.session_state.compare_ids:
        st.session_state.compare_ids = ids
        st.session_state.compare_summary = None
    if len(ids) < 2:
        st.caption("상품을 2개 이상 담으면 비교표가 나와요. 위 검색 결과의 '비교에 담기'를 눌러도 돼요.")
        return
    chosen = [by_id[pid] for pid in ids]
    if len({p["loan_type"] for p in chosen}) > 1:
        st.warning("대출 유형이 다른 상품이 섞여 있어요. 같은 유형끼리 비교하는 것이 정확해요.")
    rows = compare_rows(chosen)
    st.dataframe(
        [{"항목": k, **{r["상품"]: r[k] for r in rows}} for k in rows[0] if k != "상품"],
        hide_index=True,
        width="stretch",
    )
    st.caption("표의 값은 금감원 공시값을 그대로 옮겼어요.")

    plan = st.session_state.get("plan", {"loan": 30000, "years": 30, "method": "원리금균등"})
    st.markdown(f"**같은 계획으로 계산하면** (대출 {plan['loan']:,}만원 · {plan['years']}년 · {plan['method']})")
    st.dataframe([plan_for(p, plan["loan"], plan["years"] * 12, plan["method"], stats) for p in chosen], hide_index=True, width="stretch")
    st.caption(CALC_NOTICE)

    if st.button("차이 요약 (근거 포함)"):
        with st.status("차이를 정리하는 중…", expanded=True) as status:
            try:
                st.session_state.compare_summary = guide.compare(ids, on_status=status.write)
                status.update(label="요약 끝", state="complete", expanded=False)
            except Exception:
                status.update(label="요약 실패", state="error")
                friendly_error("차이 요약")
    if st.session_state.compare_summary:
        with st.container(border=True):
            show_answer(st.session_state.compare_summary)
