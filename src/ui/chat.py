"""길잡이: 고른 상품과의 질의응답. 대화 기록은 상품마다 따로 둔다. (F2009, F2012)"""

import streamlit as st

from src.ui.common import disclaimers, friendly_error, history_key, show_answer

SUGGESTIONS = {
    True: ["이 상품의 중도상환수수료는 어떻게 돼?", "이 상품 금리는 고정이야 변동이야?", "이 상품의 대출 한도는?"],
    False: ["LTV가 뭐야?", "고정금리와 변동금리는 뭐가 달라?", "원리금균등상환이 뭐야?"],
}


def _ask(guide, question: str, product_id: str | None, use_wiki: bool) -> None:
    key = history_key(product_id)
    history = st.session_state.histories.setdefault(key, [])
    prior = [{"role": m["role"], "content": m["content"]} for m in history]
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.status("길잡이가 찾는 중…", expanded=True) as status:
        try:
            result = guide.ask(question, product_id, prior, use_wiki=use_wiki, on_status=status.write)
        except Exception:
            status.update(label="답변 실패", state="error")
            friendly_error("답변 생성")
            history.pop()
            return
        wiki = result.get("wiki", {})
        if wiki.get("error"):
            status.write(wiki["error"])
        elif wiki.get("titles"):
            status.write("참고한 위키백과 문서: " + ", ".join(wiki["titles"]))
        status.update(label="답변 완료", state="complete", expanded=False)
    history.append({"role": "assistant", "content": result["answer"], "result": result})
    st.rerun()


def render_chat(guide, product: dict | None) -> None:
    product_id = product["product_id"] if product else None
    st.subheader("길잡이")
    if product:
        st.caption(f"{product.get('title', '')}에 대해 묻고 있어요. 다른 상품을 고르면 그 상품의 대화로 바뀌어요.")
    else:
        st.caption("상품을 고르지 않은 일반 질문이에요. 사이드바에서 상품을 고르면 상품별 대화가 따로 쌓여요.")
    use_wiki = st.toggle("질문할 때 위키백과 문서도 찾기", value=True, disabled=not product, help="상품을 고른 질문에서만 써요. 한 번 찾은 검색어는 다시 찾지 않아요.")

    history = st.session_state.histories.get(history_key(product_id), [])
    for msg in history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("result"):
                show_answer(msg["result"])
            else:
                st.markdown(msg["content"])
    if history:
        disclaimers(compact=True)

    picked = None
    cols = st.columns(3)
    for col, text in zip(cols, SUGGESTIONS[bool(product)]):
        if col.button(text, key=f"sg-{text}", width="stretch"):
            picked = text
    question = st.chat_input("길잡이에게 물어보세요") or picked
    if question:
        _ask(guide, question, product_id, use_wiki and bool(product))
