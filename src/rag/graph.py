"""LangGraph 답변 그래프: retrieve -> generate 두 노드. (F2006~F2008)

retrieve: 두 컬렉션 검색·병합 (retriever.retrieve)
generate: 프롬프트3으로 근거만 써서 답하고, 답변에 인용된 [근거 n] 만 출처로 돌려준다.
"""

import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.rag.prompts import ANSWER_PROMPT
from src.rag.retriever import format_evidence

NOT_FOUND = "제공된 문서에서 확인되지 않습니다."
NO_CITATION_NOTICE = "답변에 인용된 근거가 없습니다."


class AnswerState(TypedDict, total=False):
    question: str
    product_id: str | None
    evidence: list[dict]
    answer: str
    sources: list[dict]
    notice: str


def cited_numbers(answer: str) -> list[int]:
    """'[근거 1][근거 3]', '[근거 1, 3]' 같은 표기에서 번호를 순서대로 뽑는다."""
    nums = []
    for group in re.findall(r"\[근거\s*([\d,\s]+)\]", answer or ""):
        for n in re.findall(r"\d+", group):
            if int(n) not in nums:
                nums.append(int(n))
    return nums


def source_line(ev: dict) -> str:
    """화면·CLI 에 보일 출처 한 줄: 제목과 원문 위치(URL 또는 파일·쪽)"""
    where = ev.get("source_url", "")
    if ev.get("source_file"):
        where = ev["source_file"] + (f" {ev['page']}쪽" if ev.get("page") else "")
    extra = f" (수집 {ev['collected_at'][:10]})" if ev.get("doc_type") == "meta" and ev.get("collected_at") else ""
    return f"[근거 {ev['n']}] {ev['label']} · {ev['title']} — {where}{extra}"


def build_graph(retrieve_fn, llm: BaseChatModel):
    """retrieve_fn(question, product_id) -> 근거 목록"""

    def retrieve_node(state: AnswerState) -> AnswerState:
        return {"evidence": retrieve_fn(state["question"], state.get("product_id"))}

    def generate_node(state: AnswerState) -> AnswerState:
        evidence = state.get("evidence", [])
        if not evidence:
            return {"answer": NOT_FOUND, "sources": [], "notice": NO_CITATION_NOTICE}
        reply = llm.invoke(ANSWER_PROMPT.format_messages(question=state["question"], evidence=format_evidence(evidence)))
        answer = reply.content.strip()
        by_n = {ev["n"]: ev for ev in evidence}
        sources = [by_n[n] for n in cited_numbers(answer) if n in by_n]
        return {"answer": answer, "sources": sources, "notice": "" if sources else NO_CITATION_NOTICE}

    graph = StateGraph(AnswerState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
