"""대표 질문을 RAG 없이 한 번, RAG로 한 번 답하게 해 나란히 기록한다. (F2015)

실행:
  python -m src.eval.rag_compare                       # config/eval_questions.yaml 전부
  python -m src.eval.rag_compare --only q1 q4          # 일부만
  python -m src.eval.rag_compare --out data/processed/eval/rag_compare.md

같은 채팅 모델(settings.yaml llm.chat_model)로
- RAG 없이: 프롬프트(BASELINE_PROMPT)에 질문만 준다.
- RAG 사용: 답변 그래프(retrieve -> generate). 질문에 product_id 가 있으면 길잡이에서 그 상품을 고른 것과 같다.
세 기준으로 자동 점검하고, 사람이 최종 판단할 수 있게 근거·답변을 모두 남긴다.
- 사실 정확성: fact 는 공시값(코드가 loan_meta 에서 꺼냄)이 답변에 있는지, concept 는 핵심 낱말이 있는지
- 근거 제시: 답변이 인용한 근거(출처) 수. RAG 없이 답한 쪽은 출처를 댈 수 없다.
- 모름 인정: unknown 질문에 "확인되지 않습니다"로 답했는지
결과 파일에는 금감원 공시값이 들어가므로 기본 위치는 git 에서 빠진 data/processed/eval/ 이다.
"""

import argparse
import re
from datetime import date
from pathlib import Path

import openai

from src.config import NOTICES, ROOT_DIR, get_settings
from src.eval.search_compare import load_questions
from src.rag.graph import source_line
from src.rag.prompts import BASELINE_PROMPT
from src.services.compare import text_fields

UNKNOWN_MARKERS = ("확인되지 않", "알 수 없", "예측할 수 없", "예측하기 어렵")
NUMBER = re.compile(r"\d+(?:\.\d+)?\s*%")


def expected_values(item: dict, meta_store) -> list[str]:
    """fact 질문의 정답 값(답변에서 찾을 문자열). 공시값에서 코드가 꺼낸다."""
    check = item.get("check") or {}
    got = meta_store.get(ids=[item["product_id"]]) if item.get("product_id") else {"ids": []}
    if not got["ids"]:
        return []
    meta, text = got["metadatas"][0], got["documents"][0]
    if "meta" in check:
        value = meta.get(check["meta"])
        return [f"{value:.2f}".rstrip("0").rstrip(".")] if isinstance(value, (int, float)) else []
    if "field" in check:
        value = text_fields(text).get(check["field"], "")
        nums = [n.replace(" ", "").rstrip("%") for n in NUMBER.findall(value)]
        return nums or ([value[:15]] if value and value != "공시 없음" else [])
    return []


def judge(item: dict, answer: str, expected: list[str]) -> str:
    """자동 점검 결과: 맞음 / 틀림 / 판단 불가"""
    text = (answer or "").replace(" ", "")
    if item["kind"] == "unknown":
        return "맞음" if any(m.replace(" ", "") in text for m in UNKNOWN_MARKERS) else "틀림"
    if item["kind"] == "concept":
        return "맞음" if all(k in text for k in item.get("keywords", [])) else "틀림"
    if not expected:
        return "판단 불가"
    return "맞음" if any(v.replace(" ", "") in text for v in expected) else "틀림"


def run_item(item: dict, llm, app, meta_store) -> dict:
    baseline = llm.invoke(BASELINE_PROMPT.format_messages(question=item["question"])).content.strip()
    rag = app.invoke({"question": item["question"], "product_id": item.get("product_id")})
    expected = expected_values(item, meta_store) if item["kind"] == "fact" else []
    return {
        **item,
        "expected": expected,
        "baseline": baseline,
        "baseline_judge": judge(item, baseline, expected),
        "rag": rag["answer"],
        "rag_judge": judge(item, rag["answer"], expected),
        "sources": [source_line(ev) for ev in rag.get("sources", [])],
        "evidence_count": len(rag.get("evidence", [])),
    }


CRITERIA = {"fact": "사실 정확성", "concept": "사실 정확성", "unknown": "모름 인정"}


def to_markdown(rows: list[dict], model: str) -> str:
    def score(key):
        judged = [r for r in rows if r[key] != "판단 불가"]
        return f"{sum(r[key] == '맞음' for r in judged)}/{len(judged)}"

    out = [
        "# RAG 사용 전후 답변 비교 (F2015)",
        "",
        f"- 날짜: {date.today().isoformat()} · 모델: {model} (두 방식 같은 모델)",
        "- RAG 없이: 질문만 준 답변 / RAG 사용: loan_meta·loan_docs 근거로 답한 답변(인용 근거만 출처로 표시)",
        "- 자동 점검은 정답 값·핵심 낱말·'확인되지 않습니다' 포함 여부만 본다. 최종 판단은 아래 답변을 읽고 '사람 판단' 칸에 적는다.",
        "",
        "## 요약",
        "",
        "| 기준 | RAG 없이 | RAG 사용 |",
        "| :--- | :---: | :---: |",
        f"| 자동 점검 맞음 | {score('baseline_judge')} | {score('rag_judge')} |",
        f"| 근거 제시(출처 있는 답변) | 0/{len(rows)} | {sum(bool(r['sources']) for r in rows)}/{len(rows)} |",
        "",
        "| 번호 | 질문 | 기준 | 정답 값 | RAG 없이 | RAG 사용 | RAG 출처 수 | 사람 판단 |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | ---: | :--- |",
    ]
    for r in rows:
        expected = ", ".join(r["expected"]) or (", ".join(r.get("keywords", [])) if r["kind"] == "concept" else "-")
        out.append(
            f"| {r['id']} | {r['question']} | {CRITERIA[r['kind']]} | {expected} | {r['baseline_judge']} | "
            f"{r['rag_judge']} | {len(r['sources'])} |  |"
        )
    for r in rows:
        out += [
            "",
            f"## {r['id']}. {r['question']}" + (f" (상품 {r['product_id']})" if r.get("product_id") else ""),
            "",
            f"**RAG 없이** — 자동 점검: {r['baseline_judge']}",
            "",
            _quote(r["baseline"]),
            "",
            f"**RAG 사용** — 자동 점검: {r['rag_judge']} · 검색 근거 {r['evidence_count']}개, 인용 {len(r['sources'])}개",
            "",
            _quote(r["rag"]),
            "",
        ]
        out += [f"- {s}" for s in r["sources"]] or ["- 인용된 근거 없음"]
    out += ["", "---", NOTICES["disclaimer"], "", NOTICES["fss_source"]]
    return "\n".join(out) + "\n"


def _quote(text: str) -> str:
    return "\n".join("> " + line for line in (text or "").splitlines()) or "> (빈 답변)"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 전후 답변 비교")
    parser.add_argument("--only", nargs="*", help="질문 id (예: q1 q4)")
    parser.add_argument("--out", default=str(ROOT_DIR / "data" / "processed" / "eval" / "rag_compare.md"))
    args = parser.parse_args()

    from src.rag.ask import make_app
    from src.rag.llm import get_chat_model
    from src.vectorstore.store import get_embeddings, get_store

    settings = get_settings()
    llm = get_chat_model()
    emb = get_embeddings()
    app = make_app(llm=llm, embeddings=emb)
    meta_store = get_store(settings["vectorstore"]["meta_collection"], emb)
    items = [q for q in load_questions() if not args.only or q["id"] in args.only]

    rows, failed = [], []
    for item in items:
        print(f"[{item['id']}] {item['question']} ...", flush=True)
        try:
            row = run_item(item, llm, app, meta_store)
        except openai.APIError as e:  # 재시도 후에도 실패한 GMS 오류는 그 질문만 건너뛴다
            failed.append(item["id"])
            print(f"   모델 호출 실패({type(e).__name__}). 이 질문은 건너뜁니다.")
            continue
        print(f"   RAG 없이: {row['baseline_judge']} / RAG: {row['rag_judge']} (출처 {len(row['sources'])}개)")
        rows.append(row)
    if failed:
        print(f"\n실패한 질문: {' '.join(failed)}. 보고서에는 성공한 질문만 들어갑니다. GMS 일시 오류일 수 있으니 잠시 뒤 다시 실행하세요.")
    if not rows:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(rows, settings["llm"]["chat_model"]), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
