"""질문 하나로 답변 그래프를 돌려 확인한다. (F2006~F2008 검증용)

실행:
  python -m src.rag.ask "중도상환수수료는 어떻게 계산돼?" --product mortgage:0010001:1054
  python -m src.rag.ask "전세자금대출 금리가 낮은 상품 알려줘"
  python -m src.rag.ask "..." --show-evidence    # 검색된 근거 전체도 출력
"""

import argparse
import time
from functools import partial

from src.config import NOTICES, get_settings
from src.rag.graph import build_graph, source_line
from src.rag.llm import get_chat_model
from src.rag.retriever import retrieve
from src.vectorstore.store import get_embeddings, get_store


def make_app(llm=None, embeddings=None, persist_dir=None, retrieval_cfg=None):
    settings = get_settings()
    retrieval_cfg = retrieval_cfg or settings["retrieval"]
    embeddings = embeddings or get_embeddings()
    meta_store = get_store(settings["vectorstore"]["meta_collection"], embeddings=embeddings, persist_dir=persist_dir)
    doc_store = get_store(settings["vectorstore"]["doc_collection"], embeddings=embeddings, persist_dir=persist_dir)

    def retrieve_fn(question, product_id):
        return retrieve(question, meta_store, doc_store, retrieval_cfg, product_id=product_id)

    return build_graph(retrieve_fn, llm or get_chat_model())


def main() -> None:
    parser = argparse.ArgumentParser(description="근거 기반 답변 확인")
    parser.add_argument("question")
    parser.add_argument("--product", help="선택한 상품 ID (예: mortgage:0010001:1054)")
    parser.add_argument("--show-evidence", action="store_true")
    args = parser.parse_args()

    app = make_app()
    print("== 그래프 ==")
    print(app.get_graph().draw_mermaid())

    start = time.perf_counter()
    result = app.invoke({"question": args.question, "product_id": args.product})
    elapsed = time.perf_counter() - start

    print(f"== 검색된 근거 {len(result['evidence'])}개 ==")
    for ev in result["evidence"]:
        score = f"{ev['score']:.3f}" if ev.get("score") is not None else "-"
        print(f"[근거 {ev['n']}] {ev['label']} · {ev['title']} ({ev['id']}) [{ev.get('how', '')} {score}]")
        if args.show_evidence:
            print("   " + ev["text"][:300].replace("\n", "\n   "))
    print(f"\n== 답변 ({elapsed:.1f}초) ==\n{result['answer']}")
    print("\n== 출처 ==")
    for ev in result["sources"]:
        print(source_line(ev))
    if result.get("notice"):
        print(result["notice"])
    print(f"\n{NOTICES['disclaimer']}\n{NOTICES['fss_source']}")


if __name__ == "__main__":
    main()
