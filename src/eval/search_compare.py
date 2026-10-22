"""같은 질문에 유사도 검색과 MMR 검색을 각각 돌려 근거 목록을 비교한다. (F2014)

실행:
  python -m src.eval.search_compare "LTV가 뭐야?"
  python -m src.eval.search_compare "중도상환수수료는?" --product mortgage:0010001:1054
  python -m src.eval.search_compare --questions            # config/eval_questions.yaml 의 질문 전부
  (--out 파일.md 를 주면 결과를 마크다운으로도 저장)

비교 지표
- 근거 수, 서로 다른 원문 문서 수(같은 위키 문서의 여러 청크는 1개), 출처 종류 수(위키·사전·800선·상품)
- 같은 문서 중복: 근거 수 - 서로 다른 문서 수. MMR은 이 값이 줄어야 한다.
- 한쪽에만 나온 근거
MMR 설정은 settings.yaml 의 retrieval.mmr_fetch_k(후보 수), mmr_lambda(1에 가까울수록 관련도, 0에 가까울수록 다양성).
"""

import argparse
from pathlib import Path

import openai
import yaml

from src.config import ROOT_DIR, get_settings
from src.rag.retriever import retrieve
from src.vectorstore.store import get_embeddings, get_store

QUESTIONS_FILE = ROOT_DIR / "config" / "eval_questions.yaml"


def doc_key(ev: dict) -> str:
    """청크 ID에서 순번을 뗀 원문 문서 키 (common:wiki-355227:2 -> common:wiki-355227)"""
    return ev["id"] if ev["collection"] == "meta" else ev["id"].rsplit(":", 1)[0]


def diversity(evidence: list[dict]) -> dict:
    docs = {doc_key(ev) for ev in evidence}
    return {
        "근거 수": len(evidence),
        "서로 다른 문서": len(docs),
        "같은 문서 중복": len(evidence) - len(docs),
        "출처 종류": len({ev["doc_type"] for ev in evidence}),
    }


def compare_search(question: str, meta_store, doc_store, cfg: dict, product_id: str | None = None) -> dict:
    runs = {
        kind: retrieve(question, meta_store, doc_store, {**cfg, "search_type": kind}, product_id=product_id)
        for kind in ("similarity", "mmr")
    }
    ids = {kind: {ev["id"] for ev in evs} for kind, evs in runs.items()}
    return {
        "question": question,
        "product_id": product_id,
        "runs": runs,
        "stats": {kind: diversity(evs) for kind, evs in runs.items()},
        "only_similarity": sorted(ids["similarity"] - ids["mmr"]),
        "only_mmr": sorted(ids["mmr"] - ids["similarity"]),
    }


def _line(ev: dict) -> str:
    score = f"{ev['score']:.3f}" if ev.get("score") is not None else "-"
    return f"[{ev['n']}] {ev['label']} · {ev['title']} ({ev['id']}) [{ev.get('how', '')} {score}]"


def to_markdown(results: list[dict], cfg: dict) -> str:
    out = [
        "# 유사도 검색과 MMR 검색 비교 (F2014)",
        "",
        f"설정: top_k {cfg['top_k']}, mmr_fetch_k {cfg.get('mmr_fetch_k')}, mmr_lambda {cfg.get('mmr_lambda')}, "
        f"min_relevance {cfg.get('min_relevance')}, relative_margin {cfg.get('relative_margin')}",
        "",
        "| 질문 | 방식 | 근거 수 | 서로 다른 문서 | 같은 문서 중복 | 출처 종류 |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        for kind in ("similarity", "mmr"):
            s = r["stats"][kind]
            out.append(
                f"| {r['question']} | {kind} | {s['근거 수']} | {s['서로 다른 문서']} | {s['같은 문서 중복']} | {s['출처 종류']} |"
            )
    for r in results:
        out += ["", f"## {r['question']}" + (f" (상품 {r['product_id']})" if r["product_id"] else "")]
        for kind, title in (("similarity", "유사도"), ("mmr", "MMR")):
            out += ["", f"**{title}**", ""] + [f"- {_line(ev)}" for ev in r["runs"][kind]]
        out += ["", f"- 유사도에만: {', '.join(r['only_similarity']) or '없음'}", f"- MMR에만: {', '.join(r['only_mmr']) or '없음'}"]
    return "\n".join(out) + "\n"


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="유사도 vs MMR 근거 비교")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--product")
    parser.add_argument("--questions", action="store_true", help="config/eval_questions.yaml 의 질문 전부")
    parser.add_argument("--out", help="마크다운 결과 파일 경로")
    args = parser.parse_args()
    if not (args.question or args.questions):
        parser.error("질문을 주거나 --questions 를 주세요.")

    settings = get_settings()
    cfg = settings["retrieval"]
    emb = get_embeddings()
    meta_store = get_store(settings["vectorstore"]["meta_collection"], emb)
    doc_store = get_store(settings["vectorstore"]["doc_collection"], emb)
    items = load_questions() if args.questions else [{"question": args.question, "product_id": args.product}]

    results, failed = [], []
    for item in items:
        try:
            r = compare_search(item["question"], meta_store, doc_store, cfg, item.get("product_id"))
        except openai.APIError as e:  # 재시도 후에도 실패한 GMS 오류는 그 질문만 건너뛴다
            failed.append(item["question"])
            print(f"\n== {item['question']} ==\n임베딩 호출 실패({type(e).__name__}). 이 질문은 건너뜁니다.")
            continue
        results.append(r)
        print(f"\n== {r['question']}" + (f" (상품 {r['product_id']})" if r["product_id"] else "") + " ==")
        for kind in ("similarity", "mmr"):
            print(f"-- {kind}: {r['stats'][kind]}")
            for ev in r["runs"][kind]:
                print("   " + _line(ev))
        print(f"유사도에만: {r['only_similarity'] or '없음'}\nMMR에만: {r['only_mmr'] or '없음'}")
    if failed:
        print(f"\n실패한 질문 {len(failed)}개: {failed}. GMS 일시 오류일 수 있으니 잠시 뒤 다시 실행하세요.")
    if args.out and results:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(to_markdown(results, cfg), encoding="utf-8")
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
