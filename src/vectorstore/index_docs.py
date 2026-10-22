"""외부 문서를 청킹해 loan_docs 컬렉션에 색인한다. (F2003~F2005)

실행:
  python -m src.vectorstore.index_docs --glossary
      용어사전 CSV 3종 + 한은 800선 PDF (data/raw/docs/) 색인
  python -m src.vectorstore.index_docs --product mortgage:0010001:1054 --question "중도상환수수료가 얼마야?"
      한 상품·질문으로 위키 검색어 추출 -> 검색 -> 후보 선별 -> 수집 -> 색인
  python -m src.vectorstore.index_docs --seed 2
      대출 유형마다 상품 2개씩 기본 질문으로 위키 문서를 미리 모은다
  python -m src.vectorstore.index_docs --reset-wiki
      loan_docs 의 위키 청크와 위키 캐시를 지운다 (선별 규칙을 바꾼 뒤 처음부터 다시 모을 때)
  (--dry-run 은 색인하지 않고 건수·검색어·샘플만 출력)

청크 ID: {product_id}:{doc_id}:{순번} 고정. 다시 실행해도 덮어쓰기만 한다.
청크 크기·중복은 settings.yaml 의 chunking 에서 바꾼다.
"""

import argparse
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ROOT_DIR, get_settings
from src.data_prep.doc_collector import WikiCache, collect_wiki, wiki_to_records
from src.data_prep.fss_loader import LOAN_TYPE_NAMES, load_all
from src.data_prep.glossary_loader import load_all_glossaries
from src.vectorstore.store import get_store, upsert_documents

DEFAULT_QUESTION = "이 대출 상품의 금리, 한도, 상환 방식, 수수료 조건을 설명해줘."


def chunk_records(records: list[dict], chunk_size: int, chunk_overlap: int) -> tuple[list[Document], list[str]]:
    """레코드를 청킹한다. 짧은 용어 항목은 청크 1개가 된다."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs, ids = [], []
    for rec in records:
        for seq, piece in enumerate(splitter.split_text(rec["text"])):
            meta = {**rec["metadata"], "doc_id": rec["doc_id"], "chunk_seq": seq}
            docs.append(Document(page_content=piece, metadata=meta))
            ids.append(f"{meta['product_id']}:{rec['doc_id']}:{seq}")
    return docs, ids


def _index(records, settings, dry_run, stale_where=None, embeddings=None, persist_dir=None) -> dict:
    chunking = settings["chunking"]
    docs, ids = chunk_records(records, chunking["chunk_size"], chunking["chunk_overlap"])
    by_type = Counter(d.metadata["doc_type"] for d in docs)
    print(f"청크 {len(docs)}개 (크기 {chunking['chunk_size']}, 중복 {chunking['chunk_overlap']}) {dict(by_type)}")
    if docs:
        print(f"--- 샘플 청크 ({ids[0]}) ---\n{docs[0].page_content[:300]}")
    if dry_run or not docs:
        return {"chunks": len(docs)}
    store = get_store(settings["vectorstore"]["doc_collection"], embeddings=embeddings, persist_dir=persist_dir)
    r = upsert_documents(store, docs, ids, stale_where=stale_where)
    print(
        f"컬렉션 {settings['vectorstore']['doc_collection']}: 색인 전 {r['before']}건 -> 후 {r['after']}건 "
        f"(덮어쓰기 {r['upserted']}건, 사라진 청크 삭제 {r['removed']}건)"
    )
    return {"chunks": len(docs), **r}


def index_glossaries(dry_run=False, docs_dir=None, embeddings=None, persist_dir=None) -> dict:
    settings = get_settings()
    docs_dir = docs_dir or ROOT_DIR / settings["docs"]["dir"]
    records, stats = load_all_glossaries(docs_dir, settings["docs"])
    print("== 용어사전·PDF 적재 ==")
    for key, s in stats.items():
        print(f"[{key}] " + (f"파일 없음 ({docs_dir}/{s['missing']})" if "missing" in s else f"{s['file']}: {s['docs']}건"))
    loaded_types = [s["doc_type"] for s in stats.values() if "missing" not in s]
    return _index(
        records,
        settings,
        dry_run,
        stale_where={"doc_type": {"$in": loaded_types}} if loaded_types else None,
        embeddings=embeddings,
        persist_dir=persist_dir,
    )


def index_wiki(products, question=None, dry_run=False, llm=None, wiki=None, cache=None, embeddings=None, persist_dir=None):
    """products: fss_loader.LoanProduct 목록. 상품마다 검색어 추출부터 색인까지 한다."""
    settings = get_settings()
    if llm is None:
        from src.rag.llm import get_chat_model

        llm = get_chat_model()
    if wiki is None:
        from src.data_prep.wiki_client import WikiClient

        wiki = WikiClient()
    cache = cache or WikiCache(ROOT_DIR / settings["wiki"]["cache_file"])
    if cache.stale:
        print("위키 캐시 버전이 바뀌어 검색어를 다시 수집합니다. (이미 받은 문서는 다시 받지 않음)")

    pages, logs = [], []
    for p in products:
        own_topic = LOAN_TYPE_NAMES[p.metadata["loan_type"]]
        new_pages, log = collect_wiki(question or DEFAULT_QUESTION, p.text, own_topic, llm, wiki, cache, settings["wiki"])
        print(f"\n== {p.id} ({p.metadata.get('title', '')}) ==")
        print(f"검색어(프롬프트1): {log['queries']}")
        for step in log["steps"]:
            if step["cached"]:
                print(f"  - {step['query']}: 이미 수집함 (캐시) {step['titles']}")
            else:
                print(f"  - {step['query']}: 후보 {step['candidates']}")
                judged = ", ".join(f"{j['title']}({j['kind']}{'·관련' if j['relevant'] else ''})" for j in step["judged"])
                print(f"    판단(프롬프트2): {judged}")
                print(f"    -> 선별 {step['selected']}")
        print(f"새로 받은 문서 {len(new_pages)}건: {[x['title'] for x in new_pages]}")
        pages.extend(new_pages)
        logs.append(log)

    result = _index(wiki_to_records(pages), settings, dry_run, embeddings=embeddings, persist_dir=persist_dir)
    if not dry_run:
        cache.save()  # 색인이 끝난 뒤에만 캐시에 남긴다
    return {**result, "pages": len(pages), "logs": logs}


def reset_wiki(embeddings=None, persist_dir=None) -> int:
    """위키 청크와 캐시를 지운다. 지운 청크 수를 돌려준다."""
    settings = get_settings()
    store = get_store(settings["vectorstore"]["doc_collection"], embeddings=embeddings, persist_dir=persist_dir)
    before = store._collection.count()
    ids = store.get(where={"doc_type": "wiki"}, include=[])["ids"]
    if ids:
        store.delete(ids=ids)
    print(f"loan_docs: {before}건 -> {store._collection.count()}건")
    cache_path = ROOT_DIR / settings["wiki"]["cache_file"]
    if cache_path.exists():
        cache_path.unlink()
    print(f"위키 청크 {len(ids)}개와 캐시({cache_path.name})를 지웠습니다.")
    return len(ids)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="외부 문서 수집·색인 (loan_docs)")
    parser.add_argument("--glossary", action="store_true", help="용어사전 CSV·PDF 색인")
    parser.add_argument("--product", action="append", help="상품 ID (여러 번 줄 수 있음)")
    parser.add_argument("--question", help="질문 (없으면 기본 질문)")
    parser.add_argument("--seed", type=int, default=0, help="대출 유형마다 앞에서부터 N개 상품으로 위키 수집")
    parser.add_argument("--reset-wiki", action="store_true", help="위키 청크와 위키 캐시 삭제")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not (args.glossary or args.product or args.seed or args.reset_wiki):
        parser.error("--glossary, --product, --seed, --reset-wiki 중 하나 이상을 주세요.")

    if args.reset_wiki:
        reset_wiki()

    if args.glossary:
        index_glossaries(dry_run=args.dry_run)

    if args.product or args.seed:
        types = [t for t in settings["fss"]["product_types"] if t in LOAN_TYPE_NAMES]
        products, _ = load_all(settings["paths"]["raw_dir"], types)
        chosen = []
        if args.product:
            by_id = {p.id: p for p in products}
            missing = [pid for pid in args.product if pid not in by_id]
            if missing:
                parser.error(f"없는 상품 ID: {missing}")
            chosen += [by_id[pid] for pid in args.product]
        for t in types if args.seed else []:
            chosen += [p for p in products if p.metadata["loan_type"] == t][: args.seed]
        index_wiki(chosen, question=args.question, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
