"""대출 상품 메타데이터를 loan_meta 컬렉션에 색인한다. (F2001, F2002)

실행:
  python -m src.vectorstore.index_meta              # 3종 전부 색인
  python -m src.vectorstore.index_meta --dry-run    # 임베딩 없이 적재 건수와 샘플 텍스트만 확인
  python -m src.vectorstore.index_meta --types credit --sample 2

출력: 유형별 적재 건수, 색인 전후 청크 수, 샘플 색인 텍스트
"""

import argparse

from langchain_core.documents import Document

from src.config import get_settings
from src.data_prep.fss_loader import LOAN_TYPE_NAMES, load_all
from src.vectorstore.store import get_store, upsert_documents


def index_meta(loan_types: list[str], dry_run: bool = False, sample: int = 1, embeddings=None, persist_dir=None, raw_dir=None) -> dict:
    settings = get_settings()
    products, stats = load_all(raw_dir or settings["paths"]["raw_dir"], loan_types)

    print("== 적재 (F2001) ==")
    for loan_type, s in stats.items():
        name = LOAN_TYPE_NAMES[loan_type]
        if "missing" in s:
            print(f"[{name}] 파일 없음: {s['missing']} (python -m src.data_prep.fss_collector 먼저 실행)")
            continue
        print(
            f"[{name}] 상품 {s['products']}건 (baseList {s['base_rows']}행, 중복 {s['duplicates']}건 제거) / "
            f"금리옵션 {s['option_rows']}행 중 {s['options_matched']}행 병합, 옵션 없는 상품 {s['without_options']}건"
            + (f", 연결 안 된 옵션(상품 없음·공시월 다름) {s['options_orphan']}행" if s["options_orphan"] else "")
        )
    print(f"로드 합계: {len(products)}건")

    for p in products[:sample]:
        print(f"\n--- 샘플 색인 텍스트 ({p.id}) ---\n{p.text}")

    result = {"loaded": len(products), "stats": stats}
    if dry_run or not products:
        return result

    store = get_store(settings["vectorstore"]["meta_collection"], embeddings=embeddings, persist_dir=persist_dir)
    docs = [Document(page_content=p.text, metadata=p.metadata) for p in products]
    indexed_types = [t for t, s in stats.items() if "missing" not in s]
    r = upsert_documents(
        store,
        docs,
        [p.id for p in products],
        stale_where={"loan_type": {"$in": indexed_types}},
    )
    result.update(r)
    print("\n== 색인 (F2002) ==")
    print(
        f"컬렉션 {settings['vectorstore']['meta_collection']}: 색인 전 {r['before']}건 -> 후 {r['after']}건 "
        f"(덮어쓰기 {r['upserted']}건, 사라진 상품 삭제 {r['removed']}건)"
    )
    print("로드 건수와 색인 건수 일치" if r["after"] == len(products) else "주의: 로드 건수와 색인 건수가 다릅니다.")
    return result


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="대출 상품 메타데이터 색인 (loan_meta)")
    parser.add_argument("--types", nargs="+", choices=list(LOAN_TYPE_NAMES), help="색인할 유형 (기본: settings 의 대출 유형)")
    parser.add_argument("--dry-run", action="store_true", help="임베딩·색인 없이 적재 결과만 출력")
    parser.add_argument("--sample", type=int, default=1, help="출력할 샘플 텍스트 수")
    args = parser.parse_args()
    types = args.types or [t for t in settings["fss"]["product_types"] if t in LOAN_TYPE_NAMES]
    index_meta(types, dry_run=args.dry_run, sample=args.sample)


if __name__ == "__main__":
    main()
