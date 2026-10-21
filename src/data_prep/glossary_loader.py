"""용어사전 CSV와 한국은행 경제금융용어 800선 PDF를 색인용 레코드로 바꾼다. (F2005, F2016)

- 파일은 data/raw/docs/ 에 받은 이름 그대로 둔다. 어떤 파일을 쓸지는 settings.yaml 의 docs 에 패턴으로 적는다.
- 용어사전은 용어 1개 = 문서 1개 (doc_id = {출처키}-{용어 번호}). CSV 인코딩은 UTF-8/CP949 둘 다 읽는다.
- PDF는 페이지 1장 = 문서 1개 (doc_id = {출처키}-p{쪽}). 원문 위치는 파일명과 쪽 번호로 남긴다.
- 모두 특정 상품이 아닌 공통 문서(product_id=common)이고, 출처 종류는 doc_type 으로 구분한다.
"""

import csv
from pathlib import Path

from pypdf import PdfReader


def _norm(name: str) -> str:
    return (name or "").replace("﻿", "").replace(" ", "").strip()


def find_file(docs_dir: Path, pattern: str) -> Path | None:
    files = sorted(Path(docs_dir).glob(pattern))
    return files[-1] if files else None


def read_csv(path: Path) -> list[dict]:
    """열 이름의 공백·BOM을 없앤 dict 목록. 우체국 CSV는 CP949."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            with open(path, encoding=encoding, newline="") as f:
                return [{_norm(k): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없습니다: {path}")


def load_glossary(path: Path, src: dict) -> list[dict]:
    """src: settings 의 glossary 항목 (key, doc_type, name, url, columns{id, term, en, desc})"""
    cols = {k: _norm(v) for k, v in src["columns"].items() if v}
    rows = read_csv(path)
    if rows and (cols["term"] not in rows[0] or cols["desc"] not in rows[0]):
        raise KeyError(f"{path.name}: 열 {cols['term']}/{cols['desc']} 이 없습니다. 실제 열: {list(rows[0])}")
    records, seen = [], set()
    for i, row in enumerate(rows, start=1):
        term, desc = row.get(cols["term"], ""), row.get(cols["desc"], "")
        if not term or not desc:
            continue
        entry_id = row.get(cols.get("id", ""), "") or str(i)
        if entry_id in seen:  # 번호가 겹치면 행 번호로 구분
            entry_id = f"{entry_id}_{i}"
        seen.add(entry_id)
        en = row.get(cols.get("en", ""), "")
        records.append(
            {
                "doc_id": f"{src['key']}-{entry_id}",
                "text": f"{term}" + (f" ({en})" if en else "") + f"\n{desc}",
                "metadata": {
                    "product_id": "common",
                    "doc_type": src["doc_type"],
                    "title": term,
                    "source": src["name"],
                    "source_url": src["url"],
                    "source_file": path.name,
                },
            }
        )
    return records


def load_pdf(path: Path, src: dict) -> list[dict]:
    """src: settings 의 pdf 항목 (key, doc_type, name, url, skip_pages)"""
    reader = PdfReader(str(path))
    records = []
    for page_no, page in enumerate(reader.pages, start=1):
        if page_no <= src.get("skip_pages", 0):
            continue
        text = (page.extract_text() or "").strip()
        if len(text) < 30:  # 빈 쪽, 간지
            continue
        records.append(
            {
                "doc_id": f"{src['key']}-p{page_no}",
                "text": text,
                "metadata": {
                    "product_id": "common",
                    "doc_type": src["doc_type"],
                    "title": f"{src['name']} {page_no}쪽",
                    "source": src["name"],
                    "source_url": src["url"],
                    "source_file": path.name,
                    "page": page_no,
                },
            }
        )
    return records


def load_all_glossaries(docs_dir: Path, docs_cfg: dict) -> tuple[list[dict], dict]:
    """설정된 모든 용어사전·PDF. 파일이 없으면 건너뛰고 통계에 남긴다."""
    records, stats = [], {}
    for src in docs_cfg.get("glossaries", []):
        path = find_file(docs_dir, src["pattern"])
        if not path:
            stats[src["key"]] = {"missing": src["pattern"]}
            continue
        items = load_glossary(path, src)
        stats[src["key"]] = {"file": path.name, "docs": len(items), "doc_type": src["doc_type"]}
        records.extend(items)
    for src in docs_cfg.get("pdfs", []):
        path = find_file(docs_dir, src["pattern"])
        if not path:
            stats[src["key"]] = {"missing": src["pattern"]}
            continue
        items = load_pdf(path, src)
        stats[src["key"]] = {"file": path.name, "docs": len(items), "doc_type": src["doc_type"]}
        records.extend(items)
    return records, stats
