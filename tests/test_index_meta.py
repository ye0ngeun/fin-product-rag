"""F2002: 고정 ID 색인. 두 번 실행해도 청크 수가 늘지 않는다. (가짜 임베딩, 네트워크 없음)"""

import json
import shutil
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from src.vectorstore.index_meta import index_meta
from src.vectorstore.store import count_collection

FIXTURES = Path(__file__).parent / "fixtures"
TYPES = ["mortgage", "rent_house", "credit"]


def _run(raw_dir, chroma_dir, types=TYPES):
    return index_meta(
        types,
        sample=0,
        embeddings=DeterministicFakeEmbedding(size=16),
        persist_dir=chroma_dir,
        raw_dir=raw_dir,
    )


def test_reindex_does_not_duplicate(tmp_path):
    chroma_dir = tmp_path / "chroma"
    first = _run(FIXTURES, chroma_dir)
    assert first["loaded"] == 5
    assert first["after"] == 5

    second = _run(FIXTURES, chroma_dir)
    assert second["before"] == 5
    assert second["after"] == 5
    assert count_collection("loan_meta", persist_dir=chroma_dir) == 5


def test_removed_product_is_deleted_only_within_indexed_types(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURES / "fss", raw / "fss")
    chroma_dir = tmp_path / "chroma"
    _run(raw, chroma_dir)

    # 주담대에서 한 상품이 사라진 경우
    path = raw / "fss" / "mortgage.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["baseList"] = [r for r in data["baseList"] if r["fin_prdt_cd"] != "TEST-M2"]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    result = _run(raw, chroma_dir, types=["mortgage"])
    assert result["removed"] == 1
    # 전세·신용(3건)은 그대로, 주담대 1건
    assert result["after"] == 4
