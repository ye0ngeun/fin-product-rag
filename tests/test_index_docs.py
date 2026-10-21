"""F2003~F2005: 검색어 추출·후보 선별·캐시, 용어사전·PDF 적재, loan_docs 고정 ID 색인 (가짜 LLM·위키·임베딩)"""

from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config import get_settings
from src.data_prep.doc_collector import (
    CACHE_VERSION,
    WikiCache,
    collect_wiki,
    extract_queries,
    gather_candidates,
    select_candidates,
)
from src.data_prep.fss_loader import load_products
from src.data_prep.glossary_loader import load_all_glossaries, load_pdf
from src.vectorstore.index_docs import chunk_records, index_glossaries, index_wiki, reset_wiki

FIXTURES = Path(__file__).parent / "fixtures"
WIKI_CFG = {"max_queries": 3, "search_limit": 5, "max_docs_per_query": 1, "min_chars": 10}
LONG = "본문 " * 100


class FakeWiki:
    def __init__(self):
        self.searches, self.fetches = [], []
        self.results = {
            "주택담보대출": [{"pageid": 1, "title": "주택담보대출", "snippet": "주택을 담보로"}],
            "중도상환수수료": [
                {"pageid": 3, "title": "중도상환 (영화)", "snippet": "영화"},
                {"pageid": 2, "title": "중도상환수수료", "snippet": "대출을 미리 갚을 때"},
            ],
        }

    def search(self, query, limit):
        self.searches.append(query)
        return self.results.get(query, [])[:limit]

    def lookup_title(self, title):
        self.searches.append(f"title:{title}")
        return None

    def fetch_page(self, pageid):
        self.fetches.append(pageid)
        title = {1: "주택담보대출", 2: "중도상환수수료", 3: "중도상환 (영화)"}[pageid]
        return {"pageid": pageid, "title": title, "url": f"https://ko.wikipedia.org/?curid={pageid}", "text": LONG}


def test_extract_queries_always_includes_own_topic():
    llm = FakeListChatModel(responses=['```json\n["중도상환수수료", "주택담보대출"]\n```'])
    queries = extract_queries(llm, "수수료?", "상품", "주택담보대출", max_queries=3)
    assert queries == ["주택담보대출", "중도상환수수료"]  # 대출 유형이 맨 앞, 중복 제거


def test_select_keeps_own_topic_and_filters_index():
    cands = [{"pageid": 1, "title": "주택담보대출", "snippet": ""}, {"pageid": 9, "title": "담보", "snippet": ""}]
    llm = FakeListChatModel(responses=["[1, 7]"])  # 7 은 없는 번호
    picked = select_candidates(llm, "q", "p", "주택담보대출", cands, max_docs=1, own_topic="주택담보대출")
    assert [c["title"] for c in picked] == ["주택담보대출"]


def test_collect_wiki_uses_cache(tmp_path):
    cache = WikiCache(tmp_path / "cache.json")
    wiki = FakeWiki()
    # 프롬프트1, 프롬프트2(주택담보대출), 프롬프트2(중도상환수수료)
    llm = FakeListChatModel(responses=['["중도상환수수료"]', "[0]", "[1]"])
    pages, log = collect_wiki("수수료?", "상품", "주택담보대출", llm, wiki, cache, WIKI_CFG)
    assert [p["title"] for p in pages] == ["주택담보대출", "중도상환수수료"]
    assert log["steps"][1]["selected"] == ["중도상환수수료"]  # 영화 문서는 고르지 않음
    cache.save()

    # 같은 검색어는 다시 검색·수집하지 않는다 (NF2002)
    cache2 = WikiCache(tmp_path / "cache.json")
    wiki2 = FakeWiki()
    pages2, log2 = collect_wiki("수수료?", "상품", "주택담보대출", FakeListChatModel(responses=['["중도상환수수료"]']), wiki2, cache2, WIKI_CFG)
    assert pages2 == [] and wiki2.searches == [] and wiki2.fetches == []
    assert all(s["cached"] for s in log2["steps"])


def test_glossaries_load_with_encodings():
    records, stats = load_all_glossaries(FIXTURES / "docs", get_settings()["docs"])
    assert stats["hf"]["docs"] == 2  # 설명 빈 행 제외
    assert stats["epost"]["docs"] == 1  # CP949
    assert stats["epostbot"]["docs"] == 1
    assert "missing" in stats["bok800"]
    by_id = {r["doc_id"]: r for r in records}
    assert by_id["hf-W001"]["text"].startswith("담보인정비율 (LTV)")
    assert by_id["epost-1"]["metadata"]["doc_type"] == "glossary_epost"
    assert by_id["epostbot-1"]["metadata"]["title"] == "변동금리"


def _write_pdf(path: Path, pages: list[str]) -> None:
    """한 쪽에 영문 한 줄씩 들어간 최소 PDF"""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        content_no = len(objs)
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_no} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


def test_pdf_pages_keep_file_and_page(tmp_path):
    path = tmp_path / "경제금융용어 800선.pdf"
    _write_pdf(path, ["Cover", "LTV Loan To Value ratio is the ratio of a loan to the value of an asset."])
    src = {"key": "bok800", "doc_type": "glossary_bok", "name": "800선", "url": "u"}
    records = load_pdf(path, src)
    assert [r["doc_id"] for r in records] == ["bok800-p2"]  # 짧은 표지는 건너뜀
    assert records[0]["metadata"]["page"] == 2
    assert records[0]["metadata"]["source_file"] == path.name


def test_chunk_ids_are_stable_and_split_long_text():
    rec = {"doc_id": "wiki-1", "text": "가나다 " * 300, "metadata": {"product_id": "common", "doc_type": "wiki"}}
    docs, ids = chunk_records([rec], chunk_size=200, chunk_overlap=20)
    assert len(docs) > 1
    assert ids[:2] == ["common:wiki-1:0", "common:wiki-1:1"]
    assert chunk_records([rec], 200, 20)[1] == ids


def test_index_glossaries_twice_no_duplicates(tmp_path):
    emb = DeterministicFakeEmbedding(size=16)
    first = index_glossaries(docs_dir=FIXTURES / "docs", embeddings=emb, persist_dir=tmp_path)
    second = index_glossaries(docs_dir=FIXTURES / "docs", embeddings=emb, persist_dir=tmp_path)
    assert first["after"] == second["after"] == 4


def test_index_wiki_end_to_end(tmp_path):
    products, _ = load_products("mortgage", FIXTURES / "fss" / "mortgage.json")
    llm = FakeListChatModel(responses=['["중도상환수수료"]', "[0]", "[1]"])
    result = index_wiki(
        products[:1],
        question="중도상환수수료?",
        llm=llm,
        wiki=FakeWiki(),
        cache=WikiCache(tmp_path / "cache.json"),
        embeddings=DeterministicFakeEmbedding(size=16),
        persist_dir=tmp_path / "chroma",
    )
    assert result["pages"] == 2 and result["after"] >= 2
    assert (tmp_path / "cache.json").exists()


class ExactWiki(FakeWiki):
    """'중도상환수수료' 제목 문서가 있고, 전문 검색에는 잡음(베토벤)이 섞인 경우"""

    def search(self, query, limit):
        self.searches.append(query)
        return [
            {"pageid": 2, "title": "중도상환수수료", "snippet": "대출을 미리 갚을 때"},
            {"pageid": 7, "title": "루트비히 판 베토벤", "snippet": "독일의 작곡가"},
            {"pageid": 8, "title": "적격전환대출", "snippet": "기존 대출의 상환 부담을"},
        ]

    def lookup_title(self, title):
        if title == "중도상환수수료":
            return {"pageid": 2, "title": "중도상환수수료", "snippet": "만기 전에 갚을 때", "exact": True}
        return None


def test_gather_candidates_exact_first_and_filters_noise():
    cands = gather_candidates(ExactWiki(), "중도상환수수료", 10)
    assert [c["title"] for c in cands] == ["중도상환수수료", "적격전환대출"]  # 중복·베토벤 제외
    assert cands[0]["exact"] is True


def test_exact_title_always_kept_even_if_model_picks_none():
    cands = gather_candidates(ExactWiki(), "중도상환수수료", 10)
    picked = select_candidates(FakeListChatModel(responses=["[]"]), "q", "p", "중도상환수수료", cands, 2, "주택담보대출")
    assert [c["title"] for c in picked] == ["중도상환수수료"]


def test_cache_version_change_drops_queries_keeps_pages(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text('{"queries": {"고정금리": {"titles": []}}, "pages": {"1": {"title": "t"}}}', encoding="utf-8")
    cache = WikiCache(path)  # 버전 없는 예전 캐시
    assert cache.stale and cache.queries == {} and "1" in cache.pages
    cache.save()
    assert WikiCache(path).stale is False
    assert CACHE_VERSION >= 2


def test_reset_wiki_removes_only_wiki_chunks(tmp_path, monkeypatch):
    emb = DeterministicFakeEmbedding(size=16)
    index_glossaries(docs_dir=FIXTURES / "docs", embeddings=emb, persist_dir=tmp_path)
    products, _ = load_products("mortgage", FIXTURES / "fss" / "mortgage.json")
    index_wiki(
        products[:1],
        llm=FakeListChatModel(responses=['["중도상환수수료"]', "[0]", "[1]"]),
        wiki=FakeWiki(),
        cache=WikiCache(tmp_path / "cache.json"),
        embeddings=emb,
        persist_dir=tmp_path,
    )
    monkeypatch.setattr("src.vectorstore.index_docs.ROOT_DIR", tmp_path)
    assert reset_wiki(embeddings=emb, persist_dir=tmp_path) == 2
    from src.vectorstore.store import count_collection

    assert count_collection("loan_docs", persist_dir=tmp_path) == 4  # 용어사전만 남음
