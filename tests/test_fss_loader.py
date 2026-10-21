"""F2001: 금감원 대출 JSON 적재와 색인 텍스트 구성 (픽스처 사용, 네트워크 없음)"""

from pathlib import Path

from src.data_prep.fss_loader import load_all, load_products

FIXTURES = Path(__file__).parent / "fixtures"


def test_mortgage_merges_options_and_dedupes():
    products, stats = load_products("mortgage", FIXTURES / "fss" / "mortgage.json")
    assert stats["products"] == 2
    assert stats["duplicates"] == 1
    assert stats["options_orphan"] == 1
    assert stats["without_options"] == 1

    p = next(p for p in products if p.id == "mortgage:0010001:1054")
    assert len(p.options) == 2  # 같은 옵션 행 중복 제거
    for expected in [
        "우리은행",
        "우리아파트론",
        "주택담보대출",
        "고정금리",
        "변동금리",
        "분할상환방식",
        "최저 3.52%",
        "최고 5.25%",
        "LTV 70%",
        "중도상환대출금×0.71%×잔존기간÷대출기간",
        "가계대출 최고연체이자율 : 12%",
        "국민주택채권 매입",
        "영업점, 모집인",
        "2026-10-20 ~ 2027-10-19",
    ]:
        assert expected in p.text, expected
    assert p.metadata["rate_min"] == 3.52
    assert p.metadata["rate_max"] == 5.25
    assert p.metadata["fin_grp_nm"] == "은행"
    assert p.metadata["title"] == "우리은행 우리아파트론"


def test_metadata_has_only_scalar_values():
    products, _ = load_all(FIXTURES, ["mortgage", "rent_house", "credit"])
    for p in products:
        for value in p.metadata.values():
            assert isinstance(value, (str, int, float, bool)), (p.id, value)


def test_product_without_options_has_no_rate_metadata():
    products, _ = load_products("mortgage", FIXTURES / "fss" / "mortgage.json")
    p = next(p for p in products if p.id == "mortgage:0013175:TEST-M2")
    assert "공시된 옵션 없음" in p.text
    assert "별도 종료일 없음" in p.text
    assert "rate_min" not in p.metadata


def test_credit_uses_score_bands_and_loan_rate_row():
    products, stats = load_products("credit", FIXTURES / "fss" / "credit.json")
    assert stats["products"] == 2
    ids = sorted(p.id for p in products)
    # 같은 상품코드에 구분이 둘이면 ID 끝에 구분 코드를 붙인다
    assert ids == ["credit:0010001:TEST-C1:1", "credit:0010001:TEST-C1:2"]

    p = next(p for p in products if p.id.endswith(":1"))
    assert "일반신용대출" in p.text
    assert "900점 초과 4.50%" in p.text
    assert "501~600점 10.40%" in p.text
    # 범위는 대출금리 행만으로 계산 (기준금리 3.2 는 제외)
    assert p.metadata["rate_min"] == 4.5
    assert p.metadata["rate_max"] == 10.4
    assert p.metadata["rate_avg"] == 5.4


def test_missing_file_is_reported(tmp_path):
    products, stats = load_all(tmp_path, ["mortgage"])
    assert products == []
    assert "missing" in stats["mortgage"]
