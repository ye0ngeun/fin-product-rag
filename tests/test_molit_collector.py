"""국토부 실거래가 응답 파싱 (픽스처 XML은 응답 형식을 본뜬 가짜 데이터, 네트워크 없음)"""

from datetime import date
from pathlib import Path

import pytest

from src.data_prep.molit_collector import fetch_month, normalize, parse_response, recent_months

XML = (Path(__file__).parent / "fixtures" / "molit" / "apt_trade_page.xml").read_text(encoding="utf-8")
CFG = {"base_url": "http://test", "num_of_rows": 1, "max_retries": 2, "timeout_sec": 1, "request_interval_sec": 0}


def test_recent_months():
    assert recent_months(6, today=date(2026, 10, 21)) == ["202604", "202605", "202606", "202607", "202608", "202609"]
    assert recent_months(3, end_month="202602") == ["202512", "202601", "202602"]


def test_parse_and_normalize():
    items, total = parse_response(XML)
    assert total == 2 and len(items) == 2
    row = normalize(items[0], "11440", "마포구")
    assert row["price_manwon"] == 82500
    assert row["deal_date"] == "2026-09-07"
    assert row["area_m2"] == 84.97
    assert row["area_pyeong"] == 25.7
    assert row["floor"] == 12
    assert row["cancelled"] is False
    assert normalize(items[1], "11440", "마포구")["cancelled"] is True


def test_gateway_error_raises():
    xml = (
        "<OpenAPI_ServiceResponse><cmmMsgHeader><errMsg>SERVICE ERROR</errMsg>"
        "<returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</returnAuthMsg>"
        "<returnReasonCode>30</returnReasonCode></cmmMsgHeader></OpenAPI_ServiceResponse>"
    )
    with pytest.raises(RuntimeError, match="30"):
        parse_response(xml)


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class _Session:
    """totalCount=2 응답을 페이지당 1건씩 돌려준다."""

    def __init__(self):
        self.pages = []

    def get(self, url, params, timeout):
        self.pages.append(params["pageNo"])
        items = XML.split("<item>")
        keep = items[params["pageNo"]].split("</items>")[0]
        return _Resp(items[0] + "<item>" + keep + "</items>" + XML.split("</items>")[1])


def test_fetch_month_paginates():
    session = _Session()
    items = fetch_month(session, "key", "11440", "202609", CFG)
    assert len(items) == 2
    assert session.pages == [1, 2]


def test_seoul_region_table_from_cp949_csv():
    from src.data_prep.region_codes import find_code_file, load_seoul_regions

    path = find_code_file(Path(__file__).parent / "fixtures" / "codes")
    regions = load_seoul_regions(path)
    assert regions.lawd_cd("마포구") == "11440"
    assert regions.gu_by_code == {"11440": "마포구", "11680": "강남구"}  # 부산 제외
    assert regions.code_by_dong[("11440", "공덕동")] == "1144010200"
    assert "1144099900" not in regions.dong_by_code  # 폐지 제외
    with pytest.raises(KeyError):
        regions.lawd_cd("중구")  # 부산 중구는 서울 표에 없음


def test_normalize_attaches_dong_code_and_direction():
    from src.data_prep.molit_collector import direction_by_gu
    from src.data_prep.region_codes import find_code_file, load_seoul_regions

    regions = load_seoul_regions(find_code_file(Path(__file__).parent / "fixtures" / "codes"))
    directions = direction_by_gu({"서": ["마포구"], "남": ["강남구"]})
    items, _ = parse_response(XML)
    row = normalize(items[0], "11440", "마포구", regions, directions["마포구"])
    assert row["bjd_cd"] == "1144010200"  # umdCd 없음 -> 동 이름으로 찾음
    assert row["direction"] == "서"
