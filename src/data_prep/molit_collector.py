"""국토교통부 아파트 매매 실거래가를 구·계약월 단위로 받아 CSV/JSON으로 저장한다.

《불빛 하나》 화면의 집 불빛(과거 거래 사례)에 쓰는 데이터이며 RAG 색인 대상이 아니다.

실행: python -m src.data_prep.molit_collector
     python -m src.data_prep.molit_collector --months 3 --end-month 202609
수집할 구와 기간은 config/settings.yaml 의 molit 에서 정한다. 인증키는 .env 의 MOLIT_API_KEY.
결과: data/raw/molit/apt_trade.json (원본 필드 포함), data/raw/molit/apt_trade.csv (정리본, 해제 거래 제외)
"""

import argparse
import csv
import json
import statistics
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from urllib.parse import unquote

import requests

from src.config import get_env, get_settings

SOURCE_NAME = "국토교통부 아파트 매매 실거래가 자료 (공공데이터포털)"
SUCCESS_CODES = {"000", "00"}
# 응답 태그: 현재 영문 태그, 예전 한글 태그도 함께 받는다.
FIELDS = {
    "sgg_cd": ["sggCd", "지역코드"],
    "umd_nm": ["umdNm", "법정동"],
    "apt_nm": ["aptNm", "아파트"],
    "jibun": ["jibun", "지번"],
    "apt_dong": ["aptDong"],
    "area_m2": ["excluUseAr", "전용면적"],
    "deal_year": ["dealYear", "년"],
    "deal_month": ["dealMonth", "월"],
    "deal_day": ["dealDay", "일"],
    "price_manwon": ["dealAmount", "거래금액"],
    "floor": ["floor", "층"],
    "build_year": ["buildYear", "건축년도"],
    "dealing_type": ["dealingGbn", "거래유형"],
    "cdeal_type": ["cdealType", "해제여부"],
    "cdeal_day": ["cdealDay", "해제사유발생일"],
}
CSV_COLUMNS = [
    "lawd_cd", "gu_nm", "umd_nm", "apt_nm", "jibun", "apt_dong", "area_m2", "area_pyeong",
    "floor", "build_year", "deal_date", "deal_ym", "price_manwon", "dealing_type",
]


def recent_months(n: int, end_month: str | None = None, today: date | None = None) -> list[str]:
    """end_month(YYYYMM, 없으면 지난달)부터 거꾸로 n개월을 오래된 순으로 돌려준다."""
    if end_month:
        y, m = int(str(end_month)[:4]), int(str(end_month)[4:6])
    else:
        today = today or date.today()
        y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    months = []
    for _ in range(n):
        months.append(f"{y}{m:02d}")
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return months[::-1]


def _to_int(value) -> int | None:
    s = str(value or "").replace(",", "").strip()
    return int(s) if s.lstrip("-").isdigit() else None


def _to_float(value) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def parse_response(xml_text: str) -> tuple[list[dict], int]:
    """응답 XML에서 거래 목록(원본 태그 그대로)과 totalCount 를 뽑는다."""
    root = ET.fromstring(xml_text)
    if root.tag == "OpenAPI_ServiceResponse":  # 게이트웨이 오류 (인증키, 트래픽 초과 등)
        msg = root.findtext(".//returnAuthMsg") or root.findtext(".//errMsg") or "알 수 없는 오류"
        code = root.findtext(".//returnReasonCode") or "?"
        raise RuntimeError(f"공공데이터포털 오류 {code}: {msg}")
    code = (root.findtext("header/resultCode") or "").strip()
    if code not in SUCCESS_CODES:
        raise RuntimeError(f"API 오류 {code}: {root.findtext('header/resultMsg')}")
    items = [{child.tag: (child.text or "").strip() for child in item} for item in root.iter("item")]
    total = _to_int(root.findtext("body/totalCount")) or 0
    return items, total


def normalize(raw: dict, lawd_cd: str, gu_nm: str) -> dict:
    """원본 거래 1건을 화면에서 쓰기 좋은 형태로 바꾼다. 가격 단위는 만원."""
    def pick(name):
        for tag in FIELDS[name]:
            if raw.get(tag) not in (None, ""):
                return raw[tag]
        return ""

    y, m, d = _to_int(pick("deal_year")), _to_int(pick("deal_month")), _to_int(pick("deal_day"))
    area = _to_float(pick("area_m2"))
    return {
        "lawd_cd": lawd_cd,
        "gu_nm": gu_nm,
        "umd_nm": pick("umd_nm"),
        "apt_nm": pick("apt_nm"),
        "jibun": pick("jibun"),
        "apt_dong": pick("apt_dong"),
        "area_m2": area,
        "area_pyeong": round(area / 3.3058, 1) if area else None,
        "floor": _to_int(pick("floor")),
        "build_year": _to_int(pick("build_year")),
        "deal_date": f"{y}-{m:02d}-{d:02d}" if y and m and d else "",
        "deal_ym": f"{y}{m:02d}" if y and m else "",
        "price_manwon": _to_int(pick("price_manwon")),
        "dealing_type": pick("dealing_type"),
        "cancelled": pick("cdeal_type").upper() == "O",
    }


def fetch_month(session: requests.Session, api_key: str, lawd_cd: str, deal_ym: str, cfg: dict) -> list[dict]:
    """한 구·한 계약월의 전 페이지를 받는다. 실패하면 간격을 늘려 재시도한다. (NF2006)"""
    items, page = [], 1
    while True:
        params = {
            "serviceKey": api_key,
            "LAWD_CD": lawd_cd,
            "DEAL_YMD": deal_ym,
            "pageNo": page,
            "numOfRows": cfg["num_of_rows"],
        }
        last_error = None
        for attempt in range(1, cfg["max_retries"] + 1):
            try:
                resp = session.get(cfg["base_url"], params=params, timeout=cfg["timeout_sec"])
                resp.raise_for_status()
                page_items, total = parse_response(resp.text)
                break
            except (requests.RequestException, ET.ParseError, RuntimeError) as e:
                last_error = e
                time.sleep(cfg["request_interval_sec"] * 2**attempt)
        else:
            raise RuntimeError(f"요청 실패 ({lawd_cd}, {deal_ym}, pageNo={page}): {last_error}")
        items.extend(page_items)
        time.sleep(cfg["request_interval_sec"])
        if not page_items or len(items) >= total:
            return items
        page += 1


def main() -> None:
    settings = get_settings()
    cfg = settings["molit"]
    parser = argparse.ArgumentParser(description="국토교통부 아파트 매매 실거래가 수집")
    parser.add_argument("--months", type=int, default=cfg["months"])
    parser.add_argument("--end-month", default=cfg.get("end_month"), help="마지막 계약년월 YYYYMM (기본: 지난달)")
    args = parser.parse_args()

    # 포털의 Encoding 키(%2B 등)를 넣어도 requests 가 다시 인코딩하지 않도록 풀어 둔다.
    api_key = unquote(get_env("MOLIT_API_KEY"))
    months = recent_months(args.months, args.end_month)
    districts = {str(k): v for k, v in cfg["districts"].items()}
    out_dir = settings["paths"]["raw_dir"] / "molit"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"수집 기간 {months[0]} ~ {months[-1]} ({len(months)}개월), 구 {len(districts)}개")

    raw_rows, rows = [], []
    with requests.Session() as session:
        for lawd_cd, gu_nm in districts.items():
            count = 0
            for deal_ym in months:
                for item in fetch_month(session, api_key, lawd_cd, deal_ym, cfg):
                    raw_rows.append({"lawd_cd": lawd_cd, "deal_ym_req": deal_ym, **item})
                    rows.append(normalize(item, lawd_cd, gu_nm))
                    count += 1
            print(f"[{gu_nm}] {count}건", flush=True)

    valid = [r for r in rows if not r["cancelled"] and r["price_manwon"]]
    (out_dir / "apt_trade.json").write_text(
        json.dumps(
            {
                "source": SOURCE_NAME,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "districts": districts,
                "months": months,
                "items": raw_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # Windows 엑셀에서 한글이 깨지지 않도록 BOM 포함
    with open(out_dir / "apt_trade.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(valid, key=lambda r: (r["lawd_cd"], r["deal_date"])))

    print(f"\n전체 {len(rows)}건, 해제·가격없음 제외 {len(rows) - len(valid)}건 -> CSV {len(valid)}건")
    for lawd_cd, gu_nm in districts.items():
        prices = [r["price_manwon"] for r in valid if r["lawd_cd"] == lawd_cd]
        if prices:
            print(f"[{gu_nm}] {len(prices)}건, 중위 거래가 {statistics.median(prices) / 10000:.2f}억원")
    print(f"저장: {out_dir / 'apt_trade.csv'}, {out_dir / 'apt_trade.json'}")


if __name__ == "__main__":
    main()
