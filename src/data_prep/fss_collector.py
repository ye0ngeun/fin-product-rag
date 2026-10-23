"""금감원 금융상품한눈에 Open API에서 대출(주택담보·전세자금·개인신용) 상품을 수집한다.

실행: python -m src.data_prep.fss_collector
수집할 상품 유형과 권역은 config/settings.yaml 의 fss.product_types, fss.fin_groups 에서 정한다.
결과: data/raw/fss/{유형}.json (권역·전 페이지 합본) 과 상품 수 출력
"""

import json
import time
from datetime import datetime, timezone

import requests

from src.config import get_env, get_settings

PRODUCT_TYPES = {
    "mortgage": "mortgageLoanProductsSearch.json",  # 주택담보대출
    "rent_house": "rentHouseLoanProductsSearch.json",  # 전세자금대출
    "credit": "creditLoanProductsSearch.json",  # 개인신용대출
    "deposit": "depositProductsSearch.json",  # 정기예금
    "saving": "savingProductsSearch.json",  # 적금
}
GROUP_NAMES = {
    "020000": "은행",
    "030200": "여신전문",
    "030300": "저축은행",
    "050000": "보험",
    "060000": "금융투자",
}


def fetch_page(session: requests.Session, url: str, params: dict, fss_cfg: dict) -> dict:
    """한 페이지를 요청한다. 실패하면 간격을 늘려 재시도한다. (NF2006)"""
    last_error = None
    for attempt in range(1, fss_cfg["max_retries"] + 1):
        try:
            resp = session.get(url, params=params, timeout=fss_cfg["timeout_sec"])
            resp.raise_for_status()
            result = resp.json()["result"]
            if result.get("err_cd") != "000":
                raise RuntimeError(f"API 오류 {result.get('err_cd')}: {result.get('err_msg')}")
            return result
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
            last_error = e
            time.sleep(fss_cfg["request_interval_sec"] * 2**attempt)
    # 오류 문구의 요청 URL 에 인증키가 들어가므로 가린다 (NF2003)
    detail = str(last_error).replace(params["auth"], "***") if params.get("auth") else last_error
    raise RuntimeError(f"요청 실패 (pageNo={params.get('pageNo')}): {detail}")


def collect(product_type: str, api_key: str, fss_cfg: dict) -> dict:
    """한 상품 유형을 모든 권역, 모든 페이지에 걸쳐 수집한다."""
    url = f"{fss_cfg['base_url']}/{PRODUCT_TYPES[product_type]}"
    base_list, option_list = [], []
    with requests.Session() as session:
        for group in fss_cfg["fin_groups"]:
            page, max_page = 1, 1
            while page <= max_page:
                params = {"auth": api_key, "topFinGrpNo": group, "pageNo": page}
                result = fetch_page(session, url, params, fss_cfg)
                max_page = int(result.get("max_page_no") or 1)
                for row in result.get("baseList", []):
                    base_list.append({**row, "top_fin_grp_no": group})
                for row in result.get("optionList", []):
                    option_list.append({**row, "top_fin_grp_no": group})
                page += 1
                time.sleep(fss_cfg["request_interval_sec"])
    return {
        "product_type": product_type,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "source": "금융감독원 금융상품한눈에 Open API",
        "baseList": base_list,
        "optionList": option_list,
    }


def main() -> None:
    settings = get_settings()
    fss_cfg = settings["fss"]
    api_key = get_env("FSS_API_KEY")
    out_dir = settings["paths"]["raw_dir"] / "fss"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for product_type in fss_cfg["product_types"]:
        print(f"[{product_type}] 수집 중...", flush=True)
        data = collect(product_type, api_key, fss_cfg)
        path = out_dir / f"{product_type}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        products = {(r["fin_co_no"], r["fin_prdt_cd"]) for r in data["baseList"]}
        by_group = {
            GROUP_NAMES.get(g, g): sum(1 for r in data["baseList"] if r["top_fin_grp_no"] == g)
            for g in fss_cfg["fin_groups"]
        }
        total += len(products)
        print(f"[{product_type}] 상품 {len(products)}건 {by_group}, 금리옵션 {len(data['optionList'])}건 -> {path}")

    print(f"합계 상품 {total}건 ({'100건 이상 충족' if total >= 100 else '100건 미달'})")


if __name__ == "__main__":
    main()
