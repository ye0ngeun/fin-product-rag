"""대출 조건 비교표. (F2010)

표의 값은 모두 loan_meta 의 공시값(메타데이터와 상품 텍스트)에서 코드가 그대로 옮긴다. LLM은 쓰지 않는다.
차이 요약(근거 포함)은 Guide.compare 가 만든다.
"""

from src.services.loan_calc import Repayment, Stats, methods_for, monthly_left, repayment

# 상품 텍스트(fss_loader.build_text)의 줄 이름
TEXT_FIELDS = {"대출 한도": "한도", "중도상환 수수료": "중도상환수수료", "연체 이자율": "연체 이자율", "부대비용": "부대비용"}


def _pct(v) -> str:
    return f"{v:.2f}%" if isinstance(v, (int, float)) else "공시 없음"


def text_fields(text: str) -> dict[str, str]:
    """상품 텍스트에서 '이름: 값' 줄을 찾아 비교표 칸으로. 없으면 '공시 없음'."""
    out = {label: "공시 없음" for label in TEXT_FIELDS.values()}
    for line in (text or "").splitlines():
        name, _, value = line.partition(":")
        if name.strip() in TEXT_FIELDS and value.strip():
            out[TEXT_FIELDS[name.strip()]] = value.strip()
    return out


def compare_rows(products: list[dict]) -> list[dict]:
    """products: Guide.products() 항목(메타데이터 + text). 상품마다 한 행."""
    rows = []
    for p in products:
        rows.append(
            {
                "상품": p.get("title", ""),
                "권역": p.get("fin_grp_nm", ""),
                "유형": p.get("loan_type_nm", ""),
                "금리 범위": f"{_pct(p.get('rate_min'))} ~ {_pct(p.get('rate_max'))}",
                "평균 금리": _pct(p.get("rate_avg")),
                "금리 유형": (p.get("rate_types") or "공시 없음").replace(",", ", "),
                "상환 방식": (p.get("rpay_types") or "공시 없음").replace(",", ", "),
                **text_fields(p.get("text", "")),
                "공시월": p.get("dcls_month", ""),
            }
        )
    return rows


def plan_for(product: dict, principal: float, months: int, method: str, stats: Stats) -> dict:
    """같은 대출금·기간으로 상품마다 월 상환액과 남는 돈. 금리는 공시 평균(없으면 최저)."""
    rate = product.get("rate_avg") if product.get("rate_avg") is not None else product.get("rate_min")
    if rate is None:
        return {"상품": product.get("title", ""), "적용 금리": "공시 없음", "첫 달 상환액": None, "월 남는 돈": None}
    if method not in methods_for(product.get("rpay_types")):
        method = methods_for(product.get("rpay_types"))[0]
    r: Repayment = repayment(principal, rate, months, method)
    return {
        "상품": product.get("title", ""),
        "적용 금리": f"{rate:.2f}% (공시 {'평균' if product.get('rate_avg') is not None else '최저'})",
        "상환 방식": method,
        "첫 달 상환액": round(r.first_month, 1),
        "총이자": round(r.total_interest),
        "월 남는 돈": round(monthly_left(stats, r.first_month), 1),
    }
