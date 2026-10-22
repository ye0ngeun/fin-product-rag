"""월 상환액, 한 달 살아보기, 실거래 사례 불빛 상태 계산. (F2010)

금융 숫자는 LLM이 아니라 이 모듈이 공시 금리와 사용자 입력으로 직접 만든다.
- 금리는 연 %, 금액은 만원 단위.
- 취득세·중개보수, 실제 LTV·DSR 규제 한도는 계산하지 않는다. 화면에 그 사실을 함께 적는다.
"""

from dataclasses import dataclass

METHODS = ("원리금균등", "원금균등", "만기일시")

# 금감원 상환방식 이름 -> 계산 방식
RPAY_TO_METHODS = {
    "분할상환방식": ("원리금균등", "원금균등"),
    "만기일시상환방식": ("만기일시",),
}


@dataclass
class Repayment:
    method: str
    annual_rate: float
    first_month: float
    last_month: float
    total_interest: float
    total_payment: float


def repayment(principal: float, annual_rate: float, months: int, method: str) -> Repayment:
    """대출금 principal(만원), 연금리 annual_rate(%), 기간 months 개월."""
    if method not in METHODS:
        raise ValueError(f"상환방식은 {METHODS} 중 하나: {method}")
    if principal <= 0 or months <= 0:
        return Repayment(method, annual_rate, 0, 0, 0, 0)
    r = annual_rate / 100 / 12
    if method == "원리금균등":
        pay = principal / months if r == 0 else principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
        total = pay * months
        return Repayment(method, annual_rate, pay, pay, total - principal, total)
    if method == "원금균등":
        base = principal / months
        interest = principal * r * (months + 1) / 2  # 매달 남은 원금 x r 의 합
        return Repayment(method, annual_rate, base + principal * r, base + base * r, interest, principal + interest)
    interest = principal * r * months  # 만기일시: 매달 이자만, 마지막 달에 원금
    return Repayment(method, annual_rate, principal * r, principal * r + principal, interest, principal + interest)


def methods_for(rpay_types: str | None) -> list[str]:
    """상품이 공시한 상환방식으로 계산할 수 있는 방식. 공시가 없으면(신용대출 등) 전부."""
    if not rpay_types:
        return list(METHODS)
    allowed = [m for name in rpay_types.split(",") for m in RPAY_TO_METHODS.get(name.strip(), ())]
    return allowed or list(METHODS)


@dataclass
class Stats:
    """시작 스탯 (만원)"""

    cash: float = 30000  # 현금자산
    emergency: float = 1000  # 비상금 (집값에 쓰지 않음)
    income: float = 400  # 월 소득 (세후)
    living: float = 200  # 월 생활비
    existing_debt: float = 0  # 기존 대출 월 상환액
    margin_goal: float = 50  # 월 여유 목표

    @property
    def own_funds(self) -> float:
        return max(self.cash - self.emergency, 0)


def month_steps(stats: Stats, new_payment: float) -> list[tuple[str, float]]:
    """한 달 살아보기: 월급에서 차례로 빠져나가고 남는 돈"""
    left = stats.income
    steps = [("월급", left)]
    for label, amount in [("생활비", stats.living), ("기존 상환", stats.existing_debt), ("새 대출 상환", new_payment)]:
        left -= amount
        steps.append((f"- {label}", -amount))
    steps.append(("남은 돈", left))
    return steps


def monthly_left(stats: Stats, new_payment: float) -> float:
    return stats.income - stats.living - stats.existing_debt - new_payment


# 불빛 상태: 따뜻한 불빛(살 수 있고 생활도 됨) / 깜빡이는 불빛(가격만 맞음) / 먼 불빛(가격이 넘음)
WARM, BLINK, FAR = "따뜻한 불빛", "깜빡이는 불빛", "먼 불빛"


def light_status(price: float, stats: Stats, loan: float, new_payment: float) -> str:
    if price > stats.own_funds + loan:
        return FAR
    return WARM if monthly_left(stats, new_payment) >= stats.margin_goal else BLINK


def simple_ratios(price: float, loan: float, stats: Stats, new_payment: float) -> dict:
    """화면용 단순 비율. 제도상 LTV·DSR(지역·주택 수·스트레스 금리 반영)과 다르다."""
    yearly_income = stats.income * 12
    return {
        "loan_to_price": loan / price * 100 if price else None,
        "payment_to_income": (stats.existing_debt + new_payment) * 12 / yearly_income * 100 if yearly_income else None,
    }
