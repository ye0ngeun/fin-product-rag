"""대출 계획(장면 3·4)과 한 달 살아보기(장면 6). (F2010)

숫자는 모두 loan_calc 가 공시 금리로 계산한다. 입력값은 st.session_state.plan 에 두어 다른 화면에서도 쓴다.
"""

import streamlit as st

from src.config import ROOT_DIR, get_settings
from src.services.loan_calc import FAR, WARM, Stats, light_status, methods_for, month_steps, monthly_left, repayment, simple_ratios
from src.services.trades import TRADES_SOURCE, light_counts, load_trades, with_lights
from src.ui.common import CALC_NOTICE

RATE_PICKS = {"최저": "rate_min", "평균": "rate_avg", "최고": "rate_max"}
LIGHT_ICONS = {WARM: "🟡", "깜빡이는 불빛": "🟠", FAR: "⚫"}


def _plan() -> dict:
    return st.session_state.setdefault(
        "plan", {"price": 60000, "loan": 30000, "years": 30, "method": "원리금균등", "rate_pick": "평균", "rate": 4.0}
    )


def plan_rate(product: dict | None, plan: dict) -> tuple[float, str]:
    """(적용 연금리, 출처 설명). 상품을 고르면 그 상품의 공시 최저·평균·최고 중 하나."""
    if product and product.get(RATE_PICKS[plan["rate_pick"]]) is not None:
        rate = product[RATE_PICKS[plan["rate_pick"]]]
        return rate, f"금감원 공시 {product.get('title', '')}의 {plan['rate_pick']} 금리 (공시월 {product.get('dcls_month', '-')})"
    return plan["rate"], "직접 입력한 가정 금리"


def current_payment(product: dict | None) -> tuple[float, dict]:
    plan = _plan()
    rate, _ = plan_rate(product, plan)
    method = plan["method"] if plan["method"] in methods_for((product or {}).get("rpay_types")) else "원리금균등"
    r = repayment(plan["loan"], rate, plan["years"] * 12, method)
    return r.first_month, plan


def render_plan(product: dict | None, stats: Stats) -> None:
    plan = _plan()
    st.subheader("대출 계획")
    if product:
        st.caption(f"고른 상품: {product.get('title', '')}")
    else:
        st.caption("사이드바에서 상품을 고르면 그 상품의 공시 금리로 계산해요. 지금은 가정 금리를 씁니다.")

    c1, c2 = st.columns(2)
    plan["price"] = c1.number_input("살고 싶은 집 가격 (만원)", 1000, 500000, int(plan["price"]), step=1000)
    plan["loan"] = c2.slider("대출금 (만원)", 0, max(int(plan["price"]), 1000), min(int(plan["loan"]), int(plan["price"])), step=500)
    c3, c4 = st.columns(2)
    plan["years"] = c3.slider("기간 (년)", 1, 40, int(plan["years"]))
    methods = methods_for((product or {}).get("rpay_types"))
    if plan["method"] not in methods:
        plan["method"] = methods[0]
    plan["method"] = c4.selectbox("상환 방식", methods, index=methods.index(plan["method"]))
    if product and product.get("rate_min") is not None:
        picks = [k for k, f in RATE_PICKS.items() if product.get(f) is not None]
        if plan["rate_pick"] not in picks:
            plan["rate_pick"] = picks[0]
        plan["rate_pick"] = st.radio("적용 금리", picks, index=picks.index(plan["rate_pick"]), horizontal=True)
    else:
        plan["rate"] = st.number_input("가정 금리 (연 %)", 0.0, 20.0, float(plan["rate"]), step=0.1)

    rate, rate_source = plan_rate(product, plan)
    r = repayment(plan["loan"], rate, plan["years"] * 12, plan["method"])
    left = monthly_left(stats, r.first_month)
    m1, m2, m3 = st.columns(3)
    m1.metric("첫 달 상환액", f"{r.first_month:,.1f}만원")
    m2.metric("총이자", f"{r.total_interest:,.0f}만원")
    m3.metric("월 남는 돈", f"{left:,.1f}만원", delta=f"목표 대비 {left - stats.margin_goal:+,.1f}")
    if plan["method"] == "원금균등":
        st.caption(f"원금균등은 상환액이 줄어들어요. 마지막 달 {r.last_month:,.1f}만원")
    if plan["method"] == "만기일시":
        st.caption(f"만기일시는 매달 이자만 내고 마지막 달에 원금까지 {r.last_month:,.0f}만원을 갚아요.")
    st.caption(f"적용 금리 {rate:.2f}%: {rate_source}")

    need = plan["price"] - plan["loan"]
    status = light_status(plan["price"], stats, plan["loan"], r.first_month)
    ratios = simple_ratios(plan["price"], plan["loan"], stats, r.first_month)
    st.markdown(f"**이 집의 불빛: {LIGHT_ICONS[status]} {status}**  ·  필요한 자기자금 {need:,.0f}만원 / 쓸 수 있는 자기자금 {stats.own_funds:,.0f}만원")
    st.caption(
        f"대출금 ÷ 집값 {ratios['loan_to_price']:.0f}% · 연 상환액 ÷ 연소득 {ratios['payment_to_income'] or 0:.0f}% "
        "(단순 비율이며 제도상 LTV·DSR 한도와 다릅니다)"
    )
    st.caption(CALC_NOTICE)
    _render_lights(stats, plan["loan"], r.first_month)


def _render_lights(stats: Stats, loan: float, payment: float) -> None:
    trades = load_trades(ROOT_DIR / get_settings()["paths"]["raw_dir"] / "molit" / "apt_trade.csv")
    st.subheader("서울 실거래 사례의 불빛")
    if not trades:
        st.info("실거래 사례가 없어요. `python -m src.data_prep.molit_collector` 를 먼저 실행하세요.")
        return
    lit = with_lights(trades, stats, loan, payment)
    counts = light_counts(lit)
    st.dataframe(
        [{"구": gu, **{f"{LIGHT_ICONS[k]} {k}": v for k, v in c.items()}} for gu, c in sorted(counts.items())],
        hide_index=True,
        width="stretch",
    )
    gus = sorted(counts)
    c1, c2 = st.columns(2)
    gu = c1.selectbox("구", gus)
    light = c2.selectbox("불빛", list(LIGHT_ICONS), format_func=lambda k: f"{LIGHT_ICONS[k]} {k}")
    rows = sorted((t for t in lit if t.get("gu_nm") == gu and t["light"] == light), key=lambda t: t["price_manwon"])
    st.dataframe(
        [
            {
                "거래일": t.get("deal_date"),
                "동": t.get("umd_nm"),
                "아파트": t.get("apt_nm"),
                "평": t.get("area_pyeong"),
                "층": t.get("floor"),
                "거래가(만원)": t["price_manwon"],
                "야경 방향": t.get("direction"),
            }
            for t in rows[:200]
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(f"{len(rows)}건 중 최대 200건. 출처: {TRADES_SOURCE}. 과거 실거래 사례이며 지금 살 수 있는 매물이 아닙니다.")


def render_month(product: dict | None, stats: Stats) -> None:
    payment, plan = current_payment(product)
    rate, rate_source = plan_rate(product, plan)
    st.subheader("한 달 살아보기")
    st.caption(f"대출 {plan['loan']:,}만원 · {plan['years']}년 · {plan['method']} · 연 {rate:.2f}% 기준 (대출 계획에서 바꿀 수 있어요)")
    steps = month_steps(stats, payment)
    running, bars = 0.0, []
    for i, (label, amount) in enumerate(steps[:-1]):
        running = amount if i == 0 else running + amount
        bars.append({"단계": f"{i + 1}. {label.lstrip('- ')} 후" if i else "1. 월급", "남은 돈(만원)": round(running, 1)})
    st.bar_chart(bars, x="단계", y="남은 돈(만원)", horizontal=True)
    lines = []
    for label, amount in steps:
        if label.startswith("-"):
            lines.append(f"- {label.lstrip('- ')}: **−{abs(amount):,.1f}만원**")
        else:
            lines.append(f"- {label}: **{amount:,.1f}만원**")
    st.markdown("\n".join(lines))
    left = steps[-1][1]
    if left < 0:
        st.error(f"매달 {-left:,.1f}만원이 모자라요.")
    elif left < stats.margin_goal:
        st.warning(f"남는 돈 {left:,.1f}만원이 월 여유 목표 {stats.margin_goal:,.0f}만원보다 적어요.")
    else:
        st.success(f"월 여유 목표를 채우고 {left - stats.margin_goal:,.1f}만원이 더 남아요.")

    st.markdown("**적용 금리의 근거**")
    st.caption(rate_source)
    if product:
        option_lines = [line for line in product.get("text", "").splitlines() if line.startswith("- ")]
        if option_lines:
            st.code("\n".join(option_lines), language=None)
        st.caption(f"출처: 금감원 상품정보 · {product.get('title', '')} — {product.get('source_url', '')}")
    st.caption(CALC_NOTICE)
