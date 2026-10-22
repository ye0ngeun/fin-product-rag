"""서울 아파트 실거래 사례(data/raw/molit/apt_trade.csv)를 읽어 불빛 상태를 붙인다. (F2010)

실거래 사례는 RAG 근거가 아니라 시뮬레이션 데이터다. 과거 거래이며 매물이 아니라는 안내를 화면에 함께 둔다.
"""

import csv
from collections import Counter
from pathlib import Path

from src.services.loan_calc import BLINK, FAR, WARM, Stats, light_status

TRADES_SOURCE = "국토교통부 아파트 매매 실거래가 자료 (공공데이터포털)"


def load_trades(path: Path) -> list[dict]:
    """가격이 있는 거래만. 없으면 빈 목록."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    trades = []
    for row in rows:
        try:
            price = int(float(row.get("price_manwon") or 0))
        except ValueError:
            continue
        if price <= 0:
            continue
        trades.append({**row, "price_manwon": price, "area_pyeong": float(row.get("area_pyeong") or 0) or None})
    return trades


def with_lights(trades: list[dict], stats: Stats, loan: float, new_payment: float) -> list[dict]:
    return [{**t, "light": light_status(t["price_manwon"], stats, loan, new_payment)} for t in trades]


def light_counts(trades: list[dict]) -> dict[str, dict[str, int]]:
    """구별 불빛 수: {구: {따뜻한 불빛: n, 깜빡이는 불빛: n, 먼 불빛: n}}"""
    out: dict[str, Counter] = {}
    for t in trades:
        out.setdefault(t.get("gu_nm", ""), Counter())[t["light"]] += 1
    return {gu: {k: c.get(k, 0) for k in (WARM, BLINK, FAR)} for gu, c in out.items()}
