"""금감원 대출 상품 JSON을 읽어 상품 1건 = 색인 텍스트 1개로 만든다. (F2001)

입력: data/raw/fss/{유형}.json (fss_collector 결과, baseList + optionList)
출력: LoanProduct 목록. text 는 임베딩할 본문, metadata 는 Chroma 메타데이터(스칼라 값만).

필드는 금감원 Open API 상세(주택담보 menuNo=700055, 전세 700056, 개인신용 700057) 기준.
상품 키는 (fin_co_no, fin_prdt_cd) 이고 ID 는 `{유형}:{fin_co_no}:{fin_prdt_cd}` 로 고정한다.
개인신용대출은 같은 상품코드에 상품 구분(crdt_prdt_type)이 여러 개면 ID 끝에 구분 코드를 붙인다.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

LOAN_TYPE_NAMES = {
    "mortgage": "주택담보대출",
    "rent_house": "전세자금대출",
    "credit": "개인신용대출",
}
GROUP_NAMES = {
    "020000": "은행",
    "030200": "여신전문",
    "030300": "저축은행",
    "050000": "보험",
    "060000": "금융투자",
}
# 개인신용대출 optionList 의 신용점수 구간 필드 (금감원 API 명세)
CREDIT_GRADES = {
    "crdt_grad_1": "900점 초과",
    "crdt_grad_4": "801~900점",
    "crdt_grad_5": "701~800점",
    "crdt_grad_6": "601~700점",
    "crdt_grad_10": "501~600점",
    "crdt_grad_11": "401~500점",
    "crdt_grad_12": "301~400점",
    "crdt_grad_13": "300점 이하",
}
CREDIT_LOAN_RATE_TYPE = "A"  # crdt_lend_rate_type: A 대출금리, B 기준금리, C 가산금리, D 가감조정금리
SOURCE_NAME = "금융감독원 금융상품한눈에"
SOURCE_URL = "https://finlife.fss.or.kr"


@dataclass
class LoanProduct:
    id: str
    text: str
    metadata: dict
    options: list[dict] = field(default_factory=list)


def _to_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}%"


def _ymd(value) -> str:
    s = str(value or "").strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else (s or "-")


def _flatten(value) -> str:
    """'- 고정금리 : ...\\n- 변동금리 : ...' 같은 여러 줄 값을 한 줄로 잇는다. 내용은 그대로 둔다."""
    lines = [line.strip() for line in str(value or "").splitlines()]
    lines = [line[2:].strip() if line.startswith("- ") else line for line in lines]
    return " / ".join(line for line in lines if line)


def _rate_summary(values: list[float | None]) -> tuple[float | None, float | None]:
    nums = [v for v in values if v is not None]
    return (min(nums), max(nums)) if nums else (None, None)


def _option_line_secured(opt: dict) -> str:
    """주택담보·전세: 담보유형 · 상환방식 · 금리유형: 최저/최고/평균"""
    labels = [opt.get("mrtg_type_nm"), opt.get("rpay_type_nm"), opt.get("lend_rate_type_nm")]
    label = " · ".join(x for x in labels if x)
    rates = (
        f"최저 {_pct(_to_float(opt.get('lend_rate_min')))} · "
        f"최고 {_pct(_to_float(opt.get('lend_rate_max')))} · "
        f"평균 {_pct(_to_float(opt.get('lend_rate_avg')))}"
    )
    return f"- {label}: {rates}"


def _option_line_credit(opt: dict) -> str:
    """개인신용: 금리 구분(대출/기준/가산/가감조정) · 금리방식: 신용점수 구간별 금리"""
    labels = [opt.get("crdt_lend_rate_type_nm"), opt.get("crdt_prdt_type_nm")]
    label = " · ".join(x for x in labels if x) or "금리"
    bands = [
        f"{name} {_pct(_to_float(opt.get(key)))}"
        for key, name in CREDIT_GRADES.items()
        if _to_float(opt.get(key)) is not None
    ]
    avg = _to_float(opt.get("crdt_grad_avg"))
    if avg is not None:
        bands.append(f"평균 {_pct(avg)}")
    return f"- {label}: {', '.join(bands) if bands else '공시 금리 없음'}"


def _summarize_options(loan_type: str, options: list[dict]) -> dict:
    """옵션에서 금리 최저·최고·평균과 선택지 목록을 뽑는다."""
    if loan_type == "credit":
        # 금리 범위는 '대출금리' 행의 신용점수 구간 값으로 계산한다.
        loan_rows = [
            o
            for o in options
            if o.get("crdt_lend_rate_type") == CREDIT_LOAN_RATE_TYPE or o.get("crdt_lend_rate_type_nm") == "대출금리"
        ]
        values = [_to_float(o.get(k)) for o in loan_rows for k in CREDIT_GRADES]
        avgs = [_to_float(o.get("crdt_grad_avg")) for o in loan_rows]
        rate_min, rate_max = _rate_summary(values)
        avgs = [a for a in avgs if a is not None]
        return {
            "rate_min": rate_min,
            "rate_max": rate_max,
            "rate_avg": round(sum(avgs) / len(avgs), 2) if avgs else None,
            "rate_types": sorted({o.get("crdt_lend_rate_type_nm") for o in options if o.get("crdt_lend_rate_type_nm")}),
        }

    rate_min, _ = _rate_summary([_to_float(o.get("lend_rate_min")) for o in options])
    _, rate_max = _rate_summary([_to_float(o.get("lend_rate_max")) for o in options])
    avgs = [a for a in (_to_float(o.get("lend_rate_avg")) for o in options) if a is not None]
    return {
        "rate_min": rate_min,
        "rate_max": rate_max,
        "rate_avg": round(sum(avgs) / len(avgs), 2) if avgs else None,
        "rate_types": sorted({o.get("lend_rate_type_nm") for o in options if o.get("lend_rate_type_nm")}),
        "rpay_types": sorted({o.get("rpay_type_nm") for o in options if o.get("rpay_type_nm")}),
        "mrtg_types": sorted({o.get("mrtg_type_nm") for o in options if o.get("mrtg_type_nm")}),
    }


def build_text(loan_type: str, base: dict, options: list[dict]) -> str:
    """상품 1건을 검색용 텍스트 하나로 만든다."""
    type_nm = LOAN_TYPE_NAMES[loan_type]
    group_nm = GROUP_NAMES.get(base.get("top_fin_grp_no"), "")
    summary = _summarize_options(loan_type, options)

    lines = [
        f"[{type_nm}] {base.get('kor_co_nm', '')} {base.get('fin_prdt_nm', '')}",
        f"금융회사: {base.get('kor_co_nm', '')}" + (f" ({group_nm})" if group_nm else ""),
        f"상품명: {base.get('fin_prdt_nm', '')}",
        f"대출 유형: {type_nm}",
    ]
    if loan_type == "credit":
        if base.get("crdt_prdt_type_nm"):
            lines.append(f"신용대출 구분: {base['crdt_prdt_type_nm']}")
        if base.get("cb_name"):
            lines.append(f"신용평가회사(CB): {base['cb_name']}")
    else:
        if summary.get("mrtg_types"):
            lines.append(f"담보 유형: {', '.join(summary['mrtg_types'])}")
        if summary.get("rpay_types"):
            lines.append(f"상환 방식: {', '.join(summary['rpay_types'])}")
    if summary.get("rate_types"):
        label = "금리 구성" if loan_type == "credit" else "금리 유형"
        lines.append(f"{label}: {', '.join(summary['rate_types'])}")

    lines.append(
        f"금리 요약: 최저 {_pct(summary['rate_min'])} / 최고 {_pct(summary['rate_max'])}"
        f" / 평균 {_pct(summary['rate_avg'])}"
    )
    if options:
        lines.append("금리 옵션:")
        line_fn = _option_line_credit if loan_type == "credit" else _option_line_secured
        lines.extend(line_fn(o) for o in options)
    else:
        lines.append("금리 옵션: 공시된 옵션 없음")

    for key, label in [
        ("loan_lmt", "대출 한도"),
        ("erly_rpay_fee", "중도상환 수수료"),
        ("dly_rate", "연체 이자율"),
        ("loan_inci_expn", "부대비용"),
    ]:
        if base.get(key):
            lines.append(f"{label}: {_flatten(base[key])}")
    if base.get("join_way"):
        lines.append(f"가입 방법: {', '.join(w.strip() for w in str(base['join_way']).split(',') if w.strip())}")

    end = _ymd(base.get("dcls_end_day")) if base.get("dcls_end_day") else "별도 종료일 없음"
    lines.append(f"공시 기간: {_ymd(base.get('dcls_strt_day'))} ~ {end} (공시월 {base.get('dcls_month', '-')})")
    return "\n".join(lines)


def build_metadata(loan_type: str, base: dict, options: list[dict], product_id: str, collected_at: str) -> dict:
    """Chroma 메타데이터. 조건 검색(F2010)에 쓸 금리 범위를 숫자로 넣고, None 값은 뺀다."""
    summary = _summarize_options(loan_type, options)
    meta = {
        "product_id": product_id,
        "loan_type": loan_type,
        "loan_type_nm": LOAN_TYPE_NAMES[loan_type],
        "fin_co_no": base.get("fin_co_no"),
        "fin_prdt_cd": base.get("fin_prdt_cd"),
        "kor_co_nm": base.get("kor_co_nm"),
        "fin_prdt_nm": base.get("fin_prdt_nm"),
        "top_fin_grp_no": base.get("top_fin_grp_no"),
        "fin_grp_nm": GROUP_NAMES.get(base.get("top_fin_grp_no")),
        "crdt_prdt_type_nm": base.get("crdt_prdt_type_nm"),
        "rate_min": summary["rate_min"],
        "rate_max": summary["rate_max"],
        "rate_avg": summary["rate_avg"],
        "rate_types": ",".join(summary.get("rate_types", [])) or None,
        "rpay_types": ",".join(summary.get("rpay_types", [])) or None,
        "mrtg_types": ",".join(summary.get("mrtg_types", [])) or None,
        "option_count": len(options),
        "dcls_month": base.get("dcls_month"),
        "dcls_strt_day": base.get("dcls_strt_day"),
        "dcls_end_day": base.get("dcls_end_day"),
        "collected_at": collected_at,
        "title": f"{base.get('kor_co_nm', '')} {base.get('fin_prdt_nm', '')}".strip(),
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
    }
    return {k: v for k, v in meta.items() if v not in (None, "")}


def _base_key(loan_type: str, row: dict) -> tuple:
    key = (row.get("fin_co_no"), row.get("fin_prdt_cd"))
    return key + (row.get("crdt_prdt_type"),) if loan_type == "credit" else key


def load_products(loan_type: str, path: Path) -> tuple[list[LoanProduct], dict]:
    """한 유형의 JSON을 읽어 상품 목록과 적재 통계를 돌려준다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    collected_at = data.get("collected_at", "")
    base_rows = data.get("baseList", [])
    option_rows = data.get("optionList", [])

    # 같은 상품이 여러 번 나오면 가장 최근 공시월(같으면 먼저 나온 행)만 남긴다.
    bases: dict[tuple, dict] = {}
    for row in base_rows:
        key = _base_key(loan_type, row)
        if key not in bases or str(row.get("dcls_month", "")) > str(bases[key].get("dcls_month", "")):
            bases[key] = row

    options: dict[tuple, list[dict]] = {}
    seen_opts = set()
    for row in option_rows:
        dedup = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if dedup in seen_opts:
            continue
        seen_opts.add(dedup)
        options.setdefault(_base_key(loan_type, row), []).append(row)

    # 옵션은 (상품 키, 공시 제출월)로 연결한다. 남긴 baseList 행과 공시월이 다른 옵션은 버린다. (API 명세)
    merged = {
        key: [o for o in options.get(key, []) if o.get("dcls_month") in (None, "", base.get("dcls_month"))]
        for key, base in bases.items()
    }

    # 신용대출: 같은 (회사, 상품코드)에 구분이 여러 개일 때만 ID에 구분 코드를 붙인다.
    pair_counts: dict[tuple, int] = {}
    for key in bases:
        pair_counts[key[:2]] = pair_counts.get(key[:2], 0) + 1

    products = []
    for key, base in bases.items():
        product_id = f"{loan_type}:{key[0]}:{key[1]}"
        if loan_type == "credit" and pair_counts[key[:2]] > 1:
            product_id += f":{key[2]}"
        opts = merged[key]
        products.append(
            LoanProduct(
                id=product_id,
                text=build_text(loan_type, base, opts),
                metadata=build_metadata(loan_type, base, opts, product_id, collected_at),
                options=opts,
            )
        )

    matched = sum(len(v) for v in merged.values())
    stats = {
        "base_rows": len(base_rows),
        "products": len(products),
        "duplicates": len(base_rows) - len(bases),
        "option_rows": len(option_rows),
        "options_matched": matched,
        "options_orphan": len(seen_opts) - matched,
        "without_options": sum(1 for v in merged.values() if not v),
    }
    return products, stats


def load_all(raw_dir: Path, loan_types: list[str]) -> tuple[list[LoanProduct], dict]:
    """여러 유형을 한 번에 적재한다. 파일이 없는 유형은 건너뛰고 통계에 표시한다."""
    products, stats = [], {}
    for loan_type in loan_types:
        if loan_type not in LOAN_TYPE_NAMES:
            raise ValueError(f"대출 유형이 아닙니다: {loan_type}")
        path = Path(raw_dir) / "fss" / f"{loan_type}.json"
        if not path.exists():
            stats[loan_type] = {"missing": str(path)}
            continue
        items, stats[loan_type] = load_products(loan_type, path)
        products.extend(items)
    return products, stats
