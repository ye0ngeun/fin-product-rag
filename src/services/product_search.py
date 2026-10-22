"""자연어 조건으로 대출상품을 찾는다. (F2011)

흐름: 프롬프트4 조건 해석 -> 코드가 메타데이터로 숫자·범주 조건을 거름 -> 남은 후보를 유사도 순으로 최대
search_k 개 -> 프롬프트5가 나머지 조건(query)에 맞는 것만 max_items 개까지 고름.
- 금리·유형·상환방식처럼 메타데이터로 확인할 수 있는 조건은 코드가 판정한다. 그래서 조건과 맞지 않는 상품은
  LLM 판단과 상관없이 결과에 섞이지 않는다.
- 결과가 0건이면 어느 조건에서 후보가 사라졌는지 알려준다.
"""

from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from src.data_prep.fss_loader import LOAN_TYPE_NAMES
from src.rag.llm import parse_json_list, parse_json_object
from src.rag.prompts import CONDITION_PROMPT, PRODUCT_SELECTION_PROMPT

CONDITION_KEYS = ("loan_type", "rate_max", "rate_type", "rpay_type", "fin_grp", "company", "query")
FIN_GROUPS = ("은행", "저축은행", "보험", "여신전문")


@dataclass
class SearchResult:
    conditions: dict
    cards: list[dict] = field(default_factory=list)  # [{product_id, title, metadata, matched, reason}]
    funnel: list[tuple[str, int]] = field(default_factory=list)  # [(조건 설명, 남은 후보 수)]
    message: str = ""


def parse_conditions(llm: BaseChatModel, condition: str) -> dict:
    """프롬프트4. 알 수 없는 값은 None 으로 정리한다."""
    raw = parse_json_object(llm.invoke(CONDITION_PROMPT.format_messages(condition=condition)).content)
    cond = {k: raw.get(k) or None for k in CONDITION_KEYS}
    if cond["loan_type"] not in LOAN_TYPE_NAMES:
        cond["loan_type"] = None
    try:
        cond["rate_max"] = float(cond["rate_max"]) if cond["rate_max"] is not None else None
    except (TypeError, ValueError):
        cond["rate_max"] = None
    if cond["rate_type"] not in ("고정", "변동"):
        cond["rate_type"] = None
    if cond["rpay_type"] not in ("분할", "만기일시"):
        cond["rpay_type"] = None
    if cond["fin_grp"] not in FIN_GROUPS:
        cond["fin_grp"] = None
    return cond


def _filters(cond: dict) -> list[tuple[str, callable]]:
    """(설명, 메타데이터 판정 함수). 정보가 없는 상품은 그 조건에 맞지 않는 것으로 본다."""
    out = []
    if cond.get("loan_type"):
        out.append((LOAN_TYPE_NAMES[cond["loan_type"]], lambda m: m.get("loan_type") == cond["loan_type"]))
    if cond.get("rate_max") is not None:
        out.append(
            (f"최저 금리 {cond['rate_max']:g}% 이하", lambda m: m.get("rate_min") is not None and m["rate_min"] <= cond["rate_max"])
        )
    if cond.get("rate_type"):
        rate_word = cond["rate_type"] + "금리"
        out.append((rate_word, lambda m: rate_word in (m.get("rate_types") or "").split(",")))
    if cond.get("rpay_type"):
        rpay_word = cond["rpay_type"] + "상환방식"
        out.append((f"{cond['rpay_type']}상환", lambda m: rpay_word in (m.get("rpay_types") or "")))
    if cond.get("fin_grp"):
        out.append((cond["fin_grp"], lambda m: m.get("fin_grp_nm") == cond["fin_grp"]))
    if cond.get("company"):
        name = cond["company"].replace(" ", "")
        out.append((cond["company"], lambda m: name in (m.get("kor_co_nm") or "").replace(" ", "")))
    return out


def matched_labels(meta: dict, cond: dict) -> list[str]:
    """카드에 보일 '맞은 조건'. 코드가 메타데이터 값으로 만든다."""
    labels = []
    if cond.get("rate_max") is not None and meta.get("rate_min") is not None:
        labels.append(f"최저 {meta['rate_min']:.2f}% ≤ {cond['rate_max']:g}%")
    labels += [desc for desc, _ in _filters(cond) if not desc.startswith("최저 금리")]
    return labels


def search_products(
    condition: str,
    meta_store: Chroma,
    llm: BaseChatModel,
    search_k: int = 20,
    max_items: int = 5,
) -> SearchResult:
    cond = parse_conditions(llm, condition)
    got = meta_store.get(include=["metadatas", "documents"])
    pool = {pid: (meta, text) for pid, meta, text in zip(got["ids"], got["metadatas"], got["documents"])}
    result = SearchResult(conditions=cond, funnel=[("전체 상품", len(pool))])

    for desc, ok in _filters(cond):
        pool = {pid: v for pid, v in pool.items() if ok(v[0])}
        result.funnel.append((desc, len(pool)))
        if not pool:
            result.message = f"조건에 맞는 상품이 없습니다. '{desc}' 조건에서 후보가 0건이 되었어요. 이 조건을 넓혀 보세요."
            return result

    # 조건 밖의 말(query)이 있으면 그 말과 비슷한 순, 없으면 최저 금리 낮은 순
    ids = list(pool)
    if cond.get("query"):
        ranked = meta_store.similarity_search(cond["query"], k=min(search_k, len(ids)), filter={"product_id": {"$in": ids}})
        ids = [d.metadata["product_id"] for d in ranked]
    else:
        ids.sort(key=lambda pid: pool[pid][0].get("rate_min") if pool[pid][0].get("rate_min") is not None else 99)
    candidates = ids[:search_k]

    if cond.get("query"):
        listing = "\n\n".join(f"{i}. {pool[pid][1]}" for i, pid in enumerate(candidates))
        reply = llm.invoke(
            PRODUCT_SELECTION_PROMPT.format_messages(condition=condition, candidates=listing, max_items=max_items)
        )
        picked = []
        for item in parse_json_list(reply.content):
            ok = isinstance(item, dict) and isinstance(item.get("no"), int) and 0 <= item["no"] < len(candidates)
            if ok and item["no"] not in [p for p, _ in picked]:
                picked.append((item["no"], str(item.get("reason", ""))))
        chosen = [(candidates[i], reason) for i, reason in picked[:max_items]]
        result.funnel.append((f"'{cond['query']}' (모델 판단)", len(chosen)))
    else:
        chosen = [(pid, "") for pid in candidates[:max_items]]

    for pid, reason in chosen:
        meta = pool[pid][0]
        result.cards.append(
            {
                "product_id": pid,
                "title": meta.get("title", pid),
                "metadata": meta,
                "matched": matched_labels(meta, cond),
                "reason": reason,
            }
        )
    if not result.cards:
        result.message = f"조건에 맞는 상품이 없습니다. '{cond.get('query')}' 조건을 빼거나 바꿔 보세요."
    return result
