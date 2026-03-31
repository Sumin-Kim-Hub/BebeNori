import re

from data_loader import DISTRICTS
from typing import Optional


DOC_LOOKUP_KEYWORDS = {
    "parking_info": ("주차", "주차장", "차 세울", "차댈"),
    "reservation_url": ("예약", "예약링크", "링크", "예매"),
    "address": ("주소", "위치", "어디에", "어디야", "몇 층", "몇층"),
    "price_info_text": ("가격", "요금", "비용", "입장료", "얼마"),
    "operating_hours_text": ("운영시간", "영업시간", "몇 시", "몇시", "오픈", "마감"),
}

FOLLOWUP_MARKERS = (
    "그럼",
    "그러면",
    "그곳",
    "거기",
    "거긴",
    "여긴",
    "여기",
    "아까",
    "방금",
    "첫번째",
    "첫 번째",
    "두번째",
    "두 번째",
    "세번째",
    "세 번째",
    "이 곳",
    "이곳",
)

DETAIL_KEYWORDS = (
    "왜 추천",
    "어때",
    "어떤데",
    "분위기",
    "특징",
    "장점",
    "단점",
    "깨끗",
    "위생",
    "수유실",
    "장난감",
    "놀이",
    "리뷰",
    "좋아",
)

SEARCH_KEYWORDS = (
    "추천",
    "찾아",
    "갈만",
    "키즈카페",
    "장소",
    "근처",
    "다른 곳",
    "다시",
    "바꿔",
)

REFINE_HINTS = (
    "그럼",
    "그러면",
    "대신",
    "말고",
    "다른",
    "다시",
    "바꾸",
)


def resolve_followup(chat_history: list[dict], prompt: str) -> dict:
    prompt = (prompt or "").strip()
    last_turn_state = _get_last_turn_state(chat_history)
    last_search_state = _get_last_search_state(chat_history)
    last_assistant = _get_last_assistant_with_docs(chat_history)
    last_user_query = last_search_state.get("standalone_query") or _get_last_search_query(chat_history)
    source_docs = last_assistant.get("source_docs", []) if last_assistant else []
    explicit_target_rank = _extract_target_doc_rank(prompt)
    target_doc_rank = (
        explicit_target_rank
        if explicit_target_rank is not None
        else last_turn_state.get("active_place_rank", 0)
    )
    target_doc = _pick_doc(
        source_docs,
        target_doc_rank,
        active_place_id=last_turn_state.get("active_place_id"),
        prefer_active=explicit_target_rank is None,
    )
    lookup_field = _detect_doc_lookup_field(prompt)
    has_followup_marker = _has_followup_marker(prompt)
    likely_followup = _is_likely_followup(
        prompt,
        lookup_field=lookup_field,
        has_followup_marker=has_followup_marker,
    )
    explicit_search = _is_explicit_search(prompt)

    resolution = {
        "intent": "fresh_search",
        "standalone_query": prompt,
        "target_doc_rank": target_doc_rank,
        "lookup_field": None,
        "source_docs": source_docs,
        "target_doc": None,
        "last_answer": last_assistant.get("content", "") if last_assistant else "",
        "active_place_id": last_turn_state.get("active_place_id"),
        "retrieved_pids": last_search_state.get("retrieved_pids", []),
        "search_slots": _extract_search_slots(prompt),
    }

    if not source_docs:
        return resolution

    if lookup_field and likely_followup:
        resolution["intent"] = "doc_lookup"
        resolution["lookup_field"] = lookup_field
        resolution["target_doc"] = target_doc
        return resolution

    if _is_place_detail(prompt) and (likely_followup or target_doc_rank > 0):
        resolution["intent"] = "place_detail"
        resolution["target_doc"] = target_doc
        if target_doc and target_doc.get("place_id"):
            resolution["active_place_id"] = target_doc.get("place_id")
        return resolution

    if (
        (has_followup_marker and _has_refine_hint(prompt))
        or (_has_refine_hint(prompt) and explicit_search)
    ):
        merged_slots = _merge_search_slots(last_search_state.get("search_slots", {}), prompt)
        resolution["intent"] = "refine_search"
        resolution["search_slots"] = merged_slots
        resolution["standalone_query"] = _merge_queries(
            last_user_query,
            prompt,
            last_search_state.get("search_slots", {}),
        )
        return resolution

    return resolution


def format_doc_lookup_answer(doc: Optional[dict], field: Optional[str]) -> Optional[str]:
    if not doc or not field:
        return None

    place_name = doc.get("place_name", "방금 추천한 장소")

    if field == "parking_info":
        value = _clean_value(doc.get("parking_info"))
        if not value:
            return f"{place_name}의 주차 정보는 데이터에 따로 없어요."
        return f"{place_name} 기준으로 보면, 주차는 이렇게 안내돼 있어요.\n\n{value}"

    if field == "reservation_url":
        value = _clean_value(doc.get("reservation_url") or doc.get("detail_url"))
        if not value:
            return f"{place_name}의 예약 링크는 데이터에 따로 없어요."
        return f"{place_name} 예약은 여기서 확인하면 돼요.\n\n{value}"

    if field == "address":
        value = _clean_value(doc.get("address"))
        if not value:
            return f"{place_name}의 위치 정보는 데이터에 따로 없어요."
        return f"{place_name} 위치는 여기예요.\n\n{value}"

    if field == "price_info_text":
        value = _clean_value(doc.get("price_info_text"))
        if not value:
            return f"{place_name} 이용요금 정보는 데이터에 따로 없어요."
        return f"{place_name} 이용요금은 이렇게 안내돼 있어요.\n\n{value}"

    if field == "operating_hours_text":
        hours = _clean_value(doc.get("operating_hours_text"))
        days = _clean_value(doc.get("operating_day_text"))
        closed = _clean_value(doc.get("closed_day_text"))
        parts = [part for part in (hours, days, closed) if part]
        if not parts:
            return f"{place_name} 운영시간 정보는 데이터에 따로 없어요."
        return f"{place_name} 운영 정보는 이 정도로 확인돼요.\n\n" + "\n".join(parts)

    return None


def _get_last_assistant_with_docs(chat_history: list[dict]) -> Optional[dict]:
    for chat in reversed(chat_history):
        if chat.get("role") == "assistant" and chat.get("source_docs"):
            return chat
    return None


def _get_last_user_query(chat_history: list[dict]) -> str:
    for chat in reversed(chat_history):
        if chat.get("role") == "user":
            return str(chat.get("content", "")).strip()
    return ""


def _get_last_search_query(chat_history: list[dict]) -> str:
    return _get_last_search_state(chat_history).get("standalone_query") or _get_last_user_query(chat_history)


def _get_last_turn_state(chat_history: list[dict]) -> dict:
    for chat in reversed(chat_history):
        if chat.get("role") != "assistant":
            continue
        turn_meta = chat.get("turn_meta") or {}
        if turn_meta:
            return {
                "active_place_id": turn_meta.get("active_place_id"),
                "active_place_rank": int(turn_meta.get("active_place_rank", 0) or 0),
                "search_slots": dict(turn_meta.get("search_slots", {}) or {}),
            }
    return {}


def _get_last_search_state(chat_history: list[dict]) -> dict:
    for chat in reversed(chat_history):
        if chat.get("role") != "assistant":
            continue
        turn_meta = chat.get("turn_meta") or {}
        intent = turn_meta.get("intent")
        standalone_query = str(turn_meta.get("standalone_query", "")).strip()
        if intent in {"fresh_search", "refine_search"} and standalone_query:
            return {
                "standalone_query": standalone_query,
                "active_place_id": turn_meta.get("active_place_id"),
                "active_place_rank": int(turn_meta.get("active_place_rank", 0) or 0),
                "retrieved_pids": list(turn_meta.get("retrieved_pids", []) or []),
                "search_slots": dict(turn_meta.get("search_slots", {}) or {}),
            }
    return {}


def _extract_target_doc_rank(prompt: str) -> Optional[int]:
    normalized = prompt.replace(" ", "")
    if "세번째" in normalized or "3번째" in normalized or "세번" in normalized:
        return 2
    if "두번째" in normalized or "2번째" in normalized or "두번" in normalized:
        return 1
    if "첫번째" in normalized or "1번째" in normalized or "첫 번째" in prompt:
        return 0
    return None


def _pick_doc_by_rank(source_docs: list[dict], rank: int) -> Optional[dict]:
    if not source_docs:
        return None
    if 0 <= rank < len(source_docs):
        return source_docs[rank]
    return source_docs[0]


def _pick_doc(
    source_docs: list[dict],
    rank: int,
    active_place_id: Optional[str] = None,
    prefer_active: bool = False,
) -> Optional[dict]:
    if not source_docs:
        return None
    if prefer_active and active_place_id:
        for doc in source_docs:
            if doc.get("place_id") == active_place_id:
                return doc
    return _pick_doc_by_rank(source_docs, rank)


def _detect_doc_lookup_field(prompt: str) -> Optional[str]:
    normalized = prompt.strip()
    for field, keywords in DOC_LOOKUP_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return field
    return None


def _has_followup_marker(prompt: str) -> bool:
    normalized = prompt.strip()
    return any(marker in normalized for marker in FOLLOWUP_MARKERS)


def _is_likely_followup(
    prompt: str,
    lookup_field: Optional[str] = None,
    has_followup_marker: bool = False,
) -> bool:
    normalized = prompt.strip()
    if has_followup_marker:
        return True
    if len(normalized) <= 18 and (lookup_field or _is_place_detail(normalized)):
        return True
    return False


def _is_explicit_search(prompt: str) -> bool:
    normalized = prompt.strip()
    return any(keyword in normalized for keyword in SEARCH_KEYWORDS)


def _is_place_detail(prompt: str) -> bool:
    normalized = prompt.strip()
    if any(keyword in normalized for keyword in DETAIL_KEYWORDS):
        return True
    if re.search(r"(첫|두|세)\s*번째", normalized):
        return True
    return False


def _has_refine_hint(prompt: str) -> bool:
    normalized = prompt.strip()
    return any(keyword in normalized for keyword in REFINE_HINTS)


def _merge_queries(last_query: str, prompt: str, last_slots: Optional[dict] = None) -> str:
    last_query = last_query.strip()
    prompt = prompt.strip()
    if not last_query:
        return prompt

    last_slots = last_slots or {}
    prior_district = last_slots.get("district") or _extract_district(last_query)
    new_district = _extract_district(prompt)
    prior_age = last_slots.get("age_expr") or _extract_age_expr(last_query)
    new_age = _extract_age_expr(prompt)

    merged = last_query
    if new_district:
        if prior_district:
            merged = re.sub(prior_district, new_district, merged)
            merged = re.sub(prior_district.replace("구", ""), new_district.replace("구", ""), merged)
        elif new_district not in merged:
            merged = f"{new_district} {merged}"

    if new_age:
        if prior_age:
            merged = merged.replace(prior_age, new_age)
        elif new_age not in merged:
            merged = f"{new_age} {merged}"

    cleaned_prompt = _clean_refine_prompt(prompt, district=new_district, age_expr=new_age)
    if cleaned_prompt:
        return f"{merged} 추가 조건: {cleaned_prompt}".strip()
    return merged.strip()


def _clean_value(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _extract_district(text: str) -> str:
    normalized = text.strip()
    for district in DISTRICTS:
        if district in normalized:
            return district
        short = district[:-1] if district.endswith("구") else district
        if short and short in normalized:
            return district
    return ""


def _extract_age_expr(text: str) -> str:
    normalized = text.strip()
    month_match = re.search(r"\d+\s*개월", normalized)
    if month_match:
        return month_match.group(0).replace(" ", "")
    year_match = re.search(r"(?:만\s*)?\d+\s*(?:세|살)", normalized)
    if year_match:
        return re.sub(r"\s+", "", year_match.group(0))
    return ""


def _extract_search_slots(text: str) -> dict:
    return {
        "district": _extract_district(text),
        "age_expr": _extract_age_expr(text),
    }


def _merge_search_slots(last_slots: dict, prompt: str) -> dict:
    merged = dict(last_slots or {})
    prompt_slots = _extract_search_slots(prompt)
    for key, value in prompt_slots.items():
        if value:
            merged[key] = value
    return merged


def _remove_district_tokens(text: str, district: str) -> str:
    cleaned = text
    variants = {district, district.replace("구", ""), f"{district}로", f"{district}으로"}
    short = district.replace("구", "")
    variants.update({f"{short}로", f"{short}으로"})
    for token in sorted((v for v in variants if v), key=len, reverse=True):
        cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _clean_refine_prompt(prompt: str, district: str = "", age_expr: str = "") -> str:
    cleaned = prompt
    if district:
        cleaned = _remove_district_tokens(cleaned, district)
    if age_expr:
        cleaned = cleaned.replace(age_expr, " ")
    for token in ("그럼", "그러면", "대신", "말고", "다른", "다시", "바꾸면", "바꾸고", "바꿔"):
        cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"[?!.]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
