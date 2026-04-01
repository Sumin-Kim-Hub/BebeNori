# 수정일: 26/03/27/11:55 (진국)
# 이유: 필터링 조건에서 컨텍스트를 추출하던 문제를 수정. 유저 쿼리에만 의존하도록
# =============================================================================
# 파일 위치: 02_model_logic/backend_core.py
# 역할: RAG 파이프라인 엔진

import hashlib
import json
import math
import os
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from dotenv import load_dotenv

# 경로 설정 및 data_loader 임포트 (기존 로직 유지)
_MODELS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODELS_DIR.parent
_DATA_PREP = _PROJECT_DIR / "01_data_prep"
if str(_DATA_PREP) not in sys.path:
    sys.path.insert(0, str(_DATA_PREP))

from data_loader import (
    CHROMA_DIR,
    DISTRICTS,
    FEATURES_CSV,
    PLACES_CSV,
    PUBLIC_BOOK,
    REVIEWS_CSV,
    park_short,
)
# ⭐️ 분리한 프롬프트 템플릿 임포트
from prompt_templates import (
    SYSTEM_PROMPT,
    get_main_recommendation_prompt,
    get_followup_answer_prompt,
    get_card_recommendation_prompt,
)

# LangChain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv(_PROJECT_DIR / ".env")

# 상수 설정
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "gpt-4o-mini"
CHROMA_COLLECTION = "bebenori_v5"
INDEX_MANIFEST_NAME = "index_manifest.json"
NEAR_QUERY_TOKENS = ("근처", "주변", "인근", "가까운", "도보")
LOCATION_OVERSAMPLE = 12
LOCATION_HARD_RADIUS_KM = 2.0
REVIEW_SEARCH_K_MULTIPLIER = 10
REVIEW_EVIDENCE_LIMIT = 2
REVIEW_EVIDENCE_CHAR_LIMIT = 140
GENERIC_PLACE_TOKENS = {
    "서울형",
    "서울형키즈카페",
    "키즈카페",
    "키즈",
    "카페",
    "시립",
    "일반형",
    "특화형",
}
KOREAN_AGE_WORDS = {
    "한": 1,
    "하나": 1,
    "두": 2,
    "둘": 2,
    "세": 3,
    "셋": 3,
    "네": 4,
    "넷": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
}

# --- [LLM 체인 (LangChain ChatOpenAI + ChatPromptTemplate + StrOutputParser)] ---

@lru_cache(maxsize=1)
def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

def _get_district_value(row) -> str:
    return str(row.get("district") or row.get("address_gu") or "").strip()


def _safe_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _compact_spaces(value: object) -> str:
    return re.sub(r"\s+", "", _safe_text(value))


def _spaced_term_pattern(term: object) -> str:
    compact = _compact_spaces(term)
    return r"\s*".join(re.escape(char) for char in compact)


def _matches_spaced_phrase(query: str, term: object, whole: bool = False) -> bool:
    compact = _compact_spaces(term)
    if not compact:
        return False
    pattern = _spaced_term_pattern(compact)
    if whole:
        pattern = rf"(?<![가-힣A-Za-z0-9]){pattern}(?![가-힣A-Za-z0-9])"
    return re.search(pattern, _safe_text(query)) is not None


def _find_spaced_phrase_index(query: str, term: object, whole: bool = False) -> int:
    compact = _compact_spaces(term)
    if not compact:
        return -1
    pattern = _spaced_term_pattern(compact)
    if whole:
        pattern = rf"(?<![가-힣A-Za-z0-9]){pattern}(?![가-힣A-Za-z0-9])"
    match = re.search(pattern, _safe_text(query))
    return match.start() if match else -1


def _safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_station_names(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"([가-힣A-Za-z0-9]+역)", _safe_text(text))))


def _extract_place_keywords(name: str) -> list[str]:
    text = _safe_text(name)
    if not text:
        return []

    text = re.sub(r"서울형\s*키즈카페", " ", text)
    text = re.sub(r"서울형키즈카페", " ", text)
    text = re.sub(r"시립", " ", text)
    tokens = re.split(r"[()/,\s]+", text)

    keywords = []
    for token in tokens:
        token = token.strip()
        token = re.sub(r"점$", "", token)
        if len(token) < 2 or token in GENERIC_PLACE_TOKENS:
            continue
        keywords.append(token)
    return list(dict.fromkeys(keywords))


def _build_landmark_aliases(term: str) -> list[str]:
    term = _safe_text(term)
    if not term:
        return []

    aliases = {term}
    stripped = term
    for prefix in ("서울형", "서울형키즈카페", "서울", "시립"):
        if stripped.startswith(prefix) and len(stripped) > len(prefix) + 1:
            stripped = stripped[len(prefix):].strip()
            aliases.add(stripped)
    return sorted(alias for alias in aliases if len(_compact_spaces(alias)) >= 2)


def _strip_parenthetical(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", _safe_text(text)).strip()


def _extract_parenthetical_terms(text: str) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in re.findall(r"\(([^)]*)\)", _safe_text(text)) if part.strip()))


def _extract_place_fragments(name: str) -> list[str]:
    text = _strip_parenthetical(name)
    if not text:
        return []

    text = re.sub(r"서울형\s*키즈카페", " ", text)
    text = re.sub(r"서울형키즈카페", " ", text)
    text = re.sub(r"시립", " ", text)
    tokens = re.split(r"[()/,\s]+", text)

    fragments = []
    for token in tokens:
        token = token.strip()
        if len(_compact_spaces(token)) < 2:
            continue
        if token in GENERIC_PLACE_TOKENS:
            continue
        if token in DISTRICTS or token.endswith("구"):
            continue
        fragments.append(token)
    return list(dict.fromkeys(fragments))


def _build_place_name_match_terms(place_name: str) -> list[tuple[int, str, bool]]:
    candidates = []
    full_name = _safe_text(place_name)
    stripped_name = _strip_parenthetical(full_name)
    for term in (full_name, stripped_name):
        if term:
            candidates.append((0, term, False))

    for term in _extract_parenthetical_terms(full_name):
        candidates.append((1, term, False))

    alias_sources = [full_name, stripped_name, *_extract_parenthetical_terms(full_name)]
    for source in alias_sources:
        for alias in _build_landmark_aliases(source):
            candidates.append((2, alias, True))

    for fragment in _extract_place_fragments(full_name):
        candidates.append((3, fragment, True))

    deduped = []
    seen = set()
    for priority, term, whole in candidates:
        compact = _compact_spaces(term)
        if len(compact) < 2 or compact in seen:
            continue
        seen.add(compact)
        deduped.append((priority, term, whole))
    return deduped


def infer_answer_place(source_docs: list[dict], response_text: str) -> tuple[Optional[str], int]:
    if not source_docs:
        return None, 0

    fallback_doc = source_docs[0] if source_docs else {}
    fallback_pid = _safe_text(fallback_doc.get("place_id", ""))
    answer = _safe_text(response_text)
    if not answer:
        return (fallback_pid or None), 0

    best_match = None
    for idx, doc in enumerate(source_docs):
        pid = _safe_text(doc.get("place_id", ""))
        place_name = _safe_text(doc.get("place_name", ""))
        if not pid or not place_name:
            continue

        for priority, term, whole in _build_place_name_match_terms(place_name):
            position = _find_spaced_phrase_index(answer, term, whole=whole)
            if position < 0:
                continue
            score = (priority, position, -len(_compact_spaces(term)), idx)
            if best_match is None or score < best_match[0]:
                best_match = (score, pid, idx)

    if best_match:
        return best_match[1], best_match[2]

    return (fallback_pid or None), 0


def _extract_address_keywords(address: str) -> list[str]:
    text = _safe_text(address)
    if not text:
        return []

    tokens = re.split(r"[()/,\s]+", text)
    keywords = []
    for token in tokens:
        token = token.strip()
        token = re.sub(r"내$", "", token)
        if len(token) < 2:
            continue
        if re.search(r"\d", token):
            continue
        if token in GENERIC_PLACE_TOKENS:
            continue
        if token in {"서울", "서울시", "서울특별시"}:
            continue
        if token.endswith(("구", "동", "로", "길", "층")):
            continue
        keywords.append(token)
    return list(dict.fromkeys(keywords))


def _build_row_location_keywords(row) -> list[str]:
    keywords = []
    keywords.extend(_extract_place_keywords(row.get("place_name", "")))
    keywords.extend(_extract_address_keywords(row.get("address", "")))

    dong = _safe_text(row.get("address_dong", ""))
    if dong:
        keywords.append(dong)

    keywords.extend(_extract_station_names(row.get("subway_info", "")))

    address = _safe_text(row.get("address", ""))
    if address:
        keywords.append(address)

    return list(dict.fromkeys(k for k in keywords if k))


def _build_location_blob(row) -> str:
    parts = [
        _safe_text(row.get("place_name", "")),
        _safe_text(row.get("address", "")),
        _safe_text(row.get("address_dong", "")),
        _safe_text(row.get("subway_info", "")),
        " ".join(_build_row_location_keywords(row)),
    ]
    return " | ".join(part for part in parts if part)


def _build_location_index(df: Optional[pd.DataFrame]) -> dict:
    index = {
        "districts": [],
        "district_aliases": {},
        "dong_to_district": {},
        "station_aliases": {},
        "station_to_place_ids": {},
        "landmark_aliases": {},
        "landmark_to_place_ids": {},
        "place_coords": {},
    }
    if df is None or df.empty:
        return index

    districts = []
    ambiguous_dongs = set()
    for _, row in df.iterrows():
        place_id = _safe_text(row.get("place_id", ""))
        district = _get_district_value(row)
        dong = _safe_text(row.get("address_dong", ""))
        lat = _safe_float(row.get("latitude", row.get("lat")))
        lng = _safe_float(row.get("longitude", row.get("lng")))

        if district:
            districts.append(district)
            if district.endswith("구") and len(district) > 2:
                index["district_aliases"].setdefault(district[:-1], district)

        if dong and district:
            prev = index["dong_to_district"].get(dong)
            if prev and prev != district:
                ambiguous_dongs.add(dong)
            else:
                index["dong_to_district"][dong] = district

        if place_id and lat is not None and lng is not None:
            index["place_coords"][place_id] = (lat, lng)

        for station in _extract_station_names(row.get("subway_info", "")):
            index["station_aliases"].setdefault(station, set()).add(station)
            bare = station[:-1]
            if len(bare) >= 2:
                index["station_aliases"].setdefault(bare, set()).add(station)
            if place_id:
                index["station_to_place_ids"].setdefault(station, set()).add(place_id)

        for keyword in _extract_place_keywords(row.get("place_name", "")):
            if place_id:
                index["landmark_to_place_ids"].setdefault(keyword, set()).add(place_id)
                for alias in _build_landmark_aliases(keyword):
                    index["landmark_aliases"].setdefault(alias, set()).add(keyword)
        for keyword in _extract_address_keywords(row.get("address", "")):
            if place_id:
                index["landmark_to_place_ids"].setdefault(keyword, set()).add(place_id)
                for alias in _build_landmark_aliases(keyword):
                    index["landmark_aliases"].setdefault(alias, set()).add(keyword)

    for dong in ambiguous_dongs:
        index["dong_to_district"].pop(dong, None)

    index["districts"] = sorted(set(districts), key=len, reverse=True)
    index["district_aliases"] = {
        key: value
        for key, value in sorted(index["district_aliases"].items(), key=lambda item: len(item[0]), reverse=True)
    }
    index["dong_to_district"] = {
        key: value
        for key, value in sorted(index["dong_to_district"].items(), key=lambda item: len(item[0]), reverse=True)
    }
    index["station_aliases"] = {
        key: sorted(value, key=len, reverse=True)
        for key, value in sorted(index["station_aliases"].items(), key=lambda item: len(item[0]), reverse=True)
    }
    index["station_to_place_ids"] = {
        key: sorted(value)
        for key, value in index["station_to_place_ids"].items()
    }
    index["landmark_to_place_ids"] = {
        key: sorted(value)
        for key, value in sorted(index["landmark_to_place_ids"].items(), key=lambda item: len(item[0]), reverse=True)
    }
    index["landmark_aliases"] = {
        key: sorted(value, key=len, reverse=True)
        for key, value in sorted(index["landmark_aliases"].items(), key=lambda item: len(item[0]), reverse=True)
    }
    return index


def _extract_location_hints(query: str, df: Optional[pd.DataFrame]) -> dict:
    hints = {
        "district": "",
        "dong_terms": [],
        "station_terms": [],
        "landmark_terms": [],
        "anchor_place_ids": [],
        "near_request": any(token in query for token in NEAR_QUERY_TOKENS),
    }
    if not query:
        return hints

    index = _build_location_index(df)
    used_spans = []

    for district in index["districts"]:
        if district and _matches_spaced_phrase(query, district):
            hints["district"] = district
            break

    if not hints["district"]:
        for alias, district in index["district_aliases"].items():
            if alias and _compact_spaces(alias) in _compact_spaces(query):
                hints["district"] = district
                break

    for dong, district in index["dong_to_district"].items():
        if _matches_spaced_phrase(query, dong, whole=True):
            if any(
                _compact_spaces(dong) in _compact_spaces(existing)
                or _compact_spaces(existing) in _compact_spaces(dong)
                for existing in hints["dong_terms"]
            ):
                continue
            hints["dong_terms"].append(dong)
            used_spans.append(dong)
            if not hints["district"]:
                hints["district"] = district

    station_terms = []
    anchor_place_ids = set()
    for alias, canonicals in index["station_aliases"].items():
        if not alias:
            continue
        if alias.endswith("역"):
            matched = _matches_spaced_phrase(query, alias, whole=True)
        else:
            matched = re.search(
                rf"{_spaced_term_pattern(alias)}\s*(역|근처|주변|인근|가까운|도보)",
                _safe_text(query),
            ) is not None
        if not matched:
            continue
        station_terms.extend(canonicals)
    for station in dict.fromkeys(station_terms):
        hints["station_terms"].append(station)
        anchor_place_ids.update(index["station_to_place_ids"].get(station, []))

    for alias, canonical_keywords in index["landmark_aliases"].items():
        if not alias:
            continue
        if alias in GENERIC_PLACE_TOKENS:
            continue
        if not _matches_spaced_phrase(query, alias, whole=True):
            continue
        for keyword in canonical_keywords:
            if keyword in used_spans:
                continue
            if any(
                _compact_spaces(keyword) in _compact_spaces(existing)
                or _compact_spaces(existing) in _compact_spaces(keyword)
                for existing in hints["landmark_terms"]
            ):
                continue
            hints["landmark_terms"].append(keyword)
            anchor_place_ids.update(index["landmark_to_place_ids"].get(keyword, []))

    hints["anchor_place_ids"] = sorted(anchor_place_ids)
    return hints


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def _rank_location_candidates(df: Optional[pd.DataFrame], hints: dict) -> list[str]:
    if df is None or df.empty:
        return []
    if not (
        hints["dong_terms"]
        or hints["station_terms"]
        or hints["landmark_terms"]
        or hints["near_request"]
    ):
        return []

    location_index = _build_location_index(df)
    place_coords = location_index.get("place_coords", {})
    anchor_coords = [
        place_coords[pid]
        for pid in hints["anchor_place_ids"]
        if pid in place_coords
    ]

    scored = []
    for _, row in df.iterrows():
        pid = _safe_text(row.get("place_id", ""))
        if not pid:
            continue

        district = _get_district_value(row)
        if hints["district"] and district != hints["district"]:
            continue

        score = 0.0
        dong = _safe_text(row.get("address_dong", ""))
        subway_info = _safe_text(row.get("subway_info", ""))
        location_blob = _build_location_blob(row)

        if hints["district"] and district == hints["district"]:
            score += 2.0
        if hints["dong_terms"] and dong in hints["dong_terms"]:
            score += 8.0
        if hints["station_terms"] and any(term in subway_info for term in hints["station_terms"]):
            score += 9.0
        if hints["landmark_terms"] and any(term in location_blob for term in hints["landmark_terms"]):
            score += 8.0

        if hints["near_request"] and anchor_coords:
            lat = _safe_float(row.get("latitude", row.get("lat")))
            lng = _safe_float(row.get("longitude", row.get("lng")))
            if lat is not None and lng is not None:
                min_distance = min(
                    _haversine_km(lat, lng, anchor_lat, anchor_lng)
                    for anchor_lat, anchor_lng in anchor_coords
                )
                if min_distance <= 0.5:
                    score += 5.0
                elif min_distance <= 1.0:
                    score += 3.5
                elif min_distance <= LOCATION_HARD_RADIUS_KM:
                    score += 2.0

        if score > 0:
            scored.append((score, pid))

    return [
        pid
        for _, pid in sorted(scored, key=lambda item: item[0], reverse=True)
    ]


def _score_doc(doc, rank: int, hints: dict, place_coords: dict) -> float:
    score = float(max(LOCATION_OVERSAMPLE - rank, 1))
    metadata = doc.metadata or {}

    district = _safe_text(metadata.get("district", ""))
    dong = _safe_text(metadata.get("dong", ""))
    subway_info = _safe_text(metadata.get("subway_info", ""))
    location_blob = _safe_text(metadata.get("location_blob", ""))

    if hints["district"] and district == hints["district"]:
        score += 4.0
    if hints["dong_terms"] and dong in hints["dong_terms"]:
        score += 3.0
    if any(term in subway_info for term in hints["station_terms"]):
        score += 5.0
    if any(term in location_blob for term in hints["landmark_terms"]):
        score += 4.0

    if hints["near_request"] and hints["anchor_place_ids"]:
        lat = _safe_float(metadata.get("latitude"))
        lng = _safe_float(metadata.get("longitude"))
        if lat is not None and lng is not None:
            anchor_coords = [
                place_coords[pid]
                for pid in hints["anchor_place_ids"]
                if pid in place_coords
            ]
            if anchor_coords:
                min_distance = min(
                    _haversine_km(lat, lng, anchor_lat, anchor_lng)
                    for anchor_lat, anchor_lng in anchor_coords
                )
                if min_distance <= 0.5:
                    score += 4.0
                elif min_distance <= 1.0:
                    score += 3.0
                elif min_distance <= LOCATION_HARD_RADIUS_KM:
                    score += 1.5

    return score


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_index_manifest() -> dict:
    tracked_files = {
        "places_csv": Path(PLACES_CSV),
        "features_csv": Path(FEATURES_CSV),
        "reviews_csv": Path(REVIEWS_CSV),
        "data_loader_py": _DATA_PREP / "data_loader.py",
        "backend_core_py": _MODELS_DIR / "backend_core.py",
    }

    file_hashes = {
        name: {
            "path": str(path.resolve()),
            "sha256": _hash_file(path),
        }
        for name, path in tracked_files.items()
    }

    manifest = {
        "collection": CHROMA_COLLECTION,
        "embedding_model": EMBED_MODEL_NAME,
        "tracked_files": file_hashes,
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return manifest


def _load_index_manifest(persist_path: Path) -> Optional[dict]:
    manifest_path = persist_path / INDEX_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_index_manifest(persist_path: Path, manifest: dict) -> None:
    persist_path.mkdir(parents=True, exist_ok=True)
    manifest_path = persist_path / INDEX_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _extract_district_from_query(query: str) -> str:
    for district in DISTRICTS:
        if district != "전체" and district in query:
            return district
    return ""


def _parse_korean_age_word(token: str) -> Optional[int]:
    return KOREAN_AGE_WORDS.get(_safe_text(token))


def extract_age_months_from_text(query: str) -> Optional[int]:
    text = _safe_text(query)
    if not text:
        return None

    matches = []

    for match in re.finditer(r"(만\s*)?(\d+)\s*개월", text):
        matches.append((match.start(), int(match.group(2))))

    for match in re.finditer(r"(만\s*)?(\d+)\s*(?:세|살)", text):
        years = int(match.group(2))
        months = years * 12 if match.group(1) else max(years * 12 - 1, 0)
        matches.append((match.start(), months))

    for match in re.finditer(r"(만\s*)?(\d+)\s*돌", text):
        years = int(match.group(2))
        matches.append((match.start(), years * 12))

    for match in re.finditer(
        r"(만\s*)?(한|하나|두|둘|세|셋|네|넷|다섯|여섯|일곱|여덟|아홉|열)\s*돌",
        text,
    ):
        years = _parse_korean_age_word(match.group(2))
        if years is not None:
            matches.append((match.start(), years * 12))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def _get_dev_row_for_age(dev_df: Optional[pd.DataFrame], age_months: Optional[int]) -> Optional[pd.Series]:
    if dev_df is None or dev_df.empty or age_months is None:
        return None

    min_months = pd.to_numeric(dev_df.get("min_month"), errors="coerce")
    max_months = pd.to_numeric(dev_df.get("max_month"), errors="coerce")
    matched = dev_df[(min_months <= age_months) & (max_months >= age_months)]
    if matched.empty:
        return None

    return matched.sort_values(["min_month", "max_month"]).iloc[0]


def _split_context_terms(value: object) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []

    parts = re.split(r"[|,]", text)
    terms = []
    for part in parts:
        term = part.strip()
        if not term:
            continue
        terms.append(term)
    return terms


def _expand_facility_aliases(term: str) -> list[str]:
    aliases = []
    raw = _safe_text(term)
    if not raw:
        return aliases

    candidates = {raw}
    stripped = _strip_parenthetical(raw)
    if stripped:
        candidates.add(stripped)

    for inner in _extract_parenthetical_terms(raw):
        candidates.add(inner)
        candidates.update(part.strip() for part in re.split(r"[/,]", inner) if part.strip())

    expanded = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        expanded.add(candidate)
        if candidate.endswith("존") and len(candidate) > 2:
            expanded.add(candidate[:-1])
        if candidate.endswith("세트") and len(candidate) > 3:
            expanded.add(candidate[:-2].strip())
        parts = [part.strip() for part in re.split(r"[\s/]+", candidate) if part.strip()]
        if len(parts) >= 2:
            expanded.add(parts[0])
            expanded.add(parts[-1])

    for candidate in expanded:
        compact = _compact_spaces(candidate)
        if len(compact) < 2:
            continue
        aliases.append(candidate)

    return list(dict.fromkeys(aliases))


def _match_dev_facilities_for_place(row, dev_row: Optional[pd.Series]) -> list[str]:
    if dev_row is None:
        return []

    place_text = " ".join(
        part
        for part in (
            _safe_text(row.get("official_play_tags", "")),
            _safe_text(row.get("official_summary", "")),
            _safe_text(row.get("review_text", "")),
            _safe_text(row.get("place_name", "")),
        )
        if part
    )
    if not place_text:
        return []

    matched_terms = []
    for facility in _split_context_terms(dev_row.get("matching_facilities", "")):
        if any(_matches_spaced_phrase(place_text, alias) for alias in _expand_facility_aliases(facility)):
            matched_terms.append(facility)

    return list(dict.fromkeys(matched_terms))


@lru_cache(maxsize=1)
def _load_review_chunks() -> pd.DataFrame:
    try:
        review_df = pd.read_csv(REVIEWS_CSV, encoding="utf-8-sig")
    except UnicodeDecodeError:
        review_df = pd.read_csv(REVIEWS_CSV, encoding="cp949")
    except Exception:
        return pd.DataFrame()

    review_df = review_df.copy()
    if "chunk_text_for_embedding" not in review_df.columns and "chunk_text" in review_df.columns:
        review_df["chunk_text_for_embedding"] = review_df["chunk_text"]
    if "chunk_text" not in review_df.columns and "content" in review_df.columns:
        review_df["chunk_text"] = review_df["content"]

    if "place_id" in review_df.columns:
        review_df["place_id"] = review_df["place_id"].fillna("").astype(str).str.strip().str.upper()

    text_columns = [
        "place_name",
        "review_title",
        "chunk_text",
        "chunk_text_for_embedding",
        "section_type",
        "play_tags",
        "need_tags",
        "issue_tags",
        "source_url",
    ]
    for column in text_columns:
        if column not in review_df.columns:
            review_df[column] = ""
        review_df[column] = review_df[column].fillna("").astype(str)

    if "chunk_order" in review_df.columns:
        review_df["chunk_order"] = pd.to_numeric(review_df["chunk_order"], errors="coerce").fillna(0)

    return review_df


def _compact_preview_text(value: object, limit: int = REVIEW_EVIDENCE_CHAR_LIMIT) -> str:
    text = re.sub(r"\s+", " ", _safe_text(value)).strip()
    if not text:
        return ""
    return text[:limit].rstrip() + ("…" if len(text) > limit else "")


def _serialize_feature_text(row) -> str:
    play_tags = ", ".join(_split_context_terms(row.get("official_play_tags", ""))[:5])
    if play_tags:
        return play_tags
    features = row.get("features", [])
    if isinstance(features, list):
        return ", ".join(features[:5])
    return _safe_text(features)


def _build_place_vector_content(row) -> str:
    district = _get_district_value(row)
    dong = _safe_text(row.get("address_dong", ""))
    subway_info = _safe_text(row.get("subway_info", ""))
    location_keywords = ", ".join(_build_row_location_keywords(row))
    feature_text = _serialize_feature_text(row) or "정보 없음"
    summary = _safe_text(row.get("official_summary", ""))
    age_text = _safe_text(row.get("age_text", ""))
    return (
        f"장소명: {row.get('place_name')}\n"
        f"주소: {row.get('address')}\n"
        f"지역: {district}\n"
        f"동: {dong}\n"
        f"지하철: {subway_info}\n"
        f"위치키워드: {location_keywords}\n"
        f"이용연령: {age_text}\n"
        f"놀이/특징: {feature_text}\n"
        f"공식요약: {summary}"
    )


def _build_review_chunk_vector_content(chunk_row, place_row) -> str:
    district = _get_district_value(place_row)
    dong = _safe_text(place_row.get("address_dong", ""))
    location_keywords = ", ".join(_build_row_location_keywords(place_row))
    tags = []
    for label, value in (
        ("놀이 태그", chunk_row.get("play_tags", "")),
        ("이용 맥락", chunk_row.get("need_tags", "")),
        ("주의 태그", chunk_row.get("issue_tags", "")),
    ):
        cleaned = _safe_text(value).replace("|", ", ")
        if cleaned:
            tags.append(f"{label}: {cleaned}")

    tag_block = "\n".join(tags)
    if tag_block:
        tag_block = f"{tag_block}\n"

    return (
        f"장소명: {chunk_row.get('place_name') or place_row.get('place_name')}\n"
        f"주소: {place_row.get('address')}\n"
        f"지역: {district}\n"
        f"동: {dong}\n"
        f"위치키워드: {location_keywords}\n"
        f"리뷰 제목: {chunk_row.get('review_title', '')}\n"
        f"{tag_block}"
        f"리뷰 근거: {chunk_row.get('chunk_text_for_embedding') or chunk_row.get('chunk_text')}"
    )


def _build_evidence_entry(doc: Document, score: float) -> dict:
    metadata = doc.metadata or {}
    return {
        "chunk_id": _safe_text(metadata.get("chunk_id", "")),
        "text": _compact_preview_text(metadata.get("chunk_text", "")),
        "title": _safe_text(metadata.get("review_title", "")),
        "section_type": _safe_text(metadata.get("section_type", "")),
        "play_tags": _safe_text(metadata.get("play_tags", "")),
        "score": float(score),
    }


def _finalize_evidence_entries(entries: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for entry in sorted(entries, key=lambda item: item.get("score", 0.0), reverse=True):
        key = (entry.get("chunk_id"), entry.get("text"))
        if key in seen or not entry.get("text"):
            continue
        seen.add(key)
        deduped.append(entry)
        if len(deduped) >= REVIEW_EVIDENCE_LIMIT:
            break
    return deduped

def get_llm_chain():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.72, api_key=api_key or None)
    
    # ⭐️ SYSTEM_PROMPT 사용
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{user_message}"),
    ])
    return prompt | llm | StrOutputParser()

# --- [RAG 검색 및 컨텍스트 빌드 로직] ---

def gen_answer(chain, query: str, ctx: str) -> str:
    # 🔴 context가 비어있으면 LLM 호출 금지
    if not ctx.strip():
        return "조건에 맞는 장소를 찾지 못했어요. 다른 조건으로 다시 시도해 주세요."

    user_msg = get_main_recommendation_prompt(query, ctx)
    return _safe_invoke(chain, user_msg)


def gen_followup_answer(
    chain,
    query: str,
    ctx: str,
    last_answer: str = "",
) -> str:
    if not ctx.strip():
        return "추가로 안내할 만한 정보가 지금 데이터에는 없어요."

    user_msg = get_followup_answer_prompt(query, ctx, last_answer=last_answer)
    return _safe_invoke(chain, user_msg)

def gen_card_rec(chain, name: str, addr: str, feats: list, review: str, age_sel: str = "") -> str:
    # ⭐️ get_card_recommendation_prompt 사용
    user_msg = get_card_recommendation_prompt(name, addr, feats, review, age_sel)
    return _safe_invoke(chain, user_msg)

# --- [벡터 저장소 빌드 로직] ---

def load_or_create_vectorstore(df, force_rebuild=False):
    persist_dir = str(_PROJECT_DIR / CHROMA_DIR)
    persist_path = Path(persist_dir)
    embeddings = _get_embeddings()
    current_manifest = _build_index_manifest()

    rebuild_reason = None
    if force_rebuild:
        rebuild_reason = "force_rebuild=True"
    elif not persist_path.exists():
        rebuild_reason = "chroma_db 없음"
    else:
        saved_manifest = _load_index_manifest(persist_path)
        if saved_manifest is None:
            rebuild_reason = "index manifest 없음 또는 손상"
        elif saved_manifest.get("fingerprint") != current_manifest.get("fingerprint"):
            rebuild_reason = "데이터/코드 변경 감지"

    if rebuild_reason is None:
        print("📂 기존 DB 로드")
        return Chroma(
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION,
            persist_directory=persist_dir
        )

    if persist_path.exists():
        if persist_path.is_dir():
            shutil.rmtree(persist_path)
        else:
            persist_path.unlink()

    print(f"📦 DB 생성 중... ({rebuild_reason})")

    docs = []
    place_rows = {}

    for _, row in df.iterrows():
        pid = _safe_text(row.get("place_id", ""))
        if not pid:
            continue

        place_rows[pid] = row
        district = _get_district_value(row)
        dong = _safe_text(row.get("address_dong", ""))
        subway_info = _safe_text(row.get("subway_info", ""))
        latitude = _safe_float(row.get("latitude", row.get("lat")))
        longitude = _safe_float(row.get("longitude", row.get("lng")))

        docs.append(Document(
            page_content=_build_place_vector_content(row),
            metadata={
                "doc_type": "place",
                "place_id": pid,
                "district": district,
                "name": row.get("place_name"),
                "dong": dong,
                "subway_info": subway_info,
                "location_blob": _build_location_blob(row),
                "latitude": latitude,
                "longitude": longitude,
            }
        ))

    review_df = _load_review_chunks()
    if not review_df.empty:
        for _, chunk_row in review_df.iterrows():
            pid = _safe_text(chunk_row.get("place_id", ""))
            place_row = place_rows.get(pid)
            if not pid or place_row is None:
                continue

            district = _get_district_value(place_row)
            dong = _safe_text(place_row.get("address_dong", ""))
            subway_info = _safe_text(place_row.get("subway_info", ""))
            latitude = _safe_float(place_row.get("latitude", place_row.get("lat")))
            longitude = _safe_float(place_row.get("longitude", place_row.get("lng")))

            docs.append(Document(
                page_content=_build_review_chunk_vector_content(chunk_row, place_row),
                metadata={
                    "doc_type": "review_chunk",
                    "chunk_id": _safe_text(chunk_row.get("chunk_id", "")),
                    "place_id": pid,
                    "district": district,
                    "name": place_row.get("place_name"),
                    "dong": dong,
                    "subway_info": subway_info,
                    "location_blob": _build_location_blob(place_row),
                    "latitude": latitude,
                    "longitude": longitude,
                    "review_title": _safe_text(chunk_row.get("review_title", "")),
                    "chunk_text": _safe_text(chunk_row.get("chunk_text", "")),
                    "section_type": _safe_text(chunk_row.get("section_type", "")),
                    "play_tags": _safe_text(chunk_row.get("play_tags", "")),
                    "need_tags": _safe_text(chunk_row.get("need_tags", "")),
                    "issue_tags": _safe_text(chunk_row.get("issue_tags", "")),
                    "source_url": _safe_text(chunk_row.get("source_url", "")),
                }
            ))

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=persist_dir
    )
    _write_index_manifest(persist_path, current_manifest)

    return vectorstore


# 보조 함수: LLM 호출 시 에러 방지
def _safe_invoke(chain, user_msg):
    try:
        return chain.invoke({"user_message": user_msg})
    except Exception as e:
        return f"답변을 생성하는 중 오류가 발생했어요: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# RAG 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

def rag_retrieve(
    vectorstore: Chroma,
    query:       str,
    df:          Optional[pd.DataFrame] = None,
    district:    Optional[str] = None,
    n:           int = 4,
    return_details: bool = False,
    candidate_pids: Optional[list[str]] = None,
    exclude_pids: Optional[list[str]] = None,
) -> Union[list, dict]:
    """
    벡터 유사도 검색으로 관련 장소 ID 리스트를 반환합니다.

    Args:
        vectorstore: build_vectorstore() 가 반환한 Chroma 인스턴스
        query:       사용자 자연어 질문
        district:    자치구 필터 ("전체" 또는 빈 문자열이면 필터 없음)
        n:           반환할 최대 결과 수

    Returns:
        list[str] | dict: place_id 리스트 또는 retrieval 상세 정보
    """
    if vectorstore is None:
        return {"pids": [], "evidence_by_pid": {}} if return_details else []

    candidate_pid_set = None
    if candidate_pids:
        candidate_pid_set = {
            _safe_text(pid).upper()
            for pid in candidate_pids
            if _safe_text(pid)
        }
        if not candidate_pid_set:
            return {"pids": [], "evidence_by_pid": {}} if return_details else []

    exclude_pid_set = {
        _safe_text(pid).upper()
        for pid in (exclude_pids or [])
        if _safe_text(pid)
    }

    location_hints = _extract_location_hints(query, df)
    if district is None:
        district = location_hints["district"] or _extract_district_from_query(query)
    filter_dict = (
        {"district": district}
        if district and district not in ("전체", "")
        else None
    )
    try:
        search_k = max(n * REVIEW_SEARCH_K_MULTIPLIER, LOCATION_OVERSAMPLE * 2)
        results = vectorstore.similarity_search(query, k=search_k, filter=filter_dict)
        location_pids = _rank_location_candidates(df, location_hints)
        place_coords = _build_location_index(df).get("place_coords", {})
        place_buckets = {}

        for rank, doc in enumerate(results, start=1):
            metadata = doc.metadata or {}
            pid = _safe_text(metadata.get("place_id", "")).upper()
            if not pid:
                continue
            if pid in exclude_pid_set:
                continue
            if candidate_pid_set is not None and pid not in candidate_pid_set:
                continue

            doc_type = _safe_text(metadata.get("doc_type", "place")) or "place"
            score = _score_doc(doc, rank, location_hints, place_coords)
            if doc_type == "review_chunk":
                score += 0.75

            bucket = place_buckets.setdefault(
                pid,
                {
                    "place_score": 0.0,
                    "chunk_scores": [],
                    "location_boost": 0.0,
                    "evidence": [],
                },
            )

            if doc_type == "place":
                bucket["place_score"] = max(bucket["place_score"], score)
            else:
                bucket["chunk_scores"].append(score)
                bucket["evidence"].append(_build_evidence_entry(doc, score))

        for idx, pid in enumerate(location_pids):
            pid = _safe_text(pid).upper()
            if not pid:
                continue
            if pid in exclude_pid_set:
                continue
            if candidate_pid_set is not None and pid not in candidate_pid_set:
                continue
            bucket = place_buckets.setdefault(
                pid,
                {
                    "place_score": 0.0,
                    "chunk_scores": [],
                    "location_boost": 0.0,
                    "evidence": [],
                },
            )
            bucket["location_boost"] = max(
                bucket["location_boost"],
                max((LOCATION_OVERSAMPLE - idx) * 0.8, 0.0),
            )

        ranked_pids = []
        for pid, bucket in place_buckets.items():
            top_chunk_scores = sorted(bucket["chunk_scores"], reverse=True)[:REVIEW_EVIDENCE_LIMIT]
            final_score = (
                bucket["place_score"]
                + sum(top_chunk_scores) * 0.85
                + bucket["location_boost"]
            )
            ranked_pids.append((final_score, pid))

        ranked_pids.sort(key=lambda item: item[0], reverse=True)
        pids = [pid for _, pid in ranked_pids[:n]]
        evidence_by_pid = {
            pid: _finalize_evidence_entries(place_buckets.get(pid, {}).get("evidence", []))
            for pid in pids
        }

        if return_details:
            return {
                "pids": pids,
                "evidence_by_pid": evidence_by_pid,
            }
        return pids
    except Exception:
        return {"pids": [], "evidence_by_pid": {}} if return_details else []


def build_context(
    df:      pd.DataFrame,
    dev_df:  pd.DataFrame,
    pids:    list,
    age_months: Optional[int] = None,
    query:   str = "",
    evidence_by_pid: Optional[dict] = None,
) -> str:
    """
    검색된 장소 + 발달 정보를 LLM 에 전달할 컨텍스트 문자열로 조합합니다.

    Args:
        df:      load_places() DataFrame
        dev_df:  load_dev() DataFrame
        pids:    rag_retrieve() 가 반환한 place_id 리스트
        age_months: 사용자 질문/세션에서 추정한 아이 월령
        evidence_by_pid: 검색 단계에서 선별된 장소별 리뷰 근거

    Returns:
        str: LLM 프롬프트용 컨텍스트
    """
    if age_months is None:
        age_months = extract_age_months_from_text(query)

    rows = df[df["place_id"].isin(pids)]
    rows_by_pid = {
        _safe_text(row.get("place_id", "")).upper(): row
        for _, row in rows.iterrows()
    }
    dev_row = _get_dev_row_for_age(dev_df, age_months)
    evidence_by_pid = evidence_by_pid or {}
    parts = []
    for raw_pid in pids:
        pid = _safe_text(raw_pid).upper()
        r = rows_by_pid.get(pid)
        if r is None:
            continue
        district = _get_district_value(r)
        feat_str = _serialize_feature_text(r) or "정보 없음"

        matched_facilities = _match_dev_facilities_for_place(r, dev_row)
        dev_hint = ""
        if dev_row is not None:
            dev_keywords = _safe_text(dev_row.get("keywords", ""))
            if matched_facilities:
                dev_hint = (
                    f"{dev_keywords} | 이곳에서는 "
                    f"{', '.join(matched_facilities[:3])} 같은 놀이가 잘 맞아요."
                )
            elif dev_keywords:
                dev_hint = f"{dev_keywords} | 이 월령대 아이가 몰입하기 쉬운 놀이가 있는지 함께 봐주세요."
        
        park_str = park_short(str(r.get("parking_info", "")))
        res_url  = str(r.get("reservation_url", "")).strip() or PUBLIC_BOOK
        tip      = "미끄럼방지 양말 필수" if r.get("needs_socks") else ""
        crowded  = "주말 혼잡 주의" if r.get("is_crowded") else "예약 확인 권장"
        evidence_items = evidence_by_pid.get(pid, [])
        evidence_lines = []
        for idx, item in enumerate(evidence_items[:REVIEW_EVIDENCE_LIMIT], start=1):
            tag_bits = []
            if item.get("section_type"):
                tag_bits.append(item["section_type"])
            if item.get("play_tags"):
                tag_bits.append(item["play_tags"].replace("|", ", "))
            tag_prefix = f"[{' / '.join(tag_bits[:2])}] " if tag_bits else ""
            evidence_lines.append(
                f"  리뷰근거{idx}: {tag_prefix}{item.get('text', '')}"
            )

        if not evidence_lines:
            fallback_review = _compact_preview_text(r.get("review_text", ""), limit=REVIEW_EVIDENCE_CHAR_LIMIT)
            if fallback_review:
                evidence_lines.append(f"  리뷰요약: {fallback_review}")

        evidence_block = ""
        if evidence_lines:
            evidence_block = "\n" + "\n".join(evidence_lines)

        parts.append(
            f"[{r['place_name']}]\n"
            f"  위치: {district} {r['address']}\n"
            f"  교통: {str(r.get('subway_info', ''))[:120]}\n"
            f"  연령: {r['age_text']} | 이용료: 기본 3,000원\n"
            f"  주차: {park_str} | 혼잡: {crowded}\n"
            f"  특징: {feat_str}\n"
            f"  방문팁: {tip}\n"
            f"  예약: {res_url}\n"
            f"  발달포인트: {dev_hint}"
            + evidence_block
        )

    dev_str = ""
    if dev_row is not None:
        age_label = _safe_text(dev_row.get("age", ""))
        facilities = ", ".join(_split_context_terms(dev_row.get("matching_facilities", ""))[:5])
        month_label = f"{age_months}개월" if age_months is not None else "월령 정보 미상"
        dev_str = (
            f"[아이 발달 프로필]\n"
            f"  추정 월령: {month_label}\n"
            f"  참고 구간: {age_label}개월\n"
            f"  발달 키워드: {str(dev_row.get('keywords',''))[:90]}\n"
            f"  잘 맞는 놀이: {facilities}\n"
            f"  대근육: {str(dev_row.get('gross_motor_skills',''))[:90]}\n"
            f"  소근육: {str(dev_row.get('fine_motor_skills',''))[:90]}\n"
            f"  언어:   {str(dev_row.get('language_development',''))[:90]}\n"
            f"  인지:   {str(dev_row.get('cognitive_development',''))[:90]}\n"
            f"  사회성: {str(dev_row.get('social_development',''))[:90]}"
        )

    context_blocks = []
    if dev_str:
        context_blocks.append(dev_str)
    context_blocks.extend(parts)
    return "\n\n".join(context_blocks)
