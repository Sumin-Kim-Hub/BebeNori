# 수정일: 2026-03-27(진국)
# 수정 이유: 데이터 컬럼명 일치를 위해 수정
# =============================================================================
# 파일 위치: 01_data_prep/data_loader.py
# 역할: 원본 CSV 4종을 읽어 병합·정제한 DataFrame 을 반환하는 전처리 모듈.
#       다른 모든 모듈은 이 파일을 import 해서 데이터를 사용합니다.
# =============================================================================

from pathlib import Path
import pandas as pd

# ── 경로 상수 ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
# 전처리된 파일들이 모여있는 곳으로 경로 설정
DB_DIR = _HERE / "outputs" / "preprocessed" 
RAW_DIR = _HERE / "outputs" / "raw"

PLACES_CSV   = DB_DIR / "places.csv"
FEATURES_CSV = DB_DIR / "place_features.csv"
REVIEWS_CSV  = DB_DIR / "review_docs.csv"
DEV_CSV      = DB_DIR / "baby_development.csv"
RAW_DEV_CSV  = RAW_DIR / "baby_development.csv"

# ChromaDB 영속 저장 경로 (다른 모듈에서 import 해 사용)
CHROMA_DIR = str(_HERE.parent / "02_model_logic" / "chroma_db")

# ── 공통 상수 ─────────────────────────────────────────────────────────────
FALLBACK_IMG = (
    "https://images.unsplash.com/photo-1587654780291-39c9404d746b"
    "?auto=format&fit=crop&w=800&q=80"
)
PUBLIC_BOOK          = "https://yeyak.seoul.go.kr"
CONFIDENCE_THRESHOLD = 0.7
REVIEW_DOC_LIMIT     = 3
REVIEW_TEXT_LIMIT    = 1200

FEAT_SKIP: set = {
    "district", "has_phone", "guardian_rule_mentioned",
    "socks_rule_mentioned", "cleanliness_negative", "crowded_warning",
}

DISTRICTS: list = ["전체"] + sorted([
    "강남구","강동구","강북구","강서구","관악구","광진구","구로구","금천구",
    "노원구","도봉구","동대문구","동작구","마포구","서대문구","서초구",
    "성동구","성북구","송파구","양천구","영등포구","용산구","은평구",
    "종로구","중구","중랑구",
])


# ── 데이터 로드 함수 ──────────────────────────────────────────────────────

def _normalize_place_id(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.upper()


def _join_review_text(values, item_limit=None, char_limit=REVIEW_TEXT_LIMIT) -> str:
    texts = []
    for value in values:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            continue
        texts.append(text)

    if item_limit is not None:
        texts = texts[:item_limit]

    joined = " ".join(texts)
    if char_limit and len(joined) > char_limit:
        return joined[:char_limit].rstrip()
    return joined


def _aggregate_reviews(revs: pd.DataFrame) -> pd.DataFrame:
    revs = revs.copy()

    if "content" not in revs.columns and "chunk_text" in revs.columns:
        revs["content"] = revs["chunk_text"]

    review_text_col = "content" if "content" in revs.columns else "chunk_text"

    if "doc_id" not in revs.columns:
        return (
            revs.groupby("place_id", as_index=False)
            .agg(
                review_text=(
                    review_text_col,
                    lambda x: _join_review_text(x, item_limit=6),
                ),
                review_count=(review_text_col, "count"),
            )
        )

    revs["_source_order"] = range(len(revs))
    sort_cols = ["place_id", "doc_id", "_source_order"]
    if "chunk_order" in revs.columns:
        revs["chunk_order"] = pd.to_numeric(revs["chunk_order"], errors="coerce")
        sort_cols = ["place_id", "doc_id", "chunk_order", "_source_order"]
    revs = revs.sort_values(sort_cols, na_position="last")

    doc_level = (
        revs.groupby(["place_id", "doc_id"], as_index=False)
        .agg(
            review_doc_text=(
                review_text_col,
                lambda x: _join_review_text(x, item_limit=None),
            ),
            _doc_order=("_source_order", "min"),
        )
        .sort_values(["place_id", "_doc_order"])
    )

    return (
        doc_level.groupby("place_id", as_index=False)
        .agg(
            review_text=(
                "review_doc_text",
                lambda x: _join_review_text(x, item_limit=REVIEW_DOC_LIMIT),
            ),
            review_count=("doc_id", "nunique"),
        )
    )


def _resolve_dev_csv_path() -> Path:
    if DEV_CSV.exists():
        return DEV_CSV
    return RAW_DEV_CSV

def load_places() -> pd.DataFrame:
    try:
        places = pd.read_csv(PLACES_CSV, encoding='utf-8-sig')
        places["place_id"] = _normalize_place_id(places["place_id"])
        places["age_min"] = pd.to_numeric(places["age_min"], errors="coerce").fillna(0)
        places["age_max"] = pd.to_numeric(places["age_max"], errors="coerce").fillna(13)
        places["image_url"] = places["image_url"].fillna(FALLBACK_IMG)
        if "district" not in places.columns and "address_gu" in places.columns:
            places["district"] = places["address_gu"].fillna("")

        feats = pd.read_csv(FEATURES_CSV, encoding='utf-8-sig')
        feats["place_id"] = _normalize_place_id(feats["place_id"])

        feat_map = {}

        # ─────────────────────────────────────
        # Case A: 세로형
        # ─────────────────────────────────────
        if "feature_name" in feats.columns:

            if "confidence" in feats.columns:
                feats["confidence"] = pd.to_numeric(feats["confidence"], errors="coerce").fillna(0)
                feats = feats[feats["confidence"] >= CONFIDENCE_THRESHOLD]

            for _, r in feats.iterrows():
                pid = r["place_id"]
                fn = r["feature_name"]

                if fn in FEAT_SKIP:
                    continue

                feat_map.setdefault(pid, []).append(fn)

        # ─────────────────────────────────────
        # Case B: 가로형 (지금 데이터)
        # ─────────────────────────────────────
        else:
            for _, r in feats.iterrows():
                pid = r["place_id"]
                feat_map.setdefault(pid, [])

                for col in feats.columns:
                    if col in ["place_id", "Unnamed: 0"] or col in FEAT_SKIP:
                        continue

                    val = r[col]

                    # ⭐ 핵심: 모든 경우 커버
                    if pd.notna(val) and str(val).lower() in ["yes", "1", "true"]:
                        feat_map[pid].append(col)

        # ─────────────────────────────────────
        # 리뷰 병합
        # ─────────────────────────────────────
        revs = pd.read_csv(REVIEWS_CSV, encoding='utf-8-sig')
        revs["place_id"] = _normalize_place_id(revs["place_id"])
        agg = _aggregate_reviews(revs)

        df = places.merge(agg, on="place_id", how="left")

        df["review_count"] = df["review_count"].fillna(0).astype(int)
        df["review_text"] = df["review_text"].fillna("")

        # ⭐ 핵심: feature 매핑
        df["features"] = df["place_id"].map(feat_map).apply(lambda x: x if isinstance(x, list) else [])

        # ─────────────────────────────────────
        # socks / crowded 처리 (안정 버전)
        # ─────────────────────────────────────
        if "feature_name" in feats.columns:
            socks_ids = set(feats[feats["feature_name"] == "socks_rule_mentioned"]["place_id"])
            crowded_ids = set(feats[feats["feature_name"] == "crowded_warning"]["place_id"])

            df["needs_socks"] = df["place_id"].isin(socks_ids)
            df["is_crowded"] = df["place_id"].isin(crowded_ids)

        else:
            if "socks_rule_mentioned" in feats.columns:
                socks_ids = set(
                    feats[feats["socks_rule_mentioned"].astype(str).str.lower().isin(["yes", "true", "1"])]["place_id"]
                )
                df["needs_socks"] = df["place_id"].isin(socks_ids)
            else:
                df["needs_socks"] = False

            if "crowded_warning" in feats.columns:
                crowded_ids = set(
                    feats[feats["crowded_warning"].astype(str).str.lower().isin(["yes", "true", "1"])]["place_id"]
                )
                df["is_crowded"] = df["place_id"].isin(crowded_ids)
            else:
                df["is_crowded"] = False

        return df

    except Exception as e:
        print(f"❌ load_places 에러 발생: {e}")
        return pd.DataFrame()

def load_dev() -> pd.DataFrame:
    dev_path = _resolve_dev_csv_path()
    try:
        # 1. 먼저 utf-8-sig로 시도 (일반적인 AI 데이터 방식)
        return pd.read_csv(dev_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            # 2. 실패하면 cp949로 시도 (엑셀 기본 저장 방식) << 수정
            return pd.read_csv(dev_path, encoding='cp949')
        except Exception as e:
            print(f"❌ load_dev 최종 로드 실패: {e}")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ load_dev 에러 발생: {e}")
        return pd.DataFrame()

def park_short(raw: str) -> str:
    """주차 안내 원문을 65자 이내 단문으로 요약합니다."""
    if not raw or str(raw).lower() in ("nan", "none", ""):
        return "정보 없음"
    s     = str(raw).strip()
    first = s.split("- ")[1] if "- " in s else s
    first = first.split("\n")[0].strip()
    return first[:65] + ("…" if len(first) > 65 else "")

# ── 4. 자가 검증 (메인 실행부) ────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("       🔍 data_loader 경로 및 데이터 검증")
    print("="*50)
    
    # 파일 존재 여부 확인
    dev_path = _resolve_dev_csv_path()
    files = {
        "장소(places)": PLACES_CSV,
        "특징(features)": FEATURES_CSV,
        "리뷰(reviews)": REVIEWS_CSV,
        "발달(development)": dev_path
    }
    
    all_ok = True
    for name, path in files.items():
        if path.exists():
            print(f" ✅ {name}: 찾음")
        else:
            print(f" ❌ {name}: 못 찾음! (경로: {path})")
            all_ok = False
            
    if all_ok:
        print("\n 🔄 데이터를 불러오는 중...")
        df_test = load_places()
        dev_test = load_dev()
        
        if not df_test.empty:
            print(f" ✨ 성공: 장소 {len(df_test)}개 로드 완료")
            print(f" ✨ 성공: 발달 데이터 {len(dev_test)}행 로드 완료")
            print(f" 📌 특징 샘플: {df_test.iloc[0]['features'] if not df_test.empty else '없음'}")
        else:
            print(" ⚠️ 장소 데이터를 불러오지 못했습니다. 위 에러 메시지를 확인하세요.")
    
    print("="*50 + "\n")
