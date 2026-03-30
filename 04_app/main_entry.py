# ============================================================
# 파일: 04_app/main_entry.py
# 역할: BebeNori 메인 진입점 v7.1
# ============================================================

import random
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 1. 경로 설정 — 절대 수정 금지
# ──────────────────────────────────────────────────────────────
_APP_DIR     = Path(__file__).resolve().parent
_PROJECT_DIR = _APP_DIR.parent
_DATA_PREP   = _PROJECT_DIR / "01_data_prep"
_MODEL_LOGIC = _PROJECT_DIR / "02_model_logic"

for _p in [str(_DATA_PREP), str(_MODEL_LOGIC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ui_components  # noqa: E402

# ──────────────────────────────────────────────────────────────
# 2. 페이지 설정
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="베베노리 — 이모삼촌의 키즈카페 추천",
    page_icon="🌙",
    layout="centered",
    initial_sidebar_state="auto",
)

# ── CSS 적용 ─────────────────────────────────────────────────
try:
    with open(_APP_DIR / "style.css", "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── Google Fonts 명시 로드 ───────────────────────────────────
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Fredoka+One&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# 3. 랜덤 육아 꿀팁 (로딩 대기 시간 관리)
# ──────────────────────────────────────────────────────────────
_LOADING_TIPS: list[str] = [
    "💡 서울형 키즈카페는 매월 1일·16일에 예약이 열려요!",
    "💡 18개월 아기는 소근육 발달이 활발한 시기예요.",
    "💡 방문 전 주차 공간을 미리 확인하면 스트레스가 줄어요!",
    "💡 아이가 처음 가는 곳은 오전 10~11시가 가장 한적해요.",
    "💡 키즈카페 직원에게 인기 장난감 위치를 먼저 물어보세요!",
    "💡 유아용 실내화를 챙기면 더 편하게 놀 수 있어요.",
    "💡 서울형 키즈카페 어린이 2시간 평균 3,000원! 사설 대비 1/10이에요.",
    "💡 36개월 이상 아이는 역할놀이 공간에서 집중력이 폭발해요.",
    "💡 이유식 중인 아기라면 별도 수유실이 있는지 꼭 확인하세요!",
    "💡 베베노리 데이터는 서울시 공공 데이터로 검증됐어요. ✅",
    "💡 24개월 미만 전용 존이 있는 곳에서 더 안전하게 놀 수 있어요.",
    "💡 볼풀 소독 주기를 직접 물어보면 위생 수준을 바로 알 수 있어요!",
]

def _get_tip() -> str:
    return random.choice(_LOADING_TIPS)


# ──────────────────────────────────────────────────────────────
# 4. 백엔드 초기화 (캐싱)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def initialize_backend():
    try:
        from data_loader import load_places, load_dev           # type: ignore
        from backend_core import load_or_create_vectorstore, get_llm_chain  # type: ignore

        df          = load_places()
        dev_df      = load_dev()
        vectorstore = load_or_create_vectorstore(df)
        chain       = get_llm_chain()
        return df, dev_df, vectorstore, chain
    except Exception as exc:
        st.error(f"백엔드 초기화 실패: {exc}")
        return pd.DataFrame(), None, None, None

with st.spinner("🌙 베베노리 이모삼촌이 준비 중이에요…"):
    df, dev_df, vectorstore, chain = initialize_backend()

# ──────────────────────────────────────────────────────────────
# 5. 세션 상태 초기화
# ──────────────────────────────────────────────────────────────
if "chat_history"  not in st.session_state:
    st.session_state.chat_history  = []
if "current_input" not in st.session_state:
    st.session_state.current_input = ""

# ──────────────────────────────────────────────────────────────
# 6. 사이드바
# ──────────────────────────────────────────────────────────────
ui_components.render_sidebar(df, st.session_state.chat_history)

# ──────────────────────────────────────────────────────────────
# 7. 메인 채팅 화면
# ──────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    ui_components.render_gpt_style_welcome()

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        with st.chat_message("user", avatar="👪"):
            st.markdown(chat["content"])
    else:
        with st.chat_message("assistant", avatar="🐰"):
            ui_components.render_ai_label()
            st.markdown(chat["content"])
            if chat.get("source_docs"):
                ui_components.render_recommendation_cards(chat["source_docs"])

# ──────────────────────────────────────────────────────────────
# 8. 채팅 입력 처리 및 RAG 파이프라인
# ──────────────────────────────────────────────────────────────
raw_prompt = st.chat_input(
    "이모삼촌에게 무엇이든 물어보세요! (예: 강남구 18개월 아기 갈만한 곳 🐣)"
)

if st.session_state.current_input:
    prompt = st.session_state.current_input
    st.session_state.current_input = ""
else:
    prompt = raw_prompt

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👪"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🐰"):
        ui_components.render_ai_label()

        with st.spinner(_get_tip()):
            try:
                from backend_core import rag_retrieve, build_context, gen_answer  # type: ignore

                pids = rag_retrieve(vectorstore, prompt, df=df, n=3)
                ctx  = build_context(df, dev_df, pids, query=prompt)
                ans  = gen_answer(chain, prompt, ctx)

                st.markdown(ans)

                docs: list[dict] = []
                if not df.empty and pids:
                    docs = df[df["place_id"].isin(pids)].to_dict("records")
                if docs:
                    ui_components.render_recommendation_cards(docs)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ans, "source_docs": docs}
                )

            except Exception as exc:
                error_msg = (
                    "이모삼촌이 정보를 찾는 데 문제가 생겼어요. 😥\n\n"
                    "잠시 후 다시 질문해 주시겠어요?"
                )
                st.markdown(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )
                with st.expander("🔧 오류 상세 (개발용)"):
                    st.exception(exc)

    st.rerun()
