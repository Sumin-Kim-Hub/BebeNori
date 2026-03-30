import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# 1. 경로 설정 및 라이브러리 임포트 (기존 유지)
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _APP_DIR.parent
for folder in ["01_data_prep", "02_model_logic"]:
    path = str(_PROJECT_DIR / folder)
    if path not in sys.path: 
        sys.path.insert(0, path)

import ui_components
from data_loader import load_places, load_dev
from backend_core import (
    get_llm_chain, load_or_create_vectorstore, 
    rag_retrieve, build_context, gen_answer, gen_followup_answer
)
from followup_resolver import resolve_followup, format_doc_lookup_answer

st.set_page_config(page_title="BEBENORI", layout="centered")

# 2. CSS 로드
try:
    with open(_APP_DIR / "style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except: 
    pass

# 3. 데이터 및 모델 초기화 (기존 버전 유지)
@st.cache_resource(show_spinner=False)
def init_all_systems():
    # 실제 구현에서는 try-except 등으로 데이터를 가져옴
    # 여기서는 데이터가 있다고 가정
    df = load_places()
    dev_df = load_dev()
    vectorstore = load_or_create_vectorstore(df)
    llm_chain = get_llm_chain()
    return df, dev_df, vectorstore, llm_chain

df, dev_df, vectorstore, llm_chain = init_all_systems()

# 4. 세션 관리 초기화
if "sessions" not in st.session_state:
    st.session_state.sessions = [{
        "id": 0, "title": "첫 번째 대화",
        "chat_history": [{
            "role": "assistant", 
            "content": "반가워요! 베베노리 이모예요 🐰\n우리 아이와 함께 가기 좋은 **서울형 키즈카페**를 찾아 드릴게요!",
            "source_docs": [],
            "turn_meta": {}
        }]
    }]
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = 0

current_session = st.session_state.sessions[st.session_state.current_session_id]

# 5. UI 출력
ui_components.render_sidebar(df)
for chat in current_session["chat_history"]:
    st.markdown(ui_components.get_message_html(chat["role"], chat["content"]), unsafe_allow_html=True)

# 6. 채팅 입력 및 RAG 로직 ( turn_meta 저장 누락 해결 및 기존 로직 유지 )
if prompt := st.chat_input("이모삼촌에게 무엇이든 물어보세요! 🐣"):
    if len(current_session["chat_history"]) <= 1: 
        current_session["title"] = prompt[:12] + "..."
    
    # 사용자 메시지 저장 및 출력
    current_session["chat_history"].append({"role": "user", "content": prompt})
    st.markdown(ui_components.get_message_html("user", prompt), unsafe_allow_html=True)

    with st.chat_message("assistant", avatar=None):
        with st.spinner("베베노리 이모가 조카를 위한 최적의 장소를 찾는 중... ✨"):
            # ✅ 팀원들이 만든 followup_resolver 연동
            resolution = resolve_followup(current_session["chat_history"], prompt)
            intent = resolution.get("intent")
            age_sel = st.session_state.get("sb_age", "")
            
            response_text, source_pids = "", []
            
            if intent == "doc_lookup":
                # 단순 정보 조회
                response_text = format_doc_lookup_answer(resolution.get("target_doc"), resolution.get("lookup_field"))
            elif intent == "place_detail":
                # 특정 장소 상세 설명
                target_doc = resolution.get("target_doc")
                pids = [target_doc.get("place_id")] if target_doc else []
                ctx = build_context(df, dev_df, pids, age_sel=age_sel, query=prompt)
                response_text = gen_followup_answer(llm_chain, prompt, ctx, resolution.get("last_answer", ""))
            else:
                # 새로운 검색 또는 추가 조건 검색
                search_query = resolution.get("standalone_query", prompt)
                source_pids = rag_retrieve(vectorstore, search_query, df)
                ctx = build_context(df, dev_df, source_pids, age_sel=age_sel, query=search_query)
                
                # 대화가 길어지면 주아님의 후속 답변 프롬프트 사용
                if intent == "refine_search" or len(current_session["chat_history"]) > 3:
                    response_text = gen_followup_answer(llm_chain, prompt, ctx, resolution.get("last_answer", ""))
                else:
                    response_text = gen_answer(llm_chain, prompt, ctx)

        # AI 답변 화면 출력
        st.markdown(ui_components.get_message_html("assistant", response_text), unsafe_allow_html=True)
        
        # ✅ 최종 답변 및 메타데이터를 세션에 저장 (꼬리질문 버그 해결 핵심)
        current_session["chat_history"].append({
            "role": "assistant", "content": response_text,
            "turn_meta": {"intent": intent, "standalone_query": resolution.get("standalone_query", prompt)},
            "source_docs": df[df["place_id"].isin(source_pids)].to_dict("records") if source_pids else resolution.get("source_docs", [])
        })
        
    st.rerun()