import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# 🌟 배포 환경(SQLite 버전 문제) 대응: chromadb 로드 전 실행 필수
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# 1. 경로 설정 및 연동 유지
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
    rag_retrieve, build_context, gen_answer, gen_followup_answer,
    infer_answer_place,
)
from followup_resolver import resolve_followup, format_doc_lookup_answer

# ✅ 반응형 및 레이아웃 설정
st.set_page_config(page_title="BEBENORI", layout="centered", initial_sidebar_state="auto")

# 2. CSS 로드 (하단 여백 및 입력창 디자인)
try:
    with open(_APP_DIR / "style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except: pass

# 3. 데이터 및 모델 초기화 (에러 방지용 캐싱)
@st.cache_resource(show_spinner=False)
def init_all_systems():
    df = load_places()
    dev_df = load_dev()
    vectorstore = load_or_create_vectorstore(df)
    llm_chain = get_llm_chain()
    return df, dev_df, vectorstore, llm_chain

df, dev_df, vectorstore, llm_chain = init_all_systems()

# 4. 세션 관리 (이모티콘 제거 및 초기화)
if "sessions" not in st.session_state:
    st.session_state.sessions = [{
        "id": 0, "title": "첫 번째 대화",
        "chat_history": [{
            "role": "assistant", 
            "content": "반가워요! 베베노리 이모예요 🐰\n우리 아이와 함께 가기 좋은 서울형 키즈카페를 찾아 드릴게요!",
            "source_docs": [], "turn_meta": {}
        }]
    }]
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = 0

current_session = st.session_state.sessions[st.session_state.current_session_id]

def _ordered_source_docs(df: pd.DataFrame, pids: list) -> list:
    if not pids: return []
    rows = df[df["place_id"].isin(pids)].to_dict("records")
    by_pid = {row.get("place_id"): row for row in rows}
    return [by_pid[pid] for pid in pids if pid in by_pid]

# 5. UI 출력
ui_components.render_sidebar(df)
for chat in current_session["chat_history"]:
    # 🚀 [수정 포인트] 이전 대화 기록 렌더링 시에도 intent 정보를 넘겨줌
    past_intent = chat.get("turn_meta", {}).get("intent")
    st.markdown(ui_components.get_message_html(chat["role"], chat["content"], source_docs=chat.get("source_docs", []), intent=past_intent), unsafe_allow_html=True)

# 6. 채팅 입력 및 RAG 로직 (v3.31 통합 로직)
if prompt := st.chat_input("이모삼촌에게 무엇이든 물어보세요!"):
    if len(current_session["chat_history"]) <= 1: 
        current_session["title"] = prompt[:12] + "..."
    
    current_session["chat_history"].append({"role": "user", "content": prompt})
    st.markdown(ui_components.get_message_html("user", prompt), unsafe_allow_html=True)

    with st.chat_message("assistant", avatar=None):
        with st.spinner("최적의 장소를 찾는 중... ✨"):
            resolution = resolve_followup(current_session["chat_history"], prompt)
            intent = resolution.get("intent")
            age_sel = "" # 수민님의 요청에 따른 고정값
            
            response_text, source_pids = "", []
            saved_source_docs = []
            active_place_id = resolution.get("active_place_id")
            active_place_rank = resolution.get("target_doc_rank", 0)
            search_slots = dict(resolution.get("search_slots", {}) or {})
            
            # 인텐트 기반 처리
            if intent == "doc_lookup":
                response_text = format_doc_lookup_answer(resolution.get("target_doc"), resolution.get("lookup_field")) or "관련 정보가 부족해요."
                saved_source_docs = resolution.get("source_docs", [])
                if resolution.get("target_doc"): active_place_id = resolution["target_doc"].get("place_id")
            elif intent == "place_detail":
                target_doc = resolution.get("target_doc")
                pids = [target_doc.get("place_id")] if target_doc else []
                ctx = build_context(df, dev_df, pids, age_sel=age_sel, query=prompt)
                response_text = gen_followup_answer(llm_chain, prompt, ctx, resolution.get("last_answer", ""))
                saved_source_docs = resolution.get("source_docs", [])
                if target_doc: active_place_id = target_doc.get("place_id")
            else:
                search_query = resolution.get("standalone_query", prompt)
                source_pids = rag_retrieve(vectorstore, search_query, df)
                ctx = build_context(df, dev_df, source_pids, age_sel=age_sel, query=search_query)
                
                if intent == "refine_search" or len(current_session["chat_history"]) > 3:
                    response_text = gen_followup_answer(llm_chain, prompt, ctx, resolution.get("last_answer", ""))
                else:
                    response_text = gen_answer(llm_chain, search_query, ctx)
                
                saved_source_docs = _ordered_source_docs(df, source_pids)
                active_place_id, active_place_rank = infer_answer_place(saved_source_docs, response_text)

                # LLM이 답변에서 언급한 장소를 카드(첫 번째 doc)로 띄우기 위한 재정렬 로직
                if active_place_id:
                    for i, doc in enumerate(saved_source_docs):
                        if doc.get("place_id") == active_place_id:
                            active_doc = saved_source_docs.pop(i)
                            saved_source_docs.insert(0, active_doc)
                            break

        # [수정] 생성된 답변을 그릴 때 현재 intent를 함께 넘겨줌
        st.markdown(ui_components.get_message_html("assistant", response_text, saved_source_docs, intent=intent), unsafe_allow_html=True)
        
        # 메타데이터 저장 및 세션 업데이트
        current_session["chat_history"].append({
            "role": "assistant", "content": response_text,
            "turn_meta": {
                "intent": intent, "standalone_query": resolution.get("standalone_query", prompt),
                "active_place_id": active_place_id, "active_place_rank": active_place_rank,
                "retrieved_pids": source_pids if source_pids else resolution.get("retrieved_pids", []),
                "search_slots": search_slots,
            },
            "source_docs": saved_source_docs
        })
        
    st.rerun()