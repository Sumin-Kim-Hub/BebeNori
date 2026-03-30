import streamlit as st
from pathlib import Path
import sys
import ui_components

# 1. 경로 설정 및 초기화
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _APP_DIR.parent
for folder in ["01_data_prep", "02_model_logic"]:
    path = str(_PROJECT_DIR / folder)
    if path not in sys.path: sys.path.insert(0, path)

# 2. 페이지 설정
st.set_page_config(page_title="BEBENORI", layout="centered")

# 3. CSS 로드
try:
    with open(_APP_DIR / "style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except: pass

# 🚨 요청 반영: 앱 첫 실행 시 카카오톡 챗봇처럼 첫 인사말 자동 출력
if "chat_history" not in st.session_state or not st.session_state.chat_history:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "안녕하세요! 베베노리 이모예요 🐰\n우리 아이에게 꼭 맞는 서울형 키즈카페,\n무엇이든 물어보세요!"}
    ]

# 4. 사이드바 출력
ui_components.render_sidebar(None, st.session_state.chat_history)

# 5. 채팅 메시지 출력 (HTML 기반 좌/우 정렬 카톡 스타일)
for chat in st.session_state.chat_history:
    st.markdown(ui_components.get_message_html(chat["role"], chat["content"]), unsafe_allow_html=True)
    
    if chat.get("source_docs"):
        ui_components.render_recommendation_cards(chat["source_docs"])

# 6. 질문 입력 및 RAG 실행
if prompt := st.chat_input("이모삼촌에게 무엇이든 물어보세요! 🐣"):
    # 사용자 메시지 저장 및 화면 출력
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.markdown(ui_components.get_message_html("user", prompt), unsafe_allow_html=True)

    # RAG 응답 생성 영역
    with st.chat_message("assistant", avatar=None):
        # 여기에 팀원들의 실제 RAG 로직 (rag_retrieve 등)을 연결하세요.
        # --- RAG 로직 예시 ---
        response = "좋은 질문이네요! 안심하고 갈 수 있는 곳을 추천해 드릴게요. 🌙"
        # --------------------
        
        st.markdown(ui_components.get_message_html("assistant", response), unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    st.rerun()