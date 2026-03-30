import streamlit as st

def render_sidebar(df, chat_history):
    with st.sidebar:
        st.markdown("<h2 style='font-family:Fredoka One; color:#F2B705; text-align:center;'>BEBENORI</h2>", unsafe_allow_html=True)
        if st.button("➕ 새로운 대화 시작", use_container_width=True, type="primary"):
            st.session_state.chat_history = []
            st.rerun()
            
        st.divider()
        
        # 텍스트가 정상적으로 보이도록 복구
        with st.expander("🌱 개월별 발달단계 정보", expanded=True):
            st.markdown("""
            <div style="font-size:0.85rem; line-height:1.7;">
            <b>~12개월</b>: 오감 자극 놀이<br>
            <b>13~24개월</b>: 대근육 발달 (트램펄린)<br>
            <b>25~36개월</b>: 역할놀이 시작 (주방놀이)
            </div>
            """, unsafe_allow_html=True)
            
        with st.expander("🗺️ 전체 매장 보기"):
            st.link_button("서울형 키즈카페 공식 지도", "https://yeyak.seoul.go.kr/", use_container_width=True)

def get_message_html(role, content):
    if role == "assistant":
        # 왼쪽 정렬 (베베노리 이모)
        content = content.replace('\n', '<br>') # 줄바꿈 처리
        return f"""
        <div class="bubble-container ai-msg">
            <div class="avatar-wrap">
                <div class="avatar-circle ai-avatar"><span style="font-size: 22px;">🐰</span></div>
                <div class="avatar-text">베베노리 이모</div>
            </div>
            <div class="bubble ai-bubble">{content}</div>
        </div>
        """
    else:
        # 오른쪽 정렬 (사용자)
        return f"""
        <div class="bubble-container user-msg">
            <div class="bubble user-bubble">{content}</div>
            <div class="avatar-wrap">
                <div class="avatar-circle user-avatar"><span style="font-size: 22px;">👪</span></div>
                <div class="avatar-text">사용자</div>
            </div>
        </div>
        """

def render_recommendation_cards(docs):
    cols = st.columns(len(docs))
    for i, doc in enumerate(docs):
        with cols[i]:
            st.markdown(f"""
            <div style="background:white; padding:15px; border-radius:15px; color:#222; border:1px solid #ddd; margin-bottom:5px;">
                <p style="font-size:14px; font-weight:bold; margin-bottom:5px;">{doc.get('place_name', '키즈카페')}</p>
                <p style="color:#FF7043; font-weight:bold; font-size:13px;">💰 3,000원</p>
            </div>
            """, unsafe_allow_html=True)
            st.link_button("📅 예약", doc.get("booking_url", "https://yeyak.seoul.go.kr/"), use_container_width=True)