import streamlit as st
import pandas as pd
import pydeck as pdk
import re

# 마크다운 텍스트를 버튼(<a>) 및 볼드(<strong>)로 변환
def parse_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)
    return text

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            """<div style="background: linear-gradient(135deg, #F2B705 0%, #FFD646 100%);
            border-radius: 12px; padding: 20px 0; margin-bottom: 15px; text-align: center;">
            <span class="sidebar-logo">BEBENORI</span><br>
            <span style="color:white; font-size:0.75rem;">서울형 키즈카페 안심 가이드</span></div>""", unsafe_allow_html=True
        )
        
        if st.button("➕ 새로운 대화 시작", width='stretch', type="primary"):
            new_id = len(st.session_state.sessions)
            st.session_state.sessions.append({
                "id": new_id,
                "title": f"새 대화 {new_id + 1}",
                "chat_history": [{"role": "assistant", "content": "반가워요! 베베노리 이모예요 🐰\n궁금한 키즈카페가 있나요?"}]
            })
            st.session_state.current_session_id = new_id
            st.rerun()
            
        st.divider()
        st.markdown("<p style='font-size:0.8rem; font-weight:bold; color:#ccc;'>최근 대화 목록</p>", unsafe_allow_html=True)
        for i in range(len(st.session_state.sessions)-1, -1, -1):
            sess = st.session_state.sessions[i]
            btn_label = f"💬 {sess['title']}"
            if i == st.session_state.current_session_id: btn_label = f"⭐ {sess['title']}"
            if st.button(btn_label, key=f"nav_btn_{i}", width='stretch'):
                st.session_state.current_session_id = i
                st.rerun()

        st.divider()
        with st.expander("🌱 개월별 발달단계 정보"):
            st.selectbox("아이 개월 수", ["0~12개월", "13~24개월", "25~36개월", "37개월 이상"], key="sb_age")

        with st.expander("🗺️ 키즈카페 지도 (툴팁 포함)"):
            if df is not None and not df.empty:
                districts = ["서울시 전체"] + sorted([d for d in df["district"].dropna().unique() if d])
                selected_gu = st.selectbox("자치구 자동 이동", districts, key="sb_map_gu")
                
                map_df = df.dropna(subset=["latitude", "longitude"])
                if selected_gu != "서울시 전체":
                    map_df = map_df[map_df["district"] == selected_gu]
                
                # ✅ 수정: 토큰 없이 서울 지도가 뜨도록 map_style을 카토DB 스타일로 변경
                st.pydeck_chart(pdk.Deck(
                    map_style='light', # 또는 pdk.map_styles.LIGHT
                    initial_view_state=pdk.ViewState(
                        latitude=map_df["latitude"].mean() if not map_df.empty else 37.5665,
                        longitude=map_df["longitude"].mean() if not map_df.empty else 126.9780,
                        zoom=11 if selected_gu == "서울시 전체" else 13.5,
                        pitch=0,
                    ),
                    layers=[pdk.Layer(
                        'ScatterplotLayer', 
                        data=map_df, 
                        get_position='[longitude, latitude]',
                        get_color='[242, 183, 5, 200]', # 베베노리 노란색 점
                        get_radius=220, 
                        pickable=True
                    )],
                    tooltip={"text": "{place_name}\n({address_dong})"}
                ))
            st.link_button("🌐 공식 사이트 예약", "https://yeyak.seoul.go.kr/", width='stretch')

def get_message_html(role, content):
    content = parse_markdown(content).replace('\n', '<br>')
    icon, label, color = ("🐰", "베베노리 이모", "#F2B705") if role == "assistant" else ("👪", "사용자", "#6B9DD4")
    cls = "ai" if role == "assistant" else "user"
    return f"""
    <div class="bubble-container {cls}-msg">
        <div style="display: flex; flex-direction: column; align-items: center;">
            <div style="width: 42px; height: 42px; border-radius: 50%; background: white; border: 2px solid {color}; display: flex; justify-content: center; align-items: center;">
                <span style="font-size: 22px;">{icon}</span>
            </div>
            <div style="font-size: 0.6rem; font-weight: bold; color: white; margin-top: 4px;">{label}</div>
        </div>
        <div class="bubble {cls}-bubble">{content}</div>
    </div>
    """