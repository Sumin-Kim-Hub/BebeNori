import streamlit as st
import pandas as pd
import pydeck as pdk
import re
import urllib.parse

TAG_MAP = {
    "parking_available": "#주차가능", "parking_paid": "#유료주차", "reservation_available": "#예약가능",
    "stroller_parking": "#유모차보관소", "cafe": "#카페시설", "wi_fi": "#무선인터넷",
    "nursing_room": "#수유실완비", "diaper_table": "#기저귀교환대", "microwave": "#전자레인지(이유식)",
    "high_chair": "#아기의자구비", "baby_food_allowed": "#이유식반입가능", "nursing_pillow": "#수유쿠션비치",
    "toddler_friendly": "#영유아특화", "preschool_friendly": "#유아환영", "care_service_available": "#돌봄서비스",
    "careserviceavailable": "#돌봄서비스", "guardian_required": "#보호자동반", "guardianrequired": "#보호자동반",
    "air_purifier": "#공기청정기", "cctv": "#CCTV설치", "first_aid_kit": "#구급함비치",
    "play_zone": "#다양한놀이존", "discount_available": "#할인혜택"
}

DEFAULT_IMG = "https://images.unsplash.com/photo-1566454825481-4e48f80aa4d7?q=80&w=500&auto=format&fit=crop"

def parse_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2" class="res-btn" target="_blank">\1</a>', text)
    return text

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("""<div style="padding: 0px 0 20px 0; text-align: center;"><span class="sidebar-logo">BEBENORI</span><br><span style="color:#6B9DD4; font-size:0.85rem; font-weight: 800;">서울형 키즈카페 안심 가이드</span></div>""", unsafe_allow_html=True)
        if st.button("새로운 대화 시작", use_container_width=True, type="primary"):
            new_id = len(st.session_state.sessions)
            st.session_state.sessions.append({"id": new_id, "title": f"새 대화 {new_id + 1}", "chat_history": [{"role": "assistant", "content": "반가워요! 베베노리 이모삼촌이에요 🐰\n궁금한 키즈카페가 있나요?", "source_docs": []}]})
            st.session_state.current_session_id = new_id
            st.rerun()
        st.divider()
        st.markdown("<p style='font-size:0.8rem; font-weight:bold; color:#ccc; margin-bottom: 5px;'>최근 대화 목록</p>", unsafe_allow_html=True)
        for i in range(len(st.session_state.sessions)-1, -1, -1):
            sess = st.session_state.sessions[i]
            if st.button(sess['title'], key=f"nav_btn_{i}", use_container_width=True):
                st.session_state.current_session_id = i
                st.rerun()
        st.divider()
        with st.expander("키즈카페 지도", expanded=True):
            if df is not None and not df.empty:
                districts = ["서울시 전체"] + sorted([d for d in df["district"].dropna().unique() if d])
                selected_gu = st.selectbox("자치구 이동", districts, key="sb_map_gu", label_visibility="collapsed")
                map_df = df.dropna(subset=["latitude", "longitude"]).copy()
                if selected_gu != "서울시 전체": 
                    map_df = map_df[map_df["district"] == selected_gu]
                
                map_df["tooltip_name"] = map_df["place_name"].astype(str).str.replace("서울형 키즈카페", "").str.strip()
                map_df["tooltip_dong"] = map_df["address_dong"].fillna("").astype(str).apply(lambda x: f"({x})" if x and x.lower() != "nan" else "")
                
                layer = pdk.Layer(
                    'ScatterplotLayer', data=map_df, get_position='[longitude, latitude]', 
                    get_color='[242, 183, 5, 220]', get_radius=250, pickable=True
                )
                view_state = pdk.ViewState(
                    latitude=map_df["latitude"].mean() if not map_df.empty else 37.5665, 
                    longitude=map_df["longitude"].mean() if not map_df.empty else 126.9780, 
                    zoom=10.5
                )
                st.pydeck_chart(pdk.Deck(
                    map_style='light', initial_view_state=view_state, layers=[layer], 
                    tooltip={"html": "<b>{tooltip_name}</b><br>{tooltip_dong}", "style": {"backgroundColor": "#6B9DD4", "color": "white", "fontSize": "13px", "padding": "10px", "borderRadius": "8px", "maxWidth": "150px"}}
                ), height=230) 
        st.markdown("""<div style="padding: 10px 5px;"><a href="https://yeyak.seoul.go.kr/" target="_blank" style="color: #6B9DD4; font-weight: 800; font-size: 0.95rem; text-decoration: none;">공식 사이트 예약하러 가기</a></div>""", unsafe_allow_html=True)

def get_message_html(role, content, source_docs=None, intent=None):
    content = parse_markdown(content).replace('\n', '<br>')
    
    # 🚀 [수정 포인트 1] 이모삼촌 줄간격(line-height: 1.15) 축소, 노리메이트 귀여운 가족 그림 적용
    ai_label = "<div style='line-height: 1.15; margin-top: 2px;'>베베노리<br>이모삼촌</div>"
    user_icon = '<img src="https://img.icons8.com/color/48/family.png" style="width: 28px; height: 28px;">'
    
    icon, label, color = ("🐰", ai_label, "#F2B705") if role == "assistant" else (user_icon, "노리메이트", "#6B9DD4")
    cls = "ai" if role == "assistant" else "user"
    cards_html = ""
    
    # 🚀 [수정 포인트 2] 일반 꼬리질문(place_detail)일 경우 대형 카드 숨김
    if role == "assistant" and source_docs and intent != "place_detail":
        cards_html = "<div style='display: flex; flex-direction: column; gap: 15px; margin-top: 15px;'>"
        for doc in source_docs[:1]:
            name = doc.get("place_name", "키즈카페")
            address = doc.get("address", "주소 정보 없음")
            link = doc.get("booking_url", "https://yeyak.seoul.go.kr/")
            img_url = ""
            for k in ["image_url", "img_url", "image", "thumbnail", "url"]:
                val = str(doc.get(k, "")).strip()
                if val and val.lower() != "nan" and val.startswith("http"):
                    img_url = val
                    break
            if not img_url: img_url = DEFAULT_IMG
            
            if intent == "doc_lookup":
                cards_html += f"""
                <div style="border: 1px solid #EFEFEF; border-radius: 12px; padding: 12px; background-color: #FAFAFA; box-shadow: 0 2px 8px rgba(0,0,0,0.04); width: 100%;">
                    <div style="font-size: 0.95rem; font-weight: 800; color: #333; margin-bottom: 8px; word-break: keep-all;">📍 {name}</div>
                    <div style="display: flex; gap: 8px;">
                        <a href="{link}" target="_blank" style="flex: 1; background-color: #F2B705; color: #333; text-align: center; padding: 8px 0; border-radius: 6px; text-decoration: none; font-weight: 700; font-size: 0.85rem;">예약하기</a>
                        <a href="https://map.naver.com/v5/search/{urllib.parse.quote(name)}" target="_blank" style="flex: 1; background-color: #6B9DD4; color: #000000; text-align: center; padding: 8px 0; border-radius: 6px; text-decoration: none; font-weight: 700; font-size: 0.85rem;">지도 보기</a>
                    </div>
                </div>
                """
            else:
                raw_feats = doc.get("features", [])
                if isinstance(raw_feats, str): 
                    raw_feats = raw_feats.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",")
                
                tags_html = "".join([
                    f"<span style='color: #6B9DD4; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; background-color: #F4F8FD; margin-right: 4px; margin-bottom: 4px; display: inline-block; white-space: nowrap; border: 1px solid #6B9DD4;'>{TAG_MAP.get(f.strip().lower(), '#기타정보')}</span>" 
                    for f in raw_feats[:6] if f.strip() and f.strip().lower() in TAG_MAP
                ])

                cards_html += f"""
                <div style="border: 2px solid #FDF4D6; border-radius: 14px; overflow: hidden; background-color: #FFFFFF; box-shadow: 0 4px 10px rgba(0,0,0,0.05); width: 100%;">
                    <div style="position: relative; height: 140px; background-color: #EFEFEF;">
                        <img src="{img_url}" onerror="this.onerror=null; this.src='{DEFAULT_IMG}';" style="width: 100%; height: 100%; object-fit: cover;">
                        <div style="position: absolute; top: 10px; left: 10px;">
                            <span style="background-color: #F2B705; color: #333; padding: 4px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 800; box-shadow: 0 2px 4px rgba(0,0,0,0.15);">1순위 추천</span>
                        </div>
                    </div>
                    <div style="padding: 14px;">
                        <div style="font-size: 1.05rem; font-weight: 800; color: #333; margin-bottom: 4px; word-break: keep-all;">{name}</div>
                        <div style="font-size: 0.8rem; color: #777; margin-bottom: 10px; word-break: keep-all;">{address}</div>
                        <div style="margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 4px;">{tags_html}</div>
                        <div style="display: flex; gap: 6px;">
                            <a href="{link}" target="_blank" style="flex: 1; background-color: #F2B705; color: #333; text-align: center; padding: 8px 0; border-radius: 6px; text-decoration: none; font-weight: 800; font-size: 0.9rem;">예약하기</a>
                            <a href="https://map.naver.com/v5/search/{urllib.parse.quote(name)}" target="_blank" style="flex: 1; background-color: #6B9DD4; color: #000000; text-align: center; padding: 8px 0; border-radius: 6px; text-decoration: none; font-weight: 800; font-size: 0.9rem;">지도 보기</a>
                        </div>
                    </div>
                </div>
                """
        
        if len(source_docs) > 1 and intent not in ("doc_lookup", "place_detail"):
            next_place_name = source_docs[1].get("place_name", "다른 키즈카페")
            clean_next_name = next_place_name.replace("서울형 키즈카페", "").strip()
            cards_html += f"""<div style="margin-top: 10px; display: inline-block; background-color: #FFF4D6; border: 1px dashed #F2B705; color: #B28704; padding: 8px 14px; border-radius: 16px; font-size: 0.8rem; font-weight: 800; word-break: keep-all;">💡 추천 질문: "{clean_next_name}도 알려줘" 또는 "여기 주차 어떻게 해?" 라고 입력해보세요!</div>"""
        
        cards_html += "</div>"
        
    # 🚀 [수정 포인트 3] 프로필 이름 글씨 색상(color: white) 강제 적용 및 이미지 정렬
    profile_html = f"""<div style="display: flex; flex-direction: column; align-items: center; min-width: 55px;"><div style="width: 42px; height: 42px; border-radius: 50%; background: white; border: 2px solid {color}; display: flex; justify-content: center; align-items: center;"><span style="font-size: 22px; display: flex; align-items: center; justify-content: center;">{icon}</span></div><div style="font-size: 0.65rem; font-weight: bold; color: white; margin-top: 4px; text-align: center;">{label}</div></div>"""
    bubble_html = f'<div class="bubble {cls}-bubble">{content}{cards_html}</div>'
    
    display_content = bubble_html + profile_html if role == "user" else profile_html + bubble_html
    return f'<div class="bubble-container {cls}-msg">{display_content}</div>'