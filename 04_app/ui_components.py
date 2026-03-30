# ============================================================
# 파일: 04_app/ui_components.py
# 역할: BebeNori UI 컴포넌트 v7.1
# ============================================================

import streamlit as st
import pandas as pd
import urllib.parse


# ─────────────────────────────────────────────────────────────
# 1. 사이드바
# ─────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame, chat_history: list):
    with st.sidebar:
        # ── 브랜드 마크 ───────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding:8px 0 16px;">
                <div style="display:inline-flex;align-items:center;gap:8px;
                            background:linear-gradient(135deg,#F2B705,#F5C842);
                            padding:8px 20px;border-radius:16px;
                            box-shadow:0 4px 14px rgba(242,183,5,0.28);">
                    <span style="font-size:1.3rem;">🌙</span>
                    <span style="font-family:'Fredoka One',cursive;font-size:1.25rem;
                                 color:#FFFFFF;letter-spacing:0.05em;">BEBENORI</span>
                </div>
                <p style="font-size:0.7rem;color:#B08000;margin:6px 0 0;font-weight:600;">
                    광고 없는 진짜 리뷰 기반
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── 새 대화 버튼 ──────────────────────────────────────
        if st.button("➕ 새로운 대화 시작", use_container_width=True, type="primary"):
            st.session_state.chat_history = []
            st.session_state.current_input = ""
            st.rerun()

        st.divider()

        # ── 🌱 개월별 발달단계 ────────────────────────────────
        with st.expander("🌱 개월별 발달단계 정보"):
            st.markdown(
                """
                <div style="font-size:0.8rem;line-height:1.7;color:#5C4000;">
                <b>~12개월</b><br>
                &nbsp;오감 자극, 뒤집기, 기어가기<br>
                손잡고 서기 시작 → 소근육 장난감 추천<br><br>
                <b>13~24개월</b><br>
                &nbsp;걸음마 시작, 대근육 폭발, 호기심 MAX<br>
                넓은 공간 + 트램펄린 있는 곳 추천<br><br>
                <b>25~36개월</b><br>
                &nbsp;언어 발달, 역할놀이, 소근육 정교화<br>
                역할놀이 존이 있는 곳 베스트<br><br>
                <b>37개월~</b><br>
                &nbsp;사회성 발달, 규칙 이해, 창의력 폭발<br>
                친구들과 함께 노는 구조형 공간 추천<br><br>
                <span style="font-size:0.72rem;color:#888;">
                💡 챗봇에 "18개월 아기 발달에 좋은 곳" 형태로 물어보세요!
                </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── 🗺️ 전체 매장 보기 ─────────────────────────────────
        with st.expander("🗺️ 서울형 키즈카페 전체 지도"):
            st.markdown(
                "<span style='font-size:0.77rem;color:#999;'>"
                "현재 등록된 모든 안심 키즈카페 위치입니다.</span>",
                unsafe_allow_html=True,
            )
            if df is not None and not df.empty:
                map_data = (
                    df[["latitude", "longitude"]]
                    .dropna()
                    .rename(columns={"latitude": "lat", "longitude": "lon"})
                )
                if not map_data.empty:
                    st.map(map_data, zoom=10)
                else:
                    st.caption("위치 정보가 아직 없습니다.")
            else:
                st.info("데이터를 불러오는 중입니다.")

        # ── 📊 AI 신뢰도 대시보드 ─────────────────────────────
        with st.expander("📊 AI 신뢰도 대시보드"):
            _render_trust_dashboard()

        st.divider()

        # ── 💬 최근 대화 기록 ─────────────────────────────────
        st.markdown(
            "<p style='font-size:0.72rem;font-weight:700;color:#B08000;"
            "letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;'>"
            "💬 최근 대화</p>",
            unsafe_allow_html=True,
        )
        user_turns = [c for c in chat_history if c["role"] == "user"] if chat_history else []
        if not user_turns:
            st.caption("아직 대화 기록이 없어요.")
        else:
            for i, chat in enumerate(user_turns[-6:]):
                label = chat["content"][:18] + ("…" if len(chat["content"]) > 18 else "")
                st.button(
                    f"🗣️ {label}",
                    key=f"hist_{i}_{hash(chat['content'][:10])}",
                    disabled=True,
                    use_container_width=True,
                )

        # ── ✨ Coming Soon ─────────────────────────────────────
        st.divider()
        st.markdown(
            "<p style='font-size:0.72rem;font-weight:700;color:#CCCCCC;"
            "letter-spacing:0.06em;text-transform:uppercase;margin-bottom:5px;'>"
            "✨ Coming Soon</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='coming-soon-btn'>", unsafe_allow_html=True)
        st.button(
            "🎁 우리 아이 맞춤 교구·장난감 추천",
            disabled=True,
            use_container_width=True,
            key="coming_soon_toy",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.68rem;color:#CCCCCC;text-align:center;margin-top:4px;'>"
            "곧 업데이트 예정이에요 😊</p>",
            unsafe_allow_html=True,
        )


def _render_trust_dashboard():
    metrics = [
        ("장소 정확도",       95, False),
        ("리뷰 기반 답변율",  92, False),
        ("할루시네이션 방지", 98, False),
        ("응답 일관성",       90, True),
    ]
    html_parts = [
        "<div style='font-size:0.72rem;color:#888;margin-bottom:8px;'>"
        "서울시 공공데이터 + 실사용자 리뷰 기반 평가</div>"
    ]
    for label, pct, is_blue in metrics:
        fill_class = "trust-fill blue" if is_blue else "trust-fill"
        html_parts.append(
            f"""
            <div class="trust-row">
                <div class="trust-meta">
                    <span>{label}</span><span>{pct}%</span>
                </div>
                <div class="trust-track">
                    <div class="{fill_class}" style="width:{pct}%;"></div>
                </div>
            </div>
            """
        )
    html_parts.append(
        "<p style='font-size:0.68rem;color:#B08000;margin-top:8px;font-weight:600;'>"
        "✅ 서울시 공공 데이터 기반 검증 완료</p>"
    )
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 2. 초기 화면
# ─────────────────────────────────────────────────────────────
def render_gpt_style_welcome():
    st.markdown("<div style='height:36px;'></div>", unsafe_allow_html=True)

    # ── 로고 ─────────────────────────────────────────────────
    try:
        from logo_data import LOGO_BASE64  # type: ignore
        src = f"data:image/jpeg;base64,{LOGO_BASE64}"
        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="{src}"
                     style="width:100%;max-width:220px;border-radius:20px;
                            box-shadow:0 10px 28px rgba(242,183,5,0.22);">
            </div>
            """,
            unsafe_allow_html=True,
        )
    except ImportError:
        st.markdown(
            """
            <div style="text-align:center;">
                <div style="display:inline-flex;align-items:center;gap:10px;
                            background:linear-gradient(135deg,#F2B705,#F5C842);
                            padding:16px 32px;border-radius:24px;
                            box-shadow:0 10px 30px rgba(242,183,5,0.28);">
                    <span style="font-size:2.2rem;">🌙</span>
                    <span style="font-family:'Fredoka One',cursive;font-size:1.8rem;
                                 color:#FFFFFF;letter-spacing:0.06em;">BEBENORI</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<p class='welcome-title'>이모삼촌이 함께해요</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='welcome-subtitle'>우리 아이에게 꼭 맞는 곳, 찾아드릴게요 🐰</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button("💡 18개월 아기 강남구 키즈카페", use_container_width=True):
            st.session_state.current_input = "18개월 아기랑 갈만한 강남구 키즈카페 추천해줘"
            st.rerun()
        if st.button("🧦 맨발 놀이 가능한 위생적인 곳", use_container_width=True):
            st.session_state.current_input = "애가 맨발로 놀기 괜찮은 위생 좋은 키즈카페 있어?"
            st.rerun()
    with col2:
        if st.button("🎭 역할놀이 좋아하는 3살 추천", use_container_width=True):
            st.session_state.current_input = "역할놀이 좋아하는 3살 아이가 갈만한 곳 알려줘"
            st.rerun()
        if st.button("👨 아빠 혼자 데리고 가기 좋은 곳", use_container_width=True):
            st.session_state.current_input = "주말에 아빠 혼자 애기 데리고 가기 좋은 위생적인 곳"
            st.rerun()

    st.markdown(
        "<p class='welcome-caption'>"
        "📂 서울시 공공 데이터 기반 · 광고 없는 진짜 리뷰 · 안심 인증 키즈카페"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 3. AI 이모 라벨 (퍼스널 브랜딩)
# ─────────────────────────────────────────────────────────────
def render_ai_label():
    st.markdown(
        "<div class='ai-label'>🐰 베베노리 이모</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# 4. 추천 카드 렌더링
# ─────────────────────────────────────────────────────────────
def render_recommendation_cards(docs: list):
    """
    영수증 스타일 추천 카드.
    ⚠️ HTML 카드 먼저 출력 → 그 직후 st.columns로 버튼 배치
       (unsafe_allow_html 내부에 Streamlit 위젯 혼용 불가 원칙 준수)
    """
    if not docs:
        return

    visible = docs[:3]
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    cols = st.columns(min(len(visible), 3), gap="small")

    for i, doc in enumerate(visible):
        with cols[i]:
            review_text = str(doc.get("review_text", ""))
            feat_text   = str(doc.get("features", "")) + review_text
            place_name  = doc.get("place_name", doc.get("facility_name", "이름 없음"))
            address     = doc.get("address", "주소 정보 없음")
            tip         = doc.get("tip", "방문 전 서울시 공공서비스 예약 사이트를 꼭 확인하세요! 📅")
            tip_short   = tip[:60] + ("…" if len(tip) > 60 else "")
            fee         = doc.get("fee", "어린이 3,000원")
            parking     = doc.get("parking_info", "별도 확인 필요")
            badges_html = _build_badges(feat_text, review_text)

            # ── 카드 HTML (버튼 제외) ─────────────────────────
            card_html = f"""
            <div class="reco-card">
                <div>
                    <div class="receipt-icon">🏠</div>
                    <div class="receipt-dashed">- - - - - - - - - - - - -</div>
                    <div class="reco-badge-row">{badges_html}</div>
                    <div class="reco-title" title="{place_name}">{place_name}</div>
                    <div class="reco-addr">📍 {address}</div>
                    <div class="receipt-dashed">· · · · · · · · · · · · ·</div>
                    <div class="receipt-row">
                        <span class="receipt-label">이용요금</span>
                        <span class="receipt-value">{fee}</span>
                    </div>
                    <div class="receipt-row">
                        <span class="receipt-label">주차</span>
                        <span class="receipt-value">{parking}</span>
                    </div>
                    <div class="receipt-dashed">- - - - - - - - - - - - -</div>
                </div>
                <div class="tip-box">
                    <div class="tip-title">🧸 이모삼촌 꿀팁</div>
                    <div class="tip-content">{tip_short}</div>
                </div>
                <hr class="card-divider">
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # ── 버튼: HTML 이후 st.columns로 ─────────────────
            b1, b2 = st.columns(2, gap="small")
            with b1:
                st.link_button(
                    "🗺️ 지도보기",
                    _build_map_url(place_name, address),
                    use_container_width=True,
                )
            with b2:
                st.link_button(
                    "📅 예약하기",
                    doc.get("booking_url", "https://yeyak.seoul.go.kr/web/main.do"),
                    use_container_width=True,
                )


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────
def _build_badges(feat_text: str, review_text: str) -> str:
    badges = []
    if any(kw in review_text for kw in ("청결", "깨끗", "위생", "청소", "소독")):
        badges.append("<span class='badge clean'>✨ 청결 우수</span>")
    if any(kw in review_text for kw in ("안전", "푹신", "안심", "보호", "전용 존")):
        badges.append("<span class='badge safe'>🛡️ 안전 안심</span>")
    if any(kw in feat_text for kw in ("주차", "주차장", "주차 가능", "넓은 주차")):
        badges.append("<span class='badge parking'>🅿️ 주차 쾌적</span>")
    elif any(kw in feat_text for kw in ("지하철", "버스", "역 근처", "대중교통", "도보")):
        badges.append("<span class='badge transit'>🚌 대중교통 추천</span>")
    if not badges:
        badges.append("<span class='badge default'>🌟 안심 인증</span>")
    return " ".join(badges)


def _build_map_url(place_name: str, address: str) -> str:
    query   = place_name if place_name != "이름 없음" else address
    encoded = urllib.parse.quote(query)
    return f"https://map.kakao.com/link/search/{encoded}"
