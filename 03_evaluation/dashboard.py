import streamlit as st
import pandas as pd
import json
import os
import glob

# 페이지 기본 설정
st.set_page_config(page_title="베베노리 RAG 평가 대시보드", page_icon="🧸", layout="wide")

st.title("🧸 베베노리 RAG 파이프라인 성능 대시보드")
st.markdown("`evaluate_cli.py`를 통해 누적된 버전별 평가 지표와 개별 답변을 분석합니다.")

# =============================================================================
# 1. 파일 자동 탐색
# =============================================================================

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

def find_eval_files() -> list[str]:
    """eval_history_*.json 파일을 자동으로 탐색하여 정렬된 리스트로 반환"""
    pattern = os.path.join(EVAL_DIR, "eval_history_*.json")
    files = sorted(glob.glob(pattern))
    return files


# =============================================================================
# 2. 데이터 로드 함수
# =============================================================================

@st.cache_data
def load_files(filepaths: tuple) -> tuple:
    """선택된 파일들을 로드하여 (raw_data, df_summary) 반환"""
    merged_raw = []
    summary_rows = []

    for filepath in filepaths:
        label = os.path.splitext(os.path.basename(filepath))[0]  # 파일명 (확장자 제외)

        if not os.path.exists(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError:
                continue

        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        for item in raw_data:
            item_copy = dict(item)
            item_copy["_source"] = label
            merged_raw.append(item_copy)

            row = {
                "Source": label,
                "Timestamp": item.get("timestamp", ""),
                "Version": item.get("version", "unknown"),
                "Queries": item.get("queries_count", 0),
            }
            row.update(item.get("average_scores", {}))
            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    return merged_raw, df_summary


# =============================================================================
# 3. 사이드바: 파일 선택
# =============================================================================

with st.sidebar:
    st.header("⚙️ 데이터 소스 설정")

    all_files = find_eval_files()
    all_labels = [os.path.splitext(os.path.basename(f))[0] for f in all_files]

    if not all_files:
        st.warning("eval_history_*.json 파일이 없습니다.")
        st.stop()

    # 파일 다중 선택 (기본: 전체 선택)
    selected_labels = st.multiselect(
        "비교할 파일을 선택하세요",
        options=all_labels,
        default=all_labels,
    )

    selected_files = [f for f, l in zip(all_files, all_labels) if l in selected_labels]

    if not selected_files:
        st.warning("최소 1개 이상의 파일을 선택해 주세요.")
        st.stop()

    raw_data, df_summary = load_files(tuple(selected_files))
    st.caption(f"총 {len(selected_files)}개 파일 / {len(df_summary)}개 버전 로드됨")


# =============================================================================
# 4. 대시보드 렌더링
# =============================================================================

if df_summary.empty:
    st.warning("⚠️ 선택한 파일에 데이터가 없습니다.")
else:
    score_keys = [
        col for col in df_summary.columns
        if col not in ["Source", "Timestamp", "Version", "Queries"]
    ]

    tab1, tab2 = st.tabs(["📈 버전별 종합 요약", "🔍 개별 질문 분석"])

    # -------------------------------------------------------------------------
    # 탭 1: 전체 평균 요약
    # -------------------------------------------------------------------------
    with tab1:
        # --- 파일별 최근 버전 지표 ---
        for source_label, group in df_summary.groupby("Source"):
            latest = group.iloc[-1]
            st.subheader(f"📍 [{source_label}] 최근 버전: `{latest['Version']}`")

            cols = st.columns(5)
            for i, key in enumerate(score_keys):
                if key not in latest or pd.isna(latest[key]):
                    continue
                current_score = latest[key]
                delta = None
                if len(group) > 1:
                    prev_score = group.iloc[-2].get(key)
                    if prev_score is not None and pd.notna(prev_score):
                        delta = round(current_score - prev_score, 3)
                cols[i % 5].metric(label=key, value=f"{current_score:.2f}", delta=delta)

        st.divider()

        # --- 추이 차트: 지표별로 모든 파일을 하나의 선으로 연결 ---
        st.subheader("📈 전체 평균 성능 변화 추이")
        import plotly.graph_objects as go

        fig = go.Figure()

        if len(selected_files) == 1:
            # 단일 파일: 버전 순서대로 지표별 선 연결
            for key in score_keys:
                if key not in df_summary.columns:
                    continue
                valid = df_summary[["Version", key]].dropna()
                fig.add_trace(go.Scatter(
                    x=valid["Version"],
                    y=valid[key],
                    mode="lines+markers",
                    name=key,
                    line=dict(width=2),
                    marker=dict(size=8),
                ))
        else:
            # 다중 파일: 지표별로 Source(파일) 순서대로 연결
            for key in score_keys:
                x_vals, y_vals = [], []
                for source_label in selected_labels:  # 선택된 파일 순서 유지
                    group = df_summary[df_summary["Source"] == source_label]
                    for _, row in group.iterrows():
                        if key in row and pd.notna(row[key]):
                            x_vals.append(row["Version"])
                            y_vals.append(row[key])
                if x_vals:
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers",
                        name=key,
                        line=dict(width=2),
                        marker=dict(size=8),
                    ))

        fig.update_layout(
            xaxis_title="Version",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            height=450,
            margin=dict(l=40, r=40, t=30, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- 전체 요약 테이블 ---
        st.subheader("📋 전체 요약 데이터")
        st.dataframe(
            df_summary.style.highlight_max(axis=0, subset=score_keys, color="#e6f2ff"),
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # 탭 2: 개별 질문 상세 분석
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🧐 특정 질문에 대한 모델의 답변 변화 추적")

        all_queries = set()
        for item in raw_data:
            for indiv in item.get("individual_results", []):
                all_queries.add(indiv.get("query", ""))

        if not all_queries:
            st.info("개별 질문 데이터가 아직 없습니다.")
        else:
            selected_query = st.selectbox("비교해 볼 질문을 선택하세요:", sorted(all_queries))

            st.divider()

            # 선택된 질문의 버전별 기록 추출
            history_of_query = []
            for item in raw_data:
                version = item.get("version")
                source = item.get("_source", "unknown")
                for indiv in item.get("individual_results", []):
                    if indiv.get("query") == selected_query:
                        history_of_query.append({
                            "Source": source,
                            "Version": version,
                            "Answerable": indiv.get("answerable"),
                            "Answer": indiv.get("answer"),
                            "Scores": indiv.get("scores", {}),
                        })

            # Source별로 그룹화하여 표시
            source_groups = {}
            for record in history_of_query:
                source_groups.setdefault(record["Source"], []).append(record)

            for source_label, records in source_groups.items():
                st.markdown(f"### 📂 {source_label}")
                for record in reversed(records):
                    answerable_icon = "✅" if record["Answerable"] else "⛔"
                    with st.expander(
                        f"📌 버전: {record['Version']} (답변 가능: {answerable_icon})",
                        expanded=True,
                    ):
                        score_cols = st.columns(5)
                        for i, (metric, detail) in enumerate(record["Scores"].items()):
                            score_val = detail.get("score", "N/A")
                            score_cols[i % 5].metric(label=metric, value=score_val)

                        st.markdown("**🤖 모델 답변:**")
                        st.info(record["Answer"])