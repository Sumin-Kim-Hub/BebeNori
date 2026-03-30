# =============================================================================
# 파일 위치: 03_evaluation/evaluate_cli.py
# 역할: GPT-4o-mini를 judge로 사용해 RAG 파이프라인 성능을 평가하는 스크립트.
#
# 평가 흐름:
#   Step 1 — context에 답변 가능한 정보가 있는지 판단 (answerable 여부)
#   Step 2A — answerable=true  → 5개 지표 평가
#             (faithfulness / context_recall / context_precision /
#              answer_relevance / tone_friendliness)
#   Step 2B — answerable=false → 2개 지표 평가
#             (unanswerable_response / tone_friendliness)
#
# 사용법:
#   cd 프로젝트_루트/
#   python 03_evaluation/evaluate_cli.py
#
# 환경 변수:
#   OPENAI_API_KEY=sk-...   (.env 또는 export 로 설정)
# =============================================================================

# =============================================================================
# ★ 팀원 설정 영역 — 이 변수들만 수정해서 평가를 실행하세요 ★
# =============================================================================

EVAL_VERSION = "eval_history_3" #새로운 version 저장시 'eval_history_숫자'로 변경 부탁드립니다
TEST_QUERY   = "주말에 18개월 아이랑 강남에서 갈 수 있는 키즈카페 추천해줘"
TOP_K        = 4

BATCH_QUERIES: list[str] = [
    "강남에 갈만한 키즈카페 추천해줘",
    "20대 남성이 갈만한 키즈카페 추천",
    "경기도 키즈카페 추천해줘",
    "성북동 근처에서 주차 무료인 곳",
    "첫째는 25개월이고 둘째는 10개월인데, 두 아이가 모두 즐겁게 놀 수 있는 지점을 추천해줘",
    "천호역 근처 키즈카페 추천해줘",
    "종로구에서 일요일에도 갈 수 있는 키즈카페가 있어?",
    "서울형 키즈카페 강남10호점은 어디에 있어?",
    "마포구에 있으면서 4살 아이가 놀기 좋고, 주차가 2대 이상 가능하며 입장료가 만 원 이하인 곳 찾아줘.",
    "보호자 양말 착용이 필수가 아니거나, 현장에서 양말을 무료로 대여해주는 동작구 키즈카페 혹시 있을까?",
    "애들 맡겨놓고 남편이랑 오랜만에 데이트할 건데, 강남역 근처에 분위기 좋고 조용한 오마카세 예약되는 곳 추천해 줘.",
    "아이가 미끄럼틀을 타고 싶어하는데, 미끄럼틀이 있는 키즈카페가 있어?",
    "아이가 자동차랑 기차에 완전 빠져있어. 자동차랑 기차 장난감이 있는 곳 추천 노원구쪽에ㅈ장난감ㅁ많고 부모도쉴수잇는곳어디야",
    "12개월 아기가 열이 39도까지 나는데, 챔프 빨간색이랑 파란색 교차 복용 몇 시간 간격으로 해야 돼?",
    "강남구에서 강아지나 고양이 등 반려동물 동반 입장이 가능한 영유아 전용 대형 키즈카페 있어?"
]

JUDGE_MODEL = "gpt-4o-mini"

# =============================================================================
# 이하 평가 로직 (수정 불필요)
# =============================================================================

import sys
import time
import json
import textwrap
import datetime
from pathlib import Path

_EVAL_DIR    = Path(__file__).parent
_PROJECT_DIR = _EVAL_DIR.parent
_MODELS_DIR  = _PROJECT_DIR / "02_model_logic"
_DATA_PREP   = _PROJECT_DIR / "01_data_prep"

for _p in [str(_MODELS_DIR), str(_DATA_PREP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_PROJECT_DIR / ".env")

import openai
from data_loader import load_places, load_dev
from backend_core import (
    load_or_create_vectorstore,
    get_llm_chain,
    rag_retrieve,
    build_context,
    gen_answer,
)

# =============================================================================
# Judge 프롬프트
# =============================================================================

_JUDGE_USER_TMPL = """\
[사용자 질문]
{query}

[검색된 Context]
{context}

[모델 답변]
{answer}
"""

# ── Step 1: answerable 판단 ──────────────────────────────────────────────────
_ANSWERABLE_SYSTEM = """\
당신은 RAG 시스템의 평가자입니다.
주어진 Context만을 근거로, 사용자의 질문에 답하기 위한 정보가 충분히 존재하는지 판단하세요.

반드시 아래 JSON 형식만 출력하고, 다른 텍스트는 절대 포함하지 마세요.

{"answerable": true 또는 false, "reason": "한 문장 이유"}

판단 기준:
- true  : Context에 질문에 답할 수 있는 관련 정보가 하나 이상 존재한다
- false : Context에 질문과 관련된 정보가 전혀 없거나 극히 부족하다
"""

# ── Step 2-A: answerable=true ────────────────────────────────────────────────
_JUDGE_SYSTEM_ANSWERABLE = """\
당신은 RAG(검색 증강 생성) 시스템의 답변 품질을 평가하는 전문 평가자입니다.
주어진 입력을 바탕으로 아래 5개 지표를 각각 0.0 ~ 1.0 사이의 점수와 한 문장 이유로 평가하세요.

반드시 아래 JSON 형식만 출력하고, 다른 텍스트는 절대 포함하지 마세요.

{
  "faithfulness":      {"score": 0.0, "reason": "string"},
  "context_recall":    {"score": 0.0, "reason": "string"},
  "context_precision": {"score": 0.0, "reason": "string"},
  "answer_relevance":  {"score": 0.0, "reason": "string"},
  "tone_friendliness": {"score": 0.0, "reason": "string"}
}

각 지표 설명:
- faithfulness      : 답변이 context에 없는 정보를 지어내지 않았는가 (1.0 = 환각 없음)
- context_recall    : 질문에 답하는 데 필요한 정보가 검색된 context에 충분히 존재하는가
- context_precision : 검색된 context 중 실제로 답변에 유용하게 쓰인 비율
- answer_relevance  : 답변이 질문의 의도에 얼마나 직접적으로 답하는가
- tone_friendliness : 말투가 친절하고 자연스러우며 육아 앱 서비스에 적합한가
"""

# ── Step 2-B: answerable=false ───────────────────────────────────────────────
_JUDGE_SYSTEM_UNANSWERABLE = """\
당신은 RAG 시스템의 평가자입니다.
Context에 질문에 답할 정보가 없는 상황입니다.
이 경우 모델이 "정보를 찾을 수 없다"는 취지의 답변을 명확하게 했는지 평가하세요.

반드시 아래 JSON 형식만 출력하고, 다른 텍스트는 절대 포함하지 마세요.

{
  "unanswerable_response": {"score": 0.0, "reason": "string"},
  "tone_friendliness":     {"score": 0.0, "reason": "string"}
}

각 지표 설명:
- unanswerable_response : 모델이 정보 없음을 명확하고 솔직하게 안내했는가
                          (1.0 = 없다고 정확히 안내, 0.0 = 없는 정보를 지어내서 답변)
- tone_friendliness     : 정보가 없는 상황에서도 친절하고 자연스럽게 안내했는가
"""

# 지표 라벨 (출력용)
METRIC_LABELS_ANSWERABLE = {
    "faithfulness":      "Faithfulness      (환각 없음)",
    "context_recall":    "Context Recall    (필요 정보 존재)",
    "context_precision": "Context Precision (유용한 비율)",
    "answer_relevance":  "Answer Relevance  (질문 적합성)",
    "tone_friendliness": "Tone Friendliness (말투 친절도)",
}
METRIC_LABELS_UNANSWERABLE = {
    "unanswerable_response": "Unanswerable Response (정보 없음 안내)",
    "tone_friendliness":     "Tone Friendliness     (말투 친절도)",
}


def _call_judge(query: str, context: str, answer: str) -> dict:
    client   = openai.OpenAI()
    user_msg = _JUDGE_USER_TMPL.format(
        query=query, context=context, answer=answer,
    )

    # Step 1: answerable 판단
    resp1 = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _ANSWERABLE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
    )
    raw1       = resp1.choices[0].message.content.strip()
    raw1       = raw1.replace("```json", "").replace("```", "").strip()
    step1      = json.loads(raw1)
    answerable = bool(step1.get("answerable", True))

    # Step 2: 분기 평가
    system2 = (
        _JUDGE_SYSTEM_ANSWERABLE
        if answerable
        else _JUDGE_SYSTEM_UNANSWERABLE
    )
    resp2 = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system2},
            {"role": "user",   "content": user_msg},
        ],
    )
    raw2  = resp2.choices[0].message.content.strip()
    raw2  = raw2.replace("```json", "").replace("```", "").strip()
    step2 = json.loads(raw2)

    return {
        "answerable":        answerable,
        "answerable_reason": step1.get("reason", ""),
        "scores":            step2,
    }


# =============================================================================
# RAG 실행
# =============================================================================

def run_single(
    df, dev_df, vectorstore, chain,
    query: str,
    verbose: bool = True,
) -> dict:
    t0 = time.perf_counter()

    pids = rag_retrieve(vectorstore, query, df=df, n=TOP_K)
    if not pids:
        print("⚠️  검색 결과 없음")
        return {}

    ctx     = build_context(df, dev_df, pids, query=query)
    answer  = gen_answer(chain, query, ctx)
    elapsed = round(time.perf_counter() - t0, 2)

    print("   🔍 GPT judge 평가 중...", end="", flush=True)
    try:
        eval_result = _call_judge(query, ctx, answer)
    except Exception as e:
        eval_result = {"error": str(e)}
    print(" 완료")

    result = {
        "query":       query,
        "retrieved":   pids,
        "context":     ctx,
        "answer":      answer,
        "elapsed_sec": elapsed,
        "eval":        eval_result,
    }

    if verbose:
        _print_result(result)

    return result


# =============================================================================
# 출력 포맷
# =============================================================================

def _separator(char: str = "─", width: int = 68) -> str:
    return char * width


def _fmt_score(score) -> str:
    try:
        s      = float(score)
        filled = round(s * 10)
        bar    = "█" * filled + "░" * (10 - filled)
        return f"[{bar}] {s:.2f}"
    except Exception:
        return str(score)


def _print_scores(scores: dict, labels: dict) -> None:
    for key, label in labels.items():
        item   = scores.get(key, {})
        score  = item.get("score", "N/A")
        reason = item.get("reason", "")
        print(f"    {label}")
        print(f"    {_fmt_score(score)}")
        wrapped = textwrap.fill(
            reason, width=62,
            initial_indent="    💬 ",
            subsequent_indent="       ",
        )
        print(wrapped)
        print()


def _print_result(r: dict) -> None:
    sep  = _separator()
    thin = _separator("·")

    print(f"\n{sep}")
    print(f"  📝 질문      : {r['query']}")
    print(f"  🗄  검색 IDs  : {r['retrieved']}")
    print(f"  ⏱  답변 속도  : {r['elapsed_sec']}초")
    print(thin)

    print("  📨 모델 답변:")
    for line in r["answer"].split("\n"):
        print(f"      {line}")
    print(thin)

    eval_data = r.get("eval", {})

    if "error" in eval_data:
        print(f"  ❌ 평가 오류: {eval_data['error']}")
        print(sep)
        return

    answerable        = eval_data.get("answerable", True)
    answerable_reason = eval_data.get("answerable_reason", "")
    scores            = eval_data.get("scores", {})

    # answerable 판정 표시
    tag = "✅ 답변 가능" if answerable else "⛔ 답변 불가 (데이터 없음)"
    print(f"  🔎 Answerable : {tag}")
    if answerable_reason:
        wrapped = textwrap.fill(
            answerable_reason, width=62,
            initial_indent="              ",
            subsequent_indent="              ",
        )
        print(wrapped)
    print()

    print("  📊 평가 결과:")
    print()
    if answerable:
        _print_scores(scores, METRIC_LABELS_ANSWERABLE)
    else:
        _print_scores(scores, METRIC_LABELS_UNANSWERABLE)

    print(sep)


# =============================================================================
# 배치 요약 출력
# =============================================================================

def _print_batch_summary(results: list[dict]) -> None:
    sep  = _separator("=")
    thin = _separator("─")

    # ── 집계 ────────────────────────────────────────────────────────
    answerable_totals   = {k: [] for k in METRIC_LABELS_ANSWERABLE}
    unanswerable_totals = {k: [] for k in METRIC_LABELS_UNANSWERABLE}

    rows = []
    for i, r in enumerate(results, 1):
        elapsed    = r.get("elapsed_sec", 0)
        eval_data  = r.get("eval", {})
        answerable = eval_data.get("answerable", True)
        scores     = eval_data.get("scores", {})
        tag        = "✅" if answerable else "⛔"
        labels     = METRIC_LABELS_ANSWERABLE if answerable else METRIC_LABELS_UNANSWERABLE
        totals     = answerable_totals if answerable else unanswerable_totals

        scores_str = ""
        for k in labels:
            s = scores.get(k, {}).get("score", None)
            if s is not None:
                totals[k].append(float(s))
                scores_str += f"{s:.2f} "
            else:
                scores_str += " N/A "

        rows.append((i, tag, r["query"], elapsed, scores_str))

    # ── 출력 ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  📊 배치 평가 종합 요약")
    print(thin)

    for i, tag, query, elapsed, scores_str in rows:
        print(f"  [{i:02}] {tag} {query[:36]:<36}  ⏱{elapsed}초")
        print(f"        점수: {scores_str}")

    # ── 답변 가능 케이스 평균 ────────────────────────────────────────
    answerable_count = sum(
        1 for r in results if r.get("eval", {}).get("answerable", True)
    )
    if answerable_count:
        print(thin)
        print(f"  📈 답변 가능 케이스 ({answerable_count}건) 평균:")
        print()
        for k, label in METRIC_LABELS_ANSWERABLE.items():
            vals = answerable_totals[k]
            if vals:
                print(f"    {label}")
                print(f"    {_fmt_score(sum(vals) / len(vals))}")
            else:
                print(f"    {label}  —  데이터 없음")
            print()

    # ── 답변 불가 케이스 평균 ────────────────────────────────────────
    unanswerable_count = sum(
        1 for r in results if not r.get("eval", {}).get("answerable", True)
    )
    if unanswerable_count:
        print(thin)
        print(f"  📈 답변 불가 케이스 ({unanswerable_count}건) 평균:")
        print()
        for k, label in METRIC_LABELS_UNANSWERABLE.items():
            vals = unanswerable_totals[k]
            if vals:
                print(f"    {label}")
                print(f"    {_fmt_score(sum(vals) / len(vals))}")
            else:
                print(f"    {label}  —  데이터 없음")
            print()

    # ── 시간 요약 ───────────────────────────────────────────────────
    total_time = sum(r.get("elapsed_sec", 0) for r in results)
    avg_time   = round(total_time / len(results), 2) if results else 0
    print(thin)
    print(f"  ⏱  총 소요시간: {round(total_time, 2)}초  / 평균: {avg_time}초")
    print(f"{sep}\n")


# =============================================================================
# 평가 결과 저장
# =============================================================================

def save_eval_history(results: list[dict]):
    """평가 결과를 EVAL_VERSION 이름의 json 파일로 저장합니다."""

    safe_version = EVAL_VERSION.replace("/", "-").replace("\\", "-").replace(":", "-")
    history_file = _EVAL_DIR / f"{safe_version}.json"

    if history_file.exists():
        print(f"  ⚠️  '{history_file.name}' 파일이 이미 존재합니다. 덮어씁니다.")

    answerable_res = [r for r in results if r.get("eval", {}).get("answerable", True)]
    avg_scores = {}
    metrics = ["faithfulness", "context_recall", "context_precision", "answer_relevance", "tone_friendliness"]

    if answerable_res:
        for m in metrics:
            vals = [r["eval"]["scores"].get(m, {}).get("score", 0) for r in answerable_res]
            avg_scores[m] = round(sum(vals) / len(vals), 3)

    unanswerable_res = [r for r in results if not r.get("eval", {}).get("answerable", True)]
    if unanswerable_res:
        u_vals = [r["eval"]["scores"].get("unanswerable_response", {}).get("score", 0) for r in unanswerable_res]
        avg_scores["unanswerable_response"] = round(sum(u_vals) / len(u_vals), 3)

        t_vals = [r["eval"]["scores"].get("tone_friendliness", {}).get("score", 0) for r in unanswerable_res]
        avg_scores["unanswerable_tone_friendliness"] = round(sum(t_vals) / len(t_vals), 3)

    session = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": EVAL_VERSION,
        "queries_count": len(results),
        "average_scores": avg_scores,
        "individual_results": [
            {
                "query": r["query"],
                "answer": r["answer"],
                "answerable": r.get("eval", {}).get("answerable", True),
                "scores": r.get("eval", {}).get("scores", {})
            } for r in results
        ]
    }

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=4)

    print(f"\n  💾 [Version {EVAL_VERSION}] '{history_file.name}' 저장 완료!")


# =============================================================================
# main
# =============================================================================

def main():
    print(_separator("="))
    print("  베베노리 RAG 파이프라인 — 평가 CLI (with GPT Judge)")
    print(_separator("="))

    print("\n▶ 데이터 로드 중...")
    df     = load_places()
    dev_df = load_dev()
    print(f"   장소: {len(df)}개  발달: {len(dev_df)}행")

    print("▶ 벡터 스토어 준비 중...")
    vectorstore = load_or_create_vectorstore(df, force_rebuild=True)

    print("▶ LLM 체인 초기화 중...")
    chain = get_llm_chain()

    all_results = []

    # ── 단일 평가 ──────────────────────────────────────────────────────
    print(f"\n{_separator()}")
    print("  [단일 테스트] TEST_QUERY 평가")
    r_single = run_single(df, dev_df, vectorstore, chain, TEST_QUERY)
    if r_single:
        all_results.append(r_single)

    # ── 배치 평가 ──────────────────────────────────────────────────────
    if BATCH_QUERIES:
        print(f"\n{_separator()}")
        print(f"  [배치 테스트] {len(BATCH_QUERIES)}개 쿼리 순차 실행")
        batch_results = []
        for bq in BATCH_QUERIES:
            print(f"\n  ▷ 실행 중: {bq}")
            r = run_single(df, dev_df, vectorstore, chain, bq)
            if r:
                batch_results.append(r)
                all_results.append(r)

        if batch_results:
            _print_batch_summary(batch_results)

    # ── 평가 기록 파일로 저장 ──────────────────────────────────────────
    if all_results:
        save_eval_history(all_results)


if __name__ == "__main__":
    main()