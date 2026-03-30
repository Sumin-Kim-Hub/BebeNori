# 수정일: 26/03/27/11:55 (진국)
# 이유: 필터링 조건에서 컨텍스트를 추출하던 문제를 수정. 유저 쿼리에만 의존하도록
# =============================================================================
# 파일 위치: 프로젝트 최상위폴더
# 역할: 자신이 원하는 쿼리를 입력해서 출력을 테스트해볼 수 있도록 하는 코드

# test.py
import sys
from pathlib import Path

# 1. 현재 파일(test_run.py)의 위치를 기준으로 프로젝트 루트 경로 잡기
_PROJECT_DIR = Path(__file__).resolve().parent

# 2. 파이썬에게 "이 폴더들 안에도 코드가 있어!"라고 알려주기
# 폴더 이름이 '01_data_prep' 인지 다시 한번 꼭 확인하세요!
data_prep_path = str(_PROJECT_DIR / "01_data_prep")
model_logic_path = str(_PROJECT_DIR / "02_model_logic")

if data_prep_path not in sys.path:
    sys.path.insert(0, data_prep_path)
if model_logic_path not in sys.path:
    sys.path.insert(0, model_logic_path)

# 3. 임포트
try:
    from data_loader import load_places, load_dev
    from backend_core import get_llm_chain, gen_answer, load_or_create_vectorstore, rag_retrieve, build_context
    print("✅ 모든 모듈 임포트 성공! 엔진을 시작합니다.")
except ImportError as e:
    print(f"❌ 임포트 에러 발생: {e}")
    print("💡 힌트: 폴더 이름이 '01_data_prep'과 '02_model_logic'이 맞는지 확인해 보세요.")
    sys.exit(1) # 임포트 실패 시 여기서 실행 중단 (NameError 방지)

def main():
    print("베베노리 AI 엔진 가동 중... 잠시만 기다려주세요.")

    # 3. 데이터 로드 및 벡터 DB 구축 (준비 작업)
    df = load_places()
    dev_df = load_dev()
    vectorstore = load_or_create_vectorstore(df)
    chain = get_llm_chain()

    print("준비 완료! 질문을 던져봅니다.")
    print("-" * 50)

    # 4. 가상 질문 설정 (테스트용)
    user_query = "20개월 아이랑 갈만한 동작구 키즈카페 추천해줘"

    # 5. RAG 파이프라인 가동
    # (1) 검색
    pids = rag_retrieve(vectorstore, user_query, n=3)
    
    # (2) 컨텍스트 생성 (장소 정보 + 발달 정보 합치기)
    context = build_context(df, dev_df, pids, query=user_query)
    
    # (3) 답변 생성
    answer = gen_answer(chain, user_query, context)

    # 6. 결과 출력
    print(f"\n[나의 질문]: {user_query}")
    print(f"\n[베베노리 이모삼촌의 답변]:\n")
    print(answer)
    print("-" * 50)

if __name__ == "__main__":
    main()
