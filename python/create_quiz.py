import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import json

def create_quiz_with_gemini(full_text: str, num_quiz: int, difficulty: str) -> dict :

    #.env 파일 load
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # api_key가 존재하지 않을 시 ValueError 예외 강제 발생
        raise ValueError("⛔ 오류 : GOOGLE_API_KEY가 .env 파일에 설정되지 않음")
    # api_key 설정하여 gemini API 접근 권한 부여
    genai.configure(api_key=api_key)

    #gemini 모델 설정
    model = genai.GenerativeModel('gemini-2.5-flash')

    # prompt에 넣을 난이도에 대한 설명
    difficulty_instructions = {
        "쉬움": "텍스트에 명시적으로 언급된 핵심 용어의 정의나 사실 관계를 확인하는 문제를 출제해줘.",
        "보통": "텍스트의 내용을 종합하거나 문맥을 파악해야 풀 수 있는 문제를 출제해줘.",
        "어려움": "텍스트 내용을 바탕으로 추론하거나, 다른 상황에 적용하는 응용 문제를 출제해줘. 오답 보기는 매우 그럴 듯하게 만들어줘."
    }

    # prompt 설정
    prompt = f"""
    당신은 주어진 텍스트 내용을 바탕으로 학습 내용을 점검할 수 있는 퀴즈를 만드는 전문 출제자입니다.
    아래 텍스트를 분석하여, [난이도: {difficulty}] 수준의 객관식 문제 {num_quiz}개를 만들어주세요.

    [문제 생성 가이드라인]
    {difficulty_instructions.get(difficulty, difficulty_instructions[difficulty])}

    [출력 형식]
    - 반드시 아래와 같은 JSON 형식으로만 응답해야 합니다.
    - 다른 설명이나 인사말, 코드 블록 마크(```json) 등은 절대 추가하지 마세요.
    - 모든 문자열 값은 큰따옴표(")로 감싸야 합니다.

    {{
      "quizzes": [
        {{
          "quiz": "문제 내용...",
          "options": ["보기 1", "보기 2", "보기 3", "보기 4"],
          "answer": "정답에 해당하는 보기 내용",
          "explanation": "왜 이 보기가 정답인지에 대한 간단한 해설 ("
        }}
      ]
    }}

    --- 제공된 텍스트 ---
    {full_text}
    """
    try:
        # gemini API 응답 받아오기
        logging.info(f"{difficulty} 난이도 퀴즈 {num_quiz}개 생성 중...")
        response = model.generate_content(prompt)
        response_text = response.text

        # API 응답에서 불필요한 코드 블록 마크다운을 제거하는 전처리 과정
        # 1. 앞뒤 공백 및 줄바꿈 제거
        clean_text = response_text.strip()
        # 2. 시작 부분의 ```json 제거
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]  # "```json" 다음부터 슬라이싱
        # 3. 끝 부분의 ``` 제거
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]  # 맨 뒤 3글자 제외하고 슬라이싱

        # 이제 깨끗해진 텍스트로 JSON 파싱 시도
        logging.info(f"✅ 성공 : {difficulty} 난이도 퀴즈 {num_quiz}개 생성 완료")
        quiz_data = json.loads(clean_text)
        return quiz_data

    except Exception as e:
        logging.error("⛔ 오류 : Gemini API 호출 또는 JSON 파싱 중 예외 발생")
        logging.error(f"상세 오류 : {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print("--- API Raw Response ---")
            print(response.text)
            print("------------------------")
        return {}