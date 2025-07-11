import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from extract_text import extract_text_from_pdf
from create_quiz import create_quiz_with_gemini
from utils import format_quizzes_to_string
from chatbot import question_and_answer
from delete_quiz import delete_vector_db

# logging 설정
logging.basicConfig(level=logging.INFO)

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="퀴즈 공장 API 서버",
    description="PDF 파일을 받아 텍스트를 추출하고 퀴즈를 생성하는 API입니다.",
    version="1.0.0",
)

# ----- 요청 모델 정의 (퀴즈 출제) -----
class QuizCreateRequest(BaseModel):
    input_file_path: str
    difficulty : str
    num_quiz : int

# ----- 요청 모델 정의 (퀴즈 삭제) -----
class QuizDeleteRequest(BaseModel):
    quiz_no : int

# -----요청 모델 정의 (챗봇) -----
class ChatBotRequest(BaseModel):
  input_file_path: str
  question : str
  quiz_no : int

# ----- 응답 모델 정의 (퀴즈 출제) -----
class QuizCreateResponse(BaseModel):
    quizzes : str

# ----- 응답 모델 정의 (퀴즈 삭제) -----
class QuizDeleteResponse(BaseModel):
    message : str

# ----- 응답 모델 정의 (챗봇) -----
class ChatBotResponse(BaseModel) :
    answer : str

@app.post("/api/quiz/create", response_model=QuizCreateResponse)
async def api_quiz_create(request_data: QuizCreateRequest):
    logging.info("--- 퀴즈 생성 시작 ---")
    input_file_path = request_data.input_file_path # pdf 파일 경로
    difficulty = request_data.difficulty # 난이도
    num_quiz = request_data.num_quiz # 출제 퀴즈 수

    # pdf 파일에서 텍스트 추출
    full_text = extract_text_from_pdf(input_file_path)

    # 추출한 텍스트를 통해 퀴즈 데이터 생성 (dict 타입)
    quiz_data = create_quiz_with_gemini(full_text, num_quiz, difficulty)

    # 퀴즈 데이터를 문자열로 변환
    quizzes = format_quizzes_to_string(quiz_data)

    return QuizCreateResponse(quizzes=quizzes)

@app.post("/api/quiz/delete", response_model=QuizDeleteResponse)
async def api_quiz_delete(request_data: QuizDeleteRequest):
    logging.info("--- 퀴즈 삭제 시작 ---")
    quiz_no = request_data.quiz_no
    message = delete_vector_db(quiz_no)
    return QuizDeleteResponse(message=message)

@app.post("/api/chatbot", response_model=ChatBotResponse)
async def api_chatbot(request_data: ChatBotRequest):
    input_file_path = request_data.input_file_path # pdf 파일 경로
    question = request_data.question # 질문
    quiz_no = request_data.quiz_no # 퀴즈 번호

    try :
        logging.info("--- 챗봇 시작 ---")

        # pdf 파일에서 텍스트 추출
        full_text = extract_text_from_pdf(input_file_path)

        # RAG 시스템을 통한 질의 응답
        answer = question_and_answer(quiz_no, question, full_text)
        logging.info(f"성공적으로 답변 생성 완료: {answer[:50]}...")
        return ChatBotResponse(answer=answer)

    except Exception as e:
        logging.error(f"⛔ 오류 : 챗봇 과정 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="챗봇 서버 내부 오류 발생")