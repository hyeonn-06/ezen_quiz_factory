import os
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_text_splitters import KonlpyTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging

# --- 전역 설정 (상수로 분리하여 관리) ---
# 벡터 DB를 저장할 최상위 기본 디렉토리
PERSIST_BASE_DIRECTORY = "./persistent_vector_stores"
# 사용할 HuggingFace 임베딩 모델의 이름
EMBEDDING_MODEL_NAME = "nlpai-lab/KURE-v1"

# 프로그램 시작 시 기본 디렉토리가 없으면 생성
os.makedirs(PERSIST_BASE_DIRECTORY, exist_ok=True)


def question_and_answer(quiz_no: int, question: str, full_text: str | None = None) -> str:
    logging.info(f"--- 퀴즈 No. {quiz_no} / 질문: '{question}'에 대한 프로세스 시작 ---")

    # 1. API 키 및 환경 변수 로드
    logging.info("환경 변수 로드 및 API 키 확인 중...")
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    if not google_api_key:
        # GOOGLE_API_KEY가 존재하지 않을 시 ValueError 예외 강제 발생
        raise ValueError("⛔ 오류 : GOOGLE_API_KEY가 .env 파일에 설정되지 않음")
    if not hf_token:
        # HF_TOKEN이 존재하지 않을시 ValueError 예외 강제 발생
        logging.warning("⛔ 오류 : HF_TOKEN .env 파일에 설정되지 않음")

    # 2. 이 퀴즈의 고유 벡터 DB 경로 정의
    persist_directory = os.path.join(PERSIST_BASE_DIRECTORY, str(quiz_no))
    logging.info(f"   - 해당 퀴즈의 벡터 DB 경로 : '{persist_directory}'")

    # 3. 임베딩 모델 준비 (DB 생성과 로드 양쪽 모두에 필요)
    logging.info("임베딩 모델 준비 중...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"   - 임베딩을 위해 사용하는 디바이스: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    logging.info(f"   - 임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 완료.")

    # 4. 벡터 DB 로드 또는 생성
    logging.info("벡터 DB 로드 시도...")
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # --- IF: DB가 존재할 경우 (가장 일반적인 빠른 경로) ---
        logging.info(f"   - 벡터 DB가 존재. 디스크에서 로드 시도")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        logging.info("   - DB 로드 성공.")
    else:
        # --- ELSE: DB가 없을 경우 (최초 1회만 실행되는 경로) ---
        logging.warning(f"   - 벡터 DB가 존재하지 않음. 새로 생성 시도")

        if not full_text:
            raise ValueError(f"⛔ 오류: DB 생성을 위해 'full_text'가 필요.")

        logging.info("   - 텍스트 분할(Chunking) 시작...")
        text_splitter = KonlpyTextSplitter(chunk_size=500, chunk_overlap=50)
        input_doc = Document(page_content=full_text)
        docs = text_splitter.split_documents([input_doc])
        logging.info(f"   - 문서가 {len(docs)}개의 조각으로 분할되었습니다.")

        logging.info("   - 벡터화 및 영구 저장 시작...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logging.info(f"   - 퀴즈 No. {quiz_no}의 벡터 DB 생성 및 영구 저장 완료.")

    # 5. RAG 체인 생성
    logging.info("RAG 체인 생성 중...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=google_api_key,
        convert_system_message_to_human=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}  # 검색할 문서의 개수를 3개로 지정 (튜닝 가능)
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 가장 간단한 방식. 문서들을 모두 컨텍스트에 넣음.
        retriever=retriever,
        return_source_documents=False  # 답변 생성에 사용된 원본 문서를 반환할지 여부
    )
    logging.info("   - RAG 체인 생성 완료.")

    # 6. 체인 실행 및 결과 반환
    logging.info(f"질문에 대한 답변 생성 실행...")
    result_dict = qa_chain.invoke(question)
    answer = result_dict.get('result', '답변 키를 찾지 못했습니다.')
    logging.info("   - 답변 생성 완료.")
    logging.info(f"--- 퀴즈 No. {quiz_no}에 대한 프로세스 종료 ---\n")

    return answer