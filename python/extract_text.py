# extract_text.py

import fitz #PyMuPDF
import os
import logging

# pdf 파일로부터 텍스트를 추출하는 함수
def extract_text_from_pdf(input_file_path: str) -> str:

    # 해당 경로에 pdf 파일 존재 x
    if not os.path.exists(input_file_path):
        logging.error(f"⛔ 오류 : '{input_file_path}' 경로에서 파일을 찾을 수 없음")
        return ""

    # 해당 경로에 pdf 파일 존재 o
    try:
        # pdf 파일 열기
        document = fitz.open(input_file_path)
        full_text = ""

        # 텍스트 추출
        logging.info(f"총 {document.page_count} 페이지의 텍스트 추출 중...")
        for page_num in range(document.page_count):
            page = document.load_page(page_num) # 한 페이지씩 불러오기
            full_text += page.get_text("text") # 해당 페이지에서 텍스트만 추출
            full_text += "\n ----- 페이지 구분 ----- \n" # 페이지 구분을 위해 추가

        logging.info(f"✅ 성공 : '{input_file_path}' 경로의 파일로부터 텍스트 추출 완료")
        return full_text

    # 예외 발생
    except Exception as e :
        logging.error("⛔ 오류 : PDF 파일 처리 중 예외 발생")
        logging.error(f"상세 오류 : {e}")
        return ""