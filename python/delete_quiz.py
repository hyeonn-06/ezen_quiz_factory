import os
import logging
import shutil

PERSIST_BASE_DIRECTORY = "./persistent_vector_stores"

def delete_vector_db(quiz_no:int):

    target_directory = os.path.join(PERSIST_BASE_DIRECTORY, str(quiz_no))

    logging.info(f"--- 벡터 DB 삭제 시도: '{target_directory}' ---")

    if os.path.exists(target_directory):
        try:
            # 디렉토리와 그 안의 모든 내용을 재귀적으로 삭제
            shutil.rmtree(target_directory)
            logging.info(f"✔️ 성공: 벡터 DB 디렉토리 '{target_directory}' 삭제 성공")
            return "삭제 성공"
        except OSError as e:
            logging.error(f"⛔ 오류: 벡터 DB 디렉토리 '{target_directory}' 삭제 중 오류 발생: {e}")
    else:
        raise ValueError(f"⛔ 오류 : 벡터 DB 디렉토리 '{target_directory}'가 존재하지 않음")
    return "삭제 실패"
    logging.info(f"--- 벡터 DB 삭제 시도 종료: '{target_directory}' ---")