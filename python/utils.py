
def format_quizzes_to_string(quizzes_data: dict) -> str:
    output_lines = []

    # 'quizzes' 키가 있고 그 값이 리스트인지 확인
    if 'quizzes' in quizzes_data and isinstance(quizzes_data['quizzes'], list):
        for i, question_obj in enumerate(quizzes_data['quizzes']):
            output_lines.append(f"--- 문제 {i + 1} ---")

            # 퀴즈 질문
            quiz_text = question_obj.get('quiz', '질문 없음').replace('\n', ' ').strip()
            output_lines.append(f"Q: {quiz_text}")

            # 객관식 문항
            options = question_obj.get('options', [])
            if options:
                for j, option in enumerate(options):
                    output_lines.append(f"  {j + 1}. {option.replace('\n', ' ').strip()}")

            # 정답
            answer = question_obj.get('answer', None)
            if answer:
                output_lines.append(f"정답: {answer.replace('\n', ' ').strip()}")

            # 해설
            explanation = question_obj.get('explanation', None)
            if explanation:
                output_lines.append(f"해설: {explanation.replace('\n', ' ').strip()}")

            output_lines.append("\n")  # 각 문제 사이에 빈 줄 추가

    return "\n".join(output_lines).strip()  # 마지막에 추가된 빈 줄 제거를 위해 strip()
