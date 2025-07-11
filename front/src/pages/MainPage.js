import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

const MainPage = () => {

    const { isLoggedIn } = useAuth();
    const navigate = useNavigate();

    const handleQuizBtnClick = (e) => {
        e.preventDefault();
        if (!isLoggedIn) {
            alert("로그인 후 이용 가능합니다.");
            navigate("/auth/login");
        }
        else {
            navigate("/member/quiz/create");
        }
    }
    return (
        <>
            <div className="jumbotron bg-white"></div>
            <div className="container mt-5">
                <div className="row justify-content-center">
                    <div className="col-md-11 col-lg-9 col-xl-8">
                        <h1 className="text-center mb-4">퀴즈 공장 (Quiz Factory)</h1>

                        {/* 플랫폼 설명 박스 */}
                        <div className="card shadow-sm mb-4 mx-auto" style={{ maxWidth: "100%" }}>
                            <div className="card-body p-4">
                                <h5 className="card-title text-center mb-3">
                                    AI 퀴즈 생성 플랫폼
                                </h5>
                                <p className="card-text text-muted">
                                    퀴즈 공장은 AI를 통해 당신의 학습 자료(PDF)에서 맞춤형 퀴즈를 생성하는 플랫폼입니다.
                                    <br />
                                    PDF 파일을 업로드하면, AI가 핵심 내용을 분석하여 문제를 출제합니다.
                                    <br />
                                    궁금한 점은 챗봇을 통해 해결할 수 있습니다.
                                    <br />
                                    이제 수동적인 읽기에서 벗어나, '스스로 검증하고 질문하는' 진짜 학습을 시작하세요.
                                </p>

                                <hr />

                                <h6 className="font-weight-bold mb-2">사용 방법</h6>
                                <ol className="mb-0 text-muted">
                                    <li>퀴즈 난이도와 생성할 퀴즈 수 선택</li>
                                    <li>학습 자료(PDF) 업로드</li>
                                    <li>AI가 자동으로 퀴즈 생성</li>
                                    <li>챗봇 Q&A</li>
                                </ol>
                            </div>
                        </div>
                        {/* 퀴즈 생성 버튼 */}
                        <div className="text-center">
                            <button onClick={handleQuizBtnClick} type="button" className="btn btn-primary btn-lg px-5 py-3 shadow w-100" style={{ fontSize: "1.2rem" }}>
                                퀴즈 생성하러 가기
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
export default MainPage;