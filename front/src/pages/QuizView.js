import React, { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { API_IP } from "../Config";
import axios from "axios";
import { getTokenFromCookie } from "../utils/CookieUtils";

const QuizView = () => {

    const navigate = useNavigate();
    const [vo, setVo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [searchParams] = useSearchParams();
    const currentPage = Number.parseInt(searchParams.get("page") || "1");
    const quiz_no = searchParams.get("quiz_no");

    const accessToken = getTokenFromCookie("accessToken");
    const refreshToken = getTokenFromCookie("refreshToken");

    const messageEndRef = useRef(null);

    const [messages, setMessages] = useState([
        { sender: "bot", text: "안녕하세요! 무엇을 도와드릴까요?" },
    ]);
    const [question, setQuestion] = useState("");
    const [isAnswering, setIsAnswering] = useState(false);

    const handleListBtnClick = () => {
        navigate(`/member/quiz/list?page=${currentPage}`);
    }

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && isAnswering === false) {
            handleQuestionBtnClick(e);
        }
    };

    useEffect(() => {
        if (messageEndRef.current) {
            messageEndRef.current.scrollTop = messageEndRef.current.scrollHeight;
        }
    }, [messages]);

    const handleQuestionBtnClick = async (e) => {
        e.preventDefault();
        const requestUrl = `http://${API_IP}/member/chatbot/ask.do`
        if (question.length > 0) {
            setMessages((prev) => [...prev, { sender: "user", text: question }]);
            setQuestion("");
            setIsAnswering(true);
            try {
                const res = await axios.post(requestUrl, { question, quiz_no }, {
                    headers: { Authorization: `Bearer ${accessToken}`, 'X-Refresh-Token': refreshToken, 'Content-Type': 'application/json' }
                });
                setMessages((prev) => [...prev, { sender: "bot", text: res.data }]);
                setIsAnswering(false);
            } catch {
                setMessages((prev) => [...prev, { sender: "bot", text: "알 수 없는 오류가 발생했습니다. 다시 시도해주시기 바랍니다." }]);
                setIsAnswering(false);
            }
        }
    }

    useEffect(() => {
        const fetchData = async (quiz_no) => {
            const requestUrl = `http://${API_IP}/member/quiz/view.do?quiz_no=${quiz_no}`
            try {
                setLoading(true);
                const res = await axios.get(requestUrl, {
                    withCredentials: true
                });
                setVo(res.data);

            } catch (err) {
                if (err.response) {
                    if (err.response.status === 403) {
                        alert("로그인 후 이용 가능합니다.");
                        navigate("/auth/login");
                    }
                    else {
                        alert("예상치 못한 오류가 발생했습니다. 잠시후 다시 시도해주시기 바랍니다.");
                        navigate(`/member/quiz/list?page=${currentPage}`);
                    }
                    setVo(null);
                }
            } finally {
                setLoading(false);
            }
        }
        fetchData(quiz_no);
    }, [quiz_no]);

    const renderTextWithLineBreaks = (text) => {
        if (!text) return null; // 텍스트가 없으면 null 반환
        return text.split('\\n').map((line, index, array) => (
            // React.Fragment를 사용하여 불필요한 div/span 렌더링 방지
            // 마지막 줄이 아니면 <br /> 추가
            <React.Fragment key={index}>
                {line}
                {index < array.length - 1 && <br />}
            </React.Fragment>
        ));
    };

    const handleUpdateBtnClick = () => {
        navigate(`/member/quiz/update?quiz_no=${quiz_no}&page=${currentPage}`);
    }

    const handleDeleteBtnClick = async (e) => {
        e.preventDefault();
        if (!window.confirm("정말 이 퀴즈를 삭제하시겠습니까?")) return;
        const requestUrl = `http://${API_IP}/member/quiz/delete.do`
        try {
            await axios.post(requestUrl, { quiz_no: quiz_no }, {
                headers: { Authorization: `Bearer ${accessToken}`, 'X-Refresh-Token': refreshToken, 'Content-Type': 'application/json' }
            });
            alert("퀴즈 삭제 성공! 퀴즈 리스트 페이지로 이동합니다.");
            navigate("/member/quiz/list");
        } catch (err) {
            alert("퀴즈 삭제 실패! 다시 시도해주시기 바랍니다.");
        }
    };

    if (loading) {
        return <p>데이터를 불러오는 중입니다...</p>;
    }

    return (
        <div className="container mt-5">
            <div className="d-flex justify-content-between align-items-center">
                <h1 className="mb-4">퀴즈 상세보기</h1>
                <button className="btn btn-lg" onClick={handleListBtnClick}><i className="fa fa-long-arrow-left" style={{ fontSize: '35px' }}></i></button>
            </div>
            <div className="card border-0">
                <div className="card-header bg-white">
                    <div className="d-flex align-items-center mb-2">
                        <span className="badge badge-secondary mr-2">No : {vo.quiz_no}</span>
                    </div>
                    <h2 className="card-title mb-0">제목 : {vo.title}</h2>
                </div>

                <div className="card-body" >
                    <h4 className="text-dark mb-3">
                        <i className="fa fa-question-circle text-info mr-2"></i>
                        퀴즈
                    </h4>
                    <div className="mb-4" style={{ height: '200px', overflowY: 'auto' }}>
                        {renderTextWithLineBreaks(vo.content)}
                    </div>
                    <div className="d-flex justify-content-end">
                        <button className="btn btn-success mr-2" onClick={handleUpdateBtnClick}>수정</button>
                        <button className="btn btn-danger" onClick={handleDeleteBtnClick}>삭제</button>
                    </div>
                </div>
            </div>

            <div className="bg-white border rounded shadow-lg">
                <div className="bg-primary text-white p-3 rounded-top d-flex justify-content-between align-items-center">
                    <h6 className="mb-0 font-weight-bold">학습 도우미</h6>
                </div>

                <div className="d-flex flex-column" style={{ height: "250px" }}>
                    <div ref={messageEndRef} className="flex-fill p-3" style={{ overflowY: "auto" }}>
                        {messages.map((msg, idx) => (
                            <div
                                key={idx}
                                className={`mb-3 d-flex ${msg.sender === "user" ? "justify-content-end" : "justify-content-start"}`}
                            >
                                <div
                                    className={`px-3 py-2 rounded small ${msg.sender === "user" ? "bg-primary text-white" : "bg-light text-dark"
                                        }`}
                                    style={{ maxWidth: "400px" }}
                                >
                                    {msg.text}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="p-3 border-top">
                        <div className="input-group">
                            <input
                                type="text"
                                className="form-control form-control-sm"
                                placeholder="메시지를 입력하세요..."
                                value={question}
                                onKeyDown={handleKeyDown}
                                onChange={(e) => setQuestion(e.target.value)}
                            />
                            <div className="input-group-append">
                                <button className="btn btn-primary btn-sm" type="button" onClick={handleQuestionBtnClick}>
                                    <i className="fa fa-paper-plane"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
export default QuizView;