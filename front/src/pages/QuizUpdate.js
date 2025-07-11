import { useNavigate, useSearchParams } from "react-router-dom";
import { API_IP } from "../Config";
import React, { useEffect, useState } from "react";
import axios from "axios";
import { getTokenFromCookie } from "../utils/CookieUtils";

const QuizUpdate = () => {
    const [searchParams] = useSearchParams();
    const currentPage = Number.parseInt(searchParams.get("page") || "1");
    const quiz_no = searchParams.get("quiz_no");
    const navigate = useNavigate();

    const [loading, setLoading] = useState(true);
    const [vo, setVo] = useState(null);
    const [title, setTitle] = useState("");

    const accessToken = getTokenFromCookie("accessToken");
    const refreshToken = getTokenFromCookie("refreshToken");

    const handleViewBtnClick = () => {
        navigate(`/member/quiz/view?quiz_no=${quiz_no}&page=${currentPage}`);
    }

    useEffect(() => {
        const fetchData = async () => {
            const requestUrl = `http://${API_IP}/member/quiz/view.do?quiz_no=${quiz_no}`
            try {
                setLoading(true);
                const res = await axios.get(requestUrl, {
                    withCredentials: true
                });
                setVo(res.data);
                setTitle(res.data.title);
            } catch (err) {
                if (err.response) {
                    if (err.response.status === 403) {
                        alert("로그인 후 이용 가능합니다.");
                        navigate("/auth/login");
                    }
                    else {
                        alert("예상치 못한 오류가 발생했습니다. 잠시후 다시 시도해주시기 바랍니다.");
                        navigate(`/member/quiz/view?quiz_no=${quiz_no}&page=${currentPage}`);
                    }
                    setVo(null);
                }
            } finally {
                setLoading(false);
            }
        }
        fetchData();
    }, [quiz_no]);

    if (loading) {
        return <p>데이터를 불러오는 중입니다...</p>;
    }

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

    const handleUpdateBtnClick = async (e) => {
        e.preventDefault();
        const requestUrl = `http://${API_IP}/member/quiz/update.do`
        try {
            await axios.post(requestUrl, { quiz_no, title }, {
                headers: { Authorization: `Bearer ${accessToken}`, 'X-Refresh-Token': refreshToken, 'Content-Type': 'application/json' }
            })
            alert("퀴즈 수정 성공! 퀴즈 상세보기 페이지로 이동합니다.");
            navigate(`/member/quiz/view?quiz_no=${quiz_no}&page=${currentPage}`);
        } catch {
            alert("퀴즈 수정 실패! 다시 시도해주시기 바랍니다.");
        }
    }
    return (
        <div className="container mt-5">
            <form onSubmit={handleUpdateBtnClick}>
                <div className="d-flex justify-content-between align-items-center">
                    <h1 className="mb-4">퀴즈 수정</h1>
                    <button className="btn btn-lg" onClick={handleViewBtnClick}><i className="fa fa-long-arrow-left" style={{ fontSize: '35px' }}></i></button>
                </div>
                <div className="card border-0">
                    <div className="card-header bg-white">
                        <div className="d-flex align-items-center mb-2">
                            <span className="badge badge-secondary mr-2">No : {quiz_no}</span>
                        </div>
                        <label htmlFor="title">제목 :</label>
                        <input className="form-control" value={title} onChange={(e) => setTitle(e.target.value)} />
                    </div>
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
                        <button className="btn btn-success mr-2">수정</button>
                    </div>
                </div>
            </form>
        </div>
    )
}
export default QuizUpdate;