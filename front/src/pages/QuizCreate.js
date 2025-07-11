import axios from "axios";
import { useState } from "react";
import { useAuth } from "../contexts/AuthContext";
import { API_IP } from "../Config";
import { useNavigate } from "react-router-dom";
import { getTokenFromCookie } from "../utils/CookieUtils";

const QuizCreate = () => {
    const [difficulty, setDifficulty] = useState("");
    const [num_quiz, setNum_quiz] = useState(1);
    const [title, setTitle] = useState("");
    const { isLoggedIn } = useAuth();
    const [selectedFile, setSelectedFile] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const accessToken = getTokenFromCookie("accessToken");
    const refreshToken = getTokenFromCookie("refreshToken");

    const navigate = useNavigate();

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            setSelectedFile(e.target.files[0]);
        }
        else {
            setSelectedFile(null);
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        if (isLoggedIn === false) {
            alert("해당 작업을 수행할 권한이 없습니다. 메인페이지로 이동합니다.");
            navigate("/");
            return;
        }

        const formData = new FormData();
        formData.append("difficulty", difficulty);
        formData.append("num_quiz", num_quiz);
        formData.append("title", title);
        formData.append("file", selectedFile);
        const requestUrl = `http://${API_IP}/member/quiz/create.do`;
        try {
            await axios.post(requestUrl, formData, {
                headers: {Authorization: `Bearer ${accessToken}`, 'X-Refresh-Token': refreshToken}
            });
            setIsLoading(false);
            alert("퀴즈 생성 성공! 퀴즈 리스트 페이지로 이동합니다.");
            navigate("/member/quiz/list");
        } catch (err) {
            setIsLoading(false);
            alert("퀴즈 생성 실패! 다시 시도해주시기 바랍니다.");
            console.log("퀴즈 생성 실패 :", err);
        }
    }

    return (
        <>
            <div className="jumbotron bg-white"></div>
            <div className="container mt-5">
                <h1 className="text-center mb-4">퀴즈 생성</h1>
                <div className="card">
                    <div className="card-body">
                        <form onSubmit={handleSubmit}>
                            <div className="form-row">
                                <div className="form-group col-md-6">
                                    <label htmlFor="difficulty">난이도 :</label>
                                    <select
                                        onChange={(e) => setDifficulty(e.target.value)}
                                        className="form-control"
                                        required
                                    >
                                        <option value="">난이도 선택</option>
                                        <option value="쉬움">쉬움</option>
                                        <option value="보통">보통</option>
                                        <option value="어려움">어려움</option>
                                    </select>
                                </div>
                                <div className="form-group col-md-6">
                                    <label htmlFor="number">퀴즈 수 :</label>
                                    <input
                                        type="number"
                                        onChange={(e) => setNum_quiz(e.target.value)}
                                        max={10}
                                        min={1}
                                        className="form-control"
                                        required
                                    />
                                </div>
                            </div>
                            <div className="form-group">
                                <label htmlFor="title">제목 :</label>
                                <input
                                    type="text"
                                    value={title}
                                    onChange={(e) => setTitle(e.target.value)}
                                    className="form-control"
                                    required
                                />
                            </div>
                            <div className="card">
                                <div className="card-body">
                                    <div style={{ textAlign: 'center' }}>
                                        <i className="fa fa-file-pdf-o" style={{ fontSize: '7em' }}></i>
                                    </div>
                                    <div className="d-flex justify-content-center mt-2">
                                        <input
                                            type="file"
                                            className="border"
                                            onChange={handleFileChange}
                                            accept="application/pdf"
                                            required
                                        />
                                    </div>
                                </div>
                            </div>
                            <div className="d-flex justify-content-end mt-2">
                                <button type="submit" className="btn btn-primary">생성 {isLoading && (<span className="spinner-border spinner-border-sm"></span>)}</button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </>
    );
};

export default QuizCreate;