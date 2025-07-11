import { useState } from "react";
import { API_IP } from "../Config";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const SignUpPage = () => {
    const memberIdRegEx = /^[A-Za-z][A-Za-z0-9]{3,19}$/;
    const [member_id, setMember_id] = useState("");
    const [pw, setPw] = useState("");
    const [isMemberIdReadOnly, setIsMemberIdReadOnly] = useState(false);
    const navigate = useNavigate();

    const checkMemberId = (member_id) => {
        return memberIdRegEx.test(member_id);
    }

    const handleCheckMemberId = async (e) => {
        e.preventDefault();
        const requestUrl = `http://${API_IP}/auth/validateMemberId.do`;
        if (!member_id.trim()) {
            alert("아이디를 입력해주세요.");
            return;
        }
        else if (!checkMemberId(member_id)) {
            alert("아이디 형식이 일치하지 않습니다.\n아이디 형식 : 첫글자 영문자, 그 외 영숫자 (4~20자)");
            return;
        }
        try {
            await axios.post(requestUrl, { member_id }, {
                withCredentials: true
            });
            alert("사용 가능한 아이디입니다.");
            setIsMemberIdReadOnly(true);
        } catch (err) {
            console.log(err);
            alert("이미 사용중인 아이디입니다.");
            setIsMemberIdReadOnly(false);
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        const requestUrl = `http://${API_IP}/auth/signUp.do`;
        if (!isMemberIdReadOnly) {
            alert("아이디 중복 체크 통과 후 시도해주시기 바랍니다.");
            return;
        }
        else if (pw.length < 4) {
            alert("비밀번호 형식이 일치하지 않습니다.\n비밀번호 형식 : 4자 이상");
        }
        try {
            await axios.post(requestUrl, { member_id, pw }, {
                withCredentials: true
            });
            alert("회원가입 성공! 로그인 페이지로 이동합니다.");
            navigate("/auth/login");
        } catch {
            alert("회원가입 실패! 다시 시도해주시기 바랍니다.");
        }
    }
    return (
        <>
            <div className="jumbotron bg-white"></div>
            <div className="container mt-5">
                <div className="row justify-content-center">
                    <div className="col-md-11 col-lg-9 col-xl-8">
                        <h1 className="text-center mb-4">회원가입</h1>

                        {/* 회원가입 박스 */}
                        <div className="card shadow-sm mb-4 mx-auto">
                            <div className="card-body px-5 py-4">

                                <form onSubmit={handleSubmit}>
                                    {/* 아이디 입력 + 중복체크 */}
                                    <div className="form-group mb-3">
                                        <label htmlFor="member_id" className="form-label">
                                            아이디
                                        </label>
                                        <div className="input-group">
                                            <input
                                                type="text"
                                                className="form-control form-control-lg"
                                                value={member_id}
                                                onChange={(e) => setMember_id(e.target.value)}
                                                readOnly={isMemberIdReadOnly}
                                                placeholder="아이디를 입력하세요"
                                                required
                                            />
                                            <div className="input-group-append">
                                                <button
                                                    type="button"
                                                    onClick={handleCheckMemberId}
                                                    disabled={isMemberIdReadOnly}
                                                    className="btn btn-outline-secondary"
                                                    style={{ fontSize: "0.9rem" }}>
                                                    중복 체크
                                                </button>
                                            </div>
                                        </div>
                                    </div>

                                    {/* 비밀번호 입력 */}
                                    <div className="form-group mb-3">
                                        <label htmlFor="password" className="form-label">
                                            비밀번호
                                        </label>
                                        <input
                                            type="password"
                                            className="form-control form-control-lg"
                                            value={pw}
                                            onChange={(e) => setPw(e.target.value)}
                                            placeholder="비밀번호를 입력하세요"
                                            required
                                        />
                                    </div>

                                    {/* 회원가입 버튼 */}
                                    <div className="text-center mb-3">
                                        <button
                                            type="submit"
                                            className="btn btn-primary btn-lg w-100 shadow"
                                            style={{ fontSize: "1.2rem" }}
                                        >
                                            회원가입
                                        </button>
                                    </div>

                                    {/* 로그인 링크 */}
                                    <div className="text-center">
                                        <p className="text-muted mb-0">
                                            이미 계정이 있으신가요?
                                            <a href="/auth/login" className="text-primary ml-1">
                                                로그인
                                            </a>
                                        </p>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}

export default SignUpPage;
