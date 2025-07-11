import axios from "axios";
import { useState } from "react"
import { useNavigate } from "react-router-dom";
import { API_IP } from "../Config";
import { useAuth } from "../contexts/AuthContext";

const LoginPage = () => {
  const [member_id, setMember_id] = useState("");
  const [pw, setPw] = useState("");
  const navigate = useNavigate();
  const { checkAuthStatus, login } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const requestUrl = `http://${API_IP}/auth/login.do`;
    console.log(requestUrl);
    try{
      await axios.post(requestUrl, {member_id, pw}, {
        withCredentials: true
      })
      alert("로그인 성공! 메인페이지로 이동합니다.");
      login(member_id);
      await checkAuthStatus();
      navigate("/");
    } catch(err){
      alert("로그인 실패! 다시 시도해주시기 바랍니다.");
    }
  }
  return (
    <>
    <div className="jumbotron bg-white"></div>
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-11 col-lg-9 col-xl-8">
          <h1 className="text-center mb-4">로그인</h1>

          {/* 로그인 박스 */}
          <div className="card shadow-sm mb-4 mx-auto">
            <div className="card-body px-5 py-4">
              <form onSubmit={handleSubmit}>
                {/* 아이디 입력 */}
                <div className="form-group mb-3">
                  <label htmlFor="member_id" className="form-label">
                    아이디
                  </label>
                  <input
                    type="text"
                    className="form-control form-control-lg"
                    value={member_id}
                    onChange={(e) => setMember_id(e.target.value)}
                    placeholder="아이디를 입력하세요"
                    required
                  />
                </div>

                {/* 비밀번호 입력 */ }
                <div className="form-group mb-4">
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

                {/* 로그인 버튼 */}
                <div className="text-center mb-3">
                  <button
                    type="submit"
                    className="btn btn-primary btn-lg w-100 shadow"
                    style={{ fontSize: "1.2rem" }}
                  >
                    로그인
                  </button>
                </div>

                {/* 회원가입 링크 */}
                <div className="text-center">
                  <p className="text-muted mb-0">
                    계정이 없으신가요?
                    <a href="/auth/signup" className="text-primary ml-1">
                      회원가입
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
export default LoginPage;
