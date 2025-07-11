import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext"

const NavBar = () => {
    const { member_id, isLoggedIn, logout } = useAuth();
    const navigate = useNavigate();

    const handleLogout = async (e) => {
        e.preventDefault();
        const success = await logout();
        if(success){
            navigate("/");
        }
    };

    return (
        <header>
            <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
                <div className="container">
                    <a className="navbar-brand" href="/">
                        QF
                    </a>
                    <button
                        className="navbar-toggler"
                        type="button"
                        data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav"
                        aria-expanded="false"
                        aria-label="Toggle navigation"
                    >
                        <span className="navbar-toggler-icon"></span>
                    </button>
                    <div className="collapse navbar-collapse" id="navbarNav">
                        <ul className="navbar-nav ml-auto">
                            <li className="nav-item">
                                {isLoggedIn ? (
                                    <a className="nav-link" href="/member/quiz/list">{member_id}님</a>
                                ) : (
                                    <a className="nav-link" href="/auth/login">로그인</a>
                                )}
                            </li>
                            <li className="nav-item">
                                {isLoggedIn ? (
                                    <a className="nav-link" onClick={handleLogout}>로그아웃</a>
                                ) : (
                                    <a className="nav-link" href="/auth/signup">회원가입</a>
                                )}
                            </li>
                        </ul>
                    </div>
                </div>
            </nav>
        </header>
    )
}
export default NavBar
