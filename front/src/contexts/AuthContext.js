import axios from "axios";
import { createContext, useContext, useEffect, useState } from "react";
import { API_IP } from "../Config";

export const AuthContext = createContext({
    member_id: null,
    isLoggedIn: false,
    login: () => { },
    logout: () => { },
    checkAuthStatus: () => { }
});

export const AuthProvider = ({ children }) => {
    const [member_id, setMember_id] = useState(null);
    const [isLoggedIn, setIsLoggedIn] = useState(false);

    const checkAuthStatus = async () => {
        const requestUrl = `http://${API_IP}/auth/getId.do`;
        try {
            const res = await axios.get(requestUrl, { withCredentials: true });
            if (res.data) {
                setMember_id(res.data);
                setIsLoggedIn(true);
                console.log("인증 상태 확인: 로그인 -", res.data);
            }
            else {
                setMember_id(null);
                setIsLoggedIn(false);
                console.log("인증 상태 확인: 로그아웃");
            }
        } catch (err) {
            setMember_id(null);
            setIsLoggedIn(false);
            console.error("인증 상태 확인 오류:", err);
        }
    };

    useEffect(() => {
        checkAuthStatus();
    }, []);

    const login = (user_input) => {
        setMember_id(user_input);
        setIsLoggedIn(true);
        console.log("로그인 성공:", user_input);
    };

    const logout = async () => {
        const requestUrl = `http://${API_IP}/auth/logout.do`;
        try {
            await axios.post(requestUrl, { member_id: member_id }, {
                withCredentials: true
            });
            alert("로그아웃 성공! 메인페이지로 이동합니다.");

            setMember_id(null);
            setIsLoggedIn(false);
            console.log("로그아웃 처리 완료");
            return true;
        } catch (err) {
            alert("로그아웃 실패! 다시 시도해주시기 바랍니다.");
            console.error("로그아웃 오류:", err);
            return false;
        }
    }

    const contextValue = {
        member_id,
        isLoggedIn,
        login, // 로그인 페이지 등에서 호출하여 상태를 업데이트
        logout, // Navbar 등에서 호출하여 로그아웃 처리
        checkAuthStatus, // 필요할 때 현재 인증 상태를 강제로 다시 확인
    };

    return (
        <AuthContext.Provider value={contextValue}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if(context === undefined){
        throw new Error("useAuth must ne used within an AuthProvider");
    }
    return context;
};