export function getTokenFromCookie(name) {
    const cookies = document.cookie.split(";"); // document.cookie는 ';'을 기준으로 쿠키 구분
    for(let i=0;i<cookies.length;i++){
        let cookie = cookies[i].trim(); // 각 쿠키 문자열의 앞뒤 공백 제거
        if(cookie.startsWith(`${name}=`)) // 찾고자 하는 토큰 정보 찾기
            return cookie.substring(name.length + 1); // 쿠키로부터 토큰값만 추출 (accessToken=AbCD.... 이런식으로 되어 있기 때문)  
    }
    return null;
}