package com.qf.auth.filter;

import java.io.IOException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseCookie;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.OncePerRequestFilter;

import com.qf.auth.service.AuthService;
import com.qf.auth.utils.JwtTokenProvider;
import com.qf.auth.vo.AuthVO;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.log4j.Log4j2;

@Component
@Log4j2
public class JwtAuthenticationFilter extends OncePerRequestFilter{
	
	@Autowired
	private JwtTokenProvider jtp;
	
	@Autowired
	private AuthService service;
	
	@Override
	protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {
		String accessToken = null;
		String refreshToken = null;
		// GET 방식의 통신
		if(request.getMethod().equalsIgnoreCase("GET")) {
			// accessToken의 value
			accessToken = resolveToken(request, "accessToken");
			// refreshToken의 value
			refreshToken = resolveToken(request, "refreshToken");
		}
		// POST 방식의 통신
		else {
			accessToken = getTokenFromAuthHeader(request);
			refreshToken = getTokenFromCustomHeader(request);
		}
		// accessToken이 존재(만료되지 않음)하고 유효성 검증 통과 
		if(StringUtils.hasText(accessToken) && jtp.validateToken(accessToken)) {
			// 로그아웃을 해도 클라이언트 환경의 만료되기 전까지 accessToken은 삭제되지 않음
			// -> DB에서 관리 중인 refreshToken과 유효성 검증
			Integer result = service.validateRefreshToken(refreshToken);
			// refreshToken 유효
			if(result > 0) {
	            try {
	                // 토큰이 유효하면 Authentication 객체를 가져와서 SecurityContext에 저장
	                // 이 Authentication 객체는 JwtTokenProvider의 getAuthentication 메서드에서 생성됨
	                Authentication authentication = jtp.getAuthentication(accessToken);
	                SecurityContextHolder.getContext().setAuthentication(authentication);
	                log.info("인증 성공: 사용자 [{}]", authentication.getName()); // 인증 성공 로그
	            } catch (Exception e) {
	                log.error("JWT 토큰 유효성 검사 또는 인증 설정 중 오류 발생: {}", e.getMessage());
	                // 필요하다면 여기서 응답을 설정하여 클라이언트에게 오류를 알릴 수 있습니다.
	                // response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
	                // return; // 더 이상 필터 체인을 진행하지 않음
	            }
			}
		}
		// accessToken은 존재하지않고, refreshToken은 유효성 검증 통과
		else if(jtp.validateToken(refreshToken)) {
			// 로그아웃을 해도 클라이언트 환경의 만료되기 전까지 refreshToken은 삭제되지 않음
			// -> DB에 저장되어있는 refreshToken과 유효성 검증
			Integer result = service.validateRefreshToken(refreshToken);
			// refreshToken 유효
			if(result > 0) {
				// accessToken 재발급을 위해 refreshToken으로부터 member_id, auth를 꺼내 MemberVO 객체에 넣어 전달
				AuthVO vo = new AuthVO();
				vo.setMember_id(jtp.getMemberIdFromToken(refreshToken));
				vo.setAuth(jtp.getAuthFromToken(refreshToken));
				// 새로운 accessToken 생성
				String newAccessToken = jtp.generateAccessToken(vo);
				// 새로운 accessToken을 HTTP 쿠키(Cookie)로 생성 (클라이언트 서버로 보내기 위함)
    			ResponseCookie cookie = ResponseCookie.from("accessToken", newAccessToken)
    					.httpOnly(false)
    					.secure(false)
    					.path("/")
    					.maxAge(jtp.getAccessTokenExpiration() / 1000) // JWT 만료 시간 (초 단위)
                        .sameSite("Lax")
                        .build();
    			// 클라이언트(웹 브라우저)에게 쿠키를 설정하라는 명령을 HTTP 응답 헤더에 추가
    			response.addHeader("Set-Cookie", cookie.toString());
                // 토큰이 유효하면 Authentication 객체를 가져와서 SecurityContext에 저장
                // 이 Authentication 객체는 JwtTokenProvider의 getAuthentication 메서드에서 생성됨
                Authentication authentication = jtp.getAuthentication(newAccessToken);
                SecurityContextHolder.getContext().setAuthentication(authentication);
                log.info("access 토큰 재발급 성공"); 
			}
			else {
				log.warn("DB에서 관리 중인 refreshToken이 아닙니다.");
			}
			
		}
		// accessToken과 refreshToken 모두 존재하지 않음
		else {
			if(accessToken == null && refreshToken == null)
				log.warn("accessToken, refreshToken이 존재하지 않습니다.");
			else
				log.warn("유효하지 않은 Jwt Token입니다.");
		}
		filterChain.doFilter(request, response);
	}
	
	// 전달받은 토큰을 읽어 토큰값 반환 (GET 방식)
	private String resolveToken(HttpServletRequest request, String tokenName) {
		Cookie[] cookies = request.getCookies();
		if(cookies != null) {
			for(Cookie cookie : cookies) {
				if(tokenName.equals(cookie.getName())) {
					return cookie.getValue();
				}
			}
		}
		return null;
	}
	
	// 인증 헤더로부터 토큰(accessToken) 가져오기
	private String getTokenFromAuthHeader(HttpServletRequest request) {
		String accessToken = request.getHeader("authorization");
		if(accessToken == null)
			return null;
		return accessToken.substring(7);
	}
	
	// 커스텀 헤더로부터 토큰(refreshToken) 가져오기
	private String getTokenFromCustomHeader(HttpServletRequest request) {
		String refreshToken = request.getHeader("x-refresh-token");
		return refreshToken;
	}
	
}
