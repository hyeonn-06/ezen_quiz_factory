package com.qf.auth.controller;

import java.util.Date;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseCookie;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.qf.auth.service.AuthService;
import com.qf.auth.utils.JwtTokenProvider;
import com.qf.auth.vo.AuthVO;

import lombok.extern.log4j.Log4j2;

@RestController
@Log4j2
@RequestMapping("/auth")
public class AuthRestController {
	
	@Autowired
	private AuthService service;
	
	@Autowired
	private JwtTokenProvider jtp;
	
	@PostMapping("/validateMemberId.do")
	public ResponseEntity<String> validateMemberId(@RequestBody AuthVO vo){
		log.info("----- AuthRestController validateMemberId -----");
		Integer result = service.validateMemberId(vo);
		return (result == 0) ? new ResponseEntity<>("사용 가능한 아이디입니다.", HttpStatus.OK)
			: new ResponseEntity<>("이미 사용중인 아이디입니다.", HttpStatus.BAD_REQUEST);
	}
	// 회원가입
	@PostMapping("/signUp.do")
	public ResponseEntity<String> signUp(@RequestBody AuthVO vo){
		log.info("----- AuthRestController signUp() -----");
		vo.setAuth("ROLE_MEMBER");
		Integer result = service.signUp(vo);
		return (result != 0) ? new ResponseEntity<>("회원 가입 성공", HttpStatus.OK) 
				: new ResponseEntity<>("회원 가입 실패", HttpStatus.BAD_REQUEST);
	}
	
	// 로그인
	@PostMapping("/login.do")
	public ResponseEntity<String> login(@RequestBody AuthVO vo){
		log.info("----- AuthRestController login() -----");
		Integer result = service.login(vo);
		// 로그인 정보 일치
		if(result > 0) {
			// 사용자 입력한 로그인 정보에는 권한 정보가 없음 -> 회원 정보 DB에서 권한 정보를 가져와 추가
			vo.setAuth(service.getAuth(vo));
			// access Token 생성
			String accessToken = jtp.generateAccessToken(vo);
			// accessToken을 HTTP 쿠키(Cookie)로 생성 (클라이언트 서버로 보내기 위함)
			ResponseCookie accessTokenCookie = ResponseCookie.from("accessToken", accessToken)
					.httpOnly(false)
					.secure(false)
					.path("/")
					.maxAge(jtp.getAccessTokenExpiration() / 1000) // JWT 만료 시간 (초 단위)
                    .sameSite("Lax") // 다른 웹사이트에서 보낸 요청에 브라우저가 보안을 위해 쿠키를 자동으로 첨부하지 않도록 설정하는 쿠키 속성
                    .build();
			// refresh Token 생성
			String refreshToken = jtp.generateRefreshToken(vo);
			// accessToken을 HTTP 쿠키(Cookie)로 생성 (클라이언트 서버로 보내기 위함)
			ResponseCookie refreshTokenCookie = ResponseCookie.from("refreshToken", refreshToken)
					.httpOnly(false)
					.secure(false)
					.path("/")
					.maxAge(jtp.getRefreshTokenExpiration() / 1000) // JWT 만료 시간 (초 단위)
                    .sameSite("Lax") // 다른 웹사이트에서 보낸 요청에 브라우저가 보안을 위해 쿠키를 자동으로 첨부하지 않도록 설정하는 쿠키 속성
                    .build();
			// HTTP 응답(Response) 헤더에 생성한 쿠키를 추가
			HttpHeaders headers = new HttpHeaders();
			headers.add(HttpHeaders.SET_COOKIE, accessTokenCookie.toString());
			headers.add(HttpHeaders.SET_COOKIE, refreshTokenCookie.toString());
			
			// refresh 토큰 관리 DB에 생성한 refresh Token 정보 저장
			Date now = new Date();
			service.insertRefreshToken(vo.getMember_id(), refreshToken, new Date(now.getTime() + jtp.getRefreshTokenExpiration()));
			
			return ResponseEntity.ok()
					.headers(headers)
					.body("로그인 성공!");
		}
		return new ResponseEntity<>("로그인 실패!", HttpStatus.UNAUTHORIZED);
	}
	
	// 로그아웃
	@PostMapping("/logout.do")
	public ResponseEntity<String> logout(@RequestBody AuthVO vo){
		log.info("----- AuthRestController logout() -----");
		service.logout(vo);
		return new ResponseEntity<>("로그아웃 성공!", HttpStatus.OK);
	}
	
	@GetMapping("/getId.do")
	// @AuthenticationPrincipal : Spring Security의 SecurityContextHolder에 저장되어있는 현재 인증된 사용자 정보를 주입
	// 다수의 클라이언트가 로그인하더라도 userDetails 매개변수에는 항상 현재 요청을 보낸 특정 클라이언트의 인증 정보만 주입됨
	public String getId(@AuthenticationPrincipal UserDetails userDetails) {
		if (userDetails != null) {
            String userId = userDetails.getUsername(); // UserDetails의 getUsername()은 사용자 ID 반환
            // 권한 정보는 getAuthorities()로 가져올 수 있지만, 문자열로 깔끔하게 조합하려면
            // 스트림 API를 사용할 수 있습니다.
            String roles = userDetails.getAuthorities().stream()
                                  .map(GrantedAuthority::getAuthority) // GrantedAuthority는 인터페이스, 구현체는 SimpleGrantedAuthority
                                  .collect(Collectors.joining(", ")); // 쉼표로 구분된 문자열로 결합

            log.info("인증된 사용자 ID: {}, 권한: {}", userId, roles);
            return userId;
        } else {
            // @AuthenticationPrincipal이 null 이라는 것은 인증이 되지 않은 상태를 의미합니다.
            // SecurityConfig에서 해당 경로가 permitAll()로 설정되어 있다면 이곳으로 들어올 수 있습니다.
            log.info("인증되지 않은 사용자입니다.");
            return "";
        }
	}
}
