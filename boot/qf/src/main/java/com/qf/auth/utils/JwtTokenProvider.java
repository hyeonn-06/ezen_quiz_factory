package com.qf.auth.utils;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import javax.crypto.SecretKey;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;

import com.qf.auth.vo.AuthVO;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import lombok.extern.log4j.Log4j2;

@Component
@Log4j2
public class JwtTokenProvider {

	private SecretKey key;
	// accessToken 만료시간(15분)
	private final long accessTokenExpiration;
	// refreshToken 만료시간(1개월)
	private final long refreshTokenExpiration;
	
    public JwtTokenProvider(@Value("${jwt.secret-key}") String secretKey,
			@Value("${jwt.access-token-expiration-time}") long accessTokenExpiration,
			@Value("${jwt.refresh-token-expiration-time}") long refreshTokenExpiration){
		byte[] keyBytes = Decoders.BASE64.decode(secretKey);
		this.key = Keys.hmacShaKeyFor(keyBytes);
		this.accessTokenExpiration = accessTokenExpiration;
		this.refreshTokenExpiration = refreshTokenExpiration;
    }
    
    private SecretKey getSigningKey() {
    	return this.key;
    }
    
    // JWT 토큰 생성(accessToken)
    public String generateAccessToken(AuthVO vo) {
    	Date now = new Date();
    	Date expiryDate = new Date(now.getTime() + accessTokenExpiration);
    	return Jwts.builder() // JWT 빌더 객체를 생성하여 JWT 토큰 생성 시작
    			.subject(vo.getMember_id()) // 누구에 대한 JWT 토큰인지 설정(subject 클레임)
    			.issuedAt(now) // 해당 JWT 토큰이 언제 발행되었는지 설정(issue 클레임)
    			.expiration(expiryDate) // 해당 JWT 토큰이 언제 만료되는지 설정(expire 클레임) 
    			.claim("auth",vo.getAuth()) // 해당 JWT 토큰의 소유자가 어떤 권한을 갖게 되는지 설정(auth 클레임)
    			.signWith(getSigningKey()) // 설정을 마무리하고 JWT 문자열 생성
    			.compact();
    }

    // JWT 토큰 생성(refreshToken)
    public String generateRefreshToken(AuthVO vo) {
    	Date now = new Date();
    	Date expiryDate = new Date(now.getTime() + refreshTokenExpiration);
    	return Jwts.builder() // JWT 빌더 객체를 생성하여 JWT 토큰 생성 시작
    			.subject(vo.getMember_id()) // 누구에 대한 JWT 토큰인지 설정(subject 클레임)
    			.issuedAt(now) // 해당 JWT 토큰이 언제 발행되었는지 설정(issue 클레임)
    			.expiration(expiryDate) // 해당 JWT 토큰이 언제 만료되는지 설정(expire 클레임) 
    			.claim("auth",vo.getAuth()) // 해당 JWT 토큰의 소유자가 어떤 권한을 갖게 되는지 설정(auth 클레임)
    			.signWith(getSigningKey()) // 설정을 마무리하고 JWT 문자열 생성
    			.compact();
    }
    
    // accessToken 만료 시간(15분) 반환 (단위 : 밀리초)
    public long getAccessTokenExpiration() {
    	return accessTokenExpiration;
    }
    
    // refreshToken 만료 시간(1개월) 반환 (단위 : 밀리초)
    public long getRefreshTokenExpiration() {
    	return refreshTokenExpiration;
    }
	
    // token 유효성 검증
    public boolean validateToken(String token) {
        try {
            Jwts.parser()
                .verifyWith(getSigningKey())
                .build()
                .parseSignedClaims(token);
            return true;
        } catch (io.jsonwebtoken.security.SecurityException | io.jsonwebtoken.MalformedJwtException e) {
            // 잘못된 JWT 서명
            System.out.println("Invalid JWT signature.");
        } catch (io.jsonwebtoken.ExpiredJwtException e) {
            // 만료된 JWT 토큰
            System.out.println("Expired JWT token.");
        } catch (io.jsonwebtoken.UnsupportedJwtException e) {
            // 지원되지 않는 JWT 토큰
            System.out.println("Unsupported JWT token.");
        } catch (IllegalArgumentException e) {
            // JWT 클레임 문자열이 비어있음
            System.out.println("JWT claims string is empty.");
        }
        return false;
    }
   
    // Jwt token의 클레임 정보 반환
	private Claims parseClaims(String token) {
	       return Jwts.parser().verifyWith(getSigningKey()).build().parseSignedClaims(token).getPayload();
	}
    
    // Jwt token으로부터 권한을 읽어와 Spring Security가 이해할 수 있는 인증 정보로 변환
    // => 사용자가 어떤 권한을 가지고 있는지 시스템에 알려주는 역할
    public Authentication getAuthentication(String token) {
    	// JwtToken으로부터 클레임 정보를 가져옴
    	Claims claims = parseClaims(token);
    	// 가져온 클레임 정보로부터 auth(권한) 클레임 정보만 문자열로 가져옴
    	String authFromToken = claims.get("auth", String.class);
    	
    	List<GrantedAuthority> authorities = new ArrayList<>();
    	
    	// 권한(ROLE_USER 또는 ROLE_ADMIN)이 존재
    	if(authFromToken != null)
    		// 토큰에서 읽어온 문자열을 그대로 GrantedAuthority로 추가
    		authorities.add(new SimpleGrantedAuthority(authFromToken));
    	else
    		log.warn("Jwt token에 auth 클레임이 없거나 유효하지 않습니다.");
    	// Spring Security가 이해할 수 있는 인증 정보 객체 생성
    	// 파라미터값은 순서대로 사용자 아이디 정보, 비밀번호, 권한 정보 (비밀번호는 Jwt token으로 통신하기 때문에 빈 문자열)
        UserDetails principal = new User(claims.getSubject(), "", authorities);
        // Spring Security에게 인증 정보 전달
        return new UsernamePasswordAuthenticationToken(principal, "", authorities);
    }
    
    // token의 subject claim에 저장되어있는 member_id 반환
    public String getMemberIdFromToken(String token) {
        return Jwts.parser()
                .verifyWith(getSigningKey())
                .build()
                .parseSignedClaims(token)
                .getPayload()
                .getSubject();
    }
    
    // token의 auth claim에 저장되어있는 auth 반환
    public String getAuthFromToken(String token) {
    	Claims claims = parseClaims(token);
    	return claims.get("auth", String.class);
    }
}
