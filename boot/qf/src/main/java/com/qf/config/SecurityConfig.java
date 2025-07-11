package com.qf.config;

import java.util.Arrays;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.www.BasicAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import com.qf.auth.filter.JwtAuthenticationFilter;

@Configuration
@EnableWebSecurity // 웹 보안 활성화
public class SecurityConfig {
	
	@Autowired
	private JwtAuthenticationFilter jwtAuthFilter;
	@Bean
	public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
		http
	    // 1. CSRF 보호 비활성화 (JWT는 Stateless이므로 일반적으로 REST API에서 필요 없음)
	    .csrf(csrf -> csrf.disable())
	    // 2. CORS(Cross-Origin Resource Sharing) 설정 추가
	    .cors(cors -> cors.configurationSource(corsConfigurationSource()))
	    // 3. HTTP Basic 인증 비활성화 (브라우저 기본 인증 팝업 방지)
	    .httpBasic(httpBasic -> httpBasic.disable())
	    // 4. 폼 로그인 비활성화 (⭐ localhost/login 리다이렉션을 막는 핵심 설정 ⭐)
	    .formLogin(formLogin -> formLogin.disable())
	    // 5. 세션 사용 안 함 (JWT는 서버에 상태를 저장하지 않는 Stateless 방식)
	    .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
	    // 6. 헤더 관련 설정 (H2 Console 등)
	    // H2-Console이 iframe 내에서 작동하려면 X-Frame-Options를 비활성화해야 합니다.
	    .headers(headers -> headers.frameOptions(frameOptions -> frameOptions.disable()))       
	    // 7. 요청별 접근 권한 설정
	    .authorizeHttpRequests(authorize -> authorize
		// ⭐ 인증 없이 접근 허용할 경로들 (로그인, 회원가입 등 공개 API) ⭐
        // H2 콘솔 접근 설정 (개발용), Favicon, 기본 에러 페이지도 permitAll()
        .requestMatchers("/auth/**", "/user/**", "/favicon.ico", "/error").permitAll()    
        // Swagger UI 관련 경로도 개발 시 permitAll() 하는 경우가 많습니다.
        // .requestMatchers("/v3/api-docs/**", "/swagger-ui/**", "/swagger-resources/**", "/webjars/**").permitAll()                
        // ⭐ ADMIN은 USER 권한이 필요한 경로에도 접근 가능하도록 명시적으로 추가 ⭐
        // "user/**" 경로는 "ROLE_USER" 또는 "ROLE_ADMIN" 권한을 가진 사용자만 접근 가능
        .requestMatchers("/member/**").hasAnyRole("MEMBER", "ADMIN") 	
        // .hasAnyRole()을 사용하여 여러 역할을 지정합니다.                
        // "admin/**" 경로는 오직 "ROLE_ADMIN" 권한을 가진 사용자만 접근 가능
	    .requestMatchers("/admin/**").hasRole("ADMIN")
	    )
	    
	    // 8. JWT 인증 필터 추가
	    // Spring Security 필터 체인 중 BasicAuthenticationFilter 이전에
	    // 우리가 만든 JwtAuthenticationFilter가 실행되도록 설정합니다.
	    .addFilterBefore(jwtAuthFilter, BasicAuthenticationFilter.class);

		// 9. 예외 처리 (선택 사항이지만, REST API에서는 오류 응답을 명확히 하기 위해 권장)
		// .exceptionHandling(exceptionHandling -> exceptionHandling
		//     .authenticationEntryPoint(new CustomAuthenticationEntryPoint()) // 인증되지 않은 사용자가 보호된 리소스에 접근 시
		// 	   .accessDeniedHandler(new CustomAccessDeniedHandler())     // 인증된 사용자가 권한 없는 리소스에 접근 시
		// );

        return http.build();
	}
	
	@Bean
	public CorsConfigurationSource corsConfigurationSource() {
		CorsConfiguration configuration = new CorsConfiguration();
        
        // 허용할 오리진(프론트엔드 URL) 목록
        // React 앱의 실제 주소를 정확히 입력합니다. (예: "http://localhost:3000")
        // 운영 환경에서는 실제 배포된 프론트엔드 도메인을 사용해야 합니다.
        configuration.setAllowedOrigins(Arrays.asList("http://localhost:3000", "http://192.168.0.232:3000"));
        
        // 허용할 HTTP 메소드 목록 (OPTIONS는 Pre-flight 요청을 위해 필수)
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"));
        
        // 허용할 요청 헤더 목록. 커스텀 헤더를 명시적으로 포함해야 합니다.
        // Authorization과 X-Refresh-Token이 여기에 반드시 포함되어야 합니다.
        configuration.setAllowedHeaders(Arrays.asList("Authorization", "Content-Type", "X-Refresh-Token"));
        
        // 자격 증명(쿠키, HTTP 인증 등) 허용 여부.
        // React 앱에서 withCredentials=true를 사용하고 쿠키를 보내려 한다면 필수입니다.
        configuration.setAllowCredentials(true);
        
        // Pre-flight 요청 결과 캐시 시간 (초)
        // 브라우저가 이 시간 동안은 동일한 요청에 대해 다시 Pre-flight을 보내지 않습니다.
        configuration.setMaxAge(3600L); // 1시간

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        // 모든 경로에 대해 위 CORS 설정을 적용
        source.registerCorsConfiguration("/**", configuration);
        return source;
    }
}