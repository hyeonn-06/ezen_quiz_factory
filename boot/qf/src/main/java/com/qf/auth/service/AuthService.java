package com.qf.auth.service;

import java.util.Date;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.qf.auth.vo.AuthVO;
import com.qf.auth.mapper.AuthMapper;

import lombok.extern.log4j.Log4j2;

@Service
@Log4j2
public class AuthService {
	
	@Autowired
	private AuthMapper mapper;
	
	public Integer validateMemberId(AuthVO vo) {
		log.info("----- AuthService validateMemberId() -----");
		return mapper.validateMemberId(vo);
	}
	
	public Integer signUp(AuthVO vo) {
		log.info("----- AuthService signUp() -----");
		return mapper.signUp(vo);
	}
	
	public Integer login(AuthVO vo) {
		log.info("----- AuthService login() -----");
		return mapper.login(vo);
	}
	
	public String getAuth(AuthVO vo) {
		log.info("----- AuthService getAuth() -----");
		return mapper.getAuth(vo);
	}
	
	public Integer insertRefreshToken(String member_id, String token, Date expiry_date) {
		log.info("----- AuthService insertRefreshToken() -----");
		return mapper.insertRefreshToken(member_id, token, expiry_date);
	}
	
	public Integer logout(AuthVO vo) {
		log.info("----- AuthService logout() -----");
		return mapper.logout(vo);
	}
	
	
	// DB에서 refresh 토큰 검증
	public Integer validateRefreshToken(String token) {
		log.info("----- AuthService() validateRefreshToken() -----");
		return mapper.validateRefreshToken(token);
	}
}
