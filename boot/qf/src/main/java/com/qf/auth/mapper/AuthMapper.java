package com.qf.auth.mapper;

import java.util.Date;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import com.qf.auth.vo.AuthVO;

@Mapper
public interface AuthMapper {
	public Integer validateMemberId(AuthVO vo);
	public Integer signUp(AuthVO vo);
	public Integer login(AuthVO vo);
	public String getAuth(AuthVO vo);
	public Integer insertRefreshToken(@Param("member_id") String member_id, @Param("token") String token, @Param("expiry_date") Date expiry_date);
	public Integer logout(AuthVO vo);
	public Integer validateRefreshToken(String token);
}
