package com.qf.quiz.mapper;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import com.qf.page.utils.PageObject;
import com.qf.quiz.vo.QuizVO;

@Mapper
public interface QuizMapper {
	public Integer createQuiz(QuizVO vo);
	public Long getTotalRow(String member_id);
	public List<QuizVO> list(@Param("pageObject") PageObject pageObject, @Param("member_id") String member_id);
	public QuizVO view(QuizVO vo);
	public Integer update(QuizVO vo);
	public Integer delete(QuizVO vo);
	
	public String getFilePath(Long quiz_no);
}
