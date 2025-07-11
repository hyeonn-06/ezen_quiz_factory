package com.qf.quiz.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.qf.page.utils.PageObject;
import com.qf.quiz.mapper.QuizMapper;
import com.qf.quiz.vo.QuizVO;

import lombok.extern.log4j.Log4j2;

@Service
@Log4j2
public class QuizService {
	
	@Autowired
	private QuizMapper mapper;
	
	public Integer createQuiz(QuizVO vo) {
		log.info("----- QuizService createQuiz() -----");
		return mapper.createQuiz(vo);
	}
	
	public List<QuizVO> list(PageObject pageObject, String member_id){
		log.info("----- QuizService list() -----");
		pageObject.setTotalRow(mapper.getTotalRow(member_id));
		return mapper.list(pageObject, member_id);
	}
	
	public QuizVO view(QuizVO vo){
		log.info("----- QuizService view() -----");
		return mapper.view(vo);
	}
	
	public Integer update(QuizVO vo) {
		log.info("----- QuizService update() -----");
		return mapper.update(vo);
	}
	
	public Integer delete(QuizVO vo) {
		log.info("----- QuizService delete() -----");
		return mapper.delete(vo);
	}
	
	public String getFilePath(Long quiz_no) {
		log.info("----- QuizService getFilePath() -----");
		return mapper.getFilePath(quiz_no);
	}
}
