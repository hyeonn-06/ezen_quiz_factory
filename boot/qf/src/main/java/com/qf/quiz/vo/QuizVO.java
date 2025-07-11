package com.qf.quiz.vo;

import java.util.Date;

import lombok.Data;

@Data
public class QuizVO {
	private Long quiz_no;
	private String title;
	private String content;
	private String file_path;
	private String member_id;
	private Date created_at;
}
