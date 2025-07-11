package com.qf.chatbot.vo;

import lombok.Data;

@Data
public class ChatBotVO {
	private Long quiz_no;
	private String question;
	private String member_id;
	private String input_file_path;
	
	private String answer;
}
