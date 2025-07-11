package com.qf.member.controller;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import com.qf.chatbot.vo.ChatBotVO;
import com.qf.page.utils.PageObject;
import com.qf.quiz.service.QuizService;
import com.qf.quiz.vo.QuizVO;

import lombok.extern.log4j.Log4j2;

@RestController
@Log4j2
@RequestMapping("/member")
public class MemberRestController {
	
	private final String uploadDir = "D:/quiz_factory/upload/";
	
	private final RestTemplate restTemplate;
	
	@Autowired
	private QuizService service;
	
	public MemberRestController(RestTemplate restTemplate) {
		this.restTemplate = restTemplate;
	}

	// 퀴즈 리스트
	@GetMapping("/quiz/list.do")
	public ResponseEntity<Map<String, Object>> quizList(PageObject pageObject, @AuthenticationPrincipal UserDetails userDetails) {

		log.info("----- MemberRestController quizList() -----");
		Map<String, Object> map = new HashMap<>();
		
		map.put("list", service.list(pageObject, userDetails.getUsername()));
		map.put("pageObject", pageObject);
		
		log.info("list map= " + map);
		
		return new ResponseEntity<>(map, HttpStatus.OK);
	}
	
	// 퀴즈 상세보기
	@GetMapping("/quiz/view.do")
	public ResponseEntity<QuizVO> quizView(@RequestParam("quiz_no") Long quiz_no, @AuthenticationPrincipal UserDetails userDetails){
		log.info("----- MemberRestController quizView() -----");
		QuizVO vo = new QuizVO();
		vo.setQuiz_no(quiz_no);
		vo.setMember_id(userDetails.getUsername());
		return new ResponseEntity<>(service.view(vo), HttpStatus.OK);
	}
	
	// 퀴즈 생성
	@PostMapping("/quiz/create.do")
	public ResponseEntity<String> createQuiz(@RequestParam("file") MultipartFile file,
											@RequestParam("difficulty") String difficulty,
											@RequestParam("num_quiz") Integer num_quiz,
											@RequestParam("title") String title,
											@AuthenticationPrincipal UserDetails userDetails){
		log.info("----- MemberRestController createQuiz() -----");
		Path filePath = null;
		// 파일 업로드
		try {
			// 1. 파일 고유 이름 생성
			String originalFileName = file.getOriginalFilename(); // 원본 파일 이름 ex) sample.pdf
			String fileExtension = "";
			if(originalFileName != null && originalFileName.contains("."))
				fileExtension = originalFileName.substring(originalFileName.lastIndexOf(".")); // 확장자명(.pdf) 추출
			String savedFileName = UUID.randomUUID().toString() + fileExtension;
			
			// 2. 파일이 저장될 디렉토리 생성 (없으면 생성)
            Path uploadPath = Paths.get(uploadDir);
            if (!Files.exists(uploadPath)) {
                Files.createDirectories(uploadPath); // 디렉토리가 없으면 생성
            }
            
            // 3. 파일 저장 경로 설정
            filePath = uploadPath.resolve(savedFileName);

            // 4. MultipartFile을 실제 파일로 저장
            file.transferTo(filePath);            
            
			
        } catch (Exception e) {
        	return new ResponseEntity<>("파일 업로드 실패", HttpStatus.BAD_REQUEST);
        }
		
		QuizVO vo = new QuizVO();
		
		// 퀴즈 생성
		try {
			// 1. 요청 객체 구성
			// requestBody 구성
			Map<String, Object> requestBody = new HashMap<>();
			requestBody.put("input_file_path", filePath.toString());
			requestBody.put("difficulty", difficulty);
			requestBody.put("num_quiz", num_quiz);
			
			String fastApiUrl = "http://localhost:8000/api/quiz/create";
			
			// header 구성
			HttpHeaders headers = new HttpHeaders();
			headers.setContentType(MediaType.APPLICATION_JSON);
			
			HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);
			
			// 2. 데이터 전송 및 응답 받기
			ResponseEntity<String> response = restTemplate.postForEntity(fastApiUrl, requestEntity, String.class);
			
			String content = response.getBody();
			content = content.substring(12, content.length()-2);
			
			vo.setTitle(title);
			vo.setContent(content);
			vo.setFile_path(filePath.toString());
			vo.setMember_id(userDetails.getUsername());
			
		} catch (Exception e) {
			log.error(e);
			return new ResponseEntity<>("퀴즈 생성 실패", HttpStatus.INTERNAL_SERVER_ERROR);
		}
		
		Integer result = service.createQuiz(vo);
		
		return (result > 0) ? new ResponseEntity<>("퀴즈 등록 성공", HttpStatus.OK)
			: new ResponseEntity<>("퀴즈 등록 실패", HttpStatus.BAD_REQUEST);
	}
	
	
	@PostMapping("/quiz/update.do")
	public ResponseEntity<String> quizUpdate(@RequestBody QuizVO vo, @AuthenticationPrincipal UserDetails userDetails){
		log.info("----- MemberRestController quizUpdate() -----");
		vo.setMember_id(userDetails.getUsername());
		
		Integer result = service.update(vo);
		
		return (result > 0) ? new ResponseEntity<>("퀴즈 수정 성공", HttpStatus.OK)
				: new ResponseEntity<>("퀴즈 수정 실패", HttpStatus.BAD_REQUEST);
	}
	
	// 퀴즈 삭제
	@PostMapping("/quiz/delete.do")
	public ResponseEntity<String> quizDelete(@RequestBody QuizVO vo, @AuthenticationPrincipal UserDetails userDetails){
		log.info("----- MemberRestController quizDelete() -----");
		vo.setMember_id(userDetails.getUsername());
		
		Map<String, Object> requestBody = new HashMap<>();
		requestBody.put("quiz_no", vo.getQuiz_no());
		
		String fastApiUrl = "http://localhost:8000/api/quiz/delete";
		
		HttpHeaders headers = new HttpHeaders();
		headers.setContentType(MediaType.APPLICATION_JSON);
		
		HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<Map<String,Object>>(requestBody, headers);
		
		try {
			ResponseEntity<String> response = restTemplate.postForEntity(fastApiUrl, requestEntity, String.class);
			log.info(response.getBody());
		} catch (Exception e){
			new ResponseEntity<>("퀴즈 삭제 실패", HttpStatus.INTERNAL_SERVER_ERROR);	
		}
		
		Integer result = service.delete(vo);
		
		return (result > 0) ? new ResponseEntity<>("퀴즈 삭제 성공", HttpStatus.OK)
				: new ResponseEntity<>("퀴즈 삭제 실패", HttpStatus.BAD_REQUEST);
	}
	
	// 챗봇 Q&A
	@PostMapping("/chatbot/ask.do")
	public String chatBot(@RequestBody ChatBotVO vo,  @AuthenticationPrincipal UserDetails userDetails) {
		log.info("----- MemberController ask() -----");
		Map<String, Object> requestBody = new HashMap<>();
		requestBody.put("quiz_no", vo.getQuiz_no());
		requestBody.put("question", vo.getQuestion());
		requestBody.put("input_file_path", service.getFilePath(vo.getQuiz_no()));
		
		String fastApiUrl = "http://localhost:8000/api/chatbot";
		
		HttpHeaders headers = new HttpHeaders();
		headers.setContentType(MediaType.APPLICATION_JSON);
		
		HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<Map<String,Object>>(requestBody, headers);
		
		try {
			ResponseEntity<ChatBotVO> response = restTemplate.postForEntity(fastApiUrl, requestEntity, ChatBotVO.class);
			if(response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
				String answer = response.getBody().getAnswer();
	            log.info("챗봇 응답 성공: 추출된 answer: {}", answer);
	            return answer;
			}
			else {
	            log.error("챗봇 응답 실패: 상태 코드 = {}, 본문 = {}", response.getStatusCode(), response.getBody());
	            return "챗봇 응답을 받지 못했습니다.";
			}
		}catch (Exception e) {
	         log.error("RestTemplate 호출 중 예외 발생: {}", e.getMessage());
	         return "챗봇 서비스와 통신 중 오류가 발생했습니다: " + e.getMessage();
	    }
	}

}
