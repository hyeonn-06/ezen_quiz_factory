-- 회원 관리
DROP TABLE member CASCADE CONSTRAINTS;
CREATE TABLE member (
    member_no NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY, -- 회원 관리 uid
    member_id VARCHAR2(50) UNIQUE NOT NULL, -- 아이디
    pw VARCHAR2(100) NOT NULL, -- 비밀번호
    auth VARCHAR2(12) CHECK (auth IN('ROLE_MEMBER', 'ROLE_ADMIN')) NOT NULL -- 권한
); 

INSERT INTO member (member_id, pw, auth) VALUES ('test', '1111','ROLE_MEMBER');
INSERT INTO member (member_id, pw, auth) VALUES ('admin', '1111', 'ROLE_ADMIN');

-- 퀴즈 게시판
DROP TABLE quiz CASCADE CONSTRAINTS;
CREATE TABLE quiz (
    quiz_no NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY, -- 퀴즈 게시판 uid
    title VARCHAR2(200) NOT NULL,-- 퀴즈 게시글 제목
    content CLOB NOT NULL, -- 퀴즈 내용
    file_path VARCHAR2(200) NOT NULL, -- 파일 경로
    member_id VARCHAR2(50) NOT NULL,-- 아이디(회원)
    created_at DATE DEFAULT SYSDATE, -- 퀴즈 출제 일자
    
    CONSTRAINT fk_quiz_member
    FOREIGN KEY (member_id)
    REFERENCES member (member_id) 
    ON DELETE CASCADE
);

-- refresh 토큰 관리
DROP TABLE refresh_token CASCADE CONSTRAINTS;    
CREATE TABLE refresh_token (
    token_no NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY, -- refresh 토큰 관리 uid
    token_value VARCHAR2(255) NOT NULL, -- 토큰 값
    member_id VARCHAR2(20) NOT NULL, -- 아이디(회원)
    expiry_date DATE NOT NULL, -- 토큰 만료 일자
    issued_date DATE DEFAULT SYSDATE NOT NULL, -- 토큰 발급 일자
    CONSTRAINT fk_refresh_token_member 
    FOREIGN KEY (member_id)
    REFERENCES member (member_id)
    ON DELETE CASCADE
);

commit;