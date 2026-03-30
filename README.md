
# NeogulMap-AI
자본기반 창업 입지 큐레이션 서비스

# NEOGULMAP AI Server

FastAPI 기반의 AI 서버입니다.
Spring 서버로부터 요청을 받아 AI 처리 후 결과를 반환합니다.

---

## 🧩 아키텍처

```
React → Spring → FastAPI (AI Server)
```

---

## 🚀 실행 방법

### 1. 레포 클론

```bash
git clone <repo>
cd NEOGULMAP
```

---

### 2. 가상환경 생성 및 실행

#### Mac / Linux

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

---

### 3. 라이브러리 설치

```bash
pip install -r requirements.txt
```

---

### 4. 서버 실행

```bash
uvicorn src.app.main:app --reload
```

---

### 5. 접속

* 서버: http://localhost:8000
* Swagger UI: http://localhost:8000/docs

---

## 📁 프로젝트 구조

```
NEOGULMAP/
 ┣ app/
 ┃ ┣ main.py              # FastAPI 실행 진입점
 ┃ ┣ api/                # 라우터 (Controller)
 ┃ ┃ ┗ v1/
 ┃ ┃   ┗ ai.py
 ┃ ┣ service/            # 비즈니스 로직
 ┃ ┃ ┗ ai_service.py
 ┃ ┣ schema/             # 요청/응답 DTO
 ┃ ┃ ┗ ai_schema.py
 ┃ ┣ core/               # 설정 관리
 ┃ ┣ utils/              # 공통 유틸
 ┃ ┗ tests/              # 단위 테스트 및 통합 테스트 
 ┣ requirements.txt
 ┗ README.md
```

---

## 📡 API 명세

### POST /api/v1/ai

#### Request

```json
{
  "message": "안녕하세요"
}
```

#### Response

```json
{
  "answer": "AI 응답입니다."
}
```

---

## ⚙️ 실행 옵션 (외부 서버 배포 시)

외부에서 접근 가능하도록 실행:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 🧪 개발 환경

* Python 3.x
* FastAPI
* Uvicorn

---

## ❗ 주의사항

* `src.app.main:app` 경로를 정확히 입력해야 합니다.
* 가상환경 활성화 후 실행해야 합니다.
* 포트 충돌 시 `--port` 옵션을 변경하세요.
* .env 파일(비밀번호, API 키, DB 주소 등 들어감)은 각자 로컬에서 만들어야 합니다. 

---

## 🔥 향후 확장

* OpenAI / Ollama 연동
* 비동기 처리 적용
* Docker 기반 배포
* Nginx + Gunicorn 적용

---


