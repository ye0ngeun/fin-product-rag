# 《불빛 하나》 fin-product-rag — 작업 규칙

AI 코딩 도구(Claude Code 등)와 사람이 함께 따르는 규칙입니다. 새 작업을 시작하면 이 파일과 [README](README.md)를 먼저 읽습니다.

## 1. 한 줄 요약
금감원 대출상품 공시(주택담보·전세자금·개인신용)와 용어사전·위키백과를 근거로 답하는 RAG 길잡이 + 내집마련 대출 시뮬레이션. SSAFY AI 코어 프로젝트 2회차 과제.

## 2. 꼭 지킬 것
1. **금리·한도·상환액 숫자를 LLM이 만들지 않는다.** 공시값은 메타데이터 그대로 쓰고, 상환액·비율은 `src/services/loan_calc.py`가 계산한다. 근거에 없는 내용은 "확인되지 않습니다".
2. **`[근거 n]`과 출처 목록이 1:1로 맞아야 한다.** 인용된 근거만 출처로 표시한다 (`src/rag/graph.py`).
3. **모든 화면에 면책 문구와 금감원 출처 문구**("금융상품통합비교공시시스템(금융감독원) 금융회사 상품정보", 약관 제9조 4항)를 둔다. `src/ui/common.py::disclaimers`, `tests/test_app.py`가 확인한다.
4. **키는 `.env`에만.** `.env.example`은 빈 값. `data/raw/`, `chroma_db/`, 금감원 공시값이 든 결과 파일은 커밋·재배포하지 않는다 (`sources.md` 3-1).
5. **이용 조건을 확인한 출처만 색인한다.** 새 출처는 `sources.md` 표에 먼저 적는다.
6. **설정 값(청크·Top-k·검색 방식·호출 간격)은 `config/settings.yaml`, 프롬프트는 `src/rag/prompts.py`에만** 둔다.
7. **버전 고정** (`requirements.txt`): LangChain 1.3.14, LangGraph 1.2.10, langchain-openai 1.4.3, langchain-chroma 1.1.0, ChromaDB 1.5.9, OpenAI 2.54.0, Streamlit 1.62.0. LLM·임베딩은 SSAFY GMS.
8. **F2001~F2012, NF2001~NF2007을 깨지 않는다.** 기능을 바꾸면 [구현 매핑](docs/04_요구사항_구현매핑.md)의 해당 줄도 고친다.

## 3. 환경과 명령
- 금감원·위키·국토부 API는 클라우드 세션에서 막혀 있다. 실제 수집·색인·화면 확인은 영은 PC(Windows)에서 하고, 클라우드에서는 픽스처 기반 테스트만 돌린다.

| 목적 | 명령 |
| :--- | :--- |
| 테스트 (네트워크·키 불필요) | `python -m pytest -q` |
| 상품 색인 확인 | `python -m src.vectorstore.index_meta --dry-run --sample 3` |
| 문서 색인 확인 | `python -m src.vectorstore.index_docs --glossary --dry-run` |
| 질문 한 번 | `python -m src.rag.ask "LTV가 뭐야?" --show-evidence` |
| 화면 | `streamlit run app.py` |
| 제출 zip | `python scripts/make_submission.py --region … --cls … --name …` |

## 4. 작업 방식
- 브랜치 → PR → `main` 머지. PR 본문에 관련 요구사항 번호와 확인한 테스트를 적는다.
- 완료 전에 `python -m pytest -q`를 돌린다. 실행하지 못한 확인(키·네트워크 필요)은 "미확인"으로 적고 통과로 쓰지 않는다.
- 결과 수치(테스트 개수, 색인 건수)를 바꾸면 [검증 결과](docs/03_검증결과.md)도 같이 고친다.

## 5. 문서 지도
| 문서 | 언제 보나 |
| :--- | :--- |
| [요구사항 정의서](docs/01_요구사항정의서.md) | 기능 범위, 요구사항 번호 |
| [RAG 설계문서](docs/02_RAG_파이프라인_설계문서.md) | 컬렉션·청킹·검색·프롬프트를 바꿀 때 |
| [검증 결과](docs/03_검증결과.md) | 제출 전 점검, 스크린샷 목록 |
| [구현 매핑](docs/04_요구사항_구현매핑.md) | 요구사항 → 코드 → 테스트 찾기 |
| [실행 가이드](docs/05_실행가이드.md) | 명령 옵션, 출력 확인, 화면 구성 |
| [출처와 이용 조건](sources.md) | 데이터를 추가하거나 화면 문구를 바꿀 때 |
