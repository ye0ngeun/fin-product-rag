# fin-product-rag

정기예금·적금 상품 정보를 근거로 답하는 RAG 서비스입니다. (SSAFY AI 코어 프로젝트 2회차)

> 본 서비스는 정보 제공용이며 금융 전문가의 조언을 대신하지 않습니다.

## 구조

```
app.py                  Streamlit 진입점
config/settings.yaml    청크 크기·중복, Top-k, 검색 방식 등 실행 설정
src/
  config.py             설정·환경변수 로드
  data_prep/            데이터 준비 (도메인 의존 코드: 금감원 API 수집, 외부 문서 수집)
  vectorstore/          컬렉션 생성, 청킹, 임베딩 색인
  rag/                  LangGraph retrieve/generate, 프롬프트
  services/             도메인 특화 기능, 조건 기반 상품 탐색
  ui/                   화면 구성 요소
data/raw, data/processed  수집·가공 데이터 (git 제외)
sources.md              데이터 출처와 이용 조건
```

## 시작하기

Python 3.10 이상.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # 키 값 채우기
```

## 데이터 수집

금감원 정기예금·적금 상품(은행·저축은행, 전 페이지)을 받아 `data/raw/fss/` 에 저장하고 상품 수를 출력합니다.

```bash
python -m src.data_prep.fss_collector
```

## 실행

```bash
streamlit run app.py
```
