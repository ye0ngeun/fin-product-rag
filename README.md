# fin-product-rag

대출 상품(주택담보·전세자금·개인신용) 정보를 근거로 답하는 RAG 서비스입니다. (SSAFY AI 코어 프로젝트 2회차)

> 본 서비스는 정보 제공용이며 금융 전문가의 조언을 대신하지 않습니다.
> 상품 정보는 금융상품통합비교공시시스템(금융감독원) 금융회사 상품정보를 사용한 결과이며, 금융감독원은 정보의 정확성·최신성을 보증하지 않습니다.

## 구조

```
app.py                  Streamlit 진입점
config/settings.yaml    청크 크기·중복, Top-k, 검색 방식 등 실행 설정
src/
  config.py             설정·환경변수 로드
  data_prep/            데이터 준비 (도메인 의존 코드: 금감원 API 수집, 외부 문서 수집)
  vectorstore/          컬렉션 생성, 청킹, 임베딩 색인
  rag/                  LangGraph retrieve/generate, 프롬프트
  services/             화면이 쓰는 기능: 길잡이·비교 요약·조건 탐색(guide), 상환액·불빛 계산(loan_calc), 비교표, 실거래 사례
  eval/                 검증 스크립트: 유사도/MMR 비교, RAG 전후 비교
  ui/                   화면 구성 요소 (사이드바, 대출 계획·한 달 살아보기, 찾기·비교, 길잡이)
data/raw, data/processed  수집·가공 데이터 (git 제외)
sources.md              데이터 계획, 출처와 이용 조건
```

## 화면 안내 문구 (F2012, 금감원 이용약관 제9조 4항)

화면 하단에 항상 다음을 표시합니다. 문구는 `src/config.py` 의 `NOTICES` 에 있습니다.

- 면책: 정보 제공용이며 금융 전문가의 조언을 대신하지 않음
- 출처: "금융상품통합비교공시시스템(금융감독원) 금융회사 상품정보" 사용 결과
- 상품 정보의 수집일

## 시작하기

Python 3.10 이상.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # 키 값 채우기
```

## 데이터 수집

금감원 대출 상품 3종(주택담보·전세자금·개인신용)을 권역별 전 페이지로 받아 `data/raw/fss/{유형}.json` 에 저장하고 유형별·합계 상품 수를 출력합니다. 받을 유형과 권역은 `config/settings.yaml` 의 `fss.product_types`, `fss.fin_groups` 에서 바꿉니다. 호출 간격 때문에 끝까지 1분 정도 걸릴 수 있습니다.

```bash
python -m src.data_prep.fss_collector
```

## 메타데이터 색인 (F2001~F2002)

수집한 `data/raw/fss/{mortgage,rent_house,credit}.json` 을 읽어 상품마다 baseList 1행과 금리 옵션(optionList)을 합쳐 텍스트 1개를 만들고, GMS 임베딩으로 `loan_meta` 컬렉션(`chroma_db/`)에 넣습니다.

- 텍스트: 금융회사·권역, 상품명, 대출 유형, 담보 유형·상환 방식·금리 유형, 금리 요약(최저/최고/평균), 옵션별 금리, 대출 한도, 중도상환 수수료, 연체 이자율, 부대비용, 가입 방법, 공시 기간
- 개인신용대출은 옵션이 신용점수 구간(900점 초과 ~ 300점 이하)별 금리라서 구간별로 적고, 금리 요약은 "대출금리" 행으로 계산합니다.
- 청크 ID는 `{유형}:{fin_co_no}:{fin_prdt_cd}` 로 고정해서 다시 실행하면 덮어쓰기만 합니다. 원본에서 사라진 상품은 같은 유형 안에서 지웁니다.
- 메타데이터에 `rate_min`, `rate_max`, `rate_avg`, `loan_type`, `fin_grp_nm` 등을 숫자·문자로 넣어 조건 검색(F2010)에 씁니다.

```bash
# 임베딩 없이 적재 건수와 샘플 텍스트만 확인
python -m src.vectorstore.index_meta --dry-run --sample 3

# 색인 (OPENAI_API_KEY, OPENAI_BASE_URL 필요)
python -m src.vectorstore.index_meta
python -m src.vectorstore.index_meta --types credit   # 한 유형만
```

출력에서 유형별 로드 건수, 색인 전후 청크 수, 샘플 텍스트를 확인합니다. 두 번 실행해도 "색인 후" 건수가 그대로면 정상입니다. Streamlit 화면 왼쪽에 현재 색인 건수가 나옵니다.

## 외부 문서 수집·색인 (F2003~F2005)

외부 문서는 `loan_docs` 컬렉션에 들어갑니다. 출처 종류는 메타데이터 `doc_type`(wiki, glossary_hf, glossary_epost, glossary_epost_chatbot, glossary_bok)으로 구분합니다. 청크 ID는 `{product_id}:{doc_id}:{순번}` 이고, 위키·용어사전은 모든 상품이 같이 쓰는 문서라 `product_id` 가 `common` 입니다. 청크 크기·중복은 `config/settings.yaml` 의 `chunking` 에서 바꿉니다.

### 1) 용어사전·PDF

받아 둔 파일을 이름 그대로 `data/raw/docs/` 에 넣습니다. 어떤 파일을 읽을지는 `settings.yaml` 의 `docs` 에 파일 이름 패턴과 열 이름으로 적혀 있습니다.

| 파일 | 패턴 | 단위 |
|------|------|------|
| 한국주택금융공사_단어사전_*.csv | `*주택금융공사*단어사전*.csv` | 용어 1개 = 문서 1개 |
| 우체국금융개발원_우체국예금보험 금융용어사전_*.csv (CP949) | `*우체국예금보험*금융용어사전*.csv` | 용어 1개 = 문서 1개 |
| 우체국금융개발원_고객센터 챗봇 금융용어 사전_*.csv (CP949) | `*챗봇*금융용어*.csv` | 용어 1개 = 문서 1개 |
| 한국은행 경제금융용어 800선 PDF | `*800*.pdf` | 1쪽 = 문서 1개 (파일명·쪽 번호를 출처로 저장) |

```bash
python -m src.vectorstore.index_docs --glossary --dry-run   # 파일별 건수, 청크 수, 샘플 확인
python -m src.vectorstore.index_docs --glossary
```

### 2) 위키백과 (검색어 추출 → 검색 → 후보 선별 → 수집)

`.env` 의 `WIKI_USER_AGENT` 에 연락처를 넣어야 합니다(NF2006). 한 상품과 질문으로 실행하면 프롬프트1이 뽑은 검색어, 검색어별 후보와 프롬프트2 선별 결과, 새로 받은 문서를 출력합니다.

- 상품의 대출 유형(예: 주택담보대출)은 검색어에 항상 들어가고, 같은 제목의 문서는 선별과 상관없이 남깁니다.
- 후보는 검색어와 제목이 같은 문서(넘겨주기 포함)를 먼저 찾고, 전문 검색 결과(10개) 중 제목·요약에 검색어 글자가 들어간 것만 더합니다. 제목이 같은 문서는 선별 결과와 상관없이 남깁니다.
- 프롬프트2는 후보마다 종류(개념·제도·공공기관·기업·인물·기타)와 관련 여부만 판단하고, 코드가 개념·제도·공공기관이면서 관련 있는 문서만 고릅니다. 위키에 문서가 없는 용어(예: 중도상환수수료)는 이미 색인한 용어사전이 근거가 됩니다.
- 이미 수집한 검색어는 `data/processed/wiki_cache.json` 을 보고 다시 검색하지 않습니다(NF2002). 같은 명령을 두 번 돌리면 두 번째는 "이미 수집함 (캐시)"으로 나옵니다. 선별 규칙이 바뀌면 캐시 버전이 올라가서 검색어를 다시 모읍니다.
- 위키 문서를 처음부터 다시 모으려면 `python -m src.vectorstore.index_docs --reset-wiki` 로 위키 청크와 캐시를 지웁니다(용어사전 청크는 그대로).

```bash
python -m src.vectorstore.index_docs --product mortgage:0010001:1054 --question "중도상환수수료는 어떻게 계산돼?"
python -m src.vectorstore.index_docs --seed 2     # 유형마다 상품 2개씩 기본 질문으로 미리 수집
```

## 근거 검색·답변·근거 표시 (F2006~F2008)

LangGraph 그래프는 `retrieve` → `generate` 두 노드입니다 (`src/rag/graph.py`).

- retrieve: 상품을 고르면 그 상품의 `loan_meta` 청크를 근거 1번으로 넣고, `loan_docs` 에서 공통 문서와 그 상품 문서 중 `retrieval.top_k` 개를 찾습니다. 상품을 고르지 않으면 `loan_meta` 에서 `meta_top_k` 개를 찾습니다. 같은 청크나 같은 내용은 한 번만 남기고 `[근거 n]` 번호를 붙입니다. 검색 방식(similarity/mmr)은 `retrieval.search_type`.
- 잡음 줄이기: 관련도(코사인 유사도)가 `min_relevance` 미만이거나 1등보다 `relative_margin` 이상 낮은 근거는 버리고, 상품을 고르지 않았을 때 상품 정보는 `meta_min_relevance` 를 넘을 때만 넣습니다. 질문 용어(LTV, 담보인정비율 등)와 제목이 같은 용어사전 항목은 `keyword_top_k` 개까지 먼저 넣습니다. `--show-evidence` 로 근거별 점수와 찾은 방식(용어 일치/유사도)을 보고 값을 조정합니다.
- generate: 프롬프트3은 근거만 써서 답하고 문장마다 `[근거 n]` 을 붙입니다. 근거로 답할 수 없으면 "제공된 문서에서 확인되지 않습니다."라고 답합니다.
- 출처: 답변에 실제로 인용된 번호의 근거만 제목과 원문 위치(위키 URL, 사전 데이터 페이지, PDF 파일·쪽, 금감원 상품정보와 수집일)로 보여 줍니다. 인용이 없으면 "답변에 인용된 근거가 없습니다."를 표시합니다.

```bash
python -m src.rag.ask "중도상환수수료는 어떻게 계산돼?" --product mortgage:0010001:1054
python -m src.rag.ask "LTV가 뭐야?" --show-evidence
python -m src.rag.ask "오늘 서울 날씨 어때?" --product mortgage:0010001:1054   # 근거 없음 확인
```

## 심화: 검색 방식 비교와 RAG 전후 비교 (F2014, F2015)

대표 질문은 `config/eval_questions.yaml` 에 있습니다(특정 상품 공시값 3개, 용어 2개, 문서로 알 수 없는 질문 1개).

### MMR 검색 (F2014)

`retrieval.search_type: mmr` 이면 후보를 `mmr_fetch_k` 개 가져와 `mmr_lambda`(1에 가까울수록 관련도, 0에 가까울수록 다양성)로 `top_k` 개를 고릅니다. 길잡이 화면의 "근거 검색 방식"에서 질문마다 바꿀 수도 있습니다.

```bash
python -m src.eval.search_compare "LTV가 뭐야?"
python -m src.eval.search_compare --questions --out data/processed/eval/search_compare.md
```

같은 질문의 두 근거 목록과 근거 수, 서로 다른 문서 수, 같은 문서 중복(같은 위키 문서의 여러 청크), 출처 종류 수, 한쪽에만 나온 근거를 보여 줍니다.

### RAG 전후 비교 (F2015)

```bash
python -m src.eval.rag_compare
python -m src.eval.rag_compare --only q1 q4
```

같은 모델로 질문만 준 답과 RAG 답을 나란히 `data/processed/eval/rag_compare.md` 에 씁니다. 사실 정확성(공시값·핵심 낱말 포함), 근거 제시(인용 출처 수), 모름 인정("확인되지 않습니다")을 자동 점검하고, 표의 "사람 판단" 칸은 답을 읽고 직접 채웁니다. 결과에 금감원 공시값이 들어가므로 git 에서 빠진 폴더에 저장합니다. 제출물에는 이 파일을 복사해 넣습니다.

## 아파트 실거래가 수집 (《불빛 하나》 화면용)

국토교통부 아파트 매매 실거래가 API로 서울 구 4곳(강남·마포·노원·관악) × 지난 6개월을 받습니다. RAG 색인 대상은 아니고 화면의 거래 사례(집 불빛)에 씁니다. 공공데이터포털에서 받은 일반 인증키를 `.env` 의 `MOLIT_API_KEY` 에 넣습니다(Decoding 키 권장, Encoding 키도 됩니다). 구와 기간은 `config/settings.yaml` 의 `molit` 에서 바꿉니다.

구 이름을 지역코드(LAWD_CD)로 바꾸고 거래마다 10자리 법정동코드를 붙이는 데 공공데이터포털 "국토교통부_법정동코드" CSV를 씁니다. 받은 파일을 이름 그대로 `data/raw/codes/` 에 넣으세요(CP949 그대로 됩니다, 여러 개면 날짜가 가장 늦은 파일). 서울의 "존재" 행(구 25개, 동 467개)만 씁니다. 구별 야경 방향(중심·동·서·남·북)은 `molit.directions` 에 임시로 나눠 두었습니다.

```bash
python -m src.data_prep.molit_collector
python -m src.data_prep.molit_collector --months 3 --end-month 202609
```

결과는 `data/raw/molit/apt_trade.csv`(정리본: 구·법정동코드·동·방향·면적·층·거래일·가격(만원), 해제 거래 제외)와 `apt_trade.json`(응답 원본 필드)입니다.

## 테스트

네트워크 없이 픽스처와 가짜 임베딩·LLM으로 적재·색인·검색·계산·조건 탐색과 화면(Streamlit AppTest)을 확인합니다.

```bash
pip install pytest
python -m pytest -q
```

## 화면 《불빛 하나》 (F2009~F2012)

```bash
streamlit run app.py
```

색인(`index_meta`, `index_docs --glossary`)과 `.env` 의 `OPENAI_API_KEY` 가 있어야 합니다. 없으면 화면이 무엇을 실행할지 안내하고 멈춥니다. 실거래 사례는 `molit_collector` 결과가 있을 때만 나옵니다.

| 영역 | 내용 | 요구사항 |
| :--- | :--- | :--- |
| 사이드바 | 시작 스탯(현금자산·비상금·월 소득·생활비·기존 상환·월 여유 목표, 만원), 상품 고르기, 색인 현황, 면책·출처 | F2009, F2012 |
| 대출 계획 | 집값·대출금·기간·상환방식·적용 금리(고른 상품의 공시 최저/평균/최고) → 첫 달 상환액, 총이자, 월 남는 돈, 집의 불빛, 구별 실거래 사례 불빛 | F2010 |
| 상품 찾기·비교 | 자연어 조건 → 프롬프트4 조건 해석 → 메타데이터로 거름 → 프롬프트5가 나머지 조건 판단 → 상품 카드(맞은 조건 표시). 2~3개 비교표(공시값 그대로), 같은 계획의 상품별 월 상환액, 근거를 붙인 차이 요약 | F2010, F2011 |
| 한 달 살아보기 | 월급 → 생활비 → 기존 상환 → 새 상환 → 남은 돈 막대, 적용 금리의 공시 근거 | F2010 |
| 길잡이 | 고른 상품과의 대화. 기록은 상품마다 따로(고르지 않으면 일반 대화). 질문 때 위키 문서를 모으고(캐시 확인) 근거 찾기 → 답변 작성 진행 상황 표시, 인용된 근거 펼치기 | F2009, F2012 |

- 금리·상환액·비교표의 숫자는 코드가 공시값으로 계산합니다(`src/services/loan_calc.py`, `compare.py`). LLM은 용어 설명과 차이 요약만 근거와 함께 씁니다.
- 불빛: 자기자금(현금자산 − 비상금) + 대출금으로 살 수 있고 월 남는 돈이 목표 이상이면 따뜻한 불빛, 가격만 맞으면 깜빡이는 불빛, 가격이 넘으면 먼 불빛입니다. 취득세·중개보수와 제도상 LTV·DSR 한도는 반영하지 않고 화면에 그렇게 적습니다.
- 조건 탐색에서 금리·유형·상환방식·권역·회사 조건은 코드가 메타데이터로 판정하므로 조건과 맞지 않는 상품이 섞이지 않습니다. 0건이면 어느 조건에서 후보가 사라졌는지 알려 줍니다.
- 오류는 안내 문구만 보이고 오류 추적은 화면에 내지 않습니다(NF2004).
