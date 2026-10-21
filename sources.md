# 데이터 출처와 이용 조건 (NF2007)

수집·사용하는 모든 데이터의 출처와 이용 조건을 기록한다. `TBD` 칸은 원문 페이지에서 확인한 뒤 채운다.

| # | 출처 | 용도 | 형식 | 수집 방법 | 원문 URL | 라이선스/이용 조건 | 표기 의무 | 확인일 |
|---|------|------|------|-----------|----------|--------------------|-----------|--------|
| 1 | 금융감독원 금융상품한눈에 Open API: 정기예금 | 항목 메타데이터 | JSON | API (`depositProductsSearch.json`, 권역 020000/030300) | https://finlife.fss.or.kr/finlife/api/main.do | TBD | TBD | TBD |
| 2 | 금융감독원 금융상품한눈에 Open API: 적금 | 항목 메타데이터 | JSON | API (`savingProductsSearch.json`, 권역 020000/030300) | https://finlife.fss.or.kr/finlife/api/main.do | TBD | TBD | TBD |
| 3 | 금융감독원 금융상품한눈에 Open API: 금융회사 | 항목 메타데이터(보조) | JSON | API (`companySearch.json`) | https://finlife.fss.or.kr/finlife/api/main.do | TBD | TBD | TBD |
| 4 | 한국어 위키백과 | 외부 문서 | JSON → 텍스트 | MediaWiki API, 질문 시 수집·캐시 | https://ko.wikipedia.org | CC BY-SA 4.0 (확인 필요) | 저작자·출처 표시, 동일조건 변경허락 | TBD |
| 5 | 한국은행 경제금융용어 800선 | 외부 문서 | PDF → 텍스트 | 파일 다운로드 | TBD | TBD (공공누리 유형 확인) | TBD | TBD |
| 6 | 우체국예금보험 금융용어사전 | 외부 문서 | CSV (CP949) | 파일 다운로드 (1,198건) | TBD | TBD | TBD | TBD |
| 7 | 우체국 고객센터 챗봇 금융용어 사전 | 외부 문서 | CSV (CP949) | 파일 다운로드 (1,038건) | TBD | TBD | TBD | TBD |

## 비고

- 원본 데이터(`data/raw/`)는 저장소에 올리지 않는다. 각 출처에서 다시 받도록 README에 방법을 적는다.
- 위키백과 요청에는 연락처가 포함된 User-Agent(`WIKI_USER_AGENT`)를 붙인다.
- 우체국 CSV 두 종은 용어 133개가 겹친다. 색인 시 출처 종류를 메타데이터로 구분한다. (F2016)
