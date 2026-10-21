"""국토교통부 법정동코드 CSV에서 서울 구·동 코드표를 만든다.

실거래가 API의 지역코드(LAWD_CD, 법정동코드 앞 5자리)와 법정동 이름을 연결해
거래 사례가 어느 구·동의 집인지 표시하고, 구별 야경 방향에 배치하는 데 쓴다.

입력: data/raw/codes/ 아래 *법정동코드*.csv (공공데이터포털 원본, CP949, 열: 법정동코드,법정동명,폐지여부)
      여러 개면 파일 이름 순으로 마지막(가장 최근 날짜) 파일을 쓴다.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

SIDO_PREFIX = "11"  # 서울특별시
SIDO_NAME = "서울특별시"


@dataclass
class RegionTable:
    gu_by_code: dict[str, str] = field(default_factory=dict)  # "11440" -> "마포구"
    code_by_gu: dict[str, str] = field(default_factory=dict)  # "마포구" -> "11440"
    dong_by_code: dict[str, str] = field(default_factory=dict)  # "1144010200" -> "공덕동"
    code_by_dong: dict[tuple[str, str], str] = field(default_factory=dict)  # ("11440", "공덕동") -> "1144010200"

    def lawd_cd(self, gu_nm: str) -> str:
        if gu_nm not in self.code_by_gu:
            raise KeyError(f"서울 구 이름이 아닙니다: {gu_nm}")
        return self.code_by_gu[gu_nm]


def find_code_file(codes_dir: Path) -> Path:
    files = sorted(Path(codes_dir).glob("*법정동코드*.csv"))
    if not files:
        raise FileNotFoundError(
            f"{codes_dir} 에 법정동코드 CSV가 없습니다. 공공데이터포털 '국토교통부_법정동코드'를 받아 넣으세요."
        )
    return files[-1]


def _read_rows(path: Path) -> list[dict]:
    """원본은 CP949. UTF-8(BOM 포함)로 다시 저장한 파일도 읽는다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            with open(path, encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없습니다: {path}")


def load_seoul_regions(path: Path) -> RegionTable:
    """서울 + 폐지여부 '존재' 행만 남겨 구(시군구)와 동(읍면동) 코드표를 만든다."""
    table = RegionTable()
    for row in _read_rows(path):
        code = (row.get("법정동코드") or "").strip()
        name = (row.get("법정동명") or "").strip()
        if not code.startswith(SIDO_PREFIX) or (row.get("폐지여부") or "").strip() != "존재":
            continue
        parts = name.split()
        if len(parts) == 2 and code.endswith("00000"):
            table.gu_by_code[code[:5]] = parts[1]
            table.code_by_gu[parts[1]] = code[:5]
        elif len(parts) >= 3:
            dong = " ".join(parts[2:])
            table.dong_by_code[code] = dong
            table.code_by_dong[(code[:5], dong)] = code
    return table
