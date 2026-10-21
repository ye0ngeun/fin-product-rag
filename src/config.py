"""config/settings.yaml 과 .env 를 읽어 하나의 설정 객체로 제공한다."""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT_DIR / "config" / "settings.yaml"

# 화면에 항상 표시하는 안내 문구 (F2012, 금감원 오픈API 이용약관 제9조 4항)
NOTICES = {
    "disclaimer": "본 서비스는 정보 제공용이며 금융 전문가의 조언을 대신하지 않습니다.",
    "fss_source": (
        "상품 정보는 금융상품통합비교공시시스템(금융감독원) 금융회사 상품정보를 사용한 결과이며, "
        "금융감독원은 정보의 정확성·최신성을 보증하지 않습니다."
    ),
}


@lru_cache
def get_settings() -> dict:
    load_dotenv(ROOT_DIR / ".env")
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    for key, rel in settings["paths"].items():
        settings["paths"][key] = ROOT_DIR / rel
    return settings


def get_env(name: str, required: bool = True) -> str | None:
    load_dotenv(ROOT_DIR / ".env")
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"환경변수 {name} 가 비어 있습니다. .env 를 확인하세요.")
    return value
