"""config/settings.yaml 과 .env 를 읽어 하나의 설정 객체로 제공한다."""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT_DIR / "config" / "settings.yaml"


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
