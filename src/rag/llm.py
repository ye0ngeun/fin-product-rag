"""GMS(OpenAI 호환) 채팅 모델과 JSON 응답 파싱."""

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import get_env, get_settings


def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    return ChatOpenAI(
        model=settings["llm"]["chat_model"],
        temperature=settings["llm"]["temperature"],
        max_retries=settings["llm"].get("max_retries", 2),
        timeout=settings["llm"].get("timeout_sec"),
        api_key=get_env("OPENAI_API_KEY"),
        base_url=get_env("OPENAI_BASE_URL", required=False),
    )


def parse_json_list(text: str) -> list:
    """모델 응답에서 JSON 배열 하나를 꺼낸다. 코드 블록이나 앞뒤 설명이 붙어도 된다. 실패하면 빈 목록."""
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def parse_json_object(text: str) -> dict:
    """모델 응답에서 JSON 객체 하나를 꺼낸다. 실패하면 빈 dict."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
