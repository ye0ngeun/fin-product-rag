"""테스트를 로컬 환경의 실제 파일과 떼어 놓는다.

색인·수집을 한 번이라도 한 로컬 환경에는 data/processed/wiki_cache.json 이 있다.
테스트가 이 캐시를 읽으면 위키를 부르지 않아 결과가 달라지고, 쓰면 실제 캐시가 더러워진다.
"""

import pytest

from src.config import get_settings


@pytest.fixture(autouse=True)
def isolated_wiki_cache(tmp_path):
    """위키 캐시 경로를 테스트마다 빈 임시 파일로 바꾼다 (get_settings 는 캐시된 dict 를 돌려준다)."""
    wiki = get_settings()["wiki"]
    old = wiki["cache_file"]
    wiki["cache_file"] = tmp_path / "wiki_cache.json"
    yield
    wiki["cache_file"] = old
