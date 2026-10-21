"""한국어 위키백과 MediaWiki API 클라이언트. (F2004, NF2006)

- 모든 요청에 연락처가 들어간 User-Agent(WIKI_USER_AGENT)를 붙인다.
- 요청 사이 간격을 두고, 실패하면 간격을 늘려 재시도한다.
"""

import html
import re
import time

import requests

from src.config import get_env, get_settings

LICENSE = "CC BY-SA 4.0"


def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


class WikiClient:
    def __init__(self, session: requests.Session | None = None, cfg: dict | None = None, user_agent: str | None = None):
        self.cfg = cfg or get_settings()["wiki"]
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent or get_env("WIKI_USER_AGENT")
        self._last_call = 0.0

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json", "formatversion": 2}
        last_error = None
        for attempt in range(1, self.cfg["max_retries"] + 1):
            wait = self.cfg["request_interval_sec"] - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self.session.get(self.cfg["api_url"], params=params, timeout=self.cfg.get("timeout_sec", 10))
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"위키 API 오류: {data['error'].get('info')}")
                return data
            except (requests.RequestException, ValueError, RuntimeError) as e:
                last_error = e
                time.sleep(self.cfg["request_interval_sec"] * 2**attempt)
        raise RuntimeError(f"위키 요청 실패: {last_error}")

    def search(self, query: str, limit: int) -> list[dict]:
        """검색 후보: pageid, title, snippet(태그 제거)"""
        data = self._get({"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "srnamespace": 0})
        return [
            {"pageid": r["pageid"], "title": r["title"], "snippet": _strip_tags(r.get("snippet", ""))}
            for r in data.get("query", {}).get("search", [])
        ]

    def fetch_page(self, pageid: int) -> dict | None:
        """본문(평문)과 원문 URL. 전체 본문 추출은 한 번에 한 문서만 된다."""
        data = self._get(
            {
                "action": "query",
                "prop": "extracts|info",
                "explaintext": 1,
                "exsectionformat": "plain",
                "inprop": "url",
                "pageids": pageid,
                "redirects": 1,
            }
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        page = pages[0]
        return {
            "pageid": page["pageid"],
            "title": page["title"],
            "url": page.get("fullurl") or f"https://ko.wikipedia.org/?curid={page['pageid']}",
            "text": (page.get("extract") or "").strip(),
        }
