"""제출 zip 만들기: AI코어프로젝트2_[주제명]_지역_반_성명.zip

넣는 것
- git 이 관리하는 파일 전부 (소스, docs/, README.md, sources.md, 설정, 테스트)
- docs/screenshots/ 의 화면 캡처와 명령 출력 txt (git 제외 폴더)
- data/processed/eval/*.md -> 검증결과/ (RAG 전후 비교, 검색 비교. 금감원 공시값이 있어 git 에서 뺀 파일)

넣지 않는 것: .env, data/raw, chroma_db, 그 밖의 git 제외 파일.
묶기 전에 .env 의 값(API 키 등)이 어느 파일에도 들어 있지 않은지 검사하고, 있으면 멈춘다. (NF2003)

    python scripts/make_submission.py --topic 불빛하나 --region 서울 --cls 1 --name 홍길동
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = ROOT / "docs" / "screenshots"
EVAL_DIR = ROOT / "data" / "processed" / "eval"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [ROOT / p for p in out.decode("utf-8").split("\0") if p and (ROOT / p).is_file()]


def env_values(path: Path) -> list[str]:
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            values.append(line.split("=", 1)[1].strip().strip("'\""))
    return values


def secret_values() -> list[str]:
    """.env 의 값 중 8자 이상이고 .env.example 에 없는 것 (키·연락처). GMS 주소 같은 공개 기본값은 뺀다."""
    public = set(env_values(ROOT / ".env.example"))
    return [v for v in env_values(ROOT / ".env") if len(v) >= 8 and v not in public]


def main() -> None:
    parser = argparse.ArgumentParser(description="제출 zip 만들기")
    parser.add_argument("--topic", default="불빛하나", help="주제명")
    parser.add_argument("--region", required=True, help="지역 (예: 서울)")
    parser.add_argument("--cls", required=True, help="반 (예: 1)")
    parser.add_argument("--name", required=True, help="성명")
    args = parser.parse_args()

    entries: list[tuple[Path, str]] = [(p, p.relative_to(ROOT).as_posix()) for p in tracked_files()]
    shots = sorted(p for p in SCREENSHOT_DIR.glob("*") if p.is_file() and p.name != ".gitkeep")
    entries += [(p, p.relative_to(ROOT).as_posix()) for p in shots]
    evals = sorted(EVAL_DIR.glob("*.md"))
    entries += [(p, f"검증결과/{p.name}") for p in evals]

    secrets = secret_values()
    leaks = []
    for path, arc in entries:
        if path.suffix.lower() in IMAGE_EXT:
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        leaks += [arc for s in secrets if s in text]
    if leaks:
        sys.exit(f".env 의 값이 들어 있는 파일이 있어 멈춥니다: {sorted(set(leaks))}")

    cls = args.cls if args.cls.endswith("반") else f"{args.cls}반"
    zip_name = f"AI코어프로젝트2_{args.topic}_{args.region}_{cls}_{args.name}.zip"
    out = ROOT / zip_name
    folder = zip_name[:-4]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, arc in entries:
            zf.write(path, f"{folder}/{arc}")

    print(f"{out.name}: 파일 {len(entries)}개 (스크린샷 {len(shots)}, 검증결과 {len(evals)})")
    if not shots:
        print("주의: docs/screenshots/ 에 스크린샷이 없습니다.")
    if not evals:
        print("주의: data/processed/eval/ 에 비교 결과가 없습니다. rag_compare, search_compare 를 먼저 실행하세요.")


if __name__ == "__main__":
    main()
