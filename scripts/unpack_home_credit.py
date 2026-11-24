#!/usr/bin/env python3
"""Распаковка Kaggle-архива Home Credit Default Risk."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.data.home_credit import HOME_CREDIT_DIR


def unpack_zip(zip_path: Path, output_dir: Path) -> list[Path]:
    """Распаковывает zip Home Credit в каталог данных.

    Args:
        zip_path: Путь к home-credit-default-risk.zip.
        output_dir: Куда распаковать CSV.

    Returns:
        Список распакованных файлов.

    Raises:
        FileNotFoundError: Если zip отсутствует.
        ValueError: Если файл не является zip-архивом.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}.")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a zip archive: {zip_path}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = output_dir / Path(info.filename).name
            if target.exists():
                extracted.append(target)
                continue
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def main() -> None:
    """CLI: распаковка Home Credit zip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zip_path",
        type=Path,
        nargs="?",
        default=Path.home() / "Downloads" / "home-credit-default-risk.zip",
        help="Путь к zip-архиву Kaggle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HOME_CREDIT_DIR,
        help="Каталог для CSV (по умолчанию data/home_credit)",
    )
    args = parser.parse_args()
    files = unpack_zip(args.zip_path, args.output_dir)
    print(f"Ready: {args.output_dir} ({len(files)} files)")
    for path in files:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
