"""CLI: validate the raw file, clean it and write data/processed.csv + results/data_report.json."""
import argparse
import json
import logging

from src.config import ProjectPaths
from src.data import build_processed
from src.preprocess import EMOJI_AVAILABLE


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-emoji", action="store_true", help="skip emoji->words conversion")
    ap.add_argument("--allow-missing-emoji-package", action="store_true",
                    help="continue even if the 'emoji' package is not installed (not recommended)")
    a = ap.parse_args(argv)
    if not a.no_emoji and not EMOJI_AVAILABLE and not a.allow_missing_emoji_package:
        raise SystemExit("The 'emoji' package is not installed (pip install emoji). "
                         "Needed for emoji->word conversion; or pass --no-emoji / --allow-missing-emoji-package.")
    paths = ProjectPaths()
    if not paths.raw_file.exists():
        raise SystemExit(f"Missing {paths.raw_file}. Put the iSarcasmEval English training file there.")
    print(json.dumps(build_processed(paths, convert_emoji=not a.no_emoji), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
