"""
Regenerate the Markdown variable reference from the catalogue.

    python scripts/gen_variable_docs.py            # -> docs/source/variables.md
    python scripts/gen_variable_docs.py -o out.md
"""

import argparse
from pathlib import Path

from monarchs.variables import to_markdown

DEFAULT_OUT = Path("docs/source/variables.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"where to write the Markdown (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(to_markdown())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
