#!/usr/bin/env python3
"""Point d'entrée de l'atelier IA Beaver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from beaver.app import App  # noqa: E402


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
