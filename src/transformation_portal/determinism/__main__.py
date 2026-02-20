from __future__ import annotations

from .bootstrap import bootstrap

bootstrap()

from .cli import app

if __name__ == "__main__":
    app()
