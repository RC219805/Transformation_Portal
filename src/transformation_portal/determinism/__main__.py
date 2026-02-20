from __future__ import annotations

from .bootstrap import bootstrap

bootstrap()


def main() -> None:
    # Import after bootstrap so env/thread controls apply before NumPy stack loads.
    from .cli import app

    app()


if __name__ == "__main__":
    main()
