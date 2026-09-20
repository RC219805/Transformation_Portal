"""Additive LuxDepthV5 CLI using the existing photographic command lifecycle."""

from typing import Any

from transformation_portal.lux_depth_v4.__main__ import _main

from .lifecycle import LuxDepthV5Request, prepare


class _CLIProfile:
    name = "lux-depth-v5"
    description = (
        "LuxDepthV5 candidate: explicit depth evidence, conservative detail recovery, and reliable photographic edits."
    )
    request_type = LuxDepthV5Request
    prepare = staticmethod(prepare)

    @staticmethod
    def run(*args: Any, **kwargs: Any) -> Any:
        from .pipeline import run

        return run(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return _main(argv, profile=_CLIProfile)


if __name__ == "__main__":
    raise SystemExit(main())
