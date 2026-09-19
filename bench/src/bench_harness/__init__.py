"""bench_harness: DGX Spark 上の推論サーバーを計測するコマンドラインの道具のパッケージ。"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bench-harness")
except PackageNotFoundError:  # pragma: no cover - editable install では通常発生しない
    __version__ = "0.0.0"

__all__ = ["__version__"]
