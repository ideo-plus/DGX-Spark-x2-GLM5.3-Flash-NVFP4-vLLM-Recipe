"""serving_kit: DGX Spark 2 台に vLLM の推論サーバーを立てるための道具のパッケージ。"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("serving-kit")
except PackageNotFoundError:  # pragma: no cover - editable install では通常発生しない
    __version__ = "0.0.0"

__all__ = ["__version__"]
