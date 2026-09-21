"""試験の共通の下ごしらえ。

`tests/` は `pyproject.toml` の `[tool.pytest.ini_options] pythonpath` と
`[tool.mypy] mypy_path` に入れてあるので、`tests/unit/` と `tests/integration/`
のどちらからでも、`tests/` 直下の助けを import できる (`fake_runner.py`、
`fake_vllm.py` は、あとのタスク (1.4、1.5) で足す)。

task 1.1 (骨組み) では、疎通の試験だけを置くので、共通のフィクスチャはまだない。
"""

from __future__ import annotations
