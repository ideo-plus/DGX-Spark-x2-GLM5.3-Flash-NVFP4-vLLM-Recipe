"""種から決まる合成の入力を作る部品 (task 2.5: corpus/synth)。"""

from bench_harness.corpus.synth import (
    PREFIX_NONCE_HEX_LEN,
    SyntheticCorpus,
    TemplateCorpus,
    prefix_header_line,
)

__all__ = [
    "PREFIX_NONCE_HEX_LEN",
    "SyntheticCorpus",
    "TemplateCorpus",
    "prefix_header_line",
]
