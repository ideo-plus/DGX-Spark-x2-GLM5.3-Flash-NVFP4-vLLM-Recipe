# モデルが書いたコードを採点するための、隔離のイメージ (要件 5.4、タスク 6.1)。
#
# 公式の Python のスリムなイメージに、numpy だけを足す。HumanEval+ の検査のプログラムが
# numpy を使うためで、ほかには何も入れない。土台はダイジェストで、numpy は版とホイールの
# ハッシュで固定する。ネットワークを使うのは、このビルドの間だけである。採点のときは
# `--network none` で動かす (bench_harness.scoring.sandbox)。
#
# 作り方 (bench/ で):
#   docker build -t bench-sandbox:py3.13-numpy2.5.3 sandbox/
#   docker image inspect bench-sandbox:py3.13-numpy2.5.3 --format '{{.Id}}'
# 出てきた識別子を、config/profiles.toml の `sandbox.image_digest` に書く。
FROM python@sha256:8d9d0b8bcf6506481eae4907c18f5e3e7902e629f5f6d684f9e7c32e85e3ddf0

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# 権限のない利用者で動かす (起動の側でも `--user` を付ける)
USER 65534:65534
WORKDIR /tmp
# 起動の引数 (`python -I -` と、標準入力でのコードの受け渡し) は、採点の側が決める
