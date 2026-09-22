#!/bin/sh
# 準備専用の依頼を現在のブランチで実行する。Git の公開操作は行わない。
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
task_file=docs/tasks/tp2-preparation.md
test -f "$task_file"
exec takt --pipeline --skip-git --workflow spark-preparation --task "$(cat "$task_file")"
