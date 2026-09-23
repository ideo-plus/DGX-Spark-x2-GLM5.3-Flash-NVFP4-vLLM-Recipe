#!/bin/sh
# docs/rules/ 以下の Markdown の規則を、パスの順にすべて標準出力へ書く。
# Claude Code の UserPromptSubmit フックから呼ばれ、出力がそのまま文脈に入る。
# 規則が 1 つも読めないときは、黙って空を返さず、終了の値 1 で失敗を知らせる。
set -eu

root=${CLAUDE_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}
rules_dir="$root/docs/rules"

if [ ! -d "$rules_dir" ]; then
  echo "load-rules: $rules_dir がない" >&2
  exit 1
fi

files=$(find "$rules_dir" -type f -name '*.md' | LC_ALL=C sort)
if [ -z "$files" ]; then
  echo "load-rules: $rules_dir に規則 (*.md) が 1 つもない" >&2
  exit 1
fi

echo "以下は docs/rules/ の規則である。作業を始める前に確認し、守ること。"
printf '%s\n' "$files" | while IFS= read -r file; do
  printf '\n<!-- %s -->\n' "${file#"$root"/}"
  cat "$file"
done
