#!/bin/sh
# TAKT が claude を起動するときの入口 (TAKT_CLAUDE_CLI_PATH に絶対パスで指定する)。
# TAKT_CLAUDE_ACCOUNT_DIR のアカウント設定 (CLAUDE_CONFIG_DIR) で claude を起動する。
# 指定がない、またはディレクトリがないときは、既定のアカウントに黙って戻さず失敗する。
# 通常は scripts/run-takt.sh から使う。
set -eu

account_dir=${TAKT_CLAUDE_ACCOUNT_DIR:-}
if [ -z "$account_dir" ]; then
    echo "takt-claude: TAKT_CLAUDE_ACCOUNT_DIR が指定されていない" >&2
    exit 1
fi
if [ ! -d "$account_dir" ]; then
    echo "takt-claude: アカウントの設定ディレクトリがない: $account_dir" >&2
    exit 1
fi

CLAUDE_CONFIG_DIR=$account_dir
export CLAUDE_CONFIG_DIR
exec "${TAKT_CLAUDE_REAL_CLI:-claude}" "$@"
