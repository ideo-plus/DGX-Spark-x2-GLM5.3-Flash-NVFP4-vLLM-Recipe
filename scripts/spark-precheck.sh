#!/usr/bin/env bash
# 計測の前に、2 台の DGX Spark が「測る相手だけが動いていて、誰も使っていない」状態かを確かめる。
#
# 読み取りだけを行う (ssh で nvidia-smi と ps を読み、対象サーバーの /metrics を GET する)。
# 何も止めず、何も変えない。条件を満たさなければ、理由を表示して 1 で終わる。
#
# 使い方 (作業用の Mac で):
#   scripts/spark-precheck.sh [対象サーバーの URL] [GPU を使ってよいプロセスの名前の正規表現]
#   例: scripts/spark-precheck.sh http://10.0.1.60:8001 '^VLLM::Worker'
#
# 確かめること:
#   1. どちらの機械でも、GPU を使っているプロセスが、測る相手のものだけであること
#   2. 対象サーバーに、処理中の要求も、待っている要求もないこと
set -u

BASE_URL="${1:-http://10.0.1.60:8001}"
ALLOWED="${2:-^VLLM::Worker}"
HOSTS=(spark-153d spark-5083)
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)
failed=0

echo "# spark-precheck $(date -u +%Y-%m-%dT%H:%M:%SZ) target=${BASE_URL} allowed=${ALLOWED}"

for host in "${HOSTS[@]}"; do
  apps="$(ssh "${SSH_OPTS[@]}" "$host" \
    'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader' 2>&1)"
  if [ $? -ne 0 ]; then
    echo "NG  ${host}: GPU のプロセスを読めなかった: ${apps}"
    failed=1
    continue
  fi
  count=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    count=$((count + 1))
    name="$(printf '%s' "$line" | awk -F', ' '{print $2}')"
    if printf '%s' "$name" | grep -Eq "$ALLOWED"; then
      echo "ok  ${host}: ${line}"
    else
      echo "NG  ${host}: 測る相手ではないプロセスが GPU を使っている: ${line}"
      failed=1
    fi
  done <<<"$apps"
  if [ "$count" -eq 0 ]; then
    echo "NG  ${host}: GPU を使っているプロセスがない (測る相手が動いていない)"
    failed=1
  fi
done

metrics="$(curl -s -m 5 "${BASE_URL}/metrics")"
if [ -z "$metrics" ]; then
  echo "NG  ${BASE_URL}/metrics を読めなかった"
  failed=1
else
  for key in num_requests_running num_requests_waiting; do
    value="$(printf '%s\n' "$metrics" | awk -v k="vllm:${key}{" 'index($0, k) == 1 {s += $NF} END {print s + 0}')"
    if [ "$value" = "0" ]; then
      echo "ok  ${key} = 0"
    else
      echo "NG  ${key} = ${value} (誰かが対象サーバーを使っている)"
      failed=1
    fi
  done
fi

if [ "$failed" -ne 0 ]; then
  echo "# 結果: NG — 計測を始めないこと"
  exit 1
fi
echo "# 結果: OK"
