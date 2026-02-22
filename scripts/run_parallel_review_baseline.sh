#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_parallel_review_baseline.sh [options]

Options:
  -n, --count <num>   Number of parallel codex workers (default: 30)
  -m, --model <name>  Optional model override passed to codex exec
  -h, --help          Show this help

Notes:
  Live worker status table (in-place redraw) is shown automatically on TTY terminals.
EOF
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

COUNT=30
MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--count)
      if [[ $# -lt 2 ]]; then
        echo "Error: missing value for $1" >&2
        usage
        exit 1
      fi
      if ! is_positive_int "$2"; then
        echo "Error: --count must be a positive integer" >&2
        exit 1
      fi
      COUNT="$2"
      shift 2
      ;;
    -m|--model)
      if [[ $# -lt 2 ]]; then
        echo "Error: missing value for $1" >&2
        usage
        exit 1
      fi
      MODEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v codex >/dev/null 2>&1; then
  echo "Error: codex command not found in PATH" >&2
  exit 1
fi

INTERACTIVE=0
if [[ -t 1 ]]; then
  INTERACTIVE=1
fi

LIVE_UI=0
if (( INTERACTIVE )) && command -v tput >/dev/null 2>&1 && [[ -n "${TERM:-}" && "${TERM}" != "dumb" ]]; then
  LIVE_UI=1
fi

UI_INITIALIZED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_NAME="parallel_review_baseline"
TEST_DIR="${ROOT_DIR}/out/${TEST_NAME}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${TEST_DIR}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${LOG_DIR}"

echo "Starting test '${TEST_NAME}' with ${COUNT} workers"
echo "Run directory: ${RUN_DIR}"

declare -a PIDS=()
declare -a WORKERS=()
declare -a WORKER_STATUS=()
declare -a WORKER_STARTED_AT=()
declare -a WORKER_ENDED_AT=()
declare -a STATUSES=()

format_duration() {
  local total_seconds="$1"
  if (( total_seconds < 0 )); then
    total_seconds=0
  fi
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

render_status_table() {
  local now
  now="$(date +%s)"

  if (( LIVE_UI )); then
    if (( UI_INITIALIZED == 0 )); then
      tput civis >/dev/null 2>&1 || true
      UI_INITIALIZED=1
    fi
    tput cup 0 0 >/dev/null 2>&1 || true
    tput ed >/dev/null 2>&1 || true
  fi

  echo "Test: ${TEST_NAME} | Run: ${RUN_ID}"
  echo "Run directory: ${RUN_DIR}"
  echo
  printf "%-12s %-10s %-14s %-10s %-s\n" "Worker" "PID" "Status" "Elapsed" "Report"
  printf "%-12s %-10s %-14s %-10s %-s\n" "------------" "----------" "--------------" "----------" "----------------------"
  for idx in "${!WORKERS[@]}"; do
    local start_ts="${WORKER_STARTED_AT[$idx]}"
    local end_ts="${WORKER_ENDED_AT[$idx]}"
    local elapsed_seconds
    if [[ "${WORKER_STATUS[$idx]}" == "running" ]]; then
      elapsed_seconds=$((now - start_ts))
    else
      elapsed_seconds=$((end_ts - start_ts))
    fi
    local elapsed_text
    elapsed_text="$(format_duration "${elapsed_seconds}")"

    printf "%-12s %-10s %-14s %-10s %-s\n" \
      "worker_${WORKERS[$idx]}" \
      "${PIDS[$idx]}" \
      "${WORKER_STATUS[$idx]}" \
      "${elapsed_text}" \
      "review_${WORKERS[$idx]}.md"
  done
  echo
  echo "Progress: completed=${completed_count}/${COUNT}, ok=${success_count}, failed=${fail_count}"
}

cleanup_ui() {
  if (( LIVE_UI )) && (( UI_INITIALIZED == 1 )); then
    tput cnorm >/dev/null 2>&1 || true
    UI_INITIALIZED=0
  fi
}

stop_running_workers() {
  for idx in "${!PIDS[@]}"; do
    if [[ "${WORKER_STATUS[$idx]:-}" == "running" ]]; then
      kill "${PIDS[$idx]}" >/dev/null 2>&1 || true
    fi
  done
}

is_worker_done() {
  local pid="$1"
  local stat
  stat="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
  [[ -z "${stat}" || "${stat}" == *Z* ]]
}

on_interrupt() {
  echo
  echo "Interrupted. Stopping running workers..."
  stop_running_workers
  for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  cleanup_ui
  exit 130
}

trap on_interrupt INT TERM
trap cleanup_ui EXIT

for i in $(seq 1 "${COUNT}"); do
  worker_id="$(printf "%03d" "${i}")"
  started_at="$(date +%s)"
  review_file="${RUN_DIR}/review_${worker_id}.md"
  log_file="${LOG_DIR}/worker_${worker_id}.log"
  prompt=$(
    cat <<EOF
Сделай code review проекта в директории ./cobaia.
Фокус: баги, риски регрессий и пробелы в тестах.
Итог запиши только в markdown-файл: ${review_file}
Не изменяй другие файлы.
EOF
  )

  if [[ -n "${MODEL}" ]]; then
    codex exec --full-auto --sandbox workspace-write -C "${ROOT_DIR}" --model "${MODEL}" "${prompt}" >"${log_file}" 2>&1 &
  else
    codex exec --full-auto --sandbox workspace-write -C "${ROOT_DIR}" "${prompt}" >"${log_file}" 2>&1 &
  fi

  PIDS+=("$!")
  WORKERS+=("${worker_id}")
  WORKER_STATUS+=("running")
  WORKER_STARTED_AT+=("${started_at}")
  WORKER_ENDED_AT+=("0")
done

success_count=0
fail_count=0
completed_count=0

if (( INTERACTIVE )); then
  render_status_table
fi

while (( completed_count < COUNT )); do
  progress_made=0

  for idx in "${!PIDS[@]}"; do
    if [[ "${WORKER_STATUS[$idx]}" != "running" ]]; then
      continue
    fi

    pid="${PIDS[$idx]}"
    worker="${WORKERS[$idx]}"
    if ! is_worker_done "${pid}"; then
      continue
    fi

    if wait "${pid}"; then
      WORKER_STATUS[$idx]="ok"
      WORKER_ENDED_AT[$idx]="$(date +%s)"
      success_count=$((success_count + 1))
    else
      rc=$?
      WORKER_STATUS[$idx]="failed(${rc})"
      WORKER_ENDED_AT[$idx]="$(date +%s)"
      fail_count=$((fail_count + 1))
    fi

    completed_count=$((completed_count + 1))
    progress_made=1

    if (( ! INTERACTIVE )); then
      elapsed_seconds=$((WORKER_ENDED_AT[$idx] - WORKER_STARTED_AT[$idx]))
      elapsed_text="$(format_duration "${elapsed_seconds}")"
      echo "worker_${worker}: ${WORKER_STATUS[$idx]} (${completed_count}/${COUNT}, elapsed=${elapsed_text})"
    fi
  done

  if (( INTERACTIVE )); then
    render_status_table
  fi

  if (( completed_count < COUNT && progress_made == 0 )); then
    sleep 1
  fi
done

for idx in "${!WORKERS[@]}"; do
  elapsed_seconds=$((WORKER_ENDED_AT[$idx] - WORKER_STARTED_AT[$idx]))
  elapsed_text="$(format_duration "${elapsed_seconds}")"
  STATUSES+=("worker_${WORKERS[$idx]}: ${WORKER_STATUS[$idx]} (${elapsed_text})")
done

cleanup_ui

summary_file="${RUN_DIR}/summary.md"
{
  echo "# Parallel Codex Review Summary"
  echo
  echo "- Test: \`${TEST_NAME}\`"
  echo "- Run ID: \`${RUN_ID}\`"
  echo "- Workers requested: ${COUNT}"
  if [[ -n "${MODEL}" ]]; then
    echo "- Model override: \`${MODEL}\`"
  fi
  echo "- Success: ${success_count}"
  echo "- Failed: ${fail_count}"
  echo
  echo "## Worker status"
  for status in "${STATUSES[@]}"; do
    echo "- ${status}"
  done
  echo
  echo "## Files"
  echo "- Reviews: \`review_*.md\`"
  echo "- Logs: \`logs/worker_*.log\`"
} >"${summary_file}"

echo "Completed: success=${success_count}, failed=${fail_count}"
echo "Summary: ${summary_file}"
