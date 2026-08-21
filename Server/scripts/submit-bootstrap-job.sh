#!/usr/bin/env bash
# Submit SteerLab's one-time environment bootstrap as a small CPU Slurm job.
# This helper itself runs on the login/submit host; it performs only mkdir,
# sbatch, lightweight squeue polling, and incremental log reads. Conda, pip,
# and profile validation execute inside the allocation.
#
# RECOVERY SEMANTICS (2026-08-13). Submitting and waiting are separate acts,
# because holding one SSH session open for the whole bootstrap is how a
# submitted job gets LOST: if the Mac slept or the connection dropped after
# sbatch but before the poll loop ended, the caller had no record of the job,
# could not adopt it, and a retry submitted a second one.
#
#   * SUBMIT (the default) sbatches the job, writes a durable breadcrumb next
#     to the job's status file, prints the job id / status file / log, and
#     EXITS. Nothing is held open.
#   * --status <job> reads ONE verdict (the job's own status file first, the
#     queue second) and exits. Callers poll by re-invoking it, so every poll
#     is its own short-lived connection and an interrupted poll loses nothing.
#   * --wait keeps the historical foreground behaviour — submit, then stream
#     the job log until the status file lands — for a human at a login shell.
#
# Before submitting, a bootstrap job that is still pending/running for this
# workspace is ADOPTED (announced with STEERLAB_BOOTSTRAP_ADOPT=<id>) instead
# of resubmitted; --force-new overrides. A FAILED queue query adopts too: an
# unproven death never licenses a resubmit.

set -u

usage() {
  cat <<'EOF'
usage: submit-bootstrap-job.sh [job flags] [--dry-run] [--wait] -- [bootstrap flags]
       submit-bootstrap-job.sh --status --job-workspace PATH [--job-id N]
  --bootstrap-script PATH  bootstrap.sh on the shared filesystem
  --job-workspace PATH     scratch/work directory for script, log, and status
  --job-partition NAME     CPU partition used for environment setup
  --job-cpus N             CPUs per task (default: STEERLAB_SETUP_CPUS, else 4)
  --job-memory VALUE       Slurm memory (default: STEERLAB_SETUP_MEMORY, 16G)
  --job-walltime H:M:S     walltime (default: STEERLAB_SETUP_WALLTIME, 02:00:00)
  --job-extra-sbatch ARG   verbatim #SBATCH argument for this job; repeatable
                           (default: STEERLAB_SETUP_EXTRA_SBATCH, word-split)
  --job-account NAME       optional Slurm account
  --squeue-command NAME    queue-query executable (default: env or squeue)
  --dry-run                show bootstrap's plan without submitting a job
  --wait                   stay in the foreground and stream the job log
                           (default: submit, print the job id, and exit)
  --force-new              submit even when a bootstrap job is already in
                           flight for this workspace
  --status                 report one verdict for an already-submitted job:
                           STEERLAB_BOOTSTRAP_STATUS=<pending|running|
                           completed-ok|completed-code-N|vanished|unknown>
  --job-id N               the job to report on (default: the breadcrumb's)
  --status-file PATH       that job's status file (default: the breadcrumb's)
EOF
}

MODE="submit"
BOOTSTRAP_SCRIPT=""
JOB_WORKSPACE=""
# The setup job's shape is SITE DATA (WP5 Step 9, audit c19): the app resolves
# `scheduler.setupJob` and passes it as the flags below, which is what the
# reviewed bootstrap plan hash covers. These defaults are the second source —
# the rendered site env file, for a hand run of this helper on the login host —
# and the literals are this script's own built-in fallbacks, in step with
# ClusterProvisioner.setupJobDefaults. They are NOT any site's declared facts:
# `steerlab-cli cluster preview --site <id> --job-class setup` shows those.
JOB_PARTITION="${STEERLAB_SETUP_PARTITION:-}"
JOB_CPUS="${STEERLAB_SETUP_CPUS:-4}"
JOB_MEMORY="${STEERLAB_SETUP_MEMORY:-16G}"
JOB_WALLTIME="${STEERLAB_SETUP_WALLTIME:-02:00:00}"
# Word-split like every other rendered directive list (STEERLAB_SLURM_EXTRA_SBATCH).
# shellcheck disable=SC2206
JOB_EXTRA_SBATCH=(${STEERLAB_SETUP_EXTRA_SBATCH:-})
JOB_EXTRA_FROM_FLAGS=0
JOB_ACCOUNT=""
SQUEUE_COMMAND="${STEERLAB_SLURM_SQUEUE:-squeue}"
DRY_RUN=0
WAIT_FOR_JOB=0
FORCE_NEW=0
QUERY_JOB_ID=""
QUERY_STATUS_FILE=""
BOOTSTRAP_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --bootstrap-script) BOOTSTRAP_SCRIPT="$2"; shift 2 ;;
    --job-workspace)    JOB_WORKSPACE="$2"; shift 2 ;;
    --job-partition)    JOB_PARTITION="$2"; shift 2 ;;
    --job-cpus)         JOB_CPUS="$2"; shift 2 ;;
    --job-memory)       JOB_MEMORY="$2"; shift 2 ;;
    --job-walltime)     JOB_WALLTIME="$2"; shift 2 ;;
    --job-extra-sbatch)
      # An explicit list REPLACES the environment's, never appends to it: the
      # caller that names directives is stating the whole set.
      if [ "$JOB_EXTRA_FROM_FLAGS" -eq 0 ]; then JOB_EXTRA_SBATCH=(); fi
      JOB_EXTRA_FROM_FLAGS=1
      JOB_EXTRA_SBATCH+=("$2"); shift 2 ;;
    --job-account)      JOB_ACCOUNT="$2"; shift 2 ;;
    --squeue-command)   SQUEUE_COMMAND="$2"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    --wait)             WAIT_FOR_JOB=1; shift ;;
    --force-new)        FORCE_NEW=1; shift ;;
    --status)           MODE="status"; shift ;;
    --job-id)           QUERY_JOB_ID="$2"; shift 2 ;;
    --status-file)      QUERY_STATUS_FILE="$2"; shift 2 ;;
    --)                 shift; BOOTSTRAP_ARGS=("$@"); break ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "bootstrap-job: unknown flag: $1" >&2; usage >&2; exit 64 ;;
  esac
done

fail() { echo "bootstrap-job: $*" >&2; exit 64; }
safe_token() { printf '%s' "$1" | grep -Eq '^[A-Za-z0-9_.-]+$'; }

# The Slurm job name the guard looks for when the breadcrumb is gone.
JOB_NAME="steerlab-bootstrap"
BREADCRUMB_NAME="steerlab-bootstrap.pending"

# --- the durable job record and the one-line verdict -----------------------
#
# Two independent records, both written by the side that owns the fact: the
# JOB writes its exit code into STATUS_FILE from inside the allocation, and
# the SUBMITTER writes the breadcrumb (job id, status file, log) next to it.
# Either one alone is enough to find a job again after a dropped connection.

read_breadcrumb_field() {   # file key
  [ -f "$1" ] || return 0
  sed -n "s/^$2=//p" "$1" | sed -n '1p'
}

read_status_code() {        # status-file -> the recorded exit code, or nothing
  [ -n "${1:-}" ] || return 0
  [ -f "$1" ] || return 0
  _raw="$(tr -d '[:space:]' < "$1")"
  printf '%s' "$_raw" | grep -Eq '^[0-9]+$' || return 0
  printf '%s' "$_raw"
}

emit_verdict() {            # verdict job [exit] [status-file] [queue-state]
  _line="STEERLAB_BOOTSTRAP_STATUS=$1 job=$2"
  [ -n "${3:-}" ] && _line="$_line exit=$3"
  [ -n "${4:-}" ] && _line="$_line statusFile=$4"
  [ -n "${5:-}" ] && _line="$_line state=$5"
  printf '%s\n' "$_line"
}

emit_completed() {          # job code [status-file]
  if [ "$2" -eq 0 ]; then
    emit_verdict "completed-ok" "$1" "0" "${3:-}"
  else
    emit_verdict "completed-code-$2" "$1" "$2" "${3:-}"
  fi
}

verdict_word() {
  printf '%s' "$1" | sed -n 's/^STEERLAB_BOOTSTRAP_STATUS=\([^ ]*\).*/\1/p'
}

# The whole classification, in one place, used by --status AND by the
# pre-submit guard so both agree on what "still in flight" means.
classify_job() {            # job [status-file]
  _jid="$1"
  _sfile="${2:-}"
  _code="$(read_status_code "$_sfile")"
  if [ -n "$_code" ]; then emit_completed "$_jid" "$_code" "$_sfile"; return 0; fi
  _queue="$("$SQUEUE_COMMAND" -h -j "$_jid" -o '%T' 2>&1)"
  if [ $? -ne 0 ]; then
    # A broken/restricted query is NOT evidence the job ended.
    emit_verdict "unknown" "$_jid" "" "$_sfile"
    return 0
  fi
  _state="$(printf '%s\n' "$_queue" | sed -n '1p' | tr -d '[:space:]')"
  case "$_state" in
    PENDING|CONFIGURING|REQUEUED|RESIZING|SUSPENDED)
      emit_verdict "pending" "$_jid" "" "$_sfile" "$_state" ;;
    RUNNING|COMPLETING)
      emit_verdict "running" "$_jid" "" "$_sfile" "$_state" ;;
    "")
      # The job writes its status file just before exiting, so a shared
      # filesystem can make it visible only after the queue has dropped the
      # job. Re-read once before calling the job vanished.
      _code="$(read_status_code "$_sfile")"
      if [ -n "$_code" ]; then
        emit_completed "$_jid" "$_code" "$_sfile"
      else
        emit_verdict "vanished" "$_jid" "" "$_sfile"
      fi
      ;;
    *)
      emit_verdict "vanished" "$_jid" "" "$_sfile" "$_state" ;;
  esac
}

[ -n "$JOB_WORKSPACE" ] || fail "--job-workspace is required"
printf '%s' "$JOB_WORKSPACE" | grep -Eq '^[^[:space:]]+$' \
  || fail "job workspace may not contain whitespace"
safe_token "$SQUEUE_COMMAND" || fail "invalid --squeue-command: $SQUEUE_COMMAND"
BREADCRUMB="$JOB_WORKSPACE/$BREADCRUMB_NAME"

# --- status mode: one verdict, no scheduler writes, no waiting -------------
#
# Always exits 0 for a readable verdict. The verdict LINE carries the outcome;
# the exit code is left to mean "could this host answer at all", so a caller
# reaching this over ssh can tell a failed connection from a failed bootstrap.
if [ "$MODE" = "status" ]; then
  STATUS_JOB_ID="$QUERY_JOB_ID"
  STATUS_FILE="$QUERY_STATUS_FILE"
  CRUMB_ID="$(read_breadcrumb_field "$BREADCRUMB" jobID)"
  [ -n "$STATUS_JOB_ID" ] || STATUS_JOB_ID="$CRUMB_ID"
  [ -n "$STATUS_JOB_ID" ] \
    || fail "--status needs --job-id (no job breadcrumb at $BREADCRUMB)"
  printf '%s' "$STATUS_JOB_ID" | grep -Eq '^[0-9]+$' \
    || fail "invalid --job-id: $STATUS_JOB_ID"
  if [ -z "$STATUS_FILE" ] && [ "$CRUMB_ID" = "$STATUS_JOB_ID" ]; then
    STATUS_FILE="$(read_breadcrumb_field "$BREADCRUMB" statusFile)"
  fi
  VERDICT_LINE="$(classify_job "$STATUS_JOB_ID" "$STATUS_FILE")"
  printf '%s\n' "$VERDICT_LINE"
  # On a terminal verdict, forward the job log's tail so the caller can parse
  # bootstrap's own JSON report (its LAST line) without a second round trip.
  case "$(verdict_word "$VERDICT_LINE")" in
    completed-*|vanished)
      STATUS_LOG="$(read_breadcrumb_field "$BREADCRUMB" log)"
      if [ "$CRUMB_ID" != "$STATUS_JOB_ID" ] || [ -z "$STATUS_LOG" ]; then
        [ -n "$STATUS_FILE" ] && STATUS_LOG="${STATUS_FILE%.status}.log"
      fi
      [ -n "$STATUS_LOG" ] && [ -f "$STATUS_LOG" ] && tail -n 500 "$STATUS_LOG"
      ;;
  esac
  exit 0
fi

# --- submit mode -----------------------------------------------------------

[ -n "$BOOTSTRAP_SCRIPT" ] || fail "--bootstrap-script is required"
[ -f "$BOOTSTRAP_SCRIPT" ] || fail "bootstrap script not found: $BOOTSTRAP_SCRIPT"
safe_token "$JOB_PARTITION" || fail "invalid --job-partition: $JOB_PARTITION"
printf '%s' "$JOB_CPUS" | grep -Eq '^[1-9][0-9]*$' \
  || fail "--job-cpus must be a positive integer"
printf '%s' "$JOB_MEMORY" | grep -Eq '^[1-9][0-9]*[KMGTP]?$' \
  || fail "invalid --job-memory: $JOB_MEMORY"
printf '%s' "$JOB_WALLTIME" | grep -Eq '^[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$' \
  || fail "invalid --job-walltime: $JOB_WALLTIME"
if [ -n "$JOB_ACCOUNT" ]; then
  safe_token "$JOB_ACCOUNT" || fail "invalid --job-account: $JOB_ACCOUNT"
fi
# Verbatim #SBATCH arguments go into a generated script unquoted, so they are
# held to the shape a site fact has: one `--flag[=value]` token, no whitespace,
# no shell metacharacters. A directive that does not look like one is refused
# rather than smuggled into the job script.
if [ "${#JOB_EXTRA_SBATCH[@]}" -gt 0 ]; then
  for _extra in "${JOB_EXTRA_SBATCH[@]}"; do
    printf '%s' "$_extra" \
      | grep -Eq '^--[A-Za-z0-9][A-Za-z0-9_-]*(=[A-Za-z0-9_.:,/@%+-]+)?$' \
      || fail "invalid --job-extra-sbatch: $_extra"
  done
fi

# The dry run is intentionally scheduler-free: bootstrap.sh executes no work
# in this mode, so it is safe on the login host and returns the same JSON plan
# the real job will execute.
if [ "$DRY_RUN" -eq 1 ]; then
  if [ "${#BOOTSTRAP_ARGS[@]}" -gt 0 ]; then
    exec bash "$BOOTSTRAP_SCRIPT" "${BOOTSTRAP_ARGS[@]}" --dry-run
  else
    exec bash "$BOOTSTRAP_SCRIPT" --dry-run
  fi
fi

command -v "$SQUEUE_COMMAND" >/dev/null 2>&1 \
  || fail "queue query command is not available on this host: $SQUEUE_COMMAND"
mkdir -p "$JOB_WORKSPACE" || fail "could not create job workspace: $JOB_WORKSPACE"

# --- idempotence guard: adopt an in-flight job rather than submit a second --
#
# The breadcrumb is authoritative FOR THIS WORKSPACE. Only when it is missing
# entirely (an interrupted submit, a cleaned scratch dir) do we fall back to
# asking the queue for a live job of ours by name — a broader question that
# could match another workspace's bootstrap, which is why it is the backstop
# and not the primary, and why --force-new exists.
ADOPT_JOB_ID=""
ADOPT_STATUS_FILE=""
ADOPT_LINE=""
if [ "$FORCE_NEW" -eq 0 ]; then
  if [ -f "$BREADCRUMB" ]; then
    CANDIDATE="$(read_breadcrumb_field "$BREADCRUMB" jobID)"
    if printf '%s' "$CANDIDATE" | grep -Eq '^[0-9]+$'; then
      CANDIDATE_STATUS="$(read_breadcrumb_field "$BREADCRUMB" statusFile)"
      CANDIDATE_LINE="$(classify_job "$CANDIDATE" "$CANDIDATE_STATUS")"
      case "$(verdict_word "$CANDIDATE_LINE")" in
        pending|running|unknown)
          ADOPT_JOB_ID="$CANDIDATE"
          ADOPT_STATUS_FILE="$CANDIDATE_STATUS"
          ADOPT_LINE="$CANDIDATE_LINE"
          ;;
      esac
    fi
  else
    QUEUE_BY_NAME="$("$SQUEUE_COMMAND" -h -n "$JOB_NAME" \
      -u "${USER:-$(id -un)}" -o '%i %T' 2>/dev/null)"
    if [ $? -eq 0 ]; then
      CANDIDATE="$(printf '%s\n' "$QUEUE_BY_NAME" | sed -n '1p' \
        | awk '{ print $1 }')"
      if printf '%s' "$CANDIDATE" | grep -Eq '^[0-9]+$'; then
        CANDIDATE_LINE="$(classify_job "$CANDIDATE" "")"
        case "$(verdict_word "$CANDIDATE_LINE")" in
          pending|running)
            ADOPT_JOB_ID="$CANDIDATE"
            ADOPT_LINE="$CANDIDATE_LINE"
            ;;
        esac
      fi
    fi
  fi
fi

if [ -n "$ADOPT_JOB_ID" ]; then
  echo "bootstrap-job: a bootstrap job is already in flight for this workspace"
  echo "bootstrap-job: adopting job $ADOPT_JOB_ID (pass --force-new to submit another)"
  echo "STEERLAB_BOOTSTRAP_ADOPT=$ADOPT_JOB_ID"
  echo "STEERLAB_BOOTSTRAP_JOB_ID=$ADOPT_JOB_ID"
  [ -n "$ADOPT_STATUS_FILE" ] \
    && echo "STEERLAB_BOOTSTRAP_STATUS_FILE=$ADOPT_STATUS_FILE"
  printf '%s\n' "$ADOPT_LINE"
  JOB_ID="$ADOPT_JOB_ID"
  STATUS_FILE="$ADOPT_STATUS_FILE"
  JOB_LOG="$(read_breadcrumb_field "$BREADCRUMB" log)"
  if [ "$WAIT_FOR_JOB" -eq 0 ]; then exit 0; fi
else
  command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available on this host"

  STAMP="$(date -u +%Y%m%dT%H%M%S)-$$"
  JOB_SCRIPT="$JOB_WORKSPACE/steerlab-bootstrap.$STAMP.sbatch"
  JOB_LOG="$JOB_WORKSPACE/steerlab-bootstrap.$STAMP.log"
  STATUS_FILE="$JOB_WORKSPACE/steerlab-bootstrap.$STAMP.status"

  {
    echo '#!/usr/bin/env bash'
    echo "#SBATCH --job-name=$JOB_NAME"
    echo "#SBATCH --partition=$JOB_PARTITION"
    echo '#SBATCH --ntasks=1'
    echo "#SBATCH --cpus-per-task=$JOB_CPUS"
    echo "#SBATCH --mem=$JOB_MEMORY"
    echo "#SBATCH --time=$JOB_WALLTIME"
    echo "#SBATCH --chdir=$JOB_WORKSPACE"
    echo "#SBATCH --output=$JOB_LOG"
    echo "#SBATCH --error=$JOB_LOG"
    echo '#SBATCH --export=NONE'
    [ -n "$JOB_ACCOUNT" ] && echo "#SBATCH --account=$JOB_ACCOUNT"
    # This class's own directives (WP5 Step 9). The SITE-WIDE placement
    # directives are deliberately absent: a CPU-only job pinned to GPU node
    # features is queued forever rather than rejected (audit §6.x item 1).
    if [ "${#JOB_EXTRA_SBATCH[@]}" -gt 0 ]; then
      for extra_directive in "${JOB_EXTRA_SBATCH[@]}"; do
        echo "#SBATCH $extra_directive"
      done
    fi
    echo
    # Reconstruct the minimum submitter identity under --export=NONE. Modules
    # are initialized explicitly because non-interactive batch shells do not
    # reliably read the login shell's startup files.
    printf 'export HOME=%q\n' "$HOME"
    printf 'export USER=%q\n' "${USER:-$(id -un)}"
    echo 'if ! command -v module >/dev/null 2>&1; then'
    echo '  [ -r /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh'
    echo '  command -v module >/dev/null 2>&1 || { [ -r /etc/profile ] && . /etc/profile; }'
    echo 'fi'
    echo 'set +e'
    printf 'bash %q' "$BOOTSTRAP_SCRIPT"
    if [ "${#BOOTSTRAP_ARGS[@]}" -gt 0 ]; then
      for argument in "${BOOTSTRAP_ARGS[@]}"; do printf ' %q' "$argument"; done
    fi
    echo
    echo 'bootstrap_code=$?'
    printf 'status_tmp=%q.tmp.$SLURM_JOB_ID\n' "$STATUS_FILE"
    printf 'printf "%%s\\n" "$bootstrap_code" > "$status_tmp" && mv "$status_tmp" %q\n' \
      "$STATUS_FILE"
    echo 'exit "$bootstrap_code"'
  } > "$JOB_SCRIPT" || fail "could not write $JOB_SCRIPT"
  chmod 700 "$JOB_SCRIPT"

  SUBMIT_OUTPUT="$(cd "$JOB_WORKSPACE" && sbatch --parsable "$JOB_SCRIPT" 2>&1)"
  SUBMIT_CODE=$?
  [ "$SUBMIT_CODE" -eq 0 ] || fail "sbatch failed: $SUBMIT_OUTPUT"
  JOB_ID="${SUBMIT_OUTPUT%%;*}"
  JOB_ID="${JOB_ID##* }"
  printf '%s' "$JOB_ID" | grep -Eq '^[0-9]+$' || fail "could not parse job id: $SUBMIT_OUTPUT"

  # The breadcrumb is written BEFORE anything else can go wrong, and lands
  # atomically: from here on, the submitted job is findable from the cluster
  # side alone even if this process (or the whole SSH session) dies now.
  if {
    printf 'jobID=%s\n' "$JOB_ID"
    printf 'statusFile=%s\n' "$STATUS_FILE"
    printf 'log=%s\n' "$JOB_LOG"
    printf 'script=%s\n' "$JOB_SCRIPT"
    printf 'submittedAt=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$BREADCRUMB.tmp.$$" 2>/dev/null \
    && mv "$BREADCRUMB.tmp.$$" "$BREADCRUMB"
  then
    :
  else
    rm -f "$BREADCRUMB.tmp.$$"
    echo "bootstrap-job: WARNING could not write $BREADCRUMB — job $JOB_ID is" \
      "submitted but this workspace cannot recognise it later" >&2
  fi

  echo "bootstrap-job: submitted Slurm job $JOB_ID"
  echo "bootstrap-job: log $JOB_LOG"
  echo "STEERLAB_BOOTSTRAP_JOB_ID=$JOB_ID"
  echo "STEERLAB_BOOTSTRAP_STATUS_FILE=$STATUS_FILE"
  echo "STEERLAB_BOOTSTRAP_LOG=$JOB_LOG"

  if [ "$WAIT_FOR_JOB" -eq 0 ]; then
    echo "bootstrap-job: submitted and detached — poll it with"
    echo "bootstrap-job:   bash $0 --status --job-workspace $JOB_WORKSPACE --job-id $JOB_ID"
    exit 0
  fi
fi

# --- --wait: the historical foreground follow ------------------------------

NEXT_LINE=1
emit_new_log() {
  [ -n "$JOB_LOG" ] || return 0
  [ -f "$JOB_LOG" ] || return 0
  line_count="$(wc -l < "$JOB_LOG" | tr -d '[:space:]')"
  [ -n "$line_count" ] || line_count=0
  if [ "$line_count" -ge "$NEXT_LINE" ]; then
    sed -n "${NEXT_LINE},${line_count}p" "$JOB_LOG"
    NEXT_LINE=$((line_count + 1))
  fi
}

POLL_SECONDS="${STEERLAB_BOOTSTRAP_POLL_SECONDS:-30}"
HEARTBEAT_POLLS="${STEERLAB_BOOTSTRAP_HEARTBEAT_POLLS:-10}"
MISSING_POLLS_LIMIT="${STEERLAB_BOOTSTRAP_MISSING_POLLS:-10}"
printf '%s' "$POLL_SECONDS" | grep -Eq '^[0-9]+([.][0-9]+)?$' \
  || fail "STEERLAB_BOOTSTRAP_POLL_SECONDS must be a non-negative number"
printf '%s' "$HEARTBEAT_POLLS" | grep -Eq '^[1-9][0-9]*$' \
  || fail "STEERLAB_BOOTSTRAP_HEARTBEAT_POLLS must be a positive integer"
printf '%s' "$MISSING_POLLS_LIMIT" | grep -Eq '^[1-9][0-9]*$' \
  || fail "STEERLAB_BOOTSTRAP_MISSING_POLLS must be a positive integer"
[ -n "$STATUS_FILE" ] \
  || fail "--wait needs the job's status file; poll with --status instead"
MISSING_POLLS=0
POLL_COUNT=0
QUERY_ERRORS=0
LAST_STATE=""
while [ ! -f "$STATUS_FILE" ]; do
  emit_new_log
  POLL_COUNT=$((POLL_COUNT + 1))
  QUEUE_OUTPUT="$("$SQUEUE_COMMAND" -h -j "$JOB_ID" -o '%T' 2>&1)"
  QUEUE_CODE=$?
  if [ "$QUEUE_CODE" -ne 0 ]; then
    QUERY_ERRORS=$((QUERY_ERRORS + 1))
    # A broken/restricted query command is NOT evidence that the job ended.
    # Keep following the authoritative status file written by the job.
    if [ "$QUERY_ERRORS" -eq 1 ] || [ $((QUERY_ERRORS % HEARTBEAT_POLLS)) -eq 0 ]; then
      echo "bootstrap-job: queue query failed; job $JOB_ID may still be active: $QUEUE_OUTPUT" >&2
    fi
  elif [ -n "$(printf '%s' "$QUEUE_OUTPUT" | tr -d '[:space:]')" ]; then
    MISSING_POLLS=0
    STATE="$(printf '%s\n' "$QUEUE_OUTPUT" | sed -n '1p')"
    if [ "$STATE" != "$LAST_STATE" ]; then
      echo "bootstrap-job: job $JOB_ID state $STATE"
      LAST_STATE="$STATE"
    elif [ $((POLL_COUNT % HEARTBEAT_POLLS)) -eq 0 ]; then
      echo "bootstrap-job: job $JOB_ID still $STATE"
    fi
  else
    MISSING_POLLS=$((MISSING_POLLS + 1))
    # Tolerate scheduler/accounting propagation at both job boundaries.
    if [ "$MISSING_POLLS" -ge "$MISSING_POLLS_LIMIT" ]; then break; fi
  fi
  sleep "$POLL_SECONDS"
done
emit_new_log

if [ ! -f "$STATUS_FILE" ]; then
  echo "bootstrap-job: job $JOB_ID ended without a status record; inspect $JOB_LOG" >&2
  exit 1
fi
BOOTSTRAP_CODE="$(tr -d '[:space:]' < "$STATUS_FILE")"
printf '%s' "$BOOTSTRAP_CODE" | grep -Eq '^[0-9]+$' \
  || fail "invalid status record in $STATUS_FILE"
if [ "$BOOTSTRAP_CODE" -eq 0 ]; then
  echo "bootstrap-job: job $JOB_ID completed successfully"
else
  echo "bootstrap-job: job $JOB_ID failed with exit $BOOTSTRAP_CODE" >&2
fi
exit "$BOOTSTRAP_CODE"
