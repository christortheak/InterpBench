#!/bin/sh
# A release-directory installer: works before Python or a checkout exists.
# plan is read-only. install/repair require the exact plan hash and --yes.
set -eu
release=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
operation=${1:-plan}
[ "$#" -eq 0 ] || shift
runtime=''; expected=''; approved=no
while [ "$#" -gt 0 ]; do
    case "$1" in
        --runtime) runtime=${2:?missing runtime}; shift 2 ;;
        --expect) expected=${2:?missing plan hash}; shift 2 ;;
        --yes) approved=yes; shift ;;
        *) echo 'Unknown installer argument.' >&2; exit 64 ;;
    esac
done
json() { printf '%s' "$1" | awk 'BEGIN {printf "\""} {if (NR>1) printf "\\n"; gsub(/\\/,"\\\\"); gsub(/\"/,"\\\""); gsub(/\t/,"\\t"); gsub(/\r/,"\\r"); printf "%s",$0} END {printf "\""}'; }
# Managed tool versions, defined once for the displayed plan and the installation.
python_version=3.12.14
uv_version=0.12.5
# Every download is bounded. A connection that does not open within
# connect_timeout seconds, or moves under 1 KB/s for stall_seconds, or is
# still running after transfer_timeout seconds is abandoned, and a transient
# failure is retried download_retries times. uv's own downloads (the managed
# Python and the locked packages) get the same bounds unless the person has
# set uv's timeouts themselves.
connect_timeout=20
stall_seconds=60
transfer_timeout=900
download_retries=3
# The locked numpy and pyarrow wheels for x86_64 Linux are manylinux_2_28.
glibc_minimum_minor=28
# Free space the staging folder needs: uv, the managed Python, the package
# cache, and the environment, with room to spare.
disk_mb=1000
# Every refusal and failure is typed: a stable code, a plain reason, and a
# repair. A refusal (65) changes nothing; a failure (70) leaves the previous
# runtime active.
reported=no
report() { reported=yes; printf '{"ok":false,"changed":false,"code":'; json "$1"; printf ',"reason":'; json "$2"; printf ',"repairAction":'; json "$3"; printf '}\n'; }
refuse() { report "$1" "$2" "$3"; exit 65; }
fail() { report "$1" "$2" "$3"; exit 70; }
hash() { if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d ' ' -f1; else shasum -a 256 | cut -d ' ' -f1; fi; }
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64)
        platform=aarch64-apple-darwin
        uv_sha=5bb0e5fe008a773c3dbcb97ff79cd89e1241464fe9d2f986d52ad8f1b037bd62
        download_mb=100
        default_runtime="$HOME/Library/Application Support/SteerLab/client-runtime" ;;
    Linux/x86_64)
        platform=x86_64-unknown-linux-gnu
        uv_sha=68a509da24b06b4223a1c0175fb5eb5bc79342b76cbeff0cfe51ac3f5b17b6b2
        download_mb=150
        default_runtime="$HOME/.local/share/SteerLab/client-runtime" ;;
    *) refuse unsupportedPlatform 'This client installer supports Apple Silicon macOS and x86_64 Linux with glibc.' 'Use a supported machine; other platforms are not qualified.' ;;
esac
# Preflight, before anything is computed or written: the tools this script
# calls, and on Linux the C library the managed Python and wheels are built for.
missing=''
command -v curl >/dev/null 2>&1 || missing="${missing:+$missing, }curl"
command -v tar >/dev/null 2>&1 || missing="${missing:+$missing, }tar"
command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 || missing="${missing:+$missing, }sha256sum"
[ -z "$missing" ] || refuse missingTools "Setup needs tools this machine does not have: $missing." 'Install them with the system package manager (sha256sum comes with coreutils), then review the plan again.'
if [ "$platform" = x86_64-unknown-linux-gnu ]; then
    c_library=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
    case "$c_library" in
        'glibc '*) c_library=${c_library#glibc } ;;
        *) c_library=$( (ldd --version 2>&1 || true) | awk 'NR == 1 && (/GLIBC/ || /GNU libc/) {print $NF}') ;;
    esac
    glibc_major=${c_library%%.*}; glibc_minor=${c_library#*.}; glibc_minor=${glibc_minor%%[!0-9]*}
    case "$glibc_major/$glibc_minor" in
        */|/*|*[!0-9/]*) refuse unsupportedCLibrary 'This Linux does not use the GNU C library (glibc); musl-based systems such as Alpine are not supported. The managed Python and the client packages are built for glibc.' 'Use a glibc-based Linux, such as Debian, Ubuntu, Fedora, or Rocky Linux.' ;;
    esac
    if [ "$glibc_major" -lt 2 ] || { [ "$glibc_major" -eq 2 ] && [ "$glibc_minor" -lt "$glibc_minimum_minor" ]; }; then
        refuse oldCLibrary "This Linux has glibc $c_library; the client packages need glibc 2.$glibc_minimum_minor or newer." 'Use a newer Linux release, such as Debian 10, Ubuntu 20.04, or Rocky Linux 8, or later.'
    fi
fi
runtime=${runtime:-$default_runtime}
case "$runtime" in /*) ;; *) refuse invalidRuntimePath 'The runtime path must be absolute.' 'Choose an absolute directory outside the study workspace.' ;; esac
# Control characters are never meaningful filesystem input for this installer.
if printf '%s' "$runtime" | LC_ALL=C grep '[[:cntrl:]]' >/dev/null; then refuse invalidRuntimePath 'Control characters are not supported in runtime paths.' 'Choose an ordinary absolute path.'; fi
case "$runtime" in */|*/../*|*/./*|*/..|*/.) refuse invalidRuntimePath 'The runtime path must be normalized.' 'Remove trailing slashes and dot components.' ;; esac
for item in install-client.sh runtime-helper.py client-requirements.lock source.sha256; do
    [ -f "$release/$item" ] && [ ! -L "$release/$item" ] || refuse incompleteRelease 'Incomplete client release.' 'Download or rebuild the complete client release.'
done
set -- "$release"/*.whl
[ "$#" -eq 1 ] && [ -f "$1" ] && [ ! -L "$1" ] || refuse incompleteRelease 'The release needs exactly one client wheel.' 'Download or rebuild the complete client release.'
wheel=$1
state=missing
if [ -L "$runtime" ]; then
    [ -f "$runtime/.steerlab-client.json" ] || refuse unmanagedRuntime 'The selected runtime link is not managed by this installer.' 'Select a new runtime path or use the existing interpreter explicitly.'
    state=managed
elif [ -e "$runtime" ]; then
    refuse unmanagedRuntime 'An existing directory will not be replaced.' 'Keep that environment and choose another runtime path, or set STEERLAB_CLIENT_PYTHON explicitly.'
fi
parent=$(dirname -- "$runtime")
location_repair='Choose an install location inside a folder you own (on the command line, --runtime <absolute-path>), or change the permissions of that folder.'
# Free megabytes on the disk holding $1, or nothing when df cannot say. The
# capacity column ends in %; available space is the column before it, which
# survives spaces in device names and mount points.
free_mb() { df -Pk "$1" 2>/dev/null | awk 'NR == 2 {for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]+%$/) {print int($(i - 1) / 1024); exit}}' || true; }
# The nearest folder that already exists on the way to the runtime: where
# the plan measures writability and free space, before anything is created.
existing=$parent
while [ ! -d "$existing" ]; do
    [ ! -e "$existing" ] && [ ! -L "$existing" ] || refuse unwritableDestination "The install location cannot be created, because $existing is a file, not a folder." "$location_repair"
    existing=$(dirname -- "$existing")
done
[ -w "$existing" ] || refuse unwritableDestination "The install location cannot be written: this account cannot create files in $existing." "$location_repair"
free=$(free_mb "$existing")
case "$free" in ''|*[!0-9]*) free='' ;; esac
if [ -n "$free" ] && [ "$free" -lt "$disk_mb" ]; then
    refuse diskFull "The disk holding $existing has $free MB free; setup needs about $disk_mb MB." 'Free some disk space, or choose an install location on another disk, then review the plan again.'
fi
plan_hash=$({ printf '%s\000' "$runtime" "$state" "$platform"; readlink "$runtime" || true; for item in install-client.sh runtime-helper.py client-requirements.lock source.sha256; do hash < "$release/$item"; done; hash < "$wheel"; if [ -f "$runtime/.steerlab-client.json" ]; then hash < "$runtime/.steerlab-client.json"; fi; } | hash)
if [ "$operation" = plan ]; then
    printf '{"ok":true,"changed":false,"planSHA256":"%s","runtime":' "$plan_hash"; json "$runtime"
    printf ',"state":"%s","python":"%s","uv":"%s","platform":"%s","approximateDownloadMB":%s,"requiredDiskMB":%s,"requiresApproval":true,"actions":["Download verified uv, managed Python, and the locked client packages (about %s MB in all)","Install the locked lightweight client into an isolated environment (needs about %s MB of free disk space)","Verify imports and source identity, then activate the environment"],"execution":"Models and runner setup remain separate; no models are downloaded.","repairAction":"Review this plan, then run install or repair with --expect <planSHA256> --yes."}\n' "$state" "$python_version" "$uv_version" "$platform" "$download_mb" "$disk_mb" "$download_mb" "$disk_mb"
    exit 0
fi
case "$operation" in install|repair) ;; *) refuse unknownOperation 'Unknown setup operation.' 'Use plan, install, or repair.' ;; esac
[ "$approved" = yes ] && [ "$expected" = "$plan_hash" ] || refuse approvalRequired 'Installation requires approval of the current plan.' 'Review a fresh plan, then approve it (on the command line, pass its planSHA256 with --expect and --yes).'
network_repair='Check the internet connection (setup downloads from github.com and pypi.org), then review a fresh plan and retry. The previous runtime remains active.'
disk_repair='Free some disk space, then review a fresh plan and retry. The previous runtime remains active.'
tls_repair='If your network inspects secure traffic (some institutional networks and proxies do), ask its administrators to allow github.com and pypi.org, or use another network, and check that the date and time on this computer are correct. Then review a fresh plan and retry. The previous runtime remains active.'
lock="$runtime.setup-lock"
stage=''; child=''; locked=no; activated=no; cancelled=no
# Long steps run as a background child, so a Cancel (INT, TERM, or HUP) is
# handled at once: a trapped signal interrupts `wait`, and the cleanup stops
# the child. Children write to stderr (the setup log), never to stdout.
run_child() { "$@" >&2 </dev/null & child=$!; wait "$child" && status=0 || status=$?; child=''; return "$status"; }
stop_child() { if [ -n "$child" ]; then kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; child=''; fi; }
cleanup() {
    status=$?
    trap '' INT TERM HUP   # a second Cancel must not interrupt the cleanup itself
    stop_child
    # The helper switches the public link last; once it points here, the install finished.
    if [ -n "$stage" ] && [ "$(readlink "$runtime" || true)" = "$stage/venv" ]; then activated=yes; fi
    if [ "$activated" = no ] && [ -n "$stage" ]; then rm -rf -- "$stage"; fi
    if [ "$locked" = yes ]; then rm -f -- "$lock/owner"; rmdir -- "$lock" 2>/dev/null || true; fi
    if [ "$activated" = yes ]; then exit 0; fi
    if [ "$cancelled" = yes ] && [ "$reported" = no ]; then
        report cancelled 'Setup was cancelled.' 'Nothing was changed: the previous runtime, if any, is still active. Review a fresh plan when you want to install.'
        exit 130
    fi
    if [ "$status" -ne 0 ] && [ "$reported" = no ]; then
        report installFailed 'Client installation did not finish.' 'Read the setup log for the step that failed, then review a fresh plan and retry. The previous runtime remains active.'
    fi
}
trap cleanup EXIT
trap 'cancelled=yes; exit 130' INT TERM HUP
# After a step fails: a full disk is named as such; anything else names the step.
step_failed() {
    left=$(free_mb "$parent")
    case "$left" in ''|*[!0-9]*) left='' ;; esac
    if [ -n "$left" ] && [ "$left" -lt 100 ]; then fail diskFull "The disk ran out of space while $1 ($left MB left)." "$disk_repair"; fi
    fail installFailed "Setup stopped while $1." 'Read the setup log for the details, then review a fresh plan and retry. The previous runtime remains active.'
}
# A failed uv step, named by what its own output says went wrong.
uv_failed() {
    if grep -i -q -E 'no space left on device|os error 28|disk quota exceeded' "$2"; then
        fail diskFull "The disk ran out of space while $1." "$disk_repair"
    elif grep -i -q -E 'hash mismatch' "$2"; then
        refuse checksumMismatch "A file downloaded while $1 did not match its recorded checksum, so it was not used." 'Retry later; never bypass verification. If it happens again, something on the network (such as a proxy) is changing downloads, or the release is damaged.'
    elif grep -i -q -E 'invalid peer certificate|certificate verify|unknownissuer|self[- ]signed certificate|certificate has expired|tls handshake' "$2"; then
        fail tlsFailure "A secure connection could not be verified while $1." "$tls_repair"
    elif grep -i -q -E 'timed out|timeout' "$2"; then
        fail downloadStalled "A download stopped making progress while $1, and was abandoned after $download_retries retries." "$network_repair"
    elif grep -i -q -E 'dns error|failed to lookup address|name or service not known|nodename nor servname|temporary failure in name resolution|network is unreachable|connection refused|tcp connect error|error sending request|connection reset' "$2"; then
        fail noNetwork "The download server could not be reached while $1." "$network_repair"
    fi
    step_failed "$1"
}
uv_step() {
    what=$1; shift
    run_child "$@" 2>"$stage/step.log" && status=0 || status=$?
    cat "$stage/step.log" >&2
    [ "$status" -eq 0 ] || uv_failed "$what" "$stage/step.log"
}
download() {
    run_child curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 \
        --connect-timeout "$connect_timeout" --speed-limit 1024 --speed-time "$stall_seconds" \
        --max-time "$transfer_timeout" --retry "$download_retries" --retry-delay 2 \
        "$1" -o "$2" && return 0 || status=$?
    case "$status" in
        5|6) fail noNetwork 'The download server could not be found. This computer may be offline, or its network may not resolve the address.' "$network_repair" ;;
        7) fail noNetwork 'The download server could not be reached. This computer may be offline, or a firewall may block the connection.' "$network_repair" ;;
        28) fail downloadStalled "A download stopped making progress, and was abandoned after $download_retries retries." "$network_repair" ;;
        16|18|52|55|56|92) fail downloadInterrupted 'The connection was interrupted during a download.' "$network_repair" ;;
        22) fail downloadRefused 'The download server answered with an error; the file may be briefly unavailable, or a proxy may be blocking it.' "$network_repair" ;;
        35|51|53|54|58|59|60|64|66|77|80|82|83|90|91) fail tlsFailure 'A secure connection to the download server could not be verified.' "$tls_repair" ;;
        23) step_failed 'saving a download' ;;
        *) fail downloadFailed "A download failed (curl exit status $status)." "$network_repair" ;;
    esac
}
# The setup lock: an atomic mkdir serializes publishers. Its owner record
# names the process, machine, and start time, so a lock left by a setup that
# is no longer running is recognized and reclaimed instead of blocking every
# later attempt. A live setup's lock is never taken.
host=$(uname -n 2>/dev/null || echo unknown)
take_lock() {
    mkdir -- "$lock" 2>/dev/null || return 1
    locked=yes
    printf 'pid=%s\nhost=%s\nstarted=%s\n' "$$" "$host" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$lock/owner"
}
owner_value() { sed -n "s/^$1=//p" "$lock/owner" 2>/dev/null | head -n 1; }
owner_pid=''; owner_host=''; owner_unchecked=''
lock_is_stale() {
    owner_pid=$(owner_value pid); owner_host=$(owner_value host); owner_unchecked=''
    case "$owner_pid" in
        ''|*[!0-9]*)
            # No owner record: an earlier installer that wrote none, or a setup
            # stopped between creating the lock and recording itself. Old
            # enough, it cannot be a setup that is still starting.
            owner_pid=''
            [ -n "$(find "$lock" -prune -mmin +2 2>/dev/null || true)" ]; return ;;
    esac
    [ "$owner_host" = "$host" ] || return 1   # another machine: its processes cannot be checked from here
    [ "$owner_pid" != "$$" ] || return 0      # this very process number, recorded before a restart
    # Is a process with that number running? kill -0 asks without signalling;
    # "not permitted" means it runs, under an account this one cannot signal.
    if ! kill -0 "$owner_pid" 2>/dev/null; then
        case "$(LC_ALL=C kill -0 "$owner_pid" 2>&1 || true)" in
            *ermitted*) ;;
            *) return 0 ;;
        esac
    fi
    # It runs. The lock is stale only when inspection shows that the number now
    # belongs to something other than this installer. A process that cannot be
    # inspected (no ps, ps refused, or nothing reported) is the live setup: a
    # failed check must never delete a running setup's work.
    command -v ps >/dev/null 2>&1 || { owner_unchecked=yes; return 1; }
    owner_args=$(ps -p "$owner_pid" -o args= 2>/dev/null) || { owner_unchecked=yes; return 1; }
    case "$owner_args" in
        '') owner_unchecked=yes; return 1 ;;
        *install-client.sh*) return 1 ;;
        *) return 0 ;;
    esac
}
reclaim_lock() {
    old_stage=$(owner_value stage)
    # The interrupted setup's staging folder goes too, unless the runtime points into it.
    case "$old_stage" in
        "$parent"/.steerlab-client.*)
            case "${old_stage#"$parent"/}" in
                */*) ;;
                *) if [ -d "$old_stage" ] && [ ! -L "$old_stage" ] && [ "$(readlink "$runtime" || true)" != "$old_stage/venv" ]; then rm -rf -- "$old_stage"; fi ;;
            esac ;;
    esac
    rm -f -- "$lock/owner"
    rmdir -- "$lock" 2>/dev/null
}
mkdir -p -- "$parent" 2>/dev/null && [ -w "$parent" ] || refuse unwritableDestination "The install location cannot be written: this account cannot create files in $parent." "$location_repair"
if ! take_lock; then
    if [ -d "$lock" ] && [ ! -L "$lock" ] && lock_is_stale; then
        if reclaim_lock && take_lock; then
            printf 'Reclaimed the setup lock of an earlier setup that is no longer running.\n' >&2
        else
            refuse setupInProgress 'Another setup started for this runtime at the same moment.' 'Wait for it to finish, then review a fresh plan.'
        fi
    elif [ ! -e "$lock" ] && [ ! -L "$lock" ]; then
        refuse unwritableDestination "The install location cannot be written: this account cannot create files in $parent." "$location_repair"
    elif [ ! -d "$lock" ] || [ -L "$lock" ]; then
        refuse setupInProgress "Something that is not a setup lock is in the way at $lock." 'Move it aside, then review a fresh plan.'
    elif [ -n "$owner_pid" ] && [ "$owner_host" != "$host" ]; then
        refuse setupInProgress "A setup on another machine ($owner_host) holds this runtime's lock." "If no setup is running there, remove the folder $lock, then review a fresh plan."
    elif [ -n "$owner_pid" ] && [ -n "$owner_unchecked" ]; then
        refuse setupInProgress "Process $owner_pid holds this runtime's setup lock (started $(owner_value started)), and this environment does not allow checking whether it is a setup." "If a setup is running, wait for it to finish. If none is, remove the folder $lock, then review a fresh plan."
    elif [ -n "$owner_pid" ]; then
        refuse setupInProgress "Another setup is installing into this runtime (process $owner_pid, started $(owner_value started))." 'Wait for it to finish, or stop it, then review a fresh plan. A lock left by a setup that is no longer running is reclaimed automatically.'
    else
        refuse setupInProgress 'Another setup is starting for this runtime.' 'Wait a minute, then review a fresh plan. A lock left by a setup that is no longer running is reclaimed automatically.'
    fi
fi
# Ask a copy of the installer for the plan again, as it stands now.
replan() {
    current=$(/bin/sh "$1" plan --runtime "$runtime") && return 0
    case "$current" in '{'*) printf '%s\n' "$current"; reported=yes; exit 65 ;; esac
    fail installFailed 'The setup plan could not be computed again.' 'Review a fresh plan and retry. The previous runtime remains active.'
}
# Recheck after acquiring the lock; another setup may have completed meanwhile.
replan "$release/install-client.sh"
printf '%s' "$current" | grep -F "\"planSHA256\":\"$expected\"" >/dev/null || refuse planChanged 'The setup plan changed.' 'Review a fresh plan before installing.'
stage=$(mktemp -d "$parent/.steerlab-client.XXXXXXXX") || step_failed 'creating a staging folder'
printf 'stage=%s\n' "$stage" >> "$lock/owner"
mkdir "$stage/release" || step_failed 'creating a staging folder'
cp "$release/install-client.sh" "$release/runtime-helper.py" "$release/client-requirements.lock" "$release/source.sha256" "$wheel" "$stage/release/" || step_failed 'copying the release files'
replan "$stage/release/install-client.sh"
printf '%s' "$current" | grep -F "\"planSHA256\":\"$expected\"" >/dev/null || refuse planChanged 'Release files changed while staging.' 'Review a fresh release plan.'
printf 'Downloading verified client tools (about %s MB in all)…\n' "$download_mb" >&2
download "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-$platform.tar.gz" "$stage/uv.tar.gz"
[ "$(hash < "$stage/uv.tar.gz")" = "$uv_sha" ] || refuse checksumMismatch 'The downloaded uv did not match its recorded checksum, so it was not used.' 'Retry from the original release; never bypass verification. If it happens again, something on the network (such as a proxy) is changing downloads.'
run_child tar -xzf "$stage/uv.tar.gz" -C "$stage" || step_failed 'unpacking uv'
uv="$stage/uv-$platform/uv"
export UV_CACHE_DIR="$stage/cache" UV_PYTHON_INSTALL_DIR="$stage/python" UV_NO_CONFIG=1 UV_PYTHON_PREFERENCE=only-managed
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-$stall_seconds}" UV_HTTP_CONNECT_TIMEOUT="${UV_HTTP_CONNECT_TIMEOUT:-$connect_timeout}" UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-$download_retries}"
# A person's environment must not redirect where packages or the managed Python come from,
# replace the Python download metadata (which carries its checksums), or weaken TLS.
unset PYTHONPATH PYTHONHOME UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_INDEX UV_FIND_LINKS UV_PYTHON UV_PROJECT_ENVIRONMENT VIRTUAL_ENV \
    UV_PYTHON_INSTALL_MIRROR UV_PYPY_INSTALL_MIRROR UV_PYTHON_DOWNLOADS_JSON_URL UV_INSECURE_HOST || true
printf 'Preparing Python and installing the lightweight client…\n' >&2
uv_step 'downloading the managed Python' "$uv" venv --no-project --python "$python_version" "$stage/venv"
uv_step 'downloading and installing the client packages' "$uv" pip sync --python "$stage/venv/bin/python" --require-hashes --only-binary :all: --default-index https://pypi.org/simple "$stage/release/client-requirements.lock"
set -- "$stage/release"/*.whl
uv_step 'installing the client' "$uv" pip install --python "$stage/venv/bin/python" --no-deps "$1"
printf 'Verifying the new environment, then activating it…\n' >&2
# The helper prints the success result on stdout; it is the only child that may.
"$stage/venv/bin/python" -I "$stage/release/runtime-helper.py" "$stage" "$runtime" "$expected" "$release" </dev/null & child=$!
wait "$child" || true
child=''
# Only the switched link counts: the helper moves it last, after every check.
[ "$(readlink "$runtime" || true)" = "$stage/venv" ] || fail verificationFailed 'The new environment did not pass its final checks, so it was not activated.' 'Read the setup log for the check that failed, then review a fresh plan and retry. The previous runtime remains active.'
activated=yes
