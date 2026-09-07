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
refused=no
refuse() { refused=yes; printf '{"ok":false,"changed":false,"reason":'; json "$1"; printf ',"repairAction":'; json "$2"; printf '}\n'; exit 65; }
hash() { if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d ' ' -f1; else shasum -a 256 | cut -d ' ' -f1; fi; }
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64)
        platform=aarch64-apple-darwin
        uv_sha=5bb0e5fe008a773c3dbcb97ff79cd89e1241464fe9d2f986d52ad8f1b037bd62
        default_runtime="$HOME/Library/Application Support/SteerLab/client-runtime" ;;
    Linux/x86_64)
        platform=x86_64-unknown-linux-gnu
        uv_sha=68a509da24b06b4223a1c0175fb5eb5bc79342b76cbeff0cfe51ac3f5b17b6b2
        default_runtime="$HOME/.local/share/SteerLab/client-runtime" ;;
    *) refuse 'This client installer supports Apple Silicon macOS and x86_64 Linux with glibc.' 'Use a supported machine; other platforms are not qualified.' ;;
esac
runtime=${runtime:-$default_runtime}
case "$runtime" in /*) ;; *) refuse 'The runtime path must be absolute.' 'Choose an absolute directory outside the study workspace.' ;; esac
# Control characters are never meaningful filesystem input for this installer.
if printf '%s' "$runtime" | LC_ALL=C grep '[[:cntrl:]]' >/dev/null; then refuse 'Control characters are not supported in runtime paths.' 'Choose an ordinary absolute path.'; fi
case "$runtime" in */|*/../*|*/./*|*/..|*/.) refuse 'The runtime path must be normalized.' 'Remove trailing slashes and dot components.' ;; esac
for item in install-client.sh runtime-helper.py client-requirements.lock source.sha256; do
    [ -f "$release/$item" ] && [ ! -L "$release/$item" ] || refuse 'Incomplete client release.' 'Download or rebuild the complete client release.'
done
set -- "$release"/*.whl
[ "$#" -eq 1 ] && [ -f "$1" ] && [ ! -L "$1" ] || refuse 'The release needs exactly one client wheel.' 'Download or rebuild the complete client release.'
wheel=$1
state=missing
if [ -L "$runtime" ]; then
    [ -f "$runtime/.steerlab-client.json" ] || refuse 'The selected runtime link is not managed by this installer.' 'Select a new runtime path or use the existing interpreter explicitly.'
    state=managed
elif [ -e "$runtime" ]; then
    refuse 'An existing directory will not be replaced.' 'Keep that environment and choose another runtime path, or set STEERLAB_CLIENT_PYTHON explicitly.'
fi
plan_hash=$({ printf '%s\000' "$runtime" "$state" "$platform"; readlink "$runtime" || true; for item in install-client.sh runtime-helper.py client-requirements.lock source.sha256; do hash < "$release/$item"; done; hash < "$wheel"; if [ -f "$runtime/.steerlab-client.json" ]; then hash < "$runtime/.steerlab-client.json"; fi; } | hash)
if [ "$operation" = plan ]; then
    printf '{"ok":true,"changed":false,"planSHA256":"%s","runtime":' "$plan_hash"; json "$runtime"
    printf ',"state":"%s","python":"3.12.14","uv":"0.12.5","platform":"%s","requiresApproval":true,"actions":["Download verified uv and managed Python","Install the locked lightweight client into an isolated environment","Verify imports and source identity, then activate the environment"],"execution":"Models and runner setup remain separate; no models are downloaded.","repairAction":"Review this plan, then run install or repair with --expect <planSHA256> --yes."}\n' "$state" "$platform"
    exit 0
fi
case "$operation" in install|repair) ;; *) refuse 'Unknown setup operation.' 'Use plan, install, or repair.' ;; esac
[ "$approved" = yes ] && [ "$expected" = "$plan_hash" ] || refuse 'Installation requires approval of the current plan.' 'Run plan again and pass its planSHA256 with --expect and --yes.'
parent=$(dirname -- "$runtime")
mkdir -p "$parent"
# An atomic lock serializes our publishers. Never steal a stale lock automatically.
mkdir "$runtime.setup-lock" 2>/dev/null || refuse 'Another setup owns this runtime.' 'Wait for setup to finish; if interrupted, inspect and remove only its empty setup-lock directory.'
stage=''; activated=no
cleanup() { status=$?; if [ "$activated" = no ] && [ -n "$stage" ] && [ "$(readlink "$runtime" || true)" != "$stage/venv" ]; then rm -rf -- "$stage"; fi; rmdir "$runtime.setup-lock"; if [ "$status" -ne 0 ] && [ "$refused" = no ]; then printf '{"ok":false,"changed":false,"reason":"Client installation did not finish.","repairAction":"Check network access and the setup log, then review a fresh plan and retry. The previous runtime remains active."}\n'; fi; }
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
# Recheck after acquiring the lock; another setup may have completed meanwhile.
current=$(/bin/sh "$release/install-client.sh" plan --runtime "$runtime")
printf '%s' "$current" | grep -F "\"planSHA256\":\"$expected\"" >/dev/null || refuse 'The setup plan changed.' 'Review a fresh plan before installing.'
stage=$(mktemp -d "$parent/.steerlab-client.XXXXXXXX")
mkdir "$stage/release"
cp "$release/install-client.sh" "$release/runtime-helper.py" "$release/client-requirements.lock" "$release/source.sha256" "$wheel" "$stage/release/"
copied=$(/bin/sh "$stage/release/install-client.sh" plan --runtime "$runtime")
printf '%s' "$copied" | grep -F "\"planSHA256\":\"$expected\"" >/dev/null || refuse 'Release files changed while staging.' 'Review a fresh release plan.'
printf 'Downloading verified client tools…\n' >&2
curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 "https://github.com/astral-sh/uv/releases/download/0.12.5/uv-$platform.tar.gz" -o "$stage/uv.tar.gz"
[ "$(hash < "$stage/uv.tar.gz")" = "$uv_sha" ] || refuse 'Downloaded uv checksum mismatch.' 'Retry from the original release; do not bypass verification.'
tar -xzf "$stage/uv.tar.gz" -C "$stage"
uv="$stage/uv-$platform/uv"
export UV_CACHE_DIR="$stage/cache" UV_PYTHON_INSTALL_DIR="$stage/python" UV_NO_CONFIG=1 UV_PYTHON_PREFERENCE=only-managed
unset PYTHONPATH PYTHONHOME UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_INDEX UV_FIND_LINKS UV_PYTHON UV_PROJECT_ENVIRONMENT VIRTUAL_ENV || true
printf 'Preparing Python and installing the lightweight client…\n' >&2
"$uv" venv --no-project --python 3.12.14 "$stage/venv" >&2
"$uv" pip sync --python "$stage/venv/bin/python" --require-hashes --only-binary :all: --default-index https://pypi.org/simple "$stage/release/client-requirements.lock" >&2
set -- "$stage/release"/*.whl
"$uv" pip install --python "$stage/venv/bin/python" --no-deps "$1" >&2
"$stage/venv/bin/python" -I "$stage/release/runtime-helper.py" "$stage" "$runtime" "$expected" "$release"
activated=yes
