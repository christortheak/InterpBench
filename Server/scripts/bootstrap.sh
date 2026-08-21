#!/usr/bin/env bash
# SteerLab cluster bootstrap (TURNKEY-CLUSTER-PLAN WS5.1).
#
# Runs ON the cluster — normally from an interactive allocation or a batch
# job, since this script loads modules and runs conda solves and pip installs,
# and many sites forbid that on their login/submit nodes. WHICH hosts those
# are, and whether compute is allowed there at all, is SITE POLICY, not this
# script's: the guard below reads `policy.loginNodes` out of the rendered site
# profile (WP5 step 10), and falls back to its own built-in rule only when no
# profile reached it. The supported paths:
#   * normal run:   get an allocation (your site's `salloc`/`srun --pty`, or
#                   whatever wrapper it publishes — a site that declares
#                   `environment.interactiveAllocationCommand` has its own
#                   command printed back to it in the refusal below), then
#                   run this
#   * pip-only rerun when compute nodes lack internet egress: run it on the
#     site's transfer/data-mover node with --force-login (such nodes have
#     egress but no SLURM_JOB_ID; the site names one in
#     `environment.transferHost`). This only works where the env prefix is on
#     a filesystem both node classes mount.
#   * --dry-run works anywhere, including login nodes (prints the plan only)
# It automates the cluster-runbook install phases:
#
#   1. locate conda        (load --modules, else conda/mamba on PATH)
#   2. create-or-update    the python env at --prefix
#   3. pip install torch   (from --torch-index; skip with --no-torch)
#   4. pip install -r Server/requirements-<platform>.lock  (WP6 R1 pinned set,
#      minus the site-owned torch/nvidia-*/triton; skip with --no-lock),
#      then pip install -e "$repo/Server[all]"
#   5. install the canonical env file (~/steerlab-cluster.env by default;
#      atomic; refuses to clobber an existing file without --force). With
#      --env-file-from — what the app and `steerlab-cli cluster bootstrap`
#      now pass by default — it VALIDATES AND INSTALLS the environment
#      rendered from the site profile. Without it, the script falls back to
#      its own built-in heredoc: the manual, no-profile path (WP5 step 7),
#      whose values are defaults rather than declared site facts.
#   6. steerlab-server profile validate, sourcing that env file
#   7. optional --hello: submit a tiny GPU sbatch test job
#
# Idempotent: re-running converges (existing env is reused, installs are
# pip-idempotent, the env file is only rewritten under --force). There is
# deliberately NO `set -e`: every step reports its own failure and the final
# stdout line is a machine-readable JSON report:
#
#   {"ok":true,"steps":{"condaDetect":"ok",...},"envFile":"...","prefix":"..."}
#
# --dry-run prints the planned steps (marked "planned") without executing.

usage() {
  cat <<'EOF'
usage: bootstrap.sh [flags]
  --prefix PATH        conda env prefix          (default: $HOME/envs/steerlab)
  --python VERSION     python version            (default: 3.12)
  --repo PATH          SteerLab checkout         (default: $HOME/steerlab)
  --workspace PATH     workspace root            (default: /scratch/$USER/steerlab-workspace)
  --hf-cache PATH      HF model cache (HF_HOME)  (default: $HOME/.cache/huggingface;
                       point it at a shared/group filesystem where one exists)
  --modules "A B"      modules to load before looking for conda
                       (default: Miniforge3 — this script's BUILT-IN FALLBACK.
                       A site states its own in environment.modules, which
                       reaches this run as STEERLAB_MODULES in the pushed
                       render; pass "" to load nothing)
  --partition NAME     default Slurm partition   (default: gpu_p)
  --gres SPEC          default gres              (default: gpu:A100:1)
  --scratch-gres TOKEN node-local scratch requested as its own gres, e.g.
                       "lscratch:100" (in GB at typical sites). Absent by default: the
                       synthesized env file gains no line and nothing changes.
                       Scheduling accounting only — Slurm does not enforce it;
                       the rendered job script removes its own stage directory
                       on exit either way.
  --account NAME       Slurm --account           (default: none)
  --walltime H:M:S     default walltime          (default: 24:00:00)
  --torch-index URL    torch pip index. The default is a CUDA 12.8 build
                       (https://download.pytorch.org/whl/cu128) — this
                       script's BUILT-IN FALLBACK, not a claim about your
                       hardware. A site states environment.torchIndexURL /
                       torchVariant and the provisioner passes it here; on
                       ROCm, older CUDA, or CPU-only nodes the default is
                       wrong and torch will import but find no device.
  --no-torch           skip the torch install step
  --no-lock            resolve dependencies from pyproject's FLOORS instead of
                       the committed per-platform lock. The lock is the
                       default because floors let two sites run different
                       torch/transformers on the substrate the reproducibility
                       claims live on (WP6 R1). Pass this only when a site
                       genuinely cannot use the pinned set; the run stamp
                       records what was installed either way.
  --env-file PATH      env file to write         (default: $HOME/steerlab-cluster.env)
  --env-file-from PATH install this PRE-RENDERED env file instead of writing
                       one from the flags above (WP5 materialization). The
                       file is validated (shape + shell syntax) before it is
                       installed; the flags it supersedes are then unused.
                       The app and steerlab-cli pass this by default. Absent,
                       the script writes its own BUILT-IN FALLBACK env file:
                       the manual, no-profile path, whose GPU vocabulary, job
                       memory, purge window, offline mode, node staging, and
                       metadata root are this script's defaults rather than
                       the site's declared facts.
  --env-file-sha256 HEX  refuse --env-file-from unless the file hashes to HEX.
                       This is the integrity half of the reviewed bootstrap
                       plan: the same digest rides in the plan hash, so
                       approving a plan approves these exact env bytes.
  --force              overwrite an existing env file
  --force-login        skip the login-node guard (for pip reruns on a
                       transfer/data-mover node — those have egress but no
                       SLURM_JOB_ID, so the guard would refuse)
  --with-jlens         install the pinned J-lens reference package and
                       pre-stage the published Gemma lens artifacts (~4 GB)
                       into --hf-cache. Off by default; needs github.com +
                       huggingface.co egress. Warms the cache only — the
                       import/convert step is `steerlab-server jlens import`
  --hello              submit a tiny GPU test job via sbatch at the end
  --dry-run            print the plan + JSON report without executing
  -h | --help          this text
EOF
}

# ---------------------------------------------------------------- defaults --
PREFIX="$HOME/envs/steerlab"
PYVER="3.12"
REPO="$HOME/steerlab"
WORKSPACE="/scratch/${USER:-$(id -un)}/steerlab-workspace"
HF_CACHE="$HOME/.cache/huggingface"
PARTITION="gpu_p"
GRES="gpu:A100:1"
# Declare-or-omit (cluster-operator requirement, 2026-08-19): empty means the
# synthesized env file says nothing about node-local scratch, exactly as before.
SCRATCH_GRES=""
ACCOUNT=""
WALLTIME="24:00:00"
# c26/c30 (WP5 step 12): the module list and the torch index are SITE FACTS —
# environment.modules and environment.torchIndexURL. The values below are this
# script's built-in fallbacks for a hand run that no profile reached, exactly
# like the env heredoc and the login-node guard: MODULES_DEFAULT is overridden
# by the pushed render's STEERLAB_MODULES (resolved after argument parsing),
# and the provisioner passes --torch-index whenever the site declares one.
MODULES_DEFAULT="Miniforge3"
MODULES=""
MODULES_SOURCE=""
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
INSTALL_TORCH=1
# WP6 R1: install the exact pinned set from Server/requirements-<platform>.lock
# before the editable server install. Default on — floors are not a
# reproducibility contract. --no-lock opts out.
USE_LOCK=1
ENV_FILE="$HOME/steerlab-cluster.env"
# WP5 step 6 — opt-in materialization. Empty means "write the heredoc below",
# i.e. exactly what this script has always done.
ENV_FILE_FROM=""
ENV_FILE_SHA256=""
FORCE=0
FORCE_LOGIN=0
HELLO=0
DRY_RUN=0
WITH_JLENS=0

# Lens folders in neuronpedia/jacobian-lens, per the J-lens plan §3.1 supported
# table. The repository holds 36 models totalling ~57 GB; these two are ~4 GB,
# so the allow_patterns scoping below is load-bearing, not an optimization.
JLENS_MODELS="google/gemma-3-27b-it google/gemma-3-4b-it"
# The pre-stage free-space floor is this script's FALLBACK (WP5 step 11, audit
# c47): ~8 GiB, sized from SteerLab's own lens bytes plus working headroom, so
# it is a fact about the download and not about any site. A site that knows
# better — a shared group quota, a thin node-local cache — declares
# constraints.storage.prestageMinFreeGB and the render's
# STEERLAB_PRESTAGE_MIN_FREE_GB wins, resolved below exactly the way the
# login-node guard resolves its policy.
JLENS_MIN_FREE_KB=8388608

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)      PREFIX="$2"; shift 2 ;;
    --python)      PYVER="$2"; shift 2 ;;
    --repo)        REPO="$2"; shift 2 ;;
    --workspace)   WORKSPACE="$2"; shift 2 ;;
    --hf-cache)    HF_CACHE="$2"; shift 2 ;;
    --partition)   PARTITION="$2"; shift 2 ;;
    --gres)        GRES="$2"; shift 2 ;;
    --scratch-gres) SCRATCH_GRES="$2"; shift 2 ;;
    --account)     ACCOUNT="$2"; shift 2 ;;
    --walltime)    WALLTIME="$2"; shift 2 ;;
    --modules)     MODULES="$2"; MODULES_SOURCE="--modules"; shift 2 ;;
    --torch-index) TORCH_INDEX="$2"; shift 2 ;;
    --no-torch)    INSTALL_TORCH=0; shift ;;
    --no-lock)     USE_LOCK=0; shift ;;
    --env-file)    ENV_FILE="$2"; shift 2 ;;
    --env-file-from)   ENV_FILE_FROM="$2"; shift 2 ;;
    --env-file-sha256) ENV_FILE_SHA256="$2"; shift 2 ;;
    --force)       FORCE=1; shift ;;
    --force-login) FORCE_LOGIN=1; shift ;;
    --with-jlens)  WITH_JLENS=1; shift ;;
    --hello)       HELLO=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "bootstrap.sh: unknown flag: $1" >&2; usage >&2; exit 64 ;;
  esac
done

# Reads one `export KEY=…` out of the PUSHED render without sourcing it: at
# this point the file has not been validated yet (that happens at step 5,
# before it is installed), and nothing may execute what it has not checked.
env_file_export_value() {  # env_file_export_value <key> — prints the value, unquoted
  [ -n "$ENV_FILE_FROM" ] && [ -f "$ENV_FILE_FROM" ] && [ -r "$ENV_FILE_FROM" ] || return 0
  sed -n "s/^export $1=//p" "$ENV_FILE_FROM" 2>/dev/null | tail -n 1 \
    | sed -e "s/^'\(.*\)'\$/\1/" -e 's/^"\(.*\)"$/\1/'
}

# c26 (WP5 step 12) — the module list, resolved in the same order as the
# login-node guard below: an explicit --modules, then the pushed render's
# STEERLAB_MODULES (environment.modules), then the ambient site environment,
# then this script's built-in fallback. "Load Miniforge3" was one institution's
# way of providing conda; here it is only what happens when nobody said.
if [ -z "$MODULES_SOURCE" ]; then
  rendered_modules="$(env_file_export_value STEERLAB_MODULES)"
  if [ -n "$rendered_modules" ]; then
    MODULES="$rendered_modules"; MODULES_SOURCE="the pushed render"
  elif [ -n "${STEERLAB_MODULES:-}" ]; then
    MODULES="$STEERLAB_MODULES"; MODULES_SOURCE="the sourced site environment"
  else
    MODULES="$MODULES_DEFAULT"; MODULES_SOURCE="fallback"
  fi
fi

# A digest with nothing to hash is a caller bug, not a no-op: it would silently
# turn the integrity gate off. Refuse before anything else happens.
if [ -n "$ENV_FILE_SHA256" ] && [ -z "$ENV_FILE_FROM" ]; then
  echo "bootstrap.sh: --env-file-sha256 requires --env-file-from" >&2
  exit 64
fi

# ------------------------------------------------------- dependency lock --
# WP6 R1. `Server/pyproject.toml` declares FLOORS, which is the right contract
# for a library and the wrong one for an instrument: two sites can satisfy
# them with different torch/transformers and produce different numbers. The
# committed per-platform lock is the exact-version half; each run's config.json
# then stamps what was ACTUALLY imported (`pythonEnvironment`), so nobody has
# to trust that this step did its job.
#
# Resolved from the RUNNING platform, because that is what the wheels must
# match. A platform with no committed lock (linux-aarch64, ppc64le, …) falls
# back to the floors, loudly.
LOCK_FILE=""
if [ "$USE_LOCK" -eq 1 ]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)   LOCK_NAME="requirements-linux-x86_64.lock" ;;
    Darwin-arm64)   LOCK_NAME="requirements-macos-arm64.lock" ;;
    *)              LOCK_NAME="" ;;
  esac
  if [ -n "$LOCK_NAME" ] && [ -f "$REPO/Server/$LOCK_NAME" ]; then
    LOCK_FILE="$REPO/Server/$LOCK_NAME"
  fi
fi

# torch and its CUDA runtime are deliberately EXCLUDED from the lock install.
# Which CUDA (or ROCm, or CPU-only) build a node wants is a SITE fact — it
# arrives here as --torch-index and is installed in step 3, before this. The
# lock's torch line is the recorded reference resolution, not an instruction to
# overwrite the site's build with PyPI's; every other pin is applied exactly.
LOCK_EXCLUDE_RE='^(torch|triton|pytorch-triton|nvidia-)'

# Prints the lock with the site-owned torch stack filtered out. Comment and
# `# via` continuation lines are dropped with it — pip does not need them and
# a dangling continuation under a removed pin would be misleading.
filtered_lock() {  # filtered_lock <lockfile>
  grep -E '^[A-Za-z0-9]' "$1" | grep -Ev "$LOCK_EXCLUDE_RE"
}

# Step ledger: names are camelCase to match the JSON report contract.
STEP_NAMES="condaDetect envCreate manifestCheck torchInstall serverInstall jlensStage envFile profileValidate helloJob"
S_condaDetect="pending"; S_envCreate="pending"; S_manifestCheck="pending"; S_torchInstall="pending"
S_serverInstall="pending"; S_envFile="pending"; S_profileValidate="pending"
S_jlensStage="skipped"
S_helloJob="skipped"
OK=true
CONDA_BIN=""
HELLO_JOB_ID=""

set_step() { eval "S_$1=\"$2\""; }
get_step() { eval "printf '%s' \"\$S_$1\""; }

fail_step() {  # fail_step <step> <message>  — records, reports, keeps going only when sensible
  set_step "$1" "failed"
  OK=false
  echo "bootstrap.sh: STEP FAILED [$1]: $2" >&2
}

report() {  # the LAST stdout line: machine-readable JSON
  steps_json=""
  for name in $STEP_NAMES; do
    [ -n "$steps_json" ] && steps_json="$steps_json,"
    steps_json="$steps_json\"$name\":\"$(get_step "$name")\""
  done
  extra=""
  [ -n "$HELLO_JOB_ID" ] && extra=",\"helloJobId\":\"$HELLO_JOB_ID\""
  printf '{"ok":%s,"steps":{%s},"envFile":"%s","prefix":"%s"%s}\n' \
    "$OK" "$steps_json" "$ENV_FILE" "$PREFIX" "$extra"
}

# ------------------------------------------------------------------ dry run --
if [ "$DRY_RUN" -eq 1 ]; then
  echo "bootstrap.sh plan (dry run — nothing executed):"
  if [ -n "$MODULES" ]; then
    echo "  [condaDetect]     module load $MODULES ($MODULES_SOURCE) || use conda/mamba on PATH"
  else
    echo "  [condaDetect]     no modules declared || use conda/mamba on PATH"
  fi
  echo "  [envCreate]       conda create -p $PREFIX python=$PYVER  (reused if present)"
  if [ -f "$REPO/deployment-manifest.json" ]; then
    echo "  [manifestCheck]   verify $REPO against deployment-manifest.json (sha256)"
  else
    echo "  [manifestCheck]   (skipped: no deployment-manifest.json — dev checkout push)"
  fi
  if [ "$INSTALL_TORCH" -eq 1 ]; then
    echo "  [torchInstall]    pip install torch --index-url $TORCH_INDEX"
  else
    echo "  [torchInstall]    (skipped: --no-torch)"
  fi
  if [ "$USE_LOCK" -eq 0 ]; then
    echo "  [serverInstall]   (--no-lock: resolving from pyproject floors)"
  elif [ -n "$LOCK_FILE" ]; then
    echo "  [serverInstall]   pip install -r $LOCK_FILE (minus torch/nvidia-*/triton — site-owned)"
  else
    echo "  [serverInstall]   (no committed lock for $(uname -s)-$(uname -m): resolving from pyproject floors)"
  fi
  echo "  [serverInstall]   pip install -e \"$REPO/Server[all]\""
  if [ "$WITH_JLENS" -eq 1 ]; then
    echo "  [jlensStage]      pip install -e \"$REPO/Server[jlens]\" (git-pinned; floors transformers>=5.5)"
    echo "                    + steerlab-server jlens acquire, into $HF_CACHE"
    for m in $JLENS_MODELS; do
      echo "                      acquire: $m"
    done
  else
    echo "  [jlensStage]      (skipped: pass --with-jlens to enable)"
  fi
  if [ -n "$ENV_FILE_FROM" ]; then
    # Materialized path: the flags above no longer describe the env file, so
    # printing them here would misreport the plan.
    echo "  [envFile]         install $ENV_FILE from $ENV_FILE_FROM"
    echo "                    (pre-rendered from the site profile; validated, then installed"
    echo "                    verbatim — none of the defaults above are consulted)"
    if [ -n "$ENV_FILE_SHA256" ]; then
      echo "                    sha256 must equal $ENV_FILE_SHA256"
    else
      echo "                    no --env-file-sha256 given: contents are validated, not pinned"
    fi
  else
    echo "  [envFile]         write $ENV_FILE (workspace=$WORKSPACE hf-cache=$HF_CACHE"
    echo "                    partition=$PARTITION gres=$GRES walltime=$WALLTIME account=${ACCOUNT:-<none>})"
    echo "                    BUILT-IN FALLBACK values — no site profile is rendered on this"
    echo "                    path, so GPU vocabulary, job memory, purge window, offline mode,"
    echo "                    node staging, and metadata root come from this script's defaults"
    echo "                    (pass --env-file-from to install a rendered profile instead)"
  fi
  echo "  [profileValidate] source $ENV_FILE && steerlab-server profile validate"
  if [ "$HELLO" -eq 1 ]; then
    echo "  [helloJob]        sbatch a 10-minute GPU hello job on $PARTITION/$GRES"
  else
    echo "  [helloJob]        (skipped: pass --hello to enable)"
  fi
  for name in $STEP_NAMES; do set_step "$name" "planned"; done
  [ -f "$REPO/deployment-manifest.json" ] || set_step manifestCheck "skipped"
  [ "$INSTALL_TORCH" -eq 0 ] && set_step torchInstall "skipped"
  [ "$WITH_JLENS" -eq 0 ] && set_step jlensStage "skipped"
  [ "$HELLO" -eq 0 ] && set_step helloJob "skipped"
  report
  exit 0
fi

# -------------------------------------------------------- login-node guard --
# WP5 step 10 (audit c35/c36): login/submit-node compute policy is SITE data,
# not this script's. It arrives as three rendered keys —
#
#   STEERLAB_LOGIN_NODE_PATTERNS            regexes matched against `hostname`
#                                           (absent/empty ⇒ no host is a login
#                                           node, so the hostname rule never
#                                           refuses)
#   STEERLAB_LOGIN_NODE_ALLOW_COMPUTE       0 ⇒ refuse on a matched host
#   STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION  1 ⇒ additionally require SLURM_JOB_ID
#
# — resolved in this order:
#
#   1. the PUSHED render (--env-file-from): the reviewed, hash-pinned statement
#      of this site's policy for THIS run, available before the file is
#      installed at step 5;
#   2. the ambient environment: a hand run with the installed site env already
#      sourced;
#   3. this script's BUILT-IN FALLBACK — the rule it has always applied
#      (hosts matching `^ss-sub` are login/submit nodes where module loads,
#      conda solves and pip installs are not permitted — the historical rule,
#      kept beside the historical constants in the renderer's v1 legacy
#      defaults table). Like the fallback env heredoc below, it is a
#      DEFAULT, not a statement about the site: every rendered profile — v1
#      included, from the renderer's legacy defaults — states this policy
#      itself, so the fallback is reached only by a hand run with no profile.
#      `steerlab-cli cluster preview --site <id>` prints what a site declares.
#
# ALLOW_COMPUTE is the declaration sentinel: the renderer always emits it, so
# its presence distinguishes "this site says compute is fine here" from "nobody
# said anything". The pushed file is read with sed — three keys, never sourced:
# at this point it has not been validated yet (that happens at step 5, before
# it is installed), and the guard must not execute what it has not checked.
# --dry-run never reaches this guard.
LOGIN_POLICY_SOURCE="fallback"
LOGIN_NODE_PATTERNS='^ss-sub'
LOGIN_NODE_ALLOW_COMPUTE=0
LOGIN_NODE_REQUIRE_ALLOCATION=1
rendered_allow="$(env_file_export_value STEERLAB_LOGIN_NODE_ALLOW_COMPUTE)"
if [ -n "$rendered_allow" ]; then
  LOGIN_POLICY_SOURCE="the pushed render"
  LOGIN_NODE_PATTERNS="$(env_file_export_value STEERLAB_LOGIN_NODE_PATTERNS)"
  LOGIN_NODE_ALLOW_COMPUTE="$rendered_allow"
  LOGIN_NODE_REQUIRE_ALLOCATION="$(env_file_export_value STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION)"
elif [ -n "${STEERLAB_LOGIN_NODE_ALLOW_COMPUTE:-}" ]; then
  LOGIN_POLICY_SOURCE="the sourced site environment"
  LOGIN_NODE_PATTERNS="${STEERLAB_LOGIN_NODE_PATTERNS:-}"
  LOGIN_NODE_ALLOW_COMPUTE="$STEERLAB_LOGIN_NODE_ALLOW_COMPUTE"
  LOGIN_NODE_REQUIRE_ALLOCATION="${STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION:-0}"
fi

GUARD_HOSTNAME="$(hostname 2>/dev/null || true)"
if [ "$FORCE_LOGIN" -ne 1 ]; then
  # Word-split the pattern list, with globbing OFF: a regex like ^ss-sub[0-9]+
  # is a shell glob too, and must not be expanded against the cwd.
  set -f
  GUARD_IS_LOGIN=0
  for pattern in $LOGIN_NODE_PATTERNS; do
    if printf '%s' "$GUARD_HOSTNAME" | grep -qE "$pattern" 2>/dev/null; then
      GUARD_IS_LOGIN=1
      break
    fi
  done
  set +f
  GUARD_REFUSAL=""
  if [ "$GUARD_IS_LOGIN" -eq 1 ] && [ "$LOGIN_NODE_ALLOW_COMPUTE" != "1" ]; then
    GUARD_REFUSAL="it matches this site's login/submit-node patterns, where compute is not allowed"
  elif [ "$LOGIN_NODE_REQUIRE_ALLOCATION" = "1" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    GUARD_REFUSAL="you are outside a Slurm allocation (no SLURM_JOB_ID) and this site requires one"
  fi
  if [ -n "$GUARD_REFUSAL" ]; then
    echo "bootstrap.sh: refusing to run on '$GUARD_HOSTNAME' — $GUARD_REFUSAL," >&2
    echo "  and this script loads modules and runs conda/pip." >&2
    if [ "$LOGIN_POLICY_SOURCE" = "fallback" ]; then
      # No site policy reached this run, so the message names the built-in
      # rule and its remedy verbatim — the historical text, kept where the
      # historical constants are.
      echo "  Policy: bootstrap.sh's BUILT-IN FALLBACK (hosts matching '^ss-sub' are" >&2
      echo "  login/submit nodes where module loads and computation are refused)," >&2
      echo "  because no rendered site profile reached this run. A site states its" >&2
      echo "  own rule in policy.loginNodes." >&2
      echo "  Remedy: get an allocation with your site's command (salloc / srun --pty," >&2
      echo "  or its published wrapper), then re-run this script." >&2
    else
      echo "  Policy: this site's policy.loginNodes, from $LOGIN_POLICY_SOURCE." >&2
      echo "  Remedy: get an allocation (salloc / srun --pty, or your site's wrapper)," >&2
      echo "  or submit this script as a batch job, then re-run it." >&2
    fi
    echo "  (Pip-only rerun from a transfer node: pass --force-login. Plan preview:" >&2
    echo "  --dry-run works anywhere.)" >&2
    exit 64
  fi
  if [ "$LOGIN_POLICY_SOURCE" != "fallback" ]; then
    echo "bootstrap.sh: login-node guard: '$GUARD_HOSTNAME' is permitted by this site's" \
         "policy.loginNodes (from $LOGIN_POLICY_SOURCE)"
  fi
fi

# ------------------------------------------------------- 1. conda detection --
if command -v module >/dev/null 2>&1 && [ -n "$MODULES" ]; then
  # Site data (environment.modules), or this script's fallback when no profile
  # reached the run. Failures are silent on purpose: a module that is absent
  # here may simply be unnecessary, and the conda probe below is the real test.
  set -f
  for module_name in $MODULES; do
    module load "$module_name" >/dev/null 2>&1
  done
  set +f
fi
if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN="$(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
fi
if [ -n "$CONDA_BIN" ]; then
  set_step condaDetect "ok"
  echo "bootstrap.sh: using $CONDA_BIN"
else
  fail_step condaDetect "no conda/mamba found after loading modules [$MODULES] \
($MODULES_SOURCE). If this site provides conda as a module, name it in the site \
profile's environment.modules (or pass --modules) and re-run from a node where \
that module loads. If it provides Python another way, set \
environment.pythonProvider accordingly. Otherwise install Miniforge \
(https://github.com/conda-forge/miniforge) and re-run."
  report; exit 1
fi

# ------------------------------------------------------------ 2. env create --
if [ -x "$PREFIX/bin/python" ]; then
  set_step envCreate "ok"
  echo "bootstrap.sh: env exists at $PREFIX — reusing (idempotent)"
else
  if "$CONDA_BIN" create -p "$PREFIX" "python=$PYVER" -y; then
    set_step envCreate "ok"
  else
    fail_step envCreate "conda create -p $PREFIX python=$PYVER failed — check quota on the prefix filesystem and the python version spelling"
    report; exit 1
  fi
fi
PYBIN="$PREFIX/bin/python"

# ------------------------------------------------------- 2b. manifest check --
# Packaged payloads (scripts/make-server-payload.sh) ship a
# deployment-manifest.json listing every file's sha256. When present, refuse
# to install a payload that has drifted from it; absent (a dev checkout
# push), skip silently — nothing changes for checkout-based pushes.
verify_deployment_manifest() {
  "$PYBIN" - "$REPO/deployment-manifest.json" "$REPO" <<'PYEOF'
import hashlib, json, os, sys

manifest_path, root = sys.argv[1], sys.argv[2]
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
problems = []
for relative, expected in sorted(manifest.get("files", {}).items()):
    path = os.path.join(root, relative)
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except FileNotFoundError:
        problems.append(f"{relative} is missing from the payload")
        continue
    except OSError as error:
        problems.append(f"{relative} cannot be read ({error})")
        continue
    if digest.hexdigest() != expected.lower():
        problems.append(f"{relative} does not match the manifest")
for problem in problems:
    print(f"manifestCheck: {problem}", file=sys.stderr)
sys.exit(1 if problems else 0)
PYEOF
}
if [ -f "$REPO/deployment-manifest.json" ]; then
  if verify_deployment_manifest; then
    set_step manifestCheck "ok"
    echo "bootstrap.sh: deployment payload matches its manifest"
  else
    fail_step manifestCheck "the deployment payload at $REPO does not match \
its deployment-manifest.json — the push was incomplete or the payload was \
modified. Rebuild it with scripts/make-server-payload.sh and re-push."
    report; exit 1
  fi
else
  set_step manifestCheck "skipped"
fi

# ---------------------------------------------------------- 3. torch install --
if [ "$INSTALL_TORCH" -eq 1 ]; then
  if "$PYBIN" -m pip install torch --index-url "$TORCH_INDEX"; then
    set_step torchInstall "ok"
  else
    fail_step torchInstall "pip install torch from $TORCH_INDEX failed — no egress from this node? Re-run from the site's transfer/data-mover node (environment.transferHost) if the env prefix is on a filesystem both mount, or pass --torch-index/--no-torch"
  fi
else
  set_step torchInstall "skipped"
  echo "bootstrap.sh: skipping torch (--no-torch)"
fi

# --------------------------------------------------------- 4. server install --
# [all] includes the gemmascope extra (sae-lens) — needed only for Gemma Scope
# SAE ANALYSIS; the server imports and runs fine without it, so pip trouble
# with sae-lens can be worked around with -e "$REPO/Server[lora,test]".
if [ ! -d "$REPO/Server" ]; then
  fail_step serverInstall "$REPO/Server not found — rsync/clone the SteerLab checkout to --repo first, then rerun with --repo pointing at it (Server/README.md, 'Install')"
else
  # WP6 R1: the pinned set first, so the editable install below finds every
  # requirement already satisfied at the LOCKED version instead of resolving
  # the floors to whatever PyPI published this morning.
  lock_ok=1
  if [ "$USE_LOCK" -eq 1 ] && [ -n "$LOCK_FILE" ]; then
    lock_tmp="$(mktemp)"
    filtered_lock "$LOCK_FILE" > "$lock_tmp"
    echo "bootstrap.sh: installing pinned dependencies from $LOCK_FILE (torch/nvidia-*/triton left to the site's --torch-index build)"
    if ! "$PYBIN" -m pip install -r "$lock_tmp"; then
      lock_ok=0
    fi
    rm -f "$lock_tmp"
  elif [ "$USE_LOCK" -eq 1 ]; then
    echo "bootstrap.sh: WARNING no committed dependency lock for $(uname -s)-$(uname -m) at $REPO/Server — resolving from pyproject FLOORS, so this node's torch/transformers are not pinned to any other node's. Regenerate locks with Server/scripts/update-locks.sh."
  else
    echo "bootstrap.sh: WARNING --no-lock: resolving from pyproject FLOORS; this node's dependency versions are not pinned."
  fi
  if [ "$lock_ok" -eq 0 ]; then
    fail_step serverInstall "pip install -r $LOCK_FILE failed — see pip output above. No egress from this node? Re-run from the site's transfer/data-mover node, or pass --no-lock to resolve from pyproject floors (unpinned — the run stamp will say so)"
  elif "$PYBIN" -m pip install -e "$REPO/Server[all]"; then
    set_step serverInstall "ok"
  else
    fail_step serverInstall "pip install -e $REPO/Server[all] failed — see pip output above"
  fi
fi

# ---------------------------------------------------- 4b. J-lens pre-stage --
# CACHE WARMING ONLY. Two operations are deliberately kept apart (J-lens plan
# §11.0.1): this step installs the pinned reference package and downloads the
# published lens bytes into $HF_CACHE. Import, conversion, and record creation
# are workspace-scoped and belong to `steerlab-server jlens import` — putting a
# copy of them here would test the copy rather than the product.
#
# A green jlensStage proves egress, allow_patterns scoping, quota headroom, and
# that the bytes landed. It proves NOTHING about whether the lens loads, the
# geometry matches, or any readout is interpretable. That is the G0 gate.
stage_jlens() {
  if [ "$(get_step serverInstall)" != "ok" ]; then
    fail_step jlensStage "serverInstall did not succeed — the jlens extra installs on top of it"
    return
  fi
  # Separate from Server[all] on purpose: this floors transformers>=5.5 and
  # pulls from a git URL, so it must never ride along with a routine bootstrap.
  if ! "$PYBIN" -m pip install -e "$REPO/Server[jlens]"; then
    fail_step jlensStage "pip install -e $REPO/Server[jlens] failed — no github.com egress from this node (the pin is a git URL, not PyPI)? Re-run from the site's transfer/data-mover node with --force-login, or install the extra later and re-run with --with-jlens"
    return
  fi

  mkdir -p "$HF_CACHE" 2>/dev/null
  # The floor, resolved like the login-node guard (WP5 step 10): the pushed
  # render first, then a sourced site environment, then this script's fallback.
  # The refusal has two voices for the same reason the guard's does — only the
  # FALLBACK path may name one institution's filesystem, because only there is
  # the number that institution's.
  min_free_kb="$JLENS_MIN_FREE_KB"
  prestage_source="fallback"
  rendered_prestage_gb="$(env_file_export_value STEERLAB_PRESTAGE_MIN_FREE_GB)"
  if [ -z "$rendered_prestage_gb" ]; then
    rendered_prestage_gb="${STEERLAB_PRESTAGE_MIN_FREE_GB:-}"
    [ -n "$rendered_prestage_gb" ] && prestage_source="the sourced site environment"
  else
    prestage_source="the pushed render"
  fi
  case "$rendered_prestage_gb" in
    ''|*[!0-9]*) : ;;   # unset or not a whole number: keep the fallback
    *) min_free_kb=$((rendered_prestage_gb * 1048576)) ;;
  esac
  free_kb=$(df -Pk "$HF_CACHE" 2>/dev/null | awk 'NR==2 {print $4}')
  if [ -n "$free_kb" ] && [ "$free_kb" -lt "$min_free_kb" ]; then
    if [ "$prestage_source" = "fallback" ]; then
      fail_step jlensStage "only $((free_kb / 1048576)) GiB free on $HF_CACHE; the pinned lenses need ~4 GB plus headroom. A shared cache often sits under a GROUP quota rather than a personal one — free space or point --hf-cache elsewhere. A part-way download leaves *.incomplete files; remove them before retrying"
    else
      fail_step jlensStage "only $((free_kb / 1048576)) GiB free on $HF_CACHE, below this site's constraints.storage.prestageMinFreeGB of $((min_free_kb / 1048576)) GiB (from $prestage_source) — free space or point --hf-cache elsewhere. A part-way download leaves *.incomplete files; remove them before retrying"
    fi
    return
  fi

  # Delegates to the acquisition verb rather than carrying its own downloader:
  # one implementation, two entry points. The verb owns pattern scoping, the
  # landed-file verification, progress, cancel, and failure triage; duplicating
  # any of that here is how the two drift apart.
  #
  # HF_HOME is passed through; the verb's child forces HF_HUB_OFFLINE=0 itself
  # (the env file this script writes sets it to 1 so study runs stay
  # offline-hermetic, and inheriting that would read as a network outage).
  failed=""
  for model in $JLENS_MODELS; do
    echo "bootstrap.sh: acquiring J-lens for $model"
    if ! HF_HOME="$HF_CACHE" "$PYBIN" -m steerlab_server.cli jlens acquire "$model"; then
      failed="$failed $model"
    fi
  done
  if [ -n "$failed" ]; then
    fail_step jlensStage "lens acquisition failed for:$failed — no huggingface.co egress from this node? Compute-node egress is site-specific; a transfer/data-mover node (environment.transferHost) is the usual fallback"
  else
    set_step jlensStage "ok"
  fi
}

if [ "$WITH_JLENS" -eq 1 ]; then
  stage_jlens
else
  set_step jlensStage "skipped"
fi

# --------------------------------------------------------------- 5. env file --
# ---- WP5 step 7: this heredoc is the MANUAL FALLBACK, not the site's facts --
#
# Until step 7 this function was the only writer of the cluster env file, and
# its constants were a SECOND copy of site configuration that the site profile
# could not correct (audit a1–a7: archive root, metadata root, purge window,
# GPU vocabulary, VRAM table, compute egress). The profile-driven path — the
# app and `steerlab-cli cluster bootstrap` — now renders the env file from the
# profile and passes `--env-file-from`, so on that path NONE of the values
# below are consulted, and facts this script never knew (archive root, node
# cache root, modules, QOS, login-node policy) travel with it.
#
# What survives here is the no-profile manual path: someone who ssh'd in and
# ran `bash bootstrap.sh` by hand. Every value below is therefore a FALLBACK,
# deliberately unchanged from what it has always been so that a manual run
# still converges, and deliberately labelled as a fallback in the file it
# writes so nobody reads it as a statement about the site. To make the site's
# own facts land here, render them: `steerlab-cli cluster preview --site <id>`
# prints the exact bytes, `cluster bootstrap plan/apply` installs them.
write_fallback_env_file() {
  tmp="$ENV_FILE.tmp.$$"
  {
    echo "# generated by bootstrap.sh $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# WP5: these are bootstrap.sh's built-in FALLBACK values, NOT a site"
    echo "# profile's declared facts — this file was synthesized from command-line"
    echo "# flags plus the defaults below because no rendered environment was"
    echo "# supplied (--env-file-from). The profile-driven path renders it from"
    echo "# the site profile instead; 'steerlab-cli cluster preview --site <id>'"
    echo "# prints those bytes."
    echo "# Re-running bootstrap.sh refuses to overwrite this file without --force,"
    echo "# so local edits are preserved by default."
    echo "# The env's bin on PATH: every '. this-file && steerlab-server …'"
    echo "# must work from a bare login shell — a fresh app session has no"
    echo "# memory of the prefix (live 2026-07-17: validate failed bare)."
    echo "export STEERLAB_PREFIX=\"$PREFIX\""
    echo "export PATH=\"$PREFIX/bin:\$PATH\""
    echo "export STEERLAB_SERVER_PROFILE=cluster"
    echo "export STEERLAB_EXECUTOR=slurm"
    echo "export STEERLAB_ROOT=\"$WORKSPACE\""
    echo "export STEERLAB_RUN_ROOT=\"$WORKSPACE/runs\""
    echo "# Metadata root: a lock-capable filesystem, not a parallel one."
    echo "export STEERLAB_METADATA_ROOT=\"\$HOME/.steerlab\""
    echo "export STEERLAB_SLURM_PARTITION=\"$PARTITION\""
    echo "export STEERLAB_SLURM_GRES=\"$GRES\""
    if [ -n "$SCRATCH_GRES" ]; then
      echo "# Node-local scratch requested as a gres of its own: a job that stages"
      echo "# a model to node-local disk must ACCOUNT for the space (cluster-operator"
      echo "# requirement 2026-08-19). Accounting only — the job removes its own"
      echo "# stage directory in the rendered script's EXIT trap."
      echo "export STEERLAB_SLURM_SCRATCH_GRES=\"$SCRATCH_GRES\""
    fi
    echo "export STEERLAB_SLURM_WALLTIME=\"$WALLTIME\""
    echo "export STEERLAB_SLURM_MEMORY=\"80G\""
    if [ -n "$ACCOUNT" ]; then
      echo "export STEERLAB_SLURM_ACCOUNT=\"$ACCOUNT\""
    else
      echo "# export STEERLAB_SLURM_ACCOUNT=<lab-account>   # if your site requires --account"
    fi
    echo "# Site GPU vocabulary + VRAM table (memory-fit preflight). A fallback:"
    echo "# declare scheduler.slurm.gpus in the site profile and render, or the"
    echo "# preflight sizes jobs against GPUs this site may not have."
    echo "export STEERLAB_SLURM_GPU_TYPES=\"L4,A100,H100\""
    echo "export STEERLAB_SLURM_GPU_VRAM=\"L4:24,A100:80,H100:80\""
    echo "# Requeue interrupted jobs; the checkpoint/resume contract finishes them."
    echo "export STEERLAB_SLURM_REQUEUE=1"
    echo "export STEERLAB_PURGE_DAYS=30"
    echo "export STEERLAB_PURGE_WARN_DAYS=20"
    echo "export HF_HOME=\"$HF_CACHE\""
    echo "# Pre-stage models where there is egress, then stay offline."
    echo "export HF_HUB_OFFLINE=1"
    echo "# Node-local model staging for GPU-session loads (one sequential copy"
    echo "# to node-local disk, then loads run at local speed). SINGLE-QUOTED so"
    echo "# \$SLURM_JOB_ID expands in the loader ON THE COMPUTE NODE, never"
    echo "# here; the loader skips staging gracefully if the dir can't exist."
    echo "export STEERLAB_NODE_STAGE_DIR='/lscratch/\$SLURM_JOB_ID'"
    echo "# Bearer token for command-bearing routes (generated once, chmod 600):"
    echo "export STEERLAB_AUTH_TOKEN=\"\$(cat \"\$HOME/.steerlab-token\")\""
  } > "$tmp" && mv "$tmp" "$ENV_FILE"
}

# ---- WP5 step 6: install a PRE-RENDERED env file (opt-in, --env-file-from) --
# The renderer that produced the file is `ClusterEnvironmentRenderer` (Swift) /
# `steerlab_server.api.site_environment` (Python), both byte-pinned to
# prompts/fixtures/cluster-site-profile/. This script does not parse the
# profile and never will: it validates the SHAPE of what it was handed and
# installs it atomically, so a truncated push, a stray HTML error page, or a
# file that would not survive `.` are refusals rather than a wedged cluster.

env_file_digest() {  # env_file_digest <path> — prints lowercase hex, or nothing
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" 2>/dev/null | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$1" 2>/dev/null | awk '{print $NF}'
  fi
}

verify_env_file_from() {
  src="$ENV_FILE_FROM"
  if [ ! -e "$src" ]; then
    echo "bootstrap.sh: --env-file-from $src does not exist — the render was never pushed" >&2
    return 1
  fi
  if [ ! -f "$src" ] || [ ! -r "$src" ]; then
    echo "bootstrap.sh: --env-file-from $src is not a readable regular file" >&2
    return 1
  fi
  if [ ! -s "$src" ]; then
    echo "bootstrap.sh: --env-file-from $src is empty — a partial or failed push" >&2
    return 1
  fi
  # Shape: the renderer emits comments, blank lines, and `export KEY=…` only.
  # Anything else means this is not a rendered env file, and sourcing it would
  # run whatever it is.
  offenders="$(grep -n -E -v '^[[:space:]]*(#.*)?$|^export [A-Za-z_][A-Za-z0-9_]*=' \
      "$src" 2>/dev/null | head -3)"
  if [ -n "$offenders" ]; then
    echo "bootstrap.sh: --env-file-from $src is not a rendered env file — every line must" >&2
    echo "  be blank, a comment, or 'export KEY=…'. First offending line(s):" >&2
    printf '  %s\n' "$offenders" >&2
    return 1
  fi
  if ! bash -n "$src" 2>/dev/null; then
    echo "bootstrap.sh: --env-file-from $src is not valid shell — sourcing it would fail" >&2
    return 1
  fi
  if [ -n "$ENV_FILE_SHA256" ]; then
    expected="$(printf '%s' "$ENV_FILE_SHA256" | tr 'A-Z' 'a-z')"
    actual="$(env_file_digest "$src" | tr 'A-Z' 'a-z')"
    if [ -z "$actual" ]; then
      echo "bootstrap.sh: --env-file-sha256 was given but no sha256 tool (sha256sum/shasum/" >&2
      echo "  openssl) is on PATH here, so the reviewed bytes cannot be verified. Refusing" >&2
      echo "  rather than installing unverified. Remedy: re-run without --env-file-sha256." >&2
      return 1
    fi
    if [ "$actual" != "$expected" ]; then
      echo "bootstrap.sh: --env-file-from $src does not match the reviewed plan." >&2
      echo "  expected sha256 $expected" >&2
      echo "  actual   sha256 $actual" >&2
      echo "  The pushed file is stale or was modified — re-run the bootstrap plan and" >&2
      echo "  re-approve it, which re-renders and re-pushes." >&2
      return 1
    fi
  fi
  return 0
}

install_env_file_from() {
  verify_env_file_from || return 1
  tmp="$ENV_FILE.tmp.$$"
  cp "$ENV_FILE_FROM" "$tmp" && mv "$tmp" "$ENV_FILE"
}

# One dispatch point: with no --env-file-from this is the manual fallback,
# called exactly where it always was.
materialize_env_file() {
  if [ -n "$ENV_FILE_FROM" ]; then
    install_env_file_from
  else
    write_fallback_env_file
  fi
}

if [ ! -f "$HOME/.steerlab-token" ]; then
  if command -v openssl >/dev/null 2>&1; then
    umask_prev=$(umask); umask 077
    openssl rand -hex 32 > "$HOME/.steerlab-token"
    umask "$umask_prev"
    echo "bootstrap.sh: generated $HOME/.steerlab-token"
  else
    echo "bootstrap.sh: WARNING: openssl not found — create ~/.steerlab-token yourself (any long random string, chmod 600)" >&2
  fi
fi

if [ -f "$ENV_FILE" ] && [ "$FORCE" -ne 1 ]; then
  set_step envFile "skipped"
  echo "bootstrap.sh: $ENV_FILE exists — keeping it (pass --force to regenerate)"
elif materialize_env_file; then
  set_step envFile "ok"
  if [ -n "$ENV_FILE_FROM" ]; then
    echo "bootstrap.sh: installed $ENV_FILE from $ENV_FILE_FROM (rendered from the site profile)"
  else
    echo "bootstrap.sh: wrote $ENV_FILE from BUILT-IN FALLBACK values — no site"
    echo "  profile was rendered, so the GPU vocabulary, job memory, purge window,"
    echo "  offline mode, node staging, and metadata root in it are this script's"
    echo "  defaults, not this site's declared facts. To materialize the profile"
    echo "  instead: steerlab-cli cluster preview --site <id> (read the bytes),"
    echo "  then cluster bootstrap plan/apply (installs them via --env-file-from)."
  fi
elif [ -n "$ENV_FILE_FROM" ]; then
  fail_step envFile "could not install $ENV_FILE from $ENV_FILE_FROM — see the refusal above"
else
  fail_step envFile "could not write $ENV_FILE"
fi

# -------------------------------------------------------- 6. profile validate --
if [ -f "$ENV_FILE" ] && [ -x "$PYBIN" ] && [ "$(get_step serverInstall)" = "ok" ]; then
  # shellcheck disable=SC1090
  if ( . "$ENV_FILE"
       mkdir -p "$STEERLAB_METADATA_ROOT" "$STEERLAB_ROOT" "$STEERLAB_RUN_ROOT" 2>/dev/null
       "$PYBIN" -m steerlab_server.cli profile validate ); then
    set_step profileValidate "ok"
  else
    fail_step profileValidate "profile validate reported failures — fix what it flags (roots, sbatch on PATH, token) and re-run"
  fi
else
  set_step profileValidate "skipped"
  echo "bootstrap.sh: skipping profile validate (missing env file or server install)"
fi

# -------------------------------------------------------------- 7. hello job --
if [ "$HELLO" -eq 1 ]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    fail_step helloJob "sbatch not on PATH — run --hello from a Slurm submit host"
  elif ! mkdir -p "$WORKSPACE" 2>/dev/null; then
    fail_step helloJob "could not create workspace $WORKSPACE — pass --workspace <dir> for this site"
  else
    # Script, working dir, and logs all live workspace-side (scratch, on most
    # sites): the required-header rule (--ntasks) applies to this job too,
    # and a relative --output would land wherever bootstrap happened to run
    # (often /home, where job data must not go).
    hello_script="$WORKSPACE/steerlab-hello.sbatch"
    {
      echo "#!/bin/bash"
      echo "#SBATCH --job-name=steerlab-hello-gpu"
      echo "#SBATCH --partition=$PARTITION"
      echo "#SBATCH --gres=$GRES"
      echo "#SBATCH --ntasks=1"
      echo "#SBATCH --cpus-per-task=4"
      echo "#SBATCH --mem=16G"
      echo "#SBATCH --time=00:10:00"
      echo "#SBATCH --chdir=$WORKSPACE"
      echo "#SBATCH --output=$WORKSPACE/%x.%j.out"
      [ -n "$ACCOUNT" ] && echo "#SBATCH --account=$ACCOUNT"
      echo "nvidia-smi"
      echo "\"$PYBIN\" -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no CUDA\")'"
    } > "$hello_script"
    # Submit FROM the workspace too: --chdir moves the JOB's working dir,
    # not the submit command's cwd (sites teach "submit from /scratch").
    submit_out="$(cd "$WORKSPACE" && sbatch "$hello_script" 2>&1)"
    if [ $? -eq 0 ]; then
      HELLO_JOB_ID="$(printf '%s' "$submit_out" | awk '{print $NF}')"
      set_step helloJob "ok"
      echo "bootstrap.sh: hello job submitted: $submit_out (watch: squeue --me)"
    else
      fail_step helloJob "sbatch failed: $submit_out — check partition/gres/account for this site"
    fi
  fi
fi

report
if [ "$OK" = "true" ]; then exit 0; else exit 1; fi
