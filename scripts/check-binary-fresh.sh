#!/bin/zsh
# Is a built binary at least as new as its sources? — the stale-binary trap
# detector shared by the Finder launchers (SteerLab.app, SteerLab Server.app),
# which historically rebuilt only when the binary was MISSING and so silently
# launched yesterday's build after a source change.
#
#   usage: check-binary-fresh.sh <binary> <source-path>...
#
# Each <source-path> may be a file or a directory (directories are searched
# recursively for source files: .swift/.metal/.h/.c/.m/.json/Package.resolved).
#
# Exit codes: 0 = fresh (binary exists and no source file is newer)
#             1 = stale (prints the first newer source file found)
#             2 = binary missing, or usage error
set -u

if (( $# < 2 )); then
  echo "usage: check-binary-fresh.sh <binary> <source-path>..." >&2
  exit 2
fi

binary="$1"; shift
if [[ ! -e "$binary" ]]; then
  echo "binary missing: $binary"
  exit 2
fi

for source_path in "$@"; do
  if [[ -d "$source_path" ]]; then
    newer="$(find "$source_path" -type f \
      \( -name '*.swift' -o -name '*.metal' -o -name '*.h' -o -name '*.c' \
         -o -name '*.m' -o -name '*.json' -o -name 'Package.resolved' \) \
      -newer "$binary" -print -quit 2>/dev/null)"
    if [[ -n "$newer" ]]; then
      echo "stale: $newer is newer than $binary"
      exit 1
    fi
  elif [[ -f "$source_path" ]]; then
    if [[ "$source_path" -nt "$binary" ]]; then
      echo "stale: $source_path is newer than $binary"
      exit 1
    fi
  fi
  # Nonexistent source paths are skipped: freshness is judged on what exists.
done

echo "fresh: $binary is up to date"
exit 0
