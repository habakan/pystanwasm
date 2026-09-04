#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$HERE/node_modules/stanwasm"
DEST="$HERE/files/stanwasm"

if [ ! -d "$SRC" ]; then
  echo "error: node_modules/stanwasm not found — run 'npm install' first" >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$SRC/." "$DEST/"
rm -f "$DEST/pkg/.gitignore"
