#!/usr/bin/env bash
# Build scientific writeup PDF (requires pandoc + pdflatex).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MD="$ROOT/docs/PIPELINE_WRITEUP.md"
PDF="$ROOT/docs/PIPELINE_WRITEUP.pdf"
if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found; install pandoc to build PDF" >&2
  exit 1
fi
pandoc "$MD" -o "$PDF" \
  --pdf-engine=pdflatex \
  --toc --toc-depth=2 \
  -V geometry:margin=0.9in \
  -V colorlinks=true \
  -V linkcolor=blue
echo "Wrote $PDF"
