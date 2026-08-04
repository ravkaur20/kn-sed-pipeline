#!/usr/bin/env bash
# Build PDF user guide (requires pandoc + LaTeX or wkhtmltopdf).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MD="$ROOT/docs/PIPELINE_USER_GUIDE.md"
PDF="$ROOT/docs/PIPELINE_USER_GUIDE.pdf"
if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found; install pandoc to build PDF" >&2
  exit 1
fi
pandoc "$MD" -o "$PDF" \
  --pdf-engine=pdflatex \
  --toc --toc-depth=3 \
  -V geometry:margin=1in \
  -V colorlinks=true \
  -V linkcolor=blue
echo "Wrote $PDF"
