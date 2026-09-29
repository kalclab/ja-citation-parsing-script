#!/bin/sh
# Usage: ./eval.sh <run-dir1> [run-dir2 ...]
#
# Evaluate one or more experiment run artifacts against BibTeX gold records.
# Output goes to evaluations/<dir1>[_vs_<dir2>...]/.
#
# For custom options (--dataset-path, --styles, etc.), run directly:
#   uv run python -m scripts.evaluate --help

set -eu

if [ $# -eq 0 ]; then
  echo "Usage: ./eval.sh <run-dir1> [run-dir2 ...]" >&2
  echo "" >&2
  echo "Example:" >&2
  echo "  ./eval.sh artifacts/pilot-grobid-bibtex-001" >&2
  echo "  ./eval.sh artifacts/pilot-grobid-bibtex-001 artifacts/pilot-llm-bibtex-b1-002-sol" >&2
  exit 1
fi

RUN_ARGS=""
LABEL=""
for dir in "$@"; do
  RUN_ARGS="$RUN_ARGS --run-dir $dir"
  name=$(basename "$dir")
  if [ -z "$LABEL" ]; then
    LABEL="$name"
  else
    LABEL="${LABEL}_vs_${name}"
  fi
done

OUTPUT_DIR="evaluations/${LABEL}"

uv run python -m scripts.evaluate \
  $RUN_ARGS \
  --field-policy observable-fields-v1 \
  --gold-source derived \
  --output-dir "$OUTPUT_DIR"
