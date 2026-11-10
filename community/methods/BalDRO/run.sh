#!/usr/bin/env bash
set -euo pipefail

# Run from the Open-Unlearning repository root.
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <tofu|muse>/<script_name>.sh [script arguments]" >&2
    exit 2
fi

script="$1"
shift
exec bash "$(dirname "${BASH_SOURCE[0]}")/scripts/${script}" "$@"
