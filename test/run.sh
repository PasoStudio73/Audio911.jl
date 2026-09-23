#!/usr/bin/env bash
# Repeatable Julia entry points for Audio911.jl development.
#
#   test/run.sh instantiate   # resolve and instantiate the project env
#   test/run.sh test          # full test suite, memory-capped, timed
#   test/run.sh file X.jl     # run one test file (test/X.jl, or an absolute path)
#   test/run.sh docs          # build the Documenter site into docs/build
#   test/run.sh bench         # allocation / timing profile of the pipeline
#   test/run.sh oracle [X.py] # regenerate the audioFlux fixtures (all, or the given scripts)
#
# `oracle` runs the scripts in test/audioflux_sources/ with the Python of the
# virtual environment $AF_VENV (default ${TMPDIR:-/tmp}/audio911-oracle-venv),
# creating it with `pip install audioflux scipy numpy` when it does not exist.
# The scripts write test/audioflux_files/<feature>/*.mat.
#
# Every Julia invocation uses --project="$ROOT" (or --project=docs), a 2G heap hint,
# and, when available, a systemd-run scope that caps memory at 4G.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd -P)
export A911_ROOT="$ROOT"

JULIA=${JULIA:-julia}
# the machine-wide startup.jl activates another project; never load it
JFLAGS=(--startup-file=no --threads=auto)
HEAP=${HEAP:---heap-size-hint=2G}
LOG_DIR=${LOG_DIR:-.fm}
mkdir -p "$LOG_DIR"

# WRAP is an array of command words prepended to every Julia invocation.
WRAP=()
if command -v systemd-run >/dev/null 2>&1 && systemd-run --user --scope -p MemoryMax=4G true >/dev/null 2>&1; then
    WRAP=(systemd-run --user --scope -p MemoryMax=4G -q)
fi
TIMED=()
if [ -x /usr/bin/time ]; then
    TIMED=(/usr/bin/time -v)
fi
wrap()  { "${WRAP[@]}" "$@"; }
timed() { "${TIMED[@]}" "${WRAP[@]}" "$@"; }

case "${1:-test}" in
    instantiate)
        wrap "$JULIA" "${JFLAGS[@]}" --project="$ROOT" "$HEAP" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
        ;;
    test)
        timed "$JULIA" "${JFLAGS[@]}" --project="$ROOT" "$HEAP" -e 'using Pkg; Pkg.test()' 2>&1 | tee "$LOG_DIR/test.log"
        ;;
    file)
        shift
        # runs a single test file with the test extras available via a temporary env
        wrap "$JULIA" "${JFLAGS[@]}" --project="$ROOT" "$HEAP" -e '
            using Pkg
            Pkg.activate(; temp=true)
            Pkg.develop(path=ENV["A911_ROOT"])
            Pkg.add(["MAT", "Plots", "RecipesBase", "Test"])
            for f in ARGS; include(isabspath(f) ? f : joinpath(ENV["A911_ROOT"], "test", f)); end' "$@"
        ;;
    docs)
        wrap "$JULIA" "${JFLAGS[@]}" --project="$ROOT/docs" "$HEAP" -e 'using Pkg; Pkg.instantiate(); include(joinpath(ENV["A911_ROOT"], "docs/make.jl"))' 2>&1 | tee "$LOG_DIR/docs.log"
        ;;
    bench)
        timed "$JULIA" "${JFLAGS[@]}" --project="$ROOT" "$HEAP" "$ROOT/test/bench.jl" 2>&1 | tee "$LOG_DIR/bench.log"
        ;;
    oracle)
        shift
        AF_VENV=${AF_VENV:-${TMPDIR:-/tmp}/audio911-oracle-venv}
        if [ ! -x "$AF_VENV/bin/python" ]; then
            python3 -m venv "$AF_VENV"
            "$AF_VENV/bin/pip" install --quiet audioflux scipy numpy
        fi
        if [ $# -eq 0 ]; then
            set -- "$ROOT"/test/audioflux_sources/[a-z]*.py
        fi
        for script in "$@"; do
            case "$script" in /*) ;; *) script="$ROOT/test/audioflux_sources/$script" ;; esac
            echo "oracle: $(basename "$script")"
            wrap "$AF_VENV/bin/python" "$script"
        done
        ;;
    *)
        echo "unknown command: $1" >&2; exit 2
        ;;
esac
