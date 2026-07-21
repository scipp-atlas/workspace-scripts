#!/bin/bash
# Batch pyhs3-vs-quickFit comparison over groups of generated workspaces.
#
# Evaluates the pyhs3 NLL against the quickFit reference for many workspaces at
# once by pairing each workspaces/<stem>.json with its scans/<stem>_muscan.json
# and handing the pairs to pyhs3_eval/eval_simple_muscan.py.
#
# Run from the repository root, in a pyhs3/pytensor environment (see
# pyhs3_eval/README.md). Flags:
#   --all      -> every exported workspace
#   --events   -> the 3-channel base variant across yield scale factors
#   --channels -> the base variant across channel counts
set -uo pipefail
cd "$(dirname "$0")"

EVAL="python3 pyhs3_eval/eval_simple_muscan.py"

all() {
    local pairs=() ws stem muscan
    for ws in workspaces/*.json; do
        [[ -e "$ws" ]] || continue                 # guard: unmatched glob
        stem="$(basename "$ws" .json)"
        stem="${stem%_noaux}"                       # aux-stripped exports reuse the standard scan
        muscan="scans/${stem}_muscan.json"
        [[ -e "$muscan" ]] || { echo "skip $(basename "$ws"): no $(basename "$muscan")" >&2; continue; }
        pairs+=( --pair "$ws" "$muscan" )
    done
    $EVAL "${pairs[@]}" --plot-nll
}

num_events() {
    local pairs=() ws stem muscan
    for ws in workspaces/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield*.json; do
        [[ -e "$ws" ]] || continue                 # guard: unmatched glob
        stem="$(basename "$ws" .json)"
        muscan="scans/${stem}_muscan.json"
        [[ -e "$muscan" ]] || { echo "skip $(basename "$ws"): no $(basename "$muscan")" >&2; continue; }
        pairs+=( --pair "$ws" "$muscan" )
    done
    $EVAL "${pairs[@]}" --plot-resid "pyhs3_eval/event_comparison.pdf" --plot-resid-field "yield"
}

num_channels() {
    echo "starting channels......."
    local pairs=() ws stem muscan
    for ws in workspaces/*ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json; do
        [[ -e "$ws" ]] || continue                 # guard: unmatched glob
        stem="$(basename "$ws" .json)"
        stem="${stem%_noaux}"                       # aux-stripped exports reuse the standard scan
        muscan="scans/${stem}_muscan.json"
        [[ -e "$muscan" ]] || { echo "skip $(basename "$ws"): no $(basename "$muscan")" >&2; continue; }
        pairs+=( --pair "$ws" "$muscan" )
    done
    $EVAL "${pairs[@]}" --plot-resid "pyhs3_eval/channel_comparison.pdf"
}

for arg in "$@"; do
    case "$arg" in
        --all) all ;;
        --events) num_events ;;
        --channels) num_channels ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done
