# shell script utility to evaluate all pyhs3 and qf nlls in 
# specified workspaces, workspace_comparison.sh is run with
# the following flags:

#	--all      -> evaluates all workspaces
#	--events   -> evaluates all workspaces with variations 
#		      on number of events
#	--channels -> evaluates all workspaces with variations 
#		      on number of channels

# this shell script is meant to be run in pyhs3 with
# workspace-scripts at relative location ../workspace-scripts.
# Otherwise the location matching must be changed

all() {
    local dir="../workspace-scripts"
    local pairs=() ws stem suffix muscan
    for ws in "$dir"/workspaces/*.json; do
        [[ -e "$ws" ]] || continue                 # guard: unmatched glob
        stem="$(basename "$ws" .json)"
        stem="${stem%_noaux}"                       # aux-stripped exports reuse the standard scan
        suffix="${stem#simple_workspace}"           # "" | "_generic" | "_gensig_10_channels" | ...
        muscan="$dir/scans/${suffix}_muscan.json"
        [[ -e "$muscan" ]] || { echo "skip $(basename "$ws"): no $(basename "$muscan")" >&2; continue; }
        pairs+=( --pair "$ws" "$muscan" )
    done
    pixi run python examples/eval_simple_muscan.py "${pairs[@]}" --plot-nlls
}

num_events() {
    local dir="../workspace-scripts"
    local pairs=() ws stem muscan
    for ws in "$dir"/workspaces/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield*.json; do
        [[ -e "$ws" ]] || continue                 # guard: unmatched glob
        stem="$(basename "$ws" .json)"                       # aux-stripped exports reuse the standard scan
        muscan="$dir/scans/${stem}_muscan.json"     # full-stem lookup (no prefix to strip here)
        [[ -e "$muscan" ]] || { echo "skip $(basename "$ws"): no $(basename "$muscan")" >&2; continue; }
        pairs+=( --pair "$ws" "$muscan" )
    done
    pixi run python examples/eval_simple_muscan.py "${pairs[@]}" --plot-resid "examples/event_comparison.pdf" --plot-resid-field "yield"
}

num_channels() {
    echo "starting channels......."
    local dir="../workspace-scripts"
    local pairs=() ws stem muscan
    for ws in "$dir"/workspaces/*ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json; do
        [[ -e "$ws" ]] || continue                 # guard: unmatched glob
        stem="$(basename "$ws" .json)"
        stem="${stem%_noaux}"                       # aux-stripped exports reuse the standard scan
        muscan="$dir/scans/${stem}_muscan.json"     # full-stem lookup (no prefix to strip here)
        [[ -e "$muscan" ]] || { echo "skip $(basename "$ws"): no $(basename "$muscan")" >&2; continue; }
        pairs+=( --pair "$ws" "$muscan" )
    done
    pixi run python examples/eval_simple_muscan.py "${pairs[@]}" --plot-resid "examples/channel_comparison.pdf"
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
