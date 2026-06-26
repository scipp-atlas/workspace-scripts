#!/bin/bash
# Usage: run_simple_fit.sh [workspace.root]
#   If no argument is given, defaults to simple_workspace.root.
#   The external constraint PDF (constr_alpha_sigma) is included automatically
#   when present in the workspace.
#source "$(dirname "$0")/setup_local.sh"

wsfile=${1:-simple_workspace.root}
logdir=output_simple
mkdir -p "$logdir"

# Detect whether the workspace contains an external constraint PDF.
constr=$(python3 - "$wsfile" <<'EOF'
import sys, ROOT
ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)
f = ROOT.TFile(sys.argv[1])
ws = f.Get("combWS")
names = []
if ws:
     for pdf in ws.allPdfs():
          n = pdf.GetName()
          if n.startswith("constr_"):
               names.append(n)
print(",".join(names), end="")
EOF
)

constr_arg=()
[[ -n "$constr" ]] && constr_arg=(--externalConstraint "$constr")

stem=$(basename "$wsfile" .root)
result="${logdir}/${stem}_result.root"

# pipefail so quickFit's exit status survives the `tee`, otherwise failures are
# silently masked by tee's (always-0) status.
set -o pipefail
quickFit -f "$wsfile" \
         -w combWS -m ModelConfig -d combData \
         -p mu_sig=1_-5_10 \
         "${constr_arg[@]}" \
         -o "$result" \
    |& tee "${logdir}/${stem}_fit.log"
status=$?

if [[ $status -ne 0 || ! -f "$result" ]]; then
    echo "ERROR: quickFit failed for ${stem} (exit=${status}); no result file written." >&2
    echo "       see ${logdir}/${stem}_fit.log" >&2
    exit "${status:-1}"
fi
exit 0
