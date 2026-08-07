#!/bin/bash
# Run the standardized mu scan against a real (non-toy) workspace on the
# remote (requires ROOT + quickFit; source setup_local.sh first).
#
# muscan.py reads the post-fit parameters from the quickFit result .root files
# and writes the same muscan.json schema used for the toy workspaces, so the
# pyhs3 evaluation is identical for both:
#
#   python3 pyhs3_eval/eval_simple_muscan.py \
#       --workspace <real-workspace HS3 .json> \
#       --scan      <outdir>/<wsName>_muscan.json \
#       --analysis  CombinedPdf_combData --cache-dir pyhs3_eval/cache

basedir=/home/mhance/pyhs3
export XML=workspace_FINAL_ISOBUGFIX
export wsName=WS-bbyy-non-resonant-non-param
export wsFile=${basedir}/${XML}/${wsName}.root
export outdir=output__${XML}

muvals="-0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5"

python3 muscan.py \
    --input   ${wsFile} \
    --mu-vals "${muvals}" \
    --nll-offset \
    --logdir  ${outdir} \
    --output  ${outdir}/${wsName}_muscan.json
