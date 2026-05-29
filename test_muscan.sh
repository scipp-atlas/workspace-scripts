basedir=/home/mhance/pyhs3
export XML=workspace_FINAL_ISOBUGFIX
#export XML=workspace
export wsName=WS-bbyy-non-resonant-non-param
export wsFile=${basedir}/${XML}/${wsName}.root
export outdir=output__${XML}
mkdir -p $outdir

#muvals="-0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.861717 0.9 0.9256531815323304 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5"
muvals="-0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5"
#muvals="0 0.2 0.4 0.6 0.8 0.9 1.0 1.1 1.2 1.4 1.6 1.8 2.0"
#muvals="-0.5 0.0 0.5 0.9256531815323304 1.0 1.5 2.0"

run_quickFit=1

# get nll values for fixed mu values
rm -f ${outdir}/nlls.txt
for mu in ${muvals}; do
    if [[ ${run_quickFit} != 0 ]]; then
	quickFit -f ${wsFile} \
		 -p mu_HH=${mu} \
		 -w combWS -d combData \
		 -o ${outdir}/result_yybb__mu_${mu}.root \
		 --nllOffset 0 \
	    |& tee ${outdir}/log__mu_${mu}.txt
    fi
    
    scripts/get_nll.py ${outdir}/result_yybb__mu_${mu}.root >> ${outdir}/nlls.txt
    echo "Done with mu=${mu}"
done

    
# check best-fit value of mu, with uncertainty
#if [[ ${run_quickFit} != 0 ]]; then
#    quickFit -f ${wsFile} \
#	     -p mu_HH=1_-5_5 \
#	     -w combWS -d combData \
#	     --minos 1 \
#	     -o ${outdir}/result_yybb__mu_fit.root \
#	|& tee ${outdir}/log__mu_fit.txt
#fi
    
# don't use this snapshot
#	     -s unconditionalGlobs_muhat,unconditionalNuis_muhat \

#quickFit -f ${wsFile} \
#	 -p mu_HH=0,mu_H=1 \
#	 -w combWS -d AsimovData_1 \
#	 -s conditionalNuis_1,conditionalGlobs_1 \
#	 -o ${outdir}/result_yybb__mu_fix_0_Asimov.root \
#    |& tee ${outdir}/result_yybb__mu_fix_0_Asimov.log

