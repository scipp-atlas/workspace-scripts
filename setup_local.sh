export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
source $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "views LCG_108 x86_64-el9-gcc14-opt"

QUICKFIT_DIR=/home/mhance/pyhs3/quickFit
export PATH=${PATH}:${QUICKFIT_DIR}/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${QUICKFIT_DIR}/lib
