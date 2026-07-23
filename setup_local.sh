export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
source $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "views LCG_108 x86_64-el9-gcc14-opt"

# Point at your local quickFit build (must be compiled against the same LCG/ROOT
# sourced above). Override for a different checkout with:  export QUICKFIT_DIR=...
QUICKFIT_DIR=${QUICKFIT_DIR:-/home/allex/git-projects/quickFit}
export PATH=${PATH}:${QUICKFIT_DIR}/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${QUICKFIT_DIR}/lib
