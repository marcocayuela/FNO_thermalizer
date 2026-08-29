#!/bin/bash
# TO RUN ON MESU (login node, not a compute node -- compute nodes may not
# have internet access), NOT locally: the 2 "train"-split chunks (gamma=1.4
# "Dry_air", chunk_0 + chunk_40, ~42 GB) AND the "valid"-split chunk_40
# (~5.3 GB, same file already symlinked locally from RT_FNO_vs_WNO's own
# download -- fetched again here directly from HF rather than transiting
# through the local machine, which would burn the user's own bandwidth for
# no reason). Mirrors RT_FNO_vs_WNO/src/fetch_train_chunks_mesu.sh, adapted
# to thermalizer's own $STORE/data/<exp_dir>/... convention (cf.
# run_emul_euler.sh/run_diff_euler.sh) -- note both splits live under
# data/train/... and data/valid/... on HF, so thermalizer's own local/mesu
# layout mirrors that exactly (euler_multi_quadrants_openBC/data/{train,valid}/),
# no shortened "valid/" without the "data/" prefix.
#
# Usage (once connected to mesu, `ssh mesu`):
#   cd ~/thermalizer/src   # or wherever the code was synced
#   bash scripts/fetch_train_chunks_euler_mesu.sh /store/absolute/path/on/mesu
#
# (resolve STORE_PATH beforehand with `echo $STORE` if needed -- same
# remark as in RT_FNO_vs_WNO: don't rely on $STORE expanding through a
# script run other than interactively.)

set -e

STORE_PATH=${1:?"Usage: bash scripts/fetch_train_chunks_euler_mesu.sh /store/absolute/path/on/mesu"}

module purge
module load python/3.11
source activate fto
python -c "import huggingface_hub" 2>/dev/null || pip install huggingface_hub

LOCAL_DIR="$STORE_PATH/data/euler_multi_quadrants_openBC"
mkdir -p "$LOCAL_DIR/data/train" "$LOCAL_DIR/data/valid"

python3 - "$LOCAL_DIR" << 'PYEOF'
import sys
from huggingface_hub import hf_hub_download

local_dir = sys.argv[1]

for subdir, fname in [
    ("data/train", "euler_multi_quadrants_openBC_gamma_1.4_Dry_air_20_chunk_0.hdf5"),
    ("data/train", "euler_multi_quadrants_openBC_gamma_1.4_Dry_air_20_chunk_40.hdf5"),
    ("data/valid", "euler_multi_quadrants_openBC_gamma_1.4_Dry_air_20_chunk_40.hdf5"),
]:
    path = hf_hub_download(
        repo_id="polymathic-ai/euler_multi_quadrants_openBC",
        repo_type="dataset",
        filename=f"{subdir}/{fname}",
        local_dir=local_dir,
    )
    print("Downloaded to", path)
PYEOF

echo "Done. Check:"
ls -la "$LOCAL_DIR/data/train" "$LOCAL_DIR/data/valid"
