#!/bin/bash
# TO RUN ON MESU (login node, not a compute node -- compute nodes may not
# have internet access), NOT locally: the 2 "train"-split chunks (gamma=1.4
# "Dry_air", chunk_0 + chunk_40, ~42 GB total) are too large to transit
# through the local machine. Mirrors
# RT_FNO_vs_WNO/src/fetch_train_chunks_mesu.sh, adapted to thermalizer's own
# $STORE/data/<exp_dir>/... convention (cf. run_emul_euler.sh/run_diff_euler.sh).
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

DEST_DIR="$STORE_PATH/data/euler_multi_quadrants_openBC/data/train"
mkdir -p "$DEST_DIR"

python3 - "$DEST_DIR" << 'PYEOF'
import sys
from huggingface_hub import hf_hub_download

dest_dir = sys.argv[1]
# dest_dir ends in .../euler_multi_quadrants_openBC/data/train ; hf_hub_download
# rebuilds the data/train/... tree itself under local_dir, so go up 2 levels
# for local_dir = .../euler_multi_quadrants_openBC
local_dir = dest_dir.rsplit("/data/train", 1)[0]

for fname in [
    "euler_multi_quadrants_openBC_gamma_1.4_Dry_air_20_chunk_0.hdf5",
    "euler_multi_quadrants_openBC_gamma_1.4_Dry_air_20_chunk_40.hdf5",
]:
    path = hf_hub_download(
        repo_id="polymathic-ai/euler_multi_quadrants_openBC",
        repo_type="dataset",
        filename=f"data/train/{fname}",
        local_dir=local_dir,
    )
    print("Downloaded to", path)
PYEOF

echo "Done. Check:"
ls -la "$DEST_DIR"
