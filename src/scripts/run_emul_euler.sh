#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH -J euler_gamma1.4_emul
#SBATCH -o /scratch/cayuelam/logs/euler/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

# Train chunks (~42 GB, gamma=1.4 "Dry_air", chunk_0+chunk_40) are fetched
# once from HF onto $STORE via fetch_train_chunks_euler_mesu.sh -- rsync'd
# here from $STORE to $SCRATCH like the valid split; idempotent, so once
# already staged on $SCRATCH this is a fast no-op checksum pass, not a
# 42 GB re-transfer on every job.
rsync -av $STORE/data/euler_multi_quadrants_openBC/data/train/ $SCRATCH/data/euler_multi_quadrants_openBC/data/train/
rsync -av $STORE/data/euler_multi_quadrants_openBC/valid/ $SCRATCH/data/euler_multi_quadrants_openBC/valid/

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python entrypoints/main_emul_euler.py
source deactivate
