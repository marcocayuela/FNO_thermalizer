#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -J shear_flow_Re5e4_Sc1e0_native
#SBATCH -o /scratch/cayuelam/logs/shear_flow/%x_%j.out

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

rsync -av $STORE/data/shear_flow/ $SCRATCH/data/shear_flow

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
# Safety net against allocator fragmentation on top of the batch_size cut in
# config_command_emul_shear_flow_native.yaml (cf. the CUDA OOM this run hit).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python entrypoints/main_emul_shear_flow_native.py
source deactivate
