#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH -J kolmogorov_diff_concat
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

# Usage: sbatch --job-name=diffusion_concat_2re run_diff_concat.sh configs/config_command_diff_concat_2re.yaml
CONFIG=${1:-configs/config_command_diff_concat.yaml}

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

# Reuses Parameterized_Neural_Operator's kolmogorov_parametric dataset (Re
# 20-100, already generated) instead of thermalizer's own single-Re kolmogorov/ --
# no new data generation needed for the Re-conditioned corrector.
rsync -av $STORE/data/kolmogorov_parametric/ $SCRATCH/data/kolmogorov_parametric

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

echo "Config : $CONFIG"
python entrypoints/main_diffusion_concat.py --config "$CONFIG"
source deactivate
