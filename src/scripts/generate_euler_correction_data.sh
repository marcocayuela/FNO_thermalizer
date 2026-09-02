#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=04:00:00
#SBATCH -J euler_gen_correction_data
#SBATCH -o /scratch/cayuelam/logs/euler/%x_%j.out

# Runs the frozen fno_euler_gamma1.4_delta emulator on both the --role train
# and --role val trajectory pools (cf. training/generate_euler_correction_data.py's
# own docstring), producing the two files
# config_command_residual_corrector_euler.yaml points at. Pure inference
# (no backprop) -- the 4h budget is generous, this should be much faster.

module purge
module load python/3.11

source activate fto
pip install -r requirements.txt

mkdir -p $SCRATCH/data/euler_multi_quadrants_openBC/data/train $SCRATCH/data/euler_multi_quadrants_openBC/data/valid
rsync -av $STORE/data/euler_multi_quadrants_openBC/data/train/ $SCRATCH/data/euler_multi_quadrants_openBC/data/train/
rsync -av $STORE/data/euler_multi_quadrants_openBC/data/valid/ $SCRATCH/data/euler_multi_quadrants_openBC/data/valid/

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

N_STEPS=${1:-300}
N_TRAJ_PER_FILE=${2:-10}

python training/generate_euler_correction_data.py \
    --data_dir "$DATA_DIR" \
    --emulator_run "$LOG_DIR/euler_multi_quadrants_openBC/fno_euler_gamma1.4_delta" \
    --role train --n_traj_per_file "$N_TRAJ_PER_FILE" --n_steps "$N_STEPS" \
    --out_file "$DATA_DIR/euler_multi_quadrants_openBC/correction_data/train_rollouts.hdf5"

python training/generate_euler_correction_data.py \
    --data_dir "$DATA_DIR" \
    --emulator_run "$LOG_DIR/euler_multi_quadrants_openBC/fno_euler_gamma1.4_delta" \
    --role val --n_steps "$N_STEPS" \
    --out_file "$DATA_DIR/euler_multi_quadrants_openBC/correction_data/val_rollouts.hdf5"

echo "Copy back to \$STORE recommended once done (these files are needed by run_residual_corrector_euler.sh"
echo "on every future job, same reasoning as the raw Euler data itself):"
echo "  rsync -av $DATA_DIR/euler_multi_quadrants_openBC/correction_data/ \$STORE/data/euler_multi_quadrants_openBC/correction_data/"

source deactivate
