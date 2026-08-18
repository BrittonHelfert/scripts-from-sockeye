#!/bin/bash
#SBATCH --account=st-singha53-1
#SBATCH --partition=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=he_interp
#SBATCH --array=0-17
#SBATCH --output=/scratch/st-singha53-1/bhelfert/CBL/output/interp_v2/hne/logs/he_%A_%a.log

# H&E interpretable features, one slide per array task (raw / no stain norm).
# Sample assignment uses TRIDENT contours (corrected polygon-cluster + debris drop).
#   sbatch run_he_interp.sh                 # all 18 slides
#   sbatch --array=0 run_he_interp.sh       # single slide (test)
set -e
mkdir -p /scratch/st-singha53-1/bhelfert/CBL/output/interp_v2/hne/logs
cd "$SLURM_SUBMIT_DIR"
module load gcc
set --
source /home/bhelfert/miniconda3/bin/activate
source activate /scratch/st-singha53-1/bhelfert/conda/envs/trident_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl_cache"; mkdir -p "$MPLCONFIGDIR"

echo "Task $SLURM_ARRAY_TASK_ID on $(hostname)  $(date)"
python /scratch/st-singha53-1/bhelfert/CBL/scripts/biomni_scripts/extract_he_interp_features.py \
    --array-index "$SLURM_ARRAY_TASK_ID"
echo "=== DONE $(date) ==="
