#!/bin/bash
#SBATCH --account=st-singha53-1
#SBATCH --partition=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --job-name=cohort_ref
#SBATCH --output=/scratch/st-singha53-1/bhelfert/CBL/output/stain_references/logs/ref_%x_%A.log

# Stage 1: cohort-median stain reference for ONE stain (CPU).
#   sbatch run_cohort_reference.sh hne
#   sbatch run_cohort_reference.sh trichrome
set -e
STAIN="${1:?usage: sbatch run_cohort_reference.sh <hne|trichrome>}"
mkdir -p /scratch/st-singha53-1/bhelfert/CBL/output/stain_references/logs
cd "$SLURM_SUBMIT_DIR"
module load gcc
set --
source /home/bhelfert/miniconda3/bin/activate
source activate /scratch/st-singha53-1/bhelfert/conda/envs/trident_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl_cache"; mkdir -p "$MPLCONFIGDIR"

SD="/scratch/st-singha53-1/bhelfert/CBL/scripts/preprocessing"
echo "=== $STAIN reference ==="
python "$SD/compute_cohort_stain_reference.py" --stain "$STAIN"
echo "=== DONE ==="
