#!/bin/bash
#SBATCH --job-name=build_normed_mat
#SBATCH --account=st-singha53-1
#SBATCH --partition=interactive_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/st-singha53-1/bhelfert/CBL/output/feature_matrices/_build_%j.out
#SBATCH --error=/scratch/st-singha53-1/bhelfert/CBL/output/feature_matrices/_build_%j.err

# Build per-sample feature matrices from the cohort-median stain-normalized
# embeddings. Passthrough args go to build_normed_matrices.py, e.g.:
#   sbatch run_build_normed_matrices.sh                  # all four sets, L2-pool (_mnorm)
#   sbatch run_build_normed_matrices.sh --pool mean      # all four sets, mean-pool (_mmean)
#   sbatch run_build_normed_matrices.sh --sets uni_hne_mnorm --pool mean

module load gcc
# Scrub positional args so `source activate` does not misread passthrough flags.
ARGS=("$@")
set --
source /home/bhelfert/miniconda3/bin/activate
source activate /scratch/st-singha53-1/bhelfert/conda/envs/histolytics_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl_cache"; mkdir -p "$MPLCONFIGDIR"

SD="/scratch/st-singha53-1/bhelfert/CBL/scripts/preprocessing"
echo "=== build_normed_matrices ${ARGS[*]} ==="
echo "Node: $(hostname)  Start: $(date)"
python "$SD/build_normed_matrices.py" "${ARGS[@]}"
echo "Done: $(date)"
