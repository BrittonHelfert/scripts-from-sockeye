#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --job-name=cnorm_feat
#SBATCH --output=/scratch/st-singha53-1/bhelfert/CBL/output/features/_cnorm_logs/feat_%A.log

# Stage 2: extract cohort-median-normed embeddings for ONE (encoder, stain).
#   sbatch run_cohort_normed_features.sh resnet50 hne
#   sbatch run_cohort_normed_features.sh uni_v1   hne
#   sbatch run_cohort_normed_features.sh resnet50 trichrome
#   sbatch run_cohort_normed_features.sh conch_v1 trichrome
set -e
ENCODER="${1:?usage: sbatch run_cohort_normed_features.sh <encoder> <stain>}"
STAIN="${2:?usage: sbatch run_cohort_normed_features.sh <encoder> <stain>}"
mkdir -p /scratch/st-singha53-1/bhelfert/CBL/output/features/_cnorm_logs
cd "$SLURM_SUBMIT_DIR"
module load gcc
module load cuda
set --
source /home/bhelfert/miniconda3/bin/activate
source activate /scratch/st-singha53-1/bhelfert/conda/envs/trident_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_TOKEN=$(cat ~/.huggingface/token 2>/dev/null || true)
export http_proxy=http://hub.arc.ubc.ca:8888
export https_proxy=http://hub.arc.ubc.ca:8888
export no_proxy=.arc.ubc.ca,localhost
export TORCH_HOME="/scratch/st-singha53-1/bhelfert/cache/torch"
export HF_HOME="/scratch/st-singha53-1/bhelfert/cache/huggingface"
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl_cache"; mkdir -p "$MPLCONFIGDIR"

SD="/scratch/st-singha53-1/bhelfert/CBL/scripts/preprocessing"
echo "Running: extract_cohort_normed_features.py --encoder $ENCODER --stain $STAIN"
python "$SD/extract_cohort_normed_features.py" \
    --encoder "$ENCODER" --stain "$STAIN" \
    --batch_size 128 --num_workers "$SLURM_CPUS_PER_TASK"
echo "=== DONE ==="
