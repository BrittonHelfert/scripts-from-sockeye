#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --job-name=norm_feat
#SBATCH --output=norm_feat_%A_%a.log

# Extract features with Vahadane stain normalization.
#
# Usage:
#   # Single slide (for testing)
#   sbatch run_extract_normalized_features.sh uni_v1 hne 261085
#
#   # All H&E slides as array job
#   sbatch --array=0-17 run_extract_normalized_features.sh uni_v1 hne
#
#   # CONCH on trichrome
#   sbatch --array=0-17 run_extract_normalized_features.sh conch_v1 trichrome

set -e

ENCODER="${1:?Usage: sbatch run_extract_normalized_features.sh <encoder> <stain> [slide_name]}"
STAIN="${2:?Usage: sbatch run_extract_normalized_features.sh <encoder> <stain> [slide_name]}"
SLIDE_NAME="${3:-}"

cd "$SLURM_SUBMIT_DIR"
module load gcc
module load cuda
# Clear positional args before sourcing miniconda's activate — it forwards $@ to `conda activate`
set --
source /home/bhelfert/miniconda3/bin/activate
source activate trident_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_TOKEN=$(cat ~/.huggingface/token 2>/dev/null || true)
export http_proxy=http://hub.arc.ubc.ca:8888
export https_proxy=http://hub.arc.ubc.ca:8888
export no_proxy=.arc.ubc.ca,localhost
export TORCH_HOME="/scratch/st-singha53-1/bhelfert/cache/torch"
export HF_HOME="/scratch/st-singha53-1/bhelfert/cache/huggingface"
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl_cache"
mkdir -p "$MPLCONFIGDIR"

SCRIPT_DIR="/scratch/st-singha53-1/bhelfert/CBL/scripts/preprocessing"

ARGS="--encoder $ENCODER --stain $STAIN --batch_size 128 --num_workers $SLURM_CPUS_PER_TASK"

if [ -n "$SLIDE_NAME" ]; then
    ARGS="$ARGS --slide_name $SLIDE_NAME"
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ARGS="$ARGS --array_index $SLURM_ARRAY_TASK_ID"
fi

echo "Running: python extract_normalized_features.py $ARGS"
python "$SCRIPT_DIR/extract_normalized_features.py" $ARGS

echo "=== DONE ==="
