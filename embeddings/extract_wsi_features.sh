#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --job-name=wsi_features
#SBATCH --output=wsi_features_%j.log

# Extract patch features from original WSI SVS files via TRIDENT.
# All outputs go to /scratch/st-singha53-1/bhelfert/CBL/output/features/{job_name}/
#
# Usage:
#   sbatch extract_wsi_features.sh conch_trichrome
#   sbatch extract_wsi_features.sh resnet_hne_40x
#   sbatch extract_wsi_features.sh gigapath_hne
#   sbatch extract_wsi_features.sh all

set -e
JOB="${1:?Usage: sbatch extract_wsi_features.sh <conch_trichrome|resnet_hne_40x|gigapath_hne|all>}"

cd $SLURM_SUBMIT_DIR
module load gcc
module load cuda
# Clear positional args before sourcing miniconda's activate — it forwards $@ to `conda activate`
set --
source /home/bhelfert/miniconda3/bin/activate
source activate trident_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_TOKEN=$(cat ~/.huggingface/token)
export http_proxy=http://hub.arc.ubc.ca:8888
export https_proxy=http://hub.arc.ubc.ca:8888
export no_proxy=.arc.ubc.ca,localhost
export TORCH_HOME="/scratch/st-singha53-1/bhelfert/cache/torch"
export HF_HOME="/scratch/st-singha53-1/bhelfert/cache/huggingface"
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl_cache"
mkdir -p "$MPLCONFIGDIR"

CODE_DIR="/arc/project/st-singha53-1/bhelfert/github_repos/TRIDENT"
HNE_WSI="/arc/project/st-singha53-1/datasets/ssc/hne_10_19_2023"
TRI_WSI="/arc/project/st-singha53-1/datasets/ssc/trichrome_5_31_2024"
FEAT_ROOT="/scratch/st-singha53-1/bhelfert/CBL/output/features"

cd "$CODE_DIR"

run_conch_trichrome() {
    echo "=== CONCH v1 from trichrome WSIs (20x, 512px) ==="
    python run_batch_of_slides.py \
        --task all \
        --wsi_dir "$TRI_WSI" \
        --job_dir "$FEAT_ROOT/conch_v1_trichrome" \
        --gpu 0 \
        --patch_encoder conch_v1 \
        --patch_size 512 \
        --mag 20 \
        --batch_size 128 \
        --max_workers 8 \
        --segmenter hest --seg_conf_thresh 0.4
    echo "--> conch_trichrome done."
}

run_resnet_trichrome() {
    echo "=== ResNet50 from trichrome WSIs (20x, 256px) ==="
    python run_batch_of_slides.py \
        --task all \
        --wsi_dir "$TRI_WSI" \
        --job_dir "$FEAT_ROOT/resnet50_trichrome_20x" \
        --gpu 0 \
        --patch_encoder resnet50 \
        --patch_size 256 \
        --mag 20 \
        --batch_size 256 \
        --max_workers 8 \
        --segmenter hest --seg_conf_thresh 0.4
    echo "--> resnet_trichrome done."
}

run_resnet_hne_40x() {
    echo "=== ResNet50 from HNE WSIs (40x native, 256px) ==="
    python run_batch_of_slides.py \
        --task all \
        --wsi_dir "$HNE_WSI" \
        --job_dir "$FEAT_ROOT/resnet50_hne_40x" \
        --gpu 0 \
        --patch_encoder resnet50 \
        --patch_size 256 \
        --mag 40 \
        --batch_size 256 \
        --max_workers 8
    echo "--> resnet_hne_40x done."
}

run_gigapath_hne() {
    echo "=== GigaPath from HNE WSIs (20x, 256px) ==="
    python run_batch_of_slides.py \
        --task all \
        --wsi_dir "$HNE_WSI" \
        --job_dir "$FEAT_ROOT/gigapath_hne" \
        --gpu 0 \
        --patch_encoder gigapath \
        --patch_size 256 \
        --mag 20 \
        --batch_size 128 \
        --max_workers 8
    echo "--> gigapath_hne done."
}

case "$JOB" in
    conch_trichrome)   run_conch_trichrome ;;
    resnet_trichrome)  run_resnet_trichrome ;;
    resnet_hne_40x)    run_resnet_hne_40x ;;
    gigapath_hne)      run_gigapath_hne ;;
    all)               run_conch_trichrome; run_resnet_trichrome; run_resnet_hne_40x; run_gigapath_hne ;;
    *)                 echo "Unknown job: $JOB"; exit 1 ;;
esac

echo "=== ALL DONE ==="
