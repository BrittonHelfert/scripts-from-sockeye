#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --job-name=repl_features
#SBATCH --output=repl_features_%j.log

set -e
cd $SLURM_SUBMIT_DIR
module load gcc
module load cuda
source ~/.bashrc
conda activate trident_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_TOKEN=$(cat ~/.huggingface/token)
export http_proxy=http://hub.arc.ubc.ca:8888
export https_proxy=http://hub.arc.ubc.ca:8888
export no_proxy=.arc.ubc.ca,localhost
export TORCH_HOME="/scratch/st-singha53-1/bhelfert/cache/torch"
export HF_HOME="/scratch/st-singha53-1/bhelfert/cache/huggingface"
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl_cache"
mkdir -p "$MPLCONFIGDIR"

cd /arc/project/st-singha53-1/bhelfert/github_repos/TRIDENT

echo "=== [1/3] Extracting UNI v1 features (1024-d, 256px patches) ==="
python run_batch_of_slides.py \
    --task all \
    --wsi_dir /arc/project/st-singha53-1/datasets/ssc/hne_10_19_2023 \
    --job_dir /scratch/st-singha53-1/bhelfert/CBL/output/replication/features/uni \
    --gpu 0 \
    --patch_encoder uni_v1 \
    --patch_size 256 \
    --mag 20 \
    --batch_size 128 \
    --max_workers 8

echo "=== [2/3] Extracting CONCH v1 features (512-d, 512px patches) ==="
python run_batch_of_slides.py \
    --task all \
    --wsi_dir /arc/project/st-singha53-1/datasets/ssc/hne_10_19_2023 \
    --job_dir /scratch/st-singha53-1/bhelfert/CBL/output/replication/features/conch \
    --gpu 0 \
    --patch_encoder conch_v1 \
    --patch_size 512 \
    --mag 20 \
    --batch_size 128 \
    --max_workers 8

echo "=== [3/3] Extracting ResNet50 features (2048-d, 256px patches) ==="
python run_batch_of_slides.py \
    --task all \
    --wsi_dir /arc/project/st-singha53-1/datasets/ssc/hne_10_19_2023 \
    --job_dir /scratch/st-singha53-1/bhelfert/CBL/output/replication/features/resnet \
    --gpu 0 \
    --patch_encoder resnet50 \
    --patch_size 256 \
    --mag 20 \
    --batch_size 128 \
    --max_workers 8

echo "=== All encoders done ==="
