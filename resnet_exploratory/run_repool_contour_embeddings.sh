#!/bin/bash
#SBATCH --job-name=repool_contour
#SBATCH --account=st-singha53-1
#SBATCH --partition=interactive_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=repool_contour_%j.log

# Re-pool flagged slides' embedding rows via TRIDENT-contour containment.
# Args forwarded to the python script (e.g. --dry-run).
ARGS=("$@"); set --
source /home/bhelfert/miniconda3/bin/activate
source activate /scratch/st-singha53-1/bhelfert/conda/envs/histolytics_env

cd /scratch/st-singha53-1/bhelfert/CBL/scripts/preprocessing
export OMP_NUM_THREADS=4
python repool_contour_embeddings.py "${ARGS[@]}"
