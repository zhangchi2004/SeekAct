#!/usr/bin/env bash
#SBATCH --job-name=epic-long-camera
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/nfs_data/zhangchi/logs/lime-local/long-camera/estimate-%j.out
#SBATCH --error=/mnt/nfs_data/zhangchi/logs/lime-local/long-camera/estimate-%j.err
set -euo pipefail
LIME_REPO=/mnt/nfs_data/zhangchi/zhangchi/lime-local
LIME_PY=/mnt/nfs_data/zhangchi/miniconda3/envs/ego-transfer/bin/python
LIME_VIDEOS=/mnt/node04/zhangchi/data/epic-kitchens/demo-subset/EPIC-KITCHENS
export PYTHONPATH="$LIME_REPO/src"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/nfs_data/zhangchi/hf_cache
export TORCH_HOME=/mnt/nfs_data/zhangchi/torch_cache
test -x "$LIME_PY"
test -w /mnt/node04/zhangchi/data/lime-local
echo "job=$SLURM_JOB_ID host=$(hostname) python=$LIME_PY start=$(date -Is)"
nvidia-smi
srun "$LIME_PY" "$LIME_REPO/scripts/estimate_long_camera_da3.py" \
  --model /mnt/nfs_data/zhangchi/model/ego_transfer/depth_anything_3_small \
  --da3-repo /mnt/nfs_data/zhangchi/zhangchi/Depth-Anything-3 \
  --output /mnt/node04/zhangchi/data/lime-local/epic-long-camera-da3-v1 \
  --video "$LIME_VIDEOS/P01/videos/P01_103.MP4" \
  --video "$LIME_VIDEOS/P03/videos/P03_118.MP4" \
  --video "$LIME_VIDEOS/P04/videos/P04_114.MP4" \
  --video "$LIME_VIDEOS/P09/videos/P09_106.MP4"
