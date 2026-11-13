#!/usr/bin/env bash
#SBATCH --job-name=epic-long-segments
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=master
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/nfs_data/zhangchi/logs/lime-local/long-camera/segments-%j.out
#SBATCH --error=/mnt/nfs_data/zhangchi/logs/lime-local/long-camera/segments-%j.err
set -euo pipefail
LIME_REPO=/mnt/nfs_data/zhangchi/zhangchi/lime-local
LIME_PY=/mnt/nfs_data/zhangchi/miniconda3/envs/ego-transfer/bin/python
LIME_BACKEND=${1:-epic_fields}
LIME_POSES=/mnt/node04/zhangchi/data/epic-kitchens/EPIC-Fields/selected-json
if [[ "$LIME_BACKEND" == da3 ]]; then
  LIME_POSES=/mnt/node04/zhangchi/data/lime-local/epic-long-camera-da3-v1
fi
export PYTHONPATH="$LIME_REPO/src"
export PYTHONUNBUFFERED=1
test -x "$LIME_PY"
test -w /mnt/node04/zhangchi/data/lime-local
echo "job=$SLURM_JOB_ID host=$(hostname) python=$LIME_PY start=$(date -Is) backend=$LIME_BACKEND"
srun "$LIME_PY" "$LIME_REPO/scripts/build_long_camera_demos.py" \
  --backend "$LIME_BACKEND" --poses-root "$LIME_POSES" \
  --video-root /mnt/node04/zhangchi/data/epic-kitchens/demo-subset/EPIC-KITCHENS \
  --output "/mnt/node04/zhangchi/data/lime-local/demo-epic-long-camera-$LIME_BACKEND-v1" \
  --count 20 --per-video 10
