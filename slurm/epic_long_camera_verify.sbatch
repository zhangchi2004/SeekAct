#!/usr/bin/env bash
#SBATCH --job-name=epic-long-verify
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=master
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/nfs_data/zhangchi/logs/lime-local/long-camera/verify-%j.out
#SBATCH --error=/mnt/nfs_data/zhangchi/logs/lime-local/long-camera/verify-%j.err
set -euo pipefail
export PYTHONPATH=/mnt/nfs_data/zhangchi/zhangchi/lime-local/src
export PYTHONUNBUFFERED=1
srun /mnt/nfs_data/zhangchi/miniconda3/envs/ego-transfer/bin/python \
  /mnt/nfs_data/zhangchi/zhangchi/lime-local/scripts/verify_long_camera_demos.py \
  --root /mnt/node04/zhangchi/data/lime-local
