#!/bin/bash --login
#SBATCH --job-name=kimik31-mi355x
#SBATCH --partition=gpu_rocm
#SBATCH --account=a_rcc
#SBATCH --qos=sdf
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=1800G
#SBATCH --time=48:00:00
#SBATCH --output=logs/kimik31-%j.out
#SBATCH --error=logs/kimik31-%j.out
#
# Batch wrapper around serve-kimik31.sh for Bunya's MI355X nodes (bun159/160/161).
# Submit from the repository root on /scratch:
#
#   mkdir -p logs && sbatch serve-kimik31.sbatch
#   AGENTS=16 sbatch --export=ALL serve-kimik31.sbatch     # pick a recipe tier
#
# partition/account/qos/gres are the K3 repo's discovered MI355X values; confirm
# yours with:  sinfo -o "%P %.10l %G %f" | grep mi355x
#
# --mem=1800G is host RAM. The throughput profiles pin HiCache host pools out of
# it (sized from the node's 1133 GiB GTT; see docs/configuration.md#hicache), so
# do not lower --mem below ~1200G with AGENTS>8.
#
# Once RUNNING:  grep -A 30 'is up and serving' logs/kimik31-<jobid>.out

set -euo pipefail
# Slurm runs a COPY of this script from its spool dir, so BASH_SOURCE is not the
# checkout. SLURM_SUBMIT_DIR is where 'sbatch' was run (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[[ -x ./serve-kimik31.sh ]] || { echo "Submit from the repository root (serve-kimik31.sh not found in $PWD)" >&2; exit 1; }
echo "[kimik31-sbatch] Job $SLURM_JOB_ID on $(hostname) at $(date)"
exec ./serve-kimik31.sh serve
