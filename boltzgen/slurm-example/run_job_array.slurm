#!/bin/bash
#SBATCH -C gpu                    # Use GPU nodes
#SBATCH -q regular                # Queue/partition
#SBATCH -t 03:00:00               # Walltime per array task
#SBATCH -N 1                      # Number of nodes per task
#SBATCH --gpus-per-node=1         # 1 GPU per task
#SBATCH --array=1-2               # Job array indices. Override this.
#SBATCH --mem=64G
#SBATCH -J boltzgen

set -euo pipefail
set -x

echo "Starting job on host: $(hostname)"
echo "SLURM job ID:        ${SLURM_JOB_ID:-unknown}"
echo "Array job ID:        ${SLURM_ARRAY_JOB_ID:-unknown}"
echo "Array task ID:       ${SLURM_ARRAY_TASK_ID:-unknown}"
echo "Nodes:               ${SLURM_JOB_NUM_NODES}"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <design_spec> <outdir> <num_designs_per_job> <conda_environment>" >&2
    exit 1
fi

design_spec="$1"
outdir="$2"
num_designs="${3}"
conda_environment="${4}"
shift 4                 # Remove parsed args
extra_args=("$@")       # Store everything else

echo "Design spec: $design_spec"
echo "Output dir:  $outdir"
echo "Num designs: $num_designs"
echo "Conda environment: $conda_environment"

module load conda
conda activate "$conda_environment"

which python
which boltzgen
nvidia-smi

job_outdir="${outdir}/task-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${job_outdir}"

srun --ntasks=1 --gpus-per-task=1 \
  boltzgen run "$design_spec" \
    --output "${job_outdir}" \
    --num_designs "$num_designs" \
    "${extra_args[@]}"


