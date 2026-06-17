#!/bin/bash
#SBATCH --job-name=SensitivityTests
#SBATCH --array=0-99%100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=34:00:00
#SBATCH --mem=10G
#SBATCH --output=out3/Sensitivity_%A_%a.txt
#SBATCH --error=err3/Sensitivity_%A_%a.txt

module purge
module load python/3.13.0
module load spack/0.23

unset LD_PRELOAD
export LD_LIBRARY_PATH="/mnt/scratch/wlnw570/libs/glu:$LD_LIBRARY_PATH"

spack load py-fenics-dolfinx
spack load py-h5py
spack load py-numpy
spack load py-petsc4py@3.22.1

source /users/wlnw570/stoned_fenicsx_env/bin/activate
PYTHON=/users/wlnw570/stoned_fenicsx_env/bin/python

worker_id=${SLURM_ARRAY_TASK_ID}
id_batch=${worker_id}

echo "Worker ID = ${worker_id}"
echo "HDF5 batch ID = ${id_batch}"

for case_index in $(seq ${worker_id} 100 6999); do
    export XDG_CACHE_HOME=/mnt/scratch/wlnw570/fenics_cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${case_index}
    mkdir -p "$XDG_CACHE_HOME"

    mpirun -n 1 "$PYTHON" -u Sensitivity_tests.py \
        --case_index ${case_index} \
        --k 3.1 \
        --id_db 3 \
        --id_batch ${id_batch} \
    	--option_shear_heating False

    rm -rf "$XDG_CACHE_HOME"
done
