#!/bin/bash

USAGE="usage: myscript type, with type 0-5, where:\n 0 - no shear, k=3.1;\n 1 - no shear, k=5.0;\n 2 - no shear, non-linear;\n 3 - yes shear, k=3.1;\n 4 - yes shear, k=5.0;\n 5 - yes shear, non-linear"
USAGE2="Type must be an integer between 0-5"
Type=${1:?$(printf '%b' "$USAGE")}

if [ "$Type" -gt 5 ] ||  [ "$Type" -lt 0 ]; then
    printf '%b\n' "$USAGE2" >&2
    exit 1
fi 

# Source - https://stackoverflow.com/a/246128
# Posted by dogbane, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-16, License - CC BY-SA 4.0

pwd=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Path is ${pwd}"


echo $"id main db: ${Type}" 
bash "${pwd}/remove_file_output.sh" "${Type}"


sbatch << EOT
#!/bin/bash
#SBATCH --job-name=Sensitivity_no_shear_heating
#SBATCH --array=0-99%100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=21:00:00
#SBATCH --mem=10G
#SBATCH --output=output_bash/out_${Type}/SS_%A_%a.txt
#SBATCH --error=output_bash/err_${Type}/SS_%A_%a.txt

echo $"id main db: ${Type}"

module purge
module load python/3.13.0
module load spack/0.23

unset LD_PRELOAD
export LD_LIBRARY_PATH="/mnt/scratch/wlnw570/libs/glu:\$LD_LIBRARY_PATH"

spack load py-fenics-dolfinx
spack load py-h5py
spack load py-numpy
spack load py-petsc4py@3.22.1

source /users/wlnw570/stoned_fenicsx_env/bin/activate
PYTHON=/users/wlnw570/stoned_fenicsx_env/bin/python

worker_id=\${SLURM_ARRAY_TASK_ID}
id_batch=\${worker_id}

echo "Worker ID = \${worker_id}"
echo "HDF5 batch ID = \${id_batch}"
if [ "${Type}" -eq 0 ] || [ "${Type}" -eq 3 ]; then 
    k=3.1
elif [ "${Type}" -eq 1 ] || [ "${Type}" -eq 4 ]; then
    k=5.0
elif [ "${Type}" -eq 2 ] || [ "${Type}" -eq 5 ]; then 
    k=-1.0
fi

if [ "${Type}" -gt 2 ]; then
    shear_heating=True
else
    shear_heating=False
fi

echo " Conductivity is \${k}, shear heating is \${shear_heating}" 
echo " If k is less than 0, the models are no-linear"

for case_index in \$(seq \${worker_id} 100 99); do
    export XDG_CACHE_HOME=/mnt/scratch/wlnw570/fenics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\${case_index}
    mkdir -p "\$XDG_CACHE_HOME"

    mpirun -n 1 "\$PYTHON" -u Sensitivity_tests.py \
        --case_index \${case_index} \
        --k \${k} \
        --id_db ${Type} \
        --id_batch \$id_batch \
    	--option_shear_heating \${shear_heating} \
	    --option_viscosity_weak_zone Dislocation_Law \


    rm -rf "\$XDG_CACHE_HOME"
done
EOT
