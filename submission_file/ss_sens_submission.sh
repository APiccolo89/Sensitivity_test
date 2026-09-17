#!/bin/bash
declare group="" name_suite="" steady_state="" np=""
declare MODE=""

pwd=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Path is ${pwd}"

source "${pwd}"/parsing_argument.sh 

if [ $(($#%2)) != 0 ]; then
    die "The number of arguments i.e. (--x a --y b) is odd."  
fi
# Check 
while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
     
        if ! check_argument_bash "${1#--}"; then
            die "There is a problem: ${1} seems to not be a parameter"
        fi 

        if (( $# < 2 )) || [[ $2 == "--"* ]]; then
            die "Missing value for ${1}"
        fi 
        
        printf -v "${1#--}" "%s" "$2" # equivalent to say $name=$value Store 
        shift 2
    else 

        die "Well... unexpected arguments: ${1}"

    fi 
done

echo "group: ${group}, name_suite: ${name_suite}, steady_state: ${steady_state}, np: ${np}"

if [[ $steady_state == "1" ]]; then  
    MODE="SS"
elif [[ $steady_state == "0" ]]; then 
    MODE="TD"
fi


# Source - https://stackoverflow.com/a/246128
# Posted by dogbane, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-16, License - CC BY-SA 4.0

jid=$(sbatch --parsable << EOT
#!/bin/bash
#SBATCH --job-name=Sensitivity_no_shear_heating
#SBATCH --array=0-99%100
#SBATCH --ntasks=1
#SBATCH --exclude=node052
#SBATCH --cpus-per-task=${np}
#SBATCH --time=30:00:00
#SBATCH --mem=10G
#SBATCH --output=output_bash/out_${group}_${MODE}/${MODE}_%A_%a.txt
#SBATCH --error=output_bash/err_${group}_${MODE}/${MODE}_%A_%a.txt

echo $"id main db: ${group}"
echo $"number of processor: ${np}"

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

n_workers=100
n_cases=7000

echo "number of slaves: \${n_workers}" 
echo "number of cases: \${n_cases}"
export log_worker=output_bash/logs_worker_${group}
echo "\${log_worker}"
mkdir -p "\${log_worker}"


for case_index in \$(seq \${worker_id} \${n_workers} \$((n_cases - 1))); do
    export XDG_CACHE_HOME=/mnt/scratch/wlnw570/fenics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\${case_index}
    echo \${XDG_CACHE_HOME}
    mkdir -p "\$XDG_CACHE_HOME"

    mpirun -n ${np} "\$PYTHON" -u script_run_stoned.py \
        --case_index \${case_index} \
        --group ${group} \
        --name_suite ${name_suite}\
        --id_batch \$id_batch \
        --steady_state ${steady_state}\
        --name_csv cases_Sensitivity_steady_s_sensitivity.csv\

    rm -rf "\$XDG_CACHE_HOME"

    echo "case_index = \${case_index}" >> "\${log_worker}"/log_worker_${group}_\${worker_id}.txt


done
EOT
)
echo "job_id:${jid} name_suite:${name_suite}, group:${group}, mode:${MODE}" >> "$pwd"/log_bash_run.txt
