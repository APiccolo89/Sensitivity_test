# Sensitivity test
---
This routine has been used to produce the sensitivity analysis of (name of the manuscript). Everything is set up to reproduce these experiments; as such, the scripts have several hardcoded values. However, the workflow that is described here can be translated to use the code for other purposes.

The experiments can be divided into 6 groups, and per each group there is a number of single case experiment. So, there are n single experiment cases that differs only for three or two parameters, that are run with different common values as a function of their group. 


|**group** | **shear heating** | **k** | 
|------|-----|---|
|*k3nsh* | no  | 3.1|  
|*k5nsh* | no  | 5.0|  
|*nlnsh*  | no  | non-linear|
|*k3sh*| yes | 3.1|
|*k5sh*   | yes  | 5.0|  
|*nlsh*  | yes  | non-linear|



---

## Workflow

---
The workflow is the following:
1. Define the single experiment cases using one of the two scripts.
2. Run the experiments using a bash submission script that requires an array job of at least 100 processors (array jobs and each job only one processor).
3. Post-process the output with one of the scripts.
3.1 Merge the h5 files into one file (one per group).
3.2 Post-process the data using the pandas routine.
3.3 Generate the pictures.

Step 1 generates a csv table with the single experiment cases.
Step 2 runs these cases for the six groups of numerical experiments designed for the current project. Step 2 produces n h5 files with a reduced database storing the relevant output.
Step 3 post-processes, analyses and generates figures and the databases.

---
## Generation of the single experiment cases
---
In the main manuscript, the sensitivity study is divided into steady-state runs, which cover a wide range of conditions, and time-dependent scenarios, which cover a limited set of parameters.

There are two python scripts that generate the case table:
```
generate_case_sensitivity_time_dependent.py
generate_case_sensitivity_steady_state.py
```
### Steady-state sensitivity
Steady-state sensitivity cases are run such that 7000 random tests can be generated and run on the fly. Steady-state runs cover:

```python
    vc_arr:list[float]=[3.0, 15.0] # convergence velocity
    Tp_arr:list[float]=[1350.0, 1425.0] # potential temperature in celsius degree
    age_arr:list[float]=[30.0,100.0] # age in Myr
```

The generation of the single experiment cases is carried out with the ``scipy.stats.qmc.LatinHypercube`` library: 

```python 
    sampler = scipy.stats.qmc.LatinHypercube(d=3)
    sample = sampler.random(n=n) # n = number of cases (7000)
    scaled_sample = scipy.stats.qmc.scale(sample, lbounds, hbounds)
```

This generates a random ensemble of data. The resulting array of experiments is then saved as a csv table that will be read by the python scripts that run the numerical experiments.

### Time-dependent sensitivity

Time-dependent runs cover 100 cases (the potential temperature is dropped as a main parameter), and they are repeated for their steady-state counterparts. These tests are run for 30 Myr.

In the time-dependent cases, there is no need to run thousands of cases, as they are computationally expensive. The generation of the cases is simply a linear combination of two arrays: convergence velocity and age.

```python
    vc = np.linspace(vc_arr[0],vc_arr[1],num=10) # generate array with vc
    age = np.linspace(age_arr[0],age_arr[1],num=10) # generate array with age

    array_exp = np.zeros([100,2],dtype=float) # create the array

    vc_, age_ = np.meshgrid(vc,age)
    a = vc_.flatten()
    b = age_.flatten()
    len_array = len(age)*len(vc)
    # create the array 
    for i in range(len_array):
        array_exp[i,0] = b[i]
        array_exp[i,1] = a[i]

```
---
## Bash script
The bash scripts are stored in `submission_file`. To run:
bash submission_file/run_script.sh --case k3nsh --np 1 --steady_state 1 --name_suite steady_state_ss

- **case**: indicates the category of run:
- **np**: the number of processors
- **steady_state**: 1: steady-state simulation; 0: time-dependent
- **name_suite**: name of the batch of experiments that the user wants to run.

The bash script parses the arguments and correctly instructs the python configuration script.
There are two bash scripts: one for the steady-state sensitivity study and one for the time-dependent sensitivity study.
```bash
#!/bin/bash
#from https://keestalkstech.com/named-arguments-in-a-bash-script/ 
declare case="" name_suite="" steady_state="" np=""
declare -ar required_argument=(case name_suite steady_state np)
declare cases_sim=(k3nsh k5nsh nlnsh k3sh k5sh nlsh)
# Error
die() {
    # error message
    printf "Script failed: %s\n" "$1" >&2
    exit 1
}
# Parse the argument of the bash script
check_argument_bash(){
    # Loop over the allowed declared required arguments. Check if the 
    # value is within the set, otherwise, throw an error. 
    
    
    local parameter=$1
    local is_par 

    for is_par in "${required_argument[@]}"; do
        if [[ $parameter == "$is_par" ]]; then
           declare -p -- "$parameter" &>/dev/null
            return 
        fi
    done

    return 1
}
# Throw an error in case the wrong group is given 
check_argument_case(){
    # Loop over the allowed declared required arguments. Check if the 
    # value is within the set, otherwise, throw an error. 
    
    
    local case=$1
    local is_case 
    

    for is_case in "${cases_sim[@]}"; do
        if [[ $case == "$is_case" ]]; then
            return
        fi
    done
    die "${case} is not a valid case. valid case are: k3nsh k5nsh nlnsh k3sh k5sh nlsh"
}

check_steady_state(){
    # Loop over the allowed declared required arguments. Check if the 
    # value is within the set, otherwise, throw an error. 
    
    
    local ss=$1
 
    if [[ $ss == "1" ]]; then  
        printf "Simulations are steady state" 
        return
    elif [[ $ss == "0" ]]; then 
        printf "Simulations are time dependent" 
        return
    fi
    
    die "${ss} is not a valid steady_state flag: steady_state flag must be 1 or 0"
}

# Translation for mere mortals: 
# $# -> total arguments -> check $1-$2 -> then remove the first 2 and do the same operation till $# is zero 
# check the total number of arguments and kill immediately if it is odd
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

echo "case: ${case}, name_suite: ${name_suite}, steady_state: ${steady_state}, np: ${np}"

# Source - https://stackoverflow.com/a/246128
# Posted by dogbane, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-16, License - CC BY-SA 4.0

pwd=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Path is ${pwd}"


echo $"id main db: ${case}" 
bash "${pwd}/remove_file_output.sh" "${case}"


sbatch << EOT
#!/bin/bash
#SBATCH --job-name=Sensitivity_no_shear_heating
#SBATCH --array=0-99%100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${np}
#SBATCH --time=30:00:00
#SBATCH --mem=10G
#SBATCH --output=output_bash/out_${case}/SS_%A_%a.txt
#SBATCH --error=output_bash/err_${case}/SS_%A_%a.txt

#Print the case (group)
echo $"id main db: ${case}"
# Print the number of processor
echo $"number of processor: ${np}"

# Remove module
module purge
# Load module
module load python/3.13.0
module load spack/0.23

unset LD_PRELOAD
export LD_LIBRARY_PATH="/mnt/scratch/wlnw570/libs/glu:\$LD_LIBRARY_PATH"
# Load the module for spack 
spack load py-fenics-dolfinx
spack load py-h5py
spack load py-numpy
spack load py-petsc4py@3.22.1
source /users/wlnw570/stoned_fenicsx_env/bin/activate
PYTHON=/users/wlnw570/stoned_fenicsx_env/bin/python

worker_id=\${SLURM_ARRAY_TASK_ID}
id_batch=\${worker_id}
#id_batch == worker number
echo "Worker ID = \${worker_id}"
echo "HDF5 batch ID = \${id_batch}"
# number of worker (i.e. number of processor)
n_workers=100
# number of the case: for obviously using this number must be equal to the number of experiment
n_cases=100

# Loop over the case e.g. from 0, jump every 100 till the maximum allowed -> This is more valid for the steady state case
for case_index in \$(seq \${worker_id} \${n_workers} \$((n_cases - 1))); do
    # -> Problem with the caching information of FEniCSx 
    export XDG_CACHE_HOME=/mnt/scratch/wlnw570/fenics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\${case_index}
    mkdir -p "\$XDG_CACHE_HOME"

    # Run the script with argument
    mpirun -n ${np} "\$PYTHON" -u Sensitivity_tests_time_dep_exp.py \
        --case_index \${case_index} \ # Case index
        --case ${case} \ # Group
        --name_suite ${name_suite}\ #Name of suite
        --id_batch \$id_batch \ #worker number
        --steady_state ${steady_state}\# steady state flag


    rm -rf "\$XDG_CACHE_HOME"
done
EOT

```
:warning: **Warning:** The script has been designed to comply the specification of Aire [Aire HPC](https://arc.leeds.ac.uk/knowledge-centre/resources/aire/). It was necessary to use spack, instead of miniforge. At the moment, it should have been corrected, however the bash script has been kept in its original form. If an user, reviewer, unicorn wants to use it, they should use in accordance of their HPC. 

---

## Run experiment script 

```python
def configure_run_save():
    # parse the arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_index", type=int)
    parser.add_argument("--case", type=str)
    parser.add_argument("--id_batch", type=int)
    parser.add_argument("--name_suite",type=str)
    parser.add_argument("--steady_state",type=int,default=1)

    args_cli = parser.parse_args()
    
    # save the name of the suite
    name_suite = args_cli.name_suite
    # Post Process the arg_cli and release the effective namedspace
    arguments = parse_the_argument(args_cli,pathtotable=f"{os.
    # Find the actual csv file storing case
    path.dirname(os.path.realpath(__file__))}/cases_Sensitivity_td.csv") 
    # Perform the test    
    pathtest,_ = perform_test(args=arguments,name_suite= name_suite)
    
    print(f"Test completed for case {arguments.case_index}. Results saved in {pathtest}")
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Read the output of the test and saves the data 
    if rank == 0:
        print(f"Rank {rank} is saving the test data for case {arguments.case_index}")
        if arguments.steady_state == 0: 
            save_test_data_td_alt(name_suite=name_suite,pathtotest=pathtest,arguments=arguments)
        else: 
            save_test_data(name_suite=name_suite,pathtotest=pathtest,arguments=arguments)

    
    #remove the tests folder and saves only the case_0 
    if arguments.case_index !=0:
        shutil.rmtree(os.path.join(pathtest))    
        print(f"Removed {os.path.join(pathtest)} for case {arguments.case_index}")
    
    print(f"Finished case {arguments.case_index}")
#---------------------------------------------------------------------------------    
```

The script to run the experiments accepts several arguments that help to configure the simulation. The script has a top-level function that first create the namedspace structure that configure the simulation, then run the simulation, and then save the relevant data. 

### Parsing arguments








--- 
## Post-processing script


still working on it

