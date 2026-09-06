# Sensitivity test

<p align="center">
  <img src="Result_shear_heating_sh/Figure_3.png" alt="Slab 2D plot" width="600">
  <br>
  <em>Figure 0 — Average difference of slab temperature. [a]: Difference between non-linear test group and the group of experiment with k=3.1. [b]: Difference between non-linear test group and the group of experiment with k=5.0.</em>
</p>



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

The name of the group, in practice, represents the combination of thermal model (`k3`, `k5`, `nl`) and a flag that indicates wheter or not the shear heating is activated. The post-processing routines generate plots as a function of the **shear heating** flag (`sh` or `nsh`). This guarantee that the comparison are made only for tests sharing the same physics. 

:warning: **Warning:** : **StonedFEniCSx** is still developing, the core functionality, and the relative numeric part will be remain the same, it may be the case that a few details on the input/output can change. In the zenodo repositary, the user will find the version of the code that have been used for this work. 

The routine presented are: 
```
Sensitivity_Test/
    ├── figure_additional/
        ├── slab_surface.py
        ├── plot_material_property.py
    ├── post_process_module/
        ├── data_extractor.py
        ├── figure_plot.py
        ├── global_var.py 
        ├── post_process_data.py
        ├── post_process_data_visualization.py
    ├── generation_cases_script/
        ├── generate_cases_Mexico.py
        ├── generate_cases_Sensitivity.py
        ├── generate_cases_time_dependent.py
    ├── real_configuration_plate/ 
        ├── geometric_input_real_plate.yml
    ├── script_run_stoned.py
    ├── sensitivity_steady_state_figure_plot.py

```

---

## Workflow

---
The step 0 is to install **StonedFEniCSx**. The user should follow the instruction of the package and read how the package works ['https://github.com/APiccolo89/StonedFenicsx'](StonedFEniCSx) 


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

In the following readme.md, I try to spell out the workflow in a such way that the main insights can be exploitable by other researcher to use both StonedFEniCSx, and also for reproducing the results, and make me write a long errata in the manuscript that I am writing. Feel free to crunch my self-esteem and improve it.  

---
## Quick start: 

The user needs to install **StonedFEniCSx** (follow the guide here:['https://apiccolo89.github.io/StonedFenicsx/installation.html'](InstallationSTONED)).

After the installation, the user needs to copy the repository: 

```bash
    git clone https://github.com/APiccolo89/Sensitivity_test.git
```
Then, for running the suite of experiments: 
```bash 
    python generate_case_steady_state.py
```
or 
```bash 
    python generate_case_steady_state_td.py
```
These commands will produce either `cases_Sensitivity_td.csv` or `cases_Sensitivity.csv`

### General 

To run the sensitivity analysis, the user needs to use the bash script in `submission_file/steady_state/steady_state_sensitivity.sh`

It must run from the main folder of the repository: 
```bash
    bash /submission_file/steady_state/steady_state_sensitivity.sh --group k3nsh --np 1 --name_suite Sensitivity --steady_state 1
```
The command above will run the number of experiment, using the data associated with the group `k3nsh` with 1 processor and with the steady state mode of **StonedFEniCSx**. The data will be saved using the `name_suite`.
After a run of a single group of experiments the output layout will be: 

```
    Sensitivity_tests/
      ├── h5_outputs/
        ├──{`name_suite`}_{`group`}_h5_files/
            ├──Data_Base_{id_batch}.h5
      ├── test_folder/
         ├──{`name_suite`}_{`group`}_{`flagSS`}
            ├──{`name_suite`}_case{case_index}_{`group`}
      ├── output_bash/
            ├──out_{'group'}_{'flagSS'}
```    

where `case_index` is the actual case number (i.e., row of the .csv table), `flagSS` is a flag that identify the mode of the code (`SS`=Steady State, `TD`=Time Dependent). `id_batch` is the processor responsible for the portion of the test. 


In `h5_output`, the reduced database of the results is stored. Each sub-folder represent each group. Inside the ``{`name_suite`}_{`group`}_h5_files`` there are small h5_files that contains the relevant information for that group of simulation. 

In `test_folder` there are the real output of **StonedFEniCSx** divided into group and suite folder. The output of the single test are removed. However, the user can modify the python script to keep all the output. 

> **Important:**  **StonedFEniCSx** requires always an input.yml file. 

:warning: **Warning:** : If the user wants to reproduce the results of the manuscript, he should not modify input_tests.yml. All the main required modification are done by the configuration python scripts. 

### Post processing script

After all the run are performed, the user can use the post-processing script by indicating the name of the suite and two tags:

```bash

python sensitivity_steady_state_figure_plot.py --name_suite Sensitivity --type sh

```
`type` is signalling if the analysis is for the tests featuring shear heating or not. The `Test_plot.py` merges the `.h5` databases in one database and produces three picture (the one of the manuscript):

<p align="center">
  <img src="Result_shear_heating_sh/Figure_0.png" alt="Slab surface temperature, case" width="600">
  <br>
  <em>Figure 1 — Slab surface temperature and key isotherm data along the slab surface. [a]: Non linear cases P-T condition along slab surface (blue dots high pressure and low temperature rocks database). [b]: Difference of temperature between non-linear case and linear case with k=3.1. [c]: Seismogenetic length in all the experiment.[d]: Error measure between linear cases and non-linear cases. [d]: Depth of the characteristic isotherm .</em>
</p>



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
    declare group="" name_suite="" steady_state="" np=""
    declare MODE=""
    declare -ar required_argument=(group name_suite steady_state np)
    declare cases_sim=(k3nsh k5nsh nlnsh k3sh k5sh nlsh)

    die() {
        # error message
        printf "Script failed: %s\n" "$1" >&2
        exit 1
    }

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

    check_argument_case(){
        # Loop over the allowed declared required arguments. Check if the 
        # value is within the set, otherwise, throw an error. 


        local group=$1
        local is_case 


        for is_case in "${cases_sim[@]}"; do
            if [[ $group == "$is_case" ]]; then
                return
            fi
        done
        die "${group} is not a valid group. valid group are: k3nsh k5nsh nlnsh k3sh k5sh nlsh"
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

    echo "group: ${group}, name_suite: ${name_suite}, steady_state: ${steady_state}, np: ${np}"

    if [[ $steady_state == "1" ]]; then  
        MODE="SS"
    elif [[ $steady_state == "0" ]]; then 
        MODE="TD"
    fi


    # Source - https://stackoverflow.com/a/246128
    # Posted by dogbane, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-08-16, License - CC BY-SA 4.0

    pwd=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    echo "Path is ${pwd}"


    echo $"id main db: ${group}" 
    bash "${pwd}/remove_file_output.sh" "${group}_${MODE}"


    sbatch << EOT
    #!/bin/bash
    #SBATCH --job-name=Sensitivity_no_shear_heating
    #SBATCH --array=0-99%100
    #SBATCH --ntasks=1
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


    for case_index in \$(seq \${worker_id} \${n_workers} \$((n_cases - 1))); do
        export XDG_CACHE_HOME=/mnt/scratch/wlnw570/fenics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\${case_index}
        mkdir -p "\$XDG_CACHE_HOME"

        mpirun -n ${np} "\$PYTHON" -u script_run_stoned.py \
            --case_index \${case_index} \
            --group ${group} \
            --name_suite ${name_suite}\
            --id_batch \$id_batch \
            --steady_state ${steady_state}\
            --name_csv cases_Sensitivity.csv\


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
Parse the argument parse the arguments, and create a new namedspace which contains updated information and material properties. 

```python
def parse_the_argument(args:argparse.Namespace,pathtotable:str)->SimpleNamespace:
    """_summary_

    Args:
        args (argparse.Namespace): argument parser
        pathtotable (str): table with the cases to run


    Returns:
        SimpleNamespace: updated argument namespace with the group parameters and material properties
    """


    # Extract group. Create a dictionary of cases
    dict_case = create_case_dictionary()
    #Special case in case debug is needed
    if args.group == 'debug':
        case_index = 0 
        case_sim = dict_case['debug']
        mat_prop = choose_material_properties(case_sim['k'])
        steady_state = 1 
        name_suite = 'debug' 
        shear_heating = case_sim['shear_heating']
    else:
        # Check for the 10th time if all the required arguments are there
        required = ["case_index", "group", "id_batch"]
        missing = [name for name in required if getattr(args, name) is None]
        # Punish the violation
        if missing:
            raise ValueError(f"missing required arguments: {', '.join('--' + m for m in missing)}")
        # Extract the case index 
        case_index = args.case_index
        # Extract the relevant flag 
        group_sim = dict_case[args.group]
        # Print the metadata of the simulation
        print(print_meta_data_id_db(args.group,group_sim))
        # Fill up a dictionary of material properties to feed the configuration script
        mat_prop = choose_material_properties(case_sim['k'])
        
        steady_state = args.steady_state
        name_suite = args.name_suite 
        shear_heating = case_sim['shear_heating']

    # Extract the other parameter
    cases = pd.read_csv(pathtotable)
    
    row = cases.iloc[case_index]
    # Create a SimpleNamespace and generate a structured inoput
    args_test = SimpleNamespace(
        case_index = case_index,
        id_batch = args.id_batch,
        group = args.group,
        shear_heating=shear_heating,
        vc=float(row["vc"]),
        age=float(row["age"]),
        Tp=float(row["Tp"]),
        mat_prop = mat_prop,
        steady_state=args.steady_state,
    )
    return args_test

```
The parse arguments additionally creates a dictionary (*mat_prop*) in which the material properties are set: 
```python

def choose_material_properties(k:float)->dict:
    # Create tuples containing the main categories of phases
    materials  = ("mantle", "oceanic_crust", "upper_crust","lower_crust")
    # Create a sub dictionary with the properties that are relevant to modify
    properties = ("alpha", "density", "capacity", "conductivity","radiogenic","k","radiative","rho0",'cp','alpha0')

    # empty skeleton
    phase_properties = {m: {p: None for p in properties} for m in materials}
    
    # Constraint the properties
    # rho0 
    phase_properties['mantle']['rho0'] = 3300.0 
    phase_properties['oceanic_crust']['rho0'] = 2800.0 
    phase_properties['upper_crust']['rho0'] = 2700.0 
    phase_properties['lower_crust']['rho0'] = 2800.0 
    
    # k 
    phase_properties['upper_crust']['k'] = 2.5
    phase_properties['lower_crust']['k'] = 1.5
    phase_properties['oceanic_crust']['k'] = 1.5
    # cp & alpha
    for m in materials:
        phase_properties[m]['cp'] = 1250.0 
        phase_properties[m]['alpha0'] = 3e-5
    
    # radiogenic heating
    phase_properties['oceanic_crust']['radiogenic'] = 0.27e-6
    phase_properties['upper_crust']['radiogenic'] = 1e-6
    phase_properties['lower_crust']['radiogenic'] = 0.27e-6
    phase_properties['mantle']['radiogenic'] = 0.0
    # if k is linear
    if k in (3.1,5.0): 
        
        for m in materials: 
            
            phase_properties[m]['alpha'] = 'Constant'
            phase_properties[m]['capacity'] = 'Constant'
            phase_properties[m]['density'] = 'Constant'
            phase_properties[m]['conductivity'] = 'Constant'
            phase_properties[m]['radiogenic'] = 0.0 
            phase_properties[m]['radiative'] = 0 
        
    phase_properties['mantle']['k'] = k  
    # Set the non-linear law  
    elif k < 0.0: 
        for m in materials:
            phase_properties[m]['radiative'] = 1.0 
            phase_properties[m]['density'] = 'PT'
        phase_properties['mantle']['alpha'] = 'Mantle'
        phase_properties['mantle']['conductivity'] = 'Mantle_Richards_2018'
        phase_properties['mantle']['capacity'] = 'Mantle_Bernard_Ar_199x_FO_FA'
        for m in materials:
            if m !='mantle':
                phase_properties[m]['alpha'] = 'Oceanic_crust'
                phase_properties[m]['conductivity'] = 'Crust_Richards_2018'
                phase_properties[m]['capacity'] = 'Oceanic_crust'
            
        phase_properties['mantle']['k'] = np.abs(k)  

    else: 
        raise ValueError('Wrong k')
    #Print them 
    for m in materials: 
        print(f'{m} properties:')
        for p in phase_properties[m]:
            print(f'       .{p} = {phase_properties[m][p]} ')

    return phase_properties

```
This piece of code is necessary to programatically choose appropriate material properties per each material phases. 

## Configuration script
```python
 # ---
@timing_function
def perform_test(args:SimpleNamespace|None,name_suite:str = 'debug')-> tuple[str,str]:
    """Configuration script of StonedFEniCSx for the time-dependent sensitivity tests. The script reads the input data from a .csv table and creates the input data for StonedFEniCSx. The script also creates the material properties dictionary and assigns it to the phases. The script also creates the input data for StonedFEniCSx and runs the simulation. The results are saved in a folder, in a number of h5 files that will be merged by other scripts.  

    Args:
        args (SimpleNamespace, optional): updated Arguments of the script.
        name_suite (str, optional): Name of the test suite. Defaults to 'debug'.

    Raises:
        ValueError: _description_

    Returns:
        tuple[str,str]: general path of the tests and name of the test
    """
    from stonedfenicsx.config.input_parser import parse_input
    from stonedfenicsx.config.numerical_control import dict_shear_modes
    from stonedfenicsx.stoned_fenicsx import stoned_fenicsx

    path_general = os.path.dirname(os.path.realpath(__file__))
    
    path_input = f"{path_general}/input_tests.yaml"
    
    path_general = os.path.join(path_general,"test_folder")
    
    if not os.path.exists(path_general):
        os.makedirs(path_general)
    

    inp,Ph = parse_input(path_input)
    if args.group == 'debug':
        inp.ctrl.time_max = 0.05

    # controls
    inp.ctrl.decoupling_ctrl = 1
    inp.ctrl.steady_state = args.steady_state
    if args.steady_state == 1: 
        inp.ctrl.tol = 5e-3
        inp.ctrl.it_max = 50
    else: 
        inp.ctrl.tol = 1e-2
        inp.ctrl.it_max = 10

    
    if args.shear_heating:
        inp.ctrl.model_shear = 'SelfConsistent'
    else: 
        inp.ctrl.model_shear = 'NoShear'


    # Create input data - Input is a class populated by default dataset
    # A flag that generate the geometry of the benchmark
    # The input path for saving the results
    inp.ctrl_tbc.slab_age                           = args.age
    inp.ctrl_tbc.temp_max                           = args.Tp
    # Velocity of slab 
    inp.ctrl_ky.v_s                                 = np.array([args.vc,0],dtype=np.float64)

    # Geometrical input
    inp.g_input.cr                                   = 40.0   # Overriding crust 
    inp.g_input.lc                                   = 0.5   # relative amount of lower crust
    inp.g_input.ocr                                  = 6.0  # Crustal thickness
    inp.g_input.decoupling                           = 80  # Lithospheric mantle depth 
    inp.g_input.lab_d                                = inp.g_input.decoupling  # depth of the lab 
    inp.g_input.decoupling                           = 80 # decoupling depth
    inp.g_input.ns_depth                             = 50
    inp.g_input.transition                           = 10
    inp.g_input.slab_type                            = 'CustomRibe'
    inp.g_input.resolution_normal                    = 10.0
    inp.g_input.resolution_refine                    = 1.5 

    if args.shear_heating:
        Ph.shear_heating_disl_law                    = 'Wet_Quartzite_2001_Dislocation_creep'
    
    Ph.shear_heating_disl_phi                        = 5.0
        
    name_diffusion = 'VK_Diffusion_creep'
    name_dislocation = 'VK_Dislocation_creep'                 

    # Modify the phase with the new data: 
    Ph.subducting_plate_mantle.rho0                   = args.mat_prop['mantle']['rho0']
    Ph.subducting_plate_mantle.name_capacity          = args.mat_prop['mantle']['capacity']
    Ph.subducting_plate_mantle.name_conductivity      = args.mat_prop['mantle']['conductivity']
    Ph.subducting_plate_mantle.name_alpha             = args.mat_prop['mantle']['alpha']
    Ph.subducting_plate_mantle.name_density           = args.mat_prop['mantle']['density']
    Ph.subducting_plate_mantle.radiative_conductivity = args.mat_prop['mantle']['radiative']
    Ph.subducting_plate_mantle.radiogenic_heat        = args.mat_prop['mantle']['radiogenic']
    Ph.subducting_plate_mantle.k                      = args.mat_prop['mantle']['k']


    Ph.oceanic_crust.rho0                             = args.mat_prop['oceanic_crust']['rho0']
    Ph.oceanic_crust.name_capacity                    = args.mat_prop['oceanic_crust']['capacity']
    Ph.oceanic_crust.name_conductivity                = args.mat_prop['oceanic_crust']['conductivity']
    Ph.oceanic_crust.name_alpha                       = args.mat_prop['oceanic_crust']['alpha']
    Ph.oceanic_crust.name_density                     = args.mat_prop['oceanic_crust']['density']
    Ph.oceanic_crust.radiative_conductivity           = args.mat_prop['oceanic_crust']['radiative']
    Ph.oceanic_crust.radiogenic_heat                  = args.mat_prop['oceanic_crust']['radiogenic']
    Ph.oceanic_crust.k                                = args.mat_prop['oceanic_crust']['k']

    Ph.wedge_mantle.name_diffusion                    = name_diffusion
    Ph.wedge_mantle.name_dislocation                  = name_dislocation
    Ph.wedge_mantle.rho0                              = args.mat_prop['mantle']['rho0']
    Ph.wedge_mantle.name_capacity                     = args.mat_prop['mantle']['capacity']
    Ph.wedge_mantle.name_conductivity                 = args.mat_prop['mantle']['conductivity']
    Ph.wedge_mantle.name_alpha                        = args.mat_prop['mantle']['alpha']
    Ph.wedge_mantle.name_density                      = args.mat_prop['mantle']['density']
    Ph.wedge_mantle.radiative_conductivity            = args.mat_prop['mantle']['radiative']
    Ph.wedge_mantle.radiogenic_heat                   = args.mat_prop['mantle']['radiogenic']
    Ph.wedge_mantle.k                                 = args.mat_prop['mantle']['k']

    Ph.overriding_mantle.rho0                         = args.mat_prop['mantle']['rho0']
    Ph.overriding_mantle.name_capacity                = args.mat_prop['mantle']['capacity']
    Ph.overriding_mantle.name_conductivity            = args.mat_prop['mantle']['conductivity']
    Ph.overriding_mantle.name_alpha                   = args.mat_prop['mantle']['alpha']
    Ph.overriding_mantle.name_density                 = args.mat_prop['mantle']['density']
    Ph.overriding_mantle.radiative_conductivity       = args.mat_prop['mantle']['radiative']
    Ph.overriding_mantle.radiogenic_heat              = args.mat_prop['mantle']['radiogenic']
    Ph.overriding_mantle.k                            = args.mat_prop['mantle']['k']

    Ph.overriding_upper_crust.rho0                    = args.mat_prop['upper_crust']['rho0']
    Ph.overriding_upper_crust.name_capacity           = args.mat_prop['upper_crust']['capacity']
    Ph.overriding_upper_crust.name_conductivity       = args.mat_prop['upper_crust']['conductivity']
    Ph.overriding_upper_crust.name_alpha              = args.mat_prop['upper_crust']['alpha']
    Ph.overriding_upper_crust.name_density            = args.mat_prop['upper_crust']['density']
    Ph.overriding_upper_crust.radiative_conductivity  = args.mat_prop['upper_crust']['radiative']
    Ph.overriding_upper_crust.radiogenic_heat         = args.mat_prop['upper_crust']['radiogenic']
    Ph.overriding_upper_crust.k                       = args.mat_prop['upper_crust']['k']

    Ph.overriding_lower_crust.rho0                    = args.mat_prop['lower_crust']['rho0']
    Ph.overriding_lower_crust.name_capacity           = args.mat_prop['lower_crust']['capacity']
    Ph.overriding_lower_crust.name_conductivity       = args.mat_prop['lower_crust']['conductivity']
    Ph.overriding_lower_crust.name_alpha              = args.mat_prop['lower_crust']['alpha']
    Ph.overriding_lower_crust.name_density            = args.mat_prop['lower_crust']['density']
    Ph.overriding_lower_crust.radiative_conductivity  = args.mat_prop['lower_crust']['radiative']
    Ph.overriding_lower_crust.radiogenic_heat         = args.mat_prop['lower_crust']['radiogenic']
    Ph.overriding_lower_crust.k                       = args.mat_prop['lower_crust']['k']


    inp.ctrl_io.path_save                             = os.path.join(path_general,f"{name_suite}_{args.group}_{dict_td[f'{args.steady_state}']}")
    inp.ctrl_io.test_name                             = f'{name_suite}_case{args.case_index}_{args.group}'

    wrap_stoned(inp, Ph)
    
    return tuple([inp.ctrl_io.path_test,inp.ctrl_io.test_name])
```







--- 
## Post-processing script


still working on it


## Real cases
## Mexico 

## Japan

## Chile

## Tonga

### Usage of AI 

AI (claude/chatgpt) have been used mostly to look for specific functionality of the code and to correct my grammar. I am fully responsible for the crime against nature that I devised for this project.  
