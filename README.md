# Sensitivity test

<p align="center">
  <img src="Slab_surface_plot/eye_candy.png" alt="Temperature surface slabs and heat fluxes for each family of experiments" width="600">
  <br>
  <em>[a]: P-T conditions of the surface of the slab. The black square represents the HP-LT conditions of the exhumed rocks coming from the Whitney (2020) database. The green shaded area represents the conditions collected by the Central Honshu experimental ensemble, the red one those collected by the Mexico ensemble, while the grey ones are those related to the sensitivity cases; [b-c-d] The heat fluxes at the top of the model for each ensemble of experiments. The purple band represents the x-projection of the decoupling depth, while the orange band represents the x-projection of the depth of maximum shear stress computed along the slab surface.</em>
</p>



---
This routine has been used to produce the sensitivity analysis described in the manuscript. Everything is set up to reproduce these experiments; as such, the scripts have several hard-coded values. However, the workflow described here can be adapted to use the code for other purposes. The scripts have not been developed as a package.

The experiments can be divided into 6 groups, and for each group there is a number of single-case experiments. So, there are n single experiment cases that differ only in three or two parameters, and that are run with different common values as a function of their group.

In the main manuscript, 6 groups were run for the sensitivity test, while for the real cases **Mexico** and **Central Honshu** (called Japan) only the cases with shear heating were run.

|**group** | **shear heating** | **k** | 
|------|-----|---|
|*k3nsh* | no  | 3.1|  
|*k5nsh* | no  | 5.0|  
|*nlnsh*  | no  | non-linear|
|*k3sh*| yes | 3.1|
|*k5sh*   | yes  | 5.0|  
|*nlsh*  | yes  | non-linear temperature dependency only|
|*nlpsh*  | yes  | non-linear P-T dependency|

The name of the group, in practice, represents the combination of the thermal model (`k3`, `k5`, `nl`) and a flag that indicates whether or not shear heating is activated. The post-processing routines generate plots as a function of the **shear heating** flag (`sh` or `nsh`). This guarantees that comparisons are made only between tests sharing the same physics.



The routines presented are:
```
Sensitivity_Test/
    ├── additional_plot/
        ├── slab_surface.py
    ├── post_process_module/
        ├── data_extractor.py
        ├── figure_plot.py
        ├── global_var.py 
        ├── post_process_data.py
        ├── process_data_visualisation.py
        ├── utils.py
    ├── generation_cases/
        ├── generate_cases_real_plate.py
    ├── real_plate_configuration/ 
        ├── geometric_input_real_plate.yml
    ├── script_run_stoned.py
    ├── sensitivity_steady_state_figure_plot.py
    ├── plot_subduction_surface_real_cases.py
    ├── time_dependent.py
```

>[!NOTE]
> In the zenodo repository there will be the databases, and also the simulations metadata. There will be a guide for the databases. 
---

## Workflow

---

The workflow is the following:
1. Define the single experiment cases using one of the two scripts.
2. Run the experiments using a bash submission script that requires an array job of at least 100 processors (an array job in which each job uses only one processor).
3. Post-process the output with one of the scripts.
3.1 Merge the h5 files into one file (one per group).
3.2 Post-process the data using the pandas routine.
3.3 Generate the figures.


This README describes the workflow so that other researchers can use StonedFEniCSx and reproduce the results.

---
## Quick start: 

The user needs to install **StonedFEniCSx** (follow the guide here: [InstallationSTONED](https://apiccolo89.github.io/StonedFenicsx/installation.html)).
### Running bash scripts 
After the installation, the user needs to copy the repository: 

```bash
    git clone https://github.com/APiccolo89/Sensitivity_test.git
```
Then, to run the suite of experiments: 
```bash 
    python generation_cases/generate_cases_real_plate.py --plate Mexico
```
The argument `--plate` is the one defining the name of the suite: the names available are defined as top-level keys in `generation_cases/real_plate_parametric_config.yml` — currently `Mexico`, `Japan`, `steady_s_sensitivity`, and `td_s_sensitivity` (time-dependent counterpart of the steady-state sensitivity suite). The script will produce the `.csv` files associated with the plate: 

- cases_Sensitivity_Japan.csv
- cases_Sensitivity_Mexico.csv
- cases_Sensitivity_steady_s_sensitivity.csv
- cases_Sensitivity_td_s_sensitivity.csv

These files collect all the experiments to run.

To run the experiments, there are three bash scripts that handle all the runs; the bash scripts must be launched from the top folder:
> [!WARNING]
> 1. This is the workflow that was used to produce the data in the paper. This means that you need to adapt these files to the policies of your HPC cluster, your server or your local machine.
> 2. The generation of the .csv table relies on `scipy.stats.qmc.LatinHypercube`. A seed option has been introduced in the script that generates the cases; however, if the user generates new cases and there is a round-off error in the new table, the post-processing routines may break. Once a case .csv has been generated, the script should not be run again.


**Test Sensitivity_test with generic mesh**
```bash
bash submission_file/ss_sens_submission.sh --group k3nsh --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/ss_sens_submission.sh --group k5nsh --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/ss_sens_submission.sh --group nlnsh --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/ss_sens_submission.sh --group k3sh  --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/ss_sens_submission.sh --group k5sh  --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/ss_sens_submission.sh --group nlsh  --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/ss_sens_submission.sh --group nlpsh  --name_suite steady_s_sensitivity --np 1 --steady_state 1 

```
**Test Mexico**
```bash
bash submission_file/mexico_submission.sh --group k3sh --name_suite Mexico --np 1 --steady_state 1
bash submission_file/mexico_submission.sh --group k5sh --name_suite Mexico --np 1 --steady_state 1
bash submission_file/mexico_submission.sh --group nlsh --name_suite Mexico --np 1 --steady_state 1
bash submission_file/mexico_submission.sh --group nlpsh --name_suite Mexico --np 1 --steady_state 1
```
**Test Japan**
```bash
bash submission_file/japan_submission.sh --group k3sh --name_suite Japan --np 1 --steady_state 1
bash submission_file/japan_submission.sh --group k5sh --name_suite Japan --np 1 --steady_state 1
bash submission_file/japan_submission.sh --group nlsh --name_suite Japan --np 1 --steady_state 1
bash submission_file/japan_submission.sh --group nlpsh --name_suite Japan --np 1 --steady_state 1
```
The arguments are: 
- `--name_suite`: the name of the suite, used to generate the output name. 
- `--group`: the code that defines the type of experiments (see above)
- `--np`: the number of processors
- `--steady_state`: the steady state flag that indicates the type of simulations. 
These bash scripts run a Python configuration script called `script_run_stoned.py`.

>[!IMPORTANT]
>**Automation**  It is possible to create a wrapper that automates these operations. However, it depends on your HPC; the commands have been spelled out just to facilitate the use of this workflow.
>**bash_clean** After all the runs are completed, the user can use the same signature of the bash script to call
>```bash
>bash submission_file/bash_clean.sh --group nlsh --name_suite Japan --np 1 --steady_state 1
>```
>`bash_clean` cleans up the output and compresses all the logs of the simulations using the name of the simulation. This ensures the preservation of the logs and the reduction of the number of files.

### Python scripts

After a run of a single group of experiments, the output layout will be:

```
    Sensitivity_tests/
      ├── Output_h5_files/
        ├──{`name_suite`}_{`group`}_h5_files/
            ├──Data_Base_{id_batch}.h5
      ├── test_folder/
         ├──{`name_suite`}_{`group`}_{`flagSS`}
            ├──{`name_suite`}_case{case_index}_{`group`}
      ├── output_bash/
            ├──out_{'group'}_{'flagSS'}
            ├──err_{'group'}_{'flagSS'}
```    

where `case_index` is the actual case number (i.e., the row of the .csv table), `flagSS` is a flag that identifies the mode of the code (`SS`=Steady State, `TD`=Time Dependent), and `id_batch` is the processor responsible for that portion of the tests.


In `h5_Outputs`, the reduced database of the results is stored. Each sub-folder represents a group. Inside ``{`name_suite`}_{`group`}_h5_files`` there are small h5 files that contain the relevant information for that group of simulations.

In `test_folder` there is the actual output of **StonedFEniCSx**, divided into group and suite folders. The outputs of the single tests are removed. However, the user can modify the Python script to keep all the output. This depends on the tolerance of the cluster towards the number of files that you are allowed to store.

[!IMPORTANT]
> **StonedFEniCSx** always requires an input.yml file.


### Post-processing script

After all the runs are performed, the user can use the post-processing script:

```bash
python sensitivity_steady_state_figure_plot.py --name_suite Mexico --steady_state 1 --type sh --csv cases_Sensitivity_Mexico.csv --function_plot real
python sensitivity_steady_state_figure_plot.py --name_suite Japan --steady_state 1 --type sh --csv cases_Sensitivity_Japan.csv --function_plot real
python sensitivity_steady_state_figure_plot.py --name_suite steady_s_sensitivity --steady_state 1 --type sh --csv cases_Sensitivity_steady_s_sensitivity.csv --function_plot real
python sensitivity_steady_state_figure_plot.py --name_suite steady_s_sensitivity --steady_state 1 --type nsh --csv cases_Sensitivity_steady_s_sensitivity.csv --function_plot real

```
The script must be fed with: 
- `--name_suite`: the name of the target suite
- `--steady_state`: the steady state flag
- `--type`: the flag that tells whether the test has shear heating or not
- `--csv`: the name of the csv table 
- `--function_plot`: the flag that tells whether the suite being examined is the made-up geometry or a real plate -> changes the standard plot.

The script has been conceived such that the three thermal models must have been run prior to its use: this means that you need {name_suite}_{k3...nl}{sh} {name_suite}_{k3...nl}{sh}.
This script produces the following output folder: 
```
Result_shear_heating_sh_{name_suite}/
```

On top of this script, there is another script that produces a figure with the P-T conditions of the top surface of the slab and the heat fluxes. This script
assumes that the non-linear thermal model with shear heating of Mexico, Central Honshu and the sensitivity cases has been run.

### Main Output

Below is the overall performance of the tests. In the following table, the name of the suite of experiment, the number of the test and their average duration and number of iteration are listed. 

The residual are listed as well: 

- $`r_{cors}`$ abs: the average final combined conservation equation residual. This quantity is the absolute one, and it is dimensionless. 
- $`r_{cors}`$ rel: the average final combined equation residual (relative to the first iteration). 
-$`r_{dif}`$: the average relative combined difference between iteration. This quantity is compute by computing the relative difference of each of solution variable using the following equation $`r_v = L_{2}(v^{k+1}-v^{k})/(v^{k+1}+v^{k})`$, where `v` is the target solution variable (temperature, pressure, velocity and lithostatic pressure). Then each of the residual is combined using an eucleadian norm. 
- $`L_{\infty}(\Delta T)`$ is the average L1 norm of the absolute difference in temperature between iterations of all experiments. This norm is used to decide whether or not the problem is converged (0.01 K). 
-  $`L_{2}(\Delta T)`$ is the average L2 norm of the temperature difference of all experiments.

#### Sensitivity Tests
| group | n_tests | duration [s] | iterations | $`r_{cors}`$ abs | $`r_{cors}`$ rel | $`r_{dif}`$ | $`L_{\infty}(\Delta T)`$ [K] | $`L_{2}(\Delta T)`$ [K] |
|---|---|---|---|---|---|---|---|---|
| sensitivity_study_k3nsh | 7000 | 192.927 | 40.1876 | 1.26505e-06 | 0.00923214 | 4.56049e-06 | 0.00889464 | 0.00154495 |
| sensitivity_study_k5nsh | 7000 | 198.95 | 37.3014 | 2.49656e-07 | 0.00213116 | 5.52193e-06 | 0.00888703 | 0.00167577 |
| sensitivity_study_nlnsh | 7000 | 297.189 | 37.5694 | 2.45995e-07 | 0.00175062 | 5.28206e-06 | 0.00883886 | 0.00160054 |
| sensitivity_study_k3sh | 7000 | 189.214 | 40.1999 | 7.16584e-07 | 0.00530286 | 4.58042e-06 | 0.00889934 | 0.00155349 |
| sensitivity_study_k5sh | 7000 | 174.929 | 37.2961 | 1.31351e-07 | 0.00121911 | 5.52591e-06 | 0.00888951 | 0.00167953 |
| sensitivity_study_nlpsh | 7000 | 290.001 | 35.5666 | 1.04691e-07 | 0.000925267 | 5.87498e-06 | 0.00881412 | 0.0016635 |
| sensitivity_study_nlsh | 7000 | 281.643 | 37.5433 | 1.28771e-07 | 0.000976184 | 5.27888e-06 | 0.00883921 | 0.00160237 |

#### Japan - Central Honshu - 

#### Mexico 
| group | n_tests | duration [s] | iterations | $`r_{cors}`$ abs | $`r_{cors}`$ rel | $`r_{dif}`$ | $`L_{\infty}(\Delta T)`$ [K] | $`L_{2}(\Delta T)`$ [K] |
|---|---|---|---|---|---|---|---|---|
| Mexico_k3sh | 3500 | 87.6235 | 29.0349 | 1.48178e-07 | 4.58714e-06 | 5.32437e-05 | 0.0360183 | 0.0040375 |
| Mexico_k5sh | 3500 | 86.9429 | 28.9277 | 1.66025e-08 | 7.50896e-06 | 2.3091e-05 | 0.00980348 | 0.00119388 |
| Mexico_nlsh | 3500 | 118.762 | 28.238 | 4.22902e-08 | 3.80492e-06 | 4.47199e-05 | 0.0225374 | 0.00260068 |
| Mexico_nlpsh | 3500 | 136.022 | 28.0354 | 3.33262e-08 | 4.36059e-06 | 3.86298e-05 | 0.0180891 | 0.00208725 |

The linear tests took on average 3 hours and 44 minutes, while the non-linear tests took on average 5 hours and 50 minutes (using 100 array jobs). The linear numerical experiments cost significantly less than the non-linear experiments and require 64\% of the total resources used by the non-linear ones. 

Initially, **StonedFEniCSx** was organised to have a common set of equations, and it adjusted each member as a function of the case; however, linear models do not require the recompilation of the thermal form. The differences, thus, are a consequence of the last update of the code. 





### Usage of AI 

AI (Claude/ChatGPT) was used to look up specific code functionalities and to assist with grammar. The author is responsible for the code and its design.
