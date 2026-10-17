# Sensitivity test

<p align="center">
  <img src="Slab_surface_plot/eye_candy.png" alt="Temperature surface slabs and heat fluxes for each family of experiments" width="600">
  <br>
  <em>[a]: P-T conditions of the surface of the slab. The black square represents the HP-LT conditions of the exhumed rocks from the Whitney (2020) database. The green shaded area represents the conditions collected by the Central Honshu experimental ensemble, the red one those collected by the Mexico ensemble, and the grey ones those of the sensitivity cases; [b-c-d]: the heat fluxes at the top of the model for each ensemble of experiments. The purple band represents the x-projection of the decoupling depth, while the orange band represents the x-projection of the depth of maximum shear stress computed along the slab surface.</em>
</p>

---
>[!NOTE]
> This is the workflow that has been used for the manuscript. I tried to release all my workflow. This workflow would 
> work straight in Aire, but I cannot guarantee that in other HPC will work. In the main Zenodo repository
> I will release the h5 files that reproduce the pictures.

This routine has been used to produce the sensitivity analysis described in the manuscript. Everything is set up to reproduce these experiments; as such, the scripts have several hard-coded values. However, the workflow described here can be adapted to use the code for other purposes. The scripts have not been developed as a package.

The experiments can be divided into 6 groups, and for each group there is a number of single-case experiments. So, there are n single-experiment cases that differ only in two or three parameters, and that are run with different common values as a function of their group.

In the main manuscript, 7 groups were run for the sensitivity test, while for the real cases **Mexico** and **Central Honshu** (called `Japan`) only the cases with shear heating were run.

| **group** | **shear heating** | **k** |
|---|---|---|
| `k3nsh` | no | 3.1 |
| `k5nsh` | no | 5.0 |
| `nlnsh` | no | non-linear |
| `k3sh` | yes | 3.1 |
| `k5sh` | yes | 5.0 |
| `nlsh` | yes | non-linear, temperature dependency only |
| `nlpsh` | yes | non-linear, P-T dependency |

The name of the group, in practice, represents the combination of the thermal model (`k3`, `k5`, `nl`) and a flag that indicates whether or not shear heating is activated (`sh` or `nsh`). The post-processing routines generate plots as a function of the shear heating flag. This guarantees that comparisons are made only between tests sharing the same physics.

The routines presented are:

```
Sensitivity_test/
├── additional_plot/
│   └── slab_surface.py
├── post_process_module/
│   ├── data_extractor.py
│   ├── figure_plot.py
│   ├── global_var.py
│   ├── post_process_data.py
│   ├── process_data_visualisation.py
│   └── utils.py
├── generation_cases/
│   └── generate_cases_real_plate.py
├── real_plate_configuration/
│   └── geometric_input_real_plate.yml
├── script_run_stoned.py
├── sensitivity_steady_state_figure_plot.py
├── plot_subduction_surface_real_cases.py
└── time_dependent.py
```

> [!NOTE]
> The Zenodo repository will contain the databases and the simulation metadata, together with a guide to the databases.

---

## Workflow

The workflow is the following:

1. Define the single-experiment cases using one of the two scripts.
2. Run the experiments using a bash submission script that requires an array job of at least 100 processors (an array job in which each job uses only one processor).
3. Post-process the output with one of the scripts:
   1. Merge the h5 files into one file (one per group).
   2. Post-process the data using the pandas routine.
   3. Generate the figures.

This README describes the workflow so that other researchers can use **StonedFEniCSx** and reproduce the results.

---

## Quick start

The user needs to install **StonedFEniCSx** (follow the guide here: [Installation of StonedFEniCSx](https://apiccolo89.github.io/StonedFenicsx/installation.html)).

### Running the bash scripts

After the installation, the user needs to clone the repository:

```bash
git clone https://github.com/APiccolo89/Sensitivity_test.git
```

Then, to generate the suite of experiments:

```bash
python generation_cases/generate_cases_real_plate.py --plate Mexico
```

The argument `--plate` defines the name of the suite: the available names are defined as top-level keys in `generation_cases/real_plate_parametric_config.yml` — currently `Mexico`, `Japan`, `steady_s_sensitivity`, and `td_s_sensitivity` (the time-dependent counterpart of the steady-state sensitivity suite). The script produces the `.csv` file associated with the suite:

- `cases_Sensitivity_Japan.csv`
- `cases_Sensitivity_Mexico.csv`
- `cases_Sensitivity_steady_s_sensitivity.csv`
- `cases_Sensitivity_td_s_sensitivity.csv`

These files collect all the experiments to run.

To run the experiments, there are three bash scripts that handle all the runs; the bash scripts must be launched from the top folder.

> [!WARNING]
> 1. This is the workflow that was used to produce the data in the paper. This means that you need to adapt these files to the policies of your HPC cluster, your server or your local machine.
> 2. The generation of the `.csv` table relies on `scipy.stats.qmc.LatinHypercube`. A seed option has been introduced in the script that generates the cases; however, if the user generates new cases and there is a round-off error in the new table, the post-processing routines may break. Once a case `.csv` has been generated, the script should not be run again.

**Sensitivity test with generic mesh**

```bash
bash submission_file/ss_sens_submission.sh --group k3nsh --name_suite steady_s_sensitivity --np 1 --steady_state 1
bash submission_file/ss_sens_submission.sh --group k5nsh --name_suite steady_s_sensitivity --np 1 --steady_state 1
bash submission_file/ss_sens_submission.sh --group nlnsh --name_suite steady_s_sensitivity --np 1 --steady_state 1
bash submission_file/ss_sens_submission.sh --group k3sh  --name_suite steady_s_sensitivity --np 1 --steady_state 1
bash submission_file/ss_sens_submission.sh --group k5sh  --name_suite steady_s_sensitivity --np 1 --steady_state 1
bash submission_file/ss_sens_submission.sh --group nlsh  --name_suite steady_s_sensitivity --np 1 --steady_state 1
bash submission_file/ss_sens_submission.sh --group nlpsh --name_suite steady_s_sensitivity --np 1 --steady_state 1
```

**Mexico**

```bash
bash submission_file/mexico_submission.sh --group k3sh  --name_suite Mexico --np 1 --steady_state 1
bash submission_file/mexico_submission.sh --group k5sh  --name_suite Mexico --np 1 --steady_state 1
bash submission_file/mexico_submission.sh --group nlsh  --name_suite Mexico --np 1 --steady_state 1
bash submission_file/mexico_submission.sh --group nlpsh --name_suite Mexico --np 1 --steady_state 1
```

**Japan (Central Honshu)**

```bash
bash submission_file/japan_submission.sh --group k3sh  --name_suite Japan --np 1 --steady_state 1
bash submission_file/japan_submission.sh --group k5sh  --name_suite Japan --np 1 --steady_state 1
bash submission_file/japan_submission.sh --group nlsh  --name_suite Japan --np 1 --steady_state 1
bash submission_file/japan_submission.sh --group nlpsh --name_suite Japan --np 1 --steady_state 1
```

The arguments are:

- `--name_suite`: the name of the suite, used to generate the output name;
- `--group`: the code that defines the type of experiments (see above);
- `--np`: the number of processors;
- `--steady_state`: the flag that indicates the type of simulations.

These bash scripts run a Python configuration script called `script_run_stoned.py`.

> [!IMPORTANT]
> **Automation**: it is possible to create a wrapper that automates these operations. However, it depends on your HPC; the commands have been spelled out just to facilitate the use of this workflow.
>
> **bash_clean**: after all the runs are completed, the user can call `bash_clean.sh` with the same signature as the submission script:
> ```bash
> bash submission_file/bash_clean.sh --group nlsh --name_suite Japan --np 1 --steady_state 1
> ```
> `bash_clean.sh` cleans up the output and compresses all the logs of the simulations using the name of the simulation. This ensures the preservation of the logs and the reduction of the number of files.

### Python scripts

After a run of a single group of experiments, the output layout will be:

```
Sensitivity_test/
├── Output_h5_files/
│   └── <name_suite>_h5_files_<group>_<SSTD>/
│       └── Data_Base_<id_batch>.h5
├── test_folder/
│   └── <name_suite>_<group>_<SSTD>/
│       └── <name_suite>_case<case_index>_<group>/
└── output_bash/
    ├── out_<group>_<SSTD>/
    └── err_<group>_<SSTD>/
```

where `<case_index>` is the actual case number (i.e., the row of the `.csv` table), `<SSTD>` is the flag that identifies the mode of the code (`SS` = steady state, `TD` = time dependent), and `<id_batch>` is the index of the array job responsible for that portion of the tests.

`Output_h5_files/` stores the reduced database of the results. Each sub-folder represents a group. Inside `<name_suite>_h5_files_<group>_<SSTD>/` there are small h5 files that contain the relevant information for that group of simulations.

>[!IMPORTANT]
> **StonedFEniCSx** always requires an input.yml file.

> [!IMPORTANT]
> **StonedFEniCSx** always requires an `input.yml` file.

### Post-processing script

After all the runs are performed, the user can use the post-processing script:

```bash
python sensitivity_steady_state_figure_plot.py --name_suite Mexico --steady_state 1 --type sh --csv cases_Sensitivity_Mexico.csv --function_plot real
python sensitivity_steady_state_figure_plot.py --name_suite Japan --steady_state 1 --type sh --csv cases_Sensitivity_Japan.csv --function_plot real
python sensitivity_steady_state_figure_plot.py --name_suite steady_s_sensitivity --steady_state 1 --type sh --csv cases_Sensitivity_steady_s_sensitivity.csv --function_plot real
python sensitivity_steady_state_figure_plot.py --name_suite steady_s_sensitivity --steady_state 1 --type nsh --csv cases_Sensitivity_steady_s_sensitivity.csv --function_plot real
```

The script must be fed with:

- `--name_suite`: the name of the target suite;
- `--steady_state`: the steady-state flag;
- `--type`: the flag that indicates whether the tests have shear heating (`sh`) or not (`nsh`);
- `--csv`: the name of the `.csv` table;
- `--function_plot`: the flag that indicates whether the suite being examined uses the made-up geometry or a real plate; it changes the standard plot.

The script has been conceived such that the three thermal models must have been run prior to its use: this means that you need `<name_suite>_{k3,k5,nl}sh` and `<name_suite>_{k3,k5,nl}nsh`.
This script produces the following output folder:

```
Result_shear_heating_<type>_<name_suite>/
```

In addition to this script, there is another script that produces a figure with the P-T conditions of the top surface of the slab and the heat fluxes. This script assumes that the non-linear thermal model with shear heating has been run for Mexico, Central Honshu and the sensitivity cases.

## HDF5 data layout

The post-processing produces two levels of HDF5 files:

1. **Partial files**: one per array-job worker, written by `save_test_data` as each case finishes.
2. **Merged file**: one per group, produced by `merge_data_base` from all partial files.

### Partial files

`Output_h5_files/<name_suite>_h5_files_<group>_<SSTD>/Data_Base_<id_batch>.h5`

```
Data_Base_<id_batch>.h5
├── Xi                 (ny, nx)   x-coordinates of the 2D interpolation grid
├── Yi                 (ny, nx)   y-coordinates of the 2D interpolation grid
├── n_lc_tst           scalar     number of cases accumulated in this file
├── mean_lc_tst        (ny, nx)   running mean of the slab temperature field (Welford)
├── var_lc_tst         (ny, nx)   running sum of squared deviations M2 (Welford)
└── <case_name>/                  one group per case
    ├── <input parameters>        scalars (strings stored as UTF-8)
    ├── x_<T>, y_<T>              scalars
    ├── X_S                       (2, Ns)
    ├── T_S, P_S                  (Ns,)
    ├── X_O                       (2, No)
    ├── T_O, P_O                  (No,)
    ├── x_top, heat_flux          (Nq,)
    ├── Slab_surface              (ny, nx)
    ├── tau_shear                 (Ns,)
    └── tau_shear_{mean,std,median,min,max}   scalars
```

| Dataset | Shape | Type | Description |
|---|---|---|---|
| `<input parameters>` | scalar | float / int / str | One dataset per column of the case table (e.g. `case_index`, `id_batch`, `vc`, `age`, `shear_heating`, …); `mat_prop` is not saved |
| `x_<T>`, `y_<T>` | scalar | float32 | Coordinates of the deepest point of the `<T>` °C isotherm, for T = 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900. The isotherm is extracted from T + 0.3 K/km × depth |
| `X_S` | (2, Ns) | float32 | Slab-surface coordinates (`[0]` = x, `[1]` = y), sorted by x |
| `T_S`, `P_S` | (Ns,) | float32 | Temperature [°C] and lithostatic pressure along the slab surface |
| `X_O` | (2, No) | float32 | Oceanic-surface coordinates, sorted by x |
| `T_O`, `P_O` | (No,) | float32 | Temperature and lithostatic pressure along the oceanic surface |
| `x_top` | (Nq,) | float32 | x-coordinates of the top surface (y = 0, 0 ≤ x < 600 km) |
| `heat_flux` | (Nq,) | float32 | Vertical heat flux at `x_top` |
| `Slab_surface` | (ny, nx) | float | Steady-state temperature interpolated on `Xi`, `Yi` |
| `tau_shear` | (Ns,) | float | Effective shear stress along the slab surface [MPa]; zero if shear heating is off |
| `tau_shear_{mean,std,median,min,max}` | scalar | float | Statistics of `tau_shear` |

Arrays are stored with gzip compression (level 4, shuffle); scalars are uncompressed.
Each partial-file folder also contains `Failed_tests.jsonl`, with one JSON record per case (`case_index`, `id_batch`, `success`, `path_h5`).

### Merged file

`<name_suite>_h5_files_<group>_<SSTD>.h5`

```
<name_suite>_h5_files_<group>_<SSTD>.h5
├── Xi, Yi             (ny, nx)   2D interpolation grid
├── global_mean        (ny, nx)   mean slab temperature field over all cases
├── global_var         (ny, nx)   sample variance (global_M2 / (global_count − 1))
├── global_M2          (ny, nx)   merged sum of squared deviations
├── global_count       scalar     total number of cases
├── mean_lc_tst, var_lc_tst, n_lc_tst   local statistics of the last partial file only (not used)
└── <case_name>/                  all case groups from all partial files, unchanged
```

The global statistics are obtained by combining the local Welford statistics of each partial file with the parallel update.

## Global performance

Below is the overall performance of the tests. The following table lists the name of the suite of experiments, the number of tests, and their average duration and number of iterations.

The residuals are listed as well:

- $`r_{cors}`$ abs: the average final combined residual of the conservation equations. This quantity is absolute and dimensionless.
- $`r_{cors}`$ rel: the average final combined residual of the conservation equations, relative to the first iteration.
- $`r_{dif}`$: the average relative combined difference between iterations. This quantity is computed from the relative difference of each solution variable using $`r_v = L_{2}(v^{k+1}-v^{k})/(v^{k+1}+v^{k})`$, where $`v`$ is the target solution variable (temperature, pressure, velocity and lithostatic pressure). The residuals of the variables are then combined using a Euclidean norm.
- $`L_{\infty}(\Delta T)`$: the average $`L_{\infty}`$ norm of the absolute temperature difference between iterations over all experiments. This norm is used to decide whether or not the problem has converged (0.01 K).
- $`L_{2}(\Delta T)`$: the average $`L_{2}`$ norm of the temperature difference between iterations over all experiments.

### Sensitivity tests

| group | n_tests | duration [s] | iterations | $`r_{cors}`$ abs | $`r_{cors}`$ rel | $`r_{dif}`$ | $`L_{\infty}(\Delta T)`$ [K] | $`L_{2}(\Delta T)`$ [K] |
|---|---|---|---|---|---|---|---|---|
| steady_s_sensitivity_k3nsh | 7000 | 192.927 | 40.1876 | 1.26505e-06 | 0.00923214 | 4.56049e-06 | 0.00889464 | 0.00154495 |
| steady_s_sensitivity_k5nsh | 7000 | 198.95 | 37.3014 | 2.49656e-07 | 0.00213116 | 5.52193e-06 | 0.00888703 | 0.00167577 |
| steady_s_sensitivity_nlnsh | 7000 | 297.189 | 37.5694 | 2.45995e-07 | 0.00175062 | 5.28206e-06 | 0.00883886 | 0.00160054 |
| steady_s_sensitivity_k3sh | 7000 | 189.214 | 40.1999 | 7.16584e-07 | 0.00530286 | 4.58042e-06 | 0.00889934 | 0.00155349 |
| steady_s_sensitivity_k5sh | 7000 | 174.929 | 37.2961 | 1.31351e-07 | 0.00121911 | 5.52591e-06 | 0.00888951 | 0.00167953 |
| steady_s_sensitivity_nlpsh | 7000 | 290.001 | 35.5666 | 1.04691e-07 | 0.000925267 | 5.87498e-06 | 0.00881412 | 0.0016635 |
| steady_s_sensitivity_nlsh | 7000 | 281.643 | 37.5433 | 1.28771e-07 | 0.000976184 | 5.27888e-06 | 0.00883921 | 0.00160237 |

### Japan (Central Honshu)

### Mexico

| group | n_tests | duration [s] | iterations | $`r_{cors}`$ abs | $`r_{cors}`$ rel | $`r_{dif}`$ | $`L_{\infty}(\Delta T)`$ [K] | $`L_{2}(\Delta T)`$ [K] |
|---|---|---|---|---|---|---|---|---|
| Mexico_k3sh | 3500 | 87.6235 | 29.0349 | 1.48178e-07 | 4.58714e-06 | 5.32437e-05 | 0.0360183 | 0.0040375 |
| Mexico_k5sh | 3500 | 86.9429 | 28.9277 | 1.66025e-08 | 7.50896e-06 | 2.3091e-05 | 0.00980348 | 0.00119388 |
| Mexico_nlsh | 3500 | 118.762 | 28.238 | 4.22902e-08 | 3.80492e-06 | 4.47199e-05 | 0.0225374 | 0.00260068 |
| Mexico_nlpsh | 3500 | 136.022 | 28.0354 | 3.33262e-08 | 4.36059e-06 | 3.86298e-05 | 0.0180891 | 0.00208725 |

The linear tests took on average 3 hours and 44 minutes, while the non-linear tests took on average 5 hours and 50 minutes (using 100 array jobs). The linear numerical experiments cost significantly less than the non-linear experiments and require 64% of the total resources used by the non-linear ones.

Initially, **StonedFEniCSx** was organised to have a common set of equations, and it adjusted each member as a function of the case; however, linear models do not require the recompilation of the thermal form. The differences, thus, are a consequence of the last update of the code.

## Usage of AI

AI (Claude/ChatGPT) was used to look up specific code functionalities and to assist with grammar. The author is responsible for the code and its design.
