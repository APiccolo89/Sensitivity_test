# Sensitivity test

---
This routine has been used to produce the sensitivity analysis of (name of the manuscript). Everything is setup up to reproduce these experiments, as such, the script have a several hardcoded values. However, the workflow that is described here, can be translated to use the code for other purposes.  

---

## Work Flow

---
The work flow is the following: 
1. Run the number of the cases using an array of processor
2. Each run is saving the output of the test in the test_folder/{case}/test_case_num(x)
3. After storing the relevant data the test is automatically removed for saving the memory. 
---
## Generation single experiments cases
---
In the main manuscript, the sensitivity study is divided in steady state runs, that covers a wide range of condition, and time-dependent scenario, that covers a limited set of parameters. 

There are two python scripts that generates the cases table: 
```
generate_case_sensitivity_time_dependent.py
generate_case_sensitivity_steady_state.py
```
### Steady state Sensitivity
Steady state sensitivity case are run such that 7000 random test can be generated and run on the fly. Steady state runs cover:

```python
    vc_arr:list[float]=[3.0, 15.0] # convergence velocity
    Tp_arr:list[float]=[1350.0, 1425.0] # potential temperature in celsius degree
    age_arr:list[float]=[30.0,100.0] # age in Myr
```

The generation of the single experiments cases is carried out with ``scipy.stats.qmc.LatinHypercube`` library, and 

```python 
    sampler = scipy.stats.qmc.LatinHypercube(d=3)
    sample = sampler.random(n=n)
    scaled_sample = scipy.stats.qmc.scale(sample, lbounds, hbounds)
```

This generates a random ensemble of data. The resultant array of experiment is, then, saved as a csv table that will be read by the python scripts that run the numerical experiment. 

### Time state Sensitivity


Time dependent run 100 cases (the potential temperature is dropped as main parameter), and they are repeated for their steady state counterpart. These tests are run for 30 Myr. 


In the time dependent cases, there is no need to run thousands of cases as they are computational expensive. The generation of the cases are a simply linear combination of two arrays: velocity of convergence and age. 


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
## Bash script: 
The bash scripts are stored in submission_file. To run: 
bash submission_file/run_script.sh --case k3nsh --np 1 --steady_state 1 --name_suite steady_state_ss

- **case**: indicates the category of run: 

|**case** | **shear heating** | **k** | 
|------|-----|---|
|*k3nsh* | no  | 3.1|  
|*k5nsh* | no  | 5.0|  
|*nlnsh*  | no  | non-linear|
|*k3sh*| yes | 3.1|
|*k5sh*   | yes  | 5.0|  
|*nlsh*  | yes  | non-linear|
- **np**: the number of processor
- **steady_state**: 1: steady state simulation; 0: time dependent 
- **name_suite**: name of the batches of experiment that the user wants to run. 

The bash script can parse the argument, and instruct correctly the python configuration script. 

--- 
