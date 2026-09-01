# Sensitivity test

This routine has been used to produce the sensitivity analysis of (name of the manuscript). 
## Work Flow
The work flow is the following: 
1. Run the number of the cases using an array of processor
2. Each run is saving the output of the test in the test_folder/{case}/test_case_num(x)
3. After storing the relevant data the test is automatically removed for saving the memory. 
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
