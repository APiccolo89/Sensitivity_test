#Test Sensitivity_test with generic mesh
```bash
bash submission_file/steady_state_sensy.sh --group k3nsh --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/steady_state_sensy.sh --group k5nsh --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/steady_state_sensy.sh --group nlnsh --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/steady_state_sensy.sh --group k3sh --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/steady_state_sensy.sh --group k5sh --name_suite steady_s_sensitivity --np 1 --steady_state 1 
bash submission_file/steady_state_sensy.sh --group nlsh --name_suite steady_s_sensitivity --np 1 --steady_state 1 
```
#Test with time dependent 
```bash
bash submission_file/time_dependent.sh --group k3sh --name_suite time_dependent --np 1 --steady_state 0
bash submission_file/time_dependent.sh --group k5sh --name_suite time_dependent --np 1 --steady_state 0
bash submission_file/time_dependent.sh --group nlsh --name_suite time_dependent --np 1 --steady_state 0
```

#Test Mexico 
```bash
bash submission_file/mexico_steady_state_sensy.sh --group k3sh --name_suite Mexico --np 1 --steady_state 1
bash submission_file/mexico_steady_state_sensy.sh --group k5sh --name_suite Mexico --np 1 --steady_state 1
bash submission_file/mexico_steady_state_sensy.sh --group nlsh --name_suite Mexico --np 1 --steady_state 1
```