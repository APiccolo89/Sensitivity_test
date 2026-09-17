#!/bin/bash
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
