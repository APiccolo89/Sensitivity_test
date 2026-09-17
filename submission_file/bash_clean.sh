#!/bin/bash env 

remove_temporary_files(){
    local folder_to_remove=$1
    local out_bash=$2
    local name_file=$3

    path_bash="${PWD}"/output_bash

    echo "Remove ${folder_to_remove}"
    echo "Compress ${out_bash} and remove it" 
    # Remove the test folder related to this run
    echo "${folder_to_remove}"
    rm -rf  -- "${folder_to_remove}"

    echo "Folder removed" 
    # Compress the output files
    echo "$path_bash/$name_file.tar $out_bash"
    tar -cvf "$path_bash/$name_file.tar" "$out_bash"    
    # Remove the outbash 
    echo "$out_bash"
    rm -rf -- "$out_bash" 
}
# ---
# Private function testing
_unit_test(){
    # Create folder that mimicks the output of the submission script and python script
    # Remove them, save the outbash folder into a tar for tracking the overall goodness of the 
    # tests, and print the memory pressure on the cluster. 
    pwd_2_files=${PWD}

    echo "Creating the fake folders for the test"
    folder1="$pwd_2_files"/output_bash/test_bash
    folder2="$pwd_2_files"/test_folder/test_folder
    file_tar="$pwd_2_files"/output_bash/file_tar_test.tar

    mkdir -p "$pwd_2_files"/output_bash/test_bash
    mkdir -p "$pwd_2_files"/test_folder/test_folder
    mkdir -p "$pwd_2_files"/fenics_cache/test_folder 

    echo "Using remove_temporary_file to test it" 
    remove_temporary_files "${folder1}" "${folder2}" file_tar_test 
    echo "Asserting the tests"
    # Are the folders removed?
    failed=0
    # -- 
    if [ ! -d "$folder1" ];then
        echo "Test1 success"
    else 
        echo "Test0 Failed"
        failed=$(("$failed" + 1))
    fi
    # -- 
    if [ ! -d "$folder2" ];then
        echo "Test2 success"
    else 
        echo "Test2 Failed"
        failed=$(("$failed" + 1))
    fi
    # -- 
    if [ ! -d "$folder3" ];then
        echo "Test3 success"
    else 
        echo "Test3 Failed"
        failed=$(("$failed" + 1))
    fi
    # --- 
    if [ $failed -eq 0 ];then
        echo "all test passed"
    else
        echo "Some Tests has not passed" 
        die "Well... failed_tests: ${failed}"
    fi 

    lfs quota -h -u "$USER" /scratch 
}
# --- 


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

folder_bash=output_bash/out_${group}_${MODE}/
folder_test="${name_suite}"_"${group}"_"${MODE}"
tar_file_name="${name_suite}"_"${group}"_"${MODE}".tar
_unit_test 
remove_temporary_files ${folder_test} ${folder_bash} ${tar_file_name} 