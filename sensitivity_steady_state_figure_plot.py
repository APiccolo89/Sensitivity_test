import argparse
import gc
import os
import time as timing
import tracemalloc
from collections import namedtuple
from pathlib import Path

import cmcrameri
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.collections import LineCollection
from numba import njit

from post_process_module.figure_plot import *
from post_process_module.process_data_visualisation import wrapper_merging_h5


class ErrorParsing(Exception):
    """_summary_

    Args:
        Exception (_type_): Error of the parser. Check if the user provide the right errors, and create an instance
        of the error on the basis of the most likely mistake. 
    """
    def __init__(self,message:str,error_code:int):
        self.message = message 
        self.error_code = error_code
    def __str__(self):
        if self.error_code == -1: 
            string_error = (f'Code error{self.error_code}: Base... the required arguments has not been given:\n'
                            f'{self.message}')
        elif self.error_code == 0: 
            string_error = f'Code error {self.error_code}: Wrong sh-group: \b{self.message}'
        elif self.error_code == 1: 
            string_error = f'Code error {self.error_code}: Wrong steady state flag: \b{self.message}'
        elif self.error_code ==2: 
            string_error = (f'Code error {self.error_code}: Wrong suite name, '
                            f'all the test of the suite are not existing: \n {self.message}')
        elif self.error_code ==3: 
            string_error = (f'Code error {self.error_code}. Not all the group of tests has been performed:\n'
                            f'{self.message}')
        elif self.error_code ==4: 
            string_error = (f'Code error {self.error_code}. WRONG CSV file:\n'
                            f'{self.message}')
        else: 
            string_error = 'Ops, unknown error... '
        
        return string_error


def check_parsed_arguments(args)->None: 
    """_summary_

    Args:
        args (_type_): parsed argument of the script

    Raises:
        ErrorParsing: If the group name is wrong, or all the test are not performed, or any of the test is not 
        performed, raise ErrorParsing. 
        
    """
    required = ["sh", "name_suite", "csv", "steady_state"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        s0 = (f"missing required arguments: {', '.join('--' + m for m in missing)}")
        raise ErrorParsing(s0,-1)
    
    
    if args.sh not in  ('sh','nsh'):
        s0 = f'The --sh argument must be sh or nsh got {args.sh!r}'
        raise ErrorParsing(s0,0)
    
    if args.steady_state == 'error':
        s0 = 'Wrong steady state flag'
        raise ErrorParsing(s0,1)
    
    aux_path = Path(__file__).resolve().parent
    folder = []
    sub_group = ('k3','k5','nl')
    for i in sub_group:
        aux_path_data_base = aux_path / 'Output_h5_files' / f'{args.name_suite}_h5_files_{i}{args.sh}_{args.steady_state}'
        folder.extend([aux_path_data_base.is_dir()])    
            
    s0 = None
        # Check whether or not all the folder are not existing
    if all(not value for value in folder):
        s0 = (f'error_0: name suite error. All the groups for {args.sh} are missing. Check the spelling or run the suite\n')
        raise ErrorParsing(s0,2)
        # Check if any folder is false
    elif not all(folder): 
        indices = [index for index, value in enumerate(folder) if value == False]
        values = [f"{sub_group[i]}{args.sh}" for i in indices]
        suite_wrong = ''
        for i in values:
            suite_wrong = suite_wrong + ' ' +i 
        s0 = (f'error_1: not all the group has been executed. missing suites: {suite_wrong!r}/n')
        raise ErrorParsing(s0, 3)
    
    csv_path = aux_path / 'generation_cases' /args.csv
    if not csv_path.is_file():
        s0 = f'The csv file listed is not existing. Actual value {args.csv!r}'
        raise ErrorParsing(s0,4)
    
    
    print('The parsing is OK!')
        

# ----------------------- Main code ------------------ #
def plot_sensitivity_study_steady_state():
    """ Explorative script to plot the results of the sensitivity test. 
    This script is meant to merge the partial data files, extract the relevant data,
    and experimenting. 
    TO DO: 
    Create a more structured script for the production of the final figure: 
    -> Template function that accept function as argument (wrapped)
    -> A better data flow -> a few functions have hard coded data, which is not ideal.
    -> Better comments. 
    The script is not commented yet, as the results of the following study has not been published yet. I don't 
    want to make the life easier to those that want to use my code before my paper is out.
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--sh",            type=str,                help = "Super Group: experiment with or without shear heating: sh = yes, shear heating; nsh = no, shear heating")
    parser.add_argument("--name_suite",    type=str,                help = "Name of the test suite. This is a string that corresponds to the name of the test suite.")
    parser.add_argument("--steady_state",  type=int,                help = "Flag to run the test in steady-state or time-dependent mode. 1 = steady-state, 0 = time-dependent")
    parser.add_argument("--csv",           type=str,                help = 'csv table used to produce the cases. Important to check what are the main variables that changes within the runs.')
    parser.add_argument("--number_worker", type=int, default=100,   help = 'number of processor of the array job. The default is 100, as the run of the manuscript. User can change, but needs to be consistent with their work flow.')
    args = parser.parse_args()
    
    if args.steady_state == 1:
        args.steady_state = 'SS'
    elif args.steady_state == 0:
        args.steady_state = 'TD'
    else:
        args.steady_state = 'error'
    
    check_parsed_arguments(args)
    

    print("Merging partial data files...")

    aux_path = Path(__file__).resolve().parent
    
    aux_path_data_base = aux_path / 'Output_h5_files'
    
    # Create the main data structure to produce the figure of the paper. 

    data_table_0, surface_data_0, slab_2D_0 = wrapper_merging_h5(aux_path_data_base
                                                                 , partial=(f'k3{args.sh}',args.steady_state)
                                                                 , name_suite=args.name_suite, recreate=False,csv=args.csv)
    data_table_1, _, slab_2D_1 = wrapper_merging_h5(aux_path_data_base
                                                    , partial=(f'k5{args.sh}',args.steady_state)
                                                    , name_suite=args.name_suite, recreate=False,csv=args.csv)
    data_table_2, _, slab_2D_2 = wrapper_merging_h5(aux_path_data_base
                                                    , partial=(f'nl{args.sh}',args.steady_state),
                                                    name_suite=args.name_suite, recreate=False,csv=args.csv)

    slab_2D = (slab_2D_0, slab_2D_1, slab_2D_2)
    
    data_table = pd.concat([data_table_0, data_table_1, data_table_2], ignore_index=True)

    
    path2save = aux_path / f'Result_shear_heating_{args.sh}_{args.name_suite}'
    
    path2save.mkdir(exist_ok=True)
    
    plot_slab_variances(slab_2D, path2save = path2save, number_figure=3)
    
    pres_temp_hp_lt_db = data_base_PT_whitney()

    
    figure_slab_surface(path2save=path2save,
                        pdb=data_table,
                        crd=surface_data_0,
                        petro_db=pres_temp_hp_lt_db,
                        figure_number=0)
    
    plot_heat_flux(path2save,surface_data_0,data_table, 4)

    print(f'Saving picture into {path2save}...')
        

    #plot_slab_versus_thermalpar(path2save=path2save,pdb=data_table,figure_number=1)


if __name__ == '__main__':
    
    plot_sensitivity_study_steady_state()

