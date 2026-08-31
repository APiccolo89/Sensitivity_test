
import argparse
import os
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from mpi4py import MPI
from numpy.typing import NDArray

from post_process_module.post_process_data import (
    save_test_data,
    save_test_data_td_alt,
    timing_function,
)

_DEBUG_ = False

"""
"""
# -- 
dict_td =  {'1':'SS',
         '0':'TD'}
# -- 
def choose_material_properties(k:float)->dict:
    
    materials  = ("mantle", "oceanic_crust", "upper_crust","lower_crust")
    properties = ("alpha", "density", "capacity", "conductivity","radiogenic","k","radiative","rho0",'cp','alpha0')

    # empty skeleton
    phase_properties = {m: {p: None for p in properties} for m in materials}
    
    # Fixed value
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

    if k in (3.1,5.0): 
        
        for m in materials: 
            
            phase_properties[m]['alpha'] = 'Constant'
            phase_properties[m]['capacity'] = 'Constant'
            phase_properties[m]['density'] = 'Constant'
            phase_properties[m]['conductivity'] = 'Constant'
            phase_properties[m]['radiogenic'] = 0.0 
            phase_properties[m]['radiative'] = 0 
        
        phase_properties['mantle']['k'] = k  
        
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
    
    for m in materials: 
        print(f'{m} properties:')
        for p in phase_properties[m]:
            print(f'       .{p} = {phase_properties[m][p]} ')

    return phase_properties
# --- 
def create_case_dictionary()->dict:
    """Small function that creates the dictionary of the cases

    Returns:
        dict: dictionary of cases
    """
    
    
    # Main type
    main_case_dt = np.dtype([
        ("k",             "f8"),
        ("shear_heating", "?"),
    ])

    def make_case(k, shear_heating):
        return np.array((k, shear_heating), dtype=main_case_dt)

    dict_case = {'k3nsh':make_case(3.1,False),
                 'k5nsh':make_case(5.0,False),
                 'nlnsh':make_case(-1.0,False),
                 'k3sh':make_case(3.1,True),
                 'k5sh':make_case(5.0,True),
                 'nlsh':make_case(-1.0,True),
                 'debug':make_case(-1.0,True)}
    return dict_case
# --- 
def print_meta_data_id_db(key_from_submission:str,cases:NDArray)->str:
    
    def thermal_model(k)->str:
        if k in (3.1,5.0): 
            str_aux = f' linear cases with k = {k:d}'
        elif k < 0.0:
            str_aux = 'non linear thermal properties'
        else:
            raise ValueError('There is a problem ... ')
        return str_aux 
    
    def shear_heating(sh:bool)->str: 
        if sh:
            str_aux = 'Shear Heating is acive'
        else: 
            str_aux = 'Shear Heating is not active'
        
        return str_aux     
        
    str_2_print = f"{key_from_submission} => {thermal_model(cases['k'])} and {shear_heating(cases['shear_heating'])}"
    
    return str_2_print
    
    
# Global flag to decide wether or not to remove the results -> debug reason. 
def parse_the_argument(args:argparse.Namespace,pathtotable:str)->SimpleNamespace:
    '''
    '''
    # Extract case: 
    dict_case = create_case_dictionary()
    
    if _DEBUG_:
        case_index = 0 
        case_sim = dict_case['debug']
        mat_prop = choose_material_properties(case_sim['k'])
        steady_state = 1 
        name_suite = 'debug' 
        shear_heating = case_sim['shear_heating']
    else:
        
        required = ["case_index", "case", "id_batch"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join('--' + m for m in missing)}")

        case_index = args.case_index
        case_sim = dict_case[args.case]
        
        print(print_meta_data_id_db(args.case,case_sim))
        
        mat_prop = choose_material_properties(case_sim['k'])
        
        steady_state = args.steady_state
        name_suite = args.name_suite 
        shear_heating = case_sim['shear_heating']


    cases = pd.read_csv(pathtotable)
    
    row = cases.iloc[case_index]
    
    args_test = SimpleNamespace(
        case_index = case_index,
        id_batch = args.id_batch,
        case = args.case,
        shear_heating=shear_heating,
        vc=float(row["vc"]),
        age=float(row["age"]),
        Tp=float(row["Tp"]),
        mat_prop = mat_prop,
        steady_state=args.steady_state,
    )
    return args_test

# ---
@timing_function
def perform_test(args:argparse.Namespace = None,name_suite:str = 'debug')-> tuple[str,str]:
    from stonedfenicsx.config.input_parser import parse_input
    from stonedfenicsx.config.numerical_control import dict_shear_modes
    from stonedfenicsx.stoned_fenicsx import stoned_fenicsx

    path_general = os.path.dirname(os.path.realpath(__file__))
    
    path_input = f"{path_general}/input_tests.yaml"

    inp,Ph = parse_input(path_input)
    if args.case == 'debug':
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


    inp.ctrl_io.path_save                             = os.path.join(path_general,f"{name_suite}_{args.case}_{dict_td[f'{args.steady_state}']}")
    inp.ctrl_io.test_name                             = f'{name_suite}_case{args.case_index}_{args.case}'

    # Initialise the input
    attempt = 0 
    success = 0
    while attempt < 3 and success == 0:
        try:
            stoned_fenicsx(inp, Ph)
            print(f'Test was a success at {attempt}')
            success = 1
        except: 
            time.sleep(20)      # seconds, float allowed
            attempt = attempt + 1
            print('Trying the next time, I need to solve the issue of the cache and waited')
            
    if success==0: 
        raise ValueError('Failed attempt')
    
    return tuple([inp.ctrl_io.path_test,inp.ctrl_io.test_name])
# ---
@timing_function
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-defaults", action="store_true")
    parser.add_argument("--case_index", type=int)
    parser.add_argument("--case", type=str)
    parser.add_argument("--id_batch", type=int)
    parser.add_argument("--name_suite",type=str)
    parser.add_argument("--steady_state",type=int,default=1)

    args_cli = parser.parse_args()
    
    name_suite = args_cli.name_suite
    
    arguments = parse_the_argument(args_cli,pathtotable=f"{os.path.dirname(os.path.realpath(__file__))}/cases_Sensitivity_td.csv") 
        
    pathtest,_ = perform_test(args=arguments,name_suite= name_suite)
 
    print(f"Test completed for case {arguments.case_index}. Results saved in {pathtest}")
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    
    if arguments.steady_state == 0 and rank == 0: 
        save_test_data_td_alt(name_suite=name_suite,pathtotest=pathtest,arguments=arguments)
    else: 
        save_test_data(name_suite=name_suite,pathtotest=pathtest,arguments=arguments)

    
    #remove the tests folder 
    if arguments.case_index !=0:
        shutil.rmtree(os.path.join(pathtest))    
        print(f"Removed {os.path.join(pathtest)} for case {arguments.case_index}")
    
    print(f"Finished case {arguments.case_index}")
#---------------------------------------------------------------------------------    

if __name__ == "__main__":
    main()
    
