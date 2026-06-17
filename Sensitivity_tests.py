from stonedfenicsx.numerical_control import NumericalControls,IOControls 
from stonedfenicsx.create_mesh.aux_create_mesh import Geom_input 
from stonedfenicsx.scal import Scal
from stonedfenicsx.utils import Input,print_ph
from stonedfenicsx.create_mesh.create_mesh import create_mesh
from stonedfenicsx.Stoned_fenicx import fill_geometrical_input
import sys
import numpy as np
from numpy.typing import NDArray
import os
from stonedfenicsx.package_import import *
import argparse
from types import SimpleNamespace
import pandas as pd
from post_process_module.post_process_data import save_test_data, timing_function
import shutil


# Global flag to decide wether or not to remove the results -> debug reason. 
DEBUG = False
def parse_the_argument(args:argparse.Namespace,pathtotable:str)->SimpleNamespace:
    '''Parse the command line arguments and return a SimpleNamespace with the test parameters.'''
    
    # Extract the arguments from the command line
    case_index = args.case_index
    k_arr = args.k
    id_batch = args.id_batch
    id_db = args.id_db
    shear_heating = 'WetQuartzite_disl'
    # If debug flag is set, use default values for the parameters
    if args.debug_defaults:
        case_index = 0
        k_arr = 3.0
        id_batch = 0
        id_db = 0
        option_thermal = 0
        options_shear_heating = 'False'
    else:
        required = ["case_index", "k", "id_batch", "id_db", "option_shear_heating"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join('--' + m for m in missing)}")

        if args.option_shear_heating == 'True': 
            args.option_shear_heating = bool(1)
        else: 
            args.option_shear_heating = bool(0)
        # case index is the index of the case in the csv file, k is the thermal conductivity, id_batch and id_db are used for saving the results in the database.
        case_index = args.case_index
        k_arr = args.k
        # check if k_arr is -10.0, if so, set option_thermal to 1 and k_arr to 0.0, otherwise set option_thermal to 0. This is a way to activate the thermal non linearities in the test.
        if k_arr < 0.0:
            option_thermal = 1
            k_arr = 0.0
        else:
            option_thermal = 0
        
        

        
        id_batch = args.id_batch
        
        id_db = args.id_db

    cases = pd.read_csv(pathtotable)
    
    row = cases.iloc[case_index]


    args_test = SimpleNamespace(
        case_index = case_index,
        id_batch = id_batch,
        id_db = id_db,
        shear_heating=shear_heating,
        vc=float(row["vc"]),
        age=float(row["age"]),
        Tp=float(row["Tp"]),
        k = k_arr,
        option_thermal=option_thermal,
        option_shear_heating = args.option_shear_heating,
        steady_state=int(1),
    )
    return args_test

#-------------------------------------------------------------------------------
@timing_function
def perform_test(args:argparse.Namespace = None, plate:str = 'Tonga')-> tuple[str,str]:
    from stonedfenicsx.utils import parse_input, time_the_time, timing
    from stonedfenicsx.Stoned_fenicx import StonedFenicsx

    path_test = os.path.dirname(os.path.realpath(__file__))
    
    path_input = f"{path_test}/input_tests.yaml"

    inp,Ph = parse_input(path_input)
    # option for the benchmark

    # Create input data - Input is a class populated by default dataset
    # A flag that generate the geometry of the benchmark
    van_keken = 0
    # The input path for saving the results
    inp.path_test = f'{path_test}/{plate}'
    inp.c_age_plate = args.age
    # Velocity of slab 
    inp.v_s = np.array([args.vc,0],dtype=np.float64)
    inp.time_max = 20.0
    inp.decoupling_ctrl = 1
    if args.option_shear_heating:
        inp.model_shear = 'SelfConsistent'
    else: 
        inp.model_shear = 'NoShear'
    # Geometrical input
    inp.cr = 40.0e3   # Overriding crust 
    inp.lc = 0.2   # relative amount of lower crust
    inp.ocr = 6.0e3  # Crustal thickness
    inp.lit_mt = 80e3  # Lithospheric mantle depth 
    inp.lab_d = inp.lit_mt  # depth of the lab 
    inp.decoupling = 80e3  # decoupling depth
    inp.ns_depth = 50e3
    inp.transition = 10e3
    inp.Tmax = args.Tp  # mantle potential temperature
    inp.steady_state = args.steady_state
    inp.tol = 1e-5
    inp.it_max = 30 

    if args.steady_state == 0: 
        inp.tol = 1e-2 
        inp.tol_innerPic = 1e-2 
        inp.it_max = 5
    inp.slab_type = 'Costum'
    #inp.sub_path = f'{path_test}/{plate}_slab.pz'
    
    capacity_nameCC = 'Constant'#'Oceanic_Crust'
    conductivity_nameCC = 'Constant'#'Oceanic_Crust'
    alpha_nameCC = 'Constant'#'Oceanic_Crust'
    density_nameCC = 'PT'#'PT'
    
    inp.phi = 3.0
    
    if args.option_thermal == 1: 
        alpha_nameC = 'Crust'#'Constant'#
        alpha_nameM = 'Mantle'#'Mantle'
        density_nameC = 'PT'#'PT'
        density_nameM = 'PT'#'PT'
        capacity_nameM = 'Bermann_Aranovich_Fo_Fa_0_1'#'Bermann_Aranovich_Fo_Fa_0_1'
        capacity_nameC = 'Oceanic_Crust'#'Oceanic_Crust'
        conductivity_nameM = 'Mantle'#'Mantle'
        conductivity_nameC = 'Oceanic_Crust'#'Oceanic_Crust'
        rho0_M = 3300.0
        rho0_C = 2800.0
        H_r_LC = 0.27e-6
        H_r_UC = 1.0e-6
        H_r_M  = 0*0.06e-6
        radio_flag = 1 
        density_nameCC = 'PT'#'PT'

        print_ph('Thermal Non linearities active')
    elif args.option_thermal ==0 : 
        alpha_nameC = 'Constant'#
        alpha_nameM = 'Constant'
        density_nameC = 'Constant'#'PT'
        density_nameM = 'Constant'#'PT'
        capacity_nameM = 'Constant'#'Bermann_Aranovich_Fo_Fa_0_1'
        capacity_nameC = 'Constant'#'Oceanic_Crust'
        conductivity_nameM = 'Constant'#'Mantle'
        conductivity_nameC = 'Constant'#'Oceanic_Crust'
        rho0_M = 3300.0
        rho0_C = 2800.0
        H_r_LC = 1*0.27e-6
        H_r_UC = 1*1.0e-6 
        H_r_M  = 0*0.06e-6
        radio_flag = 0 
        density_nameCC = 'Constant'#'PT'

        print_ph('Thermal non linearities not active')

    if args.shear_heating != 'False':
        print_ph('Shear heating active')
    
    inp.dislocation_creep_wz = args.shear_heating
        
    name_diffusion = 'Van_Keken_diff'
    name_dislocation = 'Van_Keken_disl'                 

    # Modify the phase with the new data: 
    Ph.subducting_plate_mantle.rho0 = rho0_M
    Ph.subducting_plate_mantle.name_capacity = capacity_nameM
    Ph.subducting_plate_mantle.name_conductivity = conductivity_nameM
    Ph.subducting_plate_mantle.name_alpha = alpha_nameM
    Ph.subducting_plate_mantle.name_density = density_nameM
    Ph.subducting_plate_mantle.radio_flag = radio_flag
    Ph.subducting_plate_mantle.Hr = H_r_M
    Ph.subducting_plate_mantle.k = args.k 


    Ph.oceanic_crust.rho0 = rho0_C
    Ph.oceanic_crust.name_capacity = capacity_nameC
    Ph.oceanic_crust.name_conductivity = conductivity_nameC
    Ph.oceanic_crust.name_alpha = alpha_nameC
    Ph.oceanic_crust.name_density = density_nameC
    Ph.oceanic_crust.radio_flag = radio_flag
    Ph.oceanic_crust.Hr = H_r_LC 
    Ph.oceanic_crust.k = 1.5 

    Ph.wedge_mantle.name_diffusion = name_diffusion
    Ph.wedge_mantle.name_dislocation = name_dislocation
    Ph.wedge_mantle.rho0 = rho0_M
    Ph.wedge_mantle.name_capacity = capacity_nameM 
    Ph.wedge_mantle.name_conductivity = conductivity_nameM
    Ph.wedge_mantle.name_alpha = alpha_nameM
    Ph.wedge_mantle.name_density = density_nameM
    Ph.wedge_mantle.radio_flag = radio_flag
    Ph.wedge_mantle.Hr = H_r_M 
    Ph.wedge_mantle.k = args.k 

    Ph.overriding_mantle.rho0 = rho0_M 
    Ph.overriding_mantle.name_capacity = capacity_nameM
    Ph.overriding_mantle.name_conductivity = conductivity_nameM
    Ph.overriding_mantle.name_alpha = alpha_nameM
    Ph.overriding_mantle.name_density = density_nameM
    Ph.overriding_mantle.radio_flag = radio_flag
    Ph.overriding_mantle.Hr = H_r_M 
    Ph.overriding_mantle.k = args.k

    Ph.overriding_upper_crust.rho0 = rho0_C 
    Ph.overriding_upper_crust.name_capacity = capacity_nameCC
    Ph.overriding_upper_crust.name_conductivity = conductivity_nameCC
    Ph.overriding_upper_crust.name_alpha = alpha_nameCC
    Ph.overriding_upper_crust.name_density = density_nameCC
    Ph.overriding_upper_crust.radio_flag = radio_flag
    Ph.overriding_upper_crust.Hr = H_r_UC 
    Ph.overriding_upper_crust.k = 2.5 

    Ph.overriding_lower_crust.rho0 = rho0_C 
    Ph.overriding_lower_crust.name_capacity = capacity_nameC
    Ph.overriding_lower_crust.name_conductivity = conductivity_nameC
    Ph.overriding_lower_crust.name_alpha = alpha_nameC
    Ph.overriding_lower_crust.name_density = density_nameC
    Ph.overriding_lower_crust.radio_flag = radio_flag
    Ph.overriding_lower_crust.Hr = H_r_LC 
    Ph.overriding_lower_crust.k = 1.5 


    inp.sname = f'{plate}_case{args.case_index}_{args.id_db}'

    # Initialise the input
    inp.van_keken = van_keken
    StonedFenicsx(inp, Ph)
    
    return tuple([inp.path_test,inp.sname])
#-------------------------------------------------------------------------------
@timing_function
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-defaults", action="store_true")
    parser.add_argument("--case_index", type=int)
    parser.add_argument("--k", type=float)
    parser.add_argument("--id_batch", type=int)
    parser.add_argument("--id_db", type=int)
    parser.add_argument("--option_shear_heating",type=str, default='False')

    args_cli = parser.parse_args()
    
    arguments = parse_the_argument(args_cli,pathtotable=f"{os.path.dirname(os.path.realpath(__file__))}/cases_Sensitivity.csv") 


    print(f"Running case {arguments.case_index}")
    print('Case parameters:')
    print(f"  - convergence velocity: {arguments.vc} cm/yr")
    print(f"  - Mantle potential temperature: {arguments.Tp} C")
    print(f"  - Age: {arguments.age} Myr")
    print(f"  - Thermal non linearities: {'Active' if arguments.option_thermal == 1 else 'Not active'}")
    print(f"  - Thermal conductivity: {arguments.k} W/m/K")
    print(f"  - Thermal parameters : {arguments.vc * arguments.age * (1/1e5 * 1e6) * np.sin(np.radians(45))} km")
    print(f"  - Shear heating: {'Active' if arguments.option_shear_heating else 'Not active'}")
    pathtest = perform_test(args=arguments,plate='Sensitivity')
    
    print(f"Test completed for case {arguments.case_index}. Results saved in {os.path.join(pathtest[0], pathtest[1])}")
    
    save_test_data(name_suite='Sensitivity',pathtotest=pathtest,arguments=arguments)
    
    # remove the tests folder 
    if arguments.case_index !=0:
        shutil.rmtree(os.path.join(*pathtest))    
        print(f"Removed {os.path.join(pathtest[0], pathtest[1])} for case {arguments.case_index}")
    
    print(f"Finished case {arguments.case_index}")
#---------------------------------------------------------------------------------    

if __name__ == "__main__":
    main()
    
