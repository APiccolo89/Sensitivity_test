
import sys
import numpy as np
from numpy.typing import NDArray
import os
import argparse
from types import SimpleNamespace
import pandas as pd
from post_process_module.post_process_data import save_test_data, timing_function, save_test_data_td
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
    
    shear_heating = 'Constant'
    
    tau_min = None

    # If debug flag is set, use default values for the parameters
    if args.debug_defaults:
        case_index = 0
        k_arr = 3.0
        id_batch = 0
        id_db = 0
        option_thermal = 0
        options_shear_heating = 'False'
    else:
        required = ["case_index", "k", "id_batch", "id_db", "option_shear_heating","option_viscosity_weak_zone"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join('--' + m for m in missing)}")

        if args.option_shear_heating == 'True': 
            args.option_shear_heating = bool(1)
        else: 
            args.option_shear_heating = bool(0)
        if args.option_viscosity_weak_zone == 'Dislocation_Law':
            args.shear_heating = 'Wet_Quartzite_2001_Dislocation_creep'
        elif args.option_viscosity_weak_zone == 'Constant':
            args.shear_heating = 'Constant'
        else: 
            raise ValueError('Wrong viscosity law for the weak zone')
        
        
            
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
        shear_heating=args.shear_heating,
        vc=float(row["vc"]),
        age=float(row["age"]),
        Tp=float(row["Tp"]),
        k = k_arr,
        phi=5.0,
        option_thermal=option_thermal,
        option_shear_heating = args.option_shear_heating,
        tau_min = 10e6,
        steady_state=args.steady_state,
    )
    return args_test

#-------------------------------------------------------------------------------
@timing_function
def perform_test(args:argparse.Namespace = None, plate:str = 'Tonga')-> tuple[str,str]:
    from stonedfenicsx.config.input_parser import parse_input
    from stonedfenicsx.config.numerical_control import dict_shear_modes
    from stonedfenicsx.stoned_fenicsx import stoned_fenicsx

    path_general = os.path.dirname(os.path.realpath(__file__))
    
    path_input = f"{path_general}/input_tests.yaml"

    inp,Ph = parse_input(path_input)
    # controls
    inp.ctrl.decoupling_ctrl = 1
    inp.ctrl.steady_state = args.steady_state
    if args.steady_state == 1: 
        inp.ctrl.tol = 1e-4
        inp.ctrl.it_max = 50

    else: 
        inp.ctrl.tol = 1e-3
        inp.ctrl.it_max = 5

    
    if args.option_shear_heating:
        inp.ctrl.model_shear = 'SelfConsistent'
    else: 
        inp.ctrl.model_shear = 'NoShear'



    # Create input data - Input is a class populated by default dataset
    # A flag that generate the geometry of the benchmark
    # The input path for saving the results
    inp.ctrl_tbc.slab_age = args.age
    inp.ctrl_tbc.temp_max = args.Tp
    # Velocity of slab 
    inp.ctrl_ky.v_s = np.array([args.vc,0],dtype=np.float64)

    # Geometrical input
    inp.g_input.cr = 40.0   # Overriding crust 
    inp.g_input.lc = 0.5   # relative amount of lower crust
    inp.g_input.ocr = 6.0  # Crustal thickness
    inp.g_input.decoupling = 80  # Lithospheric mantle depth 
    inp.g_input.lab_d = inp.g_input.decoupling  # depth of the lab 
    inp.g_input.decoupling = 80 # decoupling depth
    inp.g_input.ns_depth = 50
    inp.g_input.transition = 10
    inp.g_input.slab_type = 'CustomRibe'
    
    if args.option_thermal == 1: 
        alpha_nameC = 'Oceanic_crust'
        alpha_nameM = 'Mantle'
        density_nameC = 'PT'
        density_nameM = 'PT'
        capacity_nameM = 'Mantle_Bernard_Ar_199x_FO_FA'
        capacity_nameC = 'Oceanic_crust'
        conductivity_nameM = 'Mantle_Richards_2018'
        conductivity_nameC = 'Crust_Richards_2018'
        conductivity_nameC = 'Constant'
        capacity_nameCC = 'Oceanic_crust'
        conductivity_nameCC = 'Crust_Richards_2018'
        alpha_nameCC = 'Oceanic_crust'
        density_nameCC = 'PT'
        rho0_M = 3300.0
        rho0_C = 2800.0
        H_r_LC = 0.27e-6
        H_r_UC = 1.0e-6
        H_r_M  = 0*0.06e-6
        radiative_conductivity = 1 
        density_nameCC = 'PT'

    elif args.option_thermal ==0 : 
        alpha_nameC = 'Constant'
        alpha_nameM = 'Constant'
        density_nameC = 'Constant'
        density_nameM = 'Constant'
        capacity_nameM = 'Constant'
        capacity_nameC = 'Constant'
        conductivity_nameM = 'Constant'
        conductivity_nameC = 'Constant'
        capacity_nameCC = 'Constant'
        conductivity_nameCC = 'Constant'
        alpha_nameCC = 'Constant'
        density_nameCC = 'Constant'

        rho0_M = 3300.0
        rho0_C = 2800.0
        H_r_LC = 1*0.27e-6
        H_r_UC = 1*1.0e-6 
        H_r_M  = 0*0.06e-6
        radiative_conductivity = 0 


    if args.shear_heating != 'False':
        Ph.shear_heating_disl_law = args.shear_heating
    Ph.shear_heating_disl_phi = args.phi
        
    name_diffusion = 'VK_Diffusion_creep'
    name_dislocation = 'VK_Dislocation_creep'                 

    # Modify the phase with the new data: 
    Ph.subducting_plate_mantle.rho0 = rho0_M
    Ph.subducting_plate_mantle.name_capacity = capacity_nameM
    Ph.subducting_plate_mantle.name_conductivity = conductivity_nameM
    Ph.subducting_plate_mantle.name_alpha = alpha_nameM
    Ph.subducting_plate_mantle.name_density = density_nameM
    Ph.subducting_plate_mantle.radiative_conductivity = radiative_conductivity
    Ph.subducting_plate_mantle.radiogenic_heat = H_r_M
    Ph.subducting_plate_mantle.k = args.k 


    Ph.oceanic_crust.rho0 = rho0_C
    Ph.oceanic_crust.name_capacity = capacity_nameC
    Ph.oceanic_crust.name_conductivity = conductivity_nameC
    Ph.oceanic_crust.name_alpha = alpha_nameC
    Ph.oceanic_crust.name_density = density_nameC
    Ph.oceanic_crust.radiative_conductivity = radiative_conductivity
    Ph.oceanic_crust.radiogenic_heat = H_r_LC 
    Ph.oceanic_crust.k = 1.5 

    Ph.wedge_mantle.name_diffusion = name_diffusion
    Ph.wedge_mantle.name_dislocation = name_dislocation
    Ph.wedge_mantle.rho0 = rho0_M
    Ph.wedge_mantle.name_capacity = capacity_nameM 
    Ph.wedge_mantle.name_conductivity = conductivity_nameM
    Ph.wedge_mantle.name_alpha = alpha_nameM
    Ph.wedge_mantle.name_density = density_nameM
    Ph.wedge_mantle.radiative_conductivity = radiative_conductivity
    Ph.wedge_mantle.radiogenic_heat = H_r_M 
    Ph.wedge_mantle.k = args.k 

    Ph.overriding_mantle.rho0 = rho0_M 
    Ph.overriding_mantle.name_capacity = capacity_nameM
    Ph.overriding_mantle.name_conductivity = conductivity_nameM
    Ph.overriding_mantle.name_alpha = alpha_nameM
    Ph.overriding_mantle.name_density = density_nameM
    Ph.overriding_mantle.radiative_conductivity = radiative_conductivity
    Ph.overriding_mantle.radiogenic_heat = H_r_M 
    Ph.overriding_mantle.k = args.k

    Ph.overriding_upper_crust.rho0 = rho0_C 
    Ph.overriding_upper_crust.name_capacity = capacity_nameCC
    Ph.overriding_upper_crust.name_conductivity = conductivity_nameCC
    Ph.overriding_upper_crust.name_alpha = alpha_nameCC
    Ph.overriding_upper_crust.name_density = density_nameCC
    Ph.overriding_upper_crust.radiative_conductivity = radiative_conductivity
    Ph.overriding_upper_crust.radiogenic_heat = H_r_UC 
    Ph.overriding_upper_crust.k = 2.5 

    Ph.overriding_lower_crust.rho0 = rho0_C 
    Ph.overriding_lower_crust.name_capacity = capacity_nameC
    Ph.overriding_lower_crust.name_conductivity = conductivity_nameC
    Ph.overriding_lower_crust.name_alpha = alpha_nameC
    Ph.overriding_lower_crust.name_density = density_nameC
    Ph.overriding_lower_crust.radiative_conductivity = radiative_conductivity
    Ph.overriding_lower_crust.radiogenic_heat = H_r_LC 
    Ph.overriding_lower_crust.k = 1.5 


    inp.ctrl_io.path_save = os.path.join(path_general,f'Sensitivity{args.id_db}_TD')
    inp.ctrl_io.test_name = f'{plate}_case{args.case_index}_{args.id_db}'

    # Initialise the input
    stoned_fenicsx(inp, Ph)
    
    return tuple([inp.ctrl_io.path_test,inp.ctrl_io.test_name])
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
    parser.add_argument("--option_viscosity_weak_zone",type=str,default='Dislocation_Law')
    parser.add_argument("--steady_state",type=int,default=1)


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

        
    pathtest,_ = perform_test(args=arguments,plate='Sensitivity')
 
    print(f"Test completed for case {arguments.case_index}. Results saved in {pathtest}")
    
    if arguments.steady_state == 0: 
        save_test_data_td(name_suite='Sensitivity_td',pathtotest=pathtest,arguments=arguments)
    else: 
        save_test_data(name_suite='Sensitivity_td',pathtotest=pathtest,arguments=arguments)

    
    #remove the tests folder 
    if arguments.case_index !=0:
        shutil.rmtree(os.path.join(pathtest))    
        print(f"Removed {os.path.join(pathtest)} for case {arguments.case_index}")
    
    print(f"Finished case {arguments.case_index}")
#---------------------------------------------------------------------------------    

if __name__ == "__main__":
    main()
    
