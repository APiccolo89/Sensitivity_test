from pathlib import Path

import pandas as pd

from post_process_module.figure_plot import *
from post_process_module.post_process_data import timing_function
from post_process_module.process_data_visualisation import wrapper_merging_h5

pres_temp_hp_lt_db = data_base_PT_whitney()

# --- 
@timing_function    
def create_data_structure(aux_path:str, args):
    print("Merging partial data files...")

    data_table_0, surface_data_0, slab_2D_0 = wrapper_merging_h5(aux_path
                                                                 , partial=(f'k3{args.sh}',args.steady_state)
                                                                 , name_suite=args.name_suite, recreate=False,csv=args.csv)
    data_table_1, _, slab_2D_1 = wrapper_merging_h5(aux_path
                                                    , partial=(f'k5{args.sh}',args.steady_state)
                                                    , name_suite=args.name_suite, recreate=False,csv=args.csv)
    data_table_2, _, slab_2D_2 = wrapper_merging_h5(aux_path
                                                    , partial=(f'nl{args.sh}',args.steady_state),
                                                    name_suite=args.name_suite, recreate=False,csv=args.csv)

    slab_2D = (slab_2D_0, slab_2D_1, slab_2D_2)
    
    data_table = pd.concat([data_table_0, data_table_1, data_table_2], ignore_index=True)  
    
    return data_table, slab_2D, surface_data_0
#---
def plot_sensitivity_study_steady_state(args)->None:
    """ Small top-level function that calls the data structure generation routine
    and produce the figure of the manuscript. 
    The following function plots the following figures:
    a. 2D plot regarding the average difference between the thermal model
    b. 1D summary plot regarding the top surface of the slab
    c. Heat flux of the numerical simulation and relative errors with respect to the non-linear model
    d. Maximum depth of the isotherm of all the experiment as a function of the thermal parameter and error measurment wrt the nonlinear model

    Args:
        args: parsed argument
    Returns: 
        None: 
    """

    check_parsed_arguments(args)
    
    aux_path = Path(__file__).resolve().parent
    
    aux_path_data_base = aux_path / 'Output_h5_files'
    
    # Create the main data structure to produce the figure of the paper. 

    data_table, slab_2D, surface_data_0 = create_data_structure(aux_path=aux_path_data_base,args=args)
    # Generate path 
    path2save = aux_path / f'Result_shear_heating_{args.sh}_{args.name_suite}'
    # Create folder
    path2save.mkdir(exist_ok=True)
    # Plot mean difference
    plot_slab_variances(slab_2D, path2save = path2save, number_figure=3)
    # Create the database
    # Plot the 1D (P-T) of the slab surface
    figure_slab_surface(path2save=path2save,
                        pdb=data_table,
                        crd=surface_data_0,
                        petro_db=pres_temp_hp_lt_db,
                        figure_number=0)
    # Plot the heat flux 
    plot_heat_flux(path2save,surface_data_0,data_table, 4)

    print(f'Saving picture into {path2save}...')
    # Plot the main 
    plot_slab_versus_thermalpar(path2save=path2save,pdb=data_table,figure_number=1)
# --- 
def plot_slab_surfaces_japan_mexico_sensitivity(): 
    """_summary_

    Args:
        args (_type_): _description_
    """
    from post_process_module.figure_plot import plot_eye_candy
        
    aux_path = Path(__file__).resolve().parent
    
    aux_path_data_base = aux_path / 'Output_h5_files'
    
    # Create the main data structure to produce the figure of the paper. 
    path2save = aux_path / 'Slab_surface_plot'
    path2save.mkdir(parents=True,exist_ok=True)

    data_table_0, surface_data_0, _ = wrapper_merging_h5(aux_path
                                                                 , partial=('nlsh','SS')
                                                                 , name_suite='steady_s_sensitivity', recreate=False,csv='cases_Sensitivity.csv')
    
    data_table_1, surface_data_1, _ = wrapper_merging_h5(aux_path
                                                    , partial=('nlsh','SS')
                                                    , name_suite='Mexico', recreate=False,csv='cases_Sensitivity_Mexico.csv')
    
    data_table_2, surface_data_2, _ = wrapper_merging_h5(aux_path
                                                    , partial=('nlsh','SS'),
                                                    name_suite='Japan', recreate=False,csv='cases_Sensitivity_Japan.csv')

    plot_eye_candy(data_table_0
                   ,data_table_1
                   ,data_table_2
                   ,data_base_PT_whitney()
                   ,surface_data_0,
                   surface_data_1
                   ,surface_data_2
                   ,path2save)


# --- 
def main():
    plot_slab_surfaces_japan_mexico_sensitivity()

# --- 
if __name__ =='__main__':
    main()
    
    
    
    


