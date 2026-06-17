import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
import gc
from collections import namedtuple
from numba import njit
import time as timing
import tracemalloc
import cmcrameri
from post_process_module.process_data_visualisation import wrapper_merging_h5
from post_process_module.figure_plot import *

# ----------------------- Main code ------------------ #
def main():
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


    plate = 'Sensitivity'

    print("Merging partial data files...")

    aux_path = f'/mnt/scratch/wlnw570/Japan_ensemble/Sensitivity_test'

    data_table_0, surface_data_0, slab_2D_0 = wrapper_merging_h5(aux_path, partial=0, plate=plate, recreate=False,sh='yes')
    data_table_1, surface_data_1, slab_2D_1 = wrapper_merging_h5(aux_path, partial=1, plate=plate, recreate=False,sh='yes')
    data_table_2, surface_data_2, slab_2D_2 = wrapper_merging_h5(aux_path, partial=2, plate=plate, recreate=False,sh='yes')

    slab_2D = (slab_2D_0, slab_2D_1, slab_2D_2)
    
    data_table = pd.concat([data_table_0, data_table_1, data_table_2], ignore_index=True)


    
    path2save = f'/mnt/scratch/wlnw570/Japan_ensemble/Sensitivity_test/Result_{plate}_SH'    
    if not os.path.isdir(path2save):
        os.mkdir(path2save)
        print(f'Created the folder {path2save}')

    print(f'Saving picture into {path2save}...')
    

    #plot_slab_variances(slab_2D_alt,path2save,name_figure='Figure0',name='T')
    
    DB = data_base_PT_whitney()

    plot_slab_surface(path2save=path2save, surface_data=surface_data, surface='slab', DB=DB,name_figure='Figure0',name='T')


    plot_slab_versus_thermalpar(path2save=path2save, DB=data_table,scale=True,name='T',name_figure='figure1')

    
    
    # Figure 1 suite: the mean of top surface of the slab and ocean 
    #plot_slab_surface(path2save=path2save, surface_data=surface_data, surface='ocean', DB=DB,name_figure='Figure1_b',name='PT')


    plot_distance_isotherm_seismic(path2save=path2save, DB=data_table,scale=True,name='PT',name_figure='figure3')
    #plot_distance_isotherm_seismic(path2save=path2save, DB=data_table_alt,scale=True,name='T',name_figure='figure3')

        


    # produce the data table
    #---------------------------------------------------------
    # Create figure for the histograms and scatter plots.
    # Plot heat fluxes
    plot_heat_flux(path2save=path2save, surface_data=surface_data, name_figure='Figure6',name='PT')


main()

