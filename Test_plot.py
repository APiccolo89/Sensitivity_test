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
from pathlib import Path

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

    aux_path =Path(__file__).resolve().parent

    sh = 'yes'
    if sh == 'no':
        partials = [0,1,2]
    else:
        partials = [3,4,5]
    data_table_0, surface_data_0, slab_2D_0 = wrapper_merging_h5(aux_path, partial=partials[0], plate=plate, recreate=False,sh=sh)
    data_table_1, _, slab_2D_1 = wrapper_merging_h5(aux_path, partial=partials[1], plate=plate, recreate=False,sh=sh)
    data_table_2, _, slab_2D_2 = wrapper_merging_h5(aux_path, partial=partials[2], plate=plate, recreate=False,sh=sh)

    slab_2D = (slab_2D_0, slab_2D_1, slab_2D_2)
    
    data_table = pd.concat([data_table_0, data_table_1, data_table_2], ignore_index=True)

    
    path2save = aux_path / f'Result_shear_heating_{sh}'
    path2save.mkdir(exist_ok=True)
    plot_slab_variances(slab_2D, path2save = path2save, number_figure=3)

    print(f'Saving picture into {path2save}...')
        
    pres_temp_hp_lt_db = data_base_PT_whitney()

    plot_slab_versus_thermalpar(path2save=path2save,pdb=data_table,figure_number=1)

    figure_slab_surface(path2save=path2save,
                        pdb=data_table,
                        crd=surface_data_0,
                        petro_db=pres_temp_hp_lt_db,
                        figure_number=0)

main()

