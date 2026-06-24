import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
import gc
from collections import namedtuple
from typing import NamedTuple
from numba import njit
import time as timing
import tracemalloc
import cmcrameri
from dataclasses import dataclass

@dataclass(slots=True)
class SlabPlot2D:
    temp_mean_2d:np.ndarray[np.float32]
    temp_var_2d:np.ndarray[np.float32]
    temp_std_2d:np.ndarray[np.float32]
    xi:np.ndarray[np.float32]
    yi:np.ndarray[np.float32]

@dataclass(slots=True)
class CoordinateSurface:
    x_heat_flux:np.ndarray[np.float32]
    z_slab:np.ndarray[np.float32]
    z_ocean:np.ndarray[np.float32]



@njit
def compute_distance(p0:list,p1:list)->float:
    return np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2) 

@njit
def find_distance_isotherm(c_md:np.ndarray,XS:np.ndarray,ell:np.ndarray)->tuple[float,float]:
    """_summary_

    Args:
        x (float): x coordinate of the max depth of the given isotherm
        y (float): y coordinate of the max depth of the given isotherm
        XS (np.ndarray): Slab geometry in 2D (shape: 2 x n_points)
        ell (np.ndarray): 1D array of the arc length along the slab geometry (shape: n_points)

    Returns:
        tuple[float,float]: dist: distance from the point (x,y) to the slab geometry. ell_point: interpolated length of the projection.
    """
    x = c_md[1]
    y = c_md[0]
    # Find the index of the segments containing the projection of the point onto the curve
    ind_x = np.nonzero(x >= XS[0, :])[0][0]
    ind_y = np.nonzero(y >= XS[1, :])[0][0]
    # create the temporary array for the segment  
    distance = np.zeros(ind_y - ind_x - 1)
    # loop over the segments and calculate the distance from the point to each segment
    for j in range(ind_x,ind_y-1):
        # calculate the distance from the point to the segment
        x_m = (XS[0, j]/2 + XS[0, j+1]/2, XS[1, j]/2 + XS[1, j+1]/2)
        distance[j-ind_x] = compute_distance(x_m,[x,y])
    # find the minimum distance and the corresponding segmen
    min_distance = np.nonzero(np.min(distance)>=distance)[0]
    # calculate the distance from the point to the curve
    dx, dy = XS[0,min_distance+1] - XS[0,min_distance], XS[1,min_distance+1] - XS[1,min_distance]
    num  = np.abs(dx * (y - XS[1,min_distance+1]) - dy * (x - XS[0,min_distance+1]))
    dist = num / np.sqrt(dx*dx + dy*dy)      
    seg_len2 = dx*dx + dy*dy
    t = ((x - XS[0, min_distance]) * dx +
         (y - XS[1, min_distance]) * dy) / seg_len2
    t = np.minimum(1.0, np.maximum(0.0, t))
    ell_point = ell[min_distance] + t * (ell[min_distance + 1] - ell[min_distance])
    return dist, ell_point


def interpolate_surfaces_slab(v:np.ndarray,x_or:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """_summary_

    Args:
        v (np.ndarray): array to interpolate
        x_or (np.ndarray): original x_or

    Returns:
        tuple[np.ndarray,np.ndarray]: new values, new coordinates
    """
    
    def interpolate(x:np.ndarray, y:np.ndarray, lim:list)->tuple[np.ndarray,np.ndarray]:
        """Small function that effectively interpolate the data, after 
        reordiering them as a function of y. 

        Args:
            x (np.ndarray): vector to interpolate
            y (np.ndarray): original coordinate system
            lim (list): new coordinate limits 

        Returns:
            tuple[np.ndarray,np.ndarray]: updated variables. 
        """
        x_new = np.linspace(lim[0], lim[1], 200)
        if np.diff(x).min() <= 0:
            x, y = zip(*sorted(zip(x, y)))
    
        return np.interp(x_new, x, y), x_new
    
    nv_interp = np.zeros((200,1),dtype=np.float32)
        
    nv_interp , x_new = interpolate(x_or, v, lim=[np.min(x_or), np.max(x_or)])

    return nv_interp, x_new 
    
def merge_data_base(list_of_partial_files, plate,partial,recreate=True,sh=''):
    output_file = f"Data_Base_{plate}_def_{partial}_{sh}.h5"
    i = 0 
    
    if recreate and os.path.exists(output_file):
        os.remove(output_file)
        print(f"Existing file {output_file} removed.")
    elif not recreate and os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping merge.")
        return "File already exists. Merge skipped."

    
    
    with h5py.File(output_file, "a") as f_out:
        for file in list_of_partial_files:
            

            if not os.path.exists(file):
                print(f"Skipping missing file: {file}")
                continue

            with h5py.File(file, "r") as f_partial:
                if i == 0:
                    T_mean_g = f_partial['mean_lc_tst'][:]
                    T_var_g = f_partial['var_lc_tst'][:]
                    n_g = f_partial['n_lc_tst'][()]
                else: 
                    T_mean_g, T_var_g, n_g = update_Welfrod_single_batch(
                        n_g, T_mean_g, T_var_g,
                        f_partial['n_lc_tst'][()],
                        f_partial['mean_lc_tst'][:],
                        f_partial['var_lc_tst'][:]
                    )
                
                for key in f_partial:

                    #  overwrite if exists
                    if key in f_out:
                        del f_out[key]
                    f_partial.copy(key, f_out)
                
                    
            i = i + 1
            if i % 10 == 0:
                print(f"Processed {i} files")   

        f_out.create_dataset("global_mean", data=T_mean_g)
        f_out.create_dataset("global_var", data=T_var_g/(n_g-1))
        f_out.create_dataset("global_M2", data=T_var_g)
        f_out.create_dataset("global_count", data=n_g)

        print(f"Global mean: {np.nanmean(T_mean_g)}, Global std: {np.sqrt(np.nanmin(T_var_g/(n_g-1)))} {np.sqrt(np.nanmax(T_var_g/(n_g-1)))}, Global count: {n_g}")
        

    return "Data merged successfully"
#---------------------------------------------------------------------------------
@njit
def update_Welfrod_single_batch(n_g:int,mean_g:float,var_g:float,n_l:int,mean_l:np.ndarray,var_l:np.ndarray)->tuple[float,float,int]:
    """_summary_

    Args:
        n_g (int): global count of samples
        mean_g (float): global mean
        var_g (float): variance of the global samples
        n_l (int): local count of samples
        mean_l (np.ndarray): local mean
        var_l (np.ndarray): variance of the local samples

    Returns:
        tuple[float,float,int]: updated global mean, variance, and count
    """
    
    
    
    n_t = n_g + n_l
    delta = mean_l-mean_g 
    mean = mean_g + delta *( n_l / n_t )
    var = var_g + var_l + delta*delta * n_g * n_l / n_t 
    
    return mean,var,n_t
#---------------------------------------------------------------------------------
def create_arrays(Temp:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]: 
    
    T_m = np.zeros_like(Temp)
    T_v = np.zeros_like(Temp)
    
    return T_m, T_v
#---------------------------------------------------------------------------------
def compute_total_length(X_S:np.ndarray)->float:

    return np.cumsum(np.sqrt(np.diff(X_S[0,:])**2 + np.diff(X_S[1,:])**2))

@njit
def interpolate_value_0D(ar:np.ndarray,x:np.ndarray,depth:float,ind:None|int)->float:
    """Interpolate 0D (aka was to lazy to look the interp function)

    Args:
        ar (np.ndarray): targer array
        x (np.ndarray): coordinate
        depth (float): target depth
        ind (None | int): ind of the current array

    Returns:
        float: interpoalted value
    """
    if ind is None: 
       ind=np.nonzero(depth >= x)[0][0]

    
    dv = ar[ind+1]-ar[ind]
    dx = x[ind+1]-x[ind]
    val = ar[ind]+(dv/dx)*(depth-ar[ind])
    
    return val,ind 

#---------------------------------------------------------------------------------
def create_data_frame(data_table:pd.DataFrame, hdf5_file:str)->tuple[pd.DataFrame,namedtuple,namedtuple]:
    def extract_max_depth(f:h5py.File, case_key:str, isotherm:str)->tuple[float,float]:
        x = f[f'{case_key}/x_{isotherm}'][:]
        y = f[f'{case_key}/y_{isotherm}'][:]
        ind = np.where(min(y) == y)[0][0]
        x_min = x[ind]
        return np.asarray([min(y),x_min])
    
 

    """Read the h5file to produce a data frame storing the relevant information for the
    sensitivity test

    Args:
        data_table (pd.DataFrame): initial data table
        hdf5_file (str): path to the h5file

    Returns:
        Data Frame with all the information, named tuple with the coordinate system of the surface, name tuple with the 2D averages of the slab
    """
    rows = []
    ind = None
    
    with h5py.File(hdf5_file, 'r') as f:
        time_A = timing.time()
        for i in f:
            time_C = timing.time()

            if i == 'Xi' or i == 'Yi' :
                print('no')
                if i == 'Xi':
                    xi = (f[i][:,:])
                else:
                    yi = (f[i][:,:])
                    
            elif i in ('global_M2','global_mean','global_var','global_count','mean_lc_tst','var_lc_tst','n_lc_tst'):
                continue
    
            else:
                
                case_id = i
                k = float(f[f'{i}/k'][()])
                vc = float(f[f'{i}/vc'][()])
                Tp = float(f[f'{i}/Tp'][()])
                age = float(f[f'{i}/age'][()])
                thermal_parameter = age * vc * (1/1e5 * 1e6)* np.sin(np.radians(45.0))
                option_thermal = int(f[f'{i}/option_thermal'][()])
                max_dpth150 = extract_max_depth(f, i, '150')
                max_dpth300 = extract_max_depth(f, i, '300')
                max_dpth350 = extract_max_depth(f, i, '350')
                max_dpth400 = extract_max_depth(f, i, '400')
                max_dpth450 = extract_max_depth(f, i, '450')
                max_dpth500 = extract_max_depth(f, i, '500')
                max_dpth550 = extract_max_depth(f, i, '550')
                max_dpth600 = extract_max_depth(f, i, '600')
                max_dpth700 = extract_max_depth(f, i, '700')
                max_dpth800 = extract_max_depth(f, i, '800')

                ell = compute_total_length(np.array(f[f'{i}/X_S']))
                slab_c_150 = find_distance_isotherm(max_dpth150,np.array(f[f'{i}/X_S']),ell)
                slab_c_300 = find_distance_isotherm(max_dpth300,np.array(f[f'{i}/X_S']),ell)
                slab_c_350 = find_distance_isotherm(max_dpth350,np.array(f[f'{i}/X_S']),ell)
                slab_c_400 = find_distance_isotherm(max_dpth400,np.array(f[f'{i}/X_S']),ell)
                slab_c_450 = find_distance_isotherm(max_dpth450,np.array(f[f'{i}/X_S']),ell)
                slab_c_500 = find_distance_isotherm(max_dpth500,np.array(f[f'{i}/X_S']),ell)
                slab_c_550 = find_distance_isotherm(max_dpth550,np.array(f[f'{i}/X_S']),ell)
                slab_c_600 = find_distance_isotherm(max_dpth600,np.array(f[f'{i}/X_S']),ell)
                slab_c_700 = find_distance_isotherm(max_dpth700,np.array(f[f'{i}/X_S']),ell)
                slab_c_800 = find_distance_isotherm(max_dpth800,np.array(f[f'{i}/X_S']),ell)

                # prepare the surface for being integrated into the database
                
                z_slab = np.array(f[f'{i}/X_S'][1,:])
                x_heat = np.array(f[f'{i}/x_top'][:])
                z_ocean = np.array(f[f'{i}/X_O'][1,:])
                
                # Interpolate value into a reduced array
                temp_s,z_slab_n = interpolate_surfaces_slab(np.array(f[f'{i}/T_S'][:]),z_slab)
                pres_s,_ = interpolate_surfaces_slab(np.array(f[f'{i}/P_S'][:]),z_slab)
                temp_o,z_ocean_n = interpolate_surfaces_slab(np.array(f[f'{i}/T_O'][:]),z_ocean)
                pres_o = interpolate_surfaces_slab(np.array(f[f'{i}/P_O'][:]),z_ocean)
                heat_f,x_heat_n = interpolate_surfaces_slab(np.array(f[f'{i}/heat_flux'][:]),x_heat)
                temp_s_dc = np.interp(-80.0,z_slab_n,temp_s)
                pres_s_dc = np.interp(-80.0,z_slab_n,pres_s)
                rows.append({
                    "case_id": case_id,
                    "k": k,
                    "age": age,
                    "vc": vc,
                    "Tp": Tp,
                    "thermal_parameter": thermal_parameter,
                    "option_thermal": option_thermal,
                    "max_dpth150": max_dpth150,
                    "max_dpth300": max_dpth300,
                    "max_dpth400": max_dpth400,
                    "max_dpth450" : max_dpth450,
                    "max_dpth500": max_dpth500,
                    "max_dpth550" : max_dpth550,
                    "max_dpth600": max_dpth600,
                    "max_dpth700": max_dpth700,
                    "max_dpth800": max_dpth800,
                    "slab_c_150" :slab_c_150,
                    "slab_c_300" :slab_c_300,
                    "slab_c_350" :slab_c_350,
                    "slab_c_400" :slab_c_400,
                    "slab_c_450" :slab_c_450,
                    "slab_c_500" :slab_c_500,
                    "slab_c_550" :slab_c_550,
                    "slab_c_600" :slab_c_600,
                    "slab_c_700" :slab_c_700,
                    "slab_c_800" :slab_c_800,
                    'T_S':temp_s,
                    'P_S':pres_s,
                    'T_O':temp_o,
                    'P_O':pres_o,
                    'Q': heat_f,
                    'T_s_dc': temp_s_dc,
                    'P_s_dc': pres_s_dc,
                })        
        slab_plot2D =  SlabPlot2D(temp_mean_2d=f['global_mean'][:],
                                  temp_var_2d=f['global_var'][:],
                                  temp_std_2d=np.sqrt(f['global_var'][:]),
                                  xi=xi,
                                  yi=yi)
        coordinate_surfaces = CoordinateSurface(x_heat_flux=x_heat_n,z_slab=z_slab_n,z_ocean=z_ocean_n)

    rows_df = pd.DataFrame(rows)
    data_table = pd.concat([data_table, rows_df], ignore_index=True)            
    
    return data_table, coordinate_surfaces, slab_plot2D
#----------------------- ------------------------------------------------------------------------------------------ ------------------ #
def wrapper_merging_h5(aux_path:str, partial:int, plate:str,sh='' ,recreate=False):

    if recreate or os.path.isfile(f'Data_Base_{plate}_def_{partial}_{sh}.h5')==False:
        aux_path = os.path.join(aux_path, f'Sensitivity_h5_files_{partial}')
        list_partial_files = []
        for i in range(100):
            list_partial_files.append(f'{aux_path}/Data_Base_{i}.h5')
    
        print(merge_data_base(list_partial_files, plate, partial, recreate=recreate,sh=sh))
    else: 
        print('Data Base is already there, and there is no need to force its recreation')
    
    data_table = pd.DataFrame({
        "case_id": [],
        "k": [],
        "age": [],
        "vc": [],
        "Tp": [],
        "option_thermal": [],
        'max_dpth150': [],
        'max_dpth300': [],
        'max_dpth450': [],
        'max_dpth500': [],
        'max_dpth600': [],
        'max_dpth700': [],
        'max_dpth800': [],
        'slab_c_150': [],
        'slab_c_300': [],
        'slab_c_450': [],
        'slab_c_500': [],
        'slab_c_550': [],
        'slab_c_600': [],
        'slab_c_700': [],
        'slab_c_800': [],
        'T_S':[],
        'P_S':[],
        'T_O':[],
        'P_O':[],
        'Q':[],
        'T_s_dc':[],
        'P_s_dc':[]
    })
    
    data_table, coordinate_system, slab_2D = create_data_frame(data_table,   f'Data_Base_{plate}_def_{partial}_{sh}.h5')
    
    return data_table, coordinate_system, slab_2D 
    
    
    
    
    
    