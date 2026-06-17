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
import matplotlib.axes as mpl_axes
from .process_data_visualisation import SlabPlot2D,CoordinateSurface

from dataclasses import dataclass
from typing import Optional

# Conversion pt - cm 
_PT_CM_ = 0.035
# Ticks, label and text font size 
_TICKSFONTSIZE_ = 0.8/_PT_CM_
_LABELFONTSIZE_ = 1.0/_PT_CM_
_TEXTFONTSIZE_ = 0.7/_PT_CM_


def create_common_data_set(a:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Function that creates common datasets: 
    -> Time to time simulations fails,so it is necessary to check how many experiments fails,
    select the common successfull data

    Args:
        a (pd.DataFrame): main database

    Returns:
        tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]: filtered database
    """

    
    keys = ["Tp", "age", "vc"]
    
    a = DB[(DB['option_thermal'] == 0) & (DB['k'] == 3.1)]
    b = DB[(DB['option_thermal'] == 0) & (DB['k'] == 5.0)]
    c = DB[(DB['option_thermal'] > 0)]    
    
    common_keys = (a[keys]
    .merge(b[keys], on=keys, how="inner")
    .merge(c[keys], on=keys, how="inner")
    .drop_duplicates()
    .sort_values(keys)
    .reset_index(drop=True))
    
    a_common = (a.merge(common_keys, on=keys, how="inner")).sort_values(keys).reset_index(drop=True)
    b_common = (b.merge(common_keys, on=keys, how="inner")).sort_values(keys).reset_index(drop=True)
    c_common = (c.merge(common_keys, on=keys, how="inner")).sort_values(keys).reset_index(drop=True)

    return a_common,b_common,c_common 

class data_base_PT_whitney():
    def __init__(self):
        self.T = [300, 550, 590, 400, 570, 610, 635, 540, 585, 510,
                   585, 590, 640, 625, 555, 595, 645, 500, 300, 450,
                   532, 350, 368, 415, 400, 320, 265, 465, 682, 425,
                   415, 510, 300, 350, 385, 300, 490, 300, 500, 335,
                   345, 360, 510, 545, 650, 570, 540, 380, 553, 500,
                   500, 570, 550, 600, 657, 600, 556, 775, 750, 605,
                   615, 570, 520, 730, 585, 420, 580, 640, 700, 750,
                   445, 360, 380, 375, 550, 545, 520, 690, 780, 660,
                   695, 580, 690, 375, 450, 615, 375, 320, 300, 320,
                   450, 360, 455, 600, 575, 575, 450, 530, 378, 420,
                   375, 480, 350, 400, 425, 438, 350, 500, 500, 350,
                   425, 320, 585, 370, 275, 530, 520, 560, 475, 495,
                   470, 470, 540, 580, 500, 475, 450, 500, 560, 530,
                   300, 550, 320, 400, 705, 350, 250, 520, 235, 450,
                   480, 350, 340, 325, 560, 638, 470, 520, 580, 350,
                   375, 350, 585, 600, 600, 420, 500, 525, 500, 360,
                   375, 525, 550, 490, 575, 560, 545, 400, 550, 450,
                   580, 550, 650, 550, 470, 350, 328, 330, 575, 670,
                   600, 300, 377, 440, 275, 515, 280, 520, 535, 550,
                   255, 510, 575, 550, 690, 550, 300, 750, 250, 410,
                   450, 495, 525, 545, 340, 580, 600, 400, 570, 600,
                   400, 475, 565, 610, 300, 425, 590, 390, 730, 480,
                   200, 555, 625, 570, 345, 420, 360, 430, 380, 650,
                   490, 375, 360, 400, 400, 375, 450, 505, 350, 350,
                   425, 450, 584, 540, 575, 490, 300, 560, 450, 510,
                   310, 625, 375, 510, 490, 410, 420, 400, 375, 350,
                   485, 520, 490, 520, 570, 460, 500, 600, 490, 380,
                   422, 555, 420, 520, 200, 450, 430, 300, 200, 290,
                   320, 415, 445, 500, 350, 190, 460, 350, 275, 360,
                   550, 550, 403, 370, 250, 620, 270, 250, 275, 450,
                   550, 650, 525, 485, 640, 450, 338, 350, 550]
        self.P =[0.8, 1.2, 2.7, 0.5, 1.7, 2.1, 2.2, 2.1, 2.0, 2.1,
               2.4, 2.2, 2.2, 2.0, 2.6, 2.5, 2.1, 2.1, 1.0, 2.5,
               1.2, 0.5, 1.0, 1.2, 0.8, 0.7, 0.5, 0.9, 1.4, 1.2,
               1.2, 1.2, 0.6, 0.6, 1.0, 0.7, 1.9, 0.9, 2.4, 0.8,
               0.9, 0.8, 2.4, 1.9, 2.1, 2.1, 1.8, 1.7, 2.2, 2.4,
               2.6, 1.5, 1.0, 1.6, 1.5, 2.0, 1.2, 1.7, 1.5, 1.8,
               1.3, 2.0, 1.7, 1.8, 2.3, 1.2, 1.9, 1.8, 3.0, 2.3,
               1.5, 1.7, 1.7, 1.7, 1.3, 1.5, 1.8, 2.1, 2.5, 2.4,
               2.3, 2.2, 2.1, 1.1, 1.3, 2.3, 1.0, 0.8, 0.6, 0.8,
               1.3, 1.6, 1.0, 1.5, 1.7, 2.0, 1.9, 2.2, 2.3, 1.0,
               1.5, 2.3, 1.0, 1.0, 1.0, 1.3, 0.8, 1.9, 1.5, 1.3,
               1.2, 0.9, 2.2, 0.9, 0.7, 1.9, 1.9, 2.2, 2.0, 2.0,
               1.5, 1.5, 2.0, 2.0, 1.5, 1.6, 1.5, 1.6, 2.2, 2.0,
               1.4, 2.2, 0.9, 2.3, 2.3, 1.2, 0.8, 2.5, 1.1, 2.1,
               2.6, 0.9, 1.2, 1.0, 2.3, 2.7, 1.0, 2.1, 1.8, 0.9,
               0.7, 1.1, 2.3, 1.8, 2.2, 1.6, 1.7, 1.1, 1.8, 1.0,
               0.9, 1.3, 2.4, 1.9, 2.2, 2.6, 2.0, 2.0, 2.6, 1.2,
               1.9, 2.7, 2.2, 1.9, 2.4, 1.0, 0.6, 0.7, 2.0, 2.0,
               2.0, 0.8, 0.9, 0.8, 0.9, 1.2, 0.7, 0.9, 1.8, 1.8,
               0.4, 1.7, 2.2, 2.4, 2.2, 1.2, 0.7, 1.6, 0.6, 1.4,
               1.6, 1.7, 1.9, 2.3, 0.8, 2.4, 1.9, 0.8, 1.3, 2.4,
               1.0, 1.5, 1.5, 1.7, 0.5, 1.2, 1.6, 1.6, 2.0, 1.6,
               0.5, 1.3, 1.7, 1.7, 0.6, 1.2, 0.8, 1.1, 0.9, 2.4,
               1.0, 1.2, 1.2, 1.3, 1.4, 1.7, 0.6, 2.4, 1.0, 0.7,
               1.3, 1.3, 1.8, 2.3, 2.8, 1.1, 1.0, 1.6, 1.2, 1.4,
               1.0, 2.0, 1.2, 1.2, 1.7, 1.3, 1.2, 1.4, 1.3, 1.2,
               1.3, 1.1, 1.1, 1.3, 1.8, 2.4, 2.6, 1.8, 1.6, 1.2,
               2.2, 2.3, 2.1, 2.6, 0.8, 1.1, 2.4, 1.2, 0.6, 0.8,
               0.7, 0.8, 0.9, 0.7, 0.8, 0.3, 1.2, 0.3, 0.7, 0.6,
               1.5, 2.0, 1.1, 0.7, 0.8, 2.5, 1.0, 0.5, 0.9, 1.1,
               1.1, 1.2, 0.6, 1.8, 1.9, 0.6, 0.7, 0.8, 1.2]

# latex Font 
TRANSPARENT = True 
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",   # Computer Modern
})

# General function to change the axis behaviour
def modify_axes(ax: mpl_axes = None,
                label_x: Optional[list] = ['n', 'n'],
                lim_x: Optional[list] = None,
                lim_y: Optional[list] = None,
                label_y: Optional[list] = ['n', 'n'],
                ticks_x: Optional[list] = None,
                ticks_y: Optional[list] = None,
                isotherms: Optional[int] = 0) -> mpl_axes:
    """Function that modifies the axes.
    Args:
        ax (mpl_axes, optional): current axis. Defaults to None.
        label_x (list, optional): [label, show_ticklabels ('y'/'n')]. Defaults to ['n','n']. -> in case of the label name = string to use as label
        lim_x (list, optional): limits of the x-axis. Defaults to None.
        lim_y (list, optional): limits of the y-axis. Defaults to None.
        label_y (list, optional): [label, show_ticklabels ('y'/'n')]. Defaults to ['n','n']. -> in case of the label name = string to use as label
        ticks_x (list, optional): x tick positions. Defaults to None.
        ticks_y (list, optional): y tick positions. Defaults to None.
        isotherms (int, optional): isotherm option. Defaults to 0.
    Returns:
        mpl_axes: modified axis
    """
    # --- spine and grid styling ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='both', colors='black', direction='in', length=5, width=2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.grid(visible=True, which='major', axis='both', color='gray', linestyle='--', linewidth=0.3, alpha=0.7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=_TICKSFONTSIZE_)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=_TICKSFONTSIZE_)

    # --- x-axis ---
    if label_x[0] != 'n':
        ax.set_xlabel(label_x[0], fontsize=_LABELFONTSIZE_, labelpad=10)
        ax.xaxis.set_label_position('top')

    if lim_x is not None:
        ax.set_xticks([lim_x[0], lim_x[1]])

    if ticks_x is not None:
        ax.set_xticks(ticks_x)
        if label_x[1] == 'y':
            ax.set_xticklabels([f"{i:.1f}" for i in ticks_x])
            ax.set_xlim(lim_x[0] - 0.1*(lim_x[1]-lim_x[0]),
                        lim_x[1] + 0.1*(lim_x[1]-lim_x[0]))
        else:
            ax.set_xticklabels([])

    # --- y-axis ---
    if label_y[0] != 'n':
        ax.set_ylabel(label_y[0], fontsize=_LABELFONTSIZE_, labelpad=10)  # fixed: was label_x[0]
        ax.yaxis.set_label_position('left')                                # fixed: was 'top'

    if lim_y is not None:                                                  # fixed: was lim_x
        ax.set_yticks([lim_y[0], lim_y[1]])

    if ticks_y is not None:
        ax.set_yticks(ticks_y)
        if label_y[1] == 'y':
            ax.set_yticklabels([f"{i:.1f}" for i in ticks_y])             # fixed: was ticks_x
            ax.set_ylim(lim_y[0] - 0.1*(lim_y[1]-lim_y[0]),              # fixed: was lim_x
                        lim_y[1] + 0.1*(lim_y[1]-lim_y[0]))
        else:
            ax.set_yticklabels([])

    return ax





def plot_slab_variances(A, path2save: str,name_figure: str,name: str, DBa:None) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    name_plot = f'Figure_{name_figure}_{name}.png'
    
    def extract_limits(k3, k5, NL):
        l_m = []
        l_M = []

        l_m.append(np.nanmin(k3.T_mean - NL.T_mean))
        l_M.append(np.nanmax(k3.T_mean - NL.T_mean))

        l_m.append(np.nanmin(k5.T_mean - NL.T_mean))
        l_M.append(np.nanmax(k5.T_mean - NL.T_mean))

        return l_m, l_M

    fig = plt.figure(figsize=(14, 6))

    k3,k5,NL = A 
    
    Xi,Yi = k3.Xi,k3.Yi
    
    
    
    bx = 0.1
    by = 0.2
    sx = 0.38
    sy = 0.68
    dx = 0.08

    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx + sx + dx, by, sx, sy])

    l_m, l_M = extract_limits(k3,k5,NL)

    levels0 = np.linspace(np.floor(l_m[0]), np.ceil(l_M[0]), 10)
    levels1 = np.linspace(np.floor(l_m[1]), np.ceil(l_M[1]), 10)
    alpha = 1.0
    if DBa is not None:
        alpha = 0.1
        z300 = DBa.loc[(DBa['option_thermal'] == 1), f'max_dpth{300}']
        z500 = DBa.loc[(DBa['option_thermal'] == 1), f'max_dpth{500}']
        z600 = DBa.loc[(DBa['option_thermal'] == 1), f'max_dpth{600}']
        x300 = DBa.loc[(DBa['option_thermal'] == 1), f'x_dpth{300}']
        x500 = DBa.loc[(DBa['option_thermal'] == 1), f'x_dpth{500}']
        x600 = DBa.loc[(DBa['option_thermal'] == 1), f'x_dpth{600}']

    a = ax0.contourf(k3.Xi, k3.Yi, k3.T_mean - NL.T_mean, levels=levels0, cmap='cmc.roma',alpha=alpha)
    if alpha != 1.0:

        
        b = ax0.scatter(x300,z300, color='firebrick', s=1,marker='x')
        d = ax0.scatter(x600,z600, color='firebrick', s=1,marker='d')

        
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    plt.colorbar(a, cax=cax, orientation="horizontal",
                 label=r'$\bar{T}_{k=3} - \bar{T}_{NL}$ [$^\circ$C]')

    b = ax1.contourf(k5.Xi, k5.Yi, k5.T_mean - NL.T_mean, levels=levels1, cmap='cmc.roma',alpha=alpha)
    if alpha != 1.0:
        b = ax1.scatter(x300,z300, color='firebrick', s=1,marker='x')
        d = ax1.scatter(x600,z600, color='firebrick', s=1,marker='d')
    divider = make_axes_locatable(ax1)
    cax2 = divider.append_axes("bottom", size="5%", pad=0.05)
    plt.colorbar(b, cax=cax2, orientation="horizontal",
                 label=r'$\bar{T}_{k=5} - \bar{T}_{NL}$ [$^\circ$C]')

    modify_axes(ax=ax0, label_x='Distance [km]', lim=[0, 1250], label_y='Depth [km]')
    modify_axes(ax=ax1, label_x='Distance [km]', lim=[0, 1250])

    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent=TRANSPARENT)
    plt.close(fig)
    
    



#------------------------------------------------------------------------------
def plot_heat_flux(path2save:str,pd0:pd.DataFrame,pd1:pd.DataFrame,pd2:pd.DataFrame, name_figure:str,name:str):

    name_plot = f'Figure_{name_figure}_{name}.png'
    
    fig = plt.figure(figsize=(8, 6))
    
    ax0 = fig.add_axes([0.1,0.1,0.4,0.8])
    ax1 = fig.add_axes([0.54,0.1,0.4,0.8])
    
    for i in range(len(surface_data[1].non_linear)):
        ax0.plot(surface_data[1].x_top, surface_data[1].heat_flux[i,:]*1000, color='olive', lw=0.5)
        ax0.plot(surface_data[0].x_top, surface_data[0].heat_flux[i,:]*1000, color='cadetblue', lw=0.5)
        ax1.plot(surface_data[2].x_top, surface_data[2].heat_flux[i,:]*1000, color='firebrick', lw=0.5)
        
        
    ax0.set_xlabel('Distance from trench [km]')
    ax0.set_ylabel(r'Heat flux $[mW/m^2]$')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.tick_params(axis='both', colors='black')
    ax1.spines['left'].set_linewidth(2.5)
    ax1.spines['bottom'].set_linewidth(2.5)
    ax0.set_xlim(10, 600)
    ax0.set_ylim(0,100)
    ax1.set_xlim(10, 600)
    ax1.set_ylim(0,100)
    ax1.set_yticklabels([])
    ax1.set_xlabel('Distance from trench [km]')
    
    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent = TRANSPARENT)




def colored_line_set(x_list, y_list, vc, case_ids, cmap='viridis', lw=2, ax=None):
    """
    Plot multiple lines with color mapped to scalar vc.

    Parameters
    ----------
    x_list : list of arrays
        List containing x coordinates for each line.
    y_list : list of arrays
        List containing y coordinates for each line.
    vc : array-like
        Scalar values used to color each line.
    cmap : str
        Matplotlib colormap name.
    lw : float
        Line width.
    ax : matplotlib axis or None
    """

    if ax is None:
        fig, ax = plt.subplots()

    norm = colors.Normalize(vmin=np.min(vc), vmax=np.max(vc))
    cmap = cm.get_cmap(cmap)

    for x, c, case_id in zip(x_list, vc, case_ids):
        print(f'Plotting case {case_id} with vc={c:.2f} cm/yr')
        ax.plot(x, y_list, color=cmap(norm(c)), lw=0.5)
        ax.axhline(y=-80.0, color='black', lw=0.5, ls='--')
        ax.axhline(y=-120.0, color='black', lw=0.5, ls='--')

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='vc')

    return ax

def figure_slab_surface(path2save:str, pd:pd.DataFrame,C:CoordinateSurface,DB:data_base_PT_whitney, figure_number:int)->None:
    
    name_plot = f'Figure_{figure_number}.png'
    
    from matplotlib.collections import LineCollection
    
    fig = plt.figure(figsize=(16, 10))

    # Top row: 
    
    bx = 0.1
    by = 0.1 
    sy = 0.40
    dy = 0.08
    sx_top = 0.28 
    dx_top_a = 0.08 
    dx_top_b = 0.05 
    sx_bot = 0.35
    dx_bot = 0.05 
    sx_bot_b = 1 - sx_bot * 2 + dx_bot # Legend box 
    
    top_row = by+sy+dy
    # --- 
    # Top row: figure ax0 -> NL slab surface // black line + scatter DB + band 80 km 
    #        : figure ax1 -> dT LN_k=3.1 - NL // black line + band 80 km 
    #        : figure ax2 -> dT LN_k=5 - NL // blacl line   + band 80 km 
    # Bot row: figure ax3 -> scatter plot dT vs gamma (thermal parameter) : cadet_blue => k=3.1 olive_green = k=5.0 
    #        : figure ax4 -> scatter plot dT vs Tp (potential temperature): 
    # --- 
    # ax5 => empty axes where to put the legend.  
    ax0 = fig.add_axes([bx,top_row,sx_top,sy])
    ax1 = fig.add_axes([bx+sx_top+dx_top_a,top_row,sx_top,sy])
    ax2 = fig.add_axes([bx+2*sx_top+dx_top_a+dx_top_b,top_row,sx_top,sy])
    
    ax3 = fig.add_axes([bx,by,sx_bot,sy])
    ax4 = fig.add_axes([bx+sx_bot+dx_bot,by,sx_bot,sy])
    ax5 = fig.add_axes([bx+2*sx_bot+dx_bot,by,sx_bot,sy])
    
    k3, k5, knl = create_common_data_set(a = pd)
    
    temp_min_gr = 0 + 5 * np.abs(C.z_slab)
    temp_max_gr = 0 + 20 * np.abs(C.z_slab)
    temp_mean_gr = 0 + 10 * np.abs(C.z_slab)
    p_fake = np.abs(C.z_slab) * 9.81 * 3200 


    k_nl_slab_temp = knl.loc[(DBa['option_thermal'] == 1), f'T_S']
    k_nl_slab_pres = knl.loc[(DBa['option_thermal'] == 1), f'P_S']  
    k_3_slab_temp = knl.loc[(DBa['k'] == 3.1), f'T_S'] 
    k_5_slab_temp = knl.loc[(DBa['k'] == 3.1), f'T_S'] 
    dtemp_k3 =  k_nl_slab_temp - k_3_slab_temp
    dtemp_k5 =  k_nl_slab_temp - k_5_slab_temp

    fig.savefig(f'{path2save}/inset_{name_plot}', dpi=600, transparent = TRANSPARENT)



# ----------------------- Functions ------------------ #

# ---------------- 
def plot_distance_to_isotherms(path2save:str, DB:pd.DataFrame, scale=False, name:str='PT',name_figure:str=''):
    
    name_plot = f'{name_figure}_{name}.png'

    
    def modify_axes(ax=None,label_x=None,lim_x=None,lim_y=None,label_y=None,isotherms=None):
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        
        ax.tick_params(axis='both', colors='black', direction='in', length=5, width=2.0)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.grid(visible=True, which='major', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        if label_x is not None: 
            ax.set_xlabel(label_x, fontsize=14, labelpad=10)
        
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)

        if label_y is not None and isotherms is not None:
            ax.set_ylabel(label_y, fontsize=14)
        
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
   
        
        return ax
    

    def plot_histogram(ax ,k3, k5,bins):
        import scipy.stats as stats


        kde1 = stats.gaussian_kde(k3)
        kde2 = stats.gaussian_kde(k5)
        
        k3_fit = kde1(bins)
        k5_fit = kde2(bins)

        k3_fit = k3_fit 
        k5_fit = k5_fit 
        
        
        LN_3 = k3_fit
        LN_5 = k5_fit
        
        ax.hist(k5,bins, alpha=0.1, color='olive',density=True)
        ax.hist(k3,bins,alpha=0.1, color='cadetblue',density=True)

        ax.plot(bins,LN_3,color='cadetblue', linewidth=1.2)
        ax.plot(bins,LN_5,color='olive', linewidth=1.2)

        max_fit = []
        
        return ax,max_fit    


    def extract_data(a:pd.DataFrame,b:pd.DataFrame,c:pd.DataFrame,isotherms:int,scale=False):
        # Select couple of experiments from k3/k5 and NL
        
        x0 = a[f'ell_point_{isotherms}']
        x1 = b[f'ell_point_{isotherms}']
        x2 = c[f'ell_point_{isotherms}']

 
        d_k3 = (x0-x2)/x2 #compute_distance(x2,x0)
        d_k5 = (x1-x2)/x2 #compute_distance(x2,x1)


        return d_k3, d_k5
        
        
        
    keys = ["Tp", "age", "vc"]
    

    
    k3 = DB[(DB['option_thermal'] == 0) & (DB['k'] == 3.0)]
    k5 = DB[(DB['option_thermal'] == 0) & (DB['k'] == 5.0)]
    NL = DB[(DB['option_thermal'] > 0)]    
    
    common_keys = (k3[keys]
    .merge(k5[keys], on=keys, how="inner")
    .merge(NL[keys], on=keys, how="inner")
    .drop_duplicates()
    .sort_values(keys)
    .reset_index(drop=True))
    
    
    
    
    k3_common = (k3.merge(common_keys, on=keys, how="inner")).sort_values(keys).reset_index(drop=True)
    k5_common = (k5.merge(common_keys, on=keys, how="inner")).sort_values(keys).reset_index(drop=True)
    NL_common = (NL.merge(common_keys, on=keys, how="inner")).sort_values(keys).reset_index(drop=True)

    d_300_k3, d_300_k5 = extract_data(k3_common,k5_common,NL_common, 300,scale=scale)
    d_500_k3, d_500_k5 = extract_data(k3_common,k5_common,NL_common, 500,scale=scale)
    d_600_k3, d_600_k5 = extract_data(k3_common,k5_common,NL_common, 600,scale=scale)

    fig = plt.figure(figsize=(10, 6))
    bx = 0.1
    by = 0.1
    sx = 0.25
    dx = 0.06
    sy = 0.35
    dy = 0.1 
    ax1 = fig.add_axes([bx,by,sx,sy])
    ax2 = fig.add_axes([bx+dx+sx,by,sx,sy])
    ax3 = fig.add_axes([bx+2*dx+2*sx,by,sx,sy])
    ax4 = fig.add_axes([bx,by+sy+dy,sx,sy])
    ax5 = fig.add_axes([bx+dx+sx,by+sy+dy,sx,sy])
    ax6 = fig.add_axes([bx+2*dx+2*sx,by+sy+dy,sx,sy])
    
    ax1,_  = plot_histogram(ax1, d_300_k3, d_300_k5, bins=np.linspace(np.min([d_300_k3.min(), d_300_k5.min()]), np.max([d_300_k3.max(), d_300_k5.max()]), 50))
    ax2,_  = plot_histogram(ax2, d_500_k3, d_500_k5, bins=np.linspace(np.min([d_500_k3.min(), d_500_k5.min()]), np.max([d_500_k3.max(), d_500_k5.max()]), 50))
    ax3,_  = plot_histogram(ax3, d_600_k3, d_600_k5, bins=np.linspace(np.min([d_600_k3.min(), d_600_k5.min()]), np.max([d_600_k3.max(), d_600_k5.max()]), 50))
    
    ax4.scatter(k3_common['ell_point_300'],d_300_k3, c='cadetblue', marker='d', s=10, label='k=3.0')
    ax4.scatter(k5_common['ell_point_300'],d_300_k5, c='olive', marker='d', s=10, label='k=5.0')
    ax5.scatter(k3_common['ell_point_500'],d_500_k3, c='cadetblue', marker='d', s=10, label='k=3.0')  
    ax5.scatter(k5_common['ell_point_500'],d_500_k5, c='olive', marker='d', s=10, label='k=5.0')
    ax6.scatter(k3_common['ell_point_600'],d_600_k3, c='cadetblue', marker='d', s=10, label='k=3.0')
    ax6.scatter(k5_common['ell_point_600'],d_600_k5, c='olive', marker='d', s=10, label='k=5.0')
    
    ax4 = modify_axes(ax4, label_x=r'$\gamma$ [km]', label_y=r'$err_{abs}$ [n.d.]', lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=300)
    ax5 = modify_axes(ax5, label_x=r'$\gamma$ [km]', label_y=None, lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=500)
    ax6 = modify_axes(ax6, label_x=r'$\gamma$ [km]', label_y=None, lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=600)

    ax1 = modify_axes(ax1, label_x=r'$err_{abs}$ [n.d.]', label_y=r'f []', lim_x=[0.0, 1.0], lim_y=None, isotherms=300)
    ax2 = modify_axes(ax2, label_x=r'$err_{abs}$ [n.d.]', label_y=None, lim_x=[0.0, 1.0], lim_y=None, isotherms=500)
    ax3 = modify_axes(ax3, label_x=r'$err_{abs}$ [n.d.]', label_y=None, lim_x=[0.0, 1.0], lim_y=None, isotherms=600)


    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent=TRANSPARENT)
    # Extract data 
    
#---------------------------------------------------------------------------------    
    

    
def plot_distance_isotherm_seismic(path2save:str, DB:pd.DataFrame, scale=False, name='PT',name_figure:str=''):
    name_plot = f'{name_figure}_{name}.png'
    
    def plot_histogram(ax ,k3, k5, nl,bins):
        import scipy.stats as stats


        kde1 = stats.gaussian_kde(k3)
        kde2 = stats.gaussian_kde(k5)
        if nl is not None:
            kde3 = stats.gaussian_kde(nl)
        k3_fit = kde1(bins)
        k5_fit = kde2(bins)
        if nl is not None:
            nl_fit = kde3(bins)
        else:
            nl_fit = 0.0

        k3_fit = k3_fit 
        k5_fit = k5_fit 
        
        nl_fit = nl_fit
        
        LN_3 = k3_fit
        LN_5 = k5_fit
        LN_NL = nl_fit

        ax.hist(k5,bins, alpha=0.1, color='olive',density=True)
        ax.hist(k3,bins,alpha=0.1, color='cadetblue',density=True)
        if nl is not None:
            ax.hist(nl,bins,alpha=0.1, color='firebrick',density=True)

        ax.plot(bins,LN_3,color='cadetblue', linewidth=1.2)
        ax.plot(bins,LN_5,color='olive', linewidth=1.2)
        if nl is not None:
            ax.plot(bins,LN_NL,color='firebrick', linewidth=1.2)

        max_fit = []
        
        return ax,max_fit    


    def extract_data(a:pd.DataFrame,b:pd.DataFrame,c:pd.DataFrame,isotherms:int,scale=False):
        # Select couple of experiments from k3/k5 and NL
        
        a = a[f'max_dpth{isotherms}']
        b = b[f'max_dpth{isotherms}']
        c = c[f'max_dpth{isotherms}']

        

        return a,b,c
        
        
        
    keys = ["Tp", "age", "vc"]
    

    
    k3 = DB[(DB['option_thermal'] == 0) & (DB['k'] == 3.1)]
    k5 = DB[(DB['option_thermal'] == 0) & (DB['k'] == 5.0)]
    NL = DB[(DB['option_thermal'] > 0)]    
    
    common_keys = (k3[keys]
    .merge(k5[keys], on=keys, how="inner")
    .merge(NL[keys], on=keys, how="inner")
    .drop_duplicates()
    .sort_values(keys)
    .reset_index(drop=True))
    
    
    
    
    k3_common = (k3.merge(common_keys, on=keys, how="inner")).sort_values(keys).reset_index(drop=True)
    k5_common = (k5.merge(common_keys, on=keys, how="inner")).sort_values(keys).reset_index(drop=True)
    NL_common = (NL.merge(common_keys, on=keys, how="inner")).sort_values(keys).reset_index(drop=True)

    d_150_k3, d_150_k5, d_150_NL = extract_data(k3_common,k5_common,NL_common, 150,scale=scale)
    d_450_k3, d_450_k5, d_450_NL = extract_data(k3_common,k5_common,NL_common, 450,scale=scale)

    D_k3 = np.abs((d_450_k3 - d_150_k3))
    D_k5 = np.abs((d_450_k5 - d_150_k5))
    D_NL = np.abs((d_450_NL - d_150_NL))
    D_k3 = D_k3-D_NL
    D_k5 = D_k5-D_NL

    fig = plt.figure(figsize=(10, 6))
    bx = 0.1
    by = 0.1
    sx = 0.25
    dx = 0.08
    sy = 0.35
    dy = 0.1 
    ax1 = fig.add_axes([bx,by,sx,sy])
    ax2 = fig.add_axes([bx+dx+sx,by,sx,sy])
    ax4 = fig.add_axes([bx,by+sy+dy,sx,sy])
    ax5 = fig.add_axes([bx+dx+sx,by+sy+dy,sx,sy])
    
    ax1,_  = plot_histogram(ax1, D_k3, D_k5, None, bins=np.linspace(np.min([D_k3.min(), D_k5.min()]), np.max([D_k3.max(), D_k5.max()]), 50))
    ax2,_  = plot_histogram(ax2, D_k3/D_NL, D_k5/D_NL, None, bins=np.linspace(np.min([D_k3/D_NL.min(), D_k5/D_NL.min()]), np.max([D_k3/D_NL.max(), D_k5/D_NL.max()]), 50))
    ax4.scatter(k3_common['thermal_parameter']/1000,D_k3, c='cadetblue', marker='d', s=0.1, label='k=3.0')
    ax4.scatter(k5_common['thermal_parameter']/1000,D_k5, c='olive', marker='d', s=0.1, label='k=5.0')
    ax5.scatter(k3_common['thermal_parameter']/1000,D_k3/D_NL, c='cadetblue', marker='d', s=0.1, label='k=3.0')  
    ax5.scatter(k5_common['thermal_parameter']/1000,D_k5/D_NL, c='olive', marker='d', s=0.1, label='k=5.0')

    
    ax4 = modify_axes(ax4, label_x=r'$\gamma$ [1000 km]', label_y=r'$\Delta xi [km]$', lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=300)
    ax5 = modify_axes(ax5, label_x=r'$\gamma$ [1000 km]', label_y=r'$\Delta xi*$ [ ]', lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=500)

    ax1 = modify_axes(ax1, label_x=r'$\Delta \xi [km]$', label_y=r'f []', lim_x=[0.0, 1.0], lim_y=None, isotherms=300)
    ax2 = modify_axes(ax2, label_x=r'$\Delta \xi*$ [ ]', label_y=None, lim_x=[0.0, 1.0], lim_y=None, isotherms=500)
    ax2.text(
        0.02, 0.97,
        (
            r'$\xi^{k=3|k=5|NL} = z_{450}-z_{150}$' '\n'
            r'$\Delta \xi = \xi^{k=3|k=5}-\xi^{NL}$' '\n'
            r'$\Delta \xi* = (\Delta \xi)/\xi^{NL}$'
        ),
        transform=ax2.transAxes,
        fontsize=10,
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent=TRANSPARENT)
    # Extract data 
            
# plot the variance along the thermal parameter
def plot_slab_versus_thermalpar(path2save: str, DB: pd.DataFrame, scale=False, name='PT',name_figure:str=''):
    
    name_plot = f'{name_figure}_{name}.png'


    def compute_values(db: pd.DataFrame,db_NL: pd.DataFrame, Tl0: float, Tl1: float, isotherms: int, normalised=False):
        db = db[(db['thermal_parameter'] >= Tl0) & (db['thermal_parameter'] < Tl1)]
        if normalised:
            buf0 = (db_NL[f'max_dpth{isotherms}']-db[f'max_dpth{isotherms}'])/db_NL[f'max_dpth{isotherms}']
        else: 
            buf0 = (db[f'max_dpth{isotherms}'])
        d = buf0.median()
        std = buf0.std()
        vmin = buf0.min()
        vmax = buf0.max()
        
        return d, std, vmin, vmax


    def extract_arrays(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, isotherms: int):
        thermal_parameters = a['thermal_parameter']
        t_p_m = np.min(thermal_parameters)
        t_p_M = np.max(thermal_parameters)
        TPar = np.linspace(t_p_m, t_p_M, 100)
        n = len(TPar) - 1  # np.len() non esiste — si usa len() builtin di Python
        
        a_array = np.zeros([n, 4])
        b_array = np.zeros([n, 4])
        c_array = np.zeros([n, 4])
        newT = np.zeros(n)
        
        for i in range(n):
            Tl0 = TPar[i]
            Tl1 = TPar[i + 1]
            a_array[i, 0], a_array[i, 1], a_array[i, 2], a_array[i, 3] = compute_values(a, Tl0, Tl1, isotherms,normalised=normalised)
            b_array[i, 0], b_array[i, 1], b_array[i, 2], b_array[i, 3] = compute_values(b, Tl0, Tl1, isotherms,normalised=normalised)
            c_array[i, 0], c_array[i, 1], c_array[i, 2], c_array[i, 3] = compute_values(c, Tl0, Tl1, isotherms,normalised=normalised)
            newT[i] = (Tl0 + Tl1) / 2.0  # centro del bin
        
        return a_array, b_array, c_array, newT

    def plot_values(ax, a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, isotherms: int,
                     labels=('k=3.0', 'k=5.0', 'NL'),normalised=False):
        a_array, b_array, c_array, x = extract_arrays(a, b, c, isotherms)
        if normalised:
            dat = (a_array, b_array, c_array)
        else: 
            dat = (a_array, b_array)
        colors = ('cadetblue', 'olive', 'firebrick')
        
        for idx, arr in enumerate(dat):
            ax.plot(x, arr[:, 0], color=colors[idx], label=labels[idx])  # mediana
            ax.fill_between(x,
                            arr[:, 0] - arr[:, 1],  # mediana - std
                            arr[:, 0] + arr[:, 1],  # mediana + std
                            color=colors[idx], alpha=0.4)
            ax.plot(x, arr[:, 2], color=colors[idx], linestyle=':', linewidth=0.8)  # min
            ax.plot(x, arr[:, 3], color=colors[idx], linestyle=':', linewidth=0.8)  # max
        
        return ax

    k3 = DB[(DB['option_thermal'] == 0) & (DB['k'] == 3.1)]
    k5 = DB[(DB['option_thermal'] == 0) & (DB['k'] == 5.0)]
    NL = DB[(DB['option_thermal'] > 0)]

    keys = ["Tp", "age", "vc"]
    
    common_keys = (k3[keys]
                   .merge(k5[keys], on=keys, how="inner")
                   .merge(NL[keys], on=keys, how="inner")
                   .drop_duplicates()
                   .sort_values(keys)
                   .reset_index(drop=True))

    k3_common = k3.merge(common_keys, on=keys, how="inner").sort_values(keys).reset_index(drop=True)
    k5_common = k5.merge(common_keys, on=keys, how="inner").sort_values(keys).reset_index(drop=True)
    NL_common = NL.merge(common_keys, on=keys, how="inner").sort_values(keys).reset_index(drop=True)

    fig = plt.figure(figsize=(12, 12))
    bx = 0.15
    by = 0.1
    dy = 0.01
    dx = 0.08 
    sx = 0.28
    sy = 0.28
    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx, by + sy + dy, sx, sy])
    ax2 = fig.add_axes([bx, by + 2 * (sy + dy), sx, sy])
    
    ax3 = fig.add_axes([bx+dx+sx, by, sx, sy])
    ax4 = fig.add_axes([bx+dx+sx, by + sy + dy, sx, sy])
    ax5 = fig.add_axes([bx+dx+sx, by + 2 * (sy + dy), sx, sy])
    
    ax = fig.gca()
    
    ax2 = plot_values(ax2, k3_common, k5_common, NL_common, isotherms=600)


    
    ax1 = plot_values(ax1, k3_common, k5_common, NL_common, isotherms=450)


    
    ax0 = plot_values(ax0, k3_common, k5_common, NL_common, isotherms=150)

    
    ax5 = plot_values(ax2, k3_common, k5_common, NL_common, isotherms=600,normalised=True)


    
    ax4 = plot_values(ax1, k3_common, k5_common, NL_common, isotherms=450,normalised=True)


    
    ax3 = plot_values(ax0, k3_common, k5_common, NL_common, isotherms=150,normalised=True)

    fig.savefig(f'{path2save}/Variance_{name}_{scale}_alt.png', dpi=600, transparent=TRANSPARENT)    


