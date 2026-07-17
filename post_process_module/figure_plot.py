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
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from dataclasses import dataclass
from typing import Optional

# Conversion pt - cm 
_PT_CM_ = 0.035
# Ticks, label and text font size 
_TICKSFONTSIZE_ = 0.8/_PT_CM_
_LABELFONTSIZE_ = 1.0/_PT_CM_
_TEXTFONTSIZE_ = 0.65/_PT_CM_

def sign(num):
    return -1 if num < 0 else 1

def ranges(x,y,z): 
    lim = [np.floor(np.nanmin([np.nanmin(x),np.nanmin(y),np.nanmin(z)])),np.ceil(np.nanmax([np.nanmax(x),np.nanmax(y),np.nanmax(z)]))]
    return lim

def create_list_tick(lim)->list:
    
    d = (lim[1]-lim[0])/4
    d = check_precision(d)
    t = lim[0]
    ticks = [lim[0]]
    while t < lim[1]:
        t = t + d     
        if t > lim[1]:
            t = lim[1]
        
        ticks.append(t)
    
    return ticks

def check_precision(value:float)-> float: 
    
    if value == 0.0:
        return 0.0
    
    value_u = np.abs(value)
    direction = sign(value)
    exponent = np.floor(np.log10(value_u))
    scale = 10**(exponent)
    value_u = value_u/scale 
    buf = value_u - np.floor(value_u)
    
    precision = [0.25,0.5,0.75,1.0]
    
    for i in range(len(precision)):
        if i == 0:
            interval = buf<precision[i]
        else: 
            interval = buf>=precision[i-1] and buf<precision[i]
        if interval: 
            corr = precision[i]
    
    value_u = (np.floor(value_u)+corr) * scale * direction
    
    return value_u 


def create_common_data_set(pdb:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Function that creates common datasets: 
    -> Time to time simulations fails,so it is necessary to check how many experiments fails,
    select the common successfull data

    Args:
        a (pd.DataFrame): main database

    Returns:
        tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]: filtered database
    """

    
    keys = ["Tp", "age", "vc"]
    
    a = pdb[(pdb['k'] == 3.1)]
    b = pdb[(pdb['k'] == 5.0)]
    c = pdb[(pdb['k'] == 0.0)]    
    
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
                precision: Optional[str] = None,
                letter:Optional[str]='[a]') -> mpl_axes:
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
    
    precision = '%.0f'
    
    ax.tick_params(axis='both', which='major', pad=9)
    
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
        ax.xaxis.set_label_position('bottom')

    if lim_x is not None:
        ax.set_xlim(lim_x[0],lim_x[1])

    if ticks_x is not None:
        ax.set_xticks(ticks_x)


            
    if label_x[1] == 'y':
        ax.xaxis.set_major_formatter(FormatStrFormatter(precision))
    else:
        ax.set_xticklabels([])

    # --- y-axis ---
    if label_y[0] != 'n':
        ax.set_ylabel(label_y[0], fontsize=_LABELFONTSIZE_, labelpad=10)  # fixed: was label_x[0]
        ax.yaxis.set_label_position('left')                                # fixed: was 'top'

    if lim_y is not None:                                                  # fixed: was lim_x
        ax.set_ylim(lim_y[0],lim_y[1])


    if ticks_y is not None:
        ax.set_yticks(ticks_y)

    if label_y[1] == 'y':
        ax.yaxis.set_major_formatter(FormatStrFormatter(precision))
    else:
        ax.set_yticklabels([])
        
        
    ax.text(
        0.02, 0.97,letter,
        transform=ax.transAxes,
        c='w',
        fontweight='demibold',
        fontsize=10,
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='k', alpha=0.7)
    )

    return ax





def plot_slab_variances(A, path2save: str,number_figure:int) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    name_plot = f'Figure_{number_figure}.png'
    
    def extract_limits(k3, k5, NL):
        l_m = []
        l_M = []

        l_m.append(np.nanmin(NL.temp_mean_2d-k3.temp_mean_2d))
        l_M.append(np.nanmax(NL.temp_mean_2d-k3.temp_mean_2d))

        l_m.append(np.nanmin(NL.temp_mean_2d-k5.temp_mean_2d))
        l_M.append(np.nanmax(NL.temp_mean_2d-k5.temp_mean_2d))

        return l_m, l_M

    fig = plt.figure(figsize=(12, 12))

    k3,k5,NL = A 
    
    Xi,Yi = k3.xi,k3.yi
    
    
    
    bx = 0.12
    by = 0.1
    sx = 0.38
    sy = 0.68
    dx = 0.08

    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx + sx + dx, by, sx, sy])

    lima,limb = extract_limits(k3,k5,NL)

    lim_x = [0.0, 800.0]
    lim_y = [-600,0.0]

    levels0 = np.linspace(np.floor(lima[0]), np.ceil(limb[0]), 10)
    levels1 = np.linspace(np.floor(lima[1]), np.ceil(limb[1]), 10)
    alpha = 1.0


    a = ax0.pcolormesh(Xi, Yi, NL.temp_mean_2d-k3.temp_mean_2d, cmap='cmc.roma',alpha=alpha)
        
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("bottom", size="5%", pad=0.8)
    cbar = plt.colorbar(a, cax=cax, orientation="horizontal",location="top")

    b = ax1.pcolormesh(Xi, Yi, NL.temp_mean_2d-k5.temp_mean_2d, cmap='cmc.roma',alpha=alpha)
    cbar.set_label( label=r'$\bar{T}_{k=3} - \bar{T}_{NL}$ [$^\circ$C]',fontsize=14)
    cbar.ax.tick_params(labelsize=12)   
    divider = make_axes_locatable(ax1)
    cax2 = divider.append_axes("bottom", size="5%", pad=0.8)
    cbar2 = plt.colorbar(b, cax=cax2, orientation="horizontal",location="top")

    cbar2.set_label(label=r'$\bar{T}_{k=5} - \bar{T}_{NL}$ [$^\circ$C]' ,fontsize=14)
    cbar2.ax.tick_params(labelsize=12)
    #modify_axes(ax=ax0, label_x='Distance [km]', lim=[0, 1250], label_y='Depth [km]')
    #modify_axes(ax=ax1, label_x='Distance [km]', lim=[0, 1250])
    ticks_x = [0,400,800]
    ax0 = modify_axes(ax0,label_x=[r'x, [km]','y'],label_y=[r'z, [km]','y'],lim_y=lim_y,lim_x=lim_x,letter='[a]',ticks_x=ticks_x)
    ax1 = modify_axes(ax1,label_x=[r'x, [km]','y'],label_y=[r'','n'],lim_y=lim_y,lim_x=lim_x,letter='[b]',ticks_x=ticks_x)
    ax0.tick_params(axis='both', which='major', pad=9)
    
    ax0.spines['top'].set_visible(True)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_color('black')
    ax0.spines['bottom'].set_color('black')
    ax0.tick_params(axis='both', colors='black', direction='in', length=5, width=2.0)
    ax0.spines['left'].set_linewidth(2.0)
    ax0.spines['top'].set_linewidth(2.0)
    ax0.grid(visible=True, which='major', axis='both', color='gray', linestyle='--', linewidth=0.3, alpha=0.7)
    ax0.set_yticklabels(ax0.get_yticklabels(), fontsize=_TICKSFONTSIZE_)
    ax0.set_xticklabels(ax0.get_xticklabels(), fontsize=_TICKSFONTSIZE_)
    ax0.xaxis.tick_top()                  # Move ticks to the top
    ax0.xaxis.set_label_position("top")   # Move x-axis label to the top
    
    ax1.spines['top'].set_visible(True)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.tick_params(axis='both', colors='black', direction='in', length=5, width=2.0)
    ax1.spines['left'].set_linewidth(2.0)
    ax1.spines['top'].set_linewidth(2.0)
    ax1.grid(visible=True, which='major', axis='both', color='gray', linestyle='--', linewidth=0.3, alpha=0.7)
    ax1.set_yticklabels(ax0.get_yticklabels(), fontsize=_TICKSFONTSIZE_)
    ax1.set_xticklabels(ax0.get_xticklabels(), fontsize=_TICKSFONTSIZE_)
    ax1.xaxis.tick_top()                  # Move ticks to the top
    ax1.xaxis.set_label_position("top")   # Move x-axis label to the top
    
    
    
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

def figure_slab_surface(path2save:str, pdb:pd.DataFrame,crd:CoordinateSurface,petro_db:data_base_PT_whitney, figure_number:int)->None:
    def create_array(buf:pd.DataFrame)->np.ndarray:
        
        if len(buf.shape)>1:
            raise ValueError('The function accept only 1D dataframe')
        
        array = np.zeros([buf.size,len(buf.iloc[0])],dtype = np.float32)
        
        for i in range(buf.size):
            array[i,:] = buf.iloc[i]
    
    
        return array 
    
    def difference_surface(pdb_a:pd.DataFrame,pdb_b:pd.DataFrame)->np.ndarray:
        """Enclosure to find the dT array
        -> check if the same row indicate the same [Tp,vc,age]-> if so, perform 
        the difference-> update array. 

        Args:
            pdb_a (pd.DataFrame): Non linear database
            pdb_b (pd.DataFrame): Linear database

        Returns:
            final array with all the experiment
        """
        
        
        count = len(pdb_a['case_id'])
        keys = ['Tp','vc','age']
        array = np.zeros([count,len(pdb_a.loc[0,'T_S'])],dtype = np.float32) 
        n_exp = 0
        
        for i in range(count):
            same = False
            
            for k in keys: 
                if pdb_a.loc[i,k] != pdb_b.loc[i,k]:
                    same = False
                    break 
                else: 
                    same = True
            if same:
                array[i,:] = pdb_a.loc[i,'T_S']-pdb_b.loc[i,'T_S']
                n_exp = n_exp + 1
        
        print(f'Number of experiments {n_exp}, {n_exp/count:.1f}')
    
        return array 
    
    
    
    name_plot = f'Figure_{figure_number}.png'
    
    from matplotlib.collections import LineCollection
    
    fig = plt.figure(figsize=(16, 10))

    # Top row: 
    
    bx = 0.1
    by = 0.1 
    sy = 0.35
    dy = 0.1
    sx_top = 0.25 
    dx_top_a = 0.07
    dx_top_b = 0.07 
    sx_bot = 0.25
    dx_bota = 0.07 
    dx_botb = 0.07
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
    ax4 = fig.add_axes([bx+sx_bot+dx_bota,by,sx_bot,sy])
    ax5 = fig.add_axes([bx+2*sx_bot+dx_bota+dx_botb,by,sx_bot,sy])
    
    k3, k5, knl = create_common_data_set(pdb = pdb)
    
    temp_min_gr = 0 + 5 * np.abs(crd.z_slab)
    temp_max_gr = 0 + 20 * np.abs(crd.z_slab)
    temp_mean_gr = 0 + 10 * np.abs(crd.z_slab)
    p_fake = np.abs(crd.z_slab) * 9.81 * 3200 

    # Extract the temperature of the non linear experiments
    k_nl_slab_temp = create_array(knl.loc[(knl['k'] == 0), f'T_S'])
    k_nl_slab_pres = create_array(knl.loc[(knl['k'] == 0), f'P_S']) 
    
    # Extract the difference between the groups 
    dtemp_k3 = difference_surface(knl,k3)
    dtemp_k5 = difference_surface(knl,k5)
    
    for i in range(k_nl_slab_temp.shape[0]):
        ax0.plot(k_nl_slab_temp[i,:],k_nl_slab_pres[i,:],c='firebrick',alpha=0.01)
        ax1.plot(dtemp_k3[i,:],k_nl_slab_pres[0,:],c='cadetblue',alpha=0.01)
        ax2.plot(dtemp_k5[i,:],k_nl_slab_pres[0,:],c='olive',alpha=0.01)
    
    ax0.axvline(150,linewidth=0.8,c='k',linestyle=':')
    ax0.axvline(450,linewidth=0.8,c='k',linestyle=':')

    
    ax0.scatter(petro_db.T,petro_db.P,c='b',s=5.0,marker='x')
    ax3.scatter(k3['thermal_parameter']/1000,knl['d_seismic_ell'],color='firebrick',s=0.5)
    ax3.scatter(k3['thermal_parameter']/1000,k3['d_seismic_ell'],color='cadetblue',s=0.5)
    ax3.scatter(k3['thermal_parameter']/1000,k5['d_seismic_ell'],color='olive',s=0.5)

    ax4.scatter(k5['thermal_parameter']/1000,k5['d_seismic_150'],color='olive',s=0.5)
    ax4.scatter(k3['thermal_parameter']/1000,k3['d_seismic_150'],color='cadetblue',s=0.5)
    ax4.scatter(knl['thermal_parameter']/1000,knl['d_seismic_150'],color='firebrick',s=0.5)

    ax5.scatter(knl['thermal_parameter']/1000,knl['d_seismic_450'],color='firebrick',s=0.5)
    ax5.scatter(k3['thermal_parameter']/1000,k3['d_seismic_450'],color='cadetblue',s=0.5)
    ax5.scatter(k5['thermal_parameter']/1000,k5['d_seismic_450'],color='olive',s=0.5)

    
    # Homogeneise the layout of the figure. 
    
    ax0 = modify_axes(ax0,label_x=[r'T [$^{\circ}$C]','y'],lim_x=[0,1000],lim_y=[0,4],label_y=[r'P [GPa]','y'],ticks_x=[0.0,250,500,750,1000], ticks_y=[0,1,2,3,4],letter='[a]')
    
    
    lim_x = ranges(dtemp_k3,dtemp_k5,np.nan)
    lim_x = [check_precision(lim_x[0]),check_precision(lim_x[1])]
    ticks_x = create_list_tick(lim_x)
    
    
    ax1 = modify_axes(ax1,label_x=[r'$\Delta$ T [$^{\circ}$C]','y'],lim_y=[0,4],lim_x=lim_x,ticks_x=ticks_x, ticks_y=[0,1,2,3,4],letter='[b]')
    ax2 = modify_axes(ax2,label_x=[r'$\Delta$ T [$^{\circ}$C]','y'],lim_y=[0,4],lim_x=lim_x,ticks_x=ticks_x,ticks_y=[0,1,2,3,4],letter='[c]')
    
    lim_x = ranges(knl['thermal_parameter']/1000,k3['thermal_parameter']/1000, k5['thermal_parameter']/1000)
    lim_x = [check_precision(lim_x[0]),check_precision(lim_x[1])]
    ticks_x = create_list_tick(lim_x)

    
    lim_y = ranges(knl['d_seismic_ell'],k3['d_seismic_ell'],k5['d_seismic_ell'])
    lim_y = [check_precision(lim_y[0]),check_precision(lim_y[1])]
    ticks_y = create_list_tick(lim_y)

    
    ax3 = modify_axes(ax3,label_x=[r'$\gamma$ [1000 km]','y'],label_y=[r'$\Delta \ell^{450-150}$ [km]','y'],lim_y=lim_y,lim_x=lim_x,ticks_x=ticks_x,ticks_y=ticks_y,letter='[d]')

    lim_y =  ranges(knl['d_seismic_150'],k5['d_seismic_150'],k3['d_seismic_150'])
    lim_y = [check_precision(lim_y[0]),check_precision(lim_y[1])]
    ticks_y = create_list_tick(lim_y)
    
    ax4 = modify_axes(ax4,label_x=[r'$\gamma$ [1000 km]','y'],label_y=[r'$z^{\mathrm{slab}}_{150}$ [km]','y'],lim_y=lim_y,lim_x=lim_x,ticks_x=ticks_x,ticks_y=ticks_y,letter='[e]')
    
    lim_y = ranges(knl['d_seismic_450'],k5['d_seismic_450'],k3['d_seismic_450'])
    lim_y = [check_precision(lim_y[0]),check_precision(lim_y[1])]
    ticks_y = create_list_tick(lim_y)
    
    ax5 = modify_axes(ax5,label_x=[r'$\gamma$ [1000 km]','y'],label_y=[r'$z^{\mathrm{slab}}_{450}$ [km]','y'],lim_y=lim_y,lim_x=lim_x,ticks_x=ticks_x,ticks_y=ticks_y)


    # Custom legend entries
    legend_elements = [
        Line2D([0], [0], marker='o', color='firebrick',
               linestyle='None', markersize=8,
               label='Non linear'),

        Line2D([0], [0], marker='o', color='cadetblue',
               linestyle='None', markersize=8,
               label='k = 3.1 [W/m/K]'),

        Line2D([0], [0], marker='o', color='olive',
               linestyle='None', markersize=8,
               label='k = 5 [W/m/K]'),
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        prop={
        'size': _TEXTFONTSIZE_,
        'weight': 'demibold'
    }
    )


    
    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent = TRANSPARENT)



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
    

    
def plot_distance_isotherm_seismic(path2save:str, pdb:pd.DataFrame, figure_number:int =0):
    name_plot = f'Figure_{figure_number}.png'
    
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
    
    k3, k5, knl = create_common_data_set(pdb = pdb)

    

    d_150_k3, d_150_k5, d_150_nl = extract_data(k3,k5,knl, 150)
    d_450_k3, d_450_k5, d_450_nl = extract_data(k3,k5,knl, 450)

    diff_k3 = np.abs((d_450_k3 - d_150_k3))
    diff_k5 = np.abs((d_450_k5 - d_150_k5))
    diff_nl = np.abs((d_450_nl - d_150_nl))
    diff_k3 = diff_k3-diff_nl
    diff_k5 = diff_k5-diff_nl

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
    bin_dimensional = np.linspace(np.min([np.min(diff_k3), diff_k5.min()]), np.max([diff_k3.max(), diff_k5.max()]), 50)
    bin_dimensionless = np.linspace(np.min([diff_k3/diff_nl.min(), diff_k5/diff_nl.min()]), np.max([diff_k3/diff_nl.max(), diff_k5/diff_nl.max()]), 50)
    
    
    ax1,_  = plot_histogram(ax1, diff_k3, diff_k5, None, bins=np.linspace(np.min([np.min(diff_k3), diff_k5.min()]), np.max([diff_k3.max(), diff_k5.max()]), 50))
    ax2,_  = plot_histogram(ax2, diff_k3/diff_NL, diff_k5/diff_NL, None, bins=np.linspace(np.min([diff_k3/diff_nl.min(), diff_k5/diff_nl.min()]), np.max([diff_k3/diff_nl.max(), diff_k5/diff_nl.max()]), 50))
    ax4.scatter(k3_common['thermal_parameter']/1000,diff_k3, c='cadetblue', marker='d', s=0.1, label='k=3.0')
    ax4.scatter(k5_common['thermal_parameter']/1000,diff_k5, c='olive', marker='d', s=0.1, label='k=5.0')
    ax5.scatter(k3_common['thermal_parameter']/1000,diff_k3/diff_nl, c='cadetblue', marker='d', s=0.1, label='k=3.0')  
    ax5.scatter(k5_common['thermal_parameter']/1000,diff_k5/diff_nl, c='olive', marker='d', s=0.1, label='k=5.0')

    
    ax4 = modify_axes(ax4, label_x=r'$\gamma$ [1000 km]', label_y=r'$\Delta xi [km]$', lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=300)
    ax5 = modify_axes(ax5, label_x=r'$\gamma$ [1000 km]', label_y=r'$\Delta xi*$ [ ]', lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=500)

    ax1 = modify_axes(ax1, label_x=r'$\Delta \xi [km]$', label_y=r'f []', lim_x=[0.0, 1.0], lim_y=None, isotherms=300)
    ax2 = modify_axes(ax2, label_x=r'$\Delta \xi*$ [ ]', label_y=None, lim_x=[0.0, 1.0], lim_y=None, isotherms=500)
    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent=TRANSPARENT)
    # Extract data 
            
# plot the variance along the thermal parameter
def plot_slab_versus_thermalpar(path2save: str, pdb: pd.DataFrame,figure_number:int=0):
    
    name_plot = f'Figure_{figure_number}.png'


    def compute_values(db: pd.DataFrame,db_nl: pd.DataFrame, al0: float, bl1: float, isotherms: int, normalised=False):
        db = db[(db['thermal_parameter'] >= al0) & (db['thermal_parameter'] < bl1)]
        db_nl = db_nl[(db_nl['thermal_parameter'] >= al0) & (db_nl['thermal_parameter'] < bl1)]

        buf0 = np.zeros([len(db_nl[f'max_dpth{isotherms}'])],dtype = np.float32)
        for i in range(len(db_nl[f'max_dpth{isotherms}'])):
            bufa =  db_nl.iloc[i]
            bufb =  db.iloc[i]
            if normalised:

                buf0[i] = np.abs((bufa[f'max_dpth{isotherms}'][0]-bufb[f'max_dpth{isotherms}'][0]))
            else: 
                buf0[i] = (bufb[f'max_dpth{isotherms}'][0])
        
        d = np.nanmedian(buf0)
        std = np.nanstd(buf0)
        vmin = np.nanpercentile(buf0,25)
        vmax = np.nanpercentile(buf0,75)
        
        return d, std, vmin, vmax


    def extract_arrays(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, isotherms: int, normalised:bool = False):
        thermal_parameters = a['thermal_parameter']
        t_p_min = np.min(thermal_parameters)
        t_p_max = np.max(thermal_parameters)
        thpar = np.linspace(t_p_min, t_p_max,100)
        n = len(thpar) - 1  # np.len() non esiste — si usa len() builtin di Python
        
        a_array = np.zeros([n, 4])
        b_array = np.zeros([n, 4])
        c_array = np.zeros([n, 4])
        newthpar = np.zeros(n)
        
        for i in range(n):
            tpl0 = thpar[i]
            tpl1 = thpar[i + 1]
            a_array[i, 0], a_array[i, 1], a_array[i, 2], a_array[i, 3] = compute_values(a,c, tpl0, tpl1, isotherms,normalised=normalised)

            b_array[i, 0], b_array[i, 1], b_array[i, 2], b_array[i, 3] = compute_values(b,c, tpl0, tpl1, isotherms,normalised=normalised)
            if not normalised: 
                c_array[i, 0], c_array[i, 1], c_array[i, 2], c_array[i, 3] = compute_values(c,c ,tpl0, tpl1, isotherms,normalised=normalised)
            newthpar[i] = (tpl0 + tpl1) / 2.0  # centro del bin
            if isotherms == 450 and normalised:
                print(f'{a_array[i,3]},{a_array[i,2]},{newthpar[i]}')
        


        return a_array, b_array, c_array, newthpar

    def plot_values(ax, a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, isotherms: int,
                     labels=('k=3.0', 'k=5.0', 'NL'),normalised:bool=False):
        a_array, b_array, c_array, x = extract_arrays(a, b, c, isotherms,normalised=normalised)
        if not normalised:
            dat = (a_array, b_array, c_array)
            lim_a = ranges(a_array[:,0]- a_array[:, 1],b_array[:,0]-  b_array[:, 1],c_array[:,0]- c_array[:, 1])    
            lim_b = ranges(a_array[:,0]+ a_array[:, 1],b_array[:,0]+ b_array[:, 1],c_array[:,0]+ c_array[:, 1])
            
        else: 
            lim_a = ranges(a_array[:, 2],b_array[:, 2],np.nan)    
            lim_b = ranges(a_array[:, 3], b_array[:, 3],np.nan)
            
            dat = (a_array, b_array)
        
        lim_y = ranges(lim_a,lim_b,np.nan)
        lim_x = ranges(x/1000,np.nan,np.nan)
        colors = ('cadetblue', 'olive', 'firebrick')
        
        for idx, arr in enumerate(dat):
            ax.plot(x/1000, arr[:, 0], color=colors[idx], label=labels[idx])  # mediana
            ax.fill_between(x/1000,
                            arr[:, 2],  # mediana - std
                            arr[:, 3],  # mediana + std
                            color=colors[idx], alpha=0.4)

        
        
        
        
        return ax, lim_x, lim_y 

    
    k3, k5, knl = create_common_data_set(pdb = pdb)

    fig = plt.figure(figsize=(12, 12))
    bx = 0.15
    by = 0.1
    dy = 0.02
    dx = 0.1 
    sx = 0.35
    sy = 0.26
    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx, by + sy + dy, sx, sy])
    ax2 = fig.add_axes([bx, by + 2 * (sy + dy), sx, sy])
    
    ax3 = fig.add_axes([bx+dx+sx, by, sx, sy])
    ax4 = fig.add_axes([bx+dx+sx, by + sy + dy, sx, sy])
    ax5 = fig.add_axes([bx+dx+sx, by + 2 * (sy + dy), sx, sy])
        
    ax2,lim_x2,lim_y2 = plot_values(ax=ax2, a= k3, b= k5, c= knl, isotherms=600)
    ax1,lim_x1,lim_y1 = plot_values(ax= ax1, a =k3, b = k5,c=knl, isotherms=450)
    ax0,lim_x0,lim_y0 = plot_values(ax =ax0, a= k3, b = k5,c=knl, isotherms=150)
    ax5,lim_x5,lim_y5 = plot_values(ax= ax5, a= k3, b = k5,c=knl, isotherms=600,normalised=True)
    ax4,lim_x4,lim_y4 = plot_values(ax= ax4, a= k3, b = k5,c=knl, isotherms=450,normalised=True)
    ax3,lim_x3,lim_y3 = plot_values(ax= ax3, a= k3, b = k5,c=knl, isotherms=150,normalised=True)
    
    
    # Modify the axis 
    lim_x = [check_precision(lim_x0[0]),check_precision(lim_x0[1])]
    ticks_x = create_list_tick(lim_x)

    lim_y = [check_precision(lim_y0[0]),check_precision(lim_y0[1])]
    ticks_y = create_list_tick(lim_y)
    ax0 = modify_axes(ax0,label_x=[r'$\gamma$ [1000 km]','y'],lim_x = lim_x0,ticks_x=ticks_x,label_y=[r'd$_{150}$ [km]','y'],ticks_y=ticks_y,letter='[c]')
    
    lim_y = [check_precision(lim_y1[0]),check_precision(lim_y1[1])]
    ticks_y = create_list_tick(lim_y)
    ax1 = modify_axes(ax1,label_x=['n','n'],lim_x = lim_x0,ticks_x=ticks_x,label_y=[r'd$_{450}$ [km]','y'],ticks_y=ticks_y,letter='[b]')

    lim_y = [check_precision(lim_y2[0]),check_precision(lim_y2[1])]
    ticks_y = create_list_tick(lim_y)
    ax2 = modify_axes(ax2,label_x=['n','n'],lim_x = lim_x0,ticks_x=ticks_x,label_y=[r'd$_{600}$ [km]','y'],ticks_y=ticks_y,letter='[a]')

    lim_y = [check_precision(lim_y3[0]),check_precision(lim_y3[1])]
    ticks_y = create_list_tick(lim_y)
    ax3 = modify_axes(ax3,label_x=[r'$\gamma$ [1000 km]','y'],lim_x = lim_x0,ticks_x=ticks_x,label_y=[r'$\xi_{150}$ [%]','y'],ticks_y=ticks_y,letter='[e]')
    
    lim_y = [check_precision(lim_y4[0]),check_precision(lim_y4[1])]
    ticks_y = create_list_tick(lim_y)
    ax4 = modify_axes(ax4,label_x=['n','n'],lim_x = lim_x0,ticks_x=ticks_x,label_y=[r'$\xi_{450}$ [%]','y'],ticks_y=ticks_y,letter='[d]')

    lim_y = [check_precision(lim_y4[0]),check_precision(lim_y4[1])]
    ticks_y = create_list_tick(lim_y)
    ax5 = modify_axes(ax5,label_x=['n','n'],lim_x = lim_x0,ticks_x=ticks_x,label_y=[r'$\xi_{600}$ [%]','y'],ticks_y=ticks_y,letter='[c]')

    # Custom legend entries
    legend_elements = [
        Line2D([0], [0], marker='o', color='firebrick',
               linestyle='None', markersize=8,
               label='Non linear'),

        Line2D([0], [0], marker='o', color='cadetblue',
               linestyle='None', markersize=8,
               label='k = 3.1 [W/m/K]'),

        Line2D([0], [0], marker='o', color='olive',
               linestyle='None', markersize=8,
               label='k = 5 [W/m/K]'),
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=True,
        prop={
        'size': _TEXTFONTSIZE_,
        'weight': 'demibold'
    }
    )

    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent=TRANSPARENT)    



def test_picture_check(path2save:str, pdba:pd.DataFrame,pdbb:pd.DataFrame,pdbc:pd.DataFrame,crd:CoordinateSurface,petro_db:data_base_PT_whitney, figure_number:int)->None:
    def create_array(buf:pd.DataFrame)->np.ndarray:
        
        if len(buf.shape)>1:
            raise ValueError('The function accept only 1D dataframe')
        
        array = np.zeros([buf.size,len(buf.iloc[0])],dtype = np.float32)
        
        for i in range(buf.size):
            array[i,:] = buf.iloc[i]
    
    
        return array 
    
    def difference_surface(pdb_a:pd.DataFrame,pdb_b:pd.DataFrame)->np.ndarray:
        """Enclosure to find the dT array
        -> check if the same row indicate the same [Tp,vc,age]-> if so, perform 
        the difference-> update array. 

        Args:
            pdb_a (pd.DataFrame): Non linear database
            pdb_b (pd.DataFrame): Linear database

        Returns:
            final array with all the experiment
        """
        
        
        count = len(pdb_a['case_id'])
        keys = ['Tp','vc','age']
        array = np.zeros([count,len(pdb_a.loc[0,'T_S'])],dtype = np.float32) 
        n_exp = 0
        
        for i in range(count):
            same = False
            
            for k in keys: 
                if pdb_a.loc[i,k] != pdb_b.loc[i,k]:
                    same = False
                    break 
                else: 
                    same = True
            if same:
                array[i,:] = pdb_a.loc[i,'T_S']-pdb_b.loc[i,'T_S']
                n_exp = n_exp + 1
        
        print(f'Number of experiments {n_exp}, {n_exp/count:.1f}')
    
        return array 
    
    
    
    name_plot = f'Figure_{figure_number}.png'
    
    from matplotlib.collections import LineCollection
    
    fig = plt.figure(figsize=(16, 10))

    # Top row: 
    
    bx = 0.1
    by = 0.1 
    sy = 0.7
    dy = 0.1
    sx_top = 0.25 
    dx_top_a = 0.08 
    dx_top_b = 0.05 
    sx_bot = 0.25
    dx_bota = 0.04 
    dx_botb = 0.08
    top_row = by+sy+dy
    # --- 
    # Top row: figure ax0 -> NL slab surface // black line + scatter DB + band 80 km 
    #        : figure ax1 -> dT LN_k=3.1 - NL // black line + band 80 km 
    #        : figure ax2 -> dT LN_k=5 - NL // blacl line   + band 80 km 
    # Bot row: figure ax3 -> scatter plot dT vs gamma (thermal parameter) : cadet_blue => k=3.1 olive_green = k=5.0 
    #        : figure ax4 -> scatter plot dT vs Tp (potential temperature): 
    # --- 
    # ax5 => empty axes where to put the legend.  
    ax0 = fig.add_axes([bx,by,sx_top,sy])
    ax1 = fig.add_axes([bx+sx_top+dx_top_a,by,sx_top,sy])
    ax2 = fig.add_axes([bx+2*sx_top+dx_top_a+dx_top_b,by,sx_top,sy])
    
    temp_min_gr = 0 + 5 * np.abs(crd.z_slab)
    temp_max_gr = 0 + 20 * np.abs(crd.z_slab)
    temp_mean_gr = 0 + 10 * np.abs(crd.z_slab)
    p_fake = np.abs(crd.z_slab) * 9.81 * 3200 

    # Extract the temperature of the non linear experiments
    a0x = create_array( pdba[f'T_S'])
    a0y = create_array( pdba[f'P_S']) 
    
    b0x = create_array( pdbb[f'T_S'])
    b0y = create_array( pdbb[f'P_S']) 
    
    c0x = create_array( pdbc[f'T_S'])
    c0y = create_array( pdbc[f'P_S']) 
    

    
    for i in range(6424):
        ax0.plot(a0x[i,:],a0y[i,:],c='firebrick',alpha=0.01)
        ax1.plot(b0x[i,:],c0y[i,:],c='cadetblue',alpha=0.01)
        ax2.plot(c0x[i,:],c0y[i,:],c='olive',alpha=0.01)
        print(i)


    

    
    ax0.scatter(petro_db.T,petro_db.P,c='b',s=5.0,marker='x')


    
    # Homogeneise the layout of the figure. 
    
    ax0 = modify_axes(ax0,label_x=[r'T [$^{\circ}$C]','y'],lim_x=[0,1100],lim_y=[0,4],label_y=[r'P [GPa]','y'],ticks_x=[0.0,250,500,750,1000], ticks_y=[0,1,2,3,4],letter='[a]')
    
    
    
    
    ax1 = modify_axes(ax1,lim_y=[0,4],lim_x=[0,1100], ticks_y=[0,1,2,3,4],letter='[b]')
    ax2 = modify_axes(ax2,lim_y=[0,4],lim_x=[0,1100],ticks_y=[0,1,2,3,4],letter='[c]')
    

    # Custom legend entries
    legend_elements = [
        Line2D([0], [0], marker='o', color='firebrick',
               linestyle='None', markersize=8,
               label='Shear Heating Wet Qtz'),

        Line2D([0], [0], marker='o', color='cadetblue',
               linestyle='None', markersize=8,
               label='No Shear Heating'),

        Line2D([0], [0], marker='o', color='olive',
               linestyle='None', markersize=8,
               label='Constant shear heating '),
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        prop={
        'size': _TEXTFONTSIZE_,
        'weight': 'demibold'
    }
    )


    
    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent = TRANSPARENT)