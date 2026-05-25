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
# latex Font 
TRANSPARENT = True 
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",   # Computer Modern
})



def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def plot_slab_variances(A, path2save: str,name_figure: str,name: str, DBa:None) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    name_plot = f'Figure_{name_figure}_{name}.png'
    
    def modify_axes(ax=None, label_x=None, lim=None, label_y=None):
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['left'].set_color('black')
        ax.spines['top'].set_color('black')

        ax.tick_params(axis='both', colors='black', direction='in', length=5, width=2.0)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['top'].set_linewidth(2.0)

        ax.grid(visible=True, which='major', axis='both',
                color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        if label_x is not None:
            ax.set_xlabel(label_x, fontsize=28, labelpad=10)
            ax.tick_params(axis='x', labelsize=15, labeltop=True)
        else:
            ax.tick_params(axis='x', labeltop=False)

        if label_y is not None:
            ax.set_ylabel(label_y, fontsize=28)
            ax.tick_params(axis='y', labelsize=15, labelleft=True)
        else:
            ax.tick_params(axis='y', labelleft=False)

        if lim is not None:
            ax.set_xlim(lim)

        return ax

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
    
    
class data_base_PT_whitney:
    def __init__(self):
        self.T  = [300, 550, 590, 400, 570, 610, 635, 540, 585, 510,
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
        self.P = [0.8, 1.2, 2.7, 0.5, 1.7, 2.1, 2.2, 2.1, 2.0, 2.1,
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

class Surfaces():
    def __init__(self, XS, T_S, X_O, T_O, x_top, heat_flux,non_linear,vc,case_ids,P_O,P_S,k_val):
        self.X_S = XS
        self.T_S = T_S
        self.X_O = X_O
        self.T_O = T_O
        self.P_O = P_O 
        self.P_S = P_S
        self.x_top = x_top
        self.vc = vc
        self.heat_flux = heat_flux
        self.non_linear = non_linear
        self.case_ids = case_ids
        self.k_val = k_val
    def post_process_data(self):
        def interpolate(x, y, lim):
            x_new = np.linspace(lim[0], lim[1], 200)
            if np.diff(x).min() <= 0:
                x, y = zip(*sorted(zip(x, y)))
        
            return np.interp(x_new, x, y), x_new
        
        num = len(self.non_linear)
        T_S_interp = np.zeros((num, 200))
        T_O_interp = np.zeros((num, 200))
        P_S_interp = np.zeros((num, 200))
        P_O_interp = np.zeros((num, 200))

        heat_flux_interp = np.zeros((num, 200))
        for i in range(num):
            T_S_interp[i,:], x_S_new = interpolate(self.X_S[i], self.T_S[i], lim=[np.min(self.X_S[i]), 0])
            T_O_interp[i,:], x_O_new = interpolate(self.X_O[i], self.T_O[i], lim=[np.min(self.X_O[i]), 0])
            p, _  = interpolate(self.X_O[i],self.P_O[i],lim=[np.min(self.X_O[i]), 0])
            if (p[0] <1) & (x_O_new[0] < -1.0): 
                p = p[::-1]
            
            P_O_interp[i,:] = p
            p = []
            p, _ =  interpolate(self.X_S[i],self.P_S[i],lim=[np.min(self.X_S[i]), 0])
            if (p[0] <1) & (x_S_new[0] < -1.0): 
                p = p[::-1]            

            P_S_interp[i,:] = p
        
            heat_flux_interp[i,:], x_heat_new = interpolate(self.x_top[i], self.heat_flux[i], lim=[np.min(self.x_top[i]), np.max(self.x_top[i])])
        
        self.T_S = T_S_interp
        self.X_S = x_S_new
        self.T_O = T_O_interp
        self.X_O = x_O_new
        self.P_S = P_S_interp
        self.P_O = P_O_interp
        self.heat_flux = heat_flux_interp
        self.x_top = x_heat_new        
        return self
    
#------------------------------------------------------------------------------
def plot_heart_flux(path2save:str, surface_data:Surfaces,name_figure:str,name:str):

    name_plot = f'Figure_{name_figure}_{name}.png'
    
    fig = plt.figure(figsize=(8, 6))
    
    ax0 = fig.add_axes([0.1,0.1,0.4,0.8])
    ax1 = fig.add_axes([0.54,0.1,0.4,0.8])
    
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_color('black')
    ax0.spines['bottom'].set_color('black')
    ax0.tick_params(axis='both', colors='black')
    ax0.spines['left'].set_linewidth(2.5)
    ax0.spines['bottom'].set_linewidth(2.5)
    
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

def plot_slab_surface(path2save:str, surface_data:Surfaces,surface:str,DB:data_base_PT_whitney, name_figure:str, name:str):
    
    name_plot = f'Figure_{name_figure}_{name}.png'
    
    from matplotlib.collections import LineCollection
    
    fig = plt.figure(figsize=(8, 6))

    ax0 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    ax1 = fig.add_axes([0.54, 0.1, 0.4, 0.8])
    if surface == 'slab':
        T_min_gr = 0 + 5 * np.abs(surface_data[1].X_S)
        T_max_gr = 0+ 20 * np.abs(surface_data[1].X_S)
        T_mean_gr = 0 + 10 * np.abs(surface_data[1].X_S)
        gr =  np.array(np.abs(surface_data[1].X_S) * 0.3)
        data = [surface_data[0].T_S+gr, surface_data[1].T_S+gr, surface_data[2].T_S+gr]
        X = [surface_data[0].P_S, surface_data[1].P_S, surface_data[2].P_S]#np.abs(surface_data.X_S)*3000*9.81/1e6
    else:
        T_min_gr = 0 + 5 * np.abs(surface_data[1].X_O)
        T_max_gr = 0+ 20 * np.abs(surface_data[1].X_O)
        T_mean_gr = 0 + 10 * np.abs(surface_data[1].X_O)        
        gr =  np.array(np.abs(surface_data[1].X_O) * 0.3)
        data = [d + gr for d in [surface_data[0].T_O, surface_data[1].T_O, surface_data[2].T_O]]
        X = [surface_data[0].P_O, surface_data[1].P_O, surface_data[2].P_O]#np.abs(surface_data.X_O)*3000*9.81/1e6

    group0 = data[2]
    group_k3 = data[0]
    group_k5 = data[1]
    

    for ax in (ax0, ax1):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.tick_params(axis='both', colors='black')
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['bottom'].set_linewidth(2.5)

    # convert lists of profiles into 2D arrays
    A0 = np.asarray(group0)   # shape: (n_profiles, n_points)
    
    K3 = np.asarray(group_k3)
    K5 = np.asarray(group_k5)
    
    # min / max / mean profiles
    T_smin  = np.min(A0, axis=0) + gr  
    T_max   = np.max(A0, axis=0) + gr 
    T_medk3  = np.median(K3, axis=0) + gr
    T_medk5 = np.median(K5,axis=0)+gr
    T_med2 = np.median(A0, axis=0) + gr

    # plot all curves
    for i in range(len(group_k5)):
        ax0.plot(group_k5[i] ,X[1][i] , color='olive', alpha=0.01, linewidth=0.5)
        ax0.plot(group_k3[i] ,X[0][i] , color='cadetblue', alpha=0.01, linewidth=0.5)


    for i in range(len(group_k5)):
        ax1.plot(group0[i],X[2][i], color='firebrick', alpha=0.01, linewidth=0.5)

    # shaded envelopes

    # mean lines
    ax0.plot(T_medk3,  X[0][0], color='cadetblue', linestyle='--', linewidth=1.5)
    ax0.plot(T_medk5,  X[1][0], color='olive', linestyle='--', linewidth=1.5)

    ax1.plot(T_med2, X[2][0], color='firebrick', linestyle='--', linewidth=1.5)

    
    ax0.axvline(x=300.0, color='black', lw=0.5, ls='--')
    ax0.axvline(x=400.0, color='black', lw=0.5, ls='--')
    ax0.axvline(x=500.0, color='black', lw=0.5, ls='--')
    ax0.axvline(x=600.0, color='black', lw=0.5, ls='--')
    
    ax1.axvline(x=300.0, color='black', lw=0.5, ls='--')
    ax1.axvline(x=400.0, color='black', lw=0.5, ls='--')
    ax1.axvline(x=500.0, color='black', lw=0.5, ls='--')
    ax1.axvline(x=600.0, color='black', lw=0.5, ls='--') 
    
    
    

    ax0.plot(T_min_gr, X[0][0], color='green', linestyle=':', linewidth=1.5)
    ax0.plot(T_max_gr, X[0][0], color='green', linestyle=':', linewidth=1.5)
    ax0.plot(T_mean_gr, X[0][0], color='green', linestyle=':', linewidth=1.5)
        
    ax1.plot(T_min_gr, X[2][0], color='green', linestyle=':', linewidth=1.5)
    ax1.plot(T_max_gr, X[2][0], color='green', linestyle=':', linewidth=1.5)
    ax1.plot(T_mean_gr,X[2][0], color='green', linestyle=':', linewidth=1.5)
    ax0.scatter(DB.T, DB.P, color='blue', s=10, label='Whitney et al. (2009)',alpha=0.5,zorder=10)
    ax1.scatter(DB.T, DB.P, color='blue', s=10, label='Whitney et al. (2009)',alpha=0.5,zorder=10)
    
    ax0.set_xlabel('Temperature [°C]')
    ax0.set_ylabel('Pressure [GPa]')
    ax1.set_xlabel('Temperature [°C]')

    ax0.set_ylim(0.0, 4.0)
    ax1.set_ylim(0.0, 4.0)
    ax0.set_xlim(0, 1400)
    ax1.set_xlim(0, 1400)
    ax1.set_yticklabels([])
    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent = TRANSPARENT)


    fig = plt.figure(figsize=(3, 8))

    ax0 = fig.add_axes([0.18, 0.1, 0.8, 0.8])

    for ax in (ax0, ax1):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.tick_params(axis='both', colors='black')
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['bottom'].set_linewidth(2.5)

    # convert lists of profiles into 2D arrays



    # shaded envelopes

    # mean lines
    ax0.plot(T_medk3-T_med2,  X[0][1], color='cadetblue', linestyle='--', linewidth=1.5)
    ax0.plot(T_medk5-T_med2,  X[0][0], color='olive', linestyle='--', linewidth=1.5)
    ax0.axvline(x=0.0, color='black', lw=0.8, ls='--')


    

    

    
    ax0.set_xlabel(r'$\Delta$ T [°C]')
    ax0.set_ylabel('Pressure [GPa]')
    ax1.set_xlabel('Temperature [°C]')

    ax0.set_ylim(0.0, 4.0)
    ax1.set_ylim(0.0, 4.0)
    ax1.set_yticklabels([])
    fig.savefig(f'{path2save}/inset_{name_plot}', dpi=600, transparent = TRANSPARENT)





def plot_suite(path2save:str, data_table:pd.DataFrame, lim:list=[0,100]):
    
    def modify_axes(ax=None,label_x=None,lim=None,label_y=None,isotherms=None):
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_color('black')
        
        ax.tick_params(axis='both', colors='black', direction='in', length=5, width=2.0)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['top'].set_linewidth(2.0)
        ax.grid(visible=True, which='major', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        if label_x is not None:
            ax.set_xlabel(label_x, fontsize=28, labelpad=10)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
            ax.xaxis.set_label_position('top')
            ax.set_xticks([lim[0], lim[1]])
            ax.set_xticklabels([f"{lim[0]:.1f}", f"{lim[1]:.1f}"])        
            ax.set_xlim(lim[0]-0.1*(lim[1]-lim[0]), lim[1]+0.1*(lim[1]-lim[0]))
        else:
            ax.set_xticklabels([])
            ax.set_xticks([lim[0], lim[1]])
            ax.set_xlim(lim[0]-0.1*(lim[1]-lim[0]), lim[1]+0.1*(lim[1]-lim[0]))



            
        ax.xaxis.set_ticks_position('top') # the rest is the same
        if label_y is True and isotherms is not None:
            string = f'$d_{{{isotherms}^\\circ C}}$ [km]'
            ax.set_ylabel(string, fontsize=28)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
        else:
            ax.set_yticklabels([])
   
        
        return ax
    
    def select_data_histo(dt:pd.DataFrame, isotherms:int):
        
        LNN = dt[(dt['option_thermal'] == 0)][f'max_dpth{isotherms}']
        LN_1 = dt[(dt['option_thermal'] == 0) & (dt['k']==3.0)][f'max_dpth{isotherms}']
        LN_2 = dt[(dt['option_thermal'] == 0) & (dt['k']==5.0)][f'max_dpth{isotherms}']
        NL = dt[(dt['option_thermal'] == 1)][f'max_dpth{isotherms}']
        bins = np.linspace(min(data_table[f'max_dpth{isotherms}']), max(data_table[f'max_dpth{isotherms}']), 100)
        
        LN = [LNN,LN_1,LN_2]
        
        return LN, NL, bins
        
    def plot_histogram(ax ,LNN, NL, bins):
        import scipy.stats as stats
        LN = LNN[0]
        LN_K3 = LNN[1]
        LN_K5 = LNN[2]

        kde1 = stats.gaussian_kde(LN)
        kde2 = stats.gaussian_kde(NL)
        kd3  = stats.gaussian_kde(LN_K3)
        kd5  = stats.gaussian_kde(LN_K5)

        
        LN_fit = kde1(bins)
        NL_fit = kde2(bins)
        LN_3 = kd3(bins)
        LN_5 = kd5(bins)
        
        ax.hist(NL,bins, alpha=0.1, color='firebrick',orientation='horizontal',density=True)
        ax.hist(LN_K3,bins,alpha=0.1, color='cadetblue',orientation='horizontal',density=True)
        ax.hist(LN_K5,bins,alpha=0.1, color='olive',orientation='horizontal',density=True)

        ax.plot(NL_fit,bins,color='firebrick', linewidth=1.2)
        ax.plot(LN_3,bins,color='cadetblue', linewidth=1.2)
        ax.plot(LN_5,bins,color='olive', linewidth=1.2)

        max_fit = np.max([np.max(NL_fit),np.max(LN_3),np.max(LN_5)])
        
        return ax,max_fit
    
    def create_scatter_plot(ax,dt:pd.DataFrame, field:str,isotherms:int):
        
        y0a = dt.loc[(dt['option_thermal'] == 0) & (dt['k'] == 3.0), f'max_dpth{isotherms}']
        y0b = dt.loc[(dt['option_thermal'] == 0) & (dt['k'] == 5.0), f'max_dpth{isotherms}']

        y1a = dt.loc[(dt['option_thermal'] == 1), f'max_dpth{isotherms}']
        
        x0a = dt.loc[(dt['option_thermal'] == 0) & (dt['k'] == 3.0), field]
        x0b = dt.loc[(dt['option_thermal'] == 0) & (dt['k'] == 5.0), field]
        
        
        x1a = dt.loc[(dt['option_thermal'] == 1), field]
        
        ax.scatter(x0a,y0a,c='cadetblue',marker='d',s=10)
        ax.scatter(x0b,y0b,c='olive',marker='d',s=10)
        ax.scatter(x1a,y1a,c='firebrick',marker='s',s=15)
        
        lm = np.min([min(x0a), min(x1a)])
        lM = np.max([max(x0a), max(x1a)])
        lim = [lm,lM]
        
        return ax,lim
    
    fig = plt.figure(figsize=(18, 18))
    bx = 0.1
    by = 0.1
    sx = 0.10
    dx = 0.02
    sy = 0.25
    dy = 0.02
    ax00 = fig.add_axes([bx,by+2*sy+2*dy,sx,sy])
    ax10 = fig.add_axes([bx+dx+sx,by+2*sy+2*dy,sx,sy])
    ax20 = fig.add_axes([bx+2*dx+2*sx,by+2*sy+2*dy,sx,sy])
    ax30 = fig.add_axes([bx+3*dx+3*sx,by+2*sy+2*dy,sx,sy])
    ax40 = fig.add_axes([bx+4*dx+4*sx,by+2*sy+2*dy,sx,sy])
    
    
    ax01 = fig.add_axes([bx,by+sy+dy,sx,sy])
    ax11 = fig.add_axes([bx+dx+sx,by+sy+dy,sx,sy])
    ax21 = fig.add_axes([bx+2*dx+2*sx,by+sy+dy,sx,sy])
    ax31 = fig.add_axes([bx+3*dx+3*sx,by+sy+dy,sx,sy])
    ax41 = fig.add_axes([bx+4*dx+4*sx,by+sy+dy,sx,sy])

    
    ax02 = fig.add_axes([bx,by,sx,sy])
    ax12 = fig.add_axes([bx+dx+sx,by,sx,sy])
    ax22 = fig.add_axes([bx+2*dx+2*sx,by,sx,sy])
    ax32 = fig.add_axes([bx+3*dx+3*sx,by,sx,sy])
    ax42 = fig.add_axes([bx+4*dx+4*sx,by,sx,sy])

    
    # Plot the histograms     
    ax00,max_fit0 = plot_histogram(ax00, *select_data_histo(data_table, 300))
    ax01,max_fit1 = plot_histogram(ax01, *select_data_histo(data_table, 500))
    ax02,max_fit2 = plot_histogram(ax02, *select_data_histo(data_table, 600))
    # Plot the scatter plots 
    ax10, lim10  = create_scatter_plot(ax10,data_table, field='vc',isotherms=300)
    ax20, lim20  = create_scatter_plot(ax20,data_table, field='Tp',isotherms=300)
    ax30, lim30  = create_scatter_plot(ax30,data_table, field='age',isotherms=300)
    ax40, lim40  = create_scatter_plot(ax40,data_table, field='thermal_parameter',isotherms=300)

    
    ax11, lim11 = create_scatter_plot(ax11,data_table, field='vc',isotherms=500)
    ax21, lim21 = create_scatter_plot(ax21,data_table, field='Tp',isotherms=500)
    ax31, lim31 = create_scatter_plot(ax31,data_table, field='age',isotherms=500)
    ax41, lim41 = create_scatter_plot(ax41,data_table, field='thermal_parameter',isotherms=500)

    
    ax12, lim02 = create_scatter_plot(ax12,data_table, field='vc',isotherms=600)
    ax22, lim12 = create_scatter_plot(ax22,data_table, field='Tp',isotherms=600)
    ax32, lim22 = create_scatter_plot(ax32,data_table, field='age',isotherms=600)
    ax42, lim42 = create_scatter_plot(ax42,data_table, field='thermal_parameter',isotherms=600)

    
    lim_vc = [3.0, 10.0]
    lim_Tp = [1350, 1425]
    lim_dc = [30.0, 100]
    lim_phi = [1000.0, 10000]
    
    ax10 = modify_axes(ax10, label_x=r'$v_c$ [cm/yr]',lim=lim_vc)
    ax20 = modify_axes(ax20, label_x=r'$T_p$ [°C]',lim=lim_Tp)
    ax30 = modify_axes(ax30, label_x=r'age [Myr]',lim=lim_dc)
    ax40 = modify_axes(ax40, label_x=r'$\gamma$ [km]',lim=lim_phi)

    
    ax11 = modify_axes(ax11, label_x=None,lim=lim_vc)
    ax21 = modify_axes(ax21, label_x=None,lim=lim_Tp)
    ax31 = modify_axes(ax31, label_x=None,lim=lim_dc)
    ax41 = modify_axes(ax41, label_x=None,lim=lim_phi)

    
    ax12 = modify_axes(ax12, label_x=None,lim=lim_vc)
    ax22 = modify_axes(ax22, label_x=None,lim=lim_Tp)
    ax32 = modify_axes(ax32, label_x=None,lim=lim_dc)
    ax42 = modify_axes(ax42, label_x=None,lim=lim_phi)

    maxF = np.max([max_fit0,max_fit1,max_fit2])
    maxF = np.round(maxF,2)
    
    ax00 = modify_axes(ax00, label_x=r'$f$ ,[]',label_y=True,lim=[0.0,maxF+0.1*maxF],isotherms=300)    
    ax01 = modify_axes(ax01, label_x=None,label_y=True,lim=[0.0,maxF+0.1*maxF],isotherms=500)
    ax02 = modify_axes(ax02, label_x=None,label_y=True,lim=[0.0,maxF+0.1*maxF],isotherms=600)
    
    fig.savefig(f'{path2save}/Summary_Plots.png', dpi=600,transparent = TRANSPARENT)
    

    
    
    
    
    


# ----------------------- Functions ------------------ #

def create_histograms(path2save:str,data_table:pd.DataFrame,isotherms:int):
    max_depth_LN = data_table[(data_table['option_thermal'] == 0)][f'max_dpth{isotherms}']
    max_depth_NL = data_table[(data_table['option_thermal'] == 1)][f'max_dpth{isotherms}']

    fig = plt.figure(figsize=(8, 6))
    bins = np.linspace(min(data_table[f'max_dpth{isotherms}']), max(data_table[f'max_dpth{isotherms}']), 20)
    plt.hist(max_depth_LN, bins, alpha=0.5, label='Linear material properties')
    plt.hist(max_depth_NL, bins, alpha=0.5, label='Non linear material properties')
    plt.legend(loc='upper right')
    plt.xlabel('Depth (km)')
    plt.title(rf'Histogram of Maximum Depth at {isotherms} $^\circ$C')
    fig.savefig(f'{path2save}/{plate}_ISO{isotherms}_histogram.png', dpi=300)
    print(f'{300} deg C, histograms plotted')
    
def create_scatter_plot(path2save:str, data_table:pd.DataFrame, isotherms:int):
    
    def modify_axes(ax=None,label_x=None,label_y=None):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        
        ax.tick_params(axis='both', colors='black')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_xlabel(label_x)
        if label_y != None: 
            ax.set_ylabel(label_y)
        else: 
            ax.set_ylabel([])
            ax.set_yticklabels([])
        
        return ax 
    
    
    def extract_relevant_data(dt:pd.DataFrame, field:str): 

        v0a = dt.loc[(dt['option_thermal'] == 0), field]
        v1a = dt.loc[(dt['option_thermal'] == 1), field]
    
    
        return v0a,v1a
    # ---------
    def create_scatter_plot(ax=None,field=None,yval=None,db=None): 
        
        v0a,v1a = extract_relevant_data(db,field)
        
        axa = ax.scatter(v0a,yval[0],c='blue',marker='x',s=10)
        axc = ax.scatter(v1a,yval[1],c='firebrick',marker='x',s=15)
        
        return ax,axa,axc 
        
    #----------------------------------------------------------
        
    y_data = [data_table.loc[(data_table['option_thermal'] == 0) ,f'max_dpth{isotherms}'],
                data_table.loc[(data_table['option_thermal'] == 1),f'max_dpth{isotherms}']]
    
    fig = plt.figure(figsize=(18, 9))
    
    bx = 0.1
    by = 0.1
    sx = 0.28
    dx = 0.01 
    sy = 0.35
    dy = 0.1 
    
    # phi + Rheology y label 
    ax0 = fig.add_axes([bx,by,sx,sy])
    # Edis 
    ax1 = fig.add_axes([bx+dx+sx,by,sx,sy])
    # Vdis
    ax2 = fig.add_axes([bx+2*dx+2*sx,by,sx,sy])
    # Tp y label
    ax3 = fig.add_axes([bx,by+sy+dy,sx,sy])
    # dc 
    ax4 = fig.add_axes([bx+dx+sx,by+sy+dy,sx,sy])
    # trans
    #ax5 = fig.add_axes([bx+2*dx+2*sx,by+sy+dy,sx,sy])
    
    ax0 = modify_axes(ax = ax0, label_x = r'$v_c$ [cm/yr]',label_y=rf'$d_{{isotherms}}$')
    
    ax1 = modify_axes(ax = ax1, label_x = r'$E^{W}_{\mathrm{dis}}$ $[\frac{kJ}{mol}]$',label_y=None)
    
    ax2 = modify_axes(ax = ax2, label_x = r'$V^{W}_{\mathrm{dis}}$  $[\frac{cm^3}{mol}]$',label_y=None)

    ax3 = modify_axes(ax = ax3, label_x = r'$T_{p}$, $[^{\circ}C]$',label_y=rf'$d_{{isotherms}}$')    
    
    ax4 = modify_axes(ax = ax4, label_x = r'$z_{dc}$ $[km]$',label_y=None)

    #ax5 = modify_axes(ax = ax5, label_x = r'$\delta_{trans}$ $[km]$',label_y=None)
    
    ax0, ax0a,ax0b = create_scatter_plot(ax=ax0,field='vc',yval=y_data,db=data_table)
    
    ax1, ax1a,ax1b = create_scatter_plot(ax=ax1,field='Edis',yval=y_data,db=data_table)
    
    ax2, ax2a,ax2b = create_scatter_plot(ax=ax2,field='Vdis',yval=y_data,db=data_table)
    
    ax3, ax3a,ax3b = create_scatter_plot(ax=ax3,field='Tp',yval=y_data,db=data_table)
    
    ax4, ax4a,ax4b = create_scatter_plot(ax=ax4,field='decoupling',yval=y_data,db=data_table)
    
    fig.savefig(fname=f'{path2save}/Scatter_d{isotherms}.png',dpi=600)
#-----------------------------------------------------------------------------------
def merge_data_base(list_of_partial_files, plate,partial,recreate=True):
    output_file = f"Data_Base_{plate}_def_{partial}.h5"
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
                    if partial == 3:
                        if key == 'Xi' or key == 'Yi' or key == 'global_M2' or key == 'global_mean' or key == 'global_var' or key == 'n_lc_tst' or key == 'mean_lc_tst' or key == 'var_lc_tst':
                            print('no')
                        else:
                            f_out[f'{key}/option_thermal'][()] = 2
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
def create_data_frame(data_table:pd.DataFrame, hdf5_file:str):
    def extract_max_depth(f:h5py.File, case_key:str, isotherm:str):
        x = f[f'{case_key}/x_{isotherm}'][:]
        y = f[f'{case_key}/y_{isotherm}'][:]
        ind = np.where(min(y) == y)[0][0]
        x_min = x[ind]
        return min(y),x_min

    """_summary_

    Args:
        data_table (pd.DataFrame): _description_
        hdf5_file (str): _description_

    Returns:
        _type_: _description_
    """
    T_slabs = []
    X_slabs = []
    T_oceanics = []
    X_oceanics = []
    x_tops = []
    heat_fluxes = []
    non_linear = []
    vc_l = []
    case_ids = []
    P_O = []
    P_S = []
    k_val = []
    
    
    cnt_k3 = 0 
    cnt_k5 = 0 
    cnt_NL = 0
    rows = []
    Slab_plot2D = namedtuple('Slab_plot2D',['T_mean','T_var','T_std','Xi','Yi'])

    
    with h5py.File(hdf5_file, 'r') as f:
        time_A = timing.time()
        for i in f:
            time_C = timing.time()

            if i == 'Xi' or i == 'Yi' :
                print('no')
                if i == 'Xi':
                    Xi = (f[i][:,:])
                else:
                    Yi = (f[i][:,:])
                    
            elif i == 'global_M2' or i == 'global_mean' or i == 'global_var' or i == 'global_count' or i == 'mean_lc_tst' or i == 'var_lc_tst' or i == 'n_lc_tst':
                print('no')
                continue
    
            else:
                
                case_id = i
                k = float(f[f'{i}/k'][()])
                vc = float(f[f'{i}/vc'][()])
                Tp = float(f[f'{i}/Tp'][()])
                age = float(f[f'{i}/age'][()])
                thermal_parameter = age * vc * (1/1e5 * 1e6)* np.sin(np.radians(45.0))
                option_thermal = int(f[f'{i}/option_thermal'][()])

                max_dpth300, x_dpth300 = extract_max_depth(f, i, '300')
                max_dpth400, x_dpth400 = extract_max_depth(f, i, '400')
                max_dpth500, x_dpth500 = extract_max_depth(f, i, '500')
                max_dpth600, x_dpth600 = extract_max_depth(f, i, '600')
                max_dpth700, x_dpth700 = extract_max_depth(f, i, '700')
                max_dpth800, x_dpth800 = extract_max_depth(f, i, '800')

                rows.append({
                    "case_id": case_id,
                    "k": k,
                    "age": age,
                    "vc": vc,
                    "Tp": Tp,
                    "thermal_parameter": thermal_parameter,
                    "option_thermal": option_thermal,
                    "max_dpth300": max_dpth300,
                    "max_dpth400": max_dpth400,
                    "max_dpth500": max_dpth500,
                    "max_dpth600": max_dpth600,
                    "max_dpth700": max_dpth700,
                    "max_dpth800": max_dpth800,
                    "x_dpth300": x_dpth300,
                    "x_dpth400": x_dpth400,
                    "x_dpth500": x_dpth500,
                    "x_dpth600": x_dpth600,
                    "x_dpth700": x_dpth700,
                    "x_dpth800": x_dpth800,
                })
                T_slabs.append(np.array(f[f'{i}/T_S'][:]))
                X_slabs.append(np.array(f[f'{i}/X_S'][1,:]))
                T_oceanics.append(np.array(f[f'{i}/T_O'][:]))
                X_oceanics.append(np.array(f[f'{i}/X_O'][1,:]))
                P_S.append(np.array(f[f'{i}/P_S'][:]))
                P_O.append(np.array(f[f'{i}/P_O'][:]))
                x_tops.append(np.array(f[f'{i}/x_top'][:]))
                heat_fluxes.append(np.array(f[f'{i}/heat_flux'][:]))
                non_linear.append(option_thermal)
                if option_thermal== 0: 
                    k_val.append(k)
                else: 
                    k_val.append(-1)
                vc_l.append(float(f[f'{i}/vc'][()]))
                case_ids.append(i)
        slab_2D = Slab_plot2D(T_mean=f['global_mean'][:], T_var=f['global_var'][:], T_std=np.sqrt(f['global_var'][:]), Xi=Xi, Yi=Yi)

                


            
    

    rows_df = pd.DataFrame(rows)
    data_table = pd.concat([data_table, rows_df], ignore_index=True)

    
    #print(f'Total experiments: k == 3 -> {cnt_k3},k == 5 -> {cnt_k5}, NL --> {cnt_NL}')
    
    
    # Generate variance of the data along the ensemble axis
            
    surface_data = Surfaces(XS=X_slabs, T_S=T_slabs, X_O=X_oceanics, T_O=T_oceanics, x_top=x_tops, heat_flux=heat_fluxes,non_linear=non_linear,vc=vc_l, case_ids=case_ids,P_O=P_O,P_S=P_S,k_val=k_val)
    return data_table,surface_data, slab_2D

def wrapper_merging_h5(aux_path:str, partial:int, plate:str, recreate=False):
    aux_path = os.path.join(aux_path, f'Sensitivity_h5_files_{partial}')
    list_partial_files = []
    for i in range(100):
        list_partial_files.append(f'{aux_path}/Data_Base_{i}.h5')
    
    print(merge_data_base(list_partial_files, plate, partial, recreate=recreate))
    
    data_table = pd.DataFrame({
        "case_id": [],
        "k": [],
        "age": [],
        "vc": [],
        "Tp": [],
        "option_thermal": [],
        'max_dpth300': [],
        'max_dpth400': [],
        'max_dpth500': [],
        'max_dpth600': [],
    })
    
    data_table, surface_data, slab_2D = create_data_frame(data_table,   f'Data_Base_{plate}_def_{partial}.h5')
    
    return data_table, surface_data, slab_2D
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
        
        a = a[f'max_dpth{isotherms}']
        b = b[f'max_dpth{isotherms}']
        c = c[f'max_dpth{isotherms}']

        if scale == True: 
            d_k3 = np.abs((c - a)/c)
            d_k5 = np.abs((c - b)/c)
        else:
            d_k3 = ((c - a))
            d_k5 = ((c - b))

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
    
    ax4.scatter(k3_common['thermal_parameter'],d_300_k3, c='cadetblue', marker='d', s=10, label='k=3.0')
    ax4.scatter(k5_common['thermal_parameter'],d_300_k5, c='olive', marker='d', s=10, label='k=5.0')
    ax5.scatter(k3_common['thermal_parameter'],d_500_k3, c='cadetblue', marker='d', s=10, label='k=3.0')  
    ax5.scatter(k5_common['thermal_parameter'],d_500_k5, c='olive', marker='d', s=10, label='k=5.0')
    ax6.scatter(k3_common['thermal_parameter'],d_600_k3, c='cadetblue', marker='d', s=10, label='k=3.0')
    ax6.scatter(k5_common['thermal_parameter'],d_600_k5, c='olive', marker='d', s=10, label='k=5.0')
    
    ax4 = modify_axes(ax4, label_x=r'$\gamma$ [km]', label_y=r'$err_{abs}$ [n.d.]', lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=300)
    ax5 = modify_axes(ax5, label_x=r'$\gamma$ [km]', label_y=None, lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=500)
    ax6 = modify_axes(ax6, label_x=r'$\gamma$ [km]', label_y=None, lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=600)

    ax1 = modify_axes(ax1, label_x=r'$err_{abs}$ [n.d.]', label_y=r'f []', lim_x=[0.0, 1.0], lim_y=None, isotherms=300)
    ax2 = modify_axes(ax2, label_x=r'$err_{abs}$ [n.d.]', label_y=None, lim_x=[0.0, 1.0], lim_y=None, isotherms=500)
    ax3 = modify_axes(ax3, label_x=r'$err_{abs}$ [n.d.]', label_y=None, lim_x=[0.0, 1.0], lim_y=None, isotherms=600)


    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent=TRANSPARENT)
    # Extract data 
    
#---------------------------------------------------------------------------------    
    
def plot_depth(path2save:str, DB:pd.DataFrame, scale=False, name='PT'):
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
    

    def plot_histogram(ax ,k3, k5, nl,bins):
        import scipy.stats as stats


        kde1 = stats.gaussian_kde(k3)
        kde2 = stats.gaussian_kde(k5)
        kde3 = stats.gaussian_kde(nl)
        k3_fit = kde1(bins)
        k5_fit = kde2(bins)
        nl_fit = kde3(bins)

        k3_fit = k3_fit 
        k5_fit = k5_fit 
        nl_fit = nl_fit
        
        LN_3 = k3_fit
        LN_5 = k5_fit
        LN_NL = nl_fit

        ax.hist(k5,bins, alpha=0.1, color='olive',density=True)
        ax.hist(k3,bins,alpha=0.1, color='cadetblue',density=True)
        ax.hist(nl,bins,alpha=0.1, color='firebrick',density=True)

        ax.plot(bins,LN_3,color='cadetblue', linewidth=1.2)
        ax.plot(bins,LN_5,color='olive', linewidth=1.2)
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

    d_300_k3, d_300_k5, d_300_NL = extract_data(k3_common,k5_common,NL_common, 300,scale=scale)
    d_500_k3, d_500_k5, d_500_NL = extract_data(k3_common,k5_common,NL_common, 500,scale=scale)
    d_600_k3, d_600_k5, d_600_NL = extract_data(k3_common,k5_common,NL_common, 600,scale=scale)

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
    
    ax1,_  = plot_histogram(ax1, d_300_k3, d_300_k5, d_300_NL, bins=np.linspace(np.min([d_300_k3.min(), d_300_k5.min()]), np.max([d_300_k3.max(), d_300_k5.max()]), 50))
    ax2,_  = plot_histogram(ax2, d_500_k3, d_500_k5, d_500_NL, bins=np.linspace(np.min([d_500_k3.min(), d_500_k5.min()]), np.max([d_500_k3.max(), d_500_k5.max()]), 50))
    ax3,_  = plot_histogram(ax3, d_600_k3, d_600_k5, d_600_NL, bins=np.linspace(np.min([d_600_k3.min(), d_600_k5.min()]), np.max([d_600_k3.max(), d_600_k5.max()]), 50))
    
    ax4.scatter(k3_common['thermal_parameter'],d_300_k3, c='cadetblue', marker='d', s=0.1, label='k=3.0')
    ax4.scatter(k5_common['thermal_parameter'],d_300_k5, c='olive', marker='d', s=0.1, label='k=5.0')
    ax4.scatter(NL_common['thermal_parameter'],d_300_NL, c='firebrick', marker='d', s=5.1, label='NL')
    ax5.scatter(k3_common['thermal_parameter'],d_500_k3, c='cadetblue', marker='d', s=0.1, label='k=3.0')  
    ax5.scatter(k5_common['thermal_parameter'],d_500_k5, c='olive', marker='d', s=0.1, label='k=5.0')
    ax5.scatter(NL_common['thermal_parameter'],d_500_NL, c='firebrick', marker='d', s=5.1, label='NL')
    ax6.scatter(k3_common['thermal_parameter'],d_600_k3, c='cadetblue', marker='d', s=0.1, label='k=3.0')
    ax6.scatter(k5_common['thermal_parameter'],d_600_k5, c='olive', marker='d', s=0.1, label='k=5.0')
    ax6.scatter(NL_common['thermal_parameter'],d_600_NL, c='firebrick', marker='d', s=5.1, label='NL')
    
    ax4 = modify_axes(ax4, label_x=r'$\gamma$ [km]', label_y=r'$err_{abs}$ [n.d.]', lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=300)
    ax5 = modify_axes(ax5, label_x=r'$\gamma$ [km]', label_y=None, lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=500)
    ax6 = modify_axes(ax6, label_x=r'$\gamma$ [km]', label_y=None, lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=600)

    ax1 = modify_axes(ax1, label_x=r'$err_{abs}$ [n.d.]', label_y=r'f []', lim_x=[0.0, 1.0], lim_y=None, isotherms=300)
    ax2 = modify_axes(ax2, label_x=r'$err_{abs}$ [n.d.]', label_y=None, lim_x=[0.0, 1.0], lim_y=None, isotherms=500)
    ax3 = modify_axes(ax3, label_x=r'$err_{abs}$ [n.d.]', label_y=None, lim_x=[0.0, 1.0], lim_y=None, isotherms=600)


    fig.savefig(f'{path2save}/Depth_{name}_{scale}.png', dpi=600, transparent=TRANSPARENT)
    # Extract data 
    
    
    
def plot_distance_isotherm_seismic(path2save:str, DB:pd.DataFrame, scale=False, name='PT',name_figure:str=''):
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

    d_300_k3, d_300_k5, d_300_NL = extract_data(k3_common,k5_common,NL_common, 300,scale=scale)
    d_600_k3, d_600_k5, d_600_NL = extract_data(k3_common,k5_common,NL_common, 600,scale=scale)

    D_k3 = np.abs((d_600_k3 - d_300_k3))
    D_k5 = np.abs((d_600_k5 - d_300_k5))
    D_NL = np.abs((d_600_NL - d_300_NL))
    D_k3 = D_k3-D_NL
    D_k5 = D_k5-D_NL

    fig = plt.figure(figsize=(10, 6))
    bx = 0.1
    by = 0.1
    sx = 0.25
    dx = 0.06
    sy = 0.35
    dy = 0.1 
    ax1 = fig.add_axes([bx,by,sx,sy])
    ax2 = fig.add_axes([bx+dx+sx,by,sx,sy])
    ax4 = fig.add_axes([bx,by+sy+dy,sx,sy])
    ax5 = fig.add_axes([bx+dx+sx,by+sy+dy,sx,sy])
    
    ax1,_  = plot_histogram(ax1, D_k3, D_k5, None, bins=np.linspace(np.min([D_k3.min(), D_k5.min()]), np.max([D_k3.max(), D_k5.max()]), 50))
    ax2,_  = plot_histogram(ax2, D_k3/D_NL, D_k5/D_NL, None, bins=np.linspace(np.min([D_k3/D_NL.min(), D_k5/D_NL.min()]), np.max([D_k3/D_NL.max(), D_k5/D_NL.max()]), 50))
    ax4.scatter(k3_common['thermal_parameter'],D_k3, c='cadetblue', marker='d', s=0.1, label='k=3.0')
    ax4.scatter(k5_common['thermal_parameter'],D_k5, c='olive', marker='d', s=0.1, label='k=5.0')
    ax5.scatter(k3_common['thermal_parameter'],D_k3/D_NL, c='cadetblue', marker='d', s=0.1, label='k=3.0')  
    ax5.scatter(k5_common['thermal_parameter'],D_k5/D_NL, c='olive', marker='d', s=0.1, label='k=5.0')

    
    ax4 = modify_axes(ax4, label_x=r'$\gamma$ [km]', label_y=r'$err_{abs}$ [n.d.]', lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=300)
    ax5 = modify_axes(ax5, label_x=r'$\gamma$ [km]', label_y=None, lim_x=[0.0, 1.0], lim_y=[0.0, 1.0], isotherms=500)

    ax1 = modify_axes(ax1, label_x=r'$err_{abs}$ [km]', label_y=r'f []', lim_x=[0.0, 1.0], lim_y=None, isotherms=300)
    ax2 = modify_axes(ax2, label_x=r'$err_{abs}$ [n.d.]', label_y=None, lim_x=[0.0, 1.0], lim_y=None, isotherms=500)
    ax2.text(0.02, 0.97, r'$\rm{err}(T) = (d(T)^{k}-d(T)^{NL}) / d(T)^{NL}$',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent=TRANSPARENT)
    # Extract data 
            
# plot the variance along the thermal parameter
def plot_slab_versus_thermalpar(path2save: str, DB: pd.DataFrame, scale=False, name='PT',name_figure:str=''):
    
    name_plot = f'{name_figure}_{name}.png'

    def modify_axes(ax=None):
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
        
        return ax


    def compute_values(db: pd.DataFrame, Tl0: float, Tl1: float, isotherms: int, scale=False):
        db = db[(db['thermal_parameter'] >= Tl0) & (db['thermal_parameter'] < Tl1)]
        
        d = db[f'max_dpth{isotherms}'].median()
        std = db[f'max_dpth{isotherms}'].std()
        vmin = db[f'max_dpth{isotherms}'].min()
        vmax = db[f'max_dpth{isotherms}'].max()
        return d, std, vmin, vmax

    def extract_arrays(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, isotherms: int, scale=False):
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
            a_array[i, 0], a_array[i, 1], a_array[i, 2], a_array[i, 3] = compute_values(a, Tl0, Tl1, isotherms)
            b_array[i, 0], b_array[i, 1], b_array[i, 2], b_array[i, 3] = compute_values(b, Tl0, Tl1, isotherms)
            c_array[i, 0], c_array[i, 1], c_array[i, 2], c_array[i, 3] = compute_values(c, Tl0, Tl1, isotherms)
            newT[i] = (Tl0 + Tl1) / 2.0  # centro del bin
        
        return a_array, b_array, c_array, newT

    def plot_values(ax, a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, isotherms: int,
                    scale=False, labels=('k=3.0', 'k=5.0', 'NL')):
        a_array, b_array, c_array, x = extract_arrays(a, b, c, isotherms, scale=scale)
        dat = (a_array, b_array, c_array)
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

    k3 = DB[(DB['option_thermal'] == 0) & (DB['k'] == 3.0)]
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

    fig = plt.figure(figsize=(6, 12))
    bx = 0.15
    by = 0.1
    dy = 0.01 
    sx = 0.8
    sy = 0.28
    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx, by + sy + dy, sx, sy])
    ax2 = fig.add_axes([bx, by + 2 * (sy + dy), sx, sy])
    
    ax = fig.gca()
    
    ax2 = plot_values(ax2, k3_common, k5_common, NL_common, isotherms=700)
    ax2.set_xticklabels([])  # Rimuove le etichette sull'asse x
    ax2.set_ylabel(r'Mean depth [km] T=700 [C]', fontsize=14)

    
    ax1 = plot_values(ax1, k3_common, k5_common, NL_common, isotherms=500)
    ax1.set_xticklabels([])  # Rimuove le etichette sull'asse x
    ax1.set_ylabel(r'Mean depth [km] T=600 [C]', fontsize=14)

    
    ax0 = plot_values(ax0, k3_common, k5_common, NL_common, isotherms=300)
    ax0.set_xlabel(r'$\gamma$ [km]', fontsize=14)
    ax0.set_ylabel(r'Mean depth [km] T=300 [C]', fontsize=14)
    ax0 = modify_axes(ax0)
    ax1 = modify_axes(ax1)
    ax2 = modify_axes(ax2)
    fig.savefig(f'{path2save}/Variance_{name}_{scale}_alt.png', dpi=600, transparent=TRANSPARENT)    
#-----------------------------------------------------------------------------
def plot_slab_versus_thermalpar_normalised(path2save: str, DB: pd.DataFrame, scale=False, name='PT',name_figure=''):
    name_plot = f'Figure_{name_figure}_{name}_{scale}.png'
    
    #-----------------------------------------------------------------------------
    
    def modify_axes(ax=None):
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
        
        return ax

    #-----------------------------------------------------------------------------

    def compute_values(db: pd.DataFrame,db_NL: pd.DataFrame, Tl0: float, Tl1: float, isotherms: int, scale=False):
        db = db[(db['thermal_parameter'] >= Tl0) & (db['thermal_parameter'] < Tl1)]
        
        buf0 = (db_NL[f'max_dpth{isotherms}']-db[f'max_dpth{isotherms}'])/db_NL[f'max_dpth{isotherms}']
        d = buf0.median()
        std = buf0.std()
        vmin = buf0.min()
        vmax = buf0.max()
        
        return d, std, vmin, vmax

    #-----------------------------------------------------------------------------

    def extract_arrays(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, isotherms: int, scale=False):
        thermal_parameters = a['thermal_parameter']
        t_p_m = np.min(thermal_parameters)
        t_p_M = np.max(thermal_parameters)
        TPar = np.linspace(t_p_m, t_p_M, 100)
        n = len(TPar) - 1  # np.len() non esiste — si usa len() builtin di Python
        
        a_array = np.zeros([n, 4])
        b_array = np.zeros([n, 4])
        newT = np.zeros(n)
        
        for i in range(n):
            Tl0 = TPar[i]
            Tl1 = TPar[i + 1]
            a_array[i, 0], a_array[i, 1], a_array[i, 2], a_array[i, 3] = compute_values(a,c ,Tl0, Tl1, isotherms)
            b_array[i, 0], b_array[i, 1], b_array[i, 2], b_array[i, 3] = compute_values(b,c ,Tl0, Tl1, isotherms)
            newT[i] = (Tl0 + Tl1) / 2.0  # centro del bin
        
        return a_array, b_array, newT

    #-----------------------------------------------------------------------------

    def plot_values(ax, a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, isotherms: int,
                    scale=False, labels=('k=3.0', 'k=5.0', 'NL')):
        a_array, b_array, x = extract_arrays(a, b, c, isotherms, scale=scale)
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
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    
    k3 = DB[(DB['option_thermal'] == 0) & (DB['k'] == 3.0)]
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

    fig = plt.figure(figsize=(6, 12))
    bx = 0.15
    by = 0.1
    dy = 0.01 
    sx = 0.8
    sy = 0.28
    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx, by + sy + dy, sx, sy])
    ax2 = fig.add_axes([bx, by + 2 * (sy + dy), sx, sy])
    
    ax = fig.gca()
    
    ax2 = plot_values(ax2, k3_common, k5_common, NL_common, isotherms=600)
    ax2.set_xticklabels([])  # Rimuove le etichette sull'asse x
    ax2.set_ylabel(r'err(600)', fontsize=14)

    
    ax1 = plot_values(ax1, k3_common, k5_common, NL_common, isotherms=500)
    ax1.set_xticklabels([])  # Rimuove le etichette sull'asse x
    ax1.set_ylabel(r'err(500)', fontsize=14)

    
    ax0 = plot_values(ax0, k3_common, k5_common, NL_common, isotherms=300)
    ax0.set_xlabel(r'$\gamma$ [km]', fontsize=14)
    ax0.set_ylabel(r'err(300)', fontsize=14)
    ax0 = modify_axes(ax0)
    ax1 = modify_axes(ax1)
    ax2 = modify_axes(ax2)
    ax0.text(0.02, 0.97, r'$\rm{err}(T) = (d(T)^{k}-d(T)^{NL}) / d(T)^{NL}$',
        transform=ax0.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    fig.savefig(f'{path2save}/{name_plot}', dpi=600, transparent=TRANSPARENT)    



# ----------------------- Main code ------------------ #
if __name__ == "__main__":
    plate = 'Sensitivity'

    print("Merging partial data files...")

    aux_path = f'/mnt/scratch/wlnw570/Japan_ensemble/Sensitivity_test'

    data_table_0, surface_data_0, slab_2D_0 = wrapper_merging_h5(aux_path, partial=0, plate=plate)
    data_table_1, surface_data_1, slab_2D_1 = wrapper_merging_h5(aux_path, partial=1, plate=plate)
    data_table_2, surface_data_2, slab_2D_2 = wrapper_merging_h5(aux_path, partial=2, plate=plate)
    data_table_3, surface_data_3, slab_2D_3 = wrapper_merging_h5(aux_path, partial=3, plate=plate)

    slab_2D = (slab_2D_0, slab_2D_1, slab_2D_2)
    slab_2D_alt = (slab_2D_0, slab_2D_1, slab_2D_3)
    
    data_table = pd.concat([data_table_0, data_table_1, data_table_2], ignore_index=True)
    data_table_alt = pd.concat([data_table_0, data_table_1, data_table_3], ignore_index=True)
    
    surface_data = (surface_data_0.post_process_data(), surface_data_1.post_process_data(), surface_data_2.post_process_data())


    
    path2save = f'/mnt/scratch/wlnw570/Japan_ensemble/Sensitivity_test/Result_{plate}'    
    if not os.path.isdir(path2save):
        os.mkdir(path2save)
        print(f'Created the folder {path2save}')

    print(f'Saving picture into {path2save}...')
    
    
    # Figure 0 : the mean of the slab (2D)
    plot_slab_variances(slab_2D,path2save,name_figure='Figure0',name='PT',DBa = None)
    plot_slab_variances(slab_2D,path2save,name_figure='Figure0_alt',name='PT',DBa=data_table)

    #plot_slab_variances(slab_2D_alt,path2save,name_figure='Figure0',name='T')
    
    DB = data_base_PT_whitney()

    
    # Figure 1 suite: the mean of top surface of the slab and ocean 
    plot_slab_surface(path2save=path2save, surface_data=surface_data, surface='slab', DB=DB,name_figure='Figure1',name='PT')
    plot_slab_surface(path2save=path2save, surface_data=surface_data, surface='ocean', DB=DB,name_figure='Figure1_b',name='PT')

    
    plot_distance_to_isotherms(path2save=path2save, DB=data_table,scale=False,name='PT',name_figure='figure2_b')

    plot_distance_to_isotherms(path2save=path2save, DB=data_table_alt,scale=False,name='T',name_figure='figure2_b')

    plot_distance_isotherm_seismic(path2save=path2save, DB=data_table,scale=True,name='PT',name_figure='figure3')
    plot_distance_isotherm_seismic(path2save=path2save, DB=data_table_alt,scale=True,name='T',name_figure='figure3')

        
    plot_slab_versus_thermalpar(path2save=path2save, DB=data_table_alt,scale=True,name='T',name_figure='figure4')
    plot_slab_versus_thermalpar(path2save=path2save, DB=data_table,scale=True,name='PT',name_figure='figure4')

    plot_slab_versus_thermalpar_normalised(path2save=path2save, DB=data_table_alt,scale=True,name='T',name_figure='figure5')
    plot_slab_versus_thermalpar_normalised(path2save=path2save, DB=data_table,scale=True,name='PT',name_figure='figure5')

    # produce the data table
    #---------------------------------------------------------
    # Create figure for the histograms and scatter plots.
    # Plot heat fluxes
    plot_heart_flux(path2save=path2save, surface_data=surface_data, name_figure='Figure6',name='PT')




