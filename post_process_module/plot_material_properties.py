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
from pathlib import Path
from stonedfenicsx.config.phase_db import compute_thermal_properties
from stonedfenicsx.config.simulation_config import configure_simulation
from stonedfenicsx.config.input_parser import parse_input
import matplotlib.axes as mpl_axes
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from dataclasses import dataclass
from typing import Optional
from scipy.integrate import quad

# Conversion pt - cm 
_PT_CM_ = 0.035
# Ticks, label and text font size 
_TICKSFONTSIZE_ = 0.8/_PT_CM_
_LABELFONTSIZE_ = 1.0/_PT_CM_
_TEXTFONTSIZE_ = 0.65/_PT_CM_

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
    
    precision = '%.1f'
    
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
        current_x_ticks = ax.get_xticks()
        precision = '%.0f'
        for i in current_x_ticks: 
            if i-np.floor(i) != 0:
                precision = '%.1f'
                break 
        
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
        current_y_ticks = ax.get_yticks()
        precision = '%.0f'
        for i in current_y_ticks: 
            if i-np.floor(i) != 0:
                precision = '%.1f'
                break 
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
# ---



# --- 
temp = np.linspace(0,1600,num=100)+273.15

pres = np.linspace(0,20,num=10000)*1e9
# --- 
tt,pp = np.meshgrid(pres,temp)
# --- 
pkg_root = Path(__file__)

input_file = Path(pkg_root.parents[1], "input_tests.yaml")

inp,Ph = parse_input(input_file)

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

# --- 
inp.ctrl_tbc.recalculate = 1
inp.ctrl.pressure_dependency = 1
# ---
Ph.subducting_plate_mantle.rho0 = rho0_M
Ph.subducting_plate_mantle.name_capacity = capacity_nameM
Ph.subducting_plate_mantle.name_conductivity = conductivity_nameM
Ph.subducting_plate_mantle.name_alpha = alpha_nameM
Ph.subducting_plate_mantle.name_density = density_nameM
Ph.subducting_plate_mantle.radiative_conductivity = radiative_conductivity
Ph.subducting_plate_mantle.radiogenic_heat = H_r_M
Ph.subducting_plate_mantle.k =3.0 
Ph.oceanic_crust.rho0 = rho0_C
Ph.oceanic_crust.name_capacity = capacity_nameC
Ph.oceanic_crust.name_conductivity = conductivity_nameC
Ph.oceanic_crust.name_alpha = alpha_nameC
Ph.oceanic_crust.name_density = density_nameC
Ph.oceanic_crust.radiative_conductivity = radiative_conductivity
Ph.oceanic_crust.radiogenic_heat = H_r_LC 
Ph.oceanic_crust.k = 1.5 


Ph.wedge_mantle.rho0 = rho0_M
Ph.wedge_mantle.name_capacity = capacity_nameM 
Ph.wedge_mantle.name_conductivity = conductivity_nameM
Ph.wedge_mantle.name_alpha = alpha_nameM
Ph.wedge_mantle.name_density = density_nameM
Ph.wedge_mantle.radiative_conductivity = radiative_conductivity
Ph.wedge_mantle.radiogenic_heat = H_r_M 
Ph.overriding_mantle.rho0 = rho0_M 
Ph.overriding_mantle.name_capacity = capacity_nameM
Ph.overriding_mantle.name_conductivity = conductivity_nameM
Ph.overriding_mantle.name_alpha = alpha_nameM
Ph.overriding_mantle.name_density = density_nameM
Ph.overriding_mantle.radiative_conductivity = radiative_conductivity
Ph.overriding_mantle.radiogenic_heat = H_r_M 
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

ctrl_sim, mesh, pdb, sc = configure_simulation(ph_in=Ph,inp=inp)
# --- 
k = np.zeros_like(tt)
cp = np.zeros_like(tt)
rho = np.zeros_like(tt)
# --- 
k_c = np.zeros_like(tt)
cp_c = np.zeros_like(tt)
rho_c = np.zeros_like(tt)

# --- 

# --- 
for i,t in enumerate(temp):
    for j,p in enumerate(pres):
        cp[i,j],rho[i,j],k[i,j] = compute_thermal_properties(pdb,t/sc.temp,p/sc.stress,2)
        cp_c[i,j],rho_c[i,j],k_c[i,j] = compute_thermal_properties(pdb,t/sc.temp,p/sc.stress,1)


kappa = (k/cp/rho)*sc.length**2/sc.time
cp = cp *sc.cp
rho = rho * sc.rho 
k = k * sc.k 
kappa_ref = kappa[:, 0]
kappa_mean = np.zeros_like(kappa_ref)

for i in range(1, len(temp)):
    integral = np.trapezoid(kappa_ref[:i+1], temp[:i+1])
    kappa_mean[i] = integral / (temp[i] - temp[0])

# Define the first point
kappa_mean[0] = kappa_ref[0]

print(f'max diffusivity = {np.max(kappa):.3e}, min diffusivity = {np.min(kappa):.3e}')
# --- Plot material property as a function of temperature with shaded pressure-dependent property

kappa0 = (5.0/3300.0/1250.0)*1e6
kappa1 = (3.1/3300.0/1250.0)*1e6
k0 = 5.0 
k1 = 3.1 
rho_01 = 3300.0 
cp_01 = 1250.0 

# --- 
fig = plt.figure(figsize=[12,16])
bx = 0.15
by = 0.12
dx = 0.0 
dy = 0.05
sy = (1.0-dy*(4-1))/5.0
sx = 0.8
# --- 
ax0 = fig.add_axes([bx,by,sx,sy])
ax1 = fig.add_axes([bx,by+dy+sy,sx,sy])
ax2 = fig.add_axes([bx,by+2*dy+2*sy,sx,sy])
ax3 = fig.add_axes([bx,by+3*dy+3*sy,sx,sy])
# --- 
ax0.plot(temp-273.15,cp[:,0],c='forestgreen',linewidth=1.5)
ax0.axhline(cp_01,c='k',linewidth=2.0,linestyle=':')


ax1.plot(temp-273.15,rho[:,0],c='forestgreen',linewidth=1.5)
ax1.fill_between(temp-273.15,np.min(rho,1),np.max(rho,1),alpha=0.1,color='grey')
ax1.axhline(rho_01,c='k',linewidth=2.0,linestyle=':')

ax2.plot(temp-273.15,k[:,0],c='forestgreen',linewidth=1.5)
ax2.fill_between(temp-273.15,np.min(k,1),np.max(k,1),alpha=0.1,color='grey')
ax2.axhline(k0,c='k',linewidth=2.0,linestyle=':')
ax2.axhline(k1,c='k',linewidth=2.0,linestyle=':')


ax3.plot(temp-273.15,kappa[:,0]*1e6,c='firebrick',linewidth=1.5)
ax3.fill_between(temp-273.15,np.min(kappa,1)*1e6,np.max(kappa,1)*1e6,alpha=0.1,color='grey')
ax3.axhline(kappa0,c='k',linewidth=2.0,linestyle=':')
ax3.axhline(kappa1,c='k',linewidth=2.0,linestyle=':')
# --- 
ax0 = modify_axes(ax0,label_x=[r'Temperature, $[^{\circ}C]$','y'],lim_x = [50,1300],lim_y=[750,1400],label_y=[r'Cp [$J/kg/K$]','y'],ticks_y=[750,1000,1250,1400],letter='[a]')
ax1 = modify_axes(ax1,label_x=['n','n'],lim_x = [50,1300],lim_y=[3000,3800],label_y=[r'$\rho$ [$kg/m^{3}$]','y'],ticks_y=[3000,3200,3400,3600,3800],letter='[b]')
ax2 = modify_axes(ax2,label_x=['n','n'],lim_x = [50,1300],lim_y=[2.5,10.0],label_y=[r'k [$W/m/K$]','y'],ticks_y=[2.5,5.0,7.5,10.0],letter='[c]')
ax3 = modify_axes(ax3,label_x=['n','n'],lim_x = [50,1300],lim_y=[0.6,3.2],label_y=[r'$\kappa$ [$mm^2/s$]','y'],ticks_y=[0.8,1.6,2.4,3.2],letter='[d]')
# --- 
fig.savefig('thermal_property_mantle.png',dpi=800,transparent=True)
# --- 
kappa_c = (k_c/cp_c/rho_c)*sc.length**2/sc.time
cp_c = cp_c *sc.cp
rho_c = rho_c * sc.rho 
k_c = k_c * sc.k 
# --- 
fig = plt.figure(figsize=[12,16])
bx = 0.15
by = 0.12
dx = 0.0 
dy = 0.05
sy = (1.0-dy*(4-1))/5.0
sx = 0.8
# --- 
ax0 = fig.add_axes([bx,by,sx,sy])
ax1 = fig.add_axes([bx,by+dy+sy,sx,sy])
ax2 = fig.add_axes([bx,by+2*dy+2*sy,sx,sy])
ax3 = fig.add_axes([bx,by+3*dy+3*sy,sx,sy])
# --- 
ax0.plot(temp-273.15,cp[:,0],c='forestgreen',linewidth=1.5)
ax0.axhline(cp_01,c='k',linewidth=2.0,linestyle=':')


ax1.plot(temp-273.15,rho_c[:,0],c='forestgreen',linewidth=1.5)
ax1.fill_between(temp-273.15,np.min(rho_c,1),np.max(rho_c,1),alpha=0.1,color='grey')
ax1.axhline(rho_01,c='k',linewidth=2.0,linestyle=':')

ax2.plot(temp-273.15,k_c[:,0],c='forestgreen',linewidth=1.5)
ax2.fill_between(temp-273.15,np.min(k_c,1),np.max(k_c,1),alpha=0.1,color='grey')
ax2.axhline(k0,c='k',linewidth=2.0,linestyle=':')
ax2.axhline(k1,c='k',linewidth=2.0,linestyle=':')


ax3.plot(temp-273.15,kappa_c[:,0]*1e6,c='firebrick',linewidth=1.5)
ax3.fill_between(temp-273.15,np.min(kappa_c,1)*1e6,np.max(kappa_c,1)*1e6,alpha=0.1,color='grey')
ax3.axhline(kappa0,c='k',linewidth=2.0,linestyle=':')
ax3.axhline(kappa1,c='k',linewidth=2.0,linestyle=':')
# --- 
ax0 = modify_axes(ax0,label_x=[r'Temperature, $[^{\circ}C]$','y'],lim_x = [50,1300],lim_y=[750,1400],label_y=[r'Cp [$J/kg/K$]','y'],ticks_y=[750,1000,1250,1400],letter='[a]')
ax1 = modify_axes(ax1,label_x=['n','n'],lim_x = [50,1300],lim_y=[2600,3200],label_y=[r'$\rho$ [$kg/m^{3}$]','y'],ticks_y=[2600,2800,3000,3200],letter='[b]')
ax2 = modify_axes(ax2,label_x=['n','n'],lim_x = [50,1300],lim_y=[1.0,3.5],label_y=[r'k [$W/m/K$]','y'],ticks_y=[1.0,1.5,2.0,3.0,3.5],letter='[c]')
ax3 = modify_axes(ax3,label_x=['n','n'],lim_x = [50,1300],lim_y=[0.5,1.3],label_y=[r'$\kappa$ [$mm^2/s$]','y'],ticks_y=[0.5,0.75,1.0,1.25],letter='[d]')
# --- 
fig.savefig('thermal_property_crust.png',dpi=800,transparent=True)