
from scipy.interpolate import CubicSpline
from stonedfenicsx.create_mesh.aux_create_mesh import create_slab_surface
from numpy.typing import NDArray
import numpy as np 
from scipy.signal import savgol_filter as sgf 
import matplotlib.pyplot as plt 
import matplotlib.axes as mpl_axes
from typing import Optional
from matplotlib.ticker import FormatStrFormatter

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




def apply_Savitzky_Golay(yd,order,cf):
    
    
    yd_smooth = sgf(yd,cf,order)
    
    return yd_smooth

def function_bending_data(ell:NDArray[float],theta:NDArray[float])->float:
    
    cs=CubicSpline(ell,theta)
    
    
    def f2(x:float| NDArray[float]):
        
        return cs(x)
    

    return f2  

def curve_fitting(xd:NDArray[float],yd:NDArray[float],path:str , name:str)->NDArray[float]:

    xdo = xd 
    ydo = yd 
    
    
    theta, ell, xd,yd = compute_bending_angle_length(xd,yd)

    theta_sgf = apply_Savitzky_Golay(theta,2,100)        
    
    f_theta = function_bending_data(ell,theta_sgf)


    slab_top,theta_mean,ell_s = create_slab_surface(f_theta,np.min(yd),stp = 5.0)
  
    from scipy.spatial import cKDTree

    # reconstructed surface, clip to depth < -10 km
    xf = slab_top[:, 0]
    yf = slab_top[:, 1]
    xf = xf[yf < -10]
    yf = yf[yf < -10]

    # original (tail-augmented) curve, clipped the same way
    xr = xd[yd < -10]
    yr = yd[yd < -10]

    tree = cKDTree(np.column_stack([xf, yf]))
    dist, _ = tree.query(np.column_stack([xr, yr]))

    rmse = np.sqrt(np.mean(dist ** 2))
    max_dev = np.max(dist)
    i_max = np.argmax(dist)

    print(f"RMSE = {rmse:.2f} km, max = {max_dev:.2f} km "
          f"(at x={xr[i_max]:.1f}, depth={yr[i_max]:.1f} km)")

    
  
  
  
    fig = plt.figure(figsize=(10,6))
    bx = 0.13
    by = 0.15
    sx= 0.35
    dx = 0.15 
    sy = 0.8 
    ax0 = fig.add_axes([bx,by,sx,sy])
    ax1 = fig.add_axes([bx+dx+sx,by,sx,sy])
    ax0.plot(ell,theta*180/np.pi,c='k',alpha=0.1,label=r'$\theta$ original')
    ax0.plot(ell,theta_sgf*180/np.pi, c='forestgreen', label = r'$\theta$ filtered data')
    ax0.plot(ell_s,theta_mean*180/np.pi,c='firebrick',label=r'$\theta$  post processed')
    ax1.plot(xd,yd,c='k',alpha=0.1,label='original,coordinate corrected')
    ax1.plot(xdo,ydo,c='forestgreen',alpha=1.0,label='original')
    ax1.plot(slab_top[:,0],slab_top[:,1],c='r',label='post processed')
    ax0 = modify_axes(ax0,label_x=[r'$\ell$,[km]','y'],label_y=[r'$\theta$, [$^{\circ}$]','y'],lim_x=[0,1200],ticks_x=[0,400,800,1200],lim_y=[0,90],ticks_y=[0,30,60,90],letter='[a]')
    ax1 = modify_axes(ax1,label_x=[r'$x$,[km]','y'],label_y=[r'Depth, [km]','y'],lim_x=[0,1200],ticks_x=[0,400,800,1200],lim_y=[-500,0.0],ticks_y=[-500,-300,-150,0],letter='[b]')

    ax0.legend()
    ax1.legend()
    fig.savefig(f'{name}_surface.png',dpi=600)
    
    
    

    


    return slab_top,theta_mean




def compute_bending_angle_length(x:NDArray[float],z:NDArray[float])->tuple[NDArray[float],NDArray[float]]:
    """From the given subduction surface compute the bending angle and the total length
    of the slab

    Args:
        x (NDArray[float]): coordinate x of the top surface of the slab
        z (NDArray[float]): coordinate z of the top surface of the slab

    Returns:
        theta: bending angle of the top surface of the slab
        ell : total length of the surface of the slab. 
    """
    
    def cx(x0:float,x1:float)->float: 
        """Correct coordinate with the first point x/y coordinate

        Args:
            x0 (float): coordinate x or y of the first point 
            x1 (float): coordinate x or y of the second point

        Returns:
            float: dx/dy of the given couple of point
        """
        return x1-x0

    def distance(x0:NDArray[float],x1:NDArray[float])->float:
        """Compute the distance between two points 

        Args:
            x0 (NDArray[float]): point 1 [x,y] coordinate
            x1 (NDArray[float]): point 2 [x,y] coordinate

        Returns:
            float: distance between the two points 
        """
        return np.sqrt(cx(x0[0],x1[0])**2+cx(x0[1],x1[1])**2)

    pmax = len(x)-1 
    
    theta = np.zeros(pmax+1,dtype=np.float64)
    
    ell = np.zeros(pmax+1,dtype=np.float64)
    
    for i in range(pmax): 
        x0 = [x[i],z[i]]
        x1 = [x[i+1],z[i+1]]
        dx = cx(x0[0],x1[0])
        d  = distance(x0,x1)
        theta[i] = dx/d
        if i == pmax-1: 
            theta[i+1] = theta[i]  
        elif i ==0:
            # Compute the distance of the slab assuming that theta is constant and equal to the first segment of the slab           
            dy = cx(x0[1],x1[1])
            x_t = -x0[1]*(dx/dy)+x0[0]
            sx = cx(x_t,x0[0])
            dl = distance([x_t,0.0],x0)
            theta_0 = sx/dl
            ell[i] = dl            
        
        ell[i+1] = ell[i]+d 

    # Adding an additional portion of the vector 
    # Create a tail vector: 
    imax = np.int32(np.floor(dl/np.mean(np.diff(ell)))) 
    dl_tail = dl/(imax)
    
    theta_tail = np.zeros(imax)
    ell_tail = np.zeros(imax)
    x_tail = np.zeros(imax)
    y_tail = np.zeros(imax)

    theta = np.acos(theta)
    theta_0 = np.acos(theta_0)

    for i in range(imax): 
        
        if i == 0: 
            theta_tail[i] = theta_0 
            ell_tail[i] = 0.0 
            x_tail[i] = x_t
            y_tail[i] = 0.0 
        else:
            ell_tail[i] = ell_tail[i-1]+dl_tail
        if i < imax-1:
            theta_tail[i+1] = theta_0
            x_tail[i+1] = x_tail[i] + dl_tail*(np.cos(theta_tail[i]))
            y_tail[i+1] = y_tail[i] - dl_tail*(np.sin(theta_tail[i]))
        
    
        

    theta = np.concatenate((theta_tail,theta))
    ell = np.concatenate((ell_tail,ell))
    x = np.concatenate((x_tail,x))
    z = np.concatenate((y_tail,z))


    
    return theta, ell, x-x_t, z 


def read_file_slab(file_path:str)->tuple[NDArray[float],NDArray[float]]:
    
    from pathlib import Path
    
    slab= np.loadtxt(file_path)
    ax = slab[:,0]
    ay = -slab[:,1]
    dz = slab[:,2]
    
    ax = ax[ay<-10]
    dz = dz[ay<-10]
    ay = ay[ay<-10]
    
    path = Path(file_path)

    slab_top, theta_mean = curve_fitting(ax,ay,path.parent,path.stem)


    return slab_top*1000, theta_mean 




def generate_parallel_layer_subducting_plate(sx:NDArray[np.float64],
                                      sy:NDArray[np.float64],
                                      th:NDArray[np.float64],
                                      lt:float)->tuple[NDArray[np.float64],NDArray[np.float64]]:
    """
    Compute the coordinates of an internal layer surface within the subducting plate.
    
    Given the slab top surface coordinates and a local slab bending angle, this
    function constructs a parallel/offset surface at a distance `lt` (e.g., oceanic
    crust thickness or slab thickness), representing an internal layer boundary
    within the subducting plate.
    
    Parameters
    ----------
    sx : NDArray[np.float64]
        x-coordinates of the slab top surface.
    sy : NDArray[np.float64]
        y-coordinates of the slab top surface.
    th : float
        Local slab bending angle associated with the surface points (degrees or
        radians depending on the implementation; must be consistent with the
        trigonometric functions used).
    lt : float
        Layer thickness used to offset the surface (e.g., oceanic crust thickness
        or slab thickness) (SI units: [m]).
    
    Returns
    -------
    cx : NDArray[np.float64]
        x-coordinates of the layer-defining surface.
    cy : NDArray[np.float64]
        y-coordinates of the layer-defining surface.
    """
    from stonedfenicsx.create_mesh.aux_create_mesh import _find_e_node,_correct_
    cx = np.zeros([np.amax(sx.shape)])
    cy = np.zeros([np.amax(sx.shape)])
        # ── arc-length along the top surface ─────────────────────────────────────
    ds    = np.sqrt(np.diff(sx)**2 + np.diff(sy)**2)
    dth = np.diff(th)
    dth_ds = dth/ds                  # [rad / m]


    
    kappa = np.zeros_like(sx)
    kappa[1:] = np.abs(dth_ds)
    
    if 0.8 * np.min(1/kappa) < lt: 
        lt = np.floor(0.8 * np.min(1/kappa))
    
  
    # Loop over the interface of the slab and find the points on the top of the surface of the subduction channel: the point on the of the top of the channel are perpendicular to the slab interface#
    # Compute the top surface of the subduction channel
    cx = sx - lt*np.sin(th)
    cy = sy - lt*np.cos(th)

    e_node2,t_ex1 = _find_e_node(cx,cy,cx*0.0,-np.min(sy),flag=False)
    cx,cy,t  = _correct_(cx,cy,e_node2,-np.min(sy),cy*0.0,0.0)

    cx_n = cx[(cx>=0.0) & (cy>=np.min(sy))]
    cy_n = cy[(cx>=0.0) & (cy>=np.min(sy))]
    # Shift the first node: 
    
    if cx_n[0] != 0.0: 
        cx_n[0] = 0.0 
        cy_n[0] = -lt/np.cos(th[0])
    
    
    

    return cx_n,cy_n


def main():
    file_path= '/Users/wlnw570/Work/Leeds/Fenics_tutorial/examples/data/Tonga_slab.pz'
    read_file_slab(file_path=file_path)
    
    
    

if __name__ == '__main__':
    main()