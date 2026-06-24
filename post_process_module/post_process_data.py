import pandas as pd
import numpy as np
import h5py
import argparse
import os
import csv
from collections import namedtuple
import gc 
import time as timing
from functools import wraps
from post_process_module.data_extractor import Test
import json
import fcntl
from datetime import datetime
from pathlib import Path


def save_dataset(group, name, data, compress=True):
    if compress and hasattr(data, "shape") and data.shape != ():
        group.create_dataset(
            name,
            data=data,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    else:
        group.create_dataset(name, data=data)


def save_namedtuple_fields(group, data, aliases=None, skip_fields=None):
    """Save all values from a namedtuple using its field names."""
    aliases = aliases or {}
    skip_fields = skip_fields or set()

    for field_name in getattr(data, "_fields", ()):
        if field_name in skip_fields:
            continue
        dataset_name = aliases.get(field_name, field_name)
        save_dataset(group, dataset_name, getattr(data, field_name))


def write_case_log(log_path, case_index, id_batch, success, status, path_h5):
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "case_index": int(case_index),
        "id_batch": int(id_batch),
        "success": bool(success),
        "status": str(status),
        "path_h5": str(path_h5),
    }

    with open(log_path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
#-------------------------------------------------
def timing_function(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        time_A = timing.time()
        result = fun(*args, **kwargs)
        time_B = timing.time()
        dt = time_B - time_A
        if dt > 60.0:
            m, s = divmod(dt, 60)
            print(f".  {fun.__name__} took {m:.2f} min and {s:.2f} sec")
        elif dt > 3600.0:
            m, s = divmod(dt, 60)
            h, m = divmod(m, 60)
            print(f".  {fun.__name__} took {dt/3600:.2f} hr, {m:.2f} min and {s:.2f} sec")
        else:
            print(f".  {fun.__name__} took {dt:.2f} sec")
        return result
    return wrapper


#--------------------------------------------------
def test_success(name_test: str):
    """Check if the test produced the expected output

    Args:
        name_test (str): path2test

    Returns:
        _type_: Bool: False if the test did not produce the expected output, True if it did
    """
    if os.path.isfile(f'{name_test}/Steady_state.h5')==False:
        return False
    else:
        return True
#------------------------------------------------
def extract_slab_surface(T:Test)->np.ndarray:
    
    T_slab_surf = T._interpolate_data_slab('SteadyState.Temp')
    
    return T_slab_surf
#-------------------------------------------------
def compute_tau_rheological_law(eps:float,temp_slab:np.ndarray,pres_slab:np.ndarray,tag:str)->np.ndarray:

    from stonedfenicsx.material_property.phase_db import _check_rheological 
    
    temp_slab = temp_slab + 273.15 
    pres_slab = pres_slab * 1e9
    A = _check_rheological(tag)
    water = (np.exp(- (A.EH20+A.VH20*pres_slab)/(A.R * temp_slab))/np.exp(- (A.EH20+A.VH20*1e5)/(A.R * 298.15)))
    
    cds = A.B * np.exp(-(A.E + pres_slab * A.V)/(A.R * temp_slab)) * water **(A.r)
    
    tau_viscous = cds ** (-1/A.n) * eps**(1/A.n)



    return tau_viscous

#-------------------------------------------------
def extract_shearzone_data(temp_slab:np.ndarray,pres_slab:np.ndarray,args:argparse.Namespace)->np.ndarray:
    """Function that extract the stress at slab interface.

    Args:
        temp_slab (np.ndarray): temperature at the surface of the slab
        PS (np.ndarray): pressure at the surface of the slab
        args (args.NamedSpace): input data -> to improve, need to find way to save the metadata of the simulation

    Returns:
        np.ndarray: tau_eff
    """
    
    vel = args.vc*1e-2/365.25/60/60/24
    eps = 0.5 * (vel *  1 /500) 
    if args.shear_heating == 'Parametrised':
        bdisl = args.eps_ref/args.tau_ref**(args.n_ref)
        tau_min = args.tau_min
        tau_v = bdisl**(-1/args.n_ref)*eps**(1/args.n_ref)
        pres_crit = tau_v/np.sin(np.radians(args.phi))/1e9
        p_norm = (pres_slab-pres_crit)/pres_crit
        tau_viscous = tau_min + (tau_v-tau_min)*np.exp(-5.0*p_norm)
        
    else: 
        tau_viscous = compute_tau_rheological_law(eps,temp_slab,pres_slab,args.shear_heating)   
        
    tau_fr = pres_slab*1e9*np.sin(np.radians(args.phi))
    tau_eff = tau_viscous * np.tanh(tau_fr/tau_viscous)
    tau_val = [np.nanmean(tau_eff),np.nanstd(tau_eff),np.nanmedian(tau_eff),np.nanmin(tau_eff),np.nanmax(tau_eff)]
    
    
    return tau_eff,tau_val


#--------------------------------------------------
def extract_slab_oceanic_surface_Temperature(T: Test)->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    def surface_data(surface):
        if surface == 'slab':
            x = T.MeshData.X[(T.MeshData.mesh_tag == 9) | (T.MeshData.mesh_tag == 8),0]
            y = T.MeshData.X[(T.MeshData.mesh_tag == 9) | (T.MeshData.mesh_tag == 8),1]
            T_surface = T.Data_raw.SteadyState.Temp[(T.MeshData.mesh_tag == 9) | (T.MeshData.mesh_tag == 8)]
            P_surface = T.Data_raw.SteadyState.LitPres[(T.MeshData.mesh_tag == 9) | (T.MeshData.mesh_tag == 8)]
        
        elif surface == 'ocean':
            x = T.MeshData.X[T.MeshData.mesh_tag == 10,0]
            y = T.MeshData.X[T.MeshData.mesh_tag == 10,1]
            T_surface = T.Data_raw.SteadyState.Temp[T.MeshData.mesh_tag == 10]
            P_surface = T.Data_raw.SteadyState.LitPres[T.MeshData.mesh_tag == 10]
        sort = np.argsort(x)
        x = x[sort]
        y = y[sort]
        T_surface = T_surface[sort]
        P_surface = P_surface[sort]
        X = np.array([x, y],dtype=np.float32)  
        T_surface = np.float32(T_surface)
        P_surface = np.float32(P_surface)   
        return X, T_surface, P_surface

    # Extract the mesh data: 
    X_S, T_S, P_S = surface_data('slab')
    X_O, T_O, P_O = surface_data('ocean')
    
    return X_S, X_O, T_S, T_O, P_S, P_O
#-------------------------------------------------
def extract_heat_flux(T: Test)->tuple[np.ndarray, np.ndarray]:
    
    q = T.Data_raw.SteadyState.qy


    x = T.MeshData.X[:,0]
    y = T.MeshData.X[:,1]
    cond = (x>=0.0) & (x<600.0) & (np.isclose(y, 0.0, atol=1e-3))
    q = q[cond==True]
    x_top = x[cond==True]
    sort = np.argsort(x_top)
    x_top = x_top[sort]
    q = q[sort]
    
    x_top = np.float32(x_top)
    q = np.float32(q)
    
    
    return x_top, q
#-------------------------------------------------
def extract_isotherm(X, Y, T, level):
    import matplotlib.pyplot as plt
    T = T + 0.3 * np.abs(Y) 
    cs = plt.contour(X, Y, T, levels=[level])
    paths = cs._paths
    cx = paths[0].vertices[:, 0]
    cy = paths[0].vertices[:, 1]
    plt.close()
    sorted_indices = np.argsort(cx)
    cx = np.float32(cx)
    cy = np.float32(cy)
    del cs 
    gc.collect()
    return cx, cy
#------------------------------------------------
@timing_function
def extract_relevant_data(path:str,args:argparse.Namespace)->list:
    T = Test(path_2_test=path)
    Slab_surface = extract_slab_surface(T=T)
    x_150, y_150 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 150.0)
    x_200, y_200 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 200.0)
    x_250, y_250 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 250.0)
    x_300, y_300 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 300.0)
    x_350, y_350 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 350.0)
    x_400, y_400 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 400.0)
    x_450, y_450 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 450.0)
    x_500, y_500 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 500.0)
    x_550, y_550 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 550.0)
    x_600, y_600 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 600.0)
    x_700, y_700 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 700.0)
    x_800, y_800 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 800.0)
    x_900, y_900 = extract_isotherm(T.MeshData.Xi, T.MeshData.Yi, Slab_surface, 900.0)



    X_S, X_O, T_S, T_O, P_S, P_O = extract_slab_oceanic_surface_Temperature(T)
    tau_slab, tau_val = extract_shearzone_data(T_S,P_S,args)
    Slab_surface = extract_slab_surface(T=T)
    x_top, heat_flux = extract_heat_flux(T)
    data_extracted = namedtuple('data_extracted',['x_150','y_150',
                                                  'x_200','y_200',
                                                  'x_250','y_250',
                                                  'x_300','y_300',
                                                  'x_350','y_350',
                                                  'x_400','y_400',
                                                  'x_450','y_450',
                                                  'x_500','y_500',
                                                  'x_550','y_550',
                                                  'x_600','y_600',
                                                  'x_700','y_700',
                                                  'x_800','y_800',
                                                  'x_900','y_900'
                                                  ,'X_S','X_O',
                                                  'T_S','T_O',
                                                  'P_S','P_O',
                                                  'x_top','heat_flux'
                                                  ,'Slab_surface'
                                                  ,'Xi','Yi','tau_slab','tau_shear_mean','tau_shear_std','tau_shear_median','tau_shear_min','tau_shear_max'])
    list_data = data_extracted(x_150, y_150,
                               x_200, y_200,
                               x_250, y_250,
                               x_300, y_300,
                               x_350, y_350,
                               x_400, y_400,
                               x_450, y_450,
                               x_500, y_500,
                               x_550, y_550,
                               x_600, y_600,
                               x_700, y_700,
                               x_800, y_800,
                               x_900, y_900,                               
                               X_S, X_O,
                               T_S, T_O,
                               P_S, P_O,
                               x_top, heat_flux,
                               Slab_surface,
                               T.MeshData.Xi, T.MeshData.Yi,tau_slab,tau_val[0],tau_val[1],tau_val[2],tau_val[3],tau_val[4]) 
    
    del T 
    gc.collect()   


    return list_data
    
#-------------------------------------------------
@timing_function
def save_test_data(name_suite:str = 'Sensitivity' ,arguments:argparse.Namespace = None, pathtotest:tuple[str,str] = None):

    print(f"Saving test data for case {arguments.case_index} to Data_Base.h5")
    
    main_dir = Path(__file__).resolve().parents[1]
    path2save = main_dir / f"{name_suite}_h5_files_{arguments.id_db}"
    
    if not os.path.isdir(path2save):
        os.mkdir(path2save)

    print(f"Test data will be saved in {path2save}")
    
    save_grid = True    
    # Paths
    name_test = pathtotest[1]
    real_test = os.path.join(pathtotest[0], name_test)

    print(f"Extracting relevant data for case {arguments.case_index} and saving to Data_Base.h5")
    print(f"name_test: {name_test}")
    test_successful = test_success(real_test)
    if test_successful:
        list_data = extract_relevant_data(path=f'{real_test}',args=arguments)
        
        print(f"  Print Meta_data for case {arguments.case_index} and {name_test}")
        for key, value in vars(arguments).items():
            print(f"  - {key}: {value}")    
       
        print(f"Case {arguments.case_index}: {name_test} - extracted isotherms at 400, 600, 800 C")
        print(f"  - 300 C: {min(list_data.y_300)} km")
        print(f"  - 400 C: {min(list_data.y_400)} km")
        print(f"  - 500 C: {min(list_data.y_500)} km")
        print(f"  - 600 C: {min(list_data.y_600)} km")
        print(f"  - 700 C: {min(list_data.y_700)} km")
        print(f"  - 800 C: {min(list_data.y_800)} km")
        print(f"  - 900 C: {min(list_data.y_900)} km")
        print(f"  --------------------------- ")



        with h5py.File(f'{path2save}/Data_Base_{arguments.id_batch}.h5', 'a') as f:
            print('  - Saving data to Data_Base.h5')
            print(f"{path2save}/Data_Base_{arguments.id_batch}.h5 ")
            if save_grid:
                if "Xi" not in f:
                    f.create_dataset("Xi", data=list_data.Xi)

                if "Yi" not in f:
                    f.create_dataset("Yi", data=list_data.Yi)

                if "Xi" in f and "Yi" in f:
                    save_grid = False
                

            if name_test in f:
                print(f"Overwriting existing case group: {name_test}")
                del f[name_test]
            grp = f.create_group(name_test)
            
            for key, value in vars(arguments).items():
                if isinstance(value, str):
                    grp.create_dataset(
                        key,
                        data=value,
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                else:
                    grp.create_dataset(key, data=value)

            save_namedtuple_fields(
                grp,
                list_data,
                aliases={"tau_slab": "tau_shear"},
                skip_fields={"Xi", "Yi"},
            )
            
            # update the local mean and variance for the Welford algorithm 

            if "n_lc_tst" not in f:
                f.create_dataset("n_lc_tst", data=0) 
                T_mean, T_var = create_arrays(list_data.Slab_surface)
                f.create_dataset("mean_lc_tst", data=T_mean)
                f.create_dataset("var_lc_tst", data=T_var)
            
            n_lc_tst = f["n_lc_tst"][()]
            n_lc_tst += 1
            f["n_lc_tst"][()] = n_lc_tst
            T_mean = f["mean_lc_tst"][()]
            T_var = f["var_lc_tst"][()]
            n_lc_tst = f["n_lc_tst"][()]
            T_mean, T_var = update_Welfrod_ALG(list_data.Slab_surface,T_mean,T_var,n_lc_tst)
            f["mean_lc_tst"][()] = T_mean
            f["var_lc_tst"][()] = T_var
            print(f"  - Updated mean and variance for the Welford algorithm for case {arguments.case_index} and {n_lc_tst} tests")
            del list_data 
            gc.collect()
            write_case_log(log_path=f'{path2save}/case_log.jsonl', case_index=arguments.case_index, id_batch=arguments.id_db, success=True, status="Test successful", path_h5=f'{path2save}/Data_Base_{arguments.id_db}.h5')

        
    return 0

def update_Welfrod_ALG(Temp:np.ndarray,mean:np.ndarray,var:np.ndarray,n:int):
    delta = Temp-mean 
    mean = mean + delta / n 
    delta2 = Temp-mean 
    var = var + delta*delta2 
    
    return mean,var
 
def create_arrays(Temp:np.ndarray)->tuple[np.ndarray,np.ndarray]: 
    
    T_m = np.zeros_like(Temp)
    T_v = np.zeros_like(Temp)
    
    return T_m, T_v



