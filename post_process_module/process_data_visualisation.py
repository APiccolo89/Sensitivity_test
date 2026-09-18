import gc
import os
import time as timing
import tracemalloc
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cmcrameri
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.collections import LineCollection
from numba import njit
from numpy.typing import NDArray

# dictionary of the groups 
dict_partial = {
    "nlsh": 0,
    "k3sh": 3.1,
    "k5sh": 5.0,
    "k3nsh": 3.1,
    "k5nsh": 5.0,
    "nlnsh": 0,
}


@dataclass(slots=True)
class SlabPlot2D:
    """Small data class containing the average temperature
    the variance and the standard deviation computed with
    the Welford alg. 
    """
    temp_mean_2d: np.ndarray[np.float32]
    temp_var_2d: np.ndarray[np.float32]
    temp_std_2d: np.ndarray[np.float32]
    xi: np.ndarray[np.float32]
    yi: np.ndarray[np.float32]


@dataclass(slots=True)
class CoordinateSurface:
    x_heat_flux: np.ndarray[np.float32]
    z_slab: np.ndarray[np.float32]
    z_ocean: np.ndarray[np.float32]


@njit
def compute_distance(p0: list, p1: list) -> float:
    return np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)

# ---
def interpolate_surfaces_slab(
    v: np.ndarray, x_or: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        v (np.ndarray): array to interpolate
        x_or (np.ndarray): original x_or

    Returns:
        tuple[np.ndarray,np.ndarray]: new values, new coordinates
    """

    def interpolate(
        x: np.ndarray, y: np.ndarray, lim: list
    ) -> tuple[np.ndarray, np.ndarray]:
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

    nv_interp = np.zeros((200, 1), dtype=np.float32)

    nv_interp, x_new = interpolate(x_or, v, lim=[np.min(x_or), np.max(x_or)])

    return nv_interp, x_new


def merge_data_base(name_suite, list_of_partial_files, plate, partial, recreate=True):
    output_file = f"{name_suite}_h5_files_{partial[0]}_{partial[1]}.h5"
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
                    T_mean_g = f_partial["mean_lc_tst"][:]
                    T_var_g = f_partial["var_lc_tst"][:]
                    n_g = f_partial["n_lc_tst"][()]
                else:
                    T_mean_g, T_var_g, n_g = update_Welfrod_single_batch(
                        n_g,
                        T_mean_g,
                        T_var_g,
                        f_partial["n_lc_tst"][()],
                        f_partial["mean_lc_tst"][:],
                        f_partial["var_lc_tst"][:],
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
        f_out.create_dataset("global_var", data=T_var_g / (n_g - 1))
        f_out.create_dataset("global_M2", data=T_var_g)
        f_out.create_dataset("global_count", data=n_g)

        print(
            f"Global mean: {np.nanmean(T_mean_g)}, Global std: {np.sqrt(np.nanmin(T_var_g / (n_g - 1)))} {np.sqrt(np.nanmax(T_var_g / (n_g - 1)))}, Global count: {n_g}"
        )

    return "Data merged successfully"


# ---------------------------------------------------------------------------------
@njit
def update_Welfrod_single_batch(
    n_g: int,
    mean_g: float,
    var_g: float,
    n_l: int,
    mean_l: np.ndarray,
    var_l: np.ndarray,
) -> tuple[float, float, int]:
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
    delta = mean_l - mean_g
    mean = mean_g + delta * (n_l / n_t)
    var = var_g + var_l + delta * delta * n_g * n_l / n_t

    return mean, var, n_t


# ---------------------------------------------------------------------------------
def create_arrays(Temp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    T_m = np.zeros_like(Temp)
    T_v = np.zeros_like(Temp)

    return T_m, T_v


# ---------------------------------------------------------------------------------
def compute_total_length(X_S: np.ndarray) -> float:

    return np.cumsum(np.sqrt(np.diff(X_S[0, :]) ** 2 + np.diff(X_S[1, :]) ** 2))


@njit
def interpolate_value_0D(
    ar: np.ndarray, x: np.ndarray, depth: float, ind: None | int
) -> float:
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
        ind = np.nonzero(depth >= x)[0][0]

    dv = ar[ind + 1] - ar[ind]
    dx = x[ind + 1] - x[ind]
    val = ar[ind] + (dv / dx) * (depth - ar[ind])

    return val, ind


# ---
def find_real_ell_z(
    isotherm: float, ell: NDArray, z: NDArray, temp: NDArray
) -> tuple[float, float]:
    def interpolate_value(ind, dtemp, temp0, dval, val0, val1):
        val = (dval / dtemp) * (isotherm - temp0) + val0
        return val

    ind_ = np.where(temp >= isotherm)[0][0]
    if temp[ind_] == isotherm:
        ell_p = ell[ind_]
        z_p = z[ind_]
    else:
        dtemp_ = temp[ind_] - temp[ind_ - 1]
        dell_ = ell[ind_] - ell[ind_ - 1]
        dz_ = z[ind_] - z[ind_ - 1]
        ell_p = interpolate_value(
            ind_, dtemp_, temp[ind_ - 1], dell_, ell[ind_ - 1], ell[ind_]
        )
        z_p = interpolate_value(ind_, dtemp_, temp[ind_ - 1], dz_, z[ind_ - 1], z[ind_])

    return ell_p, z_p


# ---------------------------------------------------------------------------------
def create_data_frame(
    data_table: pd.DataFrame, hdf5_file: str, partial, dict_column
) -> tuple[pd.DataFrame, namedtuple, namedtuple]:
    def extract_max_depth(f: h5py.File, case_key: str, isotherm: str) -> float:
        y = f[f"{case_key}/y_{isotherm}"][()]
        return y

    """Read the h5file to produce a data frame storing the relevant information
    for the sensitivity test.

    Args:
        data_table (pd.DataFrame): initial data table
        hdf5_file (str): path to the h5file
    Returns:
        Data Frame with all the information, named tuple with the coordinate system of the surface, name tuple with the 2D averages of the slab
    """
    rows = []
    ind = None

    with h5py.File(hdf5_file, "r") as f:
        time_A = timing.time()
        for i in f:
            time_C = timing.time()

            if i == "Xi" or i == "Yi":
                print("no")
                if i == "Xi":
                    xi = f[i][:, :]
                else:
                    yi = f[i][:, :]

            elif i in (
                "global_M2",
                "global_mean",
                "global_var",
                "global_count",
                "mean_lc_tst",
                "var_lc_tst",
                "n_lc_tst",
            ):
                continue

            else:
                case_id = i
                option_thermal = 1 if partial[0] == "nl" else 0
                max_dpth150 = extract_max_depth(f, i, "150")
                max_dpth300 = extract_max_depth(f, i, "300")
                max_dpth400 = extract_max_depth(f, i, "400")
                max_dpth450 = extract_max_depth(f, i, "450")
                max_dpth500 = extract_max_depth(f, i, "500")
                max_dpth550 = extract_max_depth(f, i, "550")
                max_dpth600 = extract_max_depth(f, i, "600")
                max_dpth700 = extract_max_depth(f, i, "700")
                max_dpth800 = extract_max_depth(f, i, "800")

                ell = compute_total_length(np.array(f[f"{i}/X_S"]))
                temp_slab = np.array(f[f"{i}/T_S"][:])
                z = np.array(f[f"{i}/X_S"])[1, :]

                slab_surf_150 = find_real_ell_z(150.0, ell, z, temp_slab)
                slab_surf_450 = find_real_ell_z(450.0, ell, z, temp_slab)
                slab_surf_600 = find_real_ell_z(600.0, ell, z, temp_slab)
                d_seismic = slab_surf_450[0] - slab_surf_150[0]
                d_seismic_450 = slab_surf_450[1]
                d_seismic_150 = slab_surf_150[1]

                # prepare the surface for being integrated into the database
                z_slab = np.array(f[f"{i}/X_S"][1, :])
                x_heat = np.array(f[f"{i}/x_top"][:])
                z_ocean = np.array(f[f"{i}/X_O"][1, :])

                # Interpolate value into a reduced array
                temp_s, z_slab_n = interpolate_surfaces_slab(
                    np.array(f[f"{i}/T_S"][:]), z_slab
                )
                pres_s, _ = interpolate_surfaces_slab(
                    np.array(f[f"{i}/P_S"][:]), z_slab
                )
                temp_o, z_ocean_n = interpolate_surfaces_slab(
                    np.array(f[f"{i}/T_O"][:]), z_ocean
                )
                pres_o, _ = interpolate_surfaces_slab(
                    np.array(f[f"{i}/P_O"][:]), z_ocean
                )
                tau_surf, _ = interpolate_surfaces_slab(
                    np.array(f[f"{i}/tau_shear"]), z_slab
                )
                heat_f, x_heat_n = interpolate_surfaces_slab(
                    np.array(f[f"{i}/heat_flux"][:]), x_heat
                )
                x_slab, _ = interpolate_surfaces_slab(
                    np.array(f[f"{i}/X_S"][0, :]), z_slab
                )

                if f"{i}/decoupling" in f:
                    dc = -f[f"{i}/decoupling"][()]
                else:
                    dc = -80
                temp_s_dc = np.interp(dc, z_slab_n, temp_s)
                pres_s_dc = np.interp(dc, z_slab_n, pres_s)
                dc_x = np.interp(dc, z_slab_n, x_slab)
                # Extract the maximum tau x position
                buf = f[f"{i}/tau_shear"][()]
                maxtau = np.max(buf)
                ind = np.where(buf == maxtau)[0]
                x_maxtau = np.interp(z_slab[ind], z_slab_n, x_slab)

                dict_data_base = {
                    "case_id": case_id,
                    "option_thermal": option_thermal,
                    "max_dpth150": max_dpth150,
                    "max_dpth300": max_dpth300,
                    "max_dpth400": max_dpth400,
                    "max_dpth450": max_dpth450,
                    "max_dpth500": max_dpth500,
                    "max_dpth550": max_dpth550,
                    "max_dpth600": max_dpth600,
                    "max_dpth700": max_dpth700,
                    "max_dpth800": max_dpth800,
                    "d_seismic_ell": d_seismic,
                    "d_seismic_450": d_seismic_450,
                    "d_seismic_150": d_seismic_150,
                    "slab_surf_150": slab_surf_150,
                    "slab_surf_450": slab_surf_450,
                    "slab_surf_600": slab_surf_600,
                    "T_S": temp_s,
                    "P_S": pres_s,
                    "T_O": temp_o,
                    "P_O": pres_o,
                    "Q": heat_f,
                    "T_s_dc": temp_s_dc,
                    "P_s_dc": pres_s_dc,
                    "tau_mean": f[f"{i}/tau_shear_mean"][()],
                    "tau_max": f[f"{i}/tau_shear_max"][()],
                    "tau_min": f[f"{i}/tau_shear_min"][()],
                    "tau_std": f[f"{i}/tau_shear_max"][()],
                    "tau_srf": tau_surf,
                    "maxtauxc": x_maxtau,
                    "maxdcxc": dc_x,
                }
                for m in dict_column:
                    if m == "case_index":
                        continue
                    dict_data_base[m] = f[f"{i}/{m}"][()]

                dict_data_base["thermal_parameter"] = (
                    dict_data_base["age"]
                    * dict_data_base["vc"]
                    * (1 / 1e5 * 1e6)
                    * np.sin(np.radians(45.0))
                )
                dict_data_base["k"] = dict_partial[partial[0]]

                rows.append(dict_data_base)
        slab_plot2D = SlabPlot2D(
            temp_mean_2d=f["global_mean"][:],
            temp_var_2d=f["global_var"][:],
            temp_std_2d=np.sqrt(f["global_var"][:]),
            xi=xi,
            yi=yi,
        )
        coordinate_surfaces = CoordinateSurface(
            x_heat_flux=x_heat_n, z_slab=z_slab_n, z_ocean=z_ocean_n
        )

    rows_df = pd.DataFrame(rows)
    data_table = pd.concat([data_table, rows_df], ignore_index=True)

    return data_table, coordinate_surfaces, slab_plot2D


# ----------------------- ------------------------------------------------------------------------------------------ ------------------ #
def wrapper_merging_h5(
    aux_path: str,
    partial: tuple,
    name_suite: str,
    csv: str = "",
    recreate: bool = False,
    n_workers: int = 100,
) -> tuple[pd.DataFrame, CoordinateSurface, SlabPlot2D]:
    """Function that merges and post process the data coming from the small h5
    files.

    Args:
        aux_path (str): path to the main Output_h5_files
        partial (tuple): [group name, steady state flag]
        name_suite (str): name of the suite
        csv (str): name of the auxilary csv list table where all the data are stored.
        recreate (bool, optional): Just force the recreate. Defaults to False.
        n_workers (int, optional): The number of processor used in the array job. Defaults to 100.

    Returns:
        tuple[pd.DataFrame,CoordinateSurface,SlabPlot2D]: data frame containing all the post-processed data,
                                                          coordinate system for plotting 1D profile and 2D plot
                                                          slab 2D meshes to visualise them
    """
    name_merged_file = f"{name_suite}_h5_files_{partial[0]}_{partial[1]}.h5"
    aux_path2 = aux_path / f"{name_suite}_h5_files_{partial[0]}_{partial[1]}"

    path2csv = Path(__file__).resolve().parents[1] / "generation_cases" / csv
    if recreate or os.path.isfile(name_merged_file) == False:
        list_partial_files = []
        for i in range(n_workers):
            list_partial_files.append(f"{aux_path2}/Data_Base_{i}.h5")

        print(
            merge_data_base(
                name_suite, list_partial_files, name_suite, partial, recreate=recreate
            )
        )
    else:
        print("Data Base is already there, and the flag to recreate has not been set")

    cases = pd.read_csv(path2csv)
    cases_columns = cases.columns
    args_column = {m: [] for m in cases_columns if m != "case_index"}
    keys = [
        "option_thermal",
        "max_dpth150",
        "max_dpth300",
        "max_dpth450",
        "max_dpth500",
        "max_dpth600",
        "max_dpth700",
        "max_dpth800",
        "d_seismic_ell",
        "d_seismic_dpth",
        "d_seismic_150",
        "slab_surf_150",
        "slab_surf_450",
        "slab_surf_600",
        "tau_mean",
        "tau_max",
        "tau_min",
        "tau_std",
        "Tau_s",
        "T_S",
        "P_S",
        "T_O",
        "P_O",
        "Q",
        "T_s_dc",
        "P_s_dc",
        "maxtauxc",
        "maxdcxc",
    ]
    args_test = args_column.copy()
    for i in keys:
        args_test[i] = []

    data_table = pd.DataFrame(args_test)

    data_table, coordinate_system, slab_2D = create_data_frame(
        data_table, name_merged_file, partial, args_column
    )

    return data_table, coordinate_system, slab_2D
