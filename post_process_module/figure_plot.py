"""
Module that collects all the function that assist the plot
of the results. In this module, the data are also post-processed
and transformed when it is required. 
"""
import csv

import matplotlib.axes as mpl_axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from .process_data_visualisation import CoordinateSurface

# Conversion pt - cm
_PT_CM_ = 0.035
# Ticks, label and text font size
_TICKSFONTSIZE_ = 0.5 / _PT_CM_
_LABELFONTSIZE_ = 0.6 / _PT_CM_
_TEXTFONTSIZE_ = 0.5 / _PT_CM_
# latex Font
TRANSPARENT = True
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",  # Computer Modern
    }
)
# --- 
# ---
def nice_ticks(vmin: float, vmax: float, n: int = 4,
               steps: tuple = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)) -> tuple[list, list]:
    """Return exactly n round ticks covering [vmin, vmax].

    The tick spacing is chosen from steps x 10^k (smallest that covers the
    data); the first tick is the largest multiple of the spacing <= vmin.
    The axis limits are the first and last tick.

    Args:
        vmin (float): minimum of the data
        vmax (float): maximum of the data
        n (int, optional): number of ticks. Defaults to 4.
        steps (tuple, optional): allowed mantissas of the spacing.

    Returns:
        tuple[list, list]: limits [first, last] and the list of ticks
    """
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmax == vmin:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    raw = (vmax - vmin) / (n - 1)
    exp = np.floor(np.log10(raw))
    for e in (exp, exp + 1):
        for s in steps:
            step = s * 10.0**e
            start = np.floor(vmin / step + 1e-9) * step
            if start + (n - 1) * step >= vmax - 1e-9 * step:
                ticks = start + step * np.arange(n)
                ticks = np.round(ticks, int(max(0, -np.floor(np.log10(step)) + 1)))
                return [float(ticks[0]), float(ticks[-1])], [float(t) for t in ticks]
    raise RuntimeError("no nice step found")



# ---
class data_base_PT_whitney:
    """Class that stores the data of HPLT database, 
    and read the csv file. 
    """
    def __init__(self):
        from pathlib import Path
        path = Path(__file__).resolve().parents[0]
        T, P = np.loadtxt(f"{path}/PT_whitney.csv", delimiter=",", skiprows=1, unpack=True)
        self.T = T
        self.P = P
# ---
def print_save_IQ_information(pdb: pd.DataFrame, pt2save: str, prop: str):
    """Generate basic statistics for a given property and a given database

    Args:
        pdb (pd.DataFrame): Data frame containing the merged database
        pt2save (str): Path to save the csv file
        prop (str): Property
    """


    def compute_stats(a: pd.DataFrame, name: str) -> dict:
        """Enclosure computing the basic statistics

        Args:
            a (pd.DataFrame): The Target database - i.e., the reduced one
            name (str): name of the property

        Returns:
            dict: Dictionary containing all the related data
        """
        
        dict_stat = {}
        quartiles = a.quantile([0.05, 0.25, 0.75, 0.95])
        quartiles = quartiles.to_numpy()
        median = a.median()
        dict_stat["name"] = name
        dict_stat["IQR"] = quartiles[2] - quartiles[1]
        dict_stat["95%"] = quartiles[3]
        dict_stat["5%"] = quartiles[0]
        dict_stat["median"] = median

        return dict_stat

    # ---
    # Check if all the test have been run and create a common dataset
    k3, k5, knl, knlp  = create_common_data_set(pdb=pdb)

    knl_a = knl[prop]
    k3_b = k3[prop]
    k5_c = k5[prop]
    knlp_d = knlp[prop]

    dab = knl_a - k3_b
    dac = knl_a - k5_c
    
    ddb = knlp_d - k3_b
    ddc = knlp_d - k5_c
    
    # ---
    stat_nla = compute_stats(knl_a, "NL")
    stat_k3b = compute_stats(k3_b, "k=3.1")
    stat_k5c = compute_stats(k5_c, "k=5.0")
    stat_knlp = compute_stats(knlp_d, "NLP")

    # ---
    stat_dab = compute_stats(dab, "NL-k3")
    stat_dac = compute_stats(dac, "NL-k5")
    
    stat_ddb = compute_stats(ddb, "NLP-k3")
    stat_ddc = compute_stats(ddc, "NLP-k5")

    for i in stat_nla:
        if i == "name":
            continue
        else:
            print(
                f"   {i}: k3 = {stat_k3b[i]:.2f}, k5 = {stat_k5c[i]:.2f}, nl = {stat_nla[i]:.2f}, nlp = {stat_knlp[i]:.2f}"
            )
            print(f"   {i}: (nl-k3) = {stat_dab[i]:.2f}, (nl-k5) = {stat_dac[i]:.2f}")
            print(f"   {i}: (nlp-k3) = {stat_ddb[i]:.2f}, (nlp-k5) = {stat_ddc[i]:.2f}")

    rows = []
    rows.append(stat_nla)
    rows.append(stat_k3b)
    rows.append(stat_k5c)
    rows.append(stat_knlp)
    rows.append(stat_dab)
    rows.append(stat_dac)
    rows.append(stat_ddb)
    rows.append(stat_ddc)

    pathcsv = f"{pt2save}/table_csv_{prop}.csv"
    # Prepare list of field
    field = [i for i in stat_nla]
    with open(pathcsv, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=field,
        )
        writer.writeheader()
        writer.writerows(rows)


# ---
def create_array(buf: pd.DataFrame) -> np.ndarray:
    """Create array: helper function that creates an array. 
    In the DataFrame, I saved arrays to be forcing the connection
    between input data and 1D arrays representing the state of the 
    surface of the slab. However, it is difficult to plot en-mass 
    these data, thus it is necessary to oderly collect the data
    and transform in an array of (num_test,num_points). 

    Args:
        buf (pd.DataFrame): The reduced database

    Raises:
        ValueError: If this function is used for scalar, it defeats its purpose

    Returns:
        array(np.ndarray): The output array
    """
    if len(buf.shape) > 1:
        raise ValueError("The function accept only 1D dataframe")

    array = np.zeros([buf.size, len(buf.iloc[0])], dtype=np.float32)

    for i in range(buf.size):
        array[i, :] = buf.iloc[i]

    return array


# ---
def sign(num:float)->float|int:
    """Helper function to extract the sign of a given number

    Args:
        num (float | int): a number

    Returns:
        int: -1 if num is negative, 1 otherwise
    """
    return -1 if num < 0 else 1


# ---
def ranges(x, y, z,w=np.nan)->list:
    """Helper function that finds the global maximum 
    and global minimum of three arrays. This function
    helps to standardise the limit 
    --- To be removed, as I find a better approach.
    Args:
        x (np.ndarray | float): array 1
        y (np.ndarray | float): array 2
        z (np.ndarray | float): array 3
        w (np.ndarray | float, optional): array 4. Defaults to np.nan.
    Returns:
        list: [global_min, global_max]
    """
    lim = [
        np.floor(np.nanmin([np.nanmin(x), np.nanmin(y), np.nanmin(z), np.nanmin(w)])),
        np.ceil(np.nanmax([np.nanmax(x), np.nanmax(y), np.nanmax(z), np.nanmax(w)])),
    ]
    return lim


# ---
def create_list_tick(lim: list) -> list:
    """Generate a standardized set of ticks for the plot.

    Args:
        lim (list): Axis limits.

    Returns:
        list: ticks list
    """
    d = np.abs(lim[1] - lim[0])

    exp = np.floor(np.log10(d)) - 1

    stp = 2.5 * 10**exp

    ticks_buf = np.arange(lim[0], lim[1] + stp, stp)

    if len(ticks_buf) > 4:
        ticks = []
        for i, v in enumerate(ticks_buf):
            if i == 0:
                ticks.append(v)
            else:
                if (v - ticks_buf[0]) % (stp * 3) == 0:
                    ticks.append(v)

        if ticks[-1] != ticks_buf[-1]:
            if ticks_buf[-1] - ticks[-1] < stp * 1.5:
                ticks[-1] = ticks_buf[-1]
            else:
                ticks.append(ticks_buf[-1])

    else:
        ticks = ticks_buf

    return ticks


# ---
def check_precision(value: float, m: bool) -> float:
    """Function to check the log10 of a given value. 
    This allow to round up the value to a definite number
    such that the ticks are created automatically

    Args:
        value (float): the target value
        m (True): is the minimum value or the maximum value

    Returns:
        float: corrected value
    """
    if value == 0.0:
        return 0.0

    value_u = np.abs(value)
    direction = sign(value)
    exponent = np.floor(np.log10(value_u))
    scale = 10 ** (exponent)
    value_u = value_u / scale
    buf = value_u - np.floor(value_u)

    precision = np.arange(0, 1.0 + 0.005, 0.005)

    for i in range(len(precision)):
        if i == 0:
            interval = buf < precision[i]
        else:
            interval = buf >= precision[i - 1] and buf < precision[i]
        if interval:
            if m:
                corr = precision[i - 1]
            else:
                corr = precision[i]

    value_u = (np.floor(value_u) + corr) * scale * direction

    return value_u
# ---
def create_common_data_set(
    pdb: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function that creates common datasets:
    -> Time to time simulations fails,so it is necessary to check how many experiments fails,
    select the common successfull data

    Args:
        pdb (pd.DataFrame): main database

    WARNING: this function has hard coded value related to the main manuscript!!!!

    Returns:
        tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]: filtered database
    """
    keys = ["Tp", "age", "vc"]

    a = pdb[(pdb["k"] == 3.1)]
    b = pdb[(pdb["k"] == 5.0)]
    c = pdb[(pdb["k"] == 0.0)]
    d = pdb[(pdb["k"] == -1.0)]
    

    common_keys = (
        a[keys]
        .merge(b[keys], on=keys, how="inner")
        .merge(c[keys], on=keys, how="inner")
        .drop_duplicates()
        .sort_values(keys)
        .reset_index(drop=True)
    )

    a_common = (
        (a.merge(common_keys, on=keys, how="inner"))
        .sort_values(keys)
        .reset_index(drop=True)
    )
    b_common = (
        (b.merge(common_keys, on=keys, how="inner"))
        .sort_values(keys)
        .reset_index(drop=True)
    )
    c_common = (
        (c.merge(common_keys, on=keys, how="inner"))
        .sort_values(keys)
        .reset_index(drop=True)
    )

    d_common = (
        (d.merge(common_keys, on=keys, how="inner"))
        .sort_values(keys)
        .reset_index(drop=True)
    )

    return a_common, b_common, c_common, d_common
# ---
def extract_limit_array(ar: np.ndarray) -> tuple[list,list]:
    """Function that extract the limit of an array 
    -> This array is the concatenated arrays that have been flatten into a 0D array

    Args:
        ar (np.ndarray): target array

    Returns:
        tuple of lim and ticks list
    """


    # Find the min and max
    lim = [np.min(ar.flatten()), np.max(ar.flatten())]
    # Check the precision and round up the limits to the 0.05 highest multiply or the lowest 0.05 multiply
    lim = [check_precision(lim[0], True), check_precision(lim[1], False)]
    # Create the ticks 
    ticks = create_list_tick(lim)
    # Enlarge the limits of 0.1 just for being sure to not cut the plot using the mean
    mean_lim = np.abs(np.mean(lim))
    exp = np.floor(np.log10(mean_lim))
    mean_lim = (mean_lim * 10 ** (-exp)) * 0.5
    # Correct additionally the limits
    lim = [lim[0] - mean_lim, lim[1] + mean_lim]

    return lim, ticks
# ---
def modify_axes(
    ax: mpl_axes = None,
    label_x: list = ["n", "n"],
    lim_x: list|None = None,
    lim_y: list|None = None,
    label_y: list|None = ["n", "n"],
    ticks_x: list|None = None,
    ticks_y: list|None = None,
    label_x_c: str = "Numeric",
    label_y_c: str = "Numeric",
    ticks_label_x: list|None= None,
    ticks_label_y: list|None = None,
    letter: str|None = "[a]",
    precision: str = "%.1f",
) -> mpl_axes:
    """Function that modifies the axes and standardisze the axes
    Args:
        ax (mpl_axes, optional): current axis. Defaults to None.
        label_x (list, optional): [label, show_ticklabels ('y'/'n')].
                Defaults to ['n','n']. -> in case of the label name = string to use as label
        lim_x (list, optional): limits of the x-axis. Defaults to None.
        lim_y (list, optional): limits of the y-axis. Defaults to None.
        label_y (list, optional): 
                    [label, show_ticklabels ('y'/'n')]. Defaults to ['n','n']
                    . -> in case of the label name = string to use as label
        ticks_x (list, optional): x tick positions. Defaults to None.
        ticks_y (list, optional): y tick positions. Defaults to None.
        label_x_c (str, optional): x tick-label format; "categorical" uses
                ticks_label_x verbatim, anything else formats numerically
                with `precision`. Defaults to "Numeric".
        label_y_c (str, optional): same as label_x_c, for the y-axis.
                Defaults to "Numeric".
        ticks_label_x (list, optional): explicit x tick labels, used when
                label_x_c == "categorical". Defaults to None.
        ticks_label_y (list, optional): explicit y tick labels. Defaults to None.
        letter (str, optional): panel letter/label drawn in the top-left
                corner of the axis. Defaults to "[a]".
        precision (str, optional): printf-style format string for numeric
                tick labels. Defaults to "%.1f".
    Returns:
        mpl_axes: modified axis
    """
    # --- spine and grid styling ---

    ax.tick_params(axis="both", which="major", pad=9)
    ax.tick_params(which="minor", length=4, color="k")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", colors="black", direction="in", length=5, width=2.0)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.grid(
        visible=True,
        which="both",
        axis="both",
        color="gray",
        linestyle="--",
        linewidth=0.3,
        alpha=0.7,
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=_TICKSFONTSIZE_)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=_TICKSFONTSIZE_)

    # --- x-axis ---
    if label_x[0] != "n":
        ax.set_xlabel(label_x[0], fontsize=_LABELFONTSIZE_, labelpad=10)
        ax.xaxis.set_label_position("bottom")

    if lim_x is not None:
        ax.set_xlim(lim_x[0], lim_x[1])

    if ticks_x is not None:
        ax.set_xticks(ticks_x)

    if label_x[1] == "y":
        if label_x_c != "categorical":
            ax.xaxis.set_major_formatter(FormatStrFormatter(precision))
        else:
            ax.set_xticklabels(ticks_label_x)
    else:
        ax.set_xticklabels([])

    # --- y-axis ---
    if label_y[0] != "n":
        ax.set_ylabel(
            label_y[0], fontsize=_LABELFONTSIZE_, labelpad=10
        )  # fixed: was label_x[0]
        ax.yaxis.set_label_position("left")  # fixed: was 'top'

    if lim_y is not None:  # fixed: was lim_x
        ax.set_ylim(lim_y[0], lim_y[1])

    if ticks_y is not None:
        ax.set_yticks(ticks_y)

    if label_y[1] == "y":
        ax.yaxis.set_major_formatter(FormatStrFormatter(precision))
    else:
        ax.set_yticklabels([])

    ax.text(
        0.02,
        0.97,
        letter,
        transform=ax.transAxes,
        c="w",
        fontweight="demibold",
        fontsize=10,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "k", "alpha": 0.7},
    )

    return ax


def plot_slab_variances(A, path2save: str, number_figure: str,
                        pressure_flag: bool = True, alpha: float = 1.0,
                        cmap: str = "cmc.vik", vlim: float | None = None,
                        levels_k3=(-100, -50, -25, 25, 50, 100),
                        levels_k5=(-50, -25, -10, 10, 25, 50),
                        separate_colorbars: bool = False,
                        vlim_rows: tuple | None = None) -> None:
    """Plot slab temperature differences between thermal models.

    Panels: (a) NL - k3, (b) NLP - k3, (c) NL - k5, (d) NLP - k5.
    Labelled dT contours on every panel; the k5 row uses finer levels because
    its differences are smaller (NL-k5 ~25 degC, NLP-k5 ~77 degC).

    Args:
        A (tuple): (k3, k5, NL, NLP) objects, each exposing `.xi`, `.yi` and
            `.temp_mean_2d` (mean temperature on that grid, Welford algorithm)
        path2save (str): directory where the figure is saved
        number_figure (str): figure number/letter
        pressure_flag (bool): currently unused
        alpha (float): transparency of the pcolormesh
        cmap (str): diverging colormap (default cmc.vik: blue < 0 < red)
        vlim (float | None): max |dT| of the shared colorbar; if None, from the data
        levels_k3, levels_k5: contour levels [degC] for the top (k3) and bottom (k5) row
        separate_colorbars (bool): if True, each row gets its own colour scale
            and a vertical colorbar on the right; otherwise one shared colorbar
            at the bottom (as before)
        vlim_rows (tuple | None): (vlim_k3, vlim_k5) when separate_colorbars=True;
            if None, taken from the data of each row

    Returns:
        None
    """
    name_plot = f"Figure_{number_figure}.png"
    k3, k5, NL, NLP = A
    Xi, Yi = k3.xi, k3.yi

    # --- differences ------------------------------------------------------
    diffs = [
        NL.temp_mean_2d - k3.temp_mean_2d,   # [a]
        NLP.temp_mean_2d - k3.temp_mean_2d,  # [b]
        NL.temp_mean_2d - k5.temp_mean_2d,   # [c]
        NLP.temp_mean_2d - k5.temp_mean_2d,  # [d]
    ]
    with open(f"{path2save}/max_min_dT.txt", "w") as f:
        for i, d in enumerate(diffs):
            f.write(f"Panel {i}: min={np.nanmin(d):.2f}, max={np.nanmax(d):.2f}\n")

    # --- normalisation: shared, or one per row ------------------------------
    if separate_colorbars:
        if vlim_rows is None:
            vlim_rows = (max(np.nanmax(np.abs(d)) for d in diffs[:2]),
                         max(np.nanmax(np.abs(d)) for d in diffs[2:]))
        norms = [TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
                 for v in (vlim_rows[0], vlim_rows[0], vlim_rows[1], vlim_rows[1])]
    else:
        if vlim is None:
            vlim = max(np.nanmax(np.abs(d)) for d in diffs)
        norms = [TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)] * 4
    levels = [levels_k3, levels_k3, levels_k5, levels_k5]

    fig = plt.figure(figsize=(16, 18))

    bx, by = 0.10, 0.17      # lower-left corner of the panel block
    sx, sy = 0.38, 0.38      # panel size
    dx, dy = 0.04, 0.04      # gaps between panels

    ax0 = fig.add_axes([bx,           by + dy + sy, sx, sy])  # NL  - k3
    ax1 = fig.add_axes([bx + sx + dx, by + dy + sy, sx, sy])  # NLP - k3
    ax2 = fig.add_axes([bx,           by,           sx, sy])  # NL  - k5
    ax3 = fig.add_axes([bx + sx + dx, by,           sx, sy])  # NLP - k5
    axes = [ax0, ax1, ax2, ax3]

    ims = []
    for ax, d, nrm, lev in zip(axes, diffs, norms, levels):
        im = ax.pcolormesh(Xi, Yi, d, cmap=cmap, norm=nrm, alpha=alpha, shading="auto")
        ims.append(im)
        # labelled contours, only the levels that exist in this panel
        dmin, dmax = np.nanmin(d), np.nanmax(d)
        lev_ok = [l for l in lev if dmin < l < dmax]
        if lev_ok:
            cs = ax.contour(Xi, Yi, d, levels=lev_ok, colors="k", linewidths=0.9)
            ax.clabel(cs, fmt="%+d", fontsize=_TICKSFONTSIZE_ * 0.7, inline=True, inline_spacing=3)

    lim_x = [0.0, 800.0]
    lim_y = [-600, 0.0]
    ticks_x = [0, 400, 800]

    ax0 = modify_axes(ax0, label_x=["n", "n"], label_y=["z, [km]", "y"],
                      lim_y=lim_y, lim_x=lim_x, letter="[a]", ticks_x=ticks_x)
    ax1 = modify_axes(ax1, label_x=["n", "n"], label_y=["n", "n"],
                      lim_y=lim_y, lim_x=lim_x, letter="[b]", ticks_x=ticks_x)
    ax2 = modify_axes(ax2, label_x=["x, [km]", "y"], label_y=["z, [km]", "y"],
                      lim_y=lim_y, lim_x=lim_x, letter="[c]", ticks_x=ticks_x)
    ax3 = modify_axes(ax3, label_x=["x, [km]", "y"], label_y=["n", "n"],
                      lim_y=lim_y, lim_x=lim_x, letter="[d]", ticks_x=ticks_x)

    # --- colorbar(s) ---------------------------------------------------------
    label_cb = r"$\Delta \bar{T}$ [$^\circ$C]"
    if separate_colorbars:
        cb_w, cb_gap = 0.015, 0.015
        for row, (im, y0) in enumerate(((ims[1], by + dy + sy), (ims[3], by))):
            cax = fig.add_axes([bx + 2 * sx + dx + cb_gap, y0 + 0.1 * sy, cb_w, 0.8 * sy])
            cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both")
            cb.set_label(label_cb, fontsize=_LABELFONTSIZE_, labelpad=10)
            cb.ax.tick_params(labelsize=_TICKSFONTSIZE_, length=6, width=1.2)
    else:
        cb_h, cb_gap = 0.015, 0.07
        cax = fig.add_axes([bx, by - cb_gap - cb_h, 2 * sx + dx, cb_h])
        cb = fig.colorbar(ims[-1], cax=cax, orientation="horizontal", extend="both")
        cb.set_label(label_cb, fontsize=_LABELFONTSIZE_, labelpad=10)
        cb.ax.tick_params(labelsize=_TICKSFONTSIZE_, length=6, width=1.2)

    # --- panel tags ----------------------------------------------------------
    for ax, tag in zip(axes, ("NL-k3", "NLP-k3", "NL-k5", "NLP-k5")):
        ax.text(0.05, 0.1, tag, transform=ax.transAxes, c="w", fontweight="demibold",
                fontsize=20, va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "k", "alpha": 0.7})

    fig.savefig(f"{path2save}/{name_plot}", dpi=600, transparent=TRANSPARENT)
    plt.close(fig)
# ---
# ---
def nice_ticks(vmin: float, vmax: float, n: int = 4,
               steps: tuple = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)) -> tuple[list, list]:
    """Return exactly n round ticks covering [vmin, vmax].

    The tick spacing is chosen from steps x 10^k (smallest that covers the
    data); the first tick is the largest multiple of the spacing <= vmin.
    The axis limits are the first and last tick.

    Args:
        vmin (float): minimum of the data
        vmax (float): maximum of the data
        n (int, optional): number of ticks. Defaults to 4.
        steps (tuple, optional): allowed mantissas of the spacing.

    Returns:
        tuple[list, list]: limits [first, last] and the list of ticks
    """
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmax == vmin:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    raw = (vmax - vmin) / (n - 1)
    exp = np.floor(np.log10(raw))
    for e in (exp, exp + 1):
        for s in steps:
            step = s * 10.0**e
            start = np.floor(vmin / step + 1e-9) * step
            if start + (n - 1) * step >= vmax - 1e-9 * step:
                ticks = start + step * np.arange(n)
                ticks = np.round(ticks, int(max(0, -np.floor(np.log10(step)) + 1)))
                return [float(ticks[0]), float(ticks[-1])], [float(t) for t in ticks]
    raise RuntimeError("no nice step found")


# ---
def tick_format(*tick_lists: list) -> str:
    """'%.0f' if all ticks are integers, '%.1f' otherwise."""
    all_ticks = np.concatenate([np.asarray(t, dtype=float) for t in tick_lists])
    return "%.0f" if np.allclose(all_ticks, np.round(all_ticks)) else "%.1f"


# ---
def create_bin(df: pd.DataFrame, x: str, y: str, nbins: int = 20) -> pd.DataFrame:
    """Bin df[x] into nbins equal-width bins and compute the statistics of
    df[y] in each bin.

    Args:
        df (pd.DataFrame): data
        x (str): column used for the bins
        y (str): column whose statistics are computed
        nbins (int, optional): number of bins. Defaults to 20.

    Returns:
        pd.DataFrame: one row per bin with x_mid, y_med, y_mean, y_std, y_q25, y_q75
    """
    x_bins = pd.cut(df[x], bins=nbins)
    return df.groupby(x_bins, observed=True).agg(
        x_mid=(x, "median"),
        y_med=(y, "median"),
        y_mean=(y, "mean"),
        y_std=(y, "std"),
        y_q25=(y, lambda s: s.quantile(0.25)),
        y_q75=(y, lambda s: s.quantile(0.75)),
    )


# ---
def figure_slab_surface(
    path2save: str,
    pdb: pd.DataFrame,
    crd: CoordinateSurface,
    petro_db: data_base_PT_whitney,
    figure_number: str,
) -> None:
    """Top row: P-T conditions along the slab surface (NL/NLP) with the
    petrological database, and the differences between the non-linear and the
    linear thermal models.
    Middle row: total seismogenic length (150-450 C) against plate age, v_c,
    T_p and the thermal parameter (binned median +- 1 std, one per ensemble).
    Bottom row: same for the locked seismogenic length (150-350 C).

    Args:
        path2save (str): path to save the figure
        pdb (pd.DataFrame): The dataframe containing all the simulation data
        crd (CoordinateSurface): coordinate data structure
        petro_db (data_base_PT_whitney): Petrological database
        figure_number (str): the number of the figure
    """
    name_plot = f"Figure_{figure_number}.png"

    fig = plt.figure(figsize=(16, 14))

    # --- layout (three rows)
    bx, by = 0.1, 0.07
    sy, dy = 0.23, 0.07
    sx_top, dx_top = 0.25, 0.07
    row_bot = by                      # locked length
    row_mid = by + sy + dy            # total length
    row_top = by + 2 * (sy + dy)      # slab surface

    width_total = 3 * sx_top + 2 * dx_top  # length rows span the same width as the top row
    dx_bot = 0.03
    sx_bot = (width_total - 3 * dx_bot) / 4

    # Top row: ax0 -> NL/NLP slab surface + petrological DB
    #          ax1 -> dT NL - k3.1 / k5
    #          ax2 -> dT NLP - k3.1 / k5
    ax0 = fig.add_axes([bx, row_top, sx_top, sy])
    ax1 = fig.add_axes([bx + sx_top + dx_top, row_top, sx_top, sy])
    ax2 = fig.add_axes([bx + 2 * (sx_top + dx_top), row_top, sx_top, sy])

    # Length rows: ell vs age, vc, Tp, gamma
    # (no sharey: modify_axes blanks tick labels, which would propagate to shared axes)
    mid = [fig.add_axes([bx + j * (sx_bot + dx_bot), row_mid, sx_bot, sy]) for j in range(4)]
    bot = [fig.add_axes([bx + j * (sx_bot + dx_bot), row_bot, sx_bot, sy]) for j in range(4)]

    k3, k5, knl, knl_p = create_common_data_set(pdb=pdb)

    # all ensembles are sorted on the same keys -> share the thermal parameter
    for df in (k5, knl, knl_p):
        if "thermal_parameter" not in df.columns:
            df["thermal_parameter"] = k3["thermal_parameter"].to_numpy()

    # --- top row
    ax0 = plot_slab_slab_real(ax=ax0, pdb=knl, c="firebrick")
    ax0 = plot_slab_slab_real(ax=ax0, pdb=knl_p, c="darkgreen")

    ax1 = plot_slab_slab_real(ax=ax1, pdb=k3, c="cadetblue", non_linear_db=knl)
    ax1 = plot_slab_slab_real(ax=ax1, pdb=k5, c="olive", non_linear_db=knl)

    ax2 = plot_slab_slab_real(ax=ax2, pdb=k3, c="cadetblue", non_linear_db=knl_p)
    ax2 = plot_slab_slab_real(ax=ax2, pdb=k5, c="olive", non_linear_db=knl_p)

    ax0.axvline(150, linewidth=0.8, c="k", linestyle=":")
    ax0.axvline(350, linewidth=0.8, c="k", linestyle=":")
    ax0.axvline(450, linewidth=0.8, c="k", linestyle=":")
    ax0.scatter(petro_db.T, petro_db.P, c="b", s=5.0, marker="x")

    lim_T, ticks_T = nice_ticks(0.0, 900.0)        # 0, 300, 600, 900
    lim_P, ticks_P = nice_ticks(0.0, 4.0)          # 0, 1.5, 3, 4.5
    lim_dT, ticks_dT = [-20.0, 100.0], [-20.0, 20.0, 60.0, 100.0]

    ax0 = modify_axes(
        ax0,
        label_x=[r"T [$^{\circ}$C]", "y"],
        label_y=[r"P [GPa]", "y"],
        lim_x=lim_T,
        lim_y=lim_P,
        ticks_x=ticks_T,
        ticks_y=ticks_P,
        letter="[a]",
        precision=tick_format(ticks_T, ticks_P),
    )
    ax1 = modify_axes(
        ax1,
        label_x=[r"$\Delta$ T [$^{\circ}$C]", "y"],
        lim_x=lim_dT,
        lim_y=lim_P,
        ticks_x=ticks_dT,
        ticks_y=ticks_P,
        letter="[b]",
        precision=tick_format(ticks_dT, ticks_P),
    )
    ax2 = modify_axes(
        ax2,
        label_x=[r"$\Delta$ T [$^{\circ}$C]", "y"],
        lim_x=lim_dT,
        lim_y=lim_P,
        ticks_x=ticks_dT,
        ticks_y=ticks_P,
        letter="[c]",
        precision=tick_format(ticks_dT, ticks_P),
    )

    # --- length rows: binned median +- 1 std
    ensembles = (
        (k3, "cadetblue"),
        (k5, "olive"),
        (knl, "firebrick"),
        (knl_p, "darkgreen"),
    )

    # (column, x label, scale)
    columns = (
        ("age", r"age [Myr]", 1.0),
        ("vc", r"$v_c$ [cm/yr]", 1.0),
        ("Tp", r"$T_p$ [$^{\circ}$C]", 1.0),
        ("thermal_parameter", r"$\gamma$ [1000 km]", 1000.0),
    )

    def plot_length_row(axes: list, y: str, ylabel: str, letters: tuple, show_x: bool) -> None:
        """One row of panels: y against each column, one band per ensemble."""
        binned = {
            (j, k): create_bin(df, x=col[0], y=y)
            for j, col in enumerate(columns)
            for k, (df, _) in enumerate(ensembles)
        }
        lo = np.concatenate([(s["y_med"] - s["y_std"]).to_numpy() for s in binned.values()])
        hi = np.concatenate([(s["y_med"] + s["y_std"]).to_numpy() for s in binned.values()])
        lim_y, ticks_y = nice_ticks(np.nanmin(lo), np.nanmax(hi))

        for j, (x, xlabel, scale) in enumerate(columns):
            xs = k3[x] / scale
            lim_xb, ticks_xb = nice_ticks(np.nanmin(xs), np.nanmax(xs))

            for k, (_, colour) in enumerate(ensembles):
                s = binned[(j, k)]
                xm = s["x_mid"] / scale
                axes[j].fill_between(
                    xm,
                    s["y_med"] - s["y_std"],
                    s["y_med"] + s["y_std"],
                    color=colour,
                    alpha=0.15,
                    linewidth=0,
                )
                axes[j].plot(xm, s["y_med"], c=colour, linewidth=1.5)

            shown = ([ticks_xb] if show_x else []) + ([ticks_y] if j == 0 else [])
            axes[j] = modify_axes(
                axes[j],
                label_x=[xlabel, "y"] if show_x else ["n", "n"],
                label_y=[ylabel, "y"] if j == 0 else ["n", "n"],
                lim_x=lim_xb,
                lim_y=lim_y,
                ticks_x=ticks_xb,
                ticks_y=ticks_y,
                letter=letters[j],
                precision=tick_format(*shown) if shown else "%.0f",
            )

    plot_length_row(
        mid,
        y="d_seismic_ell_total",
        ylabel=r"$\ell^{\mathrm{seismic}}_{\mathrm{total}}$ [km]",
        letters=("[d]", "[e]", "[f]", "[g]"),
        show_x=False,
    )
    plot_length_row(
        bot,
        y="d_seismic_ell_locked",
        ylabel=r"$\ell^{\mathrm{seismic}}_{\mathrm{locked}}$ [km]",
        letters=("[h]", "[i]", "[j]", "[k]"),
        show_x=True,
    )

    # --- legend
    legend_elements = [
        Line2D([0], [0], color="firebrick", linewidth=2, label="NL"),
        Line2D([0], [0], color="cadetblue", linewidth=2, label="k = 3.1 [W/m/K]"),
        Line2D([0], [0], color="olive", linewidth=2, label="k = 5 [W/m/K]"),
        Line2D([0], [0], color="darkgreen", linewidth=2, label="NLP"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=True,
        prop={"size": _TEXTFONTSIZE_, "weight": "demibold"},
    )

    fig.savefig(f"{path2save}/{name_plot}", dpi=600, transparent=TRANSPARENT)
    plt.close(fig)
    # --- 
def figure_slab_surface2(
    path2save: str,
    pdb: pd.DataFrame,
    crd: CoordinateSurface,
    petro_db: data_base_PT_whitney,
    figure_number: str,
) -> None:
    """Plot the P-T conditions along the slab surface (NL vs k3.1/k5.0
    thermal-model differences) together with the petrological database and
    the seismogenic-length statistics.

    Args:
        path2save (str): path to save the figure
        pdb (pd.DataFrame): The dataframe containing all the simulation data
        crd (CoordinateSurface): coordinate data structure
        petro_db (data_base_PT_whitney): Petrological database
        figure_number (str): the number of the figure
    """


    def difference_surface(pdb_a: pd.DataFrame, pdb_b: pd.DataFrame) -> np.ndarray:
        """Enclosure to find the dT array
        -> check if the same row indicate the same [Tp,vc,age]-> if so, perform
        the difference-> update array.

        Args:
            pdb_a (pd.DataFrame): Non linear database
            pdb_b (pd.DataFrame): Linear database

        Returns:
            final array with all the experiment
        """
        count = len(pdb_a["case_id"])
        keys = ["Tp", "vc", "age"]
        array = np.zeros([count, len(pdb_a.loc[0, "T_S"])], dtype=np.float32)
        n_exp = 0

        for i in range(count):
            same = False

            for k in keys:
                if pdb_a.loc[i, k] != pdb_b.loc[i, k]:
                    same = False
                    break
                else:
                    same = True
            if same:
                array[i, :] = pdb_a.loc[i, "T_S"] - pdb_b.loc[i, "T_S"]
                n_exp = n_exp + 1

        print(f"Number of experiments {n_exp}, {n_exp / count:.1f}")

        return array

    name_plot = f"Figure_{figure_number}.png"

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
    top_row = by + sy + dy
    # ---
    # Top row: figure ax0 -> NL slab surface // black line + scatter DB + band 80 km
    #        : figure ax1 -> dT LN_k=3.1 - NL // black line + band 80 km
    #        : figure ax2 -> dT LN_k=5 - NL // blacl line   + band 80 km
    # Bot row: figure ax3 -> scatter plot dT vs gamma (thermal parameter) : cadet_blue => k=3.1 olive_green = k=5.0
    #        : figure ax4 -> scatter plot dT vs Tp (potential temperature):
    # ---
    # ax5 => empty axes where to put the legend.
    ax0 = fig.add_axes([bx, top_row, sx_top, sy])
    ax1 = fig.add_axes([bx + sx_top + dx_top_a, top_row, sx_top, sy])
    ax2 = fig.add_axes([bx + 2 * sx_top + dx_top_a + dx_top_b, top_row, sx_top, sy])

    ax3 = fig.add_axes([bx, by, sx_bot, sy])
    ax4 = fig.add_axes([bx + sx_bot + dx_bota, by, sx_bot, sy])
    ax5 = fig.add_axes([bx + 2 * sx_bot + dx_bota + dx_botb, by, sx_bot, sy])

    k3, k5, knl, knl_p = create_common_data_set(pdb=pdb)


    # Extract the temperature of the non linear experiments
    k_nl_slab_temp = create_array(knl.loc[(knl["k"] == 0), "T_S"])
    k_nl_slab_pres = create_array(knl.loc[(knl["k"] == 0), "P_S"])

    # Extract the difference between the groups
    ax0 = plot_slab_slab_real(ax=ax0,pdb=knl,c='firebrick')
    ax0 = plot_slab_slab_real(ax=ax0,pdb=knl_p,c='darkgreen')
    
    ax1 = plot_slab_slab_real(ax=ax1, pdb=k3, c="cadetblue",non_linear_db=knl)
    ax1 = plot_slab_slab_real(ax=ax1, pdb=k5, c="olive",non_linear_db=knl)

    ax2 = plot_slab_slab_real(ax=ax2, pdb=k3, c="cadetblue",non_linear_db=knl_p)
    ax2 = plot_slab_slab_real(ax=ax2, pdb=k5, c="olive",non_linear_db=knl_p)



    ax0.axvline(150, linewidth=0.8, c="k", linestyle=":")
    ax0.axvline(450, linewidth=0.8, c="k", linestyle=":")

    ax0.scatter(petro_db.T, petro_db.P, c="b", s=5.0, marker="x")
    ax3.scatter(
        k3["thermal_parameter"] / 1000, knl["d_seismic_ell_total"], color="firebrick", s=0.5
    )
    ax3.scatter(
        k3["thermal_parameter"] / 1000, knl_p["d_seismic_ell_total"], color="darkgreen", s=0.5
    )
    ax3.scatter(
        k3["thermal_parameter"] / 1000, k3["d_seismic_ell_total"], color="cadetblue", s=0.5
    )
    ax3.scatter(
        k3["thermal_parameter"] / 1000, k5["d_seismic_ell_total"], color="olive", s=0.5
    )

    err_k3 = (
        ((knl["d_seismic_ell_total"] - k3["d_seismic_ell_total"]))
    )
    err_k5 = (
        ((knl["d_seismic_ell_total"] - k5["d_seismic_ell_total"]))
    )
    
    err_k3_p = (
        ((knl_p["d_seismic_ell_total"] - k3["d_seismic_ell_total"]))
    )
    err_k5_p = (
        ((knl_p["d_seismic_ell_total"] - k5["d_seismic_ell_total"]))
    )

    ax4.scatter(k5["thermal_parameter"] / 1000, err_k5, color="olive", s=0.5)
    ax4.scatter(k3["thermal_parameter"] / 1000, err_k3, color="cadetblue", s=0.5)


    ax5.scatter(k5["thermal_parameter"] / 1000, err_k5_p, color="olive", s=0.5)
    ax5.scatter(k3["thermal_parameter"] / 1000, err_k3_p, color="cadetblue", s=0.5)


    # Homogeneise the layout of the figure.

    ax0 = modify_axes(
        ax0,
        label_x=[r"T [$^{\circ}$C]", "y"],
        lim_x=[0, 1000],
        lim_y=[0, 4],
        label_y=[r"P [GPa]", "y"],
        ticks_x=[0.0, 250, 500, 750, 1000],
        ticks_y=[0, 1, 2, 3, 4],
        letter="[a]",
    )

    lim_x = ranges(-10, 100, np.nan)
    lim_x = [check_precision(lim_x[0], True), check_precision(lim_x[1], False)]
    ticks_x = create_list_tick(lim_x)

    ax1 = modify_axes(
        ax1,
        label_x=[r"$\Delta$ T [$^{\circ}$C]", "y"],
        lim_y=[0, 4],
        lim_x=lim_x,
        ticks_x=ticks_x,
        ticks_y=[0, 1, 2, 3, 4],
        letter="[b]",
    )
    ax2 = modify_axes(
        ax2,
        label_x=[r"$\Delta$ T [$^{\circ}$C]", "y"],
        lim_y=[0, 4],
        lim_x=lim_x,
        ticks_x=ticks_x,
        ticks_y=[0, 1, 2, 3, 4],
        letter="[c]",
    )

    lim_x = ranges(
        knl["thermal_parameter"] / 1000,
        k3["thermal_parameter"] / 1000,
        k5["thermal_parameter"] / 1000,
    )
    lim_x = [check_precision(lim_x[0], True), check_precision(lim_x[1], False)]
    ticks_x = [0,2,4,6,8]

    lim_y = ranges(knl["d_seismic_ell_total"], k3["d_seismic_ell_total"], k5["d_seismic_ell_total"])
    lim_y = [50.0, check_precision(lim_y[1], False)]
    ticks_y = create_list_tick(lim_y)

    ax3 = modify_axes(
        ax3,
        label_x=[r"$\gamma$ [1000 km]", "y"],
        label_y=[r"$\ell^{\mathrm{seismic}}$ [km]", "y"],
        lim_y=lim_y,
        lim_x=lim_x,
        ticks_x=ticks_x,
        ticks_y=ticks_y,
        letter="[d]",
    )

    lim_y = ranges(err_k3, err_k5, np.nan)
    lim_y = [check_precision(lim_y[0], True), check_precision(lim_y[1], False)]
    ticks_y = create_list_tick(lim_y)

    ax4 = modify_axes(
        ax4,
        label_x=[r"$\gamma$ [1000 km]", "y"],
        label_y=[r"$\Delta\ell$ [km]", "y"],
        lim_y=lim_y,
        lim_x=lim_x,
        ticks_x=ticks_x,
        ticks_y=ticks_y,
        letter="[e]",
    )
    ax5 = modify_axes(
        ax5,
        label_x=[r"$\gamma$ [1000 km]", "y"],
        label_y=['n', "y"],
        lim_y=lim_y,
        lim_x=lim_x,
        ticks_x=ticks_x,
        ticks_y=ticks_y,
        letter="[f]",
    )
    # Custom legend entries
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="firebrick",
            linestyle="None",
            markersize=8,
            label="NL",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="cadetblue",
            linestyle="None",
            markersize=8,
            label="k = 3.1 [W/m/K]",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="olive",
            linestyle="None",
            markersize=8,
            label="k = 5 [W/m/K]",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="darkgreen",
            linestyle="None",
            markersize=8,
            label="NLP",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=True,
        prop={"size": _TEXTFONTSIZE_, "weight": "demibold"},
    )

    fig.savefig(f"{path2save}/{name_plot}", dpi=600, transparent=TRANSPARENT)
# ---
def create_data_set_thermal_picture(
    a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, d: pd.DataFrame,
    isotherms: tuple = (350, 450, 600), nbins: int = 20,
) -> dict:
    """Binned statistics (median, std, 25th and 75th percentiles) of the maximum
    isotherm depth against the thermal parameter, for each thermal model.

    Returns:
        dict: {'isotherm_T': {'k3': (n,4), 'k5': (n,4), 'NL': (n,4), 'NLP': (n,4),
                              'thermal_parameter': (n,)}}
              columns: median, std, p25, p75
    """
    gamma = a["thermal_parameter"].to_numpy()
    edges = np.linspace(np.nanmin(gamma), np.nanmax(gamma), nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    def binned_stats(db: pd.DataFrame, col: str) -> np.ndarray:
        g = db["thermal_parameter"].to_numpy()
        v = db[col].to_numpy(dtype=float)
        out = np.full((nbins, 4), np.nan)
        for i in range(nbins):
            m = (g >= edges[i]) & ((g < edges[i + 1]) if i < nbins - 1 else (g <= edges[i + 1]))
            if m.sum() > 0:
                out[i] = [np.nanmedian(v[m]), np.nanstd(v[m]),
                          np.nanpercentile(v[m], 25), np.nanpercentile(v[m], 75)]
        return out

    main_dictionary = {}
    for t in isotherms:
        col = f"max_dpth{t}"
        main_dictionary[f"isotherm_{t}"] = {
            "k3": binned_stats(a, col),
            "k5": binned_stats(b, col),
            "NL": binned_stats(c, col),
            "NLP": binned_stats(d, col),
            "thermal_parameter": centres,
        }
    return main_dictionary


# ---
def plot_slab_versus_thermalpar(
    path2save: str, pdb: pd.DataFrame, figure_number: str = 0
) -> None:
    """Maximum depth of the 350, 450 and 600 C isotherms within the slab against
    the thermal parameter (top row: binned median and 25-75 % range per thermal
    model; bottom row: difference of the binned medians, non-linear minus linear).

    Args:
        path2save (str): path to save
        pdb (pd.DataFrame): DataFrame containing all the simulation data
        figure_number (str, optional): figure number
    """
    name_plot = f"Figure_{figure_number}.png"
    isotherms = (350, 450, 600)

    k3, k5, knl, knl_p = create_common_data_set(pdb=pdb)
    for df in (k5, knl, knl_p):
        if "thermal_parameter" not in df.columns:
            df["thermal_parameter"] = k3["thermal_parameter"].to_numpy()

    main_data = create_data_set_thermal_picture(a=k3, b=k5, c=knl, d=knl_p, isotherms=isotherms)

    fig = plt.figure(figsize=(16, 12))
    bx, by = 0.1, 0.1
    sx, sy = 0.25, 0.36
    dx, dy = 0.07, 0.03
    top = [fig.add_axes([bx + j * (sx + dx), by + sy + dy, sx, sy]) for j in range(3)]
    bot = [fig.add_axes([bx + j * (sx + dx), by, sx, sy]) for j in range(3)]

    models = (("k3", "cadetblue"), ("k5", "olive"), ("NL", "firebrick"), ("NLP", "darkgreen"))
    diffs = (
        ("NL", "k3", "cadetblue", ":"),
        ("NL", "k5", "olive", ":"),
        ("NLP", "k3", "cadetblue", "-"),
        ("NLP", "k5", "olive", "-"),
    )

    gamma_all = k3["thermal_parameter"].to_numpy() / 1000
    lim_x, ticks_x = nice_ticks(np.nanmin(gamma_all), np.nanmax(gamma_all))

    for j, t in enumerate(isotherms):
        data = main_data[f"isotherm_{t}"]
        x = data["thermal_parameter"] / 1000

        # --- top: depths
        lo, hi = [], []
        for name, colour in models:
            arr = data[name]
            top[j].fill_between(x, arr[:, 2], arr[:, 3], color=colour, alpha=0.3, linewidth=0)
            top[j].plot(x, arr[:, 0], color=colour, linewidth=1.8)
            lo.append(np.nanmin(arr[:, 2]))
            hi.append(np.nanmax(arr[:, 3]))
        lim_y, ticks_y = nice_ticks(min(lo), 0.0,8)
        ticks_y = ticks_y[0::2]
        # Local patch for the nice_ticks
        if not 0 in ticks_y:
            d_tick = ticks_y[2]-ticks_y[1]
            d_tick2 = 0 - ticks_y[-1]
            if d_tick2/d_tick < 0.7: 
                ticks_y[-1] = 0.0 
            else: 
                ticks_y.append(0.0)
        
        
        modify_axes(
            top[j],
            label_x=["n", "n"],
            label_y=[rf"$z^{{{t}\,^\circ C}}_{{max}}$ [km]", "y"],
            lim_x=lim_x, ticks_x=ticks_x,
            lim_y=lim_y, ticks_y=ticks_y[0::2],
            letter=f"[{'abc'[j]}]",
            precision=tick_format(ticks_y),
        )

        # --- bottom: differences of the binned medians (positive = shallower in non-linear)
        lo, hi = [], []
        for nl, lin, colour, ls in diffs:
            dz = data[nl][:, 0] - data[lin][:, 0]
            bot[j].plot(x, dz, color=colour, linestyle=ls, linewidth=1.8)
            lo.append(np.nanmin(dz))
            hi.append(np.nanmax(dz))
        bot[j].axhline(0, color="0.4", linewidth=0.8, linestyle="--")
        lim_y, ticks_y = nice_ticks(min(lo + [0.0]), max(hi + [0.0]))
        modify_axes(
            bot[j],
            label_x=[r"$\gamma$ [1000 km]", "y"],
            label_y=[rf"$\Delta z^{{{t}\,^\circ C}}_{{max}}$ [km]", "y"],
            lim_x=lim_x, ticks_x=ticks_x,
            lim_y=lim_y, ticks_y=ticks_y,
            letter=f"[{'def'[j]}]",
            precision=tick_format(ticks_x, ticks_y),
        )

    legend_elements = [
        Line2D([0], [0], color="cadetblue", linewidth=2, label="k = 3.1 [W/m/K]"),
        Line2D([0], [0], color="olive", linewidth=2, label="k = 5 [W/m/K]"),
        Line2D([0], [0], color="firebrick", linewidth=2, label="NL"),
        Line2D([0], [0], color="darkgreen", linewidth=2, label="NLP"),
        Line2D([0], [0], color="k", linewidth=1.8, linestyle=":", label="NL − linear"),
        Line2D([0], [0], color="k", linewidth=1.8, linestyle="-", label="NLP − linear"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=True,
        prop={"size": _TEXTFONTSIZE_, "weight": "demibold"},
    )

    fig.savefig(f"{path2save}/{name_plot}", dpi=600, transparent=TRANSPARENT)
    plt.close(fig)
# ---
# ---
def box_plot_thermal_model(
    path2save: str, pdb: pd.DataFrame, figure_number: str = 0
) -> None:
    """Box plot of the seismogenic length per thermal model (k=3.1, k=5.0,
    NL, NLP), plus the binned median +- 1 std of the same quantity against
    the thermal parameter, the decoupling depth and the potential temperature.

    Args:
        path2save (str): path to save the figure
        pdb (pd.DataFrame): main database
        figure_number (str, optional): figure number/letter used in the
                output filename. Defaults to 0.
    """
    name_plot = f"Figure_{figure_number}.png"
    y = "d_seismic_ell"
    x_c = "decoupling"           # panel [c]: use "phi" for the friction-angle suite

    k3, k5, knl, knlp = create_common_data_set(pdb=pdb)
    for df in (k5, knl, knlp):
        if "thermal_parameter" not in df.columns:
            df["thermal_parameter"] = k3["thermal_parameter"].to_numpy()

    ensembles = (
        (k3, "cadetblue", "k=3.1"),
        (k5, "olive", "k=5.0"),
        (knl, "firebrick", "NL"),
        (knlp, "darkgreen", "NLP"),
    )

    fig = plt.figure(figsize=(12, 12))
    bx, by = 0.13, 0.1
    sx, sy = 0.38, 0.38
    dx, dy = 0.07, 0.1
    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx + dx + sx, by, sx, sy])
    ax2 = fig.add_axes([bx, by + dy + sy, sx, sy])
    ax3 = fig.add_axes([bx + dx + sx, by + dy + sy, sx, sy])

    # --- [a] box plot
    data = [df[y].to_numpy() for df, _, _ in ensembles]
    bplot = ax0.boxplot(data, patch_artist=True, tick_labels=[lab for _, _, lab in ensembles])
    for patch, (_, colour, _) in zip(bplot["boxes"], ensembles):
        patch.set_facecolor(colour)

    # --- [b]-[d] binned median +- 1 std
    # (axis, column, x label, scale, letter, show y labels)
    panels = (
        (ax1, "thermal_parameter", r"$\gamma$ [1000 km]", 1000.0, "[b]", False),
        (ax2, x_c, r"$d_c$ [km]" if x_c == "decoupling" else r"$\phi$ [$^\circ$]", 1.0, "[c]", True),
        (ax3, "Tp", r"$T_p$ [$^\circ$C]", 1.0, "[d]", False),
    )

    binned = {
        (p, k): create_bin(df, x=col, y=y)
        for p, (_, col, _, _, _, _) in enumerate(panels)
        for k, (df, _, _) in enumerate(ensembles)
    }

    # common y-limits: box-plot data and the +-1 std bands
    lo = [np.nanmin(np.concatenate(data))]
    hi = [np.nanmax(np.concatenate(data))]
    for s in binned.values():
        lo.append(np.nanmin(s["y_med"] - s["y_std"]))
        hi.append(np.nanmax(s["y_med"] + s["y_std"]))
    lim_y, ticks_y = nice_ticks(min(lo), max(hi))

    for p, (ax, col, xlabel, scale, letter, show_y) in enumerate(panels):
        for k, (_, colour, _) in enumerate(ensembles):
            s = binned[(p, k)]
            xm = s["x_mid"] / scale
            ax.fill_between(
                xm, s["y_med"] - s["y_std"], s["y_med"] + s["y_std"],
                color=colour, alpha=0.15, linewidth=0,
            )
            ax.plot(xm, s["y_med"], c=colour, linewidth=1.5)

        xs = k3[col] / scale
        lim_x, ticks_x = nice_ticks(np.nanmin(xs), np.nanmax(xs))
        shown = [ticks_x, ticks_y] if show_y else [ticks_x]
        modify_axes(
            ax,
            label_x=[xlabel, "y"],
            label_y=[r"$\ell^{\mathrm{seismic}}$ [km]", "y"] if show_y else ["n", "n"],
            lim_x=lim_x,
            lim_y=lim_y,
            ticks_x=ticks_x,
            ticks_y=ticks_y,
            letter=letter,
            precision=tick_format(*shown),
        )

    modify_axes(
        ax0,
        label_x=[r"Thermal models", "y"],
        label_y=[r"$\ell^{\mathrm{seismic}}$ [km]", "y"],
        lim_y=lim_y,
        ticks_y=ticks_y,
        label_x_c="categorical",
        ticks_label_x=[lab for _, _, lab in ensembles],
        letter="[a]",
        precision=tick_format(ticks_y),
    )

    fig.savefig(f"{path2save}/{name_plot}", dpi=600, transparent=TRANSPARENT)
    plt.close(fig)
# ---
def extract_max_dpth2(db: pd.DataFrame, isotherms: int) -> np.ndarray:
    """Extract the per-experiment maximum depth reached by a given isotherm.

    Args:
        db (pd.DataFrame): reduced database, must contain a
                `max_dpth{isotherms}` column
        isotherms (int): isotherm temperature used to pick the column
                (e.g. 150, 450, 600)

    Returns:
        np.ndarray: 1D array, one value per experiment/row in `db`
    """
    buf0 = np.zeros(len(db[f"max_dpth{isotherms}"]), dtype=float)
    for i in range(len(db[f"max_dpth{isotherms}"])):
        bufb = db.iloc[i]
        buf0[i] = bufb[f"max_dpth{isotherms}"]
    return buf0
# ---
def box_plot_thermal_model(
    path2save: str, pdb: pd.DataFrame, figure_number: str = 0
) -> None:
    """Box plot of the seismogenic length per thermal model (k=3.1, k=5.0,
    NL), plus scatter plots of the same quantity against thermal parameter,
    decoupling depth and potential temperature.

    Args:
        path2save (str): path to save the figure
        pdb (pd.DataFrame): main database
        figure_number (str, optional): figure number/letter used in the
                output filename. Defaults to 0.
    """
    name_plot = f"Figure_{figure_number}.png"
    # Create common dataset
    k3, k5, knl, knlp = create_common_data_set(pdb=pdb)
    # Compute the seismogenic length
    d_seismic_nl = knl["d_seismic_ell_total"]
    d_seismic_k3 = k3["d_seismic_ell_total"]
    d_seismic_k5 = k5["d_seismic_ell_total"]
    d_seismic_knlp = knlp["d_seismic_ell_total"]


    fig = plt.figure(figsize=(12, 12))
    bx = 0.13
    by = 0.1
    sx = 0.38
    dx = 0.07
    sy = 0.38
    dy = 0.1
    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx + dx + sx, by, sx, sy])
    ax2 = fig.add_axes([bx, by + dy + sy, sx, sy])
    ax3 = fig.add_axes([bx + dx + sx, by + dy + sy, sx, sy])

    data = np.array([d_seismic_k3, d_seismic_k5, d_seismic_nl, d_seismic_knlp])
    lim_y, ticks_y = extract_limit_array(data)

    labels = ["k=3.1", "k=5.0", "NL", "NLP"]
    colors = ["cadetblue", "olive", "firebrick", "darkgreen"]
    bplot = ax0.boxplot(
        data.T,
        patch_artist=True,
        tick_labels=labels,  # fill with color
    )  # will be used to label x-ticks

    # fill with colors
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    ax1.scatter(knl["thermal_parameter"], d_seismic_nl, c="firebrick", s=1.5)
    ax1.scatter(k3["thermal_parameter"], d_seismic_k3, c="cadetblue", s=1.5)
    ax1.scatter(k5["thermal_parameter"], d_seismic_k5, c="olive", s=1.5)
    ax1.scatter(knlp["thermal_parameter"], d_seismic_knlp, c="darkgreen", s=1.5)

    ax2.scatter(knl["phi"], d_seismic_nl, c="firebrick", s=1.5)
    ax2.scatter(k3["phi"], d_seismic_k3, c="cadetblue", s=1.5)
    ax2.scatter(k5["phi"], d_seismic_k5, c="olive", s=1.5)
    ax2.scatter(knlp["phi"], d_seismic_knlp, c="darkgreen", s=1.5)

    ax3.scatter(knl["Tp"], d_seismic_nl, c="firebrick", s=1.5)
    ax3.scatter(k3["Tp"], d_seismic_k3, c="cadetblue", s=1.5)
    ax3.scatter(k5["Tp"], d_seismic_k5, c="olive", s=1.5)
    ax3.scatter(knlp["Tp"], d_seismic_knlp, c="darkgreen", s=1.5)

    data1 = np.array(
        [
            knl["thermal_parameter"].to_numpy(),
            k3["thermal_parameter"].to_numpy(),
            k5["thermal_parameter"].to_numpy(),
            knlp["thermal_parameter"].to_numpy(),
        ]
    )
    lim1_x, ticks_x1 = extract_limit_array(data1)

    data2 = np.array(
        [
            knl["phi"].to_numpy(),
            k3["phi"].to_numpy(),
            k5["phi"].to_numpy(),
            knlp["phi"].to_numpy(),
        ]
    )
    lim2_x, ticks_x2 = extract_limit_array(data2)

    data3 = np.array([knl["Tp"].to_numpy(), k3["Tp"].to_numpy(), k5["Tp"].to_numpy(), knlp["Tp"].to_numpy()])
    lim3_x, ticks_x3 = extract_limit_array(data3)

    ax0 = modify_axes(
        ax0,
        label_x=[r"Thermal models", "y"],
        label_y=[r"$\ell^{\mathrm{seismic}}$ [km]", "y"],
        ticks_y=ticks_y,
        lim_y=lim_y,
        label_x_c="categorical",
        ticks_label_x=["k=3.1", "k=5", "NL", "NLP"],
        letter="[a]",
    )

    ax1 = modify_axes(
        ax1,
        label_x=[r"$\gamma$ [1000 km]", "y"],
        lim_x=lim1_x,
        ticks_x=ticks_x1,
        label_y=["n", "n"],
        ticks_y=ticks_y,
        lim_y=lim_y,
        letter="[b]",
    )
    ax2 = modify_axes(
        ax2,
        label_x=[r"$\phi$ [$^\circ$]", "y"],
        label_y=[r"$\ell^{\mathrm{seismic}}$ [km]", "y"],
        lim_x=lim2_x,
        ticks_x=ticks_x2,
        ticks_y=ticks_y,
        lim_y=lim_y,
        letter="[c]",
    )

    ax3 = modify_axes(
        ax3,
        label_x=[r"$T_p$ $[^\circ C]$", "y"],
        lim_x=lim3_x,
        ticks_x=ticks_x3,
        label_y=["n", "n"],
        ticks_y=ticks_y,
        lim_y=lim_y,
        letter="[d]",
    )

    fig.savefig(f"{path2save}/{name_plot}", dpi=600, transparent=TRANSPARENT)
# ---
def box_plot_thermal_model_max_depth(
    path2save: str, pdb: pd.DataFrame, figure_number: str = 0
) -> None:
    """Box plot of the maximum depth reached by the 150, 450 and 600 C
    isotherms per thermal model (k=3.1, k=5.0, NL, NLP).

    Args:
        path2save (str): path to save the figure
        pdb (pd.DataFrame): main database
        figure_number (str, optional): figure number/letter used in the
                output filename. Defaults to 0.
    """
    def produce_small_data_base(ln_k3:pd.DataFrame
                           ,ln_k4:pd.DataFrame
                           ,ln_nl:pd.DataFrame
                           ,ln_nlp:pd.DataFrame,isotherms:int)->dict:
        
        extracted_data = {}
        extracted_data["k3"] = extract_max_dpth(ln_k3, isotherms)
        extracted_data["k5"] = extract_max_dpth(ln_k4, isotherms)
        extracted_data["NL"] = extract_max_dpth(ln_nl, isotherms)
        extracted_data["NLP"] = extract_max_dpth(ln_nlp, isotherms)
        extracted_data["k3_diff"] = extracted_data["NL"] - extracted_data["k3"]
        extracted_data["k5_diff"] = extracted_data["NL"] - extracted_data["k5"]
        extracted_data["NLP_diff_k5"] = extracted_data["NLP"] - extracted_data["k5"]
        extracted_data["NLP_diff_k3"] = extracted_data["NLP"] - extracted_data["k3"]
        
        
        
        return extracted_data 
    
    

    name_plot = f"Figure_{figure_number}.png"

    # Create common dataset
    k3, k5, knl, knlp = create_common_data_set(pdb=pdb)
    data_150 = produce_small_data_base(k3, k5, knl, knlp, 150)
    data_450 = produce_small_data_base(k3, k5, knl, knlp, 450)
    data_600 = produce_small_data_base(k3, k5, knl, knlp, 600)
    
    
    
    # Compute the seismogenic length
    fig = plt.figure(figsize=(16, 12))
    bx = 0.1
    by = 0.1
    sx = 0.24
    dx = 0.08
    dy = 0.14
    sy = 0.3
    ax0 = fig.add_axes([bx, by + sy + dy, sx, sy])
    ax1 = fig.add_axes([bx + dx + sx, by + sy + dy, sx, sy])
    ax2 = fig.add_axes([bx + 2 * dx + 2 * sx, by + sy + dy, sx, sy])
    ax3 = fig.add_axes([bx, by, sx, sy])
    ax4 = fig.add_axes([bx + dx + sx, by, sx, sy])
    ax5 = fig.add_axes([bx + 2 * dx + 2 * sx, by, sx, sy])
    # ax0
    # Compute the seismogenic length

    data = np.array([data_150['k3'], data_150['k5'], data_150['NL'], data_150['NLP']])
    lim_y0, ticks_y0 = extract_limit_array(data)

    labels = ["k3", "k5", "NL", "NLP"]
    colors = ["cadetblue", "olive", "firebrick", "darkgreen"]
    bplot = ax0.boxplot(
        data.T,
        patch_artist=True,
        tick_labels=labels,  # fill with color
    )  # will be used to label x-ticks

    # fill with colors
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    # ax1
    # Compute the seismogenic length

    data = np.array([data_450['k3'], data_450['k5'], data_450['NL'], data_450['NLP']])
    lim_y1, ticks_y1 = extract_limit_array(data)
    labels = ["k3", "k5", "NL","NLP"]
    colors = ["cadetblue", "olive", "firebrick", "darkgreen"]
    bplot = ax1.boxplot(
        data.T,
        patch_artist=True,
        tick_labels=labels,  # fill with color
    )  # will be used to label x-ticks

    # fill with colors
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    # ax2
    # Compute the seismogenic length

    data = np.array([data_600['k3'], data_600['k5'], data_600['NL'], data_600['NLP']])
    lim_y2, ticks_y2 = extract_limit_array(data)
    labels = ["k3", "k5", "NL", "NLP"]
    colors = ["cadetblue", "olive", "firebrick", "darkgreen"]
    bplot = ax2.boxplot(
        data.T,
        patch_artist=True,
        tick_labels=labels,  # fill with color
    )  # will be used to label x-ticks

    # fill with colors
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    # ax3
    # Compute the seismogenic length

    data = np.array([data_150['k3_diff'], data_150['k5_diff'],data_150['NLP_diff_k3'], data_150['NLP_diff_k5']])
    lim_y3, ticks_y3 = extract_limit_array(data)

    labels = ["NLk3", "NLk5", "NLPk3", "NLPk5"]
    colors = ["cadetblue", "olive", "cadetblue", "olive"]
    bplot = ax3.boxplot(
        data.T,
        patch_artist=True,
        tick_labels=labels,  # fill with color
    )  # will be used to label x-ticks

    # fill with colors
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    ax4.axhline(y=0.0, linewidth=2, color="firebrick")

    # ax4
    # Compute the seismogenic length

    data = np.array([data_450['k3_diff'], data_450['k5_diff'],data_450['NLP_diff_k3'], data_450['NLP_diff_k5']])
    lim_y4, ticks_y4 = extract_limit_array(data)
    labels = ["NLk3", "NLk5", "NLPk3", "NLPk5"]
    colors = ["cadetblue", "olive", "cadetblue", "olive"]
    bplot = ax4.boxplot(
        data.T,
        patch_artist=True,
        tick_labels=labels,  # fill with color
    )  # will be used to label x-ticks
    ax4.axhline(y=0.0, linewidth=2, color="firebrick")

    # fill with colors
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    # ax5
    # Compute the seismogenic length

    data = np.array([data_600['k3_diff'], data_600['k5_diff'], data_600['NLP_diff_k3'], data_600['NLP_diff_k5']])
    lim_y5, ticks_y5 = extract_limit_array(data)
    labels = ["NLk3", "NLk5", "NLPk3", "NLPk5"]
    colors = ["cadetblue", "olive", "cadetblue", "olive"]
    bplot = ax5.boxplot(
        data.T,
        patch_artist=True,
        tick_labels=labels,  # fill with color
    )  # will be used to label x-ticks
    ax5.axhline(y=0.0, linewidth=2, color="firebrick")

    # fill with colors
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    ax0.text(
        0.1,
        1.3,
        r"max depth 150 $^{\circ}C$",
        transform=ax0.transAxes,
        c="w",
        fontweight="demibold",
        fontsize=20,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "k", "alpha": 0.7},
    )

    ax1.text(
        0.1,
        1.3,
        r"max depth 450 $^{\circ}C$",
        transform=ax1.transAxes,
        c="w",
        fontweight="demibold",
        fontsize=20,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "k", "alpha": 0.7},
    )

    ax2.text(
        0.1,
        1.3,
        r"max depth 600 $^{\circ}C$",
        transform=ax2.transAxes,
        c="w",
        fontweight="demibold",
        fontsize=20,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "k", "alpha": 0.7},
    )

    # Check precision generate limits that should be 0.5 or 1.0 or 0.0

    ax0 = modify_axes(
        ax0,
        label_x=[r"Thermal models", "y"],
        label_y=[r"depth [km]", "y"],
        ticks_y=ticks_y0,
        lim_y=lim_y0,
        label_x_c="categorical",
        ticks_label_x=["k=3.1", "k=5", "NL", "NLP"],
        letter="[a]",
    )

    ax1 = modify_axes(
        ax1,
        label_x=[r"Thermal models", "y"],
        label_y=["n", "y"],
        ticks_y=ticks_y1,
        lim_y=lim_y1,
        label_x_c="categorical",
        ticks_label_x=["k=3.1", "k=5", "NL","NLP"],
        letter="[b]",
    )

    ax2 = modify_axes(
        ax2,
        label_x=[r"Thermal models", "y"],
        label_y=["n", "y"],
        ticks_y=ticks_y2,
        lim_y=lim_y2,
        label_x_c="categorical",
        ticks_label_x=["k=3.1", "k=5", "NL","NLP"],
        letter="[c]",
    )

    ax3 = modify_axes(
        ax3,
        label_x=[r"Thermal models", "y"],
        label_y=[r"$\Delta$depth [km]", "y"],
        ticks_y=ticks_y3,
        lim_y=lim_y3,
        label_x_c="categorical",
        ticks_label_x=[r"$NL_{k3}$", r"$NL_{k5}$", r"$NLP_{k3}$", r"$NLP_{k5}$"],
        letter="[d]",
    )

    ax4 = modify_axes(
        ax4,
        label_x=[r"Thermal models", "y"],
        label_y=["n", "y"],
        ticks_y=ticks_y4,
        lim_y=lim_y4,
        label_x_c="categorical",
        ticks_label_x=[r"$NL_{k3}$", r"$NL_{k5}$", r"$NLP_{k3}$", r"$NLP_{k5}$"],
        letter="[e]",
    )

    ax5 = modify_axes(
        ax5,
        label_x=[r"Thermal models", "y"],
        label_y=["n", "y"],
        ticks_y=ticks_y5,
        lim_y=lim_y5,
        label_x_c="categorical",
        ticks_label_x=[r"$NL_{k3}$", r"$NL_{k5}$", r"$NLP_{k3}$", r"$NLP_{k5}$"],
        letter="[f]",
    )

    fig.savefig(f"{path2save}/{name_plot}", dpi=600, transparent=TRANSPARENT)
# ---
def plot_heat_flux_slab_real(
    ax, pdb: pd.DataFrame, x_heat: np.ndarray, c="forestgreen"
) -> tuple[mpl_axes, list, list]:
    """Plot the min/max heat-flux envelope along the slab, shaded, with
    bands marking the decoupling-depth range and the range of the depth of
    maximum shear stress.

    Args:
        ax (mpl_axes): the current axis
        pdb (pd.DataFrame): database containing per-experiment heat flux
                ("Q"), decoupling depth ("maxdcxc") and depth of maximum
                shear stress ("maxtauxc")
        x_heat (np.ndarray): x-coordinate (along-slab distance) shared by
                all the heat-flux profiles
        c (str, optional): line/fill color. Defaults to "forestgreen".

    Returns:
        mpl_axes: the axis with the heat-flux envelope drawn on it
    """
    heat_flux = create_array(pdb["Q"]) * 1000
    # -extract max,min
    hf_min = np.min(heat_flux, 0)
    hf_max = np.max(heat_flux, 0)

    # -extract min_max tau_x
    x_min_tau = pdb["maxtauxc"].min()[0]
    x_max_tau = pdb["maxtauxc"].max()[0]
    # -extract min_max dc
    x_min_dc = pdb["maxdcxc"].min()
    x_max_dc = pdb["maxdcxc"].max()

    if x_min_dc != x_max_dc:
        ax.axvspan(x_min_dc, x_max_dc, alpha=0.2, color="purple")

    else:
        ax.axvline(x_min_dc, color="purple", linewidth=0.8)

    ax.axvspan(x_min_tau, x_max_tau, alpha=0.2, color="orange")

    ax.plot(x_heat, hf_min, color=c, linewidth=0.7, linestyle=":")
    ax.plot(x_heat, hf_max, color=c, linewidth=0.7, linestyle=":")
    ax.fill_between(x_heat, hf_min, hf_max, color=c, alpha=0.4)

    return ax
# ---
def plot_slab_slab_real(
    ax, pdb: pd.DataFrame, c="forestgreen"
,**kwargs) -> mpl_axes:
    """Helper function for the eyecandie figure

    Args:
        ax (mpl_axes): the current object axis
        pdb (pd.DataFrame): the database containing the P-T condition of the slab
        c (str, optional): Color.

    Returns:
        mpl_axes: the axis with the min/max T-P envelope of the slab drawn on it
    """
    if kwargs.get('non_linear_db') is not None:
        pdb_nl = kwargs['non_linear_db']
        target_array = create_array(pdb_nl["T_S"] - pdb["T_S"])
    else: 
        target_array = create_array(pdb["T_S"])
    P_S = create_array(pdb["P_S"])
    # Extract the minimum T and the maximum T per each depth/Pressure
    temp_min = np.min(target_array, 0)
    temp_max = np.max(target_array, 0)
    # -extract max,min
    # Create the plot: plot the bounding curve
    ax.plot(temp_min, P_S[0, :], color=c, linewidth=0.7, linestyle=":")
    ax.plot(temp_max, P_S[0, :], color=c, linewidth=0.7, linestyle=":")
    # Fill the bounding lines
    ax.fill_betweenx(P_S[0, :], temp_min, temp_max, color=c, alpha=0.4)

    return ax
# ---
def plot_eye_candy(
    a: pd.DataFrame,
    b: pd.DataFrame,
    c: pd.DataFrame,
    hplt: np.ndarray,
    s0: CoordinateSurface,
    s1: CoordinateSurface,
    s2: CoordinateSurface,
    ptsave: str,
):
    """Figure plotting function: it plots the figure depicting the 
    T-P condition of the surface of the slab, together with the HPLT
    database from Whitney. It plots, additionally, the heatfluxes from
    the main suite of experiments that has been performed. This 
    function can be used also for plotting other groups of the experiments
    it has been conceived to show the non-linear with shear heating experiment. 

    Args:
        a (pd.DataFrame): Sensitivity tests database
        b (pd.DataFrame): Mexico tests databases
        c (pd.DataFrame): Japan tests database
        hplt (np.ndarray): Database of HPLT
        s0 (CoordinateSurface): Coordinate of the Sensitivity tests
        s1 (CoordinateSurface): Coordinate of the Mexican tests
        s2 (CoordinateSurface): Coordinate of the Japan tests
        ptsave (str): path to save
    """
    # name plot
    name_plot = f"{ptsave}/Figure_f.png"
    # Main figure canvas
    fig = plt.figure(figsize=(16, 12))
    # Slab panel surface
    bx_a = 0.08
    by_a = 0.10
    sx_a = 1 / 2 - bx_a
    sy_a = 0.85
    # Heat fluxes panel
    bx_b = bx_a + sx_a + 0.08
    by_b = 0.1
    sx_b = 0.38
    sy_b = 0.22
    dy_b = 0.08

    ax0 = fig.add_axes([bx_a, by_a, sx_a, sy_a])
    ax1 = fig.add_axes([bx_b, by_b, sx_b, sy_b])
    ax2 = fig.add_axes([bx_b, by_b + 1 * dy_b + 1 * sy_b, sx_b, sy_b])
    ax3 = fig.add_axes([bx_b, by_b + 2 * dy_b + 2 * sy_b, sx_b, sy_b])

    # Slab P-T -> create arrays

    # Plot the petrological database

    # -> Heat Flux Sensitivity
    ax3 = plot_heat_flux_slab_real(ax3, a, s0.x_heat_flux, "silver")
    # -> Heat Flux Mexico
    ax2 = plot_heat_flux_slab_real(ax2, b, s1.x_heat_flux, "firebrick")
    # -> Heat Flux Japan
    ax1 = plot_heat_flux_slab_real(ax1, c, s2.x_heat_flux, "forestgreen")

    # Create Envelope P-T for each of the three esembles
    ax0 = plot_slab_slab_real(ax=ax0, pdb=a, c="silver")
    ax0 = plot_slab_slab_real(ax=ax0, pdb=b, c="firebrick")
    ax0 = plot_slab_slab_real(ax=ax0, pdb=c, c="forestgreen")
    ax0.scatter(hplt.T, hplt.P, s=5.0, c="k", marker="s")
    # Few stuff
    ax3.text(
        0.30,
        1.1,
        r"Sensitivity cases",
        transform=ax3.transAxes,
        c="w",
        fontweight="demibold",
        fontsize=15,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "k", "alpha": 0.7},
    )

    ax2.text(
        0.30,
        1.1,
        r"Mexico",
        transform=ax2.transAxes,
        c="w",
        fontweight="demibold",
        fontsize=15,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "k", "alpha": 0.7},
    )

    ax1.text(
        0.30,
        1.1,
        r"Central Honshu",
        transform=ax1.transAxes,
        c="w",
        fontweight="demibold",
        fontsize=15,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "k", "alpha": 0.7},
    )

    # Create Envelope for each heat fluxes
    ax0 = modify_axes(
        ax0,
        label_x=[r"T [$^{\circ}C$]", "y"],
        label_y=[r"P [GPa]", "y"],
        lim_y=[0, 4],
        ticks_y=[0, 1, 2, 3, 4],
        lim_x=[0, 1100],
        letter="[a]",
    )

    ax1 = modify_axes(
        ax1,
        label_x=["Distance [km]", "y"],
        label_y=[r"q $[mW]$", "y"],
        ticks_y=[30, 60, 90],
        ticks_label_y=[30, 60, 90],
        lim_y=[20, 100],
        lim_x=[0, 500],
        letter="[d]",
    )

    ax2 = modify_axes(
        ax2,
        label_x=["n", "n"],
        label_y=[r"q $[mW]$", "y"],
        ticks_y=[30, 60, 90],
        ticks_label_y=[30, 60, 90],
        lim_y=[20, 100],
        lim_x=[0, 500],
        letter="[c]",
    )

    ax3 = modify_axes(
        ax3,
        label_x=["n", "n"],
        label_y=[r"q $[mW]$", "y"],
        ticks_y=[30, 60, 90],
        ticks_label_y=[30, 60, 90],
        lim_y=[20, 100],
        lim_x=[0, 500],
        letter="[b]",
    )

    fig.savefig(name_plot, dpi=600, transparent=TRANSPARENT)
# ---
# ---
# ---
def figure_seismic_length_difference(
    path2save: str,
    pdb: pd.DataFrame,
    figure_number: str,
    y_total: str = "d_seismic_ell_total",
    y_locked: str = "d_seismic_ell_locked",
) -> None:
    """Paired differences of the seismogenic length between the non-linear and
    the linear thermal models, plotted against the parameter that controls each
    length: plate age for the total length (150-450 C) and convergence velocity
    for the locked length (150-350 C). Lines: binned median; shading: 25-75 %.

    Also prints the medians needed for the text.

    Args:
        path2save (str): path to save the figure
        pdb (pd.DataFrame): main database
        figure_number (str): figure number used in the file name
        y_total (str, optional): column of the total seismogenic length
        y_locked (str, optional): column of the locked seismogenic length
    """
    name_plot = f"Figure_{figure_number}.png"

    k3, k5, knl, knl_p = create_common_data_set(pdb=pdb)

    # the four ensembles must contain the same experiments in the same order
    keys = ["Tp", "age", "vc"]
    for df in (k5, knl, knl_p):
        assert len(df) == len(k3), "ensembles have different numbers of experiments"
        assert np.allclose(df[keys].to_numpy(), k3[keys].to_numpy()), "ensembles not aligned"

    def paired(y: str) -> pd.DataFrame:
        """Per-experiment differences, non-linear minus linear, for column y."""
        out = k3[keys].copy()
        out["NL-k3"] = knl[y].to_numpy() - k3[y].to_numpy()
        out["NL-k5"] = knl[y].to_numpy() - k5[y].to_numpy()
        out["NLP-k3"] = knl_p[y].to_numpy() - k3[y].to_numpy()
        out["NLP-k5"] = knl_p[y].to_numpy() - k5[y].to_numpy()
        return out

    styles = (
        ("NL-k3", "cadetblue", "--"),
        ("NL-k5", "olive", "--"),
        ("NLP-k3", "cadetblue", "-"),
        ("NLP-k5", "olive", "-"),
    )

    # (y column, x column, x label, y label, letter)
    panels = (
        (y_total, "age", r"age [Myr]",
         r"$\Delta\ell^{\mathrm{seismic}}_{\mathrm{total}}$ [km]", "[a]"),
        (y_locked, "vc", r"$v_c$ [cm/yr]",
         r"$\Delta\ell^{\mathrm{seismic}}_{\mathrm{locked}}$ [km]", "[b]"),
    )

    fig = plt.figure(figsize=(12, 5.5))
    bx, by = 0.09, 0.15
    sx, sy = 0.38, 0.68
    dx = 0.11
    axes = [fig.add_axes([bx + j * (sx + dx), by, sx, sy]) for j in range(2)]

    for ax, (y, x, xlabel, ylabel, letter) in zip(axes, panels):
        d = paired(y)

        lo, hi = [0.0], [0.0]
        for col, colour, ls in styles:
            s = create_bin(d, x=x, y=col)
            ax.fill_between(s["x_mid"], s["y_q25"], s["y_q75"],
                            color=colour, alpha=0.12, linewidth=0)
            ax.plot(s["x_mid"], s["y_med"], color=colour, linestyle=ls, linewidth=1.8)
            lo.append(np.nanmin(s["y_q25"]))
            hi.append(np.nanmax(s["y_q75"]))
        ax.axhline(0, color="0.4", linewidth=0.8, linestyle=":")

        lim_x, ticks_x = nice_ticks(np.nanmin(d[x]), np.nanmax(d[x]))
        lim_y, ticks_y = nice_ticks(min(lo), max(hi))
        modify_axes(
            ax,
            label_x=[xlabel, "y"],
            label_y=[ylabel, "y"],
            lim_x=lim_x,
            lim_y=lim_y,
            ticks_x=ticks_x,
            ticks_y=ticks_y,
            letter=letter,
            precision=tick_format(ticks_x, ticks_y),
        )

        # --- numbers for the text
        print(f"\n{y} against {x} (paired median [25-75 %], km):")
        s_all = {col: create_bin(d, x=x, y=col) for col, _, _ in styles}
        first = {c: s.iloc[0] for c, s in s_all.items()}
        last = {c: s.iloc[-1] for c, s in s_all.items()}
        for c in s_all:
            print(f"  {c:7s} lowest {x} bin: {first[c]['y_med']:6.1f} "
                  f"[{first[c]['y_q25']:.1f}, {first[c]['y_q75']:.1f}]   "
                  f"highest {x} bin: {last[c]['y_med']:6.1f} "
                  f"[{last[c]['y_q25']:.1f}, {last[c]['y_q75']:.1f}]   "
                  f"all: {d[c].median():6.1f}")
        if x == "age":
            old = d[d["age"] > 60]
            for c in s_all:
                print(f"  {c:7s} age > 60 Myr: median {old[c].median():.1f}, "
                      f"max |diff| of bin medians {np.nanmax(np.abs(s_all[c]['y_med'][s_all[c]['x_mid'] > 60])):.1f}")

    legend_elements = [
        Line2D([0], [0], color="cadetblue", linewidth=1.8, linestyle="-", label="NLP − k=3.1"),
        Line2D([0], [0], color="olive", linewidth=1.8, linestyle="-", label="NLP − k=5"),
        Line2D([0], [0], color="cadetblue", linewidth=1.8, linestyle="--", label="NL − k=3.1"),
        Line2D([0], [0], color="olive", linewidth=1.8, linestyle="--", label="NL − k=5"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=True,
        prop={"size": _TEXTFONTSIZE_, "weight": "demibold"},
    )

    fig.savefig(f"{path2save}/{name_plot}", dpi=600, transparent=TRANSPARENT)
    plt.close(fig)
    
    
def plot_heat_flux_slab_real(
    ax, pdb: pd.DataFrame, x_heat: np.ndarray, c="forestgreen", bands: bool = True
) -> tuple[mpl_axes, list, list]:
    """Plot the min/max heat-flux envelope along the slab, shaded, with
    bands marking the decoupling-depth range and the range of the depth of
    maximum shear stress.

    Args:
        ax (mpl_axes): the current axis
        pdb (pd.DataFrame): database containing per-experiment heat flux
                ("Q"), decoupling depth ("maxdcxc") and depth of maximum
                shear stress ("maxtauxc")
        x_heat (np.ndarray): x-coordinate (along-slab distance) shared by
                all the heat-flux profiles
        c (str, optional): line/fill color. Defaults to "forestgreen".
        bands (bool, optional): draw the decoupling-depth and max-shear-stress
                bands. Set to False when several ensembles are overlaid on the
                same axis, so the bands are drawn only once. Defaults to True.

    Returns:
        mpl_axes: the axis with the heat-flux envelope drawn on it
    """
    heat_flux = create_array(pdb["Q"]) * 1000
    # -extract max,min
    hf_min = np.min(heat_flux, 0)
    hf_max = np.max(heat_flux, 0)

    if bands:
        # -extract min_max tau_x
        x_min_tau = pdb["maxtauxc"].min()[0]
        x_max_tau = pdb["maxtauxc"].max()[0]
        # -extract min_max dc
        x_min_dc = pdb["maxdcxc"].min()
        x_max_dc = pdb["maxdcxc"].max()

        if x_min_dc != x_max_dc:
            ax.axvspan(x_min_dc, x_max_dc, alpha=0.2, color="purple")

        else:
            ax.axvline(x_min_dc, color="purple", linewidth=0.8)

        ax.axvspan(x_min_tau, x_max_tau, alpha=0.2, color="orange")

    ax.plot(x_heat, hf_min, color=c, linewidth=0.7, linestyle=":")
    ax.plot(x_heat, hf_max, color=c, linewidth=0.7, linestyle=":")
    ax.fill_between(x_heat, hf_min, hf_max, color=c, alpha=0.4)

    return ax
# ---
def relative_error_heat_flux(
    pdb_ref: pd.DataFrame, pdb_lin: pd.DataFrame
) -> np.ndarray:
    """Relative error of the surface heat flux of a linear model with respect to
    a non-linear (reference) model, computed experiment by experiment.

    The experiments are paired on the input parameters [Tp, vc, age], so that
    each linear experiment is compared with the non-linear experiment that has
    the same input. Experiments missing in either ensemble are dropped.

        err = 100 * (Q_ref-Q_lin) / Q_ref     [%]

    Args:
        pdb_ref (pd.DataFrame): non-linear (reference) ensemble, with column "Q"
        pdb_lin (pd.DataFrame): linear ensemble, with column "Q"

    Returns:
        np.ndarray: array (n_pairs, n_points) of relative errors in %
    """
    keys = ["Tp", "vc", "age"]
    pairs = pdb_ref[keys + ["Q"]].merge(
        pdb_lin[keys + ["Q"]], on=keys, how="inner", suffixes=("_ref", "_lin")
    )
    print(f"Paired experiments: {len(pairs)} of {len(pdb_ref)} (reference)")

    q_ref = create_array(pairs["Q_ref"])
    q_lin = create_array(pairs["Q_lin"])

    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(np.abs(q_ref) > 0.0, 100.0 * (q_ref - q_lin) / q_ref, np.nan)

    return err

def plot_heat_flux_relative_error(
    pdb: pd.DataFrame,
    crd: CoordinateSurface,
    path2save: str,
    figure_number: str = "d",
    k_codes: dict = {"k3": 3.1, "k5": 5.0, "nl": 0.0, "nlp": -1.0},
) -> None:
    """Two-panel figure for the sensitivity study:
    [a] surface heat flux (min-max envelope) of each thermal model along the
        trench-normal distance, with the decoupling-depth and max-shear-stress bands;
    [b] relative error of the linear models with respect to the non-linear ones,
        err = 100 (Q_ref-Q_lin) / Q_ref, paired experiment by experiment
        (median and 25th-75th percentiles).

    Args:
        pdb (pd.DataFrame): merged database of all the ensembles
        crd (CoordinateSurface): coordinates (x_heat_flux) shared by all experiments
        path2save (str): folder where to save the figure
        figure_number (str, optional): figure label. Defaults to "d".
        k_codes (dict, optional): value of the "k" column identifying each ensemble.
            NL and NLP must have different codes. Ensembles not found are skipped.
    """
    name_plot = f"{path2save}/Figure_{figure_number}.png"

    k3, k5, knl, knl_p = create_common_data_set(pdb=pdb)


    # Split the merged database into the ensembles
    ensembles = {'k3': k3, 'k5': k5, 'nl': knl, 'nlp': knl_p}


    colors = {"k3": "cadetblue", "k5": "olive", "nl": "firebrick", "nlp": "darkgreen"}
    labels = {"k3": r"$k=3.1$", "k5": r"$k=5$", "nl": "NL", "nlp": "NLP"}
    styles_ref = {"nl": "--", "nlp": "-"}

    fig = plt.figure(figsize=(16, 7))
    bx, by, sx, sy, dx = 0.08, 0.14, 0.38, 0.76, 0.10
    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx + sx + dx, by, sx, sy])

    x = crd.x_heat_flux

    # [a] heat flux envelopes; the bands are drawn only with the first ensemble
    first = True
    for key in ["k3", "k5", "nl", "nlp"]:
        if key not in ensembles:
            continue
        ax0 = plot_heat_flux_slab_real(ax0, ensembles[key], x, c=colors[key], bands=first)
        first = False

    # [b] relative error of the linear models wrt the non-linear ones
    all_err = []
    for ref in ["nl", "nlp"]:
        if ref not in ensembles:
            continue
        for lin in ["k3", "k5"]:
            if lin not in ensembles:
                continue
            err = relative_error_heat_flux(ensembles[ref], ensembles[lin])
            med = np.nanmedian(err, 0)
            p25 = np.nanpercentile(err, 25, axis=0)
            p75 = np.nanpercentile(err, 75, axis=0)
            ax1.plot(x, med, color=colors[lin], linestyle=styles_ref[ref], linewidth=1.8)
            ax1.fill_between(x, p25, p75, color=colors[lin], alpha=0.2, linewidth=0)
            all_err.append(np.concatenate([p25, p75]))
    ax1.axhline(0.0, color="k", linewidth=0.8)

    # Legends: models (colours) in [a]; references (line styles) in [b]
    handles_a = [
        Line2D([0], [0], color=colors[k], linewidth=6, alpha=0.4, label=labels[k])
        for k in ["k3", "k5", "nl", "nlp"]
        if k in ensembles
    ]
    ax0.legend(handles=handles_a, loc="upper right", fontsize=_TEXTFONTSIZE_, frameon=False)
    handles_b = [
        Line2D([0], [0], color=colors["k3"], linewidth=1.8, label=r"$k=3.1$"),
        Line2D([0], [0], color=colors["k5"], linewidth=1.8, label=r"$k=5$"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.8, label="vs NL"),
    ]
    if "nlp" in ensembles:
        handles_b.append(Line2D([0], [0], color="k", linestyle="-", linewidth=1.8, label="vs NLP"))
    ax1.legend(handles=handles_b, loc="best", fontsize=_TEXTFONTSIZE_, frameon=True,
               facecolor="w", edgecolor="none", framealpha=0.8)

    # Limits rounded to 5 %, ticks every 5 or 10 %
    err_all = np.concatenate(all_err)
    err_all = err_all[np.isfinite(err_all)]
    lo = 5.0 * np.floor(min(err_all.min(), 0.0) / 5.0)
    hi = 5.0 * np.ceil(max(err_all.max(), 0.0) / 5.0)
    step = 5.0 if (hi - lo) <= 30.0 else 10.0
    ticks_y_err = list(np.arange(lo, hi + 0.5 * step, step))
    lim_y_err = [lo - 0.5 * step, hi + 0.5 * step]

    ax0 = modify_axes(
        ax0,
        label_x=["Distance [km]", "y"],
        label_y=[r"q [mW m$^{-2}$]", "y"],
        ticks_y=[30, 60, 90],
        lim_y=[20, 100],
        lim_x=[0, 500],
        letter="[a]",
        precision="%.0f",
    )
    ax1 = modify_axes(
        ax1,
        label_x=["Distance [km]", "y"],
        label_y=[r"Relative error of $q$ [%]", "y"],
        lim_y=lim_y_err,
        ticks_y=ticks_y_err,
        lim_x=[0, 500],
        letter="[b]",
        precision="%.0f",
    )

    fig.savefig(name_plot, dpi=600, transparent=TRANSPARENT)
    plt.close(fig)
# ---