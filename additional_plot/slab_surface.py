"""Build a StonedFEniCSx mesh for a given slab/plate and plot its boundary
conditions, for inspection or for the manuscript's mesh figures.
"""

import sys
from importlib.resources import files
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "post_process_module"))

from figure_plot import modify_axes
from utils import change_geometry

# Conversion pt - cm
_PT_CM_ = 0.035
# Ticks, label and text font size
_TICKSFONTSIZE_ = 0.8 / _PT_CM_
_LABELFONTSIZE_ = 0.9 / _PT_CM_
_TEXTFONTSIZE_ = 0.65 / _PT_CM_


# ---
def extract_boundary(GD, X, dict_boundary: dict) -> dict:
    """Small function that collects the node associated with the boundaries


    Args:
        GD: Global Domain object (StonedFEniCSx mesh + boundary tags)
        X (np.ndarray): Coordinates of the point
        dict_boundary (dict): Internal dictionary of the boundary condition

    Returns:
        dict: dictionary that associated the boundary condition with ordered coordinates -> yes, I am abusing of the dictionary
    """

    import dolfinx

    tdim = GD.mesh.topology.dim
    fdim = tdim - 1

    GD.mesh.topology.create_connectivity(fdim, tdim)

    x_real = X[:, 0]
    z_real = X[:, 1]

    boundary_structure = {}

    for b in dict_boundary.keys():
        # Copy these horrible arrays
        x = x_real.copy()
        z = z_real.copy()
        # Find the facets
        facets = GD.facets.find(dict_boundary[b])  # or your own facet array

        # (n_facets, n_nodes_per_facet) array of geometry dof indices
        geom = dolfinx.mesh.entities_to_geometry(GD.mesh, fdim, facets)
        # Check for duplicate and purge them
        nodes = np.unique(geom.reshape(-1))
        # Select the right node from the lesser copies
        x = x[nodes]
        z = z[nodes]
        # Check which direction should happen the sorting (I had to change the boudary, because it appears that one method that works with a
        # processed mesh, is working wrongly with the real data of the mesh.) -> WRONG FUNCTION INITIALLY ARGSORT!!!!!!
        if b in ["Top", "Bottom_sla", "Bottom_wed"]:
            ind = x.argsort()
        else:
            ind = z.argsort()
        # Readapt the array following the ordering
        x = x[ind]
        z = z[ind]
        # Transpose the vector because is (1,nx) and it is annoying and create the
        # array that host the chosen one
        x_bd = np.zeros([len(x.T), 2], dtype=np.float32)
        # put the data in the new structure
        x_bd[:, 0] = x[:]
        x_bd[:, 1] = z[:]
        # update the empty dictionary
        boundary_structure[b] = x_bd
        # Just warn me whether or not you did the task
        print(f"Collecting the data for the boundary {b}")

    return boundary_structure


# ---
def plot_mesh_boundary(mesh, name_mesh: str, path2save: str) -> None:
    """Plot the mesh nodes and the boundary polylines (colored by whether
    the boundary is an ocean/crust boundary or an internal one) and save
    the figure.

    Args:
        mesh: StonedFEniCSx mesh object (must expose `.global_domain`)
        name_mesh (str): name used for the output file
        path2save (str): directory to save the figure in
    """

    name_file = f"{path2save}/{name_mesh}.png"

    GD = mesh.global_domain
    # Extract mesh:
    X = GD.mesh.geometry.x[:]
    # Extract the main boundary
    boundary_structure = extract_boundary(GD=GD, X=X, dict_boundary=GD.bc_dict)
    # loop over the main boundary and extract the nodes

    fig = plt.Figure(figsize=[14, 10])
    ax0 = fig.gca()

    ax0.scatter(X[:, 0], X[:, 1], c="forestgreen", s=0.1, alpha=0.6)

    for i in GD.bc_dict.keys():
        if i in ["Oceanic", "LCrust_overplate", "Crust_overplate"]:
            c = "k"
            linewidth = 0.8
        else:
            c = "firebrick"
            linewidth = 2.0

        ax0.plot(
            boundary_structure[i][:, 0],
            boundary_structure[i][:, 1],
            linewidth=linewidth,
            c=c,
        )

    ax0 = modify_axes(
        ax=ax0, label_x=["Distance [km]", "y"], label_y=["Depth [km]", "y"]
    )

    fig.savefig(name_file, dpi=600, transparent=True)


# ---
def process_mesh(slab: str = "CustomRibe", name_plate: str = "Default") -> None:
    """Script that uses a few functionalities of stonedfenicsx to build up a mesh.xmf file
    The main aim is to use the full functionalities to produce a mesh.xdmf file, and using the
    routines to produce the plot of each of the meshes used in this work. It is an advanced
    topic that requires a bit of knowledge on how I organised this monstruosity.

    Args:
        slab (str, optional): slab-geometry name; also used as the plate
                name when name_plate != "CustomRibe". Defaults to 'CustomRibe'.
        name_plate (str, optional): "CustomRibe" to use the built-in slab
                geometry, anything else to load `slab`'s geometry from a
                real-plate file instead. Defaults to 'Default'.
    """

    import stonedfenicsx.create_mesh.create_mesh as create_mesh
    from stonedfenicsx.config.input_parser import parse_input

    # Find the path of the file
    path_general = Path(__file__).resolve().parents[1]
    # Read the input file
    path_input = f"{path_general}/input_tests.yaml"
    inp, ph = parse_input(path_input)
    # Read input and modify the geometric input as a function of the directives.
    inp.g_input.cr = 40.0  # Overriding crust
    inp.g_input.lc = 0.5  # relative amount of lower crust
    inp.g_input.ocr = 6.0  # Crustal thickness
    inp.g_input.decoupling = 80
    inp.g_input.redo_mesh = True
    inp.g_input.lab_d = inp.g_input.decoupling  # depth of the lab
    inp.g_input.ns_depth = 50  # depth of the noslip boundary condition
    inp.g_input.transition = 10  # transition delta depth -> the distance at which the transition from uncoupled and coupled flow happens
    inp.g_input.slab_type = "CustomRibe"  # type of slab geometry
    inp.g_input.resolution_normal = 10.0  # Resolution normal
    inp.g_input.resolution_refine = 1.5  # resolution around the slab top surface and the oceanic crust within the slab.

    if name_plate != "CustomRibe":
        change_geometry(plate=slab, args=None, inp=inp)
        inp.g_input.slab_type = "FromFile"  # type of slab geometry
        inp.g_input.sub_path = str(
            files("stonedfenicsx").resolve().parents[0]
            / "examples"
            / "data"
            / f"{slab}_slab.pz"
        )
    # create the fake folders:
    inp.ctrl_io.path_save = Path(__file__).resolve().parents[0] / "mesh"
    inp.ctrl_io.test_name = Path(__file__).resolve().parents[0] / "mesh"

    inp.ctrl_io.generate_io()
    # convert the unit of measure of the angle
    inp.g_input.check_class_consistency()
    # create the mesh object
    mesh = create_mesh.create_mesh(
        ioctrl=inp.ctrl_io, g_input=inp.g_input, ctrl=inp.ctrl
    )
    # Extract coordinate
    # Plot geometry:
    path = Path(__file__).resolve().parents[1]
    path_2_save = path / "additional_figure"
    path_2_save.mkdir(parents=True, exist_ok=True)
    plot_mesh_boundary(mesh, name_mesh=slab, path2save=str(path_2_save))


# ---
def main():
    """Build and plot the mesh for the Japan real-plate geometry."""
    plate_name = "FromFile"
    slab = "Japan"
    process_mesh(slab=slab, name_plate=plate_name)


# ---
# ---
if __name__ == "__main__":
    main()
