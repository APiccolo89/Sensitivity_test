import argparse
import os
import shutil
import time
from importlib.resources import files
from pathlib import Path

import numpy as np
from mpi4py import MPI

import post_process_module.global_var as _GVARIABLES_
from post_process_module.post_process_data import (
    save_test_data,
    save_test_data_td_alt,
    timing_function,
)
from post_process_module.utils import (
    change_geometry,
    define_material_properties,
    parse_the_argument,
)

# --- function "wrap"


def wrap_stoned(inp, Ph, arguments: dict) -> tuple:
    """A fake wrap.

    Args:
        inp (_type_): input configuration data
        Ph (_type_): phase data base

    Raises:
        ValueError: failed attempt

    This is a quick hack solution. In the original plan, I would have been able to configure a no-race condition
    for the mesh generation and cache generation for the thermal properties. In practice, I was not able to find a proper solution.
    This should be possible, but, I am not willing to spend a few hours in this task. So, here you are the quick and dirty solution.
    TO DO: Reintegrate the h5 file routine for the thermal properties and create a control called: caching.
    """
    import copy

    from stonedfenicsx.stoned_fenicsx import stoned_fenicsx

    # Initialise the input
    attempt = 0
    success = 0
    last_error = None
    # Try to run the test, if it fails, wait for 20 seconds and try again. This is to avoid the problem of the cache that can cause the test to fail.
    # I need to solve the issue of the cache. In the future I will split the problem of mesh and cache generation, before the actual run
    # Or find a proper way to cleanly use race conditions. Meanwhile, this quick fix seems robust.

    while attempt < 3 and success == 0:
        # Copy the classes otherwise it crashes
        A = copy.deepcopy(inp)
        B = copy.deepcopy(Ph)
        try:
            print(f"Slab type is : {A.g_input.slab_type}")
            if A.g_input.slab_type == "FromFile":
                print(f"Slab type is : {A.g_input.sub_path}")
            stoned_fenicsx(A, B)
            print(f"Test was a success at {attempt}")
            success = 1
            # Copy the post processed classes
            inp = copy.deepcopy(A)
            Ph = copy.deepcopy(B)
        except Exception as e:
            # ok, understood. This method is not safe guarding the test path, this create a few issue, and error I patch
            # it. The reason that I am doing it is the following: it is not only for lazyness. I want to use the same version
            # of the code for this paper. If i start aiming the perfection, I will enter in an never ending loop that produce
            # a wonderful code, that will not have any paper attached.
            last_error = e

            time.sleep(20)  # seconds, float allowed
            attempt = attempt + 1
            print(f"Test failed at {attempt} with error: {e}")
        del A, B

    if success == 0:
        raise ValueError("Failed attempt") from last_error

    return inp, Ph


# ---
@timing_function
def perform_test(args: dict | None, name_suite: str = "debug") -> tuple[str, str]:
    """Configuration script of StonedFEniCSx for the time-dependent sensitivity
    tests.

    The script reads the input data from a .csv table and creates the input data for StonedFEniCSx. The script also creates the material properties dictionary and assigns it to the phases. The script also creates the input data for StonedFEniCSx and runs the simulation. The results are saved in a folder, in a number of h5 files that will be merged by other scripts.
        Args:
            args (SimpleNamespace, optional): updated Arguments of the script.
            name_suite (str, optional): Name of the test suite. Defaults to 'debug'.

        Raises:
            ValueError: _description_

        Returns:
            tuple[str,str]: general path of the tests and name of the test
    """
    # To configure the simulation, the code must ALWAYS read the input.yml file. This is a template file that contains the default values of the input data. The code will then modify the values of the input data according to the arguments passed to the script.
    from stonedfenicsx.config.input_parser import parse_input

    # Find the path of the file
    path_general = os.path.dirname(os.path.realpath(__file__))
    # Read the input file
    path_input = f"{path_general}/input_tests.yaml"
    # Set the general path for the tests
    path_general = os.path.join(path_general, "test_folder")
    # Create the test folder if does not exist
    if not os.path.exists(path_general):
        os.makedirs(path_general)

    # Parse the blue print of the test
    inp, Ph = parse_input(path_input)
    # If you are debugging, use group debug
    if args["group"] == "debug":
        inp.ctrl.time_max = 0.05

    # controls
    # Assume that the decoupling is always active
    inp.ctrl.decoupling_ctrl = 1
    # Choose the mode of stonedFEniCSx
    inp.ctrl.steady_state = args["steady_state"]
    if args["steady_state"] == 1:
        inp.ctrl.tol = 1e-6
        inp.ctrl.it_max = 50
        inp.ctrl.tol_dtemp = 1e-2
    else:
        inp.ctrl.tol = 1e-2
        inp.ctrl.it_max = 10

    # Choose the shear heating mode
    if args["shear_heating"]:
        inp.ctrl.model_shear = "SelfConsistent"
    else:
        inp.ctrl.model_shear = "NoShear"

    # Create input data - Input is a class populated by default dataset
    # A flag that generate the geometry of the benchmark
    # The input path for saving the results
    inp.ctrl_tbc.slab_age = args["age"]
    inp.ctrl_tbc.temp_max = args["Tp"]
    # Velocity of slab
    inp.ctrl_ky.v_s = np.array([args["vc"], 0], dtype=np.float64)

    # Geometrical input
    inp.g_input.cr = 40.0  # Overriding crust
    inp.g_input.lc = 0.5  # relative amount of lower crust
    inp.g_input.ocr = 6.0  # Crustal thickness
    inp.g_input.decoupling = args.get(
        "decoupling", _GVARIABLES_._DECOUPLING_
    )  # Lithospheric mantle depth
    inp.g_input.redo_mesh = [
        args.get("decoupling", _GVARIABLES_._DECOUPLING_) != _GVARIABLES_._DECOUPLING_
    ]
    inp.g_input.lab_d = inp.g_input.decoupling  # depth of the lab
    inp.g_input.ns_depth = 50  # depth of the noslip boundary condition
    inp.g_input.transition = 10  # transition delta depth -> the distance at which the transition from uncoupled and coupled flow happens
    inp.g_input.slab_type = "CustomRibe"  # type of slab geometry
    inp.g_input.resolution_normal = 10.0  # Resolution normal
    inp.g_input.resolution_refine = (
        1.5  # resolution around the slab top surface and the oceanic crust within the slab.
    )

    if args.get("plate_real") is not None:
        change_geometry(plate=args["plate_real"], args=args, inp=inp)
        inp.g_input.slab_type = "FromFile"  # type of slab geometry
        inp.g_input.sub_path = str(
            files("stonedfenicsx").resolve().parents[0]
            / "examples"
            / "data"
            / f"{args['plate_real']}_slab.pz"
        )

    if args["shear_heating"]:
        Ph.shear_heating_disl_law = args.get("shear_heating_law", _GVARIABLES_._SHEAR_HEATING_LAW)
        Ph.shear_heating_disl_phi = args.get("phi", _GVARIABLES_._PHI_)

    define_material_properties(Ph, args)

    # Path to save the results

    inp.ctrl_io.path_save = os.path.join(
        path_general,
        f"{name_suite}_{args['group']}_{_GVARIABLES_.dict_td[str(args['steady_state'])]}",
    )
    inp.ctrl_io.test_name = f"{name_suite}_case{args['case_index']}_{args['group']}_{_GVARIABLES_.dict_td[str(args['steady_state'])]}"

    # Run the test
    inp, Ph = wrap_stoned(inp=inp, Ph=Ph, arguments=args)

    return inp.ctrl_io.path_test, inp.ctrl_io.test_name


# ---
@timing_function
def configure_run_save():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case_index",
        type=int,
        help="Index of the case to run. This is an integer that corresponds to the row of the .csv table.",
    )
    parser.add_argument("--group", type=str, help="Group to which the case belongs.")
    parser.add_argument(
        "--id_batch",
        type=int,
        help="Id of the batch. This is the processor id. It always assume an array job.",
    )
    parser.add_argument(
        "--name_suite",
        type=str,
        help="Name of the test suite. This is a string that corresponds to the name of the test suite.",
    )
    parser.add_argument(
        "--steady_state",
        type=int,
        default=1,
        help="Flag to run the test in steady-state or time-dependent mode. 1 = steady-state, 0 = time-dependent",
    )
    parser.add_argument(
        "--name_csv",
        type=str,
        default="cases_Sensitivity_td.csv",
        help="Name of the .csv table that contains the cases to run. This is a string that corresponds to the name of the .csv table.",
    )
    parser.add_argument(
        "--remove_results",
        type=int,
        default=1,
        help="Flag to remove the results after the test is completed. 1 = remove, 0 = keep",
    )
    parser.add_argument(
        "--plate_real",
        type=str,
        default="None",
        help="Name of the plate to use for the real geometry (only the basic one stored in StonedFEniCSx is available). If None, the default geometry is used.",
    )
    parser.add_argument(
        "--shear_heating_law",
        type=str,
        default="NotDefined",
        help="Name of the shear heating law to use. If None, the default law is used.",
    )

    args_cli = parser.parse_args()

    name_suite = args_cli.name_suite

    path2table = Path(__file__).resolve().parents[0] / "generation_cases" / args_cli.name_csv

    arguments = parse_the_argument(args_cli, pathtotable=path2table)

    pathtest, _ = perform_test(args=arguments, name_suite=name_suite)

    print(f"Test completed for group {arguments['case_index']}. Results saved in {pathtest}")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print(f"Rank {rank} is saving the test data for group {arguments['group']}")
        if arguments["steady_state"] == 0:
            save_test_data_td_alt(name_suite=name_suite, pathtotest=pathtest, arguments=arguments)
        else:
            save_test_data(name_suite=name_suite, pathtotest=pathtest, arguments=arguments)
        # remove the tests folder

    print(f"Removing path test {pathtest}")
    shutil.rmtree(Path(pathtest))
    print(f"Removed pathtest {pathtest}")
    print(f"Finished group {arguments['case_index']}")


# ---

if __name__ == "__main__":
    configure_run_save()
