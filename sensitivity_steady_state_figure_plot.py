import argparse
from pathlib import Path

import pandas as pd

from post_process_module.figure_plot import (
    box_plot_thermal_model,
    box_plot_thermal_model_max_depth,
    data_base_PT_whitney,
    figure_slab_surface,
    plot_slab_variances,
    plot_slab_versus_thermalpar,
    print_save_IQ_information,
)
from post_process_module.post_process_data import timing_function
from post_process_module.process_data_visualisation import wrapper_merging_h5


# ---
class ErrorParsing(Exception):
    """_summary_

    Args:
        Exception (_type_): Error of the parser. Check if the user provide the right errors, and create an instance
        of the error on the basis of the most likely mistake.
    """

    def __init__(self, message: str, error_code: int):
        self.message = message
        self.error_code = error_code

    def __str__(self):
        if self.error_code == -1:
            string_error = (
                f"Code error{self.error_code}: Base... the required arguments has not been given:\n"
                f"{self.message}"
            )
        elif self.error_code == 0:
            string_error = f"Code error {self.error_code}: Wrong sh-group: \b{self.message}"
        elif self.error_code == 1:
            string_error = (
                f"Code error {self.error_code}: Wrong steady state flag: \b{self.message}"
            )
        elif self.error_code == 2:
            string_error = (
                f"Code error {self.error_code}: Wrong suite name, "
                f"all the test of the suite are not existing: \n {self.message}"
            )
        elif self.error_code == 3:
            string_error = (
                f"Code error {self.error_code}. Not all the group of tests has been performed:\n"
                f"{self.message}"
            )
        elif self.error_code == 4:
            string_error = f"Code error {self.error_code}. WRONG CSV file:\n{self.message}"
        elif self.error_code == 5:
            string_error = f"Code error {self.error_code}. Wrong function plot:\n{self.message}"
        else:
            string_error = "Ops, unknown error... "

        return string_error


# ---
def check_parsed_arguments(args) -> None:
    """_summary_

    Args:
        args (_type_): parsed argument of the script

    Raises:
        ErrorParsing: If the group name is wrong, or all the test are not performed, or any of the test is not
        performed, raise ErrorParsing.
    """
    required = ["sh", "name_suite", "csv", "steady_state"]

    if args.steady_state == 1:
        args.steady_state = "SS"
    elif args.steady_state == 0:
        args.steady_state = "TD"
    else:
        args.steady_state = "error"
        raise ErrorParsing("Wrong Steady State", -1)

    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        s0 = f"missing required arguments: {', '.join('--' + m for m in missing)}"
        raise ErrorParsing(s0, -1)

    if args.sh not in ("sh", "nsh"):
        s0 = f"The --sh argument must be sh or nsh got {args.sh!r}"
        raise ErrorParsing(s0, 0)

    if args.steady_state == "error":
        s0 = "Wrong steady state flag"
        raise ErrorParsing(s0, 1)

    aux_path = Path(__file__).resolve().parent
    folder = []
    sub_group = ("k3", "k5", "nl")
    for i in sub_group:
        aux_path_data_base = (
            aux_path
            / "Output_h5_files"
            / f"{args.name_suite}_h5_files_{i}{args.sh}_{args.steady_state}"
        )
        aux_path_post_process = (
            aux_path / f"{args.name_suite}_h5_files_{i}{args.sh}_{args.steady_state}.h5"
        )
        if aux_path_post_process.is_file():
            folder.extend([True])
            print("The merged file exist, so the check is not necessary")
        else:
            folder.extend([aux_path_data_base.is_dir()])

    s0 = None
    # Check whether or not all the folder are not existing
    if all(not value for value in folder):
        s0 = f"error_0: name suite error. All the groups for {args.sh} are missing. Check the spelling or run the suite\n"
        raise ErrorParsing(s0, 2)
        # Check if any folder is false
    elif not all(folder):
        indices = [index for index, value in enumerate(folder) if not value]
        values = [f"{sub_group[i]}{args.sh}" for i in indices]
        suite_wrong = ""
        for i in values:
            suite_wrong = suite_wrong + " " + i
        s0 = f"error_1: not all the group has been executed. missing suites: {suite_wrong!r}/n"
        raise ErrorParsing(s0, 3)

    csv_path = aux_path / "generation_cases" / args.csv
    if not csv_path.is_file():
        s0 = f"The csv file listed is not existing. Actual value {args.csv!r}"
        raise ErrorParsing(s0, 4)

    print("Updated the flag of steady state from 1 or 0 to SS or TD")

    print("The parsing is OK!")


# ---
@timing_function
def create_data_structure(aux_path: str, args):
    print("Merging partial data files...")

    data_table_0, surface_data_0, slab_2D_0 = wrapper_merging_h5(
        aux_path,
        partial=(f"k3{args.sh}", args.steady_state),
        name_suite=args.name_suite,
        recreate=False,
        csv=args.csv,
    )
    data_table_1, _, slab_2D_1 = wrapper_merging_h5(
        aux_path,
        partial=(f"k5{args.sh}", args.steady_state),
        name_suite=args.name_suite,
        recreate=False,
        csv=args.csv,
    )
    data_table_2, _, slab_2D_2 = wrapper_merging_h5(
        aux_path,
        partial=(f"nl{args.sh}", args.steady_state),
        name_suite=args.name_suite,
        recreate=False,
        csv=args.csv,
    )
    no_pressure = False
    try:
        data_table_3, _, slab_2D_3 = wrapper_merging_h5(
        aux_path,
        partial=(f"nlp{args.sh}", args.steady_state),
        name_suite=args.name_suite,
        recreate=False,
        csv=args.csv,
        )    
        no_pressure = True

    except Exception as e:
        print(f"Error in merging nl{args.sh} and nlp{args.sh}: {e}")
        print("Proceeding with available data.")
        slab_2D = (slab_2D_0, slab_2D_1, slab_2D_2)
        data_table = pd.concat([data_table_0, data_table_1, data_table_2], ignore_index=True)
    finally:
        if no_pressure:
            print("Merging completed with pressure data.")
            slab_2D = (slab_2D_0, slab_2D_1, slab_2D_2, slab_2D_3)
            data_table = pd.concat([data_table_0, data_table_1, data_table_2, data_table_3], ignore_index=True)
        else:
            print("Merging completed without pressure dependecy.")
            data_table = pd.concat([data_table_0, data_table_1, data_table_2], ignore_index=True)
            slab_2D = (slab_2D_0, slab_2D_1, slab_2D_2)

        
        
        
        print("Merging completed.")

    return data_table, slab_2D, surface_data_0


# ---
def plot_sensitivity_study_steady_state(args) -> None:
    """Small top-level function that calls the data structure generation routine
    and produce the figure of the manuscript.
    The following function plots the following figures:
    a. 2D plot regarding the average difference between the thermal model
    b. 1D summary plot regarding the top surface of the slab
    c. Heat flux of the numerical simulation and relative errors with respect to the non-linear model
    d. Maximum depth of the isotherm of all the experiment as a function of the thermal parameter and error measurment wrt the nonlinear model

    Args:
        args: parsed argument
    Returns:
        None:
    """
    check_parsed_arguments(args)

    aux_path = Path(__file__).resolve().parent

    aux_path_data_base = aux_path / "Output_h5_files"

    # Create the main data structure to produce the figure of the paper.

    if args.steady_state == 1:
        args.steady_state = "SS"
    elif args.steady_state == 0:
        args.steady_state = "TD"

    data_table, slab_2D, surface_data_0 = create_data_structure(
        aux_path=aux_path_data_base, args=args
    )
    # Generate path
    path2save = aux_path / f"Result_shear_heating_{args.sh}_{args.name_suite}_exp"
    # Create folder
    path2save.mkdir(exist_ok=True)
    # Plot mean difference
    plot_slab_variances(slab_2D, path2save=path2save, number_figure="a")
    # Create the database
    # Plot the 1D (P-T) of the slab surface
    figure_slab_surface(
        path2save=path2save,
        pdb=data_table,
        crd=surface_data_0,
        petro_db=data_base_PT_whitney(),
        figure_number="b",
    )
    # Plot the heat flux

    print(f"Saving picture into {path2save}...")
    # Plot the main
    plot_slab_versus_thermalpar(path2save=path2save, pdb=data_table, figure_number="c")


# ---
def plot_real_case_study_steady_state(args):
    """_summary_

    Args:
        args (_type_): _description_
    """

    check_parsed_arguments(args)

    aux_path = Path(__file__).resolve().parent

    aux_path_data_base = aux_path / "Output_h5_files"

    # Create the main data structure to produce the figure of the paper.
    path2save = aux_path / f"Result_shear_heating_{args.sh}_{args.name_suite}"

    data_table, _, _ = create_data_structure(aux_path=aux_path_data_base, args=args)
    print_save_IQ_information(data_table, path2save, "d_seismic_ell")
    print_save_IQ_information(data_table, path2save, "max_dpth150")
    print_save_IQ_information(data_table, path2save, "max_dpth450")
    print_save_IQ_information(data_table, path2save, "max_dpth600")
    # Generate path
    # Create folder
    path2save.mkdir(exist_ok=True)
    # Box plot for the seismogenetic zone
    box_plot_thermal_model(path2save=path2save, pdb=data_table, figure_number="d")
    # Box plot for the maximum depth of the isotherm within the slab
    box_plot_thermal_model_max_depth(path2save=path2save, pdb=data_table, figure_number="e")


# ---
def main():
    """Generation.

    Raises:
        ErrorParsing: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sh",
        type=str,
        help="Super Group: experiment with or without shear heating: sh = yes, shear heating; nsh = no, shear heating",
    )
    parser.add_argument(
        "--name_suite",
        type=str,
        help="Name of the test suite. This is a string that corresponds to the name of the test suite.",
    )
    parser.add_argument(
        "--steady_state",
        type=int,
        help="Flag to run the test in steady-state or time-dependent mode. 1 = steady-state, 0 = time-dependent",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="csv table used to produce the cases. Important to check what are the main variables that changes within the runs.",
    )
    parser.add_argument(
        "--number_worker",
        type=int,
        default=100,
        help="number of processor of the array job. The default is 100, as the run of the manuscript. User can change, but needs to be consistent with their work flow.",
    )
    parser.add_argument(
        "--function_plot",
        type=str,
        default="basic",
        help="choose the template for the plotting and analysis",
    )
    args = parser.parse_args()
    if args.function_plot == "basic":
        plot_sensitivity_study_steady_state(args)
    elif args.function_plot == "real":
        plot_real_case_study_steady_state(args)
    else:
        raise ErrorParsing("wrong function plot option", -1)


# ---
# ---
if __name__ == "__main__":
    main()
