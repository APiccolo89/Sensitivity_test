from pathlib import Path

import pandas as pd

from post_process_module.figure_plot import data_base_PT_whitney, plot_eye_candy
from post_process_module.post_process_data import timing_function
from post_process_module.process_data_visualisation import wrapper_merging_h5


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

    slab_2D = (slab_2D_0, slab_2D_1, slab_2D_2)

    data_table = pd.concat([data_table_0, data_table_1, data_table_2], ignore_index=True)

    return data_table, slab_2D, surface_data_0


# ---
def plot_slab_surfaces_japan_mexico_sensitivity():
    """_summary_

    Args:
        args (_type_): _description_
    """

    aux_path = Path(__file__).resolve().parent

    # Create the main data structure to produce the figure of the paper.
    path2save = aux_path / "Slab_surface_plot"
    path2save.mkdir(parents=True, exist_ok=True)

    data_table_0, surface_data_0, _ = wrapper_merging_h5(
        aux_path,
        partial=("nlsh", "SS"),
        name_suite="steady_s_sensitivity",
        recreate=False,
        csv="cases_Sensitivity.csv",
    )

    data_table_1, surface_data_1, _ = wrapper_merging_h5(
        aux_path,
        partial=("nlsh", "SS"),
        name_suite="Mexico",
        recreate=False,
        csv="cases_Sensitivity_Mexico.csv",
    )

    data_table_2, surface_data_2, _ = wrapper_merging_h5(
        aux_path,
        partial=("nlsh", "SS"),
        name_suite="Japan",
        recreate=False,
        csv="cases_Sensitivity_Japan.csv",
    )

    plot_eye_candy(
        data_table_0,
        data_table_1,
        data_table_2,
        data_base_PT_whitney(),
        surface_data_0,
        surface_data_1,
        surface_data_2,
        path2save,
    )


# ---
def main():
    plot_slab_surfaces_japan_mexico_sensitivity()


# ---
if __name__ == "__main__":
    main()
