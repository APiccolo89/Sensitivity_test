"""Produce the manuscript's "eye-candy" figure: P-T conditions along the
slab surface and heat fluxes for the Japan/Mexico real cases and the
steady-state sensitivity suite, overlaid on the Whitney P-T database.
"""

from pathlib import Path

from post_process_module.figure_plot import data_base_PT_whitney, plot_eye_candy
from post_process_module.process_data_visualisation import wrapper_merging_h5


# ---
def plot_slab_surfaces_japan_mexico_sensitivity():
    """Merge the Japan, Mexico and steady-state-sensitivity (nl, shear
    heating) h5 databases and produce the manuscript's eye-candy figure
    from them.
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
        csv="cases_Sensitivity_steady_s_sensitivity.csv",
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
    """Entry point: produce the eye-candy figure."""
    plot_slab_surfaces_japan_mexico_sensitivity()


# ---
if __name__ == "__main__":
    main()
