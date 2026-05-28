from pathlib import Path
from typing import Annotated, Any, Literal

import distributed
import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr
from cycler import cycler
from dask.base import is_dask_collection
from dask.distributed import Client
from pypalettes import load_cmap
from rich.progress import track

from rojak.core.constants import TWENTIES_CLIMATOLOGICAL_PARAMETER
from rojak.core.data import shift_longitude
from rojak.core.indexing import concat_new_dim
from rojak.orchestrator.configuration import (
    RelationshipBetweenTypes,
    TurbulenceDiagnostics,
)
from rojak.orchestrator.lite_controller import DISTRIBUTION_PARAMS_TYPE_ADAPTER
from rojak.plot.utilities import _PLATE_CARREE, StandardColourMaps, get_a_default_cmap
from rojak.turbulence.analysis import RelationshipBetweenXAndTurbulence, TransformToEDR
from rojak.turbulence.metrics import (
    conditional_odds_ratio,
)
from rojak.utilities.types import DistributionParameters

IMAGE_EXT: str = "pdf"

# 0.03 approximate range of latitudinal mean
# 0.22 approx. standard deviation => use 0.11 for half a standard deviation from 0
VERTICAL_VELOCITY_THRESHOLDS: list[float] = [-0.22, -0.03, 0.03, 0.22]
EDR_THRESHOLD: float = 0.22
COOL_WARM_CMAP = get_a_default_cmap(
    StandardColourMaps.CORRELATION_COOL_WARM, load_kwargs={"cmap_type": "continuous"}
).resampled(18)
CMAP_SUNSET_TEN = load_cmap(
    "ag_Sunset", cmap_type="continuous", reverse=True
).resampled(10)

app = typer.Typer(
    help="Process atmospheric data and generate plots.",
    pretty_exceptions_show_locals=True,
)
cc = cycler(color=load_cmap("Color_Blind").colors)
plt.rc("axes", prop_cycle=cc)


def blocking_wait_futures(dask_collection: object) -> None:
    if is_dask_collection(dask_collection):
        _ = distributed.wait(distributed.futures_of(dask_collection))

    if isinstance(dask_collection, list):
        for item in dask_collection:
            del item

    del dask_collection


def load_distribution_params(file_path: str) -> dict[str, DistributionParameters]:
    dist_params_file = Path(file_path)
    return DISTRIBUTION_PARAMS_TYPE_ADAPTER.validate_json(dist_params_file.read_text())


def load_single_level_data(target_path: Path, target_var: str) -> xr.DataArray:
    loaded_ds: xr.Dataset = xr.open_mfdataset(
        str(target_path / "*.nc"),
        chunks={"pressure_level": 6, "latitude": 721, "longitude": 1440},
        parallel=True,
        engine="h5netcdf",
        decode_coords=True,
        decode_cf=True,
        decode_timedelta=True,
    ).rename({"valid_time": "time"})
    return shift_longitude(loaded_ds[target_var]).persist()


def sel_for_season[T: (xr.Dataset, xr.DataArray)](
    target_data: T,
    season: Literal["DJF", "MAM", "JJA", "SON"],
    time_dim_name: str = "time",
) -> T:
    return target_data.isel(
        indexers={time_dim_name: (target_data[time_dim_name].dt.season == season)}
    )


def map_conditional_odds_on_turb(
    is_turb_da: xr.DataArray,
    effect_of: xr.DataArray,
    *control_var: xr.DataArray,
    sum_over: list[str] | None = None,
    use_log: bool = True,
    new_dim_name: str = "concat_dim",
    dim_values: list[Any] | None = None,
) -> xr.DataArray:
    if dim_values is None:
        target_length = (
            2 if (control_var_len := len(control_var)) == 1 else control_var_len
        )
        dim_values = np.arange(target_length).tolist()

    return concat_new_dim(
        conditional_odds_ratio(
            effect_of, is_turb_da, *control_var, sum_over=sum_over, use_log=use_log
        ),
        dim_name=new_dim_name,
        dim_values=dim_values,
    )


def plot_dataset(
    target_dataset: xr.Dataset,
    cbar_label: str,
    plot_dir: Path,
    name_prefix: str,
    **kwargs: Any,
) -> None:
    for key, value in track(
        target_dataset.data_vars.items(), description=f"Plotting {name_prefix}"
    ):
        fg = value.plot(
            cbar_kwargs={
                "orientation": "horizontal",
                "spacing": "uniform",
                "pad": 0.02,
                "shrink": 0.6,
                "label": cbar_label.format(key),
            },
            transform=_PLATE_CARREE,
            subplot_kws={"projection": _PLATE_CARREE},
            rasterized=True,
            **kwargs,
        )
        fg.map(lambda: plt.gca().coastlines())
        fg.fig.savefig(
            plot_dir / f"{name_prefix}_{key}.{IMAGE_EXT}", dpi=400, bbox_inches="tight"
        )
        plt.close(fg.fig)


def plot_diff_marginal_conditional(
    conditional_odds_of_cases: xr.Dataset,
    marginal_case: xr.Dataset,
    name_prefix: str,
    plot_dir: Path,
    titles: list[str],
) -> None:
    marginal_sign = np.sign(marginal_case)
    difference = marginal_sign * (conditional_odds_of_cases - marginal_case)
    for key, value in track(
        difference.data_vars.items(), description=f"Plotting {name_prefix}"
    ):
        fg = value.plot(
            row="conditions",
            x="latitude",
            cbar_kwargs={
                "orientation": "horizontal",
                "spacing": "uniform",
                "shrink": 0.6,
                "label": "$\\log(\\frac{\\theta_{XY (k)}}{\\theta})$",
            },
            robust=True,
            center=0,
            rasterized=True,
            vmax=1,
            vmin=-1,
            cmap=COOL_WARM_CMAP,
            yincrease=False,
            xincrease=False,
            aspect=1.4,
            size=2.2,
        )

        for index, this_ax in enumerate(fg.axs.flat):
            this_ax.set_title(titles[index])

        plt.savefig(
            plot_dir / f"{name_prefix}_{key}.{IMAGE_EXT}", dpi=400, bbox_inches="tight"
        )
        plt.close()


def plot_means(
    target_dataset: xr.Dataset,
    cbar_label: str,
    plot_dir: Path,
    name_prefix: str,
    titles: list[str] | None = None,
    **kwargs: Any,
) -> None:
    for key, value in track(
        target_dataset.data_vars.items(), description=f"Plotting {name_prefix}"
    ):
        fg = value.plot(
            rasterized=True,
            **kwargs,
        )
        if titles is not None and "hue" not in kwargs:
            for ax, title in zip(fg.axes.flat, titles, strict=True):
                ax.set_title(title)

        if "hue" in kwargs:
            plt.gca().set_xlabel(cbar_label)
            plt.gca().legend(titles)
            plt.gca().grid()

        plt.savefig(
            plot_dir / f"{name_prefix}_{key}.{IMAGE_EXT}", dpi=400, bbox_inches="tight"
        )
        plt.close()


def compute_vertical_velocity_masks(
    vertical_velocities: xr.DataArray, pressure_levels: xr.DataArray
) -> list[xr.DataArray]:
    all_thresholds = VERTICAL_VELOCITY_THRESHOLDS
    masks = []
    for i in range(len(all_thresholds) + 1):
        if i == 0:
            # Velocities < minimum threshold
            condition = vertical_velocities < all_thresholds[0]
        elif i == len(all_thresholds):
            # Velocities >= maximum threshold
            condition = vertical_velocities >= all_thresholds[-1]
        else:
            # Velocities between two thresholds
            condition = (vertical_velocities >= all_thresholds[i - 1]) & (
                vertical_velocities < all_thresholds[i]
            )

        masks.append(
            condition.isel(pressure_level=0)
            .expand_dims(dim={"pressure_level": pressure_levels})
            .assign_coords({"pressure_level": pressure_levels})
        )
    return masks


def case_labels_vertical_velocity_thresholds(as_latex: bool = False) -> list[str]:
    all_thresholds = VERTICAL_VELOCITY_THRESHOLDS
    labels = []
    leq = " \\leq " if as_latex else "<="
    geq = " \\geq " if as_latex else ">="

    for i in range(len(all_thresholds) + 1):
        if i == 0:
            # First bin: w < minimum
            labels.append(f"w<{all_thresholds[0]}")
        elif i == len(all_thresholds):
            # Last bin: w >= maximum
            labels.append(f"w{geq}{all_thresholds[-1]}")
        else:
            # Middle bins: threshold[i-1] <= w < threshold[i]
            lower = all_thresholds[i - 1]
            upper = all_thresholds[i]
            labels.append(f"{lower}{leq}w<{upper}")

    if as_latex:
        labels = [f"${item}$" for item in labels]

    return labels


def conditional_odds_ratio_cases(
    is_turb: xr.Dataset,
    iss_regions: xr.DataArray,
    vertical_velocity_mask: list[xr.DataArray],
    plots_dir: Path,
    target_season: Literal["DJF", "MAM", "JJA", "SON"] | None,
    sum_over_dim: None | str | list[str],
) -> None:
    target_args = [iss_regions, *vertical_velocity_mask]
    dim_names = case_labels_vertical_velocity_thresholds()

    conditional_odds_of_cases = is_turb.map(
        map_conditional_odds_on_turb,
        keep_attrs=False,
        args=target_args,
        sum_over=sum_over_dim,
        use_log=True,
        new_dim_name="conditions",
        dim_values=dim_names,
    )
    marginal_case = RelationshipBetweenXAndTurbulence(
        iss_regions,
        is_turb,
        RelationshipBetweenTypes.SAMPLE_ODDS_RATIO,
        sum_over_dim=sum_over_dim,
    ).execute()
    marginal_case_w_coords = marginal_case.assign_coords(conditions="marginal")
    all_odds_ratios = xr.concat(
        [conditional_odds_of_cases, marginal_case_w_coords], dim="conditions"
    )

    titles = [
        *case_labels_vertical_velocity_thresholds(as_latex=True),
        "marginal",
    ]

    if sum_over_dim is None:
        overall_mean = all_odds_ratios.compute().to_pandas()
        overall_mean.to_csv(
            plots_dir
            / f"overall_conditional_log_odds_ratio_{target_season if not None else 'all'}.csv",
            mode="w",
        )
    elif len(sum_over_dim) == 1 or isinstance(sum_over_dim, str):
        all_odds_ratios = all_odds_ratios.compute()
        conditions_coordinate = all_odds_ratios["conditions"].values
        for index in range(conditions_coordinate.size):
            plot_dataset(
                all_odds_ratios.isel(conditions=index).reset_coords(
                    "conditions", drop=True
                ),
                "$\\log(\\theta)$ {}",
                plots_dir,
                f"cond_log_odds_ratio_{conditions_coordinate[index]}_{target_season}",
                cmap=COOL_WARM_CMAP,
                robust=True,
                center=0,
                vmax=3,
                vmin=-3,
                col="pressure_level",
            )
            plot_dataset(
                all_odds_ratios.isel(conditions=index).reset_coords(
                    "conditions", drop=True
                ),
                "$\\log(\\theta)$ {}",
                plots_dir,
                f"cond_log_odds_ratio_vert_{conditions_coordinate[index]}_{target_season}",
                cmap=COOL_WARM_CMAP,
                robust=True,
                center=0,
                vmax=3,
                vmin=-3,
                row="pressure_level",
            )
    elif len(sum_over_dim) == 2:
        plot_means(
            all_odds_ratios,
            "$\\log(\\theta)$",
            plots_dir,
            f"cond_log_odds_ratio_lat_mean_{target_season}",
            titles=titles,
            row="conditions",
            x="latitude",
            cbar_kwargs={
                "orientation": "horizontal",
                "spacing": "uniform",
                "shrink": 0.6,
                "label": "$\\log(\\theta)$",
            },
            robust=True,
            center=0,
            vmax=3,
            vmin=-3,
            cmap=COOL_WARM_CMAP,
            col_wrap=3,
            yincrease=False,
            aspect=1.4,
            size=2.2,
            xincrease=False,
        )
        plot_diff_marginal_conditional(
            conditional_odds_of_cases,
            marginal_case,
            f"cond_log_odds_ratio_rel_marginal_lat_mean_{target_season}",
            plots_dir,
            titles,
        )
    elif len(sum_over_dim) == 3:
        plot_means(
            all_odds_ratios,
            "$\\log(\\theta)$",
            plots_dir,
            f"cond_log_odds_ratio_pl_mean_{target_season}",
            titles=titles,
            y="pressure_level",
            yincrease=False,
            hue="conditions",
        )
    else:
        raise ValueError("invalid length of sum_over_dim")

    blocking_wait_futures([conditional_odds_of_cases, marginal_case, all_odds_ratios])


def seasonal_variation(
    is_turb: xr.Dataset,
    iss_regions: xr.DataArray,
    vertical_velocities_masks: list[xr.DataArray],
    plots_dir: Path,
) -> None:
    if len(is_turb.data_vars.keys()) < 2:
        return

    seasonal_months: list[Literal["JJA", "DJF"]] = ["JJA", "DJF"]
    lat_mean_dims = ["time", "longitude"]
    dim_names = case_labels_vertical_velocity_thresholds()

    marginal_case = RelationshipBetweenXAndTurbulence(
        iss_regions,
        is_turb,
        RelationshipBetweenTypes.SAMPLE_ODDS_RATIO,
        sum_over_dim=lat_mean_dims,
    ).execute()
    marginal_sign = np.sign(marginal_case)

    for target_season in seasonal_months:
        vertical = [
            sel_for_season(this_mask, target_season)
            for this_mask in vertical_velocities_masks
        ]
        seasonal_issr = sel_for_season(iss_regions, target_season)
        seasonal_turb = sel_for_season(is_turb, target_season)
        season_marginal_case = (
            RelationshipBetweenXAndTurbulence(
                seasonal_issr,
                seasonal_turb,
                RelationshipBetweenTypes.SAMPLE_ODDS_RATIO,
                sum_over_dim=lat_mean_dims,
            )
            .execute()
            .assign_coords({"conditions": "marginal"})
        )
        conditional_odds_of_cases = seasonal_turb.map(
            map_conditional_odds_on_turb,
            keep_attrs=False,
            args=[seasonal_issr, *vertical],
            sum_over=lat_mean_dims,
            use_log=True,
            new_dim_name="conditions",
            dim_values=dim_names,
        )
        all_odds_ratios = xr.concat(
            [conditional_odds_of_cases, season_marginal_case], dim="conditions"
        )

        difference = marginal_sign * (all_odds_ratios - marginal_case)
        _fe = difference.to_dataarray("diagnostics").plot(
            row="conditions",
            col="diagnostics",
            x="latitude",
            cbar_kwargs={
                "orientation": "horizontal",
                "spacing": "uniform",
                "shrink": 0.6,
                "label": "$\\vartheta$",
            },
            robust=True,
            center=0,
            rasterized=True,
            vmax=1,
            vmin=-1,
            cmap=COOL_WARM_CMAP,
            col_wrap=3,
            yincrease=False,
            aspect=1.4,
            size=2.2,
            xincrease=False,
        )

        plt.savefig(
            plots_dir / f"diff_from_marginal_{target_season}.{IMAGE_EXT}",
            dpi=400,
            bbox_inches="tight",
        )
        plt.close()


def create_plots(
    is_turb: xr.Dataset,
    iss_regions: xr.DataArray,
    vertical_velocities_masks: list[xr.DataArray],
    plots_dir: Path,
    target_season: Literal["DJF", "MAM", "JJA", "SON"] | None,
) -> None:
    if target_season is not None:
        vertical_velocities_masks = [
            sel_for_season(this_mask, target_season)
            for this_mask in vertical_velocities_masks
        ]
        is_turb = sel_for_season(is_turb, target_season)

    # sum_over_cases = [None, ["time", "longitude"], ["time", "longitude", "latitude"]]
    sum_over_cases = [
        None,
        ["time"],
        ["time", "longitude"],
        ["time", "longitude", "latitude"],
    ]

    for this_sum_over_case in sum_over_cases:
        conditional_odds_ratio_cases(
            is_turb,
            iss_regions,
            vertical_velocities_masks,
            plots_dir,
            target_season,
            this_sum_over_case,
        )


def turbulence_probability(
    is_turb: xr.Dataset,
    plots_dir: Path,
    target_season: Literal["DJF", "MAM", "JJA", "SON"] | None,
) -> None:
    if target_season is not None:
        is_turb = sel_for_season(is_turb, target_season)

    plot_dataset(
        is_turb.mean(dim="time"),
        "Probability MOG {}",
        plots_dir,
        f"turb_probability_{target_season}",
        cmap=CMAP_SUNSET_TEN,
        robust=True,
        vmax=0.05,
        vmin=0,
        col="pressure_level",
    )


def matthews_correlation(
    is_turb: xr.Dataset,
    iss_regions: xr.DataArray,
    plots_dir: Path,
    target_season: Literal["DJF", "MAM", "JJA", "SON"] | None,
) -> None:
    if target_season is not None:
        iss_regions = sel_for_season(iss_regions, target_season)
        is_turb = sel_for_season(is_turb, target_season)

    matthews_corr = RelationshipBetweenXAndTurbulence(
        iss_regions,
        is_turb,
        RelationshipBetweenTypes.MATTHEWS_CORRELATION,
        sum_over_dim="time",
    ).execute()
    plot_dataset(
        matthews_corr,
        "$\\varphi \\, {}$",
        plots_dir,
        f"matthews_corr_{target_season}",
        cmap=COOL_WARM_CMAP,
        robust=True,
        center=0,
        vmax=0.15,
        vmin=-0.15,
        col="pressure_level",
    )
    plot_dataset(
        matthews_corr,
        "$\\varphi \\, {}$",
        plots_dir,
        f"matthews_corr_vert_{target_season}",
        cmap=COOL_WARM_CMAP,
        robust=True,
        center=0,
        vmax=0.15,
        vmin=-0.15,
        row="pressure_level",
    )
    plot_means(
        matthews_corr.mean(dim="longitude"),
        "$\\varphi",
        plots_dir,
        f"matthews_corr_lat_mean_{target_season}",
        x="latitude",
        y="pressure_level",
        cbar_kwargs={
            "orientation": "horizontal",
            "spacing": "uniform",
            "shrink": 0.6,
            "label": "$\\varphi",
        },
        robust=True,
        center=0,
        vmax=3,
        vmin=-3,
        cmap=COOL_WARM_CMAP,
        yincrease=False,
        aspect=1.4,
        size=2.2,
    )

    del matthews_corr


def main(
    diagnostics: xr.Dataset,
    iss_regions: xr.DataArray,
    vertical_velocity: xr.DataArray,
    plots_dir: Path,
) -> None:
    diagnostics = diagnostics.sel(pressure_level=slice(300, 200))

    pressure_levels = diagnostics["pressure_level"]

    vertical_velocity_masks: list[xr.DataArray] = compute_vertical_velocity_masks(
        vertical_velocity, pressure_levels
    )

    is_turb = diagnostics > EDR_THRESHOLD
    is_turb = is_turb.reset_coords(names=["altitude", "expver"], drop=True)

    seasonal_variation(is_turb, iss_regions, vertical_velocity_masks, plots_dir)

    seasons: list[Literal["DJF", "MAM", "JJA", "SON"] | None] = [None, "DJF", "JJA"]

    correlation_dir = plots_dir / "correlation"
    correlation_dir.mkdir(parents=True, exist_ok=True)
    probability_dir = plots_dir / "probability"
    probability_dir.mkdir(parents=True, exist_ok=True)

    for this_season in seasons:
        turbulence_probability(is_turb, probability_dir, this_season)
        create_plots(
            is_turb, iss_regions, vertical_velocity_masks, plots_dir, this_season
        )
        matthews_correlation(is_turb, iss_regions, correlation_dir, this_season)


@app.command()
def start(
    pressure_level_dir: Annotated[
        Path,
        typer.Argument(help="Path to pressure level data folder"),
    ],
    issr_dir: Annotated[
        Path,
        typer.Argument(help="Path to pressure level data folder"),
    ],
    vertical_velocity_dir: Annotated[
        Path,
        typer.Argument(help="Path to vertical velocity data folder"),
    ],
    plots_dir: Annotated[
        Path,
        typer.Argument(help="Path to output plots directory"),
    ],
    distribution_params_file: Annotated[
        Path,
        typer.Argument(help="Path to distribution parameters file"),
    ],
    index: Annotated[
        int,
        typer.Argument(help="Index parameter for processing"),
    ],
) -> None:
    """
    Process atmospheric data and generate visualisation plots.

    This application takes pressure level, vertical velocity, and surface
    pressure data, applies distribution parameters, and generates output plots.
    """
    # Validate input paths exist
    if not pressure_level_dir.exists() or not pressure_level_dir.is_dir():
        typer.echo(
            f"Error: Pressure level directory does not exist: {pressure_level_dir}",
            err=True,
        )
        raise typer.Exit(code=1)

    if not vertical_velocity_dir.exists() or not vertical_velocity_dir.is_dir():
        typer.echo(
            f"Error: Vertical velocity directory does not exist: {vertical_velocity_dir}",
            err=True,
        )
        raise typer.Exit(code=1)

    if not distribution_params_file.exists() or not distribution_params_file.is_file():
        typer.echo(
            f"Error: Distribution parameters file does not exist: {distribution_params_file}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Create output directory if it doesn't exist
    plots_dir.mkdir(parents=True, exist_ok=True)

    pl_path = Path(pressure_level_dir)
    w_path = Path(vertical_velocity_dir)

    client: Client = Client()

    distribution_parameters = load_distribution_params(str(distribution_params_file))

    if index == -1:
        target_diagnostics = [
            TurbulenceDiagnostics.TI1,
            TurbulenceDiagnostics.NCSU1,
            TurbulenceDiagnostics.BROWN2,
        ]
        diagnostics = xr.Dataset(
            data_vars={
                name: TransformToEDR(
                    xr.open_zarr(
                        pl_path / f"{name}.zarr",
                        consolidated=False,
                    )[name],
                    c1=TWENTIES_CLIMATOLOGICAL_PARAMETER.c1,
                    c2=TWENTIES_CLIMATOLOGICAL_PARAMETER.c2,
                    mean=distribution_parameters[str(name)].mean,
                    variance=distribution_parameters[str(name)].variance,
                ).execute()
                for name in target_diagnostics
            },
        )
    else:
        index = index - 1  # pbs array values are base 1 instead of base 0

        target_diagnostics = [
            # TurbulenceDiagnostics.F2D,
            # TurbulenceDiagnostics.F3D,
            # TurbulenceDiagnostics.UBF,
            TurbulenceDiagnostics.TI1,
            # TurbulenceDiagnostics.TI2,
            TurbulenceDiagnostics.NCSU1,
            # TurbulenceDiagnostics.ENDLICH,
            # TurbulenceDiagnostics.COLSON_PANOFSKY,
            # TurbulenceDiagnostics.RICHARDSON,
            # TurbulenceDiagnostics.WIND_SPEED,
            # TurbulenceDiagnostics.BRUNT_VAISALA,
            # TurbulenceDiagnostics.VWS,
            # TurbulenceDiagnostics.DEF,
            # TurbulenceDiagnostics.TEMPERATURE_GRADIENT,
            # TurbulenceDiagnostics.NGM1,
            # TurbulenceDiagnostics.NGM2,
            # TurbulenceDiagnostics.BROWN1,
            TurbulenceDiagnostics.BROWN2,
            # TurbulenceDiagnostics.EDR_LUNNON,
            # TurbulenceDiagnostics.DUTTON,
        ]
        loaded_diagnostics = xr.open_zarr(
            pl_path / f"{target_diagnostics[index]}.zarr",
            consolidated=False,
        )
        diagnostics = xr.Dataset(
            data_vars={
                name: TransformToEDR(
                    dvar,
                    c1=TWENTIES_CLIMATOLOGICAL_PARAMETER.c1,
                    c2=TWENTIES_CLIMATOLOGICAL_PARAMETER.c2,
                    mean=distribution_parameters[str(name)].mean,
                    variance=distribution_parameters[str(name)].variance,
                ).execute()
                for name, dvar in loaded_diagnostics.data_vars.items()
            },
        )

    diagnostics = diagnostics.chunk(
        {"longitude": -1, "latitude": -1, "pressure_level": -1}
    )

    plots_dir = plots_dir / str(target_diagnostics[index])
    plots_dir.mkdir(parents=True, exist_ok=True)

    iss_regions = xr.open_zarr(
        issr_dir / "issr.zarr",
        consolidated=False,
        zarr_format=2,
    )["issr"]
    iss_regions = iss_regions.chunk(
        {"longitude": -1, "latitude": -1, "pressure_level": -1}
    )

    vertical_velocity = load_single_level_data(w_path, "w")

    main(diagnostics, iss_regions, vertical_velocity, plots_dir)

    _ = client.close()


if __name__ == "__main__":
    app()
