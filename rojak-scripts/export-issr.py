from pathlib import Path
from typing import TYPE_CHECKING

from dask.distributed import Client

from rojak.core.data import load_from_folder
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import SpatialDomain

if TYPE_CHECKING:
    from rojak.core.data import CATData


def check_path(folder_path: str) -> Path:
    as_path = Path(folder_path)
    assert as_path.exists()
    assert as_path.is_dir()
    return as_path


def instantiate_era5_data(
    folder_path: Path, spatial_domain: SpatialDomain
) -> "CATData":
    target_chunks = {"pressure_level": 6, "latitude": 721, "longitude": 1440}
    return Era5Data(
        load_from_folder(folder_path, chunks=target_chunks, engine="h5netcdf")
    ).to_clear_air_turbulence_data(spatial_domain)


def main() -> None:
    export_dir = Path("/path/to/exported/issrs")
    export_dir.mkdir(parents=True, exist_ok=True)

    spatial_domain = SpatialDomain(
        minimum_latitude=-70,
        maximum_latitude=70,
        minimum_longitude=-180,
        maximum_longitude=180,
        grid_size=0.25,
    )
    folder_path = check_path("/path/to/raw/era5/data")
    data = instantiate_era5_data(folder_path, spatial_domain)
    data.ice_supersaturated_regions().to_zarr(
        export_dir / "issr.zarr", mode="w", zarr_format=2
    )


if __name__ == "__main__":
    client: Client = Client()

    main()

    client.close()
