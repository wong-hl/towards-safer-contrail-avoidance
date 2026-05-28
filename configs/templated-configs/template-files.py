# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2>=3.1.6",
# ]
# ///
from jinja2 import Environment, FileSystemLoader
import itertools
from pathlib import Path

OUTPUT_DIR: Path = Path("generated")
diagnostic_names: list[str] = [
    "richardson",
    "negative_richardson",
    "f2d",
    "f3d",
    "ubf",
    "ti1",
    "ti2",
    "ncsu1",
    "endlich",
    "colson_panofsky",
    "wind_speed",
    "bunt_vaisala",
    "vertical_wind_shear",
    "deformation",
    "directional_shear",
    "temperature_gradient",
    "horizontal_divergence",
    "ngm1",
    "ngm2",
    "brown1",
    "brown2",
    "gradient_pv",
    "nva",
    "dutton",
    "edr_lunnon",
    "vorticity_squared",
]
data_for_case = ["name-of-folder-for-case"]

if __name__ == "__main__":
    env = Environment(
        loader=FileSystemLoader("."), trim_blocks=True, lstrip_blocks=True
    )
    config_template = env.get_template("lite-export-diagnostics-config.yaml.jinja")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, (diagnostic_name, target_case) in enumerate(
        itertools.product(diagnostic_names, data_for_case)
    ):
        print(
            config_template.render(target_diagnostic=diagnostic_name, case=target_case),
            file=Path(OUTPUT_DIR / f"lite-export-config-{index + 1}.yaml").open(
                mode="w"
            ),
        )
