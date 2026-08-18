# Towards safer contrail avoidance: understanding the link between ice supersaturated regions and clear-air turbulence

This repo is for reproducing the results in this paper

[`rojak`](https://github.com/ImperialCollegeLondon/rojak) version of the code for this is [v1.0.1](https://github.com/ImperialCollegeLondon/rojak/releases/tag/v1.0.1)

## Obtaining ERA5 data

This uses ERA5 data. For convenience, `rojak` contains methods to get this data. For example, the command below gets the required data for the year 2023, 

```console
rojak data meteorology retrieve -s era5 -y 2023 -m -1 -d -1 -n pressure-level --default-name cat -o /path/to/store/data/in
```

## Steps To Reproduce on HPC with PBS Queue

> [!note]
> The folder structure in the repository does not directly map to the folder structure these script were run in.
> Path modifications are required to match your system to reproduce the results

### Step 1: Export Diagnostics to Zarr

> [!important]
> [`uv`](https://docs.astral.sh/uv/) has been used to run this [script so that it can directly manage the dependencies for it](https://docs.astral.sh/uv/guides/scripts/)

1. Use `jinja` to template the configuration files for each of the turbulence diagnostics. This is to run the exporting in a PBS job array, i.e. each PBS job exports one of the turbulence diagnostics. In the [`configs/templated-configs`](configs/templated-configs) folder, run

   ```console
   uv run template-files.py
   ```

2. This uses `rojak`'s `lite` interface to export the turbulence diagnostics to `zarr` file format. Submit a job to the PBS queue using the [export-diagnostics-to-zarr.pbs](pbs-scripts/compute-distribution-params.pbs) script,

   ```console
   qsub -v "SOURCE_DIR=/path/to/rojak/source/code, CONFIG_FILE_PATH=path/to/templated/configs, LITE_COMMAND=export-diagnostic" -N "export-diagnostics" export-diagnostics-to-zarr.pbs
   ```

### Step 2: Export Distribution Parameters

This step is to speed up the conversion of the turbulence diagnostics to EDR by precomputing what the mean and standard deviation of each of the turbulence diagnostics. It uses `rojak`'s `lite` interface to perform the computation.

To run, submit a job to the PBS queue using the [compute-distribution-params.pbs](pbs-scripts/compute-distribution-params.pbs) script,

```console
qsub -v "SOURCE_DIR=/path/to/rojak/source/code, CONFIG_FILE_PATH=/path/to/dist-params-config.yaml, LITE_COMMAND=distribution-parameters, LOAD_FROM=precomputed_from_zarr" -N "name-of-run" compute-distribution-params.pbs
```

### Step 3: Export Ice Supersaturated Regions

Similarly, this step is to speed up the computing of the odds ratio by precomputing the ISSRs. This uses the [export-issr.py](rojak-scripts/export-issr.py) script.

To run, submit a PBS job using the [export-issr.pbs](pbs-scripts/export-issr.pbs)

```console
qsub -v "IDENTIFIER=export-issr" -N export-issr export-issr.pbs
```

### Step 4: Compute the odds ratio for the effect of ISSRs on MOG CAT

This step uses the exported ISSRs, distribution parameters for mapping the diagnostics to EDR and the exported turbulence diagnostics stored in the `zarr` file format.

To run, submit a PBS job using the [compute-odds-ratio.pbs](pbs-scripts/compute-odds-ratio.pbs) script,

```console
qsub -v "IDENTIFIER=odds-ratio" -N odds-ratio compute-odds-ratio.pbs
```
