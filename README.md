# Finding hidden patterns in high resolution wind flow model simulations

## Overview

This folder contains:
- The dataset for the challenge
- The description of the data
- Python requirements for creating a virtualenv to load the data
- A Quickstart notebook in Python

## Folder structure

    ├── README.md                               <- This file
    │
    ├── data                                    <- Folder containing the full dataset for the challenge
    │   ├── perdigao_era5_2020.nc               <- ERA5 hourly timeseries at single location
    │   ├── perdigao_high_res_1H_2020.nc        <- LES hourly grid timeseries at 80m x 80m resolution (8GB)
    │   └── perdigao_low_res_1H_2020.nc         <- LES hourly grid timeseries at 160m x 160m resolution (2GB)
    |
    ├── data_samples                            <- Folder containing LES data of the first month of the full dataset (2020-01)
    │   ├── perdigao_high_res_1H_2020_01.nc     <- Sample LES hourly grid timeseries at 80m x 80m resolution
    │   └── perdigao_low_res_1H_2020_01.nc      <- Sample LES hourly grid timeseries at 160m x 160m resolution
    │
    ├── requirements.txt                        <- Recommended Python libraries for the virtual environment to load the data
    |
    └── quickstart.ipynb                        <- Quickstart notebook

## Data

The dataset is composed of three data sources: ERA5 data at single location, a high and a low resolution LES simulation output.

### ERA5 data

The data from ERA5 has been downloaded from [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/cdsapp#!/home). 
`data/perdigao_era5_2020.nc` corresponds to ERA5 hourly data single levels for the year 2020 
([check the documentation here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)). 
The data has been extracted at single point (-7.737°E, 39.7°N) since ERA5 spatial resolution is about 30km.

The file format is NetCDF and can be easily opened with `xarray` (see quickstart notebook).

The data represents hourly timeseries of following quantities (variables are also described in the NetCDF file and Copernicus documentation):
- `u100`: 100 meter above ground level U wind component in m/s.
- `v100`: 100 meter above ground level V wind component in m/s.
- `t2m`: 2 meter above ground level temperature in K.
- `i10fg`: 10 meter above ground level instantaneous wind gust.

### LES data

The full data from LES is available at two different spatial resolutions:
- `data/perdigao_high_res_1H_2020.nc`: 80m x 80m at 1H frequency
- `data/perdigao_low_res_1H_2020.nc`: 160m x 160m at 1H frequency

Some samples of the full data is available (NetCDF containing the first month of the full dataset):
- `perdigao_high_res_1H_2020_01.nc`: 80m x 80m at 1H frequency
- `perdigao_low_res_1H_2020_01.nc`: 160m x 160m at 1H frequency

Both dataset are available at 100m height above ground level i.e. terrain following slices.

Note that the following description of the dataset is also available from the NetCDF files.

Coordinates:
- `height`: Height in meter above ground level (only 100m). This is the height of the terrain following slice for all variables.
- `time`: Timestamps at 1H frequency.
- `xf`: Horizontal cartesian coordinate in meter of the simulated domain (West to East).
- `yf`: Vertical cartesian coordinate in meter of the simulated domain (South to North). 

Variables:
- `absolute_height`:  Height above sea level in meter, note that this variable only depends on (xf, yf) not on time.
- `std`: 1H average of standard deviation of horizontal wind speed in m/s originally recorded at 10min frequency.
- `temp`: 1H average of temperature in Kelvin.
- `u`: 1H average of U component of wind speed (along `xf`) in m/s.
- `v`: 1H average of V component of wind speed (along `yf`) in m/s.
- `vel`: 1H average of horizontal wind speed in m/s.

**Note:**

- Since averages are calculated during the LES simulation at higher time frequency: 
$$\overline{vel} \ge \sqrt{\overline{u}^2 + \overline{v}^2}$$
- Some missing timestamps and NaNs values might be present in the data.

## Data format

### xarray

File formats:
- [NetCDF](https://en.wikipedia.org/wiki/NetCDF)

NetCDF files can be opened with `xarray` python library which is a N-dimensional generalization of `pandas`.

Besides the quickstart notebook provided with the data, here are some useful links of the documentation to get familiar with `xarray`:
- [Quickstart](https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html)
- [IO operations](https://xarray.pydata.org/en/stable/user-guide/io.html)
- [Plotting functionalities](https://docs.xarray.dev/en/stable/user-guide/plotting.html)

### Final considerations

Have a look to the quickstart notebook and good luck !
