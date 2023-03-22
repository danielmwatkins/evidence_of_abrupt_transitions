# Introduction
This repository contains the code and data used for the paper "Evidence of abrupt transitions between sea ice dynamical regimes in the East Greenland marginal ice zone" by Watkins et al. This initial version contains all the code necessary to reproduce the results in the paper, *provided all the external datasets have been downloaded*. So that it is possible to review the analysis of the paper, I have separated the python scripts into code needed to prepare the data for analysis, and code needed to create the figures. The figure generation code can be run using only the data files provided in the repository. In a future version of the repository, more detailed instructions for obtaining the external datasets will be provided.

# Setup
The files environment.yml and environment-tidal.yml provide the list of package versions used to run the script. After installing miniconda (instructions: https://docs.conda.io/en/latest/miniconda.html) create the miz_dynamics environment via

`conda env create --file miz_dynamics.yml`

External datasets needed:
- Buoy drift tracks from the Arctic Data Center
- IBCAO_v4_2_400m.nc
- AMSR2 25 km sea ice concentration
- ERA5 hourly u10, v10 for the region bounded by 60-90 latitude and -30 to 30 longitude

# Data preparation
1. `clean_buoy_data.py` Reads the MOSAiC buoy tracks downloaded from the Arctic Data Center, applies quality control, and resamples to hourly resolution. Requires drift tracks and metadata from the Arctic Data Center. Results saved to `data/interpolated_tracks'.
2. `clean_ft_data.py` Reads the CSV files with IFT floe positions, interpolates to daily resolution, interpolates ERA5 daily wind speeds to floe positions, and computes turning angle and drift speed ratios relative to the wind speeds. Results saved to `data/floe_tracker/interpolated` and `data/floe_tracker/ft_with_wind.csv`.
3. `compile_amsr2_data.py` Reads daily hd5 files from NSIDC and merges them to a netcdf file for the merge_data code to access.
4. `merge_data.py` Interpolates sea ice concentration, depth, and ERA5 winds to the buoy positions. Ice concentration and depth are added to the daily positions, and wind is added to the hourly positions. Calculates drift velocity using forward differences for the daily positions and using centered differences for the hourly positions. Results saved to `data/daily_merged_buoy_data` and `data/hourly_merged_buoy_data`. 
5. `prepare_bathymetry_for_plotting.py` Regrids the IBCAO bathymetry to lat/lon coordinates to make plotting easier.
6. `calculate_spectra.py` Uses the PyCurrents spectrum function to calculate the rotary power spectral densities and saves the results in `data/spectra`.
7. `calculate_tidal_fit.py` Applies the harmonic regression model to the buoy drift track anomalies, and saves the predicted positions, maximum daily currents, and the squared correlation coefficients to `data/harmonic_fit`.

# Figure generation
1. `figure1_subplots.py` Uses the merged buoy data, deformation time series, and the regridded IBCAO depths to create the components of Figure 1. These include the map with the buoy trajectories, the time series of drift speed, wind speed, SIC, buoy count, depth, and deformation, and the satellite imagery overlays.
2. `figure2_subplots.py` Uses the merged buoy data to generate plots of drift speed magnitude, predicted drift speed based on the wind model, and time series and histograms of turning angle and drift speed ratio.
3. `figure3_subplots.py` Uses the Floe Tracker data and the merged buoy data to show the Floe Tracker estimates of drift speed, direction, and variability, and the joint histograms of turning angle and drift speed ratio as a function of wind speed for Floe Tracker and for the MOSAiC buoys.
4. `figure4_subplots.py` Uses the PyCurrents spectra results and the harmonic fit results to plot the rotary power spectra, the map with the trajectory segments, and the coefficient of determination and maximum current boxplots.

# Supporting modules
1. `drifter.py` Helper functions used for calculating drift speed and for buoy trajectory quality control
2. `pycurrents_spectra.py` Code from the University of Hawaii PyCurrents package used to calculate rotary power spectra.
