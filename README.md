This folder contains the code and data used for the paper "Evidence of abrupt transitions between sea ice dynamical regimes in the East Greenland marginal ice zone". In the final version, we will include a list of package versions and instructions for setting up a conda environment to ensure reproducibility. 

The scripts used to prepare the data take a more complex Python environment and require the external datasets to be downloaded. The results of these scripts are saved in the folder data_for_analysis which includes subfolders for daily downsampled observations, hourly observations, Ice Floe Tracker observations,  spectra, and the results of the harmonic fit. The daily resolution data includes sea ice concentration, depth, and daily median wind speed interpolated to buoy positions and derived quantities. The hourly observations include derived quantities, depth, and the interpolated ERA5 winds.

Python version 3.9.13

Python libraries used:
- cartopy
- h5py
- netcdf4
- numpy 1.23.3
- proplot 0.9.5
- pyproj 
- pycurrents
- pandas
- scipy
- rioxarray
- xarray

External datasets needed:
- Buoy drift tracks from the Arctic Data Center
- IBCAO_v4_2_400m.nc
- AMSR2 25 km sea ice concentration
- ERA5 hourly u10, v10 for the region bounded by 60-90 latitude and -30 to 30 longitude

# Data preparation
## clean_buoy_data.py
Reads the MOSAiC buoy tracks downloaded from the Arctic Data Center, applies quality control, and resamples to hourly resolution.

## clean_ft_data.py
Reads the CSV files with IFT floe positions, interpolates to daily resolution, computes turning angle and drift speed ratios.

## compile_amsr2_data.py
Script to select AMSR2 data for the study region and merge it into a single smaller netcdf file (amsr2_sea_ice_concentration.nc). Only run this if recalculating sea ice concentration from the daily AMSR2 files.

## merge_data.py
Takes the ocean depth, winds, and sea ice concentration data, interpolates it to the buoy locations.

## rotary_spectra.py
Reads the files in data/buoy_data_for_analysis and applies the PyCurrents spectra function, then plots the results.
TBD split this function into calculating spectra and plotting

# Figure creation
## figure1_subplots.py
Uses the merged buoy data, deformation time series, and the regridded IBCAO depths to create the components of Figure 1. These include the map with the buoy trajectories, the time series of drift speed, wind speed, SIC, buoy count, depth, and deformation, and the satellite imagery overlays.

## figure2_subplots.py
Uses the merged buoy data to 