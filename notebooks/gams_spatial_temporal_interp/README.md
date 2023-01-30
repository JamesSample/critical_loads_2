# Updating critical loads for water - 2023

The Norwegian National Lakes Survey undertaken in 2019 provides a basis for updating input datasets used to estimate critical loads for water. These notebooks document this process.

## Workflow

 1. **[Extracting updated water chemistry data](https://nbviewer.org/github/JamesSample/critical_loads_2/blob/master/notebooks/gams_spatial_temporal_interp/02a_get_data.ipynb)**. Getting the raw data necessary to update the critical loads models
 
 2. **[GAMs for Critical Loads](https://nbviewer.org/github/JamesSample/critical_loads_2/blob/master/notebooks/gams_spatial_temporal_interp/02b_statistical_models.ipynb)**. Exploring Generalised Additive Models to create (i) a spatial model for TOC, and (ii) a spatio-temporal model for NO3
 
 3. **[Predictive modelling](https://nbviewer.org/github/JamesSample/critical_loads_2/blob/master/notebooks/gams_spatial_temporal_interp/02c_interpolate_data.ipynb)**. Using the fitted GAMs to predict TOC and NO3 on a 1 km grid over Norway (and through time for NO3)
 
 4. **[Spatial summaries](https://nbviewer.org/github/JamesSample/critical_loads_2/blob/master/notebooks/gams_spatial_temporal_interp/02d_blr_summaries.ipynb)**. Using zonal statistics to summarise the updated water chemsitry for each BLR grid cell used for the critical loads calculations
 
 5. **[Estimating critical loads for water](https://nbviewer.org/github/JamesSample/critical_loads_2/blob/master/notebooks/gams_spatial_temporal_interp/03_generate_inputs.ipynb)**. Using the updated inputs to estimate critical loads for water
 
 ## Development and testing
 
 * **[Initial exploration of spatio-tempoeral GAMs](https://nbviewer.org/github/JamesSample/critical_loads_2/blob/master/notebooks/gams_spatial_temporal_interp/01_interpolate_toc_no3.ipynb)**. Exploring General Additive models for spatio-temporal interpolation of NO3 and TOC
 
 