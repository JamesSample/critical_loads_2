# Updating exceedances for water (November 2026)

This notebook estimates exceedances for water for 2040 and 2050, based on data provided by NILU and EMEP.

## Workflow overview

### 1. Historical data

The notebook `critical_loads_2/notebooks/workflow_update_2023/05a_exceedances_for_water.ipynb` is the main notebook for calculating **historic** critical load exceedances. First upload any new historic deposition data and run this notebook. 

 * Deposition rasters will be saved to `critical_loads/raster/deposition/{short_name}`
 * Critical loads rasters for water will be saved to `critical_loads/raster/water/{short_name}`
 * Exceedance grids will be saved to `critical_loads/raster/exceedance/{short_name}`

Note that, in section 3 of the notebook (*Rasterise critical loads*), you have the option of using updated water chemistry based on statistical analysis undertaken in 2023 (recommended), or using the original parameters from the old critical loads database.

### 2. Add EMEP deposition data

New EMEP grids have been delivered as part of the GP review. Relevant grids are uploaded to the database in `critical_loads_2/notebooks/calculate_exceedance_2026/01_update_deposition.ipynb`.

### 3. Bias correct and calculate exceedances

In `critical_loads_2/notebooks/calculate_exceedance_2026/02_future_exceedance.ipynb`, deposition for 2040 and 2050 are estimated by calculating change factors compared EMEP-2015 and applying them to the NILU data for 2012-16 (0.1 degree grid; short name `1216`). Exceedances are then calculated using the bias-corrected deposition data and the critical loads grids for 2017-21 (based on data from met.no; short name `1721_blrgrid`).

### 4. Maps and summaries

`projects/critical_loads_2/notebooks/calculate_exceedance_2026/03_maps.ipynb` creates maps and prints summary data.