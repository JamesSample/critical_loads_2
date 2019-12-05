# Critical Loads (DSToolkit workflow)

Improved data handling for the Critical Loads project. All core datasets are now centralised on the JupyterHub/Data Science Toolkit. 

Key datasets include:

 * **Atmospheric deposition of N and S** supplied by NILU using a 0.1 degree resolution grid
 
 * **Raster vegetation data** with a cell resolution of 30 m
 
 * **Water- and soil-related parameters**, aggregated to the level of the old "BLR" grid
 
Exceedances of critical loads are calculated following the method described by the [Manual for Modelling and Mapping Critical Loads & Levels](https://www.umweltbundesamt.de/en/manual-for-modelling-mapping-critical-loads-levels?parent=68093) (see especially [Chapter 5](https://www.umweltbundesamt.de/sites/default/files/medien/4292/dokumente/ch5-mapman-2017-09-10.pdf)).

To make it easier to update and re-calculate critical loads for **water**, an Excel template is also available [here](https://github.com/JamesSample/critical_loads_2/blob/master/notebooks/input_template_critical_loads_water.xlsx). This can be used in conjunction with the Python function [here](https://github.com/JamesSample/critical_loads_2/blob/ffb874835037b6b23db1edf8c56d1b0aded3ed2a/notebooks/critical_loads.py#L918) to estimate critical loads for any sites/catchments/regions of interest.

## Workflow

 1. **[Data migration to the DSToolkit](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/01_migrate_to_jhub.ipynb)**. Migrate all relevant datasets from NIVA's internal servers to a new database structure on the JupyterHub/DSToolkit
 
 2. **[Deposition](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/02_deposition_new_grid.ipynb)**. Process deposition datasets from NILU, supplied using the 0.1 degree grid
  
 3. **[Vegetation](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/03_vegetation_new_grid.ipynb)**. Workflow for calculating exceedances of critical loads for vegetation using the 0.1 degree deposition data and raster land cover 
 
 4. **[Water](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/04_water_new_grid.ipynb)**. Workflow for calculating exceedances of critical loads for water using the 0.1 degree deposition data and two different exceedance models (**SSWC** and **FAB**)
 
 5. **[Soil](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/05_soil_new_grid.ipynb)**. Workflow for calculating exceedances of critical loads for soil using the 0.1 degree deposition data and the "old" (BLR-based) soil critical loads 
 
## Development code

 * **[Prototype data migration](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/migrate_to_docker_test.ipynb)**. Initial migration of key project datasets into a Dockerised PostGIS workflow
