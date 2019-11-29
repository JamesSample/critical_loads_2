# Critical Loads (DSToolkit workflow)

Improved data handling for the Critical Loads project. All core datasets are now centralised on the JupyterHub/Data Science Toolkit. 

The workflow focuses on data supplied from 2018 onwards, primarily using the 0.1 degree deposition grid from NILU.

## Workflow

 1. **[Data migration to the DSToolkit](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/01_migrate_to_jhub.ipynb)**. Migrate all relevant datasets from NIVA's internal servers to a new database strcuture on the JupyterHub/DSToolkit
 
 2. **[Deposition](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/02_deposition_new_grid.ipynb)**. Process deposition datasets from NILU, supplied using the 0.1 degree grid
  
 3. **[Vegetation](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/03_vegetation_new_grid.ipynb)**. Workflow for calculating critical loads for vegetation using the 0.1 degree deposition data and raster land cover 
 
 4. **[Water](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/04_water_new_grid.ipynb)**. To do.
 
 5. **[Soil](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/04_soil_new_grid.ipynb)**. Workflow for calculating critical loads for soil using the 0.1 degree deposition data and the "old" (BLR-based) soil critical loads 
 
## Development code

 * **[Prototype data migration](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/migrate_to_docker_test.ipynb)**. Initial migration of key project datasets into a Dockerised PostGIS workflow
