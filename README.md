# Critical Loads (2018 workflow)

Improved data handling for the Critical Loads project. All core datasets are now centralised in the Data Science Toolkit. 

## Workflow for 2018 onwards

 1. **[Data migration to the DSToolkit](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/01_migrate_to_jhub.ipynb)**. Migrate all relevant datasets from NIVA's internal servers to a new database strcuture on the JupyterHub/DSToolkit
 
 2. **[Critical loads for vegetation (2018 onwards)](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/02_deposition_and_vegetation_new_grid.ipynb)**. Workflow for calculating critical loads for vegetation using the 0.1 degree deposition data from NILU (first supplied in 2018)
 
 3. **[Critical loads for water (2018 onwards)](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/03_water_new_grid.ipynb)**. To do.
 
 4. **[Critical loads for soil (2018 onwards)](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/04_soil_new_grid.ipynb)**. Workflow for calculating critical loads for soil using the 0.1 degree deposition data from NILU (first supplied in 2018) 
 
## Development code

 * **[Prototype data migration to Docker/PostGIS](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/migrate_to_docker_test.ipynb)**. Initial migration of key project datasets into a Dockerised PostGIS workflow
