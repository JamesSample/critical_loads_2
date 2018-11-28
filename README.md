# Critical Loads (2018 workflow)

Improved data handling for the Critical Loads project. All core datasets are now centralised in PostGIS running on Docker, giving a system that is both neater and more compatible with the developing NIVA data platform. 

## Workflow

 1. **[Data migration](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/01_migrate_to_postgis.ipynb)**. Migrating key project datasets into PostGIS
 
 2. **[Critical loads for vegetation (2018 onwards)](http://nbviewer.jupyter.org/github/JamesSample/critical_loads_2/blob/master/notebooks/03_vegetation_new_grid.ipynb)**. Calculating citical loads for vegetation using the 0.1 degree deposition data from NILU (first supplied in 2018)
