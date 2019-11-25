#-------------------------------------------------------------------------------
# Name:        critical_loads.py
# Purpose:     Functions to implement the updated (November 2018) Critical Loads 
#              workflow.
#
# Author:      James Sample
#
# Created:     20/11/2018
# Copyright:   (c) James Sample and NIVA, 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def view_dep_series(eng):
    """ Create a new deposition series in the database.
    
    Args:
        eng: Obj. Active database connection object
        
    Returns:
        Query grid.
    """
    import pandas as pd
    import nivapy3 as nivapy
    
    # Get existing series from db
    sql = "SELECT * FROM deposition.dep_series_defs"
    df = pd.read_sql(sql, eng)
    ser_grid = nivapy.da.make_query_grid(df, editable=True)
    
    # Instructions
    print("Click 'Add Row', then edit the last row in the table to reflect the new data you wish to upload.")
    
    return ser_grid

def add_dep_series(qgrid, eng):
    """ Inserts rows added using cl.view_dep_series() to the database.
    
    Args:
        qgrid: Obj. Manually edited query grid table returned by 
               cl.view_dep_series()
        eng:   Obj. Active database connection object
         
    Returns:
        DataFrame of rows added
    """
    import pandas as pd

    # Get existing series from db
    sql = "SELECT * FROM deposition.dep_series_defs"
    df = pd.read_sql(sql, eng)
    
    # Get new rows to be added
    df2 = qgrid.get_changed_df()
    ids = df['series_id'].astype(list)
    df2 = df2.query("series_id not in @ids")
    
    # Add to db
    df2.to_sql('dep_series_defs', 
               eng,
               'deposition',
               if_exists='append',
               index=False)
    
    print('%s new row(s) added successfully.' % len(df2))
    
    return df2

def upload_nilu_0_1deg_dep_data(data_fold, eng, series_id):
    """ Process .dat files containing deposition data supplied by NILU. This 
        function is based on the data supplied by NILU during 2017, which uses 
        the new 0.1 degree deposition grid.
    
    Args:
        dat_fold:  Str. Path to folder containing .dat files provided by NILU
        eng:       Obj. Active database connection object connect to the Docker
                   PostGIS db
        series_id: Int. 'series_id' for this dataset from the table 
                   deposition.dep_series_defs
        
    Returns:
        DataFrame of the data added to the database.
    """
    import glob
    import os
    import pandas as pd

    # Read NILU data
    search_path = os.path.join(data_fold, '*.dat')
    file_list = glob.glob(search_path)

    df_list = []
    for fpath in file_list:
        # Get par name
        name = os.path.split(fpath)[1].split('_')[:2]
        name = '_'.join(name)

        # Read file
        df = pd.read_csv(fpath, delim_whitespace=True, header=None,
                         names=['lat', 'lon', name])
        df.set_index(['lat', 'lon'], inplace=True)    
        df_list.append(df)

    # Combine
    df = pd.concat(df_list, axis=1)
    df.reset_index(inplace=True)

    # Calculate unique integer cell ID as latlon 
    # (both *100 and padded to 4 digits)
    df['cell_id'] = ((df['lat']*100).astype(int).map('{:04d}'.format) + 
                     (df['lon']*100).astype(int).map('{:04d}'.format))
    df['cell_id'] = df['cell_id'].astype(int)
    del df['lat'], df['lon'], df['tot_n']

    # Rename
    df.rename(columns={'tot_nhx':2, # N (red)
                       'tot_nox':1, # N (oks)
                       'tot_s':4},  # Non-marine S
              inplace=True)

    # Melt 
    df = pd.melt(df, var_name='param_id', id_vars='cell_id')

    # Add series ID
    df['series_id'] = series_id

    # Add to db
    df.to_sql('dep_values_0_1deg_grid', 
              con=eng, 
              schema='deposition', 
              if_exists='append', 
              index=False)
    
    print ('%s new rows added successfully.' % len(df))
    
    return df

def n_deposition_as_gdf_0_1deg(ser_id, eng, shp_path=None):
    """ Extracts N deposition data for the specified series as a geodataframe
        using NILU's 0.1 degree grid. Optionally, the data can be saved as a
        shapefile.
        
    Args: 
        ser_id:   Int. ID for deposition series of interest
        eng:      Obj. Active database connection object
        shp_path: Str. Raw path for shapefile to be created
        
    Returns:
        GeoDataFrame.
    """
    import geopandas as gpd
    
    # Get dep values
    sql_args={'ser_id':ser_id}
    sql = ("SELECT ST_Transform(b.geom, 32633) AS geom, "
           "  a.cell_id, "
           "  ROUND(a.n_dep) AS n_dep "
           "FROM (SELECT cell_id, SUM(value) as n_dep "
           "      FROM deposition.dep_values_0_1deg_grid "
           "      WHERE param_id IN (1, 2) "
           "      AND series_id = {ser_id} " 
           "      GROUP BY cell_id) AS a, "
           "deposition.dep_grid_0_1deg AS b "
           "WHERE a.cell_id = b.cell_id").format(**sql_args)
    gdf = gpd.read_postgis(sql, eng)
    
    if shp_path:
        gdf.to_file(shp_path)
        
    return gdf
    
def create_n_deposition_raster_0_1deg(ser_id, out_tif, snap_tif, eng, ndv=-9999):
    """ Create a raster of N deposition values from a PostGIS vector table. The 
        raster pixel size, extent, alignment etc. will be set from 'snap_ras'.
        
        You will be asked to enter a PostGIS user name and password.
        
        NOTE: The output is a 16-bit signed integer grid.
        
    Args:
        ser_id:   Int. ID for the data series of interest
        out_tif:  Str. Raw string for the .tif file to create
        snap_tif: Str. Raw string for .tif file to use as a 'snap raster'
        eng:      Obj. Active database connection object
        ndv:      Int. Value to use for NoData
        
    Returns:
        None. The grid is saved to the specified path.
    """
    import gdal
    import ogr
    import getpass

    # Get credentials
    pg_dict = {'dbname':'critical_loads',
               'host':'host.docker.internal',
               'port':'25432'}
    pg_dict['user'] = getpass.getpass(prompt='Username: ')
    pg_dict['pw'] = getpass.getpass(prompt='Password: ')

    # Build conn_str for GDAL
    conn_str = ('PG: dbname={dbname}'
                '    host={host}'
                '    user={user}'
                '    password={pw}'
                '    port={port}').format(**pg_dict)
       
    # Create vector dep grid
    # Get dep values
    sql = ("CREATE TABLE deposition.temp_ndep AS ( "
           "SELECT ST_Transform(b.geom, 32633) AS geom, "
           "  a.cell_id, "
           "  ROUND(a.n_dep) AS n_dep "
           "FROM (SELECT cell_id, SUM(value) as n_dep "
           "      FROM deposition.dep_values_0_1deg_grid "
           "      WHERE param_id IN (1, 2) "
           "      AND series_id = %s " 
           "      GROUP BY cell_id) AS a, "
           "deposition.dep_grid_0_1deg AS b "
           "WHERE a.cell_id = b.cell_id)" % ser_id)
    eng.execute(sql)

    # Use 'cell_id' col as primary key
    sql = ("ALTER TABLE deposition.temp_ndep "
           "ADD CONSTRAINT temp_ndep_pk "
           "PRIMARY KEY (cell_id)")
    eng.execute(sql)

    # Enforce geom type
    sql = ("ALTER TABLE deposition.temp_ndep "
           "ALTER COLUMN geom TYPE geometry(MULTIPOLYGON, 32633) "
           "USING ST_Multi(ST_SetSRID(geom, 32633))")
    eng.execute(sql)

    # Create sp. index
    sql = ("CREATE INDEX temp_ndep_spidx "
           "ON deposition.temp_ndep "
           "USING GIST (geom)")
    eng.execute(sql)   
    
    # 1. Create new, empty raster with correct dimensions
    # Get properties from snap_tif
    snap_ras = gdal.Open(snap_tif)
    cols = snap_ras.RasterXSize
    rows = snap_ras.RasterYSize
    proj = snap_ras.GetProjection()
    geotr = snap_ras.GetGeoTransform()
    
    # Create out_tif
    driver = gdal.GetDriverByName('GTiff')
    out_ras = driver.Create(out_tif, cols, rows, 1, gdal.GDT_Int16)
    out_ras.SetProjection(proj)
    out_ras.SetGeoTransform(geotr)
    
    # Fill output with NoData
    out_ras.GetRasterBand(1).SetNoDataValue(ndv)
    out_ras.GetRasterBand(1).Fill(ndv)

    # 2. Rasterize vector
    conn = ogr.Open(conn_str, gdal.GA_ReadOnly)
    lyr = conn.GetLayer('deposition.temp_ndep')

    gdal.RasterizeLayer(out_ras, [1], lyr, 
                        options=['ATTRIBUTE=n_dep'])

    # Flush and close
    snap_ras = None
    out_ras = None
    shp_ds = None
    
    # Drop temp table
    sql = ("DROP TABLE deposition.temp_ndep")
    eng.execute(sql)  

def vec_to_ras(in_shp, out_tif, snap_tif, attrib, ndv, data_type,
               fmt='GTiff'):
    """ Converts a shapefile to a raster with values taken from
        the 'attrib' field. The 'snap_tif' is used to set the 
        resolution and extent of the output raster.
        
    Args:
        in_shp:    Str. Raw string to shapefile
        out_tif:   Str. Raw string for geotiff to create
        snap_tif:  Str. Raw string to geotiff used to set resolution
                   and extent
        attrib:    Str. Shapefile field for values
        ndv:       Int. No data value
        data_type: Str. GDAL bit depth:
                        'Byte'
                        'Int16'
                        'UInt16'
                        'UInt32'
                        'Int32'
                        'Float32'
                        'Float64'
        fmt:       Str. Format string.
        
    Returns:
        None. Raster is saved.
    """
    import ogr
    import gdal
    
    # Bit depth dict
    bit_dict = {'Byte':gdal.GDT_Byte,
                'Int16':gdal.GDT_Int16,
                'UInt16':gdal.GDT_UInt16,
                'UInt32':gdal.GDT_UInt32,
                'Int32':gdal.GDT_Int32,
                'Float32':gdal.GDT_Float32,
                'Float64':gdal.GDT_Float64}
    assert data_type in bit_dict.keys(), 'ERROR: Invalid data type.'
    
    # 1. Create new, empty raster with correct dimensions
    # Get properties from snap_tif
    snap_ras = gdal.Open(snap_tif)
    cols = snap_ras.RasterXSize
    rows = snap_ras.RasterYSize
    proj = snap_ras.GetProjection()
    geotr = snap_ras.GetGeoTransform()
    
    # Create out_tif
    driver = gdal.GetDriverByName(fmt)
    out_ras = driver.Create(out_tif, cols, rows, 1, bit_dict[data_type])
    out_ras.SetProjection(proj)
    out_ras.SetGeoTransform(geotr)
    
    # Fill output with NoData
    out_ras.GetRasterBand(1).SetNoDataValue(ndv)
    out_ras.GetRasterBand(1).Fill(ndv)

    # 2. Rasterize shapefile
    shp_ds = ogr.Open(in_shp)
    shp_lyr = shp_ds.GetLayer()

    gdal.RasterizeLayer(out_ras, [1], shp_lyr, 
                        options=['ATTRIBUTE=%s' % attrib])

    # Flush and close
    snap_ras = None
    out_ras = None
    shp_ds = None

def reclassify_raster(in_tif, mask_tif, out_tif, reclass_df, reclass_col, ndv):
    """ Reclassify categorical values in a raster using a mapping
        in a dataframe. The dataframe index must contain the classes
        in in_tif and the 'reclass_col' must specify the new classes.
        
        Only cells with value=1 in 'mask_tif' are written to output.

    Args:
        in_tif:      Str. Raw path to input raster
        mask_tif:    Str. Raw path to mask grid defining land area
        out_tif:     Str. Raw path to .tif file to create
        reclass_df:  DataFrame. Reclassification table
        reclass_col: Str. Name of column with new raster values
        ndv:         Int. Value to use as NoData in the new raster
        
    Returns:
        None. A new raster is saved.
    """
    import gdal
    import ogr
    from gdalconst import GA_ReadOnly as GA_ReadOnly
    import numpy as np
    import pandas as pd

    # Open source file, read data
    src_ds = gdal.Open(in_tif, GA_ReadOnly)
    assert(src_ds)
    rb = src_ds.GetRasterBand(1)
    src_data = rb.ReadAsArray()

    # Open mask, read data
    mask_ds = gdal.Open(mask_tif, GA_ReadOnly)
    assert(mask_ds)
    mb = mask_ds.GetRasterBand(1)
    mask_data = mb.ReadAsArray()
    
    # Reclassify
    rc_data = src_data.copy()
    for idx, row in reclass_df.iterrows():
        rc_data[src_data==idx] = row[reclass_col]

    # Apply mask
    rc_data[mask_data!=1] = ndv
    
    # Write output
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.CreateCopy(out_tif, src_ds, 0)
    out_band = dst_ds.GetRasterBand(1)
    out_band.SetNoDataValue(ndv)
    out_band.WriteArray(rc_data)

    # Flush data and close datasets
    dst_ds = None
    src_ds = None
    mask_ds = None
    
def calc_vegetation_exceedance_0_1deg(dep_tif, cl_tif, ex_tif, ex_tif_bool, ser_id):
    """ Calculate exceedances for vegetation.
    
    Args:
        dep_tif:     Str. Raw string to deposition grid
        cl_tif:      Str. Raw string to critical loads grid
        ex_tif:      Str. Raw string to exceedance grid to be created
        ex_tif_bool: Str. Raw string to exceedance grid with Boolean values (i.e. 1
                     where exceeded and 0 otherwise)
        ser_id:      Int. Deposition series ID for the data of interest
        
    Returns:
        Summary dataframe.
    """
    import nivapy3 as nivapy
    import pandas as pd
    import numpy as np
    import gdal
    
    # Check series ID is compatible
    assert ser_id > 27, ''
    # Container for output
    data_dict = {'total_area_km2':[],
                 'exceeded_area_km2':[]}

    # Read grids
    cl_grid, cl_ndv, cl_epsg, cl_ext = nivapy.spatial.read_raster(cl_tif)
    dep_grid, dep_ndv, dep_epsg, dep_ext = nivapy.spatial.read_raster(dep_tif)
    
    # Work out cell size
    cs = (cl_ext[1] - cl_ext[0]) / cl_grid.shape[1]
    cs = float(int(cs + 0.5))

    # Upcast to float32 for safe handling of negative values
    cl_grid = cl_grid.astype(np.float32)
    dep_grid = dep_grid.astype(np.float32)
   
    # Set ndv
    cl_grid[cl_grid==cl_ndv] = np.nan
    dep_grid[dep_grid==dep_ndv] = np.nan

    # Get total area of non-NaN from dep grid
    nor_area = np.count_nonzero(~np.isnan(dep_grid))*cs*cs/1.E6

    # Apply scaling factor to CLs
    cl_grid = cl_grid*100.

    # Exceedance
    ex_grid = dep_grid - cl_grid
    del dep_grid, cl_grid  
    
    # Get total area exceeded
    ex_area = np.count_nonzero(ex_grid > 0)*cs*cs/1.E6

    # Set <0 to 0
    ex_grid[ex_grid<0] = 0
    
    # Reset ndv
    ex_grid[np.isnan(ex_grid)] = -1

    # Downcast to int16 to save space
    ex_grid = ex_grid.round(0).astype(np.int16)
    
    # Append results
    data_dict['total_area_km2'].append(nor_area)
    data_dict['exceeded_area_km2'].append(ex_area)
    
    # Write exceedance output  
    write_geotiff(ex_grid, ex_tif, cl_tif, -1, gdal.GDT_Int16)
    
    # Convert to bool grid
    ex_grid[ex_grid>0] = 1
    ex_grid[ex_grid==-1] = 255
    
    # Write bool output  
    write_geotiff(ex_grid, ex_tif_bool, cl_tif, 255, gdal.GDT_Byte)
    del ex_grid

    # Build output df
    ex_df = pd.DataFrame(data_dict)
    ex_df['exceeded_area_pct'] = 100 * ex_df['exceeded_area_km2'] / ex_df['total_area_km2']
    ex_df = ex_df.round(0).astype(int)
    ex_df['series_id'] = ser_id
    ex_df['medium'] = 'vegetation'
    ex_df = ex_df[['series_id', 'medium', 'total_area_km2', 
                   'exceeded_area_km2', 'exceeded_area_pct']]

    return ex_df

def write_geotiff(data, out_tif, snap_tif, ndv, data_type):
    """ Write a numpy array to a geotiff using 'snap_tif' to define
        raster properties.
    
    Args:
        data:      Array.
        out_tif:   Str. File to create
        snap_tif:  Str. Path to existing tif with same resolution
                   and extent as target
        ndv:       Int. No data value
        data_type: Bit depth etc. e.g. gdal.GDT_UInt32
        
    Returns:
        None. Geotiff is saved.        
    """
    from osgeo import ogr
    from osgeo import gdal
    
    # 1. Create new, empty raster with correct dimensions
    # Get properties from snap_tif
    snap_ras = gdal.Open(snap_tif)
    cols = snap_ras.RasterXSize
    rows = snap_ras.RasterYSize
    proj = snap_ras.GetProjection()
    geotr = snap_ras.GetGeoTransform()
    
    # Create out_tif
    driver = gdal.GetDriverByName('GTiff')
    out_ras = driver.Create(out_tif, cols, rows, 1, data_type)
    out_ras.SetProjection(proj)
    out_ras.SetGeoTransform(geotr)
    
    # Write data
    out_band = out_ras.GetRasterBand(1)
    out_band.SetNoDataValue(ndv)
    out_band.WriteArray(data)

    # Flush and close
    snap_ras = None
    out_band = None
    out_ras = None
    
def bbox_to_pixel_offsets(gt, bbox):
    """ Helper function for zonal_stats(). Modified from:

        https://gist.github.com/perrygeo/5667173

        Original code copyright 2013 Matthew Perry
    """
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    
    return (x1, y1, xsize, ysize)

def remap_categories(category_map, stats):
    """ Modified from https://gist.github.com/perrygeo/5667173
        Original code copyright 2013 Matthew Perry
    """
    def lookup(m, k):
        """ Dict lookup but returns original key if not found
        """
        try:
            return m[k]
        except KeyError:
            return k

    return {lookup(category_map, k): v
            for k, v in stats.items()}

def exceedance_stats_per_0_1deg_cell(ex_tif, ser_id, eng, write_to_db=True, nodata_value=-1, 
                                     global_src_extent=False, categorical=False, category_map=None):
    """ Summarise exceedance values for each 0.1 degree grid cell. 

    Args:
        raster_path:       Raw str. Path to exceedance raster
        ser_id:            Int. Deposition series ID
        eng:               Obj. Active database connection object
        write_to_db:       Bool. If True, results will be written to the database
        nodata_value:      Float. Value in raster to treat as NoData
        global_src_extent: Bool. If True, reads all data into memory in a single
                           pass. May be faster, but also takes up loats of memory
                           when used with large vector or raster datasets
        categorical:       Bool. If true, raster is assumed to be categorical, with
                           integer values representing different categories (e.g. land
                           use). In this case, the statistics returned are pixel counts
                           of each category within each vector zone
        category_map:      Dict. Only used when "categorical" is True. Dict mapping
                           integer values to category names {int_id:'cat_name'}. If
                           supplied, the integer categories in the results dataframe
                           will be mapped to the specified category names

    Returns:
        GeoDataFrame of cell statistics.
    """
    import gdal
    import ogr
    import numpy as np
    import pandas as pd
    import sys
    import geopandas as gpd
    import os
    from gdalconst import GA_ReadOnly
    
    gdal.PushErrorHandler('CPLQuietErrorHandler')

    # Read vector
    temp_fold = os.path.split(ex_tif)[0]
    temp_shp = os.path.join(temp_fold, 'temp.shp')
    gdf = n_deposition_as_gdf_0_1deg(ser_id, eng, shp_path=temp_shp)    
    
    # Read raster
    rds = gdal.Open(ex_tif, GA_ReadOnly)
    assert(rds)
    rb = rds.GetRasterBand(1)
    rgt = rds.GetGeoTransform()
    
    # Get cell size
    cs = rgt[1]

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(temp_shp, GA_ReadOnly)  # TODO maybe open update if we want to write stats
    assert(vds)
    vlyr = vds.GetLayer(0)

    # create an in-memory numpy array of the source raster data
    # covering the whole extent of the vector layer
    if global_src_extent:
        # use global source extent
        # useful only when disk IO or raster scanning inefficiencies are your limiting factor
        # advantage: reads raster data in one pass
        # disadvantage: large vector extents may have big memory requirements
        src_offset = bbox_to_pixel_offsets(rgt, vlyr.GetExtent())
        src_array = rb.ReadAsArray(*src_offset)

        # calculate new geotransform of the layer subset
        new_gt = (
            (rgt[0] + (src_offset[0] * rgt[1])),
            rgt[1],
            0.0,
            (rgt[3] + (src_offset[1] * rgt[5])),
            0.0,
            rgt[5]
        )

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature()
    while feat is not None:

        if not global_src_extent:
            # use local source extent
            # fastest option when you have fast disks and well indexed raster (ie tiled Geotiff)
            # advantage: each feature uses the smallest raster chunk
            # disadvantage: lots of reads on the source raster
            src_offset = bbox_to_pixel_offsets(rgt, feat.geometry().GetEnvelope())
            src_array = rb.ReadAsArray(*src_offset)

            # calculate new geotransform of the feature subset
            new_gt = (
                (rgt[0] + (src_offset[0] * rgt[1])),
                rgt[1],
                0.0,
                (rgt[3] + (src_offset[1] * rgt[5])),
                0.0,
                rgt[5]
            )

        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())

        # Rasterize it
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()

        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
        masked = np.ma.MaskedArray(
            src_array,
            mask=np.logical_or(
                src_array == nodata_value,
                np.logical_not(rv_array)
            )
        )

        if categorical:
            # Get cell counts for each category
            keys, counts = np.unique(masked.compressed(), return_counts=True)
            pixel_count = dict(zip([np.asscalar(k) for k in keys],
                                   [np.asscalar(c) for c in counts]))

            feature_stats = dict(pixel_count)
            if category_map:
                feature_stats = remap_categories(category_map, feature_stats)
        
        else:
            # Get summary stats
            feature_stats = {
                'min': float(masked.min()),
                'mean': float(masked.mean()),
                'max': float(masked.max()),
                'std': float(masked.std()),
                'sum': float(masked.sum()),
                'count': int(masked.count()),
                'fid': int(feat.GetFID())}                        
                        
        stats.append(feature_stats)

        rvds = None
        mem_ds = None
        feat = vlyr.GetNextFeature()

    # Tidy up
    vds = None
    rds = None
    for fname in ['temp.cpg', 'temp.dbf', 'temp.prj', 'temp.shp', 'temp.shx']:
        os.remove(os.path.join(temp_fold, fname))    
    
    # Combine results
    df = pd.DataFrame(stats)
    df.fillna(0, inplace=True)
    df['series_id'] = ser_id
    df['fid'] = df.index
    gdf['fid'] = gdf.index
    gdf = gdf.merge(df, on='fid')
    
    # Calc areas
    gdf['exceeded_area_km2'] = gdf['exceeded']*cs*cs/1E6 
    gdf['total_area_km2'] = (gdf['exceeded'] + gdf['not_exceeded'])*cs*cs/1E6
    gdf['pct_exceeded'] = 100*gdf['exceeded_area_km2']/gdf['total_area_km2']    
    del gdf['n_dep'], gdf['fid'], gdf['exceeded'], gdf['not_exceeded']
    gdf.dropna(how='any', inplace=True)
    
    if write_to_db:
        gdf2 = gdf.copy()
        del gdf2['geom']
        df = pd.DataFrame(gdf2)
        df.to_sql('exceedance_stats_0_1deg_grid', 
                  eng,
                  'vegetation',
                  if_exists='append',
                  index=False)    
    
    return gdf

def exceedance_stats_per_land_use_class(ex_tif_bool, veg_tif, ser_id, eng, write_to_db=True, 
                                        nodata_value=255):
    """ Summarise exceedance values for each land use class. 

    Args:
        ex_tif_bool:       Raw str. Path to boolean exceedance raster
        veg_tif:           Str. Path to vegetation data with same resolution as ex_tif_bool
        ser_id:            Int. Deposition series ID
        eng:               Obj. Active database connection object
        write_to_db:       Bool. If True, results will be written to the database
        nodata_value:      Float. Value in rasters to treat as NoData

    Returns:
        GeoDataFrame of land use statistics.
    """
    import gdal
    import ogr
    import numpy as np
    import pandas as pd
    import sys
    import geopandas as gpd
    import os
    from gdalconst import GA_ReadOnly
    
    gdal.PushErrorHandler('CPLQuietErrorHandler')

    # Read LU table
    sql = ("SELECT * FROM vegetation.land_class_crit_lds")
    lu_df = pd.read_sql(sql, eng)    
    
    # Read exceedance raster
    rds = gdal.Open(ex_tif_bool, GA_ReadOnly)
    assert(rds)
    rb = rds.GetRasterBand(1)

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)
        
    ex_array = rb.ReadAsArray()

    # Get cell size
    rgt = rds.GetGeoTransform()
    cs = rgt[1]

    # Read vegetation raster
    rds = gdal.Open(veg_tif, GA_ReadOnly)
    assert(rds)
    rb = rds.GetRasterBand(1)

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)
        
    veg_array = rb.ReadAsArray()

    # Loop through land classes
    stats = []
    for idx, row in lu_df.iterrows():
        # Mask the source data array
        masked = np.ma.MaskedArray(ex_array,
                                   mask=np.logical_or(ex_array==nodata_value,
                                                      veg_array!=row['norut_code']))

        # Get cell counts for each category
        keys, counts = np.unique(masked.compressed(), return_counts=True)
        pixel_count = dict(zip([np.asscalar(k) for k in keys],
                               [np.asscalar(c) for c in counts]))

        feature_stats = dict(pixel_count)
        feature_stats = remap_categories({1:'exceeded',
                                          0:'not_exceeded'},
                                         feature_stats)                     
        stats.append(feature_stats)

    # Tidy up
    rds = None  
    
    # Combine results
    df = pd.DataFrame(stats)
    df.fillna(0, inplace=True)
    df['norut_code'] = lu_df['norut_code']
    df['series_id'] = ser_id
    
    # Calc areas
    df['exceeded_area_km2'] = df['exceeded']*cs*cs/1E6 
    df['total_area_km2'] = (df['exceeded'] + df['not_exceeded'])*cs*cs/1E6
    df['pct_exceeded'] = 100*df['exceeded_area_km2']/df['total_area_km2']    
    del df['exceeded'], df['not_exceeded']
    df.dropna(how='any', inplace=True)
    
    if write_to_db:
        df.to_sql('exceedance_stats_land_class', 
                  eng,
                  'vegetation',
                  if_exists='append',
                  index=False)    
    
    return df

def veg_exceedance_as_gdf_0_1deg(ser_id, eng, shp_path=None):
    """ Extracts exceedance statistics for the specified series as a 
        geodataframe using NILU's 0.1 degree grid. Optionally, the data
        can be saved as a shapefile.
        
    Args: 
        ser_id:   Int. ID for deposition series of interest
        eng:      Obj. Active database connection object
        shp_path: Str. Raw path for shapefile to be created
        
    Returns:
        GeoDataFrame.
    """
    import geopandas as gpd
    
    # Get dep values
    sql_args={'ser_id':ser_id}
    sql = ("SELECT ST_Transform(b.geom, 32633) AS geom, "
           "  a.cell_id, "
           "  a.exceeded_area_km2, "
           "  a.total_area_km2, "
           "  a.pct_exceeded "
           "FROM (SELECT cell_id, "
           "             exceeded_area_km2, "
           "             total_area_km2, "
           "             pct_exceeded "
           "      FROM vegetation.exceedance_stats_0_1deg_grid "
           "      WHERE series_id = {ser_id}) AS a, "
           "deposition.dep_grid_0_1deg AS b "
           "WHERE a.cell_id = b.cell_id").format(**sql_args)
    gdf = gpd.read_postgis(sql, eng)
    
    if shp_path:
        gdf.to_file(shp_path)
        
    return gdf