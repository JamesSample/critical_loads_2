# -------------------------------------------------------------------------------
# Name:        critical_loads.py
# Purpose:     Functions to implement the updated (November 2018) Critical Loads
#              workflow.
#
# Author:      James Sample
# -------------------------------------------------------------------------------


def view_dep_series(eng):
    """View table of deposition series already in the database.

    Args:
        eng: Obj. Active database connection object

    Returns:
        Dataframe.
    """
    import pandas as pd

    # Get existing series from db
    sql = "SELECT * FROM deposition.dep_series_defs"
    df = pd.read_sql(sql, eng)

    return df


def add_dep_series(series_id, name, short_name, grid, desc, eng):
    """Add new deposition series to the database.

    Args:
        series_id:  Int. Unique integer ID for series
        name:       Str. Full name for this series
        short_name: Str. Short name for this series
        grid:       Str. One of ['blr', '0_1deg', 'emep']
        desc:       Str. Description of series
        eng:        Obj. Active database connection object

    Returns:
        None. Row is added.
    """
    import pandas as pd
    from sqlalchemy.sql import text

    assert isinstance(series_id, int), "'series_id' must be an integer."
    assert grid in (
        "blr",
        "0_1deg",
        "emep",
    ), "'grid' must be one of ('blr', '0_1deg', 'emep')."

    # Get existing series from db
    sql = (
        "INSERT INTO deposition.dep_series_defs "
        "(series_id, name, short_name, grid, description) "
        "VALUES "
        "(:series_id, :name, :short_name, :grid, :desc)"
    )

    param_dict = {
        "series_id": series_id,
        "name": name,
        "short_name": short_name,
        "grid": grid,
        "desc": desc,
    }

    sql = text(sql)
    eng.execute(sql, param_dict)

    print("Series added successfully.")

    return None


def upload_nilu_0_1deg_dep_data(data_fold, eng, series_id):
    """Process .dat files containing deposition data supplied by NILU. This
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
    search_path = os.path.join(data_fold, "*.dat")
    file_list = glob.glob(search_path)

    df_list = []
    for fpath in file_list:
        # Get par name
        name = os.path.split(fpath)[1].split("_")[:2]
        name = "_".join(name)

        # Read file
        df = pd.read_csv(
            fpath, delim_whitespace=True, header=None, names=["lat", "lon", name]
        )
        df.set_index(["lat", "lon"], inplace=True)
        df_list.append(df)

    # Combine
    df = pd.concat(df_list, axis=1)
    df.reset_index(inplace=True)

    # Calculate unique integer cell ID as latlon
    # (both *100 and padded to 4 digits)
    df["cell_id"] = (df["lat"] * 100).astype(int).map("{:04d}".format) + (
        df["lon"] * 100
    ).astype(int).map("{:04d}".format)
    df["cell_id"] = df["cell_id"].astype(int)
    del df["lat"], df["lon"], df["tot_n"]

    # Rename
    df.rename(
        columns={
            "tot_nhx": 2,  # N (red)
            "tot_nox": 1,  # N (oks)
            "tot_s": 4,
        },  # Non-marine S
        inplace=True,
    )

    # Melt
    df = pd.melt(df, var_name="param_id", id_vars="cell_id")

    # Add series ID
    df["series_id"] = series_id

    # Add to db
    df.to_sql(
        "dep_values_0_1deg_grid",
        con=eng,
        schema="deposition",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    print("%s new rows added successfully." % len(df))

    return df


def extract_deposition_as_gdf(series_id, par, eng, veg_class=None):
    """Extracts deposition data for the specified series as a geodataframe.

    Args:
        series_id: Int. ID for deposition series of interest
        par:       Str. One of ['nitrogen', 'sulphur']
        eng:       Obj. Active database connection object
        veg_class: Str or None. Only applies for data using the EMEP grid, which
                   reports deposition values for different vegetation classes.
                   For EMEP, must be one of ['grid average', 'forest', 'semi-natural'];
                   otherwise, pass None

    Returns:
        GeoDataFrame.
    """
    import warnings

    import geopandas as gpd
    from sqlalchemy.sql import text

    veg_class_dict = {"grid average": 1, "forest": 2, "semi-natural": 3}

    assert isinstance(series_id, int), "'series_id' must be an integer."
    assert par in (
        "nitrogen",
        "sulphur",
    ), "'par' must be one of ('nitrogen', 'sulphur')."

    # Identify grid
    param_dict = {"series_id": series_id}
    sql = "SELECT grid FROM deposition.dep_series_defs " "WHERE series_id = :series_id"
    sql = text(sql)
    grid = eng.execute(sql, param_dict).fetchall()[0][0]

    assert (
        grid is not None
    ), "'grid' is not defined for this series in the 'dep_series_defs' table.\n"

    if (grid == "emep") and (veg_class is None):
        assert veg_class in ["grid average", "forest", "semi-natural"], (
            "The specified series ID refers to the EMEP grid, "
            "so you must also specify the 'veg_class' parameter.\n"
            "Choose one of ['grid average', 'forest', 'semi-natural'] "
            "and pass e.g. veg_class='grid average'."
        )

    if (grid != "emep") and (veg_class is not None):
        print(
            "WARNING: The specified series ID does NOT refer to the EMEP grid. "
            "The 'veg_class' parameter will be ignored."
        )

    if par == "nitrogen":
        unit_factor = 1 / 14.01

        if grid == "emep":
            param_dict["veg_class_id"] = veg_class_dict[veg_class]

            # Choose 'grid-average' for veg class
            sql = (
                f"SELECT ST_Multi(ST_Transform(b.geom, 32633)) AS geom, "
                f"  a.cell_id, "
                f"  ROUND(a.n_dep) AS ndep_mgpm2pyr "
                f"FROM (SELECT cell_id, SUM(value) as n_dep "
                f"      FROM deposition.dep_values_{grid}_grid "
                f"      WHERE param_id IN (1, 2) "
                f"      AND veg_class_id = :veg_class_id "
                f"      AND series_id = :series_id "
                f"      GROUP BY cell_id) AS a, "
                f"deposition.dep_grid_{grid} AS b "
                f"WHERE a.cell_id = b.cell_id"
            )

        else:
            # No veg classes to consider
            sql = (
                f"SELECT ST_Multi(ST_Transform(b.geom, 32633)) AS geom, "
                f"  a.cell_id, "
                f"  ROUND(a.n_dep) AS ndep_mgpm2pyr "
                f"FROM (SELECT cell_id, SUM(value) as n_dep "
                f"      FROM deposition.dep_values_{grid}_grid "
                f"      WHERE param_id IN (1, 2) "
                f"      AND series_id = :series_id "
                f"      GROUP BY cell_id) AS a, "
                f"deposition.dep_grid_{grid} AS b "
                f"WHERE a.cell_id = b.cell_id"
            )
    else:
        unit_factor = 2 / 32.06

        if grid == "emep":
            param_dict["veg_class_id"] = veg_class_dict[veg_class]

            # Choose 'grid-average' for veg class
            sql = (
                f"SELECT ST_Multi(ST_Transform(b.geom, 32633)) AS geom, "
                f"  a.cell_id, "
                f"  ROUND(a.s_dep) AS sdep_mgpm2pyr "
                f"FROM (SELECT cell_id, SUM(value) as s_dep "
                f"      FROM deposition.dep_values_{grid}_grid "
                f"      WHERE param_id = 4 "
                f"      AND veg_class_id = :veg_class_id "
                f"      AND series_id = :series_id "
                f"      GROUP BY cell_id) AS a, "
                f"deposition.dep_grid_{grid} AS b "
                f"WHERE a.cell_id = b.cell_id"
            )

        else:
            # No veg classes to consider
            sql = (
                f"SELECT ST_Multi(ST_Transform(b.geom, 32633)) AS geom, "
                f"  a.cell_id, "
                f"  ROUND(a.s_dep) AS sdep_mgpm2pyr "
                f"FROM (SELECT cell_id, SUM(value) as s_dep "
                f"      FROM deposition.dep_values_{grid}_grid "
                f"      WHERE param_id = 4 "
                f"      AND series_id = :series_id "
                f"      GROUP BY cell_id) AS a, "
                f"deposition.dep_grid_{grid} AS b "
                f"WHERE a.cell_id = b.cell_id"
            )

    sql = text(sql)
    gdf = gpd.read_postgis(sql, eng, params=param_dict)

    # Convert units
    gdf[par[0] + "dep_meqpm2pyr"] = gdf[par[0] + "dep_mgpm2pyr"] * unit_factor
    gdf[par[0] + "dep_kgphapyr"] = gdf[par[0] + "dep_mgpm2pyr"] / 100

    return gdf


def create_deposition_raster(
    series_id,
    par,
    unit,
    cell_size,
    eng,
    ndv=-9999,
    bit_depth="Int16",
    fname=None,
    veg_class=None,
):
    """Create a raster of deposition values from a Geodataframe.

    Args:
        series_id: Int. ID for deposition series of interest
        par:       Str. One of ['nitrogen', 'sulphur']
        unit:      Str. One of ['mgpm2pyr', kgphapyr, 'meqpm2pyr']
        cell_size: Int. Output cell size in metres. Determines the "snap raster" to be used
                   One of (30, 60, 120)
        eng:       Obj. Active database connection object
        ndv:       Int. Value to use for No Data
        bit_depth: Str. GDAL bit depth:
                       'Byte'
                       'Int16'
                       'UInt16'
                       'UInt32'
                       'Int32'
                       'Float32'
                       'Float64'
        fname:     Str or None. File path for output. If None, the raster will be saved to
        veg_class: Str or None. Only applies for data using the EMEP grid, which
                   reports deposition values for different vegetation classes.
                   For EMEP, must be one of ['grid average', 'forest', 'semi-natural'];
                   otherwise, pass None

                       shared/critical_loads/raster/deposition/short_name.tif

                   where 'short_name' is as defined in the 'dep_series_defs' table.

    Returns:
        None. The grid is saved to the specified path.
    """
    import os

    import geopandas as gpd
    from sqlalchemy.sql import text

    assert unit in (
        "mgpm2pyr",
        "kgphapyr",
        "meqpm2pyr",
    ), "'unit' must be one of ('mgpm2pyr', 'kgphapyr', 'meqpm2pyr')."

    assert cell_size in (30, 60, 120), "'cell_size' must be one of (30, 60, 120)."

    # Get data
    gdf = extract_deposition_as_gdf(series_id, par, eng, veg_class=veg_class)

    # Save temporary file
    gdf.to_file("temp_ndep.geojson", driver="GeoJSON")

    # Convert to raster
    col_name = f"{par[0]}dep_{unit}"

    if fname is None:
        # Get short_name from db
        param_dict = {"series_id": series_id}
        sql = "SELECT short_name FROM deposition.dep_series_defs WHERE series_id = :series_id"
        sql = text(sql)
        res = eng.execute(sql, param_dict).fetchall()[0][0]
        assert res is not None, (
            "'short_name' is not defined for this series in the 'dep_series_defs' table.\n"
            "Consider explicitly specifying a file name?"
        )

        fname = f"/home/jovyan/shared/critical_loads/raster/deposition/{col_name}_{res}_{cell_size}m.tif"

    snap_tif = (
        f"/home/jovyan/shared/critical_loads/raster/blr_land_mask_{cell_size}m.tif"
    )

    vec_to_ras("temp_ndep.geojson", fname, snap_tif, col_name, ndv, bit_depth)

    # Delete temp file
    os.remove("temp_ndep.geojson")


def vec_to_ras(in_shp, out_tif, snap_tif, attrib, ndv, data_type, fmt="GTiff"):
    """Converts a shapefile to a raster with values taken from
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
    from osgeo import gdal, ogr

    # Bit depth dict
    bit_dict = {
        "Byte": gdal.GDT_Byte,
        "Int16": gdal.GDT_Int16,
        "UInt16": gdal.GDT_UInt16,
        "UInt32": gdal.GDT_UInt32,
        "Int32": gdal.GDT_Int32,
        "Float32": gdal.GDT_Float32,
        "Float64": gdal.GDT_Float64,
    }
    assert data_type in bit_dict.keys(), "ERROR: Invalid data type."

    # 1. Create new, empty raster with correct dimensions
    # Get properties from snap_tif
    snap_ras = gdal.Open(snap_tif)
    cols = snap_ras.RasterXSize
    rows = snap_ras.RasterYSize
    proj = snap_ras.GetProjection()
    geotr = snap_ras.GetGeoTransform()

    # Create out_tif
    driver = gdal.GetDriverByName(fmt)
    out_ras = driver.Create(
        out_tif, cols, rows, 1, bit_dict[data_type], options=["COMPRESS=LZW"]
    )
    out_ras.SetProjection(proj)
    out_ras.SetGeoTransform(geotr)

    # Fill output with NoData
    out_ras.GetRasterBand(1).SetNoDataValue(ndv)
    out_ras.GetRasterBand(1).Fill(ndv)

    # 2. Rasterize shapefile
    shp_ds = ogr.Open(in_shp)
    shp_lyr = shp_ds.GetLayer()

    gdal.RasterizeLayer(
        out_ras, [1], shp_lyr, options=["ATTRIBUTE=%s" % attrib, "COMPRESS=LZW"]
    )

    # Flush and close
    snap_ras = None
    out_ras = None
    shp_ds = None


def reclassify_raster(in_tif, mask_tif, out_tif, reclass_df, reclass_col, ndv):
    """Reclassify categorical values in a raster using a mapping
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
    from osgeo import gdal, ogr
    import numpy as np
    import pandas as pd
    from osgeo.gdalconst import GA_ReadOnly as GA_ReadOnly

    # Open source file, read data
    src_ds = gdal.Open(in_tif, GA_ReadOnly)
    assert src_ds
    rb = src_ds.GetRasterBand(1)
    src_data = rb.ReadAsArray()

    # Open mask, read data
    mask_ds = gdal.Open(mask_tif, GA_ReadOnly)
    assert mask_ds
    mb = mask_ds.GetRasterBand(1)
    mask_data = mb.ReadAsArray()

    # Reclassify
    rc_data = src_data.copy()
    for idx, row in reclass_df.iterrows():
        rc_data[src_data == idx] = row[reclass_col]

    # Apply mask
    rc_data[mask_data != 1] = ndv

    # Write output
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.CreateCopy(out_tif, src_ds, 0, options=["COMPRESS=LZW"])
    out_band = dst_ds.GetRasterBand(1)
    out_band.SetNoDataValue(ndv)
    out_band.WriteArray(rc_data)

    # Flush data and close datasets
    dst_ds = None
    src_ds = None
    mask_ds = None


def calc_vegetation_exceedance(dep_tif, cl_tif, ex_tif, ex_tif_bool, ser_id):
    """Calculate exceedances for vegetation. 'dep_tif' and 'cl_tif' must have the same
    extent, cell size, CRS etc.

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
    from osgeo import gdal, ogr
    import nivapy3 as nivapy
    import numpy as np
    import pandas as pd

    # Container for output
    data_dict = {"total_area_km2": [], "exceeded_area_km2": []}

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
    cl_grid[cl_grid == cl_ndv] = np.nan
    dep_grid[dep_grid == dep_ndv] = np.nan

    # Get total area of non-NaN from dep grid
    nor_area = np.count_nonzero(~np.isnan(dep_grid)) * cs * cs / 1.0e6

    # Apply scaling factor to CLs
    cl_grid = cl_grid * 100.0

    # Exceedance
    ex_grid = dep_grid - cl_grid
    del dep_grid, cl_grid

    # Get total area exceeded
    ex_area = np.count_nonzero(ex_grid > 0) * cs * cs / 1.0e6

    # Set <0 to 0
    ex_grid[ex_grid < 0] = 0

    # Reset ndv
    ex_grid[np.isnan(ex_grid)] = -1

    # Downcast to int16 to save space
    ex_grid = ex_grid.round(0).astype(np.int16)

    # Append results
    data_dict["total_area_km2"].append(nor_area)
    data_dict["exceeded_area_km2"].append(ex_area)

    # Write exceedance output
    write_geotiff(ex_grid, ex_tif, cl_tif, -1, gdal.GDT_Int16)

    # Convert to bool grid
    ex_grid[ex_grid > 0] = 1
    ex_grid[ex_grid == -1] = 255

    # Write bool output
    write_geotiff(ex_grid, ex_tif_bool, cl_tif, 255, gdal.GDT_Byte)
    del ex_grid

    # Build output df
    ex_df = pd.DataFrame(data_dict)
    ex_df["exceeded_area_pct"] = (
        100 * ex_df["exceeded_area_km2"] / ex_df["total_area_km2"]
    )
    ex_df = ex_df.round(0).astype(int)
    ex_df["series_id"] = ser_id
    ex_df["medium"] = "vegetation"
    ex_df = ex_df[
        [
            "series_id",
            "medium",
            "total_area_km2",
            "exceeded_area_km2",
            "exceeded_area_pct",
        ]
    ]

    return ex_df


def write_geotiff(data, out_tif, snap_tif, ndv, data_type):
    """Write a numpy array to a geotiff using 'snap_tif' to define
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
    from osgeo import gdal, ogr

    # 1. Create new, empty raster with correct dimensions
    # Get properties from snap_tif
    snap_ras = gdal.Open(snap_tif)
    cols = snap_ras.RasterXSize
    rows = snap_ras.RasterYSize
    proj = snap_ras.GetProjection()
    geotr = snap_ras.GetGeoTransform()

    # Create out_tif
    driver = gdal.GetDriverByName("GTiff")
    out_ras = driver.Create(out_tif, cols, rows, 1, data_type, options=["COMPRESS=LZW"])
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
    """Helper function for zonal_stats(). Modified from:

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
    """Modified from https://gist.github.com/perrygeo/5667173
    Original code copyright 2013 Matthew Perry
    """

    def lookup(m, k):
        """Dict lookup but returns original key if not found"""
        try:
            return m[k]
        except KeyError:
            return k

    return {lookup(category_map, k): v for k, v in stats.items()}


def exceedance_stats_per_0_1deg_cell(
    ex_tif,
    ser_id,
    eng,
    write_to_db=True,
    nodata_value=-1,
    global_src_extent=False,
    categorical=False,
    category_map=None,
):
    """Summarise exceedance values for each 0.1 degree grid cell.

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
    import os
    import sys

    from osgeo import gdal, ogr
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from osgeo.gdalconst import GA_ReadOnly

    gdal.PushErrorHandler("CPLQuietErrorHandler")

    # Read vector
    temp_fold = os.path.split(ex_tif)[0]
    temp_shp = os.path.join(temp_fold, "temp.shp")
    gdf = extract_deposition_as_gdf(ser_id, "nitrogen", eng, veg_class="grid_average")
    gdf.to_file(temp_shp)

    # Read raster
    rds = gdal.Open(ex_tif, GA_ReadOnly)
    assert rds
    rb = rds.GetRasterBand(1)
    rgt = rds.GetGeoTransform()

    # Get cell size
    cs = rgt[1]

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(
        temp_shp, GA_ReadOnly
    )  # TODO maybe open update if we want to write stats
    assert vds
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
            rgt[5],
        )

    mem_drv = ogr.GetDriverByName("Memory")
    driver = gdal.GetDriverByName("MEM")

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
                rgt[5],
            )

        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource("out")
        mem_layer = mem_ds.CreateLayer("poly", None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())

        # Rasterize it
        rvds = driver.Create(
            "", src_offset[2], src_offset[3], 1, gdal.GDT_Byte, options=["COMPRESS=LZW"]
        )
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()

        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
        masked = np.ma.MaskedArray(
            src_array,
            mask=np.logical_or(src_array == nodata_value, np.logical_not(rv_array)),
        )

        if categorical:
            # Get cell counts for each category
            keys, counts = np.unique(masked.compressed(), return_counts=True)
            pixel_count = dict(
                zip([np.asscalar(k) for k in keys], [np.asscalar(c) for c in counts])
            )

            feature_stats = dict(pixel_count)
            if category_map:
                feature_stats = remap_categories(category_map, feature_stats)

        else:
            # Get summary stats
            feature_stats = {
                "min": float(masked.min()),
                "mean": float(masked.mean()),
                "max": float(masked.max()),
                "std": float(masked.std()),
                "sum": float(masked.sum()),
                "count": int(masked.count()),
                "fid": int(feat.GetFID()),
            }

        stats.append(feature_stats)

        rvds = None
        mem_ds = None
        feat = vlyr.GetNextFeature()

    # Tidy up
    vds = None
    rds = None
    for fname in ["temp.cpg", "temp.dbf", "temp.prj", "temp.shp", "temp.shx"]:
        os.remove(os.path.join(temp_fold, fname))

    # Combine results
    df = pd.DataFrame(stats)
    df.fillna(0, inplace=True)
    df["series_id"] = ser_id
    df["fid"] = df.index
    gdf["fid"] = gdf.index
    gdf = gdf.merge(df, on="fid")

    # Calc areas
    gdf["exceeded_area_km2"] = gdf["exceeded"] * cs * cs / 1e6
    gdf["total_area_km2"] = (gdf["exceeded"] + gdf["not_exceeded"]) * cs * cs / 1e6
    gdf["pct_exceeded"] = 100 * gdf["exceeded_area_km2"] / gdf["total_area_km2"]
    gdf.drop(
        [
            "fid",
            "exceeded",
            "not_exceeded",
            "ndep_mgpm2pyr",
            "ndep_meqpm2pyr",
            "ndep_kgphapyr",
        ],
        axis="columns",
        inplace=True,
    )
    gdf.dropna(how="any", inplace=True)

    if write_to_db:
        gdf2 = gdf.copy()
        del gdf2["geom"]
        df = pd.DataFrame(gdf2)
        df.to_sql(
            "exceedance_stats_0_1deg_grid",
            eng,
            "vegetation",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

    return gdf


def exceedance_stats_per_land_use_class(
    ex_tif_bool, veg_tif, ser_id, eng, write_to_db=True, nodata_value=255
):
    """Summarise exceedance values for each land use class.

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
    import os
    import sys

    from osgeo import gdal, ogr
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from osgeo.gdalconst import GA_ReadOnly

    gdal.PushErrorHandler("CPLQuietErrorHandler")

    # Read LU table
    sql = "SELECT * FROM vegetation.land_class_crit_lds"
    lu_df = pd.read_sql(sql, eng)

    # Read exceedance raster
    rds = gdal.Open(ex_tif_bool, GA_ReadOnly)
    assert rds
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
    assert rds
    rb = rds.GetRasterBand(1)

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    veg_array = rb.ReadAsArray()

    # Loop through land classes
    stats = []
    for idx, row in lu_df.iterrows():
        # Mask the source data array
        masked = np.ma.MaskedArray(
            ex_array,
            mask=np.logical_or(
                ex_array == nodata_value, veg_array != row["norut_code"]
            ),
        )

        # Get cell counts for each category
        keys, counts = np.unique(masked.compressed(), return_counts=True)
        pixel_count = dict(
            zip([np.asscalar(k) for k in keys], [np.asscalar(c) for c in counts])
        )

        feature_stats = dict(pixel_count)
        feature_stats = remap_categories(
            {1: "exceeded", 0: "not_exceeded"}, feature_stats
        )
        stats.append(feature_stats)

    # Tidy up
    rds = None

    # Combine results
    df = pd.DataFrame(stats)
    df.fillna(0, inplace=True)
    df["norut_code"] = lu_df["norut_code"]
    df["series_id"] = ser_id

    # Calc areas
    df["exceeded_area_km2"] = df["exceeded"] * cs * cs / 1e6
    df["total_area_km2"] = (df["exceeded"] + df["not_exceeded"]) * cs * cs / 1e6
    df["pct_exceeded"] = 100 * df["exceeded_area_km2"] / df["total_area_km2"]
    del df["exceeded"], df["not_exceeded"]
    df.dropna(how="any", inplace=True)

    if write_to_db:
        df.to_sql(
            "exceedance_stats_land_class",
            eng,
            "vegetation",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

    return df


def veg_exceedance_as_gdf_0_1deg(ser_id, eng, shp_path=None):
    """Extracts exceedance statistics for the specified series as a
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
    sql_args = {"ser_id": ser_id}
    sql = (
        "SELECT ST_Transform(b.geom, 32633) AS geom, "
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
        "WHERE a.cell_id = b.cell_id"
    ).format(**sql_args)
    gdf = gpd.read_postgis(sql, eng)

    if shp_path:
        gdf.to_file(shp_path)

    return gdf


def calc_anclimit_cla_clminmax(df, bc0):
    """Calculates the ANC Limit, the CLA and the CLmin and CLmax values using
        the specified version of BC0, both with and without a correction for organic
        acids).

        Used by calculate_critical_loads_for_water()

    Args:
        df:  Dataframe.
        bc0: Str. One of ['BC0', 'BC0_Ffac', 'BC0_magic']

    Returns:
        Dataframe with new columns for

            ['ANClimit_{bc0}',
             'ANClimitOAA_{bc0}',
             'CLA_{bc0}',
             'CLAOAA_{bc0}',
             'CLmaxS_{bc0}',
             'CLmaxSoaa_{bc0}',
             'CLmaxN_{bc0}',
             'CLmaxNoaa_{bc0}',
            ]
    """
    import numpy as np

    assert bc0 in [
        "BC0",
        "BC0_Ffac",
        "BC0_magic",
    ], "'bc0' must be one of ['BC0', 'BC0_Ffac', 'BC0_magic']."

    method = bc0.replace("BC0", "")

    # ANC limit
    df[f"ANClimit{method}"] = np.minimum(
        50, (0.25 * df["Runoff"] * df[bc0]) / (1 + 0.25 * df["Runoff"])
    )

    # ANC limit OAA
    df[f"ANClimitOAA{method}"] = np.minimum(
        40,
        (0.2 * df["Runoff"] * (df[bc0] - 3.4 * df["TOC"])) / (1 + 0.2 * df["Runoff"]),
    )

    # CLA
    df[f"CLA{method}"] = df["Runoff"] * (df[bc0] - df[f"ANClimit{method}"])

    # CLA OAA
    df[f"CLAOAA{method}"] = df["Runoff"] * (
        df[bc0] - df[f"ANClimitOAA{method}"] - 3.4 * df["TOC"]
    )

    # CLmaxS
    df[f"CLmaxS{method}"] = df[f"CLA{method}"] / df["alphaS"]

    # CLmaxSoaa
    df[f"CLmaxSoaa{method}"] = df[f"CLAOAA{method}"] / df["alphaS"]

    # CLmaxN
    df[f"CLmaxN{method}"] = df["CLminN"] + (df[f"CLA{method}"] / df["alphaN"])

    # CLmaxNoaa
    df[f"CLmaxNoaa{method}"] = df["CLminN"] + (df[f"CLAOAA{method}"] / df["alphaN"])

    return df


def calculate_critical_loads_for_water(
    xl_path=None, req_df=None, opt_df=None, mag_df=None, out_csv=None
):
    """Calculates critical loads for water based on values entered in an
        Excel template (input_template_critical_loads_water.xlsx) or from
        the database. See the Excel file for full details of the input
        data requirements.

        You must provide EITHER the 'xl_path' OR the three separate
        dataframes - NOT BOTH.

        This function performs broadly the same calculations as Tore's 'CL'
        and 'CALKBLR' packages in RESA2, but generalised to allow for more
        flexible input data. The original critical loads calculations were
        implemented by Tore's 'cl.clcalculations' function, which has been
        documented by Kari:

        K:\Avdeling\317 Klima- og miljømodellering\KAU\Focal Centre\Data\CL script 23032015_notes.docx

        These notes form the basis for much of the code here.

    Args:
        xl_path: Str. Path to completed copy the Excel input template
        req_df:  Dataframe of required parameters
        opt_df:  Dataframe of optional parameters
        mag_df:  Dataframe of magic parameters
        out_csv: Raw str. The final dataframe is saved to the specified path

    Returns:
        Dataframe.
    """
    import numpy as np
    import pandas as pd

    # Check input
    if xl_path and (req_df is not None or mag_df is not None or opt_df is not None):
        message = (
            "ERROR: You must provide EITHER the 'xl_path' OR the three "
            "separate dataframes - NOT both."
        )
        print(message)
        raise ValueError(message)

    if xl_path and (req_df is None and mag_df is None and opt_df is None):
        # Read worksheets
        req_df = pd.read_excel(xl_path, sheet_name="required_parameters")
        opt_df = pd.read_excel(xl_path, sheet_name="optional_parameters")
        mag_df = pd.read_excel(xl_path, sheet_name="magic_parameters")

    # Dicts of constants used in script
    # 1. Default values for parameters
    default_vals = {
        "Catch_area": 1,
        "Lake_area": 0.05,
        "Forest_area": 0.95,
        "Bare_area": 0,
        "Peat_area": 0,
        "Ni": 3.57,
        "Fde": 0.1,
        "Fde_peat": 0.8,
        "SN": 5,
        "SS": 0.5,
        "TOC": 1,
        "K": 0,
        "Na": 0,
    }

    # 2. Unit conversions
    units_dict = {
        "Runoff": 1e-3,  # mm/yr => m/yr
        "Ca": 2 * 1000 / 40.08,  # mg/l  => ueq/l
        "Cl": 1 * 1000 / 35.45,  # mg/l  => ueq/l
        "Mg": 2 * 1000 / 24.31,  # mg/l  => ueq/l
        "Na": 1 * 1000 / 22.99,  # mg/l  => ueq/l
        "SO4": 2 * 1000 / 96.06,  # mg/l  => ueq/l
        "NO3N": 1 / 14.01,  # ug/l  => ueq/l
        "K": 1 * 1000 / 39.10,  # mg/l  => ueq/l
    }

    # 3. Ratios to chloride
    cl_ratios = {
        "Ca": 0.037,
        "Mg": 0.196,
        "Na": 0.859,
        "SO4": 0.103,
        "K": 0.018,
    }

    # Check region ID is unique
    assert req_df[
        "Region_id"
    ].is_unique, "'Region_id' is not unique within worksheet 'required_parameters'."
    assert opt_df[
        "Region_id"
    ].is_unique, "'Region_id' is not unique within worksheet 'optional_parameters'."
    assert mag_df[
        "Region_id"
    ].is_unique, "'Region_id' is not unique within worksheet 'magic_parameters'."

    # Join
    df = pd.merge(req_df, opt_df, how="left", on="Region_id")
    df = pd.merge(df, mag_df, how="left", on="Region_id")

    # Fill NaNs in params with defaults
    for col in default_vals.keys():
        df[col].fillna(default_vals[col], inplace=True)

    # Convert units
    for col in units_dict.keys():
        df[col] = df[col] * units_dict[col]

    # Apply sea-salt correction for required pars
    for col in cl_ratios.keys():
        df[col] = df[col] - (df["Cl"] * cl_ratios[col])
        df.loc[df[col] < 0, col] = 0  # Set negative values to zero

    # Apply sea-salt correction for Magic pars
    for col in ["Ca_magic", "Mg_magic", "Na_magic", "K_magic"]:
        par = col.split("_")[0]
        df[col] = df[col] - (df["Cl_magic"] * cl_ratios[par])
        df.loc[df[col] < 0, col] = 0  # Set negative values to zero

    # Nitrate flux
    df["ENO3_flux"] = df["Runoff"] * df["NO3N"]

    # Land cover proportions
    df["CLrat"] = df["Lake_area"] / df["Catch_area"]
    df["Ffor"] = df["Forest_area"] / df["Catch_area"]
    df["Fbare"] = df["Bare_area"] / df["Catch_area"]
    df["Fpeat"] = df["Peat_area"] / df["Catch_area"]

    # 'Nimm' is the long-term annual immobilisation (accumulation) rate of N
    # Applies to all areas that are not 'lake' or 'bare'
    df["Nimm"] = df["Ni"] * (1 - df["CLrat"] - df["Fbare"])

    # If 'Lake_area' is 0, SN and SS should be zero, else use defaults of
    # 5 and 0.5, respectively
    df.loc[df["CLrat"] == 0, "SN"] = 0
    df.loc[df["CLrat"] == 0, "SS"] = 0

    # Present-day sum of sea-salt corrected base cation concentrations
    # NB: K was not included in the original workflow before 20.3.2015
    df["BCt"] = df["Ca"] + df["Mg"] + df["Na"] + df["K"]

    # Calculate BC0 using F-Factor method
    # This was used before 2005 - 2006, but is not relevant at present
    df["SO40"] = 3 + 0.17 * df["BCt"]
    df["Ffac"] = np.sin((np.pi / 2) * df["Runoff"] * df["BCt"] * (1 / 400))
    df["BC0_Ffac"] = df["BCt"] - (df["Ffac"] * (df["SO4"] - df["SO40"] + df["NO3N"]))

    # Calculate BC0 using regression
    # This is the current approach. Note the following:
    #     - Prior to 20.3.2015, the equation BC0 = 0.1936 + 1.0409*BCt was used
    #       This is incorrect - the x and y axes were swapped
    #     - The correct equation is BC0 = 0.9431*BCt + 0.2744. Note, however, that
    #       the intercept term is not significant and should probably be omitted
    df["BC0"] = 0.9431 * df["BCt"] + 0.2744

    # Calculate BC0 from MAGIC output (if provided)
    df["BC0_magic"] = df["Ca_magic"] + df["Mg_magic"] + df["Na_magic"] + df["K_magic"]

    # Lake retention factors for N and S
    df["rhoS"] = df["SS"] / (df["SS"] + (df["Runoff"] / df["CLrat"]))
    df["rhoN"] = df["SN"] / (df["SN"] + (df["Runoff"] / df["CLrat"]))
    df.loc[df["CLrat"] == 0, "rhoS"] = 0  # If 'CLrat' is 0, rhoS is 0
    df.loc[df["CLrat"] == 0, "rhoN"] = 0  # If 'CLrat' is 0, rhoN is 0

    # Lake transmission factors. For N, takes into account what is lost to denitrification
    # before the N reaches the lake
    df["alphaS"] = 1 - df["rhoS"]
    df["alphaN"] = (
        1
        - (df["Fde"] * (1 - df["CLrat"] - df["Fpeat"] - df["Fbare"]))
        - (df["Fde_peat"] * df["Fpeat"])
    ) * (1 - df["rhoN"])

    # beta1 is the fraction of N available for uptake by the forest
    df["beta1"] = df["Ffor"] * (1 - df["Fde"]) * (1 - df["rhoN"])

    # beta2 the fraction of N available for immobilisation
    df["beta2"] = (
        ((1 - df["Fde"]) * (1 - df["CLrat"] - df["Fbare"] - df["Fpeat"]))
        + ((1 - df["Fde_peat"]) * df["Fpeat"])
    ) * (1 - df["rhoN"])

    # Calculate ANC limits, CLAs and CLmax & CLmin (using BC0, BC0_Ffac and BC0_magic, both with and without
    # corrections for organic acids)
    df["CLminN"] = ((df["beta1"] * df["Nupt"]) + (df["beta2"] * df["Nimm"])) / df[
        "alphaN"
    ]

    for bc0_method in ["BC0", "BC0_Ffac", "BC0_magic"]:
        df = calc_anclimit_cla_clminmax(df, bc0_method)

    # Rename columns to reflect unit and sea-salt corrections
    df.rename(
        {
            "Ca": "ECax",
            "Cl": "ECl",
            "Mg": "EMgx",
            "Na": "ENax",
            "SO4": "ESO4x",
            "NO3N": "ENO3",
            "K": "EKx",
        },
        inplace=True,
        axis="columns",
    )

    # Columns of interest for output (with units)
    col_dict = {
        "Nimm": "meq/m2/yr",
        "Nupt": "meq/m2/yr",
        "rhoN": "",
        "Ffor": "",
        "CLrat": "",
        "BC0": "ueq/l",
        "BC0_Ffac": "ueq/l",
        "BC0_magic": "ueq/l",
        "ANClimit": "ueq/l",
        "ANClimitOAA": "ueq/l",
        "ANClimit_Ffac": "ueq/l",
        "ANClimitOAA_Ffac": "ueq/l",
        "ANClimit_magic": "ueq/l",
        "ANClimitOAA_magic": "ueq/l",
        "CLA": "meq/m2/yr",
        "CLAOAA": "meq/m2/yr",
        "CLA_Ffac": "meq/m2/yr",
        "CLAOAA_Ffac": "meq/m2/yr",
        "CLA_magic": "meq/m2/yr",
        "CLAOAA_magic": "meq/m2/yr",
        "CLminN": "meq/m2/yr",
        "CLmaxN": "meq/m2/yr",
        "CLmaxNoaa": "meq/m2/yr",
        "CLmaxN_Ffac": "meq/m2/yr",
        "CLmaxNoaa_Ffac": "meq/m2/yr",
        "CLmaxN_magic": "meq/m2/yr",
        "CLmaxNoaa_magic": "meq/m2/yr",
        "CLmaxS": "meq/m2/yr",
        "CLmaxSoaa": "meq/m2/yr",
        "CLmaxS_Ffac": "meq/m2/yr",
        "CLmaxSoaa_Ffac": "meq/m2/yr",
        "CLmaxS_magic": "meq/m2/yr",
        "CLmaxSoaa_magic": "meq/m2/yr",
        "Runoff": "m/yr",
        "ENO3_flux": "meq/m2/yr",
        "ECax": "ueq/l",
        "ECl": "ueq/l",
        "EMgx": "ueq/l",
        "ENax": "ueq/l",
        "ESO4x": "ueq/l",
        "ENO3": "ueq/l",
        "EKx": "ueq/l",
        "TOC": "mg/l",
    }

    df = df[["Region_id"] + list(col_dict.keys())]
    cols_units = ["Region_id"] + [
        f"{i}_{col_dict[i]}" for i in col_dict.keys()
    ]  # Add units to header
    df.columns = cols_units

    if out_csv:
        df.to_csv(out_csv, index=False)

    return df


def rasterise_water_critical_loads(
    eng,
    out_fold,
    cell_size=120,
    bc0="BC0",
    req_df=None,
    opt_df=None,
    mag_df=None,
    df_to_csv=False,
):
    """Creates rasters of key critical loads parameters:

            'claoaa', 'eno3', 'clminn', 'clmaxnoaa', 'clmaxsoaa', 'clmins',
            'anclimit', 'anclimit_oaa', 'bc0'

        using the specified water chemistry, model parameters per BLR grid square, and BC0 method.

    Args:
        eng:       Obj. Valid connection object for the 'critical_loads' database
        out_fold:  Raw str. Folder in which to save grids
        cell_size: Int. Resolution of output rasters
        bc0:       Str. BC0 method to use. One of ['BC0', 'BC0_Ffac', 'BC0_magic']
        req_df:    Dataframe of required parameters or None
        opt_df:    Dataframe of optional parameters or None
        mag_df:    Dataframe of magic parameters or None
        df_to_csv: Bool. If True, the dataframe of critical loads is also written to
                   'out_fold/critical_loads.csv'

    Returns:
        None. The rasters are written to the specified folder.
    """
    import os

    import geopandas as gpd
    import nivapy3 as nivapy
    import pandas as pd

    assert bc0 in [
        "BC0",
        "BC0_Ffac",
        "BC0_magic",
    ], "'bc0' must be one of ['BC0', 'BC0_Ffac', 'BC0_magic']."

    if (mag_df is not None) and (bc0 != "BC0_magic"):
        print(
            f"WARNING: You have supplied 'mag_df', but BC0 is '{bc0}'. Magic parameters will be ignored.\n"
            "         Are you sure you don't mean 'BC0_magic' instead?\n"
        )

    if (bc0 == "BC0_magic") and (mag_df is None):
        raise ValueError(
            "To use 'BC0_magic' you must supply 'mag_df'. This probably means filling in the Excel template."
        )

    # Read parameters table from db
    par_df = pd.read_sql(
        "SELECT id as parameter_id, name, class FROM water.parameter_definitions",
        eng,
    )

    if req_df is None:
        # Read data from db
        req_df = pd.read_sql("SELECT * FROM water.blr_required_parameters", eng)

        # Restructure
        req_df = pd.merge(req_df, par_df, how="left", on="parameter_id")
        del req_df["parameter_id"], req_df["class"]
        req_df = req_df.pivot(index="region_id", columns="name", values="value")
        req_df.index.name = "Region_id"
        req_df.reset_index(inplace=True)
        req_df.columns.name = ""

    if opt_df is None:
        # Create empty dataframe with correct cols 'optional' parameters
        # (There are not used in the calculations below, but are expected by the CL function)
        opt_cols = list(par_df[par_df["class"] == "optional"]["name"].values)
        opt_df = pd.DataFrame(columns=["Region_id"] + opt_cols)

    if mag_df is None:
        # Create empty dataframe with correct cols 'magic' parameters
        # (There are not used in the calculations below, but are expected by the CL function)
        mag_cols = list(par_df[par_df["class"] == "magic"]["name"].values)
        mag_df = pd.DataFrame(columns=["Region_id"] + mag_cols)

    # Calculate critical loads
    if df_to_csv:
        out_csv = os.path.join(out_fold, "critical_loads.csv")
        cl_df = calculate_critical_loads_for_water(
            req_df=req_df, opt_df=opt_df, mag_df=mag_df, out_csv=out_csv
        )
    else:
        cl_df = calculate_critical_loads_for_water(
            req_df=req_df, opt_df=opt_df, mag_df=mag_df
        )

    # Get just cols of interest
    bc0_dict = {"BC0": "", "BC0_Ffac": "FFac_", "BC0_magic": "magic_"}
    bc0 = bc0_dict[bc0]
    cols = [
        "Region_id",
        f"CLAOAA_{bc0}meq/m2/yr",
        "ENO3_flux_meq/m2/yr",
        "CLminN_meq/m2/yr",
        f"CLmaxNoaa_{bc0}meq/m2/yr",
        f"CLmaxSoaa_{bc0}meq/m2/yr",
        f"ANClimit_{bc0}ueq/l",
        f"ANClimitOAA_{bc0}ueq/l",
        f"BC0_{bc0}ueq/l",
    ]
    cl_df = cl_df[cols]
    cols = [i.replace("/", "p").lower() for i in cols]
    cl_df.columns = cols

    cl_df.dropna(how="any", inplace=True)
    cl_df.rename({"region_id": "cell_id"}, axis="columns", inplace=True)

    # Add CLminS as 0
    cl_df["clmins_meqpm2pyr"] = 0

    # Join to BLR spatial data
    blr_gdf = nivapy.da.read_postgis("deposition", "dep_grid_blr", eng)
    blr_gdf = blr_gdf[["cell_id", "geom"]]
    blr_gdf = blr_gdf.merge(cl_df, on="cell_id")

    # Save temporary file
    blr_gdf.to_file("temp.geojson", driver="GeoJSON")

    # Snap tiff
    snap_tif = (
        f"/home/jovyan/shared/critical_loads/raster/blr_land_mask_{cell_size}m.tif"
    )

    # Rasterize columns
    cols.remove("region_id")
    cols.append("clmins_meqpm2pyr")
    for col in cols:
        print(f"Rasterising {col}...")
        # Tiff to create
        out_tif = os.path.join(out_fold, f"{col}_{cell_size}m.tif")
        vec_to_ras("temp.geojson", out_tif, snap_tif, col, -9999, "Float32")

    # Delete temp file
    os.remove("temp.geojson")
    print("Rasters saved to:")
    print(f"    {out_fold}")


def calculate_water_exceedance_sswc(
    ser_id, short_name, cl_fold, out_fold, cell_size=120, bc0="BC0", neg_to_zero=True
):
    """Calculate exceedances for water using the SSWC model.

    Args:
        ser_id:      Int. Series ID for deposition data
        short_name:  Str. Used in naming output exceedance grid. e.g. '11-16'
                     for 2011 to 2016 data series
        cl_fold:     Str. Folder containing rasters of critical loads (from rasterise_water_critical_loads)
        out_fold:    Str. Folder in which to create output raster
        cell_size:   Int. Resolution of output rasters
        bc0:         Str. BC0 method to use. One of ['BC0', 'BC0_Ffac', 'BC0_magic']
        neg_to_zero: Bool. Whether to set negative values of exceedance to zero

    Returns:
        Dataframe summarising exceedances for water.
    """
    import os

    from osgeo import gdal, ogr
    import nivapy3 as nivapy
    import numpy as np
    import pandas as pd

    assert bc0 in [
        "BC0",
        "BC0_Ffac",
        "BC0_magic",
    ], "'bc0' must be one of ['BC0', 'BC0_Ffac', 'BC0_magic']."

    # Snap tiff
    snap_tif = (
        f"/home/jovyan/shared/critical_loads/raster/blr_land_mask_{cell_size}m.tif"
    )

    # Read grids
    s_tif = f"/home/jovyan/shared/critical_loads/raster/deposition/sdep_mgpm2pyr_{short_name}_{cell_size}m.tif"
    s_dep, s_ndv, epsg, extent = nivapy.spatial.read_raster(s_tif)
    s_dep = s_dep.astype(np.float32)

    eno3_tif = os.path.join(cl_fold, f"eno3_flux_meqpm2pyr_{cell_size}m.tif")
    eno3fl, eno3_ndv, epsg, extent = nivapy.spatial.read_raster(eno3_tif)

    bc0_dict = {"BC0": "", "BC0_Ffac": "FFac_", "BC0_magic": "magic_"}
    bc0 = bc0_dict[bc0]
    claoaa_tif = os.path.join(cl_fold, f"claoaa_{bc0}meqpm2pyr_{cell_size}m.tif")

    if not os.path.exists(claoaa_tif):
        print(f"WARNING: Could not find file:")
        print(f"    {claoaa_tif}")
        print("Make sure the 'bc0' arg is set correctly.")

    claoaa, cla_ndv, epsg, extent = nivapy.spatial.read_raster(claoaa_tif)

    # Set ndv
    s_dep[s_dep == s_ndv] = np.nan
    eno3fl[eno3fl == eno3_ndv] = np.nan
    claoaa[claoaa == cla_ndv] = np.nan

    # Set negative to zero
    claoaa[claoaa < 0] = 0

    # Convert dep to meq
    s_dep = s_dep * 2 / 32.06

    # Get total area of non-NaN from dep grid
    nor_area = np.count_nonzero(~np.isnan(s_dep)) * cell_size * cell_size / 1.0e6

    # Exceedance
    sswc_ex = s_dep + eno3fl - claoaa
    del s_dep, eno3fl, claoaa

    # Get total area exceeded
    ex_area = np.count_nonzero(sswc_ex > 0) * cell_size * cell_size / 1.0e6

    if neg_to_zero:
        # Set <0 to 0
        sswc_ex[sswc_ex < 0] = 0

    # Write geotif
    sswc_tif = os.path.join(
        out_fold, f"sswc_ex_meqpm2pyr_{short_name}_{cell_size}m.tif"
    )
    write_geotiff(sswc_ex, sswc_tif, snap_tif, -1, gdal.GDT_Float32)
    del sswc_ex

    print("Exceedance grid saved to:")
    print(f"    {sswc_tif}")

    # Build df
    ex_pct = 100 * ex_area / nor_area
    ex_df = pd.DataFrame(
        {
            "exceeded_area_km2": ex_area,
            "total_area_km2": nor_area,
            "exceeded_area_pct": ex_pct,
        },
        index=[0],
    )
    ex_df = ex_df.round(0).astype(int)
    ex_df["series_id"] = ser_id
    ex_df["medium"] = "water_sswc"

    ex_df = ex_df[
        [
            "series_id",
            "medium",
            "total_area_km2",
            "exceeded_area_km2",
            "exceeded_area_pct",
        ]
    ]

    return ex_df


def exceed_ns_icpm(cln_min, cln_max, cls_min, cls_max, dep_n, dep_s):
    """Calculates exceedances based on the methodology outlined by Max
        Posch in the ICP Mapping manual (section VII.4):

        https://www.umweltbundesamt.de/sites/default/files/medien/4292/dokumente/ch7-mapman-2016-04-26.pdf

        NB: All units should be in eq/l.

    Args:
        cln_min: Float. Parameter to define "critical load function" (see PDF)
        cln_max: Float. Parameter to define "critical load function" (see PDF)
        cls_min: Float. Parameter to define "critical load function" (see PDF)
        cls_max: Float. Parameter to define "critical load function" (see PDF)
        dep_n:   Float. Total N deposition
        dep_s:   Float. Total (non-marine) S deposition

    Returns:
        Tuple (ex_n, ex_s, reg_id)

        ex_n and ex_s are the exceedances for N and S depositions dep_n and dep_s
        and the CLF defined by (cln_min, cls_max) and (cln_max, cls_min). The
        overall exceedance is (ex_n + ex_s).

        reg_id is an integer region ID, as defined in Figure VII.3 of the PDF.
    """
    # Check inputs
    assert (dep_n >= 0) and (dep_s >= 0), "Deposition cannot be negative."

    # Make sure floats
    cln_min = float(cln_min)
    cln_max = float(cln_max)
    cls_min = float(cls_min)
    cls_max = float(cls_max)
    dep_n = float(dep_n)
    dep_s = float(dep_s)

    # Handle edge cases
    # CLF pars < 0
    if (cln_min < 0) or (cln_max < 0) or (cls_min < 0) or (cls_max < 0):
        # Pars not valid
        # Updated 07.11.2020. Values < 0  do not make sense, so were originally set to -1.
        # This change is equivalent to setting values less than zero back to zero
        # return (-1, -1, -1)
        return (dep_n, dep_s, -1)

    # CL = 0
    if (cls_max == 0) and (cln_max == 0):
        # All dep is above CL
        return (dep_n, dep_s, 9)

    # Otherwise, we're somewhere on Fig. VII.3
    dn = cln_min - cln_max
    ds = cls_max - cls_min

    if (
        (dep_s <= cls_max)
        and (dep_n <= cln_max)
        and ((dep_n - cln_max) * ds <= (dep_s - cls_min) * dn)
    ):
        # Non-exceedance
        return (0, 0, 0)

    elif dep_s <= cls_min:
        # Region 1
        ex_s = 0
        ex_n = dep_n - cln_max

        return (ex_n, ex_s, 1)

    elif dep_n <= cln_min:
        # Region 5
        ex_s = dep_s - cls_max
        ex_n = 0

        return (ex_n, ex_s, 5)

    elif -(dep_n - cln_max) * dn >= (dep_s - cls_min) * ds:
        # Region 2
        ex_n = dep_n - cln_max
        ex_s = dep_s - cls_min

        return (ex_n, ex_s, 2)

    elif -(dep_n - cln_min) * dn <= (dep_s - cls_max) * ds:
        # Region 4
        ex_n = dep_n - cln_min
        ex_s = dep_s - cls_max

        return (ex_n, ex_s, 4)

    else:
        # Region 3
        dd = dn**2 + ds**2
        s = dep_n * dn + dep_s * ds
        v = cln_max * ds - cls_min * dn
        xf = (dn * s + ds * v) / dd
        yf = (ds * s - dn * v) / dd
        ex_n = dep_n - xf
        ex_s = dep_s - yf

        return (ex_n, ex_s, 3)


def vectorised_exceed_ns_icpm(cln_min, cln_max, cls_min, cls_max, dep_n, dep_s):
    """Vectorised version of exceed_ns_icpm(). Calculates exceedances based on
        the methodology outlined by Max Posch in the ICP Mapping manual (section
        VII.4):

        https://www.umweltbundesamt.de/sites/default/files/medien/4292/dokumente/ch7-mapman-2016-04-26.pdf

        NB: All units should be in meq/l.

    Args:
        cln_min: Float array. Parameter to define "critical load function" (see PDF)
        cln_max: Float array. Parameter to define "critical load function" (see PDF)
        cls_min: Float array. Parameter to define "critical load function" (see PDF)
        cls_max: Float array. Parameter to define "critical load function" (see PDF)
        dep_n:   Float array. Total N deposition
        dep_s:   Float array. Total (non-marine) S deposition

    Returns:
        Tuple of arrays (ex_n, ex_s, reg_id)

        ex_n and ex_s are the exceedances for N and S depositions dep_n and dep_s
        and the CLF defined by (cln_min, cls_max) and (cln_max, cls_min). The
        overall exceedance is (ex_n + ex_s).

        reg_id is an integer array of region IDs, as defined in Figure VII.3 of the PDF.
    """
    import numpy as np

    # Create NaN arrays for output with correct dimensions
    ex_n = np.full(shape=dep_s.shape, fill_value=np.nan)
    ex_s = np.full(shape=dep_s.shape, fill_value=np.nan)
    reg_id = np.full(shape=dep_s.shape, fill_value=np.nan)

    # Handle edge cases
    # CLF pars < 0
    mask = (cln_min < 0) | (cln_max < 0) | (cls_min < 0) | (cls_max < 0)
    # Updated 07.11.2020. Values < 0  do not make sense, so were originally set to NaN. This change is
    # equivalent to setting values less than zero back to zero
    # ex_n[mask] = np.nan
    # ex_s[mask] = np.nan
    ex_n[mask] = dep_n[mask]
    ex_s[mask] = dep_s[mask]
    reg_id[mask] = -1
    edited = mask.copy()  # Keep track of edited cells so we don't change them again
    # This is analagous to the original 'if' statement in
    # exceed_ns_icpm(), which requires the logic to be
    # implemented in a specific order i.e. once a cell has been
    # edited we do not want to change it again (just like once
    # the 'if' evaluates to True, we don't proceed any further)

    # CL = 0
    mask = (cls_max == 0) & (cln_max == 0) & (edited == 0)
    ex_n[mask] = dep_n[mask]
    ex_s[mask] = dep_s[mask]
    reg_id[mask] = 9
    edited += mask

    # Otherwise, we're somewhere on Fig. VII.3
    dn = cln_min - cln_max
    ds = cls_max - cls_min

    # Non-exceedance
    mask = (
        (dep_s <= cls_max)
        & (dep_n <= cln_max)
        & ((dep_n - cln_max) * ds <= (dep_s - cls_min) * dn)
        & (edited == 0)
    )
    ex_n[mask] = 0
    ex_s[mask] = 0
    reg_id[mask] = 0
    edited += mask

    # Region 1
    mask = (dep_s <= cls_min) & (edited == 0)
    ex_n[mask] = dep_n[mask] - cln_max[mask]
    ex_s[mask] = 0
    reg_id[mask] = 1
    edited += mask

    # Region 5
    mask = (dep_n <= cln_min) & (edited == 0)
    ex_s[mask] = dep_s[mask] - cls_max[mask]
    ex_n[mask] = 0
    reg_id[mask] = 5
    edited += mask

    # Region 2
    mask = (-(dep_n - cln_max) * dn >= (dep_s - cls_min) * ds) & (edited == 0)
    ex_n[mask] = dep_n[mask] - cln_max[mask]
    ex_s[mask] = dep_s[mask] - cls_min[mask]
    reg_id[mask] = 2
    edited += mask

    # Region 4
    mask = (-(dep_n - cln_min) * dn <= (dep_s - cls_max) * ds) & (edited == 0)
    ex_n[mask] = dep_n[mask] - cln_min[mask]
    ex_s[mask] = dep_s[mask] - cls_max[mask]
    reg_id[mask] = 4
    edited += mask

    # Region 3 (anything not already edited)
    dd = dn**2 + ds**2
    s = dep_n * dn + dep_s * ds
    v = cln_max * ds - cls_min * dn
    xf = (dn * s + ds * v) / dd
    yf = (ds * s - dn * v) / dd
    ex_n[~edited] = dep_n[~edited] - xf[~edited]
    ex_s[~edited] = dep_s[~edited] - yf[~edited]
    reg_id[~edited] = 3

    del mask, edited, dd, s, v, xf, yf, dn, ds

    return (ex_n, ex_s, reg_id)


def read_orig_blr_data(eng, dropna=False):
    """Reads the original BLR data for water critical loads from the
    database.

    Args
        eng:    Obj. Database engine object
        dropna: Bool. Default False. Whether to drop rows containing NaN

    Returns
        Tuple of dataframe (req_df, opt_df, mag_df). The last two are
        blank, but have the ocrrect column headings for adding new data,
        if desired.
    """
    import pandas as pd

    # Read parameters table from db
    par_df = pd.read_sql(
        "SELECT id as parameter_id, name, class FROM water.parameter_definitions",
        eng,
    )

    # Read data from db
    req_df = pd.read_sql("SELECT * FROM water.blr_required_parameters", eng)

    # Restructure
    req_df = pd.merge(req_df, par_df, how="left", on="parameter_id")
    del req_df["parameter_id"], req_df["class"]
    req_df = req_df.pivot(index="region_id", columns="name", values="value")
    req_df.index.name = "Region_id"
    req_df.reset_index(inplace=True)
    req_df.columns.name = ""

    # Create empty dataframe with correct cols 'optional' parameters
    # (There are not used in the calculations below, but are expected by the CL function)
    opt_cols = list(par_df[par_df["class"] == "optional"]["name"].values)
    opt_df = pd.DataFrame(columns=["Region_id"] + opt_cols)

    # Create empty dataframe with correct cols 'magic' parameters
    # (There are not used in the calculations below, but are expected by the CL function)
    mag_cols = list(par_df[par_df["class"] == "magic"]["name"].values)
    mag_df = pd.DataFrame(columns=["Region_id"] + mag_cols)

    if dropna:
        req_df.dropna(how="any", inplace=True)
        opt_df.dropna(how="any", inplace=True)
        mag_df.dropna(how="any", inplace=True)

    return (req_df, opt_df, mag_df)


def update_required_pars(orig_req_df, new_blr_df, st_per, end_per):
    """Update values in 'orig_req_df' with those from 'new_blr_df'. Values
    for TOC and runoff are taken directly from 'new_blr_df'. For NO3, the
    original values are kept for periods before 1992, otherwise a new series
    is calculated as the mean over the 5-year period from 'per_st' to 'per_end'.

    Args
        orig_req_df: Dataframe. Original data from database
        new_blr_df:  Dataframe. Updated values for each BLR
        st_per:      Int. Year at start of period
        end_per:     Int. Year at end of period

    Returns
        Dataframe with updated values.
    """
    import pandas as pd

    req_df = orig_req_df.copy()
    blr_df = new_blr_df.copy()

    cols = ["Region_id", "runoff_mmpyr", "TOC_mgpl_2019"]
    df = blr_df[cols].copy()

    # Use old BLR NO3 before 1992, otherwise 5-year mean from new data
    if st_per >= 1992:
        df["NO3_NO2_ugpl"] = blr_df[
            [f"NO3_NO2_ugpl_{yi}" for yi in range(st_per, end_per + 1)]
        ].mean(axis="columns")

    # Rename to match template
    df.rename(
        {
            "runoff_mmpyr": "Runoff",
            "TOC_mgpl_2019": "TOC",
            "NO3_NO2_ugpl": "NO3N",
        },
        axis="columns",
        inplace=True,
        errors="ignore",
    )

    # Update req_df with new cols
    to_drop = [col for col in df.columns if col != 'Region_id']
    req_df.drop(to_drop, axis="columns", inplace=True)
    req_df = pd.merge(req_df, df, how="left", on="Region_id")

    return req_df


def estimate_optional_pars(req_df, new_blr_df):
    """Add values for 'opt_df' based on data in 'new_blr_df'.

    Args
        req_df:      Dataframe. Required parameters for BLRs of
                     interest
        new_blr_df:  Dataframe. Updated values for each BLR

    Returns
        Dataframe in format required for 'opt_df'.
    """
    import numpy as np
    import pandas as pd

    req_df = req_df[["Region_id"]].copy()
    opt_df = new_blr_df[["Region_id"]].copy()

    # Use new values for land cover
    opt_df["Catch_area"] = new_blr_df["Total_area_km2"]
    opt_df["Lake_area"] = new_blr_df["Lake_area_km2"]
    opt_df["Forest_area"] = new_blr_df["Forest_area_km2"]
    opt_df["Bare_area"] = new_blr_df["Bare_area_km2"]
    opt_df["Peat_area"] = new_blr_df["Peat_area_km2"]

    # Add remaining columns as NaN (i.e. use defaults from database)
    opt_df["Ni"] = np.nan
    opt_df["Fde"] = np.nan
    opt_df["Fde_peat"] = np.nan
    opt_df["SN"] = np.nan
    opt_df["SS"] = np.nan

    # Join to match 'req_df'
    opt_df = pd.merge(req_df, opt_df, how="left", on="Region_id")

    return opt_df
