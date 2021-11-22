import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString


def read_data(pars, keep_unknown_ends=True):
    fire_stations = _read_geo_files_into_geopandas(pars['fire_stations'], pars['crs'])
    fire_stations=fire_stations[~(fire_stations['id'].isnull())]
    fire_stations['id'] = pd.to_numeric(fire_stations['id'], downcast='unsigned')
    sbws = _read_geo_files_into_geopandas(pars['sbws'], pars['crs'])
    sbws = sbws[sbws['GTYPE'] == 'P']
    sbws = _geopandas_fix_datetime(sbws,
                                   cols=['ISSUED', 'EXPIRED', 'INIT_ISS', 'INIT_EXP'],
                                   fmt='%Y%m%d%H%M%S')

    # lsrs = _read_geo_files_into_geopandas(pars['lsrs'], pars['crs'])
    # lsrs = _geopandas_fix_datetime(lsrs,
    #                                  cols=['VALID'],
    #                                  fmt='%Y%m%d%H%M%S')

    waypoints = pd.read_csv(pars['waypoints_file'][0])
    waypoints = list(tuple(x) for x in waypoints.to_records(index=False))
    tornado_db = read_tornado_file(pars)
    return fire_stations, sbws, waypoints, tornado_db


def read_tornado_file(pars, keep_unknown_ends=True):
    tornado_db = pd.read_csv(pars['tornado_db'][0])
    if keep_unknown_ends:
        tornado_db.elon = tornado_db.apply(lambda x: x.slon + 0.001, axis=1)
        tornado_db.elat = tornado_db.apply(lambda x: x.slat + 0.001, axis=1)
        ...
    else:
        tornado_db = tornado_db[~((tornado_db['elon'] == 0) & (tornado_db['elat'] == 0))]
    geometries = [LineString([(x0, y0), (x1, y1)])
                  for x0, y0, x1, y1
                  in zip(tornado_db.slon, tornado_db.slat, tornado_db.elon, tornado_db.elat)]
    tornado_db = gpd.GeoDataFrame(tornado_db, crs="EPSG:4326", geometry=geometries)
    tornado_db = tornado_db.to_crs(crs=pars['crs'])
    tornado_db['datetime'] = tornado_db['date'].str.cat(tornado_db['time'], sep=" ")
    tornado_db = _geopandas_fix_datetime(tornado_db, cols=['datetime'], fmt='%Y-%m-%d %H:%M:%S')
    tornado_db = _geopandas_fix_datetime(tornado_db, cols=['date'], fmt='%Y-%m-%d')
    tornado_db['date'] = tornado_db['date'].dt.date
    tornado_db = _geopandas_fix_datetime(tornado_db, cols=['time'], fmt='%H:%M:%S')
    tornado_db['time'] = tornado_db['time'].dt.time
    return tornado_db


def _read_geo_files_into_geopandas(files, crs="EPSG:4326"):
    if isinstance(files, str):
        print(f"Reading : {files}")
        gp_df = gpd.read_file(files)
        gp_df = gp_df.to_crs(crs=crs)
        return gp_df
    gp_df = []
    for file in files:
        print(f"Reading : {file}")
        gp_df.append(gpd.read_file(file))
        gp_df[-1] = gp_df[-1].to_crs(crs=crs)
    gp_df = pd.concat(gp_df)
    return gp_df


def _geopandas_fix_datetime(gdf, cols=None, fmt='%Y%m%d%H%M%S'):
    """Adds date time to :param cols: in a :param gdf: geopandas dataframe.  :param fmt: is the datetime format"""
    if isinstance(cols, str):
        gdf[cols] = pd.to_datetime(gdf[cols], format=fmt)
    else:
        for col in cols:
            gdf[col] = pd.to_datetime(gdf[col], format=fmt)
    return gdf
