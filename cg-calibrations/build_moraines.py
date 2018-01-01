import glob
import os

DEM_DIR = "/volumes/science-b/data/columbia/dem/"

paths = glob.glob("geojson/moraines/*.geojson")
for path in paths:
    basename = os.path.basename(path)
    date_str = os.path.splitext(basename)[0]
    dem_path = glob.glob(DEM_DIR + date_str + "*.tif")[-1]
    dem = DEM.DEM.read(dem_path)
    geo = helper.read_geojson(path, crs=32606)
    helper.elevate_geojson(geo, elevation=dem)
    geo2 = helper.ordered_geojson(geo)
    helper.write_geojson(geo2,
        path=os.path.splitext(path)[0] + ".geojson",
        decimals=[7, 7, 2],
        crs=32606)
