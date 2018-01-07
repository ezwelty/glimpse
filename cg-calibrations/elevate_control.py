import glob
import os
import re
import dem as DEM

DEM_DIR = "/volumes/science-b/data/columbia/dem/"
DATE_STR = '20121001'

# ---- Read DEM ----

dem_path = glob.glob(DEM_DIR + DATE_STR + "*.tif")[-1]
dem = DEM.DEM.read(dem_path)

# ---- Moraine lines ----

paths = glob.glob("geojson/moraines/" + DATE_STR + ".geojson")
for path in paths:
    basename = os.path.basename(path)
    geo = helper.read_geojson(path, crs=32606)
    helper.elevate_geojson(geo, elevation=dem)
    for coords in helper.geojson_itercoords(geo):
        if any(np.isnan(coords[:, 2])):
            print "Missing elevations in " + path
    geo2 = helper.ordered_geojson(geo)
    helper.write_geojson(geo2,
        path=os.path.splitext(path)[0] + ".geojson",
        decimals=[7, 7, 2],
        crs=32606)

# ---- Ground control points ----

GCP_DEM_PATH = "/volumes/science-b/data/columbia/_new/ArcticDEM/v2.0/tiles/merged_projected_clipped.tif"
dem_ref = DEM.DEM.read(GCP_DEM_PATH)

geo = helper.read_geojson("geojson/gcp.geojson", crs=32606, key="id")
keys = [key for key in geo['features'].iterkeys() if re.findall("T" + DATE_STR, key)]
keys.sort()
for key in keys:
    coords = geo['features'][key]['geometry']['coordinates']
    coords[:, 2] = dem.sample(coords[:, 0:2])
    if any(np.isnan(coords[:, 2])):
        print "Missing elevation for " + key

# Check for gross errors
z = np.vstack([geo['features'][key]['geometry']['coordinates'][:, 2] for key in keys])
z_ref = np.vstack([dem_ref.sample(geo['features'][key]['geometry']['coordinates'][:, 0:2]) for key in keys])
z - z_ref

geo2 = helper.ordered_geojson(geo, properties=['id', 'valid', 'notes'])
helper.write_geojson(geo2,
    path="geojson/gcp.geojson",
    decimals=[7, 7, 2],
    crs=32606)
