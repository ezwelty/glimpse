import cg
from cg import (glimpse, glob)
from glimpse.imports import (os, re, np)

DEM_DIR = '/volumes/science-b/data/columbia/dem'
DATE_STR = '20070922'

# ---- Read DEM ----

dem_path = glob.glob(os.path.join(DEM_DIR, DATE_STR + '*.tif'))[-1]
dem = glimpse.Raster.read(dem_path)

# ---- Moraine lines ----

paths = glob.glob('geojson/moraines/' + DATE_STR + '.geojson')
for path in paths:
    geo = glimpse.helpers.read_geojson(path, crs=32606)
    glimpse.helpers.elevate_geojson(geo, elevation=dem)
    for coords in glimpse.helpers.geojson_itercoords(geo):
        if any(np.isnan(coords[:, 2])):
            print('Missing elevations in ' + path)
    geo2 = glimpse.helpers.ordered_geojson(geo)
    glimpse.helpers.write_geojson(geo2,
        path=os.path.splitext(path)[0] + '.geojson',
        decimals=(7, 7, 2), crs=32606)

# ---- Ground control points ----

GCP_DEM_PATH = '/volumes/science-b/data/columbia/_new/ArcticDEM/v2.0/tiles/merged_projected_clipped.tif'
dem_ref = glimpse.Raster.read(GCP_DEM_PATH)

geo = glimpse.helpers.read_geojson('geojson/gcp.geojson', crs=32606, key='id')
keys = [key for key in geo['features'] if re.findall('T' + DATE_STR, key)]
keys.sort()
for key in keys:
    coords = geo['features'][key]['geometry']['coordinates']
    coords[:, 2] = dem.sample(coords[:, 0:2])
    if any(np.isnan(coords[:, 2])):
        print('Missing elevation for ' + key)

# Check for gross errors
z = np.vstack([geo['features'][key]['geometry']['coordinates'][:, 2] for key in keys])
z_ref = np.vstack([dem_ref.sample(geo['features'][key]['geometry']['coordinates'][:, 0:2]) for key in keys])
z - z_ref

geo2 = glimpse.helpers.ordered_geojson(geo, properties=('id', 'valid', 'notes'))
glimpse.helpers.write_geojson(geo2, path='geojson/gcp.geojson',
    decimals=(7, 7, 2), crs=32606)
