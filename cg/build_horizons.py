import cg
from cg import (glimpse, glob)
from glimpse.imports import (np, matplotlib)
import geojson
cg.IMAGE_PATH = '/volumes/science-b/data/columbia/timelapse'

# ---- Constants ---- #

# CG04: IfSar (ArcticDEM as one artifact)
# CG05: IfSar (ArcticDEM not tried)
# CG06: IfSar (ArcticDEM not tried)
# AK01: IfSar (ArcticDEM as one artifact)
# AK01b: IfSar (ArcticDEM better detail on far left, but couple large errors)
# AK03(b): IfSar (ArcticDEM not tried)
# AK04: IfSar (ArcticDEM not tried)
# AK09(b): IfSar (ArcticDEM not tried)
# AK10(b): IfSAR (ArcticDEM has nasty artifacts)
# AK12: IfSar (ArcticDEM not tried)
# AKJNC: IfSar (very similar to ArcticDEM, but without a couple small artifacts)
# AKST03A: IfSar (ArcticDEM not tried)
# AKST03B: IfSar (ArcticDEM not tried)

STATION = 'CG04'

# DEM_PATH = '/volumes/science-b/data/columbia/_new/arcticdem/v2.0/tiles/merged_projected_horizon.tif'
# DEM_DZ = 0
DEM_PATH = '/volumes/science-b/data/columbia/_new/ifsar/merged_projected_horizon.tif'
DEM_DZ = 17
DEM_ZLIM = (1, np.inf)
DDEG = 0.05

# --- Load DEM ---- #

dem = glimpse.DEM.read(DEM_PATH)
dem.crop(zlim=DEM_ZLIM)
dem.Z += DEM_DZ

# --- Compute horizon ---- #

eop = cg.load_calibrations(station_estimate=STATION, merge=True)
# Stamp null circle into DEM around camera
dem.fill_circle(center=eop['xyz'], radius=100)
# Sample horizon in swath in front of camera
heading = eop['viewdir'][0]
headings = np.arange(heading - 90, heading + 90, step=DDEG)
# headings = np.arange(0, 360, step=DDEG) # AK12 only
hxyz = dem.horizon(eop['xyz'], headings)

# --- Format and save GeoJSON ---- #

geo = geojson.FeatureCollection(
    [geojson.Feature(geometry=geojson.LineString(xyz.tolist())) for xyz in hxyz])
geo = glimpse.helpers.ordered_geojson(geo)
glimpse.helpers.write_geojson(geo,
    'geojson/horizons/' + STATION + '.geojson',
    crs=32606, decimals=(5, 5, 0))

# --- Check result ---- #

svg_path = glob.glob('svg/' + STATION + '_*.svg')[0]
img_path = cg.find_image(svg_path)
eop = cg.load_calibrations(path=img_path, station_estimate=STATION, merge=True)
img = glimpse.Image(img_path, cam=eop)
geo = glimpse.helpers.read_geojson('geojson/horizons/' + STATION + '.geojson', crs=32606)
lxyz = [coords for coords in glimpse.helpers.geojson_itercoords(geo)]
luv = [img.cam.project(xyz, correction=True) for xyz in lxyz]
img.plot()
for uv in luv:
    matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], color='red')
img.set_plot_limits()
