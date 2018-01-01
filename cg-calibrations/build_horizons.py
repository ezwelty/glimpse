import cgcalib
import dem as DEM
import helper
import geojson

# ---- Constants ---- #

# IfSAR DEM much better looking south (AK10). ArcticDEM has nasty artifacts.
# ArcticDEM may have the edge looking north (AK01) – very similar to IfSar but more detail.

STATION = 'AK01b'
DEM_PATH = "/volumes/science-b/data/columbia/_new/arcticdem/v2.0/tiles/merged_projected_horizon.tif"
DEM_DZ = 0
# DEM_PATH = "/volumes/science-b/data/columbia/_new/ifsar/merged_projected_horizon.tif"
# DEM_DZ = 17
DEM_ZLIM = [1, np.inf]
DDEG = 0.05

# --- Load DEM ---- #
dem = DEM.DEM.read(DEM_PATH)
dem.crop(zlim=DEM_ZLIM)
dem.Z += DEM_DZ

# --- Compute horizon ---- #

eop = cgcalib.station_eop(STATION)
# Stamp null circle into DEM around camera
dem.fill_circle(center=eop['xyz'], radius=100)
# Sample horizon in swath in front of camera
heading = eop['viewdir'][0]
headings = np.arange(heading - 90, heading + 90, step=DDEG)
hxyz = dem.horizon(eop['xyz'], headings)

# --- Format and save GeoJSON ---- #

geo = geojson.FeatureCollection(
    [geojson.Feature(geometry=geojson.LineString(xyz.tolist())) for xyz in hxyz])
geo = helper.ordered_geojson(geo)
helper.write_geojson(geo,
    "geojson/horizons/" + STATION + "-arcticdem.geojson",
    crs=32606, decimals=[5, 5, 0])

# --- Check result ---- #

eop = cgcalib.station_eop(STATION)
img = image.Image(glob.glob("svg/" + STATION + "*.JPG")[0], cam=eop)
geo = helper.read_geojson("geojson/horizons/" + STATION + "-arcticdem.geojson", crs=32606)
lxyz = [coords for coords in helper.geojson_itercoords(geo)]
luv = [img.cam.project(xyz) for xyz in lxyz]
# img.plot()
for uv in luv:
    matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], color="red")

img.set_plot_limits()