import cg
from cg import (glimpse)
from glimpse.imports import (datetime)

START_TIME = datetime.datetime(2005, 6, 2)
END_TIME = datetime.datetime(2005, 6, 25)
STATION = 'CG05'
SERVICE = '20050916'

# ---- Load anchors ----


# ---- Load images ----

# Identify services
sequences = cg.Sequences()
is_row = (
    (sequences.first_time_utc < END_TIME) &
    (sequences.last_time_utc > START_TIME) &
    (sequences.station == STATION))
sequences.loc[is_row, ('station', 'service', 'first_time_utc', 'last_time_utc')]

# Identify dems
# Identify anchor images
# Identify fixed-area mask

STATION_DIR = os.path.join('data', 'AK01b')
IMG_DIR = os.path.join(STATION_DIR, 'images')
KEYPOINT_DIR = os.path.join(STATION_DIR, 'keypoints')
MATCH_DIR = os.path.join(STATION_DIR, 'matches')
MASK_PATH = os.path.join(STATION_DIR, 'mask', 'mask.npy')
ANCHOR_BASENAME = 'AK01b_20130615_220325'

# ---- Prepare Observer ----

all_img_paths = glob.glob(os.path.join(IMG_DIR, '*.JPG'))
datetimes = np.array([glimpse.Exif(path).datetime
    for path in all_img_paths])
inrange = np.logical_and(datetimes > START_TIME, datetimes < END_TIME)
img_paths = [all_img_paths[i] for i in np.where(inrange)[0]]
basenames = [os.path.splitext(os.path.basename(path))[0]
    for path in img_paths]
cam_args = cg.load_calibration(image=ANCHOR_BASENAME)
images = [glimpse.Image(
    path, cam=cam_args.copy(), anchor=(basename == ANCHOR_BASENAME),
    keypoints_path=os.path.join(KEYPOINT_DIR, basename + '.pkl'))
    for path, basename in zip(img_paths, basenames)]
observer = glimpse.Observer(images)

# build masked keypoints
# build masked keypoint matches
# optimize image viewdir

# identify glacier area (how?)
# generate tracking points
# split images into 3-day observers
# track points
# compile results
# filter results based on quality, visibility
