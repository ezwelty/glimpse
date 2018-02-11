from .context import *
from glimpse.imports import (np, datetime)

def test_image_init_defaults():
    path = os.path.join(test_dir, 'AK10b_20141013_020336.JPG')
    img = glimpse.Image(path)
    assert img.path == path
    assert img.datetime == img.exif.datetime
    assert all(img.cam.imgsz == img.exif.size)
    sensorsz = glimpse.Camera.get_sensor_size(img.exif.make, img.exif.model)
    assert all(img.cam.f == img.exif.fmm * img.exif.size / sensorsz)

def test_image_init_custom():
    path = os.path.join(test_dir, 'AK10b_20141013_020336.JPG')
    img_datetime = datetime.datetime(2010, 1, 1, 0, 0, 0)
    cam_args = dict(imgsz=(100, 100), sensorsz=(10, 10))
    img = glimpse.Image(path, cam=cam_args, datetime=img_datetime)
    assert img.datetime == img_datetime
    assert all(img.cam.imgsz == cam_args['imgsz'])
    assert all(img.cam.f == img.exif.fmm * np.divide(cam_args['imgsz'], cam_args['sensorsz']))

def test_image_read():
    path = os.path.join(test_dir, 'AK10b_20141013_020336.JPG')
    # Default size
    img = glimpse.Image(path)
    I = img.read()
    assert all(I.shape[0:2][::-1] == img.cam.imgsz)
    # Resize camera
    img.cam.resize(0.5)
    I = img.read()
    assert all(I.shape[0:2][::-1] == img.cam.imgsz)
