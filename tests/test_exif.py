import os
import datetime
import shutil
import tempfile
from .context import glimpse, test_dir

def test_parses_file_exif():
    path = os.path.join(test_dir, 'AK10b_20141013_020336.JPG')
    exif = glimpse.Exif(path)
    assert exif.imgsz == (800, 536)
    assert all(isinstance(x, int) for x in exif.imgsz)
    assert exif.fmm == 20
    assert isinstance(exif.fmm, float)
    assert exif.make == 'NIKON CORPORATION'
    assert exif.model == 'NIKON D200'
    assert exif.iso == 200
    assert isinstance(exif.iso, int)
    assert exif.exposure == 0.0125
    assert isinstance(exif.exposure, float)
    assert exif.aperture == 8.0
    assert isinstance(exif.aperture, float)
    assert exif.datetime == datetime.datetime(2014, 10, 13, 2, 3, 36, 280000)
    assert exif.sensorsz == (23.6, 15.8)
    assert all(isinstance(x, float) for x in exif.sensorsz)

def test_inserts_exif_into_file():
    old = os.path.join(test_dir, 'AK10b_20141013_020336.JPG')
    _, new = tempfile.mkstemp()
    shutil.copy2(old, new)
    exif = glimpse.Exif(old)
    old_iso = exif.iso
    new_iso = int(old_iso * 0.5)
    exif.tags['Exif']['ISOSpeedRatings'] = new_iso
    exif.insert(new)
    exif = glimpse.Exif(new)
    assert exif.iso == new_iso
    os.remove(new)
