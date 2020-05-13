import datetime
import shutil
import tempfile
from .context import glimpse, test_dir, os

def test_parses_file_exif():
    path = os.path.join(test_dir, 'AK10b_20141013_020336.JPG')
    exif = glimpse.Exif(path)
    assert exif.size[0] == 800
    assert exif.size[1] == 536
    assert exif.fmm == 20
    assert exif.make == 'NIKON CORPORATION'
    assert exif.model == 'NIKON D200'
    assert exif.iso == 200
    assert exif.exposure == 0.0125
    assert exif.aperture == 8
    assert exif.datetime == datetime.datetime(2014, 10, 13, 2, 3, 36, 280000)

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
