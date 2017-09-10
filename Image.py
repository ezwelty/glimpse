from datetime import datetime
import numpy as np
import exifread
import scipy.misc
import math
from Camera import Camera

class Image(object):
    """
    An `Image` describes the camera settings and resulting image captured at a particular time.

    Attributes:
        path (str): Path to the image file
        info (dict): Image file metadata
        datetime (datetime): Capture date and time
        epoch (float): Capture date and time as seconds since 1970-01-01 00:00:00
        size (array): Size of the original image in pixels [nx, ny]
        shutter (float): Shutter speed in seconds
        aperture (float): Lens aperture size as f-number
        iso (float): Film speed
        ev (float): Exposure value
        cam (Camera): Camera object
    """
    
    def __init__(self, path, cam_props = None):
        """Create an Image."""
        self.path = path
        # Load image metadata
        f = open(path, 'rb')
        self.info = exifread.process_file(f, details=False)
        # Camera object
        if not cam_props:
            cam_props = {}
        if cam_props.has_key('imgsz'):
            raise UserWarning("cam_props['imgsz'] replaced by actual image size")
        cam_props['imgsz'] = self.size
        if not cam_props.has_key('f'):
            cam_props['f'] = fmm_to_fpx(self.fmm, self.sensorsz, self.size)
        self.cam = Camera(**cam_props)
    
    @property
    def size(self):
        imgsz = [
            parse_exif_tag(self.info['EXIF ExifImageWidth']),
            parse_exif_tag(self.info['EXIF ExifImageLength'])
        ]
        return(np.array(imgsz))
    
    @property
    def datetime(self):
        datetime_str = parse_exif_tag(self.info['EXIF DateTimeOriginal'])
        subsec_str = parse_exif_tag(self.info['EXIF SubSecTimeOriginal'])
        return(datetime.strptime(datetime_str + "." + subsec_str.zfill(6), "%Y:%m:%d %H:%M:%S.%f"))
        
    @property
    def epoch(self):
        dt = self.datetime - datetime(1970, 1, 1)
        return(dt.total_seconds())
    
    @property
    def shutter(self):
        return(parse_exif_tag(self.info['EXIF ExposureTime']))
        
    @property
    def aperture(self):
        return(parse_exif_tag(self.info['EXIF FNumber']))
        
    @property
    def iso(self):
        return(parse_exif_tag(self.info['EXIF ISOSpeedRatings']))
    
    @property
    def ev(self):
        # https://en.wikipedia.org/wiki/Exposure_value
        return(math.log(1000 * self.aperture ** 2 / (self.iso * self.shutter), 2))
    
    @property
    def fmm(self):
        return(parse_exif_tag(self.info['EXIF FocalLength']))
        
    @property
    def make(self):
        return(parse_exif_tag(self.info['Image Make']))
        
    @property
    def model(self):
        return(parse_exif_tag(self.info['Image Model']))
    
    @property
    def sensorsz(self):
        sensorsz = get_sensor_size(self.make, self.model)
        return(np.array(sensorsz))
    
    def read():
        """Read image file."""
        I = scipy.misc.imread(self.path)
        return(I)

def parse_exif_tag(tag):
    """Parse the value of a tag returned by exifread.process_file()."""
    value = tag.values
    if type(value) is list:
        value = value[0]
    if isinstance(value, exifread.Ratio):
        return(float(value.num) / value.den)
    else:
        return(value)

def get_sensor_size(make, model):
    """
    Get a camera model's CCD sensor width and height in mm.
    
    Data is from Digital Photography Review (https://dpreview.com).
    See also https://www.dpreview.com/articles/8095816568/sensorsizes.
    
    Arguments:
        make (str): Camera make (EXIF Make)
        model (str): Camera model (EXIF Model)
        
    Return:
        list: Camera sensor width and height in mm
    """
    make_model = make.strip() + " " + model.strip()
    sensor_sizes = { # mm
        'NIKON CORPORATION NIKON D2X': [23.7, 15.7], # https://www.dpreview.com/reviews/nikond2x/2
        'NIKON CORPORATION NIKON D200': [23.6, 15.8], # https://www.dpreview.com/reviews/nikond200/2
        'NIKON CORPORATION NIKON D300S': [23.6, 15.8], # https://www.dpreview.com/reviews/nikond300s/2
        'NIKON E8700': [8.8, 6.6], # https://www.dpreview.com/reviews/nikoncp8700/2
        'Canon Canon EOS 20D': [22.5, 15.0], # https://www.dpreview.com/reviews/canoneos20d/2
        'Canon Canon EOS 40D': [22.2, 14.8], # https://www.dpreview.com/reviews/canoneos40d/2
    }
    if sensor_sizes.has_key(make_model):
        return(sensor_sizes[make_model])
    else:
        raise KeyError("No sensor size found for " + make_model)

def fmm_to_fpx(fmm, sensorsz, imgsz):
    """Convert focal length in millimeters to pixels."""
    return(fmm * imgsz / sensorsz)
    
def fpx_to_fmm(fpx, sensorsz, imgsz):
    """Convert focal length in pixels to millimeters."""
    return(fpx * sensorsz / imgsz)
