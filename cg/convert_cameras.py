import cg
from cg import (glimpse)
from glimpse.imports import (re)

def parse_matlab_calibration(path):
    with open(path, mode='r') as fp:
        txt = fp.read()
    def parse_parameter(param, length):
        if length == 1:
            pattern = '^' + param + ' = (.*)' + ';'
        else:
            pattern = '^' + param + ' = \[ ' + ' ; '.join(['(.*)'] * length) + ' \];'
        values = re.findall(pattern, txt, flags=re.MULTILINE)[0]
        if length == 1:
            return float(values)
        else:
            return [float(value) for value in values]
    lengths = dict(
        fc=2, fc_error=2,
        cc=2, cc_error=2,
        kc=5, kc_error=5,
        nx=1, ny=1)
    return dict((param, parse_parameter(param, length)) for param, length in lengths.items())

def import_matlab_calibration(path, camera, suffix='-calib'):
    calib = parse_matlab_calibration(path)
    imgsz = (calib['nx'], calib['ny'])
    sensorsz = cg.load_calibration(camera=camera)['sensorsz']
    # mean values
    cam = glimpse.Camera.from_matlab(
        imgsz=imgsz, sensorsz=sensorsz,
        fc=calib['fc'], cc=calib['cc'], kc=calib['kc'])
    cam.write(
        path=os.path.join('cameras', camera + suffix + '.json'),
        attributes=('fmm', 'cmm', 'k', 'p', 'sensorsz'))
    # standard errors
    cam = glimpse.Camera.from_matlab(
        imgsz=imgsz, sensorsz=sensorsz,
        fc=calib['fc_error'], cc=calib['cc_error'], kc=calib['kc_error'])
    cam.vector /= 3 # 'errors are approximately three times the standard deviations'
    cam.write(
        path=os.path.join('cameras', camera + suffix + '_stderr.json'),
        attributes=('fmm', 'cmm', 'k', 'p'))

# ---- #

CAMERA = 'nikon-e8700'
CALIB_PATH = '/volumes/science-b/projects/timeseries/calibrations/cg/' + CAMERA + '/matlab-cct/Calib_Results.m'
import_matlab_calibration(CALIB_PATH, CAMERA)

CAMERA = 'canon-40d-01'
CALIB_PATH = '/volumes/science-b/projects/timeseries/calibrations/cg/' + CAMERA + '/matlab-cct/Calib_Results.m'
import_matlab_calibration(CALIB_PATH, CAMERA)
