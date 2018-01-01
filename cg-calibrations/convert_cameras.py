import sys
sys.path.append('../')
import image

CAMERA = 'nikon-e8700'

cam = image.Camera.from_matlab(
    imgsz=[3264, 2448],
    sensorsz=cgcalib.load_calibration(camera=CAMERA)['sensorsz'],
    fc=[3345.760828684510670, 3343.007845322197227],
    cc=[1626.529826546877302, 1228.053303013219875],
    kc=[-0.206256762062722, 0.206074941129924, 0.000795687419603, -0.000762555784047, 0.000000000000000])
cam.write(
    path="cameras/" + CAMERA + "-calib.json",
    attributes=["fmm", "cmm", "k", "p", "sensorsz"])

cam = image.Camera.from_matlab(
    imgsz=[3264, 2448],
    sensorsz=cgcalib.load_calibration(camera=CAMERA)['sensorsz'],
    fc=[1.535806253664732, 1.612127581079067],
    cc=[2.265762525976454, 1.727752135185034],
    kc=[0.001735351838634, 0.005240486256317, 0.000112783687547, 0.000136815030323, 0.000000000000000])
cam.vector /= 3 # "errors are approximately three times the standard deviations"
cam.write(
    path="cameras/" + CAMERA + "-calib_stderr.json",
    attributes=["fmm", "cmm", "k", "p"])
