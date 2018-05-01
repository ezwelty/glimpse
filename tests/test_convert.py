from .context import *
from glimpse.imports import (np)

params = dict(
    imgsz = (200, 100),
    f = (200, 200),
    c = (0, 0),
    k = (0.1, 0.1, 0.1),
    p = (0.01, 0.01),
    sensorsz = (20, 10)
)
rcam = glimpse.Camera(**params)

def test_matlab_camera():
    mcam = glimpse.convert.MatlabCamera(
        imgsz=rcam.imgsz, fc=rcam.f, cc=rcam.c + (rcam.imgsz / 2) - 0.5,
        kc=np.concatenate((rcam.k[0:2], rcam.p, rcam.k[2:3])), alpha_c=0)
    np.array_equal(rcam.vector, mcam.as_camera().vector)

def test_photoscan_camera():
    pscam = glimpse.convert.PhotoScanCamera(
        imgsz=rcam.imgsz, f=rcam.f[1], cx=rcam.c[0], cy=rcam.c[1],
        k1=rcam.k[0], k2=rcam.k[1], k3=rcam.k[2], p1=rcam.p[1], p2=rcam.p[0],
        b1=rcam.f[1] - rcam.f[0])
    np.array_equal(rcam.vector, pscam.as_camera().vector)
