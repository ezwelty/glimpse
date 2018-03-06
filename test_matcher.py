import numpy as np
from matplotlib import path
import matplotlib.pyplot as plt

import glimpse

import os
import cv2
import sys
import time
import datetime

import sharedmem

year_start = 2013
month_start = 6
day_start = 12
hour_start = 00
year_end = 2013
month_end = 6
day_end = 15
hour_end = 00

start_time = time.mktime(datetime.datetime(year_start,month_start,day_start,hour_start,0,0).timetuple())
end_time = time.mktime(datetime.datetime(year_end,month_end,day_end,hour_end,0,0).timetuple())

data_directory = './data/'

camera_directory = data_directory+'/AK01b/'
raw_image_directory = camera_directory+'images/' 
sift_directory = camera_directory+'sift_descriptors/'

mask = np.load(camera_directory+'mask/mask.npy')
anchor_image_name = 'AK01b_20130615_220325'
anchor_image = glimpse.Image(raw_image_directory+anchor_image_name+'.JPG',cam='cg-calibrations/images/'+anchor_image_name+'.json')

img_names = np.sort(os.listdir(raw_image_directory))[:7]
images = []

for img_name in img_names:
    basename = img_name[:-4]
    if anchor_image_name==basename:
        images.append(glimpse.Image(raw_image_directory+basename+'.JPG',cam=anchor_image.cam.copy(),siftpath=sift_directory+basename+'.p',anchor_image=True))
    else:
        images.append(glimpse.Image(raw_image_directory+basename+'.JPG',cam=anchor_image.cam.copy(),siftpath=sift_directory+basename+'.p'))

observer_ = glimpse.Observer(images)

ms = glimpse.optimize.CameraMotionSolver(observer_)

ms.generate_image_kp_and_des(masks=mask,contrastThreshold=0.02,overwrite_cached_kp_and_des=False)
matches = ms.generate_matches(match_bandwidth=10,match_path='./data/AK01b/matches/',overwrite_matches=False)
w00 = observer_.images[0].cam.viewdir.copy()
out = ms.align()


other_camera_location = np.array([499211.00725947437, 6783756.0918104537, 479.33179419876853]).reshape((1,3))

fig,axs = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(16,8)

for i,im in enumerate(observer_.images):
    im.read()

    uv = im.cam.project(other_camera_location)[0]
    if i==0:
        uv0 = uv.copy()
    axs[0].imshow(im.I,interpolation='none')
    axs[0].plot(uv[0],uv[1],'r.')
    axs[0].set_xlim(uv[0]-50,uv[0]+50)
    axs[0].set_ylim(uv[1]+50,uv[1]-50)

    axs[1].imshow(im.I,interpolation='none')
    axs[1].plot(uv[0],uv[1],'r.')
    axs[1].set_xlim(uv0[0]-50,uv0[0]+50)
    axs[1].set_ylim(uv0[1]+50,uv0[1]-50)

    fig.savefig('./data/AK01b/anims/anim_{0:03}.jpg'.format(i),dpi=300)
    axs[0].cla()
    axs[1].cla()    



