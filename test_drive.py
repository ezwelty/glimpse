import numpy as np
from matplotlib import path
import matplotlib.pyplot as plt

import optimize
import image
import dem
import helper

import os
import cv2
import sys
import time
import datetime

import sharedmem

import observer
import tracker

year_start = 2013
month_start = 6
day_start = 12
hour_start = 20
year_end = 2013
month_end = 6
day_end = 18
hour_end = 20

start_time = time.mktime(datetime.datetime(year_start,month_start,day_start,hour_start,0,0).timetuple())
end_time = time.mktime(datetime.datetime(year_end,month_end,day_end,hour_end,0,0).timetuple())

data_directory = './data/'
camera_directory_ak01 = data_directory+'/AK01b/'
image_orientation_directory_ak01 = camera_directory_ak01+'images_json/'
raw_image_directory_ak01 = camera_directory_ak01+'images/' 

camera_directory_ak10 = data_directory+'/AK10b/'
image_orientation_directory_ak10 = camera_directory_ak10+'images_json/'
raw_image_directory_ak10 = camera_directory_ak10+'images/' 

dem_list = os.listdir(data_directory+'dem/')
dem_list = [n for n in dem_list if 'geo' in n]
dem_list.sort()
dem_dates = [time.mktime(datetime.datetime(int(n[8:12]),int(n[12:14]),int(n[14:16]),0,0,0).timetuple()) for n in dem_list]

dem_indices = helper.nearest_neighbours(start_time,dem_dates)

xmin = 492000
xmax = 501500
ymin = 6.777e6
ymax = 6.781e6+5000

if np.isscalar(dem_indices):
    x,y,Z = helper.get_cropped_dem(data_directory+'dem/'+dem_list[dem_indices],xmin,xmax,ymin,ymax)
else:
    x0,y0,Z0 = helper.get_cropped_dem(data_directory+'dem/'+dem_list[dem_indices[0]],xmin,xmax,ymin,ymax)
    x1,y1,Z1 = helper.get_cropped_dem(data_directory+'dem/'+dem_list[dem_indices[1]],xmin,xmax,ymin,ymax)
    t_interval = dem_dates[dem_indices[1]] - dem_dates[dem_indices[0]]
    d0 = start_time - dem_dates[dem_indices[0]]
    d1 = dem_dates[dem_indices[1]] - start_time
    x = x0*d1/t_interval + x1*d0/t_interval
    y = y0*d1/t_interval + y1*d0/t_interval
    Z = Z0*d1/t_interval + Z1*d0/t_interval

DEM = dem.DEM(Z,x=x,y=y)
DEM.fill_crevasses_simple()

def create_observer(image_orientation_directory,raw_image_directory):
    img_base_names,img_times = helper.get_image_names_and_times(image_orientation_directory,extension='.json')

    indices = np.array([t>=start_time and t<=end_time for t in img_times])

    times = img_times[indices]
    times /= 60**2*24
    images = []
    for img_name,do_read in zip(img_base_names,indices):
        if do_read:
            images.append(image.Image(raw_image_directory+img_name+'.JPG',cam=image_orientation_directory+img_name+'.json'))

    observer_ = observer.Observer(images,times,reference_halfwidth=25,search_halfwidth=35)
    return observer_,times

observer_ak10,times_ak10 = create_observer(image_orientation_directory_ak10,raw_image_directory_ak10)
observer_ak01,times_ak01 = create_observer(image_orientation_directory_ak01,raw_image_directory_ak01)
times = np.sort(np.hstack((times_ak01,times_ak10)))

xy_0 = np.array([498800,6.78186e6])
p_tracker = tracker.Tracker(5000,xy_0[0],xy_0[1],DEM,times,observers=[observer_ak01,observer_ak10])
p_tracker.initialize_particles(2.0,2.0,10.0,10.0)
p_tracker.initialize_observers()
p_tracker.track(2.0,2.0,do_plot=True)
