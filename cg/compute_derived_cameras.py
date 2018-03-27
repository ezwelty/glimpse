import sys
sys.path.insert(0,'../')
import image
import numpy as np
from matplotlib import path
import matplotlib.pyplot as plt
import optimize
from svgpathtools import svg2paths
import os
import cv2
from helper import compute_mask_array_from_svg

animate = True

camera = 'AK01b'
anchor_image_name = 'AK01b_20130615_220325'
other_camera_location = np.array([499211.00725947437, 6783756.0918104537, 479.33179419876853]).reshape((1,3))

#camera = 'AK10b'
#anchor_image_name = 'AK10b_20130615_013704'
#other_camera_location = np.array([493219.90001682361, 6776736.585432346, 464.54523318793304]).reshape((1,3))

data_dir = '../data/'+camera+'/'
mask = np.load(data_dir+'mask/mask.npy')
image_dir = data_dir+'images/'
animation_dir = data_dir+'animation/'
derived_image_dir = data_dir+'images_json/'

anchor_image = image.Image('./images/'+anchor_image_name+'-original.jpg',cam='./images/'+anchor_image_name+'.json')
anchor_image.read()

files = np.sort(os.listdir(image_dir))

if animate:
    plt.ion()
    uv = anchor_image.cam.project(other_camera_location)[0]
    fig,ax = plt.subplots()
    im = ax.imshow(anchor_image.I)
    ax.set_xlim(250,350)
    ax.set_ylim(750,650)
    ph, = ax.plot(uv[0],uv[1],'r.')

for filename in files:
    print(filename)
    new_image = image.Image(image_dir+filename,cam='./images/'+anchor_image_name+'.json')
    new_image.read()

    #matches = optimize.corr_matches([anchor_image,new_image],masks=mask,n_points=100,hw=15,sd=25,do_highpass=True)
    matches = optimize.sift_matches([anchor_image,new_image],masks=mask,ratio=0.8,nfeatures=0)

    model = optimize.Cameras(cams=new_image.cam,controls=matches,cam_params={'viewdir':True})
    try:
        rvalues, rindex = optimize.ransac(model, sample_size=15, max_error=0.25, min_inliers=10, iterations=100)
        new_image.cam.viewdir = rvalues
        if animate:
            uv = new_image.cam.project(other_camera_location)[0]
            im.set_data(new_image.I)
            ph.set_xdata(uv[0])
            ph.set_ydata(uv[1])
            ax.set_xlim(uv[0]-50,uv[0]+50)
            ax.set_ylim(uv[1]+50,uv[1]-50)
            plt.pause(0.0000001)
            fig.savefig(animation_dir+filename,dpi=200)
        new_image.cam.write(derived_image_dir+new_image.path[-25:-4]+'.json')
    except ValueError:
        print("BAD SOLUTION")
