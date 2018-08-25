import cg
from cg import glimpse
from glimpse.imports import (re, np, os, matplotlib, collections)
import shutil
root = '/volumes/science/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')
new_root = '/volumes/boot 1/users/admin/sites/cg/glimpse/cg'
new_image_path = '/volumes/boot 1/users/admin/sites/cg/glimpse/cg/timelapse'

observer_json = glimpse.helpers.read_json('observers.json',
    object_pairs_hook=collections.OrderedDict)

# Copy dem interpolant
# source = 'dem_interpolant.pkl'
# destination = os.path.join(new_root, 'dem_interpolant.pkl')
# if not os.path.isfile(destination):
#     shutil.copyfile(source, destination)
for i in range(300, 400):
    # date_strings = []
    # Copy images
    for basenames in observer_json[i].values():
        for basename in basenames:
            source = cg.find_image(basename)
            destination = os.path.join(new_image_path, basename + '.JPG')
            if not os.path.isfile(destination):
                shutil.copyfile(source, destination)
        # date_strings.append(cg.parse_image_path(basenames[0])['date_str'])
    # datestr = min(date_strings)
    # basename = datestr + '-' + str(i)
    # # Copy points
    # source = os.path.join('points', basename + '.pkl')
    # destination = os.path.join(new_root, 'points', basename + '.pkl')
    # if not os.path.isfile(destination):
    #     shutil.copyfile(source, destination)

# Timelapse symlinks (nested)
import subprocess
regex = re.compile('timelapse')
for i in range(1066 + 1):
    print(i)
    for basenames in observer_json[i].values():
        for basename in basenames:
            source = cg.find_image(basename)
            destination = regex.sub('timelapse-hardlinks', source)
            glimpse.helpers.make_path_directories(destination)
            if not os.path.isfile(destination):
                subprocess.call(['ln', source, destination])

# Timelapse symlinks (flat)
import subprocess
new_image_path = '/volumes/science/data/columbia/timelapse-symlinks-flat/'
for i in range(100 + 1): #1066 + 1):
    print(i)
    for basenames in observer_json[i].values():
        for basename in basenames:
            source = cg.find_image(basename)
            destination = new_image_path + basename + '.JPG'
            subprocess.call(['ln', '-s', source, destination])
