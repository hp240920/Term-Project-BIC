import os
#import cv2
import matplotlib.image as mpimg


path_img = ['Rotated_img_10\Red', 'Rotated_img_10\Green', 'Rotated_img_10\Yellow']
des_path = 'Mix_imgs/'

color = 0

for paths in path_img:
    for path, dirs, files in os.walk(paths):
        for f in files:
            filename = os.path.join(path, f)
            img1 = mpimg.imread(filename)
            mpimg.imsave(des_path + f[:-4] + '__{}.png'.format(color), img1)
    color += 1
