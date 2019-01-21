import cv2
import os
from os.path import isfile, join

print(os.listdir("Poses/"))

poses = os.listdir('Poses/')

for pose in poses:
    print(">> Working on pose : " + pose)
    subdirs = os.listdir('Poses/' + pose + '/')
    for subdir in subdirs:
        files = os.listdir('Poses/' + pose + '/' + subdir + '/')
        print(">> Working on examples : " + subdir)
        for file in files:
            if(file.endswith(".png")):
                path = 'Poses/' + pose + '/' + subdir + '/' + file
                # Read image
                im = cv2.imread(path)

                height, width, channels = im.shape
                if not height == width == 28:
                    # Resize image
                    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
                    # Write image
                    cv2.imwrite(path, im)