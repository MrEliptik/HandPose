import numpy as np
import cv2
import os
from utils import detector_utils as detector_utils
import tensorflow as tf


print("Enter a name for the pose you want to add :")
name_pose = input()

print("You'll now be prompted to record the pose you want to add. \n \
        Please place your hand beforehand facing the camera, and press any key when ready. \n \
        When finished press 'q'.")

input()

if not os.path.exists('Poses/' + name_pose):
    os.makedirs('Poses/' + name_pose)

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Poses/' + name_pose + '/' + name_pose + '.avi', fourcc, 25.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # write the frame
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

vid = cv2.VideoCapture('Poses/' + name_pose + '/' + name_pose + '.avi')

# Check if the video
if (vid.isOpened() == False):
    print("Error opening video stream or file")


print(">> loading frozen model for worker")
detection_graph, sess = detector_utils.load_inference_graph()
sess = tf.Session(graph=detection_graph)
iter = 0
# Read until video is completed
while(vid.isOpened()):
    # Capture frame-by-frame
    ret, frame = vid.read()
    if ret == True:
        # Resize and convert to RGB for NN to work with
        frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect object
        boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

        # get region of interest
        res = detector_utils.get_box_image(1, 0.2, scores, boxes, 320, 180, frame)

        # Save cropped image 
        if(res is not None):
            
            cv2.imwrite('Poses/' + name_pose + '/' + name_pose + '_' + str(iter) + '.png', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

        iter += 1

    # Break the loop
    else:
        break
vid.release()