import numpy as np
import cv2
import os


print("Enter a name for the pose you want to add :")
name_pose = input()

print("You'll now be prompted to record the pose you want to add. \n \
        Please place your hand beforehand facing the camera, and press any key when ready. \n \
        When finished press 'q'.")

input()

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(name_pose + '.avi', fourcc, 25.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        # write the flipped frame
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

print("The video of the gesture will be shown to you :")

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(name_pose + '.avi')

# Check if the video
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv2.imshow(name_pose, frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

print("Pose sequences ok? y/n")
# Press Q on keyboard to  exit
if cv2.waitKey(2500) & 0xFF == ord('n'):
    #Handle no
    pass

cap = cv2.VideoCapture(name_pose + '.avi')

# Check if the video
if (cap.isOpened() == False):
    print("Error opening video stream or file")

if not os.path.exists(name_pose):
    os.makedirs(name_pose)

iter = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv2.imwrite(name_pose+"/im"+str(iter)+".png", frame)
        iter += 1

    # Break the loop
    else:
        break
cap.release()
