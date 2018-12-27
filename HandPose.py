import numpy as np
import cv2
import scipy.misc
import tensorflow as tf
from threading import Thread

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d


def handPose(frame):
    image_raw = scipy.misc.imresize(frame, (240, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v,\
        keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                             keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                            feed_dict={image_tf: image_v})

    hand_scoremap_v = np.squeeze(hand_scoremap_v)
    image_crop_v = np.squeeze(image_crop_v)
    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

    # post processing
    image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

    # Display the resulting frame
    cv2.imshow('crop', image_crop_v)


# network input
image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
# left hand (true for all samples provided)
hand_side_tf = tf.constant([[1.0, 0.0]])
evaluation = tf.placeholder_with_default(True, shape=())

# build network
net = ColorHandPose3DNetwork()
hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(
        image_tf, hand_side_tf, evaluation)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# initialize network
net.init(sess)

cap = cv2.VideoCapture(0)

while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        thread = Thread(target=handPose, args=(frame,))
        thread.start()
        

        

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

