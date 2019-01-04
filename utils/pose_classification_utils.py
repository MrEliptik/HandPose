import cv2
import numpy as np

def classify(model, im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Reshape
    res = cv2.resize(im, (28,28), interpolation=cv2.INTER_AREA)
    res = np.reshape(res, (1, 28, 28, 1))

    # Convert to float values between 0. and 1.
    res = res.astype(dtype="float32")
    res /= 255
    prediction = model.predict(res)

    return prediction[0]