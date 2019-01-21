import cv2
import numpy as np

def classify(model, im):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    im = cv2.flip(im, 1)

    '''
    cv2.imshow('To Classify', im)
    cv2.waitKey(0)
    cv2.destroyWindow('To Classify')
    '''

    # Reshape
    res = cv2.resize(im, (28,28), interpolation=cv2.INTER_AREA)

    # Convert to float values between 0. and 1.
    res = res.astype(dtype="float32")
    res = res / 255
    res = np.reshape(res, (1, 28, 28, 1))

    prediction = model.predict(res)

    return prediction[0] 

if __name__ == "__main__":

    from keras.models import load_model

    print(">> loading keras model for pose classification")
    model = load_model('cnn/models/hand_poses_2poses_14.h5')

    '''
    # Fist
    print('<< FIST >>')
    im = cv2.imread("Poses/Fist/Fist_1/Fist_1_1302.png")
    print(classify(model, im))

    # Dang
    print('<< DANG >>')
    im = cv2.imread("Poses/Dang/Dang_1/Dang_1_1223.png")
    print(classify(model, im))

    # Four
    print('<< FOUR >>')
    im = cv2.imread("Poses/Four/Four_1/Four_1_867.png")
    print(classify(model, im))
    '''
    
    # Startrek
    print('<< Startrek >>')
    im = cv2.imread("Poses/Startrek/Startrek_1/Startrek_1_867.png")
    print(classify(model, im))

    # Palm
    print('<< Palm >>')
    im = cv2.imread("Poses/Palm/Palm_1/Palm_1_867.png")
    print(classify(model, im))