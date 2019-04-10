import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IEPlugin

def readIRModels(bin_path, xml_path):
    net = IENetwork(model=xml_path, weights=bin_path)

    return net

def prepareImage(im, net):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    im = cv2.flip(im, 1)

    # Reshape
    res = cv2.resize(im, (28,28), interpolation=cv2.INTER_AREA)

    # Convert to float values between 0. and 1.
    res = res.astype(dtype="float64")
    res = res / 255
    res = np.reshape(res, (1, 28, 28))

    input_blob = next(iter(net.inputs))

    n, c, h, w = net.inputs[input_blob].shape
    prepimg = np.ndarray(shape=(n, c, h, w))

    # Change data layout from HW to NCHW
    prepimg[0,0,:,:] = res

    return prepimg

def loadToDevice(net, _device="MYRIAD"):
    # Plugin initialization for specified device
    plugin = IEPlugin(device=_device)

    # Loading model to the plugin
    exec_net = plugin.load(network=net)

    return exec_net

def infer(exec_net, input_blob, im):
    return exec_net.infer(inputs={input_blob: im})

