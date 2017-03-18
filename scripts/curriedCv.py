import cv2
import numpy as np

from doc2text import rotate, compute_skew, estimate_skew

from toolz.functoolz import pipe
from functools import partial
from toolz import curry


def load(path):
    return cv2.imread(path)


def float_to_uint8(image):
    return (image * 255).astype(np.uint8)


def threshold(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def threshold_params(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]


def canny(image):
    high = threshold_params(image); low = 0.5 * high
    return cv2.Canny(image, low, high)


def clahe_to_l_channel(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def descew(image):
    try:
        return rotate(image, compute_skew(estimate_skew(image)))
    except:
        return image


#  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
## -=-=-=-=-=-=-=-=-=-=-=-=-=-= CURRIED -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@curry
def resize(image, bigger_size=800):
    rows, cols = float(image.shape[0]), float(image.shape[1])
    p = cols/rows
    return cv2.resize(image, (int(p*bigger_size), bigger_size), interpolation=cv2.INTER_CUBIC)


@curry
def erode(image, size=(5,5)):
    """http://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html#erosion"""
    return cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, size))


@curry
def dilate(image, size=(5,5)):
    """http://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html#dilation"""
    return cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, size))


@curry
def m_open(image, size=(5,5)):
    """http://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html#opening"""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size))


@curry
def m_close(image, size=(5,5)):
    """http://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html#closing"""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, size))


@curry
def median(image, r=9):
    return cv2.medianBlur(image, r)


@curry
def level(image, black, white, gamma):
    a = (image - black) / (white - black)
    a[a > 1.0] = 1.0
    a[a < 0.0] = 0.0
    return np.power(a, gamma)


@curry
def dog(img, size=15, sigma_1=100, sigma_2=0):
    return cv2.filter2D(img, img.shape[-1], \
                        cv2.getGaussianKernel(size, sigma_1) - cv2.getGaussianKernel(size, sigma_2))


@curry
def normalize(image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F):
    return cv2.normalize(image, image, alpha=alpha, beta=beta, norm_type=norm_type, dtype=dtype)


def negate(image):
    return cv2.bitwise_not(image)


@curry
def gaussian(image, size=(15, 15), power=1):
    return cv2.GaussianBlur(image, size, power)
