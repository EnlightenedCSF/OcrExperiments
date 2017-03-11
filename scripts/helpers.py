import cv2
import numpy as np
from matplotlib import pyplot as plt

from functools import partial


to_gray = partial(cv2.cvtColor, code=cv2.COLOR_RGB2GRAY)  # RGB -> Grayscale
to_color = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2RGB) # Grayscale -> RGB

convert_to_display = lambda image: image \
                        if len(image.shape) == 3 \
                        else to_color(image)

# отрисовка
def show(i1, i2=None, i3=None, i4=None):
    t = 110
    imgs = [i1]
    for u in [i2, i3, i4]:
        if u is not None:
            imgs.append(u)
            t += 10
    plt.figure()
    for index, img in enumerate(imgs):
        plt.subplot(t + index + 1)
        plt.imshow(convert_to_display(img))
    plt.show()


def get_countour_bounding_rect(contours, index):
    """Принимает массив контуров и индекс - один из контуров.
        Возвращает прямоугольник вокруг контура
    """
    x_min = min(cs[index][:,0][:,0])
    x_max = max(cs[index][:,0][:,0])
    y_min = min(cs[index][:,0][:,1])
    y_max = max(cs[index][:,0][:,1])
    return (x_min, y_min), (x_max, y_max)


def _get_biggest_contour(cs):
    """Принимает массив контуров, возвращает самый большой"""
    area = -1
    for i,c in enumerate(cs):
        s = cv2.contourArea(c)
        if s > area:
            area = s
            best = c
    return best


def _get_best_contour(image):
    """Находит все контуры на изображении, берет самый большой"""
    _, cs, _ = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    page = _get_biggest_contour(cs) # !!! GOVNOCODE
    cs.remove(page)                 # !!! GOVNOCODE
    best = _get_biggest_contour(cs)
    return cv2.boundingRect(best)


def cut(image, blacks):
    x,y,dx,dy = _get_best_contour(blacks)
    return image[y:y+dy, x:x+dx]


def draw_lines(image, lines):
    """Принимает линии, полученные в результате Hough line detection. Рисует их на том же изображении"""
    if lines.shape[2] == 2:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 2)
    else:
        a,b,c = lines.shape
        for i in range(a):
            cv2.line(image,
                     (lines[i][0][0], lines[i][0][1]),
                     (lines[i][0][2], lines[i][0][3]),
                     (0, 0, 255), 3, cv2.LINE_AA)


def _get_optimal_erode(image):
    def count_contours(erode_level, image, ):
        p = erode(image, erode_level)
        _, contours, _ = cv2.findContours(p, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)

    f = partial(count_contours, image=result)
    erodes = [x+1 for x in range(15)]
    cntrs = [f(e) for e in erodes]
    return erodes[np.argmin(cntrs)]

def erode_optimal(image):
    """применяет erode такой величины, чтобы количество черных контуров было минимальным (типа все, что надо - слилось)
        P.s. Кажется, тоже не использую
    """
    return erode(image, size=_get_optimal_erode(image))




# image = load('data/1.jpg')
# nice_gray = pipe(image,
#                  resize, to_gray, threshold, cv2.equalizeHist,
#              )
# black_blocks = pipe(nice_gray,
#                    m_open(size=(7,1)), m_close(size=(1,7)), median(r=5),
#                    erode(size=(25,1)), erode(size=(1,21)), m_open
#                   )
# result = cut(nice_gray, black_blocks)

# p = pipe(nice_gray, m_open(size=(15,1)), m_close(size=(1,7)), erode(size=(15,15)))
# show(nice_gray, p)
