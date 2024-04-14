import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def otsu_binarization(img, th=128):
    H,W=img.shape
    max_sigma=0
    max_t=0

    # determine threshold
    for _t in range(1, 255):
        v0 = img[np.where(img < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = img[np.where(img >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # Binarization
    # print("threshold >>", max_t)
    th = max_t
    img[img < th] = 0
    img[img >= th] = 1

    return img


# img=Image.open('F:/files/datas/1_Pre_test/pic_1.jpg')
# img=np.array(img)
#
# plt.subplot(1,2,1)
# plt.imshow(img,cmap='gray')
#
# plt.subplot(1,2,2)
# img=otsu_binarization(img)
# print(img.sum())
# plt.imshow(img,cmap='gray')
# plt.show()
