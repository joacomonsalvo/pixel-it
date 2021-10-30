import numpy as np
import cv2.cv2
import tkinter as tk
from sklearn.cluster import KMeans
from tkinter import filedialog


h, w = 64, 64
n_colours = 8
window_name = "image"
wait_key = 0

ROOT = tk.Tk().withdraw()
IMG_PATH = cv2.cv2.imread(filedialog.askopenfilename())

class Pixel:

    @classmethod
    def color_clustering(cls, idx, image, k):
        cluster_values = []
        for _ in range(0, k):
            cluster_values.append([])
        
        for r in range(0, idx.shape[0]):
            for c in range(0, idx.shape[1]):
                cluster_values[idx[r][c]].append(image[r][c])

        img_c = np.copy(image)

        cluster_averages = []
        for i in range(0, k):
            cluster_averages.append(np.average(cluster_values[i], axis=0))

        for r in range(0, idx.shape[0]):
            for c in range(0, idx.shape[1]):
                img_c[r][c] = cluster_averages[idx[r][c]]
                
        return img_c


    @classmethod
    def segment_img_clr_rgb(cls, image, k):
        
        img_c = np.copy(IMG_PATH)
        
        h = image.shape[0]
        w = image.shape[1]
        
        img_c.shape = (image.shape[0] * image.shape[1], 3)
        kmeans = KMeans(n_clusters=k, random_state=0).fit_predict(img_c)
        kmeans.shape = (h, w)

        return kmeans


    @classmethod
    def k_means_image(cls, image, k):
        idx = Pixel.segment_img_clr_rgb(image, k)
        return Pixel.color_clustering(idx, image, k)


    @classmethod
    def pixelate(cls, image, w, h):
        height, width = image.shape[:2]
        temp = cv2.cv2.resize(IMG_PATH, (w, h), interpolation=cv2.cv2.INTER_LINEAR)
        return cv2.cv2.resize(temp, (width, height), interpolation=cv2.cv2.INTER_NEAREST)


if __name__ == '__main__':
    obj = Pixel()
    IMG_PATH = Pixel.pixelate(IMG_PATH, h, w)
    IMG_PATH = Pixel.k_means_image(IMG_PATH, n_colours)
    cv2.cv2.imshow(window_name, IMG_PATH)
    cv2.cv2.waitKey(wait_key)
