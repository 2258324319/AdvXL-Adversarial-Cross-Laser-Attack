from scipy import stats
import math
from PIL import Image
import numpy as np
import cv2
import os
import threading
from time import sleep
from numba import jit
import time



def img_simulate(x, y, width, rotate, img_path):
    """
    在原图片上根据参数添加激光模拟图像
    （调用了light_sim函数）
    :param x:
    :param y:
    :param rotate:
    :param img_path:
    :return:
    """
    start_time = time.time()
    mask_path = './tmpimg/mask_'+str(start_time)+".jpg"
    Image.fromarray(light_sim(x, y, width, rotate, img_path)).save(mask_path)

    img_sim_mask = cv2.imread(mask_path)
    img = cv2.imread(img_path)
    # imgadd = cv2.add(img, img_sim_mask)
    imgadd = cv2.addWeighted(img, 1, img_sim_mask, 2, 0)
    end_time = time.time()
    #print("cost", end_time-start_time)
    cv2.imwrite('./tmpimg/' + str(start_time) + '_simulated.jpg', imgadd)
    return './tmpimg/' + str(start_time) + '_simulated.jpg'

# @jit(nopython=False)
def light_sim(x, y, width, rotate, img_path):
    """
    产生一个对应参数的蒙版
    :param x:
    :param y:
    :param rotate:
    :param img_path:
    :return:激光遮罩层的numpy格式
    """
    #print("light sim processing", img_path, '......')
    img = Image.open(img_path)
    try:
        img = img.convert("RGB") # 将一个4通道转化为rgb三通道
    except Exception as e:
        print("light_sim error, convert failed.")
    img_mat = np.array(img)
    mask_mat = np.zeros_like(img_mat)
    w = len(mask_mat[0])
    h = len(mask_mat)
    key_point = (x, y)
    # y = kx + (y0-kx0)
    k1 = math.tan(math.radians(rotate))
    k2 = math.tan(math.radians(rotate+90))
    # distance1 = abs(k1*x0-y0+(y-k*x))
    loop(w, h, x, y, k1, k2, width, mask_mat)
#    for x0 in range(w):
#        for y0 in range(h):
#            distance1 = abs(k1 * x0 - y0 + (y - k1 * x)) / math.sqrt(k1**2 + 1)
#            distance2 = abs(k2 * x0 - y0 + (y - k2 * x)) / math.sqrt(k2**2 + 1)
#            attenuation1 = attenuation(distance1, width)
#            attenuation2 = attenuation(distance2, width)
#
#            # 数组下标和坐标轴的x,y是反过来的
#            if distance1 < width*2 or distance2 < width*2:
#                if mask_mat[y0, x0, 1]==0:
#                    # 这个if判断是为了防止重复的写入导致溢出
#                    mask_mat[y0, x0, 0] += max(attenuation1[0],attenuation2[0])
#                    mask_mat[y0, x0, 1] += max(attenuation1[1],attenuation2[1])
#                    mask_mat[y0, x0, 2] += max(attenuation1[2],attenuation2[2])


    return mask_mat

@jit(nopython=True)
def loop(w, h ,x, y, k1, k2, width, mask_mat):
    for x0 in range(w):
        for y0 in range(h):
            distance1 = abs(k1 * x0 - y0 + (y - k1 * x)) / math.sqrt(k1**2 + 1)
            distance2 = abs(k2 * x0 - y0 + (y - k2 * x)) / math.sqrt(k2**2 + 1)
            attenuation1 = attenuation(distance1, width)
            attenuation2 = attenuation(distance2, width)

            # 数组下标和坐标轴的x,y是反过来的
            if distance1 < width*2 or distance2 < width*2:
                if mask_mat[y0, x0, 1]==0:
                    # 这个if判断是为了防止重复的写入导致溢出
                    mask_mat[y0, x0, 0] += max(attenuation1[0],attenuation2[0])
                    mask_mat[y0, x0, 1] += max(attenuation1[1],attenuation2[1])
                    mask_mat[y0, x0, 2] += max(attenuation1[2],attenuation2[2])

@jit(nopython=True)
def attenuation(distance, width):
    """
    模拟某点相对于激光线距离的高斯衰减
    distance: 某点相对于光线的距离
    :return:
    """
    fluorescence_ratio = 0.8
    half_width = width*0.7
    if distance > half_width*fluorescence_ratio:
        top = norm_indensity(0, 0, 4)
        # 使光强在
        #return (0, 255 * (stats.norm.pdf(distance, loc=half_width * fluorescence_ratio, scale=5)/top), 0)
        return (255 * (norm_indensity(distance, half_width * fluorescence_ratio, 5))/top, 0, 0)
    else:
        return (255, 0, 0)

@jit(nopython=True)
def norm_indensity(x, miu, sigma):
    return  (1/(math.sqrt(2*math.pi)*sigma)) * math.e**(-((x-miu)**2)/2*sigma**2)


#start_time = time.time()
#img_simulate(4, 59, 10, 20, 'dataset/110.jpg')
#end_time = time.time()
#print("cost", end_time-start_time)

