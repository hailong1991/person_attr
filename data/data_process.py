import os
from PIL import Image
from torch.utils import data
import numpy as np
import cv2
from torchvision import transforms as T
import random

import torch
import matplotlib.pyplot as plt

def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def BGR2HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def HSV2BGR(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def RandomBrightness(bgr, probability=0.5):
    if random.random() < probability:
        hsv = BGR2HSV(bgr)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def RandomSaturation(bgr, probability=0.5):
    if random.random() < probability:
        hsv = BGR2HSV(bgr)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def RandomHue(bgr, probability=0.5):
    if random.random() < probability:
        hsv = BGR2HSV(bgr)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def randomBlur(bgr, probability=0.5, area=(5,5)):
    if random.random() < probability:
        bgr = cv2.blur(bgr, area)
    return bgr


def randomShift(bgr, probability=0.5):
    # 平移变换
    if random.random() < probability:
        height, width, c = bgr.shape
        after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
        after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
        shift_x = random.uniform(-width * 0.2, width * 0.2)
        shift_y = random.uniform(-height * 0.2, height * 0.2)
        # print(bgr.shape,shift_x,shift_y)
        # 原图像的平移
        if shift_x >= 0 and shift_y >= 0:
            after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x), :]
        elif shift_x >= 0 and shift_y < 0:
            after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x), :]
        elif shift_x < 0 and shift_y >= 0:
            after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):, :]
        elif shift_x < 0 and shift_y < 0:
            after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):, -int(shift_x):, :]
        return after_shfit_image
    return bgr

def randomScale(bgr, probability=0.5):
    # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
    if random.random() < probability:
        scale = random.uniform(0.8, 1.2)
        height, width, c = bgr.shape
        bgr = cv2.resize(bgr, (int(width * scale), height))
        return bgr
    return bgr


def randomCrop(bgr, probability=0.5):
    if random.random() < probability:
        height, width, c = bgr.shape
        h = random.uniform(0.6 * height, height)
        w = random.uniform(0.6 * width, width)
        x = random.uniform(0, width - w)
        y = random.uniform(0, height - h)
        x, y, h, w = int(x), int(y), int(h), int(w)
        img_croped = bgr[y:y + h, x:x + w, :]
        return img_croped
    return bgr


def subMean(bgr, mean):
    mean = np.array(mean, dtype=np.float32)
    bgr = bgr - mean
    return bgr

def random_flip(im, probability=0.5):
    if random.random() < probability:
        im_lr = np.fliplr(im).copy()
        return im_lr
    return im


def random_bright(im, delta=16):
    alpha = random.random()
    if alpha > 0.3:
        im = im * alpha + random.randrange(-delta, delta)
        im = im.clip(min=0, max=255).astype(np.uint8)
    return im