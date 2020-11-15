import cv2
import numpy as np
from collections import Counter


def define_colors():
    red = ([17, 15, 100], [50, 56, 200])
    colors = {
    "red": red
    }
    return colors

def import_image(image_path):
    image = cv2.imread(image_path)
    return image

def find_most_prominent(colors,image):
    colors_present = {
    }
    for color in colors:
        lower = np.array(colors[color][0], dtype = "uint8")
        upper = np.array(colors[color][1], dtype = "uint8")
        mask = cv2.inRange(image, lower, upper)
        pixel_counter = 0
        for row in mask:
            for pixel in row:
                if pixel != 0:
                    pixel_counter += 1
        colors_present[color] = pixel_counter
    c = Counter(colors_present)
    max = c.most_common(1)
    return max

def find_rgb(image):
    counter = 0
    r = 0
    g = 0
    b = 0
    for row in image:
        for pixel in row:
            r += pixel[2]
            g += pixel[1]
            b += pixel[0]
            counter+=1
    average_rgb = [r/counter, g/counter, b/counter]
    return average_rgb

def find_hsv(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = 0
    s = 0
    v = 0
    counter = 0
    for row in image:
        for pixel in row:
            v += pixel[2]
            s += pixel[1]
            h += pixel[0]
            counter+=1
    average_hsv = [h/counter, s/counter, v/counter]
    return average_hsv
