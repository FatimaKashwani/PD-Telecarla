#!/usr/bin/env python
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.transform import (hough_line, hough_line_peaks)
import numpy as np
import cv2
from skimage.feature import canny
from skimage import io, color
from matplotlib import pyplot as plt
import pygame
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import csv
import time
from scipy.interpolate import InterpolatedUnivariateSpline

import os
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from collections import deque
from scipy.ndimage import gaussian_filter1d

# global variable used to store data from callback
global_seg = None


def resize_image(image, scale_factor=2):
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    return cv2.resize(image, (new_width, new_height))

def apply_hough_transform(image, lines_mask):
    # applying hough transform to image
    # image: road segmentation mask
    # lines_mask: lane lines segmentation mask
    # this function uses hough transform to find the lane line thus splitting the road mask
    if lines_mask is None: return None

    gray_lane_lines_mask = cv2.cvtColor(lines_mask, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection (Canny) to lane lines mask
    edges = cv2.Canny(gray_lane_lines_mask, 50, 150)

    # Apply Hough transform to detect lines in the lane lines mask
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=65)

    # Assuming only one line detected for simplicity
    if lines is not None:

        line_mask = np.zeros_like(image)

        for line in lines:

            rho, theta = line[0][0], line[0][1]

            # Convert polar coordinates to Cartesian coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            #cv2.imshow("Line", line_mask)
            #cv2.waitKey(1)

    else:

        return None


    region_mask = np.zeros_like(image, dtype=np.uint8)


    y_indices, x_indices = np.indices(image.shape[:2])
    region_mask = np.where(a * x_indices + b * y_indices - rho > 1, 255, 0).astype(np.uint8)
    # Use the mask values to set the corresponding pixels in the region_mask
    region_mask = region_mask[:, :, np.newaxis]
    region_mask = np.repeat(region_mask, 3, axis=2)
    
    # Create masks for the regions above and below the line

    above_line_mask = cv2.bitwise_and(image, cv2.bitwise_not(region_mask))
    below_line_mask = cv2.bitwise_and(image, region_mask)

    above_line_mask = cv2.cvtColor(above_line_mask, cv2.COLOR_BGR2GRAY)
    below_line_mask = cv2.cvtColor(below_line_mask, cv2.COLOR_BGR2GRAY)


    above_area = cv2.countNonZero(above_line_mask)
    below_area = cv2.countNonZero(below_line_mask)

    # Determine which section has the most road pixels
    # With our camera angle (driver POV), the ego lane appears larger and contains more pixels
    larger_mask = above_line_mask if above_area > below_area else below_line_mask


    return larger_mask


def round_to_nearest_interval(timestamp, interval):
    # rounds timestamps for defining time points
    return round(timestamp / interval) * interval


def image_callback(msg):

    
    try:
        # directories for saving images when using calculate_dsc
        mask_dir = "/home/oem/Desktop/test_150ms/server_mask/"
        rgb_dir = "/home/oem/Desktop/test_150ms/server_rgb/"
        current_time = time.time()
        # Round timestamp to nearest 10ms interval
        aligned_timestamp = round_to_nearest_interval(current_time, 0.1)
        aligned_timestamp_str = "{:.1f}".format(aligned_timestamp)
        # Record frame with aligned timestamp
        image_file_name = f"{aligned_timestamp_str}.png"
        # obtain image from message
        cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
        print(aligned_timestamp)

        # Show RGB image
        if cv_image is not None:
            cv2.imshow("Server View", cv_image)
            cv2.waitKey(1)
        
        # Show road segmentation mask
        if global_seg is not None:
            cv2.imshow("Server Segmentation Mask", global_seg)
            cv2.waitKey(1)

    except Exception as e:
        rospy.logerr("Error processing and publishing image: {}".format(e))


def draw_mask(image):
    # Draw road and lines masks
    # purple = road color in cityscapes pallette
    # yellow = lane lines color
    purple = np.array([128, 64, 128], dtype=np.uint8)

    yellow = np.array([50, 234, 157], dtype=np.uint8)

    # Create white mask for road area
    image_array = np.zeros_like(image)
    mask = np.all((image == purple), axis=-1)
    mask = np.expand_dims(mask, axis=-1)  
    mask = np.tile(mask, (1, 1, image_array.shape[2])) 
    image_array[mask] = 255

    # Create white mask for lane lines
    # (this can be used for hough trtansform)
    image_array2 = np.zeros_like(image_array)
    mask2 = np.all((image == yellow), axis=-1)
    mask2 = np.expand_dims(mask2, axis=-1)  
    mask2 = np.tile(mask2, (1, 1, image_array2.shape[2])) 
    image_array2[mask2] = 255

    # returns road mask
    return image_array



def seg_callback(msg):
    # Callback for segmentation image
    # Draws and stores road mask
    seg = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    global global_seg
    global_seg = draw_mask(seg)



def camera_subscriber():
    # Subscribes to RGB and segmentation topics
    rospy.init_node('camera_subscriber', anonymous=True)

    camera_topic = "/carla/ego_vehicle/front/image"
    seg_topic = "/carla/ego_vehicle/front_left/image"

    rospy.Subscriber(seg_topic, Image, seg_callback)
    rospy.Subscriber(camera_topic, Image, image_callback)


    rospy.spin()

if __name__ == '__main__':
    try:
        camera_subscriber()
    except rospy.ROSInterruptException:
        pass
