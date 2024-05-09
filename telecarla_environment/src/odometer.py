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



def odometry_callback(msg, callback_args):
    # Callback for odometry data
    writer = callback_args["writer"]
    current_time = time.time()

    # x and y positions
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    # Calculate forward direction vector based on orientation (yaw)
    yaw_rad = np.deg2rad( msg.twist.twist.angular.z)
    forward_direction = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])

    # Extract velocity vector components
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    vz = msg.twist.twist.linear.z

    # Construct velocity vector
    velocity = np.array([vx, vy, vz])

    # Calculate dot product of velocity and forward direction to project velocity onto forward direction
    forward_velocity = np.dot(velocity, forward_direction)

    # Write info for this time point to CSV
    current_time = round(current_time, 1)
    writer.append_row([current_time, x, y, forward_velocity])





def camera_subscriber(writer):
    # Subscribe to odometry topic
    rospy.init_node('camera_subscriber', anonymous=True)

    odometry_topic = '/carla/ego_vehicle/odometry'

    rospy.Subscriber(odometry_topic, Odometry, odometry_callback, callback_args={'writer': writer})

    rospy.spin()


class CSVWriter:
    # Class to write info to csv file
    def __init__(self, directory, file_name):
        # Initialize writer to append to a file
        self.file_path = os.path.join(directory, file_name)
        self.append_row(["time", "x pos", "y pos", "speed"])

    def append_row(self, values):
        # Append a new row
        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(values)

if __name__ == '__main__':
    try:
        writer = CSVWriter("/home/oem/Desktop/csv_files", "server_no.csv")
        camera_subscriber(writer)
    except rospy.ROSInterruptException:
        pass