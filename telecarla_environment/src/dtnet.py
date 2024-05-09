#!/usr/bin/env python
import os
import rospy
from sensor_msgs.msg import Image as rosimage
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.transform import (hough_line, hough_line_peaks)
import numpy as np
from sensor_msgs.msg import Image

import cv2
from skimage.feature import canny
from skimage import io, color
from matplotlib import pyplot as plt
import pygame
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import tempfile
import csv
import time
from scipy.interpolate import InterpolatedUnivariateSpline

import os
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from collections import deque
from scipy.ndimage import gaussian_filter1d

from carla_msgs.msg import (
    CarlaEgoVehicleStatus
)


import argparse
import os
import pygame
from tkinter import Image
from mmseg.datasets import DATASETS
import datasets
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F
from scipy.signal import savgol_filter
#from mmseg.models.segmentors.encoder_decoder import whole_inference
#from mmseg.datasets import replace_ImageToTensor as rep
from mmcv.parallel import collate, scatter
#from mmseg.models.segmentors.encoder_decoder import inference as inferr
import pandas as pd
from mmcv.utils import DictAction
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader
from mmseg.apis.inference import inference_segmentor
# from mmseg.registry import DATASETS
# from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.datasets import build_dataset as mmseg_build_dataset
from mmseg.datasets.builder import DATASETS as MMSEG_DATASETS

from mmdet.datasets import build_dataset as mmdet_build_dataset
from mmdet.datasets.builder import DATASETS as MMDET_DATASETS
from mmseg.models import build_segmentor
from mmseg.apis import inference

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (
    build_dataloader,
    #replace_ImageToTensor,
)
from mmdet.models import build_detector
import torchvision

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random 

from pathlib import Path
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.ndimage import convolve1d

from scalabel.label.io import save
from scalabel.label.transforms import bbox_to_box2d
from scalabel.label.typing import Dataset, Frame, Label 
from torch.profiler import profile, record_function, ProfilerActivity
from mmdet.apis import init_detector, inference_detector
from mmseg.apis import init_segmentor
import os.path as osp
from typing import List
from scipy.interpolate import splprep, splev
import time
from sklearn.metrics import mean_squared_error
import math
from std_msgs.msg import String

MODEL_SERVER = "https://dl.cv.ethz.ch/bdd100k/det/models/"

frame_count = 0

# Frame skipping for real-time processing
# Skip frames to mitigate processing delay
SKIP = 5

# Predicted road mask from segmentation callback
global_image = None

# Speedometer and odometry data if needed
global_speedometer = None
global_odometry = None

def Draw_Det_Mask(outputs,dim):
    # Draw a mask of detection boxes
    
    # Empty mask 
    mask_frame = np.zeros(dim, dtype=np.uint8)

    # Iterate over bounding boxes
    for cat_idx, bbox in enumerate(outputs[0][0]):
 
        bbox_xyxy = bbox[:4].astype(int)
        class_idx = int(bbox[4])
        class_mask = np.zeros(dim, dtype=np.uint8)
        class_mask[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = 1

        #frame_mask[class_mask > 0] = class_idx
        # Create a binary mask of the bounding box with a unique class ID
        mask_bb = np.zeros_like(mask_frame, dtype=np.uint8)
        mask_bb[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = cat_idx + 1

        # Merge the bounding box mask into the frame mask
        mask_frame = cv2.bitwise_or(mask_frame, mask_bb)
    colormap = create_custom_colormap(10)
    colored_mask = cv2.applyColorMap(mask_frame, colormap)
        
    return mask_frame

def Draw_Seg_Mask(outputs):
    # Draw a mask of road segmentation
    # Pallette equivalent to blue for ego lane and green for alternate
    #PALETTE = [[219, 94, 86], [86, 211, 219], [0, 0, 0]]
    PALETTE = [[255,0,0], [0,255,0], [0, 0, 0]]
    # Create an empty mask image
    outputs = outputs[0]
    height, width = outputs.shape[:2]
    mask_image = np.zeros((height, width, 3), dtype=np.uint8)

    for pid in np.unique(outputs):
        mask = (outputs == pid).astype(np.uint8)
        # Set the corresponding pixel values in the mask image
        mask_image[mask != 0] = PALETTE[pid]

    return (mask_image)

def create_custom_colormap(num_classes):
    # Generate a set of evenly spaced colors for the number of classes
    hsv_values = [(i * 180 / num_classes, 255, 255) for i in range(num_classes)]
    colors = [list(map(lambda x: int(x), cv2.cvtColor(np.uint8([[hsv_values[i]]]), cv2.COLOR_HSV2BGR)[0][0])) for i in range(num_classes)]

    # Assign black color (0, 0, 0) to the background class (class 0)
    colors[0] = [0, 0, 0]

    # Create an empty colormap image
    colormap_image = np.zeros((256, 1, 3), dtype=np.uint8)

    # Assign colors to the colormap based on class indices
    for class_idx, color in enumerate(colors):
        colormap_image[class_idx] = color

    return colormap_image

def CombineMasks (seg, det):
    # Combine detection and segmentation masks
    # Define the RGB to Grayscale mapping
    rgb_to_grayscale_map = {

        (255, 0, 0) : 50,
        (0, 255, 0) : 30

    }
    grayscale_image = det
    corresponding_rgb_image = seg
    # Convert RGB images to Grayscale as per the mapping

    for rgb_color, grayscale_value in rgb_to_grayscale_map.items():
            mask = np.all(corresponding_rgb_image == np.array(rgb_color), axis=-1)
            corresponding_rgb_image[mask] = [grayscale_value]

    mask = grayscale_image != 0


    corresponding_rgb_image[mask, 0] = grayscale_image[mask]  
    corresponding_rgb_image[mask, 1] = grayscale_image[mask]  
    corresponding_rgb_image[mask, 2] = grayscale_image[mask]  

    # Create the "Mask image" containing only masks of pixel values corresponding to 30 and 50 in the "Fused Image"
    mask_image = np.zeros_like(corresponding_rgb_image, dtype=np.uint8)
    mask_image[corresponding_rgb_image == 30] = 1
    mask_image[corresponding_rgb_image == 50] = 2

    colored = cv2.applyColorMap(mask_image, create_custom_colormap(3))

    return colored

def init_segmentor(config, checkpoint=None, device='cuda:0'):
    # This overrides the method in mmseg.
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        #model.CLASSES = checkpoint['meta']['CLASSES']
        #model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def calculate_distance(pixel):
    # Calculate distance from pixel in depth image
    return (pixel[2] + pixel[1] * 256 + pixel[0] * 256 * 256) / (256 * 256 * 256 - 1) * 1000

def calculate_horizontal_distance(pixel):
    # Calculate distance from pixel in depth image
    # Returns the distance along the road rather than diagonal
    diagonal = (pixel[2] + pixel[1] * 256 + pixel[0] * 256 * 256) / (256 * 256 * 256 - 1) * 1000
    return np.sqrt(diagonal*diagonal - 1.7*1.7)

def find_closest_pixel(depth_column, threshold_distance, start_row=0, end_row=None):
    # Find closest point in the image to a specified distance
    # FIX ME this method doesnt display predicted distance concisely
    if end_row is None:
        end_row = len(depth_column) - 1

    closest_distance = float('inf')
    closest_pixel = None

    while start_row <= end_row:
        mid_row = (start_row + end_row) // 2
        distance = calculate_distance(depth_column[mid_row])

        if abs(distance - threshold_distance) < closest_distance:
            closest_distance = abs(distance - threshold_distance)
            
            closest_pixel = mid_row

        if distance < threshold_distance:
            start_row = mid_row + 1
            print("less")
            print("distance: ", distance)
            print("threshold distance: ", threshold_distance)
        elif distance > threshold_distance:
            end_row = mid_row - 1
            print("more")
            print("distance: ", distance)
            print("threshold distance: ", threshold_distance)
        else:
            
            break

    print("closest distance ", closest_distance)
    return closest_pixel

def find_nearest_pixel(column, distance):
    dists = []
    for item in column:
        dists.append(calculate_horizontal_distance(item))
    print(len(dists))
    closest_pixel = 449
    closest_diff = float('inf')
    
    for i in range(len(dists) - 1, -1, -1):
        diff = abs(dists[i] - distance)
        
        if diff <= closest_diff:
            closest_pixel = i
            closest_diff = diff
            print("closest dif: ",closest_diff)
            print("dif: ", diff)
        else:
            break  

    return closest_pixel
    

class Image_Processor:
    '''
    Class to pass an image through the dual transformer network.
    RGB, depth, and semantic segmentation frames are obtained from CARLA via callback functions.
    '''
    def __init__(self):
        self.odometry = None
        self.depth_image = None
        self.rgb_image = None
        self.seg_image = None
        self.binary_mask = None
        self.processed_image = None
        self.paths = []

        pygame.init()
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        self.joystick = joystick

        configseg = "/home/oem/bdd100k-models/drivable/configs/drivable/upernet_swin-s_512x1024_80k_drivable_bdd100k.py"
        configdet = "/home/oem/bdd100k-models/det/configs/det/faster_rcnn_swin-s_fpn_3x_det_bdd100k.py"

        torch.cuda.set_device(1)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        """Detection Model"""
        #Loading model config
        cfg_det = mmcv.Config.fromfile(configdet)
        if cfg_det.load_from is None:
            cfg_name = os.path.split(cfg_det)[-1].replace(".py", ".pth")
            cfg_det.load_from = MODEL_SERVER + cfg_name
        #if args.cfg_options is not None:
            #cfg_det.merge_from_dict(args.cfg_options)
        # set cudnn_benchmark
        if cfg_det.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True

        cfg_det.model.pretrained = None
        if cfg_det.model.get("neck"):
            if isinstance(cfg_det.model.neck, list):
                for neck_cfg in cfg_det.model.neck:
                    if neck_cfg.get("rfp_backbone"):
                        if neck_cfg.rfp_backbone.get("pretrained"):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg_det.model.neck.get("rfp_backbone"):
                if cfg_det.model.neck.rfp_backbone.get("pretrained"):
                    cfg_det.model.neck.rfp_backbone.pretrained = None
    



        #Build detector
        cfg_det.model.train_cfg = None
        modeldet = build_detector(cfg_det.model, test_cfg=cfg_det.get("test_cfg"))
        modeldet.to(1)
        fp16_cfg = cfg_det.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(modeldet)
        checkpoint = load_checkpoint(modeldet, cfg_det.load_from, map_location="cpu")
        #if args.fuse_conv_bn:
            #model = fuse_conv_bn(model)
        if "CLASSES" in checkpoint.get("meta", {}):
            modeldet.CLASSES = checkpoint["meta"]["CLASSES"]

    

        """Segmentation Model"""
        #Loading model config
        cfg_seg = mmcv.Config.fromfile(configseg)
        if cfg_seg.load_from is None:
            cfg_name = os.path.split(cfg_seg)[-1].replace(".py", ".pth")
            cfg_seg.load_from = MODEL_SERVER + cfg_name
        #if args.options is not None:
            #cfg_seg.merge_from_dict(args.options)
        # set cudnn_benchmark
        if cfg_seg.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True

        cfg_seg.model.pretrained = None
        cfg_seg.data.test.test_mode = True
    

        #Build segmentor
        cfg_seg.model.train_cfg = None
        modelseg = build_segmentor(cfg_seg.model, test_cfg=cfg_seg.get("test_cfg"))

        constructed_filename = str("/home/oem/bdd100k-models/upernet_swin-s_512x1024_80k_drivable_bdd100k.pth")#+cfg_seg.load_from.split()[-1].split("/")[-1])
        #print("Constructed Filename:", constructed_filename)

        modelseg = init_segmentor(cfg_seg, constructed_filename, device=0)
        modelseg.CLASSES = ("direct", "alternative", "background")
        modelseg.PALETTE = [[219, 94, 86], [86, 211, 219], [0, 0, 0]]
        #modelseg.PALETTE = [[0, 0, 255], [0, 255, 0], [0, 0, 0]]
        modelseg.to(1)

        self.segmentor = modelseg
        self.detector = modeldet

    # Get corresponding images
    # Images are obtained globally from callback functions
    def get_rgb(self):
        return self.rgb_image
    def get_depth(self):
        return self.depth_image
    def get_mask(self):
        return self.binary_mask
    def get_white_mask(self):
        if self.get_mask is None: return None
    
        blue_mask = np.all(self.get_mask() == [255, 0, 0], axis=-1)
        green_mask = np.all(self.get_mask() == [0, 255, 0], axis=-1)

        # White mask combines ego and alternate lanes
        # Used for evaluation with ground truth road mask
        white_mask = np.zeros_like(blue_mask, dtype=np.uint8)
        white_mask[(blue_mask | green_mask)] = 255

        return white_mask

    def get_processed(self):
        return self.processed_image
    
    def set_rgb(self, image):
        self.rgb_image = image
    def set_depth(self, image):
        self.depth_image = image
    def set_mask(self, image):
        self.binary_mask = image
    def set_processed(self, image):
        self.processed_image = image



    def predict(self):
        # Runs the rgb image through DTNet
        # rgb image should be set before calling
        start_time = rospy.get_time()

        # Convert image to tensor
        img_tensor = torch.from_numpy(self.rgb_image).float().permute(2, 0, 1) / 255.0  # Change format to channel-first
        img_tensor = img_tensor.unsqueeze(0).to(1)
        img_meta = [dict(
                filename="file",  # Set the filename as needed
                ori_shape=self.rgb_image.shape,
                img_shape=self.rgb_image.shape,
                pad_shape=self.rgb_image.shape,
                scale_factor=1.0,
                flip=False,
                batch_input_shape=(1, 3, self.rgb_image.shape[0], self.rgb_image.shape[1]),  # Adjust as needed
        )]

        # Pass through detection module
        with torch.no_grad():
            outputsdet = self.detector.forward_test([img_tensor], [img_meta], rescale=True)
        detector_time = rospy.get_time() - start_time
        rospy.loginfo(f"Detector processing time: {detector_time} seconds")
        det = Draw_Det_Mask(outputsdet, (self.rgb_image.shape[0], self.rgb_image.shape[1]))
        del outputsdet

        

        start_time = rospy.get_time()

        # Pass through segmentation module
        with torch.no_grad():
            outputsseg = self.segmentor.forward_test([img_tensor], [img_meta], rescale=True)
        segmentor_time = rospy.get_time() - start_time
        rospy.loginfo(f"Segmentor processing time: {segmentor_time} seconds")
        seg = Draw_Seg_Mask(outputsseg)
        del outputsseg


        # Combine results into one mask
        # Set mask
        comb = CombineMasks(seg, det)
        self.set_mask(comb)
        pygame.event.pump()

        total_time = rospy.get_time() - start_time
        rospy.loginfo(f"Total prediction time: {total_time} seconds")

    def find_center_path_smooth(self):
        # Draw path based on combined mask
        self.predict()

        binary_mask = self.binary_mask
        image = self.seg_image
        last_5_outs = self.paths
        if binary_mask.ndim == 3:
            rows, cols, _ = binary_mask.shape
        elif binary_mask.ndim == 2:
            rows, cols = binary_mask.shape
        else:
            return self.rgb_image
    
        center_locations = []

        # Obtain center points from white mask
        for i in range(rows):
            row = binary_mask[i, :]
            white_pixels = np.where(row == [255,0,0])[0]

           
            weights = np.ones_like(white_pixels)  
            if len(white_pixels) > 0:
                center_x = int(round(np.average(white_pixels, weights=weights)))
                center_locations.append((center_x, i))


        center_locations = np.array(center_locations)
        if len(center_locations) == 0:
            self.set_processed(self.rgb_image)
            return self.rgb_image
        # center_locations = np.array(center_locations)[5:-1, :]

        xo = center_locations[:, 0]
        yo = center_locations[:, 1]

        x = xo.copy()
        y = yo.copy()
        
        
        OUTns=(x,y)
        xypoints = center_locations.copy()
        
        # Subsample x and y
        step_size = len(x) // 10

        if step_size == 0:
            step_size = 1
        x = x[::step_size]
        y = y[::step_size]
        
        subsampled_xypoints = xypoints[::step_size]
        # print("points:",subsampled_xypoints)

        # Ensure the last points are the same
        x[-1] = xo[-1]
        y[-1] = yo[-1]
        
        xw=x
        yw=y
        xnn=x.copy()

               
        # Calculate the average 'out' if there are previous 'outs' to average
        if len(last_5_outs) > 1:
            # Determine the number of points to smooth (last 30%)
            num_points_to_smooth = int(len(last_5_outs[0]) * 0.5)
            starting_index = len(last_5_outs[0]) - num_points_to_smooth
            
            # Validate if the starting index is not negative
            starting_index = max(0, starting_index)

            # Extract the points to be smoothed and the remaining points
            points_to_smooth = [xnn[starting_index:] for xnn in last_5_outs if len(xnn) >= num_points_to_smooth]
            remaining_points = [xnn[:starting_index] for xnn in last_5_outs if len(xnn) >= num_points_to_smooth]
            # print("points2smooth:", points_to_smooth)
            
            if points_to_smooth:
                # Calculate the average of the points to be smoothed
                avg_smoothed_points = [sum(col) / len(col) for col in zip(*points_to_smooth)]

                # Get the actual remaining points from the current frame (not averaged)
                actual_remaining_points = xnn[:starting_index]

                # Combine the actual remaining and smoothed points
                smoothed_out = np.concatenate((actual_remaining_points, avg_smoothed_points))
        else:
            smoothed_out = xnn
            
        
        last_5_outs.append(smoothed_out)
        new_x = []
        new_y = []
        
        #v = global_speedometer if global_speedometer is not None else 0
        #rospy.loginfo(v)
        #acceleration = abs((pygame.joystick.Joystick(0).get_axis(1)-1)/2)
        #v2 = v1 + acceleration * tau
        #x2 = x1 + (v1 + v2)/2 * tau * 10
        #dist = x2 - x1
        #v = 1
        #tau = 0.1
        #dist = v*tau
        dist = 0
        #if theta is not None:
            #dist *= np.cos(theta)
        if dist < 0.001: dist = 0

        # Position Prediction
        pixel = self.depth_image[int(self.depth_image.shape[0]/2)][int(self.depth_image.shape[1]) - 1]

        pos_min = (pixel[2] + pixel[1] * 256 + pixel[0] * 256 * 256) / (256 * 256 * 256 - 1) * 1000
        pred_min = np.sqrt(pos_min*pos_min - 1.7*1.7)
        dist_from_cam = np.sqrt((pred_min + dist)*(pred_min + dist) + 1.7*1.7)
        if dist == 0: pos = pos_min
        else:

            trajectory = [(smoothed_out[i], y[i]) for i in range(min(len(smoothed_out), len(y)))]
            pixel_coor = find_closest_pixel([self.depth_image[int(y), int(x)] for x,y in trajectory], dist_from_cam)
            pixel = self.depth_image[int(pixel_coor[0])][int(pixel_coor[1])]
            pos = (pixel[2] + pixel[1] * 256 + pixel[0] * 256 * 256) / (256 * 256 * 256 - 1) * 1000

        pred_dist = np.sqrt(pos*pos - 1.7*1.7)
    
        distance = np.abs(pred_dist - pred_min)
        rospy.loginfo("START")
        #for pixel in global_depth_image[int(global_depth_image.shape[1]/2)]:
            #rospy.loginfo(pixel)
        rospy.loginfo("distance: {}".format(distance))
        #rospy.loginfo("distance1: {}".format(dist + pos_min))
        rospy.loginfo("dist: {}".format(dist))
        rospy.loginfo("dist from cam: {}".format(dist_from_cam))
        rospy.loginfo("pos: {}".format(pos))
        rospy.loginfo("pos_min: {}".format(pos_min))
        #rospy.loginfo("in meters: {}".format(pos))

        x_min = np.max(smoothed_out)
        theta = self.joystick.get_axis(0)
        new_x = []
        # Curvature Adjustment
        '''
        if theta != None:

                        theta = theta / 9.8 * 3.1
                        #print(theta)
                        
                        velocity = 1
                        xs = smoothed_out
                        ys = yw

                        #for (x, y) in trajectory[0]:
                        
                        for i in range(len(xs)):
                            x = xs[i]
                            y = ys[i]
                            #y_prime = int(y + 30*(x - x_min) * velocity * np.sin(theta))
                            x_prime = int(x - 10*(x - x_min) * velocity * np.sin(theta))
                            #print("compare" + str(y) + " "+ str(y_prime))
                            #if y_prime >= 1280: y_prime = 1279
                            #if y_prime >= binary_mask.shape[1]: y_prime = binary_mask.shape[1] - 1
                            
                            if x_prime >= binary_mask.shape[1]: x_prime = binary_mask.shape[1] - 1
                            #center_points[i] = (x, y_prime)
                            #y_prime = y

                            #if(binary_mask[x,y_prime] != (0,0,0)).all():

                            tolerance = 0
                            
                            if 0 <= y < binary_mask.shape[0] and 0 <= x_prime < binary_mask.shape[1]:
                                tolerance = 0
                                if np.array_equal(binary_mask[y, x_prime], [255, 0, 0]):
                                
                                
                                    new_x.append(y)
                                    new_y.append(x_prime)



                            else:
                                pass

                                
                            
                            #rospy.loginfo(str(x)  + " "+ str(x_prime))

        
        
        '''
        if len(new_x) >= 1:
            smoothed_out = np.array(new_x)

        # Smooth using Savitzky-Golay filter
        window_length = 7
        if window_length > len(smoothed_out):
            window_length = len(smoothed_out)
        polyorder = 2
        if polyorder >= window_length:
            # Adjust window_length to be greater than polyorder
            polyorder = window_length - 1
                
        xs = savgol_filter(smoothed_out, window_length, polyorder)

        # Further smooth using moving average
        window_size = len(xs) // 4
        if window_size == 0:
            window_size = len(xs)  # or choose another appropriate default value

        smooth_data = np.convolve(xs, np.ones(window_size)/window_size, mode='valid')
        if len(smooth_data) == 1 and len(xs) > 0:
            new_value = 2 * smooth_data[0] - xs[-1]
            smooth_data = np.array([smooth_data[0], new_value])


        # Interpolation back to the original length
        xq = np.linspace(0, len(smooth_data)-1, len(xo))
        cs = CubicSpline(range(len(smooth_data)), smooth_data)
        interpolated_smooth_data = cs(xq)

        # Round to nearest integer
        interpolated_smooth_data = np.round(interpolated_smooth_data)
        x = interpolated_smooth_data
        y = yo

        # print(out)
        
        
        smoothing_factor= 0.4
        
        smoothing_factor = min(max(smoothing_factor, 0), 1)

        # Calculate the y-values on the straight line
        x_start, x_end = x[0], x[-1]
        x_line = np.linspace(x_start, x_end, len(x))

        # Smoothing the y-coordinates
        x_smoothed = np.round(smoothing_factor * x_line + (1 - smoothing_factor) * x)
            


        n = len(x_smoothed)
        first_30_percent_index = int(n * 0.3)
        last_30_percent_index = int(n * 0.7)
        last_30_percent_index2 = int(n * 0.5)

        if n > 12 and len(x_smoothed) == len(set(x_smoothed)):
            # First 30% points
            x_first = np.linspace(0, first_30_percent_index, first_30_percent_index)
            y_first = x_smoothed[:first_30_percent_index]
            spline_first = InterpolatedUnivariateSpline(x_first, y_first, k=3)
            x_new = np.linspace(0, first_30_percent_index, n)
            y_first_extended = spline_first(x_new)

            # Last 30% points
            x_last = np.linspace(last_30_percent_index, n-1, n - last_30_percent_index)
            y_last = x_smoothed[last_30_percent_index:]
            spline_last = InterpolatedUnivariateSpline(x_last, y_last, k=3)
            x_new = np.linspace(last_30_percent_index, n-1, n)
            y_last_extended = spline_last(x_new)

            # Averaging
            x_smoothed1 = np.round(y_first_extended + y_last_extended) / 2

            x_smoothed2 = np.concatenate((x_smoothed[:last_30_percent_index2], x_smoothed1[last_30_percent_index2:]))
        elif n > 3 and len(x_smoothed) == len(set(x_smoothed)):
            x_smoothed2 = InterpolatedUnivariateSpline(x_smoothed, y, k=3)
        else:
            x_smoothed2 = x_smoothed

        def smooth_array(x):
            sigma = 20  # Standard deviation for Gaussian kernel
            return gaussian_filter1d(x, sigma)

        x_smoothed2 = smooth_array(x_smoothed2)

        out = (x_smoothed2, y)


        smoothed_out = out

        color_mask = self.rgb_image        

        prev_point = None

        # Drawing path
        for i in range(min(len(smoothed_out[0]), len(smoothed_out[1]))):
            point = (int(smoothed_out[0][i]), int(smoothed_out[1][i]))
            #rospy.loginfo(self.binary_mask.shape)
            #rospy.loginfo(self.depth_image.shape)
            #rospy.loginfo(point)

            if np.array_equal(self.binary_mask[point[1]][point[0]], [255, 0, 0]):
                cv2.circle(color_mask, point, 2, (0, 255, 0), -1)

                if prev_point is not None:
                    cv2.line(color_mask, prev_point, point, (0, 255, 0), 1)
                prev_point = point
        
        # Drawing predicted positions
        xstart = x_min
        xpos = xstart - dist
        rectangle_color = (255, 0, 0)  # BGR color (red in this case)
        width = self.rgb_image.shape[1]
        startpt = (int(width/4), xstart)
        endpt = (int(width*3/4), xpos)
        shapes = np.zeros_like(self.rgb_image, np.uint8)
        startpt = (int(startpt[0]), int(startpt[1]))
        endpt = (int(endpt[0]), int(endpt[1]))
        cv2.rectangle(
                shapes,
                startpt,
                endpt,
                rectangle_color, -1
                #rectangle_thickness,
        )
        out = color_mask.copy()
        alpha = 0.2
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(self.rgb_image, alpha, shapes, 1 - alpha, 0)[mask]
        image = out

        self.set_processed(image)
        

        return image, last_5_outs
    


    
def round_to_nearest_interval(timestamp, interval):
    # Round timestamp for collecting time points
    return round(timestamp / interval) * interval
    
def draw_path(processor):
    # Process frame
    # Skip frames if needed
    global frame_count
    frame_count += 1
    if frame_count % SKIP == 0:
        start_processing = rospy.get_time()
        processor.find_center_path_smooth()
        
    
def draw_distance(rgb_image, timestamp):
    # Draw predicted position

    # Get velocity
    v = global_speedometer if global_speedometer is not None else 0.1

    # Get time elapsed between frame and now
    current_time = time.time()
    current_time = round(current_time, 1)
    elapsed = current_time - timestamp

    # Processing time + added time delay
    tau = 0.03 + elapsed + 0.15
    global global_image
    throttle = pygame.joystick.Joystick(0).get_axis(1)
    dist = v*tau if throttle is not None else v*tau + (2)*tau*tau*30
    if global_speedometer is None:
        print("NO VELOCITY")
    #if theta is not None:
        #dist *= np.cos(theta)
    if dist < 0.0000001: dist = 0

    # Pinpoint pixel on image
    pixel = global_image[int(global_image.shape[0]/2)][int(global_image.shape[1]) - 1]
    pos_min = (pixel[2] + pixel[1] * 256 + pixel[0] * 256 * 256) / (256 * 256 * 256 - 1) * 1000

    pred_min = np.sqrt(pos_min*pos_min - 1.7*1.7)
    dist_from_cam = np.sqrt((pred_min + dist)*(pred_min + dist) + 1.7*1.7)

    print("START")
    print("dist: {}".format(dist))
    print("dist from cam: {}".format(dist_from_cam))
    print("pos_min: {}".format(pos_min))
    print("pos_min: {}".format(pred_min))

    if dist == 0: pixel_coor = 449
    else:
        # FIX ME find position for drawing predicted position
        height, width, _ = global_image.shape
        middle_column_index = width // 2
        middle_column = global_image[:, middle_column_index]

        distance = np.abs(dist + pred_min)
        print("distance: ", distance)
        pixel_coor = find_nearest_pixel(middle_column, distance)
        print(pixel_coor)
        pixel = global_image[int(global_image.shape[0]/2)][pixel_coor]
        pos = (pixel[2] + pixel[1] * 256 + pixel[0] * 256 * 256) / (256 * 256 * 256 - 1) * 1000

    
    

    # Draw predicted position
    width = rgb_image.shape[1]
    height = rgb_image.shape[0]
    if dist < 0.0000001: dist = 0
    xstart = height - 1
    xpos = pixel_coor
    print("xstart: {}".format(xstart))
    print("xpos: {}".format(xpos))
    rectangle_color = (255, 0, 0)
        
    startpt = (int(width/4), xstart)
    endpt = (int(width*3/4), xpos)
    shapes = np.zeros_like(rgb_image, np.uint8)
    startpt = (int(startpt[0]), int(startpt[1]))
    endpt = (int(endpt[0]), int(endpt[1]))
    cv2.rectangle(
                shapes,
                startpt,
                endpt,
                rectangle_color, -1
                #rectangle_thickness,
    )
    out = rgb_image.copy()
    alpha = 0.2
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(rgb_image, alpha, shapes, 1 - alpha, 0)[mask]
    image = out

 
        
    current_time = time.time() - current_time
    print("TIME TAKEN: ", current_time)
    return image, dist
    



def process_image(processor):
    # For example, convert the image to grayscale
    draw_path(processor)
    processed_image = processor.get_processed()
    mask = processor.get_white_mask()
    if processed_image is not None:
        cv2.imshow("Collision-Free Path", processed_image)
        cv2.waitKey(1)

    if mask is not None:
        cv2.imshow("Client Mask", mask)
        cv2.waitKey(1)

    #processed_image = cv_image
    return processed_image

def odometry_callback(msg):
    # Process odometry data here
    global global_odometry
    #rospy.loginfo(msg)
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    linear_velocity_x = msg.twist.twist.linear.x
    linear_velocity_y = msg.twist.twist.linear.y
    linear_velocity_z = msg.twist.twist.linear.z
    angular_velocity_x = msg.twist.twist.angular.x
    angular_velocity_y = msg.twist.twist.angular.y
    angular_velocity_z = msg.twist.twist.angular.z
    #rospy.loginfo("At time "+str(rospy.Time.now())+" vehicle position - x: {}, y: {}".format(x, y))
    #rospy.loginfo("speedometer: {}".format(np.sqrt(linear_velocity_x*linear_velocity_x + linear_velocity_y*linear_velocity_y + linear_velocity_z*linear_velocity_z)))
    yaw_rad = np.deg2rad( msg.twist.twist.angular.z)

    # Calculate forward direction vector based on orientation (yaw)
    forward_direction = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])

    # Extract velocity vector components
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    vz = msg.twist.twist.linear.z

    # Construct velocity vector
    velocity = np.array([vx, vy, vz])
    global global_speedometer
    # Calculate dot product of velocity and forward direction to project velocity onto forward direction
    forward_velocity = np.dot(velocity, forward_direction)
    global_speedometer = forward_velocity


def image_callback(msg, callback_args):
    # Callback for RGB image
    print("recieved")
    try:
        


            #print(f"{item}: {getattr(msg, item)}")


        '''
        # Convert timestamp to a readable format (if needed)
        # For example, to convert to a string:
        #timestamp_str = rospy.Time.from_sec(timestamp).to_sec()
        #print("Timestamp:", str(timestamp))
        #print("Timestamp:", str(current_time))
        print("NEXT")
        #mask_dir = "/home/oem/Desktop/seg_results_test/client_mask/"
        rgb_dir = "/home/oem/Desktop/test_150ms/client_rgb/"
        #path_dir = "/home/oem/Desktop/seg_results_test/path/"
        #seg_dir = "/home/oem/Desktop/test_1/client_mask"
        # Round timestamp to nearest 10ms interval
        aligned_timestamp = round_to_nearest_interval(current_time, 0.1)
        aligned_timestamp_str = "{:.1f}".format(aligned_timestamp)
        # Record frame with aligned timestamp
        #rospy.loginfo("Recording frame at aligned timestamp:" + aligned_timestamp_str)
        image_file_name = f"{aligned_timestamp_str}.png"
        # Sleep until the next aligned timestamp
        #time.sleep(aligned_timestamp + 0.1 - time.time())
        '''
        cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
        #processor = callback_args["processor"]

        #processed_image = process_image(processor)
        timestamp = msg.header.stamp.to_sec()
        timestamp = round(timestamp, 1)
        image, dist = draw_distance(cv_image, timestamp)
        cv2.imshow("Pos Pred", image)
        cv2.waitKey(1)
        time.sleep(0.1)
        current_time = time.time()
        current_time = round(current_time, 1)
        writer.append_row([timestamp, current_time, dist])
        #
        #rospy.loginfo(aligned_timestamp_str)
        global frame_count
        #if processor.get_white_mask() is not None and frame_count % SKIP == 0:
            #cv2.imwrite(os.path.join(mask_dir, image_file_name), processor.get_white_mask())
        #cv2.imwrite(os.path.join(rgb_dir, image_file_name), cv_image)
            #processed_image_topic = CvBridge().cv2_to_imgmsg(processed_image, encoding="bgr8")

        #cv2.imwrite(os.path.join(seg_dir, image_file_name), global_image)

            #rospy.loginfo("saved images")
        #elif processor.get_white_mask() is None:
            #rospy.loginfo("no white mask")
        #else:
            #rospy.loginfo("skipping error")
        #rospy.loginfo("At current time "+str(rospy.Time.now())+" acceleration is "+str(abs((pygame.joystick.Joystick(0).get_axis(1)-1)/2)))
        #processed_image_publisher.publish(processed_image_topic)

    except Exception as e:
        rospy.logerr("Error processing and publishing image: {}".format(e), exc_info=True)




def recolor(image):
    # Recolor segmentation mask
    target_purple = np.array([128, 64, 128], dtype=np.uint8)
    tolerance = 10

    # Create a lower and upper range for purple values
    lower_range = target_purple - tolerance
    upper_range = target_purple + tolerance

    # Ensure the values are within the valid range [0, 255]
    lower_range = np.clip(lower_range, 0, 255)
    upper_range = np.clip(upper_range, 0, 255)

    # Create the mask for the range of purple values
    mask = np.all((image >= lower_range) & (image <= upper_range), axis=-1)

    # Create an empty array with the same shape as the input image
    image_array = np.zeros_like(image)

    # Set the pixels within the mask to white (255)
    image_array[mask] = 255

    return image_array
    return image_array

def rgb_callback(msg):
    global global_image
    #global_image = recolor(CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8"))
    global_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")

def camera_subscriber(processor):
    # Subscribes to RGB and other topics if needed
    pygame.init()
    pygame.joystick.Joystick(0).init()
    rospy.init_node('camera_subscriber', anonymous=True)

    #camera_topic = '/client/front/rtsp_client/camera/rgb/image'
    depth_topic = '/client/front_right/rtsp_client/camera/rgb/image'
    #rgb_topic = '/client/front_left/rtsp_client/camera/rgb/image'
    #processed_image_topic = 'processed_image'
    rgb_topic = '/carla/ego_vehicle/front/image'
    speedometer_topic = '/carla/ego_vehicle/speedometer'
    odometry_topic = '/carla/ego_vehicle/odometry'
    #processed_image_publisher = rospy.Publisher(processed_image_topic, rosimage, queue_size=10)
    rospy.Subscriber(rgb_topic, rosimage, image_callback, callback_args={'processor': processor}, queue_size=10)#, 'processed_image_publisher': processed_image_publisher})
    #rospy.Subscriber(depth_topic, rosimage, depth_callback)
    rospy.Subscriber(depth_topic, rosimage, rgb_callback)
    #rospy.Subscriber(speedometer_topic, Float32, speedometer_callback)
    rospy.Subscriber(odometry_topic, Odometry, odometry_callback)


    rospy.spin()

class CSVWriter:
    # Writes to file for distance evaluation
    def __init__(self, directory, file_name):
        self.file_path = os.path.join(directory, file_name)
        #self.append_row(["time 1", "time 2", "dist"])

    def append_row(self, values):
        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(values)

if __name__ == '__main__':
    try:
        #processor = Image_Processor()
        writer = CSVWriter("/home/oem/Desktop/csv_files", "client_no.csv")
        camera_subscriber(writer)
    except rospy.ROSInterruptException:
        pass