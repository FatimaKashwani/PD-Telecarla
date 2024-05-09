import os
import numpy as np
from skimage import io
#from skimage.metrics import (adapted_rand_error, variation_of_information,
                              #contingency_table, hausdorff_distance,
                              #mean_squared_error, peak_signal_noise_ratio,
                              #structural_similarity, normalized_root_mse,
                              #normalized_mean_squared_error,
                              #mean_squared_error)

import os
import numpy as np
from skimage import io
import cv2

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

# Get model from server
MODEL_SERVER = "https://dl.cv.ethz.ch/bdd100k/det/models/"



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

def find_closest_pixel(depth_column, threshold_distance, start_row=0, end_row=None):
    # Find closest point in the image to a specified distance
    # FIX ME this method doesnt display predicted distance concisely
    if end_row is None:
        end_row = len(depth_column) - 1

    closest_distance = float('inf')
    closest_pixel = None

    while start_row <= end_row:
        # Look for closest point within the middle column
        mid_row = (start_row + end_row) // 2
        distance = calculate_distance(depth_column[mid_row])

        if abs(distance - threshold_distance) < closest_distance:
            closest_distance = abs(distance - threshold_distance)
            closest_pixel = mid_row

        if distance < threshold_distance:
            start_row = mid_row + 1
        elif distance > threshold_distance:
            end_row = mid_row - 1
        else:
            break

    return depth_column[closest_pixel]

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
        self.detected_mask = None

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
        if self.get_mask is None: 
            print("error")
            return None
    
        blue_mask = np.all(self.get_mask() == [255, 0, 0], axis=-1)
        green_mask = np.all(self.get_mask() == [0, 255, 0], axis=-1)

        # White mask combines ego and alternate lanes
        # Used for evaluation with ground truth road mask
        white_mask = np.zeros_like(blue_mask, dtype=np.uint8)
        white_mask[(blue_mask | green_mask)] = 255

        return white_mask

    def get_processed(self):
        return self.processed_image
    
    def get_det(self):
        return self.detected_mask
    
    def set_det(self, image):
        self.detected_mask = image
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
        if self.rgb_image is None: 
            return

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

        det = Draw_Det_Mask(outputsdet, (self.rgb_image.shape[0], self.rgb_image.shape[1]))
        del outputsdet

        

     
        # Pass through segmentation module
        with torch.no_grad():
            outputsseg = self.segmentor.forward_test([img_tensor], [img_meta], rescale=True)

        seg = Draw_Seg_Mask(outputsseg)
        del outputsseg


        # Combine results into one mask
        # Set mask
        comb = CombineMasks(seg, det)
        self.set_mask(comb)
        #self.set_det(det)
        pygame.event.pump()



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
        '''
        dist = 0
        #if theta is not None:
            #dist *= np.cos(theta)
        if dist < 0.001: dist = 0


        pixel = self.depth_image[int(self.depth_image.shape[0]/2)][int(self.depth_image.shape[1]) - 1]

        pos_min = (pixel[2] + pixel[1] * 256 + pixel[0] * 256 * 256) / (256 * 256 * 256 - 1) * 1000
        pred_min = np.sqrt(pos_min*pos_min - 1.7*1.7)
        dist_from_cam = np.sqrt((pred_min + dist)*(pred_min + dist) + 1.7*1.7)
        if dist == 0: pos = pos_min
        else:
            #rospy.loginfo(trajectory)
            #rospy.loginfo(len(smoothed_out))
            #rospy.loginfo(len(y))
            trajectory = [(smoothed_out[i], y[i]) for i in range(min(len(smoothed_out), len(y)))]
            pixel_coor = find_closest_pixel([self.depth_image[int(y), int(x)] for x,y in trajectory], dist_from_cam)
            #rospy.loginfo(pixel_coor[0])
            pixel = self.depth_image[int(pixel_coor[0])][int(pixel_coor[1])]
            pos = (pixel[2] + pixel[1] * 256 + pixel[0] * 256 * 256) / (256 * 256 * 256 - 1) * 1000

        pred_dist = np.sqrt(pos*pos - 1.7*1.7)
    
        distance = np.abs(pred_dist - pred_min)


        x_min = np.max(smoothed_out)
        theta = self.joystick.get_axis(0)
        new_x = []
        
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


            if np.array_equal(self.binary_mask[point[1]][point[0]], [255, 0, 0]):
                cv2.circle(color_mask, point, 2, (0, 255, 0), -1)

                if prev_point is not None:
                    cv2.line(color_mask, prev_point, point, (0, 255, 0), 1)
                prev_point = point
        # Drawing pred pos (currently moved to dtnet.py)
        '''
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
        '''
        image = color_mask

        self.set_processed(image)

        # Return result and last 5 paths for averaging
        return image, last_5_outs
    


    
def round_to_nearest_interval(timestamp, interval):
    # Round timestamp for collecting time points
    return round(timestamp / interval) * interval
    
def draw_path(processor):
        processor.find_center_path_smooth()
        
    


def process_images(predicted_dir, path_dir, img_dir, mask_files):
    # Process images to obtain DSC, IoU, Pixel Accuracy
    # predicted dir: predicted road area output folder
    # path dir: path drawing output folder
    # img dir: rgb image input folder
    # mask files: intersection of client and server files

    # Initialize processor object
    processor = Image_Processor()
    #ground_truth_dir = "/home/oem/Desktop/test_nodel/server_mask/"


    # Get list of images in both directories
    imgs = os.listdir(img_dir)
    overlap = set(imgs).intersection(set(mask_files))
    overlap = list(overlap)

    # Pass images through DTNet
    for i in range(len(overlap)):
        print(f"Processing image {i+1} of {len(overlap)}: {overlap[i]}")
        image_path = os.path.join(img_dir, overlap[i])
        img = cv2.imread(image_path)
        processor.set_rgb(img)
        draw_path(processor)
        processed_image = processor.get_processed()
        mask = processor.get_white_mask()
        cv2.imwrite(os.path.join(path_dir, overlap[i]), processed_image)
        cv2.imwrite(os.path.join(predicted_dir, overlap[i]), mask)


def dice_coefficient(truth, prediction):
    # Returns DSC score
    intersection = np.sum(truth * prediction)
    union = np.sum(truth) + np.sum(prediction)
    if union == 0:
        return 1.0
    else:
        return (2.0 * intersection) / union

def intersection_over_union(truth, prediction):
    # Returns IoU score
    intersection = np.sum(truth * prediction)
    union = np.sum(truth) + np.sum(prediction) - intersection
    if union == 0:
        return 1.0
    else:
        return intersection / union

def pixel_accuracy(truth, prediction):
    # Returns pixel accuracy
    correct_pixels = np.sum(truth == prediction)
    total_pixels = truth.size
    return correct_pixels / total_pixels

def calculate_metrics(ground_truth_dir, predicted_dir, path_dir):
    # Calculates metrics if predicted mask is available
    # If not, first run process_images on client-side rgb
    dsc_scores = []
    iou_scores = []
    pixel_accuracy_scores = []
    percent_ovlp = []
    
    ground_truth_files = set(os.listdir(ground_truth_dir))
    predicted_files = set(os.listdir(predicted_dir))
    path_files = set(os.listdir(path_dir))
    print(len(ground_truth_files))
    #print(len(predicted_files))
    mask_files = ground_truth_files.intersection(predicted_files)#, path_files)
    print(len(mask_files))
    # Create a set to store the updated list of files
    updated_mask_files = set(mask_files)

    # Iterate through the files in the directories
    for file in os.listdir(ground_truth_dir):
        ground_truth_file_path = os.path.join(ground_truth_dir, file)
        predicted_file_path = os.path.join(predicted_dir, file)

        # Check if both files exist and are not already in the list
        if os.path.exists(predicted_file_path) and file not in mask_files:
            # Extract timestamps from the file names
            ground_truth_timestamp = float(file.split('.')[0])
            predicted_timestamp = float(file.split('.')[0])

            # Check if the timestamps differ by one unit
            if abs(ground_truth_timestamp - predicted_timestamp) == 0.1:
                # Add the files to the updated list
                updated_mask_files.add(file)


    for mask_file in updated_mask_files:
        ground_truth_mask = io.imread(os.path.join(ground_truth_dir, mask_file), as_gray=True)
        predicted_mask = io.imread(os.path.join(predicted_dir, mask_file), as_gray=True)
        '''
        path = io.imread(os.path.join(path_dir, mask_file), as_gray=False)
        path_mask = np.zeros_like(path, dtype=np.uint8)
        path_mask[(path[:,:,0] == 0) & (path[:,:,1] == 255) & (path[:,:,2] == 0)] = [255, 255, 255]
        white_pixels_path = np.all(path_mask == [255, 255, 255], axis=-1)
        white_pixels_ground_truth = ground_truth_mask == 1

        # Count the number of overlapping white pixels
        if white_pixels_path.shape == white_pixels_ground_truth.shape:
            # Count the number of overlapping white pixels
            white_pixels_overlap = np.sum(np.logical_and(white_pixels_path, white_pixels_ground_truth))
        else:
            print("Sizes of white pixel arrays do not match. Skipping logical AND operation.")
            white_pixels_overlap = None  # Or any value you want to assign
            continue
        # Count the total number of white pixels in path_mask
        total_white_pixels_path = np.sum(white_pixels_path)
        
        # Calculate the percentage of overlap
        if total_white_pixels_path != 0:
            percentage_overlap = (white_pixels_overlap / total_white_pixels_path)
        elif np.any(ground_truth_mask == 1):
            percentage_overlap = 0
        else:
            percentage_overlap = 1.0
        
        '''
        if ground_truth_mask.shape != predicted_mask.shape:
            print(f"Shapes of {mask_file} are not equal. Skipping...")
            continue
        
        ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
        predicted_mask = (predicted_mask > 0).astype(np.uint8)
        # Get DSC, IoU, pixel accuracy
        dsc = dice_coefficient(ground_truth_mask, predicted_mask)
        iou = intersection_over_union(ground_truth_mask, predicted_mask)
        accuracy = pixel_accuracy(ground_truth_mask, predicted_mask)
        # Append to list to get mean values later
        dsc_scores.append(dsc)
        iou_scores.append(iou)
        pixel_accuracy_scores.append(accuracy)
        #percent_ovlp.append(percentage_overlap)
        
        #print(f"Metrics for {mask_file}: DSC = {dsc}, IOU = {iou}, Pixel Accuracy = {accuracy}")#, Percent Overlap = {percentage_overlap}")
    print("Number of frames: ", len(mask_files))
    print("Delay: 150ms")
    return dsc_scores, iou_scores, pixel_accuracy_scores, percent_ovlp

def count(gtdir, imgdir):
    # Isolate only timestamps found in client and server directories
    ground_truth_files = set(os.listdir(gtdir))
    img_files = set(os.listdir(imgdir))
    mask_files = ground_truth_files.intersection(img_files)
    print("Count: ", str(len(mask_files)))
    return mask_files
# Provide the paths to the directories containing ground truth and predicted masks
base_dir = "/home/oem/Desktop/test_150ms"
ground_truth_dir = os.path.join(base_dir, "server_mask")
predicted_dir = os.path.join(base_dir, "gen_mask")
path_dir = os.path.join(base_dir, "gen_path")
img_dir = os.path.join(base_dir, "client_rgb")
num = count(ground_truth_dir, img_dir)

# process images if needed
#process_images(predicted_dir, path_dir, img_dir, num)

# Calculate metrics
dsc_scores, iou_scores, pixel_accuracy_scores, percent_ovlp = calculate_metrics(ground_truth_dir, predicted_dir, path_dir)

# Obtain mean values
mean_dsc_score = np.mean(dsc_scores)
mean_iou_score = np.mean(iou_scores)
mean_pixel_accuracy = np.mean(pixel_accuracy_scores)
percent_ovlp_score = np.mean(percent_ovlp)
dsc_acc = 0
iou_acc = 0
pa_acc = 0

# Observe results based on pass/fail threshold
for i in range (len(dsc_scores)):
    if dsc_scores[i] >= 0.7:
        dsc_acc += 1
    if iou_scores[i] >= 0.7:
        iou_acc += 1
    if pixel_accuracy_scores[i] >= 0.7:
        pa_acc += 1


# Print results
print("Mean DSC Score:", mean_dsc_score)
#print("Passed: ", str(dsc_acc/len(dsc_scores)))
print("Mean IOU Score:", mean_iou_score)
#print("Passed: ", str(iou_acc/len(dsc_scores)))
print("Mean Pixel Accuracy:", mean_pixel_accuracy)
#print("Passed: ", str(pa_acc/len(dsc_scores)))
print("Path Overlap:", percent_ovlp_score)

