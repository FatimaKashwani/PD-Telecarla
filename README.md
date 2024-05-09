# PD-Telecarla
Predictive Display implementation on telecarla

Required installations to run DTNet on telecarla (ubuntu 20.04):

ROS Noetic: https://wiki.ros.org/noetic/Installation/Ubuntu

CARLA 0.9.13: https://github.com/carla-simulator/carla/tree/0.9.13

TELECARLA: https://github.com/hofbi/telecarla

Included "carla" conda environment.

Note that telecarla installation includes the CARLA-ROS bridge.

Files in this repository: 

multi_sensors.json is under src/telecarla/telecarla_manual_control/config. The original setup contains 3 views from different angles, all RGB. The json file here contains RGB (front), semantic segmentation camera (front left) and depth camera (front right). Replace the json file to use the collision-free path code.

launch and src folders are under src/telecarla/telecarla. Replace the folders, run the CARLA simulator (./CARLA_0.9.13/CarlaUE4.sh), then run using two terminals:

roslaunch telecarla telecarla_multi_server.launch

roslaunch telecarla telecarla_multi_client.launch

Shown displays:

Client-end segmentation (as per DTNet)
Client-end position prediction
Client-end collision-free path

Server-end RGB
Server-end ground truth segmentation

Teleoperator view (RGB, segmentation, and depth camera views on client end).

For evaluating masks, images must be saved.
For evaluating position, time points (csv) must be saved.
