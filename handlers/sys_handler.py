# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging as log
import subprocess as sp
import cv2

def get_v4l2() -> tuple:
    """ Get V4L2 device list [] """

    # Get V4L2 Camera
    command = sp.run("ls /dev/video*", shell=True, stdout=sp.PIPE, encoding='utf8')
    
    # Not success
    if command.returncode != 0:
        return ( False, "Camera not found")

    # 0 means success

    # Parse Each Camera to a list
    available_camera = []

    camera_list = command.stdout.strip().split('\n')
    log.debug("{}, {}".format(camera_list, type(camera_list)))
    
    # Check is failed_key in that information
    for camera in camera_list.copy():
        cam_idx = int(camera.split("video")[-1])
        if cv2.VideoCapture(cam_idx, cv2.CAP_V4L).isOpened():
            available_camera.append(camera)
        
    return True, available_camera