# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging as log
import subprocess as sp
import cv2
from typing import Tuple

def get_v4l2() -> Tuple[bool, list]:
    """Get V4L2 device list

    Returns:
        Tuple[bool, list]: ( has_camera_or_not, camera_list )
    """

    # Get V4L2 Camera
    command = sp.run("ls /dev/video*", shell=True, stdout=sp.PIPE, encoding='utf8')
    
    # Fail mean no camera here, 0 means success
    if command.returncode != 0:
        return ( False, "Camera not found")

    # Parse Each Camera to a list
    available_camera = []
    
    # Check is failed_key in that information
    for camera in command.stdout.strip().split('\n'):
        # Note: use `cv2.CAP_V4L` to make sure VideoCapture run fastest.
        cam_idx = int(camera.split("video")[-1])
        if cv2.VideoCapture(cam_idx, cv2.CAP_V4L).isOpened():
            available_camera.append(camera)
        
    return ( True, available_camera )