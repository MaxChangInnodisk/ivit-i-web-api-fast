# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import subprocess as sp
from typing import Tuple

import cv2

from handlers.db_handler import is_list_empty, parse_source_data, select_data


def get_v4l2() -> Tuple[bool, list]:
    """Get V4L2 device list

    Returns:
        Tuple[bool, list]: ( has_camera_or_not, camera_list )
    """

    # Get Exist Camera
    cam_src = []
    all_cam_src = select_data(
        table="source",
        data="*",
        condition="WHERE type='CAM' AND status IN ( 'loaded', 'run' ) ",
    )

    if not is_list_empty(all_cam_src):
        cam_src = [parse_source_data(cur_src)["input"] for cur_src in all_cam_src]

    # Get V4L2 Camera
    command = sp.run("ls /dev/video*", shell=True, stdout=sp.PIPE, encoding="utf8")

    # Fail mean no camera here, 0 means success
    if command.returncode != 0:
        return (False, "Camera not found")

    # Parse Each Camera to a list
    available_camera = []

    # Check is failed_key in that information
    for camera in command.stdout.strip().split("\n"):
        cam_idx = int(camera.split("video")[-1])

        # Note: use `cv2.CAP_V4L` to make sure VideoCapture run fastest.
        if cv2.VideoCapture(cam_idx, cv2.CAP_V4L).isOpened():
            available_camera.append(camera)

        # Note: if the camera source using then append directly
        elif camera in cam_src:
            available_camera.append(camera)

    return (True, available_camera)
