# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging as log
import os
import re
import shutil
import time
from typing import Tuple

import cv2
import numpy as np

try:
    from ..common import RT_CONF, SERV_CONF
    from ..utils import gen_uid
except:
    from common import RT_CONF, SERV_CONF
    from utils import gen_uid

from .db_handler import (
    insert_data,
    is_list_empty,
    parse_source_data,
    select_data,
    update_data,
)
from .err_handler import InvalidUidError

# iVIT Libraray
from .ivit_handler import Displayer, RtspWrapper, SourceV2
from .sys_handler import get_v4l2

# --------------------------------------------------------
# Consistant
K_SRC = "SRC"
UNEXPECTED_CHAR = ["+", "/", "?", "#", "%", "&", "="]
NAME_RULE = r"[{}]".format("".join(UNEXPECTED_CHAR))

# --------------------------------------------------------
# Helper Function


def get_src_status(source_uid):
    return select_data(
        table="source", data=["status"], condition=f"WHERE uid='{source_uid}'"
    )[0][0]


def update_src_status(source_uid, status):
    update_data(
        table="source", data={"status": status}, condition=f"WHERE uid='{source_uid}'"
    )


def get_src_info(uid: str = None):
    """Get Source Information from database"""
    if uid == None:
        data = select_data(table="source", data="*")
    else:
        data = select_data(table="source", data="*", condition=f"WHERE uid='{uid}'")
    if data == []:
        raise KeyError(f"Could not find source: {uid}")
    ret = [parse_source_data(src) for src in data]
    return ret


def is_source_using(source_uid: str) -> Tuple[bool, str]:
    """If source is using or not.

    Args:
        source_uid (str): the source uuid.

    Returns:
        Tuple[bool, str]: if source is using or not.
    """

    flag, mesg = True, "No one using this source"
    data = select_data(
        table="task",
        data=["uid", "status"],
        condition=f"WHERE source_uid='{source_uid}'",
    )

    if is_list_empty(data):
        return (flag, mesg)

    using_tasks = [uid for uid, status in data if status == "run"]
    flag = len(using_tasks) >= 1
    if not flag:
        return (flag, mesg)

    mesg = "The source is still used by {}".format(", ".join(using_tasks))
    log.warning(mesg)
    return (flag, mesg)


def is_src_loaded(source_uid) -> bool:
    """Check source object is loaded

    Args:
        source_uid (_type_): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        bool: _description_
    """
    timeout = 0
    while True:
        status = get_src_status(source_uid)

        if status in ["run", "loaded"]:
            src = RT_CONF[K_SRC].get(source_uid)
            if src is None:
                break
            return True

        elif status in ["stop", "error"]:
            update_src_status(source_uid, "loading")
            break
        else:
            print("Waiting Source Object ...")
            timeout += 1
            time.sleep(1)

        if timeout >= 10:
            raise RuntimeError("Initialize Source Object Timeout")

    return False


# --------------------------------------------------------
# Main Function: Displayer

create_displayer = Displayer
create_rtsp_displayer = RtspWrapper

# --------------------------------------------------------
# Main Function: Source


def create_source(source_uid: str) -> SourceV2:
    """Create Source, if source already created then just return the SourceV2 object

    Args:
        source_uid (str): source uid

    Raises:
        RuntimeError: _description_

    Returns:
        SourceV2: _description_
    """
    # Source Information
    source = select_data(
        table="source", data="*", condition=f"WHERE uid='{source_uid}'"
    )

    # Verify Source UID is available
    if is_list_empty(source):
        raise InvalidUidError("Could not find source uid.")

    # Parse Source Information
    src_info = parse_source_data(source[0])

    # NOTE: if it's camera then have to check twice
    if src_info["type"] == "CAM":
        ret, available_cams = get_v4l2()
        print("\n\n", available_cams)
        # if not in available camera list
        # NOTE: have to add more behaviour
        if src_info["input"] not in available_cams:
            update_src_status(src_info["uid"], "error")
            raise RuntimeError("Camera not found.")

    # Check Source
    if is_src_loaded(source_uid):
        print(f"\n\nThe Source ({src_info['input']}) is loaded.")
        return RT_CONF[K_SRC].get(source_uid)

    # Initialize Source
    try:
        src_object = SourceV2(
            input=src_info["input"],
            resolution=src_info.get("resolution"),
            fps=src_info.get("fps"),
        )
    except Exception as e:
        log.exception(e)

    # Update into RT_CONF
    RT_CONF[K_SRC].update({source_uid: src_object})
    log.info(f"Update {source_uid} to {SERV_CONF.get_name}")

    log.info("Initialized Source")
    update_src_status(source_uid, "loaded")

    return src_object


def start_source(source_uid: str):
    """Start source"""
    src = create_source(source_uid)
    src.start()
    update_src_status(source_uid, "run")


def stop_source(source_uid: str):
    """Stop source

    Args:
        source_uid (str): source uuid
    """
    if is_source_using(source_uid)[0]:
        return

    src = create_source(source_uid)
    src.release()
    update_src_status(source_uid, "stop")
    log.warning(f"Stop source: {source_uid}")


def add_source(files=None, input: str = None, option: dict = None) -> dict:
    """Add new source function"""

    def save_file(file, path):
        """Save file from fastapi"""
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        shutil.move(file.filename, path)

    # Got RTSP, V4L2
    if (files is None) or not (files[0].filename):
        name = input

        # If RTSP and V4L2 is created and load , then return source object
        source = select_data(table="source", data="*", condition=f"WHERE name='{name}'")

        # Means the source is exist
        if not is_list_empty(source):
            # Parse Source Information
            src_info = parse_source_data(source[0])
            return {
                "uid": src_info["uid"],
                "name": src_info["name"],
                "type": src_info["type"],
                "input": src_info["input"],
                "height": str(src_info["height"]),
                "width": str(src_info["width"]),
                "status": src_info["status"],
            }

    # Got Files
    else:
        # Multi File
        if len(files) > 1:
            log.info("Add Multiple File")
            folder_name = f"dataset-{gen_uid(files[0].filename)}"

            folder_path = os.path.join(SERV_CONF["DATA_DIR"], folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                log.warning(f"Create Folder: {folder_path}")

            for file in files:
                path = os.path.join(folder_path, file.filename)
                save_file(file=file, path=path)

            name = folder_name
            input = folder_path

        # Single File
        else:
            file = files[0]
            path = os.path.join(SERV_CONF["DATA_DIR"], file.filename)
            save_file(file=file, path=path)

            name = file.filename
            input = path

    resolution = option.get("resolution")
    fps = option.get("fps")
    uid = gen_uid(name)

    src = SourceV2(input=input, resolution=resolution, fps=fps)

    type = src.get_type()
    height, width = src.get_shape()

    data = {
        "uid": uid,
        "name": name,
        "type": type,
        "input": input,
        "height": str(height),
        "width": str(width),
        "status": "stop",
    }

    insert_data(table="source", data=data, replace=True)

    src.release()

    return data


def get_source_frame(source_uid: str, resolution: list = None) -> np.ndarray:
    """Get source frame with target resolution

    Args:
        source_uid (str): the uid of source
        resolution (list, optional): (height, width). Defaults to None.

    Returns:
        cv2.ndarray: _description_
    """

    # Get or create source object
    src = create_source(source_uid=source_uid)
    frame = src.frame

    # Check source is exist or not
    source = select_data(
        table="source", data="*", condition=f"WHERE uid='{source_uid}'"
    )

    # The new source object have to clean
    need_to_release = False
    if is_list_empty(source):
        need_to_release = True
    else:
        src_data = parse_source_data(source[0])
        need_to_release = src_data["status"] not in ["run"]

    if need_to_release:
        log.warning("Clear source object.")
        src.release()
        update_data(
            table="source",
            data={"status": "stop"},
            condition=f'WHERE uid="{source_uid}"',
        )

    # Setup resolution if need.
    if resolution:
        frame = cv2.resize(frame, (resolution[1], resolution[0]))

    return frame


def is_name_valid(name: str) -> bool:
    return not re.search(NAME_RULE, name)
