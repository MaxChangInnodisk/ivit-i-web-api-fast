# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import gdown
import sqlite3
import os
import sys
import time
import json
import zipfile
import logging as log

try:
    from .imagenet import IMAGE_NET_LABEL
    from .coco import COCO_LABEL
    from .utils import download_data, download_model
except:
    from imagenet import IMAGE_NET_LABEL
    from coco import COCO_LABEL
    from utils import download_data, download_model

try:
    from ..common import SERV_CONF
    from ..utils import gen_uid, json_to_str
    from ..handlers import db_handler
except:
    from common import SERV_CONF
    from utils import gen_uid, json_to_str
    from handlers import db_handler

CREATED_TIME = time.time()

def intel_sample_cls(db_path: str = SERV_CONF["DB_PATH"]):
    """ Add intel sample information into database 
    ---
    1. Download Data from Google Drive, and Update into Database
    2. Download Model from Google Drive, and Update into Database
    3. Update AI Task Information into Database
    """

    # Download data and model
    data_name = 'cat.jpg'
    data_url = "https://drive.google.com/file/d/1hq2CvCT4SRTvvkHo3QVZSh85bdUuiXbR/view?usp=sharing"
    download_data(data_name, data_url)

    model_name = "dog_cat_classification_50"
    model_url = "https://drive.google.com/file/d/1ylMHWd1KN6Ydhxryxrc-ZyPALf5rmI9F/view?usp=share_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "classification-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = 'HAILO'

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join(SERV_CONF["DATA_DIR"], source_name)
    source_uid = gen_uid(source_name)
    source_height, source_width = "425", "640"

    model_uid = gen_uid(model_name)
    model_setting = {
        "confidence_threshold": 0.1,
        "topk": 3
    }

    app_name = 'Basic_Classification'
    app_type = 'CLS'
    app_uid = task_uid
    app_setting = {
        "application": {
            "areas": [
                {
                    "name": "default",
                    "depend_on": [ "cat", "dog" ],
                    "palette": {
                        "airplane": [255, 255, 255],
                        "warpalne": [0, 0, 0],
                    }
                }
            ]
        }
    }
    event_uid = None

    # Source
    db_handler.insert_data(
        table="source",
        data={
            "uid": source_uid,
            "name": source_name,
            "type": source_type,
            "input": source_input,
            "status": source_status,
            "height": source_height,
            "width": source_width,
        },
        replace=True)

    # Application
    data = db_handler.select_data(
        table='task', data="*", condition=f"WHERE uid = '{task_uid}'")
    if data == []:
        db_handler.insert_data(
            table="app",
            data={
                "uid": app_uid,
                "name": app_name,
                "type": app_type,
                "app_setting": app_setting
            },
            replace=True
        )

    db_handler.insert_data(
        table="task",
        data={
            "uid": task_uid,
            "name": task_name,
            "source_uid": source_uid,
            "model_uid": model_uid,
            "model_setting": model_setting,
            "status": task_status,
            "device": device,
            "created_time": CREATED_TIME
        },
        replace=True
    )


def intel_sample_yolov4_leaky(db_path: str = SERV_CONF["DB_PATH"]):
    """ Add intel sample information into database 
    ---
    1. Download Data from Google Drive, and Update into Database
    2. Download Model from Google Drive, and Update into Database
    3. Update AI Task Information into Database
    """

    # Download data and model
    data_name = 'car.mp4'
    data_url = "https://drive.google.com/file/d/15X2EwgdtNmLgaLOWlMyBytJ3gM1c1Wqd/view?usp=sharing"
    download_data(data_name, data_url)

    model_name = "yolov4-leacky-512"
    model_url = "https://drive.google.com/file/d/1ntUQp9pfgKOZP_2o67Mjx8VsgNgY1Nmv/view?usp=drive_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "yolov4-leacky-512-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = 'HAILO'

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join(SERV_CONF["DATA_DIR"], source_name)
    source_uid = gen_uid(source_name)
    source_height, source_width = "720", "1280"

    model_uid = gen_uid(model_name)
    model_setting = {
        "confidence_threshold": 0.5,
        "architecture_type": "yolov4",
        "anchors": [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 164, 142, 110, 192, 243, 459, 401 ]
    }

    app_name = 'Basic_Object_Detection'
    app_type = 'OBJ'
    app_uid = task_uid
    app_setting = {
        "application": {
            "areas": [
                {
                    "name": "default",
                            "depend_on": COCO_LABEL,
                }
            ]
        }
    }
    event_uid = None

    # Source: 5
    db_handler.insert_data(
        table="source",
        data={
            "uid": source_uid,
            "name": source_name,
            "type": source_type,
            "input": source_input,
            "status": source_status,
            "height": source_height,
            "width": source_width,
        },
        replace=True)

    # Application
    data = db_handler.select_data(
        table='app', data=["uid"], condition=f"WHERE uid = '{task_uid}'")
    if data == []:
        db_handler.insert_data(
            table="app",
            data={
                "uid": app_uid,
                "name": app_name,
                "type": app_type,
                "app_setting": app_setting
            },
            replace=True
        )

    db_handler.insert_data(
        table="task",
        data={
            "uid": task_uid,
            "name": task_name,
            "source_uid": source_uid,
            "model_uid": model_uid,
            "model_setting": model_setting,
            "status": task_status,
            "device": device,
            "created_time": CREATED_TIME
        },
        replace=True
    )


def intel_sample_yolov4_tiny(db_path: str = SERV_CONF["DB_PATH"]):
    """ Add intel sample information into database 
    ---
    1. Download Data from Google Drive, and Update into Database
    2. Download Model from Google Drive, and Update into Database
    3. Update AI Task Information into Database
    """

    # Download data and model
    data_name = 'car.mp4'
    data_url = "https://drive.google.com/file/d/15X2EwgdtNmLgaLOWlMyBytJ3gM1c1Wqd/view?usp=sharing"
    download_data(data_name, data_url)

    model_name = "yolov4-tiny-608"
    model_url = "https://drive.google.com/file/d/1WirW_3aDdmMGBODHiBEelITvOS50G65w/view?usp=drive_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "yolov4-tiny-608-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = 'HAILO'

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join(SERV_CONF["DATA_DIR"], source_name)
    source_uid = gen_uid(source_name)
    source_height, source_width = "720", "1280"

    model_uid = gen_uid(model_name)
    model_setting = {
        "confidence_threshold": 0.5,
        "architecture_type": "yolov3-tiny",
        "anchors": [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 ]
    }

    app_name = 'Basic_Object_Detection'
    app_type = 'OBJ'
    app_uid = task_uid
    app_setting = {
        "application": {
            "areas": [
                {
                    "name": "default",
                            "depend_on": COCO_LABEL,
                }
            ]
        }
    }
    event_uid = None

    # Source: 5
    db_handler.insert_data(
        table="source",
        data={
            "uid": source_uid,
            "name": source_name,
            "type": source_type,
            "input": source_input,
            "status": source_status,
            "height": source_height,
            "width": source_width,
        },
        replace=True)

    # Application
    data = db_handler.select_data(
        table='app', data=["uid"], condition=f"WHERE uid = '{task_uid}'")
    if data == []:
        db_handler.insert_data(
            table="app",
            data={
                "uid": app_uid,
                "name": app_name,
                "type": app_type,
                "app_setting": app_setting
            },
            replace=True
        )

    db_handler.insert_data(
        table="task",
        data={
            "uid": task_uid,
            "name": task_name,
            "source_uid": source_uid,
            "model_uid": model_uid,
            "model_setting": model_setting,
            "status": task_status,
            "device": device,
            "created_time": CREATED_TIME
        },
        replace=True
    )


def init_hailo_samples():
    intel_sample_cls()
    intel_sample_yolov4_tiny()
    intel_sample_yolov4_leaky()


if __name__ == "__main__":
    db_handler.init_sqlite()
    intel_sample_cls()
    intel_sample_yolov4_leaky()
