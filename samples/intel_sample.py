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
except:
    from imagenet import IMAGE_NET_LABEL
    from coco import COCO_LABEL
    
try:
    from ..common import SERV_CONF
    from ..utils import gen_uid, json_to_str
    from ..handlers import db_handler
except:
    from common import SERV_CONF
    from utils import gen_uid, json_to_str
    from handlers import db_handler


def extract_file(zip_path:str, folder_name:str=None):
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError("ZIP Path is unavailable: {}".format(zip_path))
    
    if not folder_name:
        folder_name = os.path.splitext(zip_path)[0]

    print(zip_path, folder_name)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(path=folder_name)

def download_file(file_path, file_url):
    check_data = file_path
    if os.path.splitext(file_path)[1] == '.zip':
        check_data = os.path.splitext(file_path)[0]

    if os.path.exists(check_data):
        return

    gdown.download(url=file_url, output=file_path, quiet=False, fuzzy=True)

def download_model(file_name, file_url):
    ext = '.zip'
    if not ext in file_name:
        file_name += ext
    file_path = os.path.join(SERV_CONF["MODEL_DIR"], file_name)
    download_file(file_path, file_url)
    extract_file(file_path)

def download_data(file_name, file_url):
    file_path = os.path.join(SERV_CONF["DATA_DIR"], file_name)
    download_file(file_path, file_url)


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

    model_name = "resnet-v1"
    model_url = "https://drive.google.com/file/d/1H2PAciA9V0FxWkduNUo4Am7gRFoKsB15/view?usp=share_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "classification-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = 'CPU'

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
                    "depend_on": IMAGE_NET_LABEL,
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
            "device": device
        },
        replace=True
    )


def intel_sample_obj(db_path: str = SERV_CONF["DB_PATH"]):
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

    model_name = "yolo-v3-tf"
    model_url = "https://drive.google.com/file/d/1Ii8GBLEdUIC8I5e1P7YbQ6oc9dqizD0z/view?usp=share_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "object-detection-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = 'CPU'

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join(SERV_CONF["DATA_DIR"], source_name)
    source_uid = gen_uid(source_name)
    source_height, source_width = "720", "1280"

    model_uid = gen_uid(model_name)
    model_setting = {
        "confidence_threshold": 0.5
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
            "device": device
        },
        replace=True
    )


def intel_sample_detection_zone(db_path: str = SERV_CONF["DB_PATH"]):
    """ Add intel sample information into database 
    ---
    1. Download Data from Google Drive, and Update into Database
    2. Download Model from Google Drive, and Update into Database
    3. Update AI Task Information into Database
    """

    # Download data and model
    data_name = '4-corner-downtown-trim-1280x720.mp4'
    data_url = "https://drive.google.com/file/d/16xK3KBJdWKZWxjzTH-_VMxpYfasEFDtz/view?usp=share_link"
    download_data(data_name, data_url)

    model_name = "yolo-v3-tf"
    model_url = "https://drive.google.com/file/d/1Ii8GBLEdUIC8I5e1P7YbQ6oc9dqizD0z/view?usp=share_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "detection-zone-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = 'CPU'

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join(SERV_CONF["DATA_DIR"], source_name)
    source_uid = gen_uid(source_name)
    source_height, source_width = "720", "1280"

    model_uid = gen_uid(model_name)
    model_setting = {
        "confidence_threshold": 0.5
    }

    app_name = 'Detection_Zone'
    app_type = 'OBJ'
    app_uid = task_uid
    app_setting = {
        "application": {
            "palette": {
                "car": [
                    0,
                    255,
                    0
                ],
                "truck": [
                    0,
                    255,
                    0
                ]
            },
            "areas": [
                {
                    "name": "The intersection of Datong Rd",
                    "depend_on": [
                        "car",
                        "truck"
                    ],
                    "area_point": [
                        [
                            0.156,
                            0.203
                        ],
                        [
                            0.468,
                            0.203
                        ],
                        [
                            0.468,
                            0.592
                        ],
                        [
                            0.156,
                            0.592
                        ]
                    ],
                    "events": {
                        "title": "Traffic is very heavy",
                        "logic_operator": ">",
                        "logic_value": 100,
                    }
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
            "device": device
        },
        replace=True
    )


def intel_sample_tracking_zone(db_path: str = SERV_CONF["DB_PATH"]):
    """ Add intel sample information into database 
    ---
    1. Download Data from Google Drive, and Update into Database
    2. Download Model from Google Drive, and Update into Database
    3. Update AI Task Information into Database
    """

    # Download data and model
    data_name = '4-corner-downtown-trim-1280x720.mp4'
    data_url = "https://drive.google.com/file/d/16xK3KBJdWKZWxjzTH-_VMxpYfasEFDtz/view?usp=share_link"
    download_data(data_name, data_url)

    model_name = "yolo-v3-tf"
    model_url = "https://drive.google.com/file/d/1Ii8GBLEdUIC8I5e1P7YbQ6oc9dqizD0z/view?usp=share_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "tracking-zone-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = 'CPU'

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join(SERV_CONF["DATA_DIR"], source_name)
    source_uid = gen_uid(source_name)
    source_height, source_width = "720", "1280"

    model_uid = gen_uid(model_name)
    model_setting = {
        "confidence_threshold": 0.5
    }

    app_name = 'Tracking_Zone'
    app_type = 'OBJ'
    app_uid = task_uid
    app_setting = {
        "application": {
                    "palette": {
                        "car": [
                            0,
                            255,
                            0
                        ],
                        "truck": [
                            0,
                            255,
                            0
                        ]
                    },
            "areas": [
                {
                    "name": "Datong Rd",
                    "depend_on": [ 'car', 'truck'
                    ],
                    "area_point": [
                        [
                            0.156,
                            0.203
                        ],
                        [
                            0.468,
                            0.203
                        ],
                        [
                            0.468,
                            0.592
                        ],
                        [
                            0.156,
                            0.592
                        ]
                    ],
                    "events": {
                        "title": "The daily traffic is over 2",
                        "logic_operator": ">",
                        "logic_value": 100,
                    }
                },
            ],
            "draw_result":False,
            "draw_bbox":False
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
            "device": device
        },
        replace=True
    )



def intel_sample_movement_zone(db_path: str = SERV_CONF["DB_PATH"]):
    """ Add intel sample information into database 
    ---
    1. Download Data from Google Drive, and Update into Database
    2. Download Model from Google Drive, and Update into Database
    3. Update AI Task Information into Database
    """

    # Download data and model
    data_name = '4-corner-downtown-trim-1280x720.mp4'
    data_url = "https://drive.google.com/file/d/16xK3KBJdWKZWxjzTH-_VMxpYfasEFDtz/view?usp=share_link"
    download_data(data_name, data_url)

    model_name = "yolo-v3-tf"
    model_url = "https://drive.google.com/file/d/1Ii8GBLEdUIC8I5e1P7YbQ6oc9dqizD0z/view?usp=share_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "movement-zone-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = 'CPU'

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join(SERV_CONF["DATA_DIR"], source_name)
    source_uid = gen_uid(source_name)
    source_height, source_width = "720", "1280"

    model_uid = gen_uid(model_name)
    model_setting = {
        "confidence_threshold": 0.5
    }

    app_name = 'Movement_Zone'
    app_type = 'OBJ'
    app_uid = task_uid
    app_setting = {
        "application": {
            "palette": {
                        "car": [
                            0,
                            255,
                            0
                        ],
                        "truck": [
                            0,
                            255,
                            0
                        ]
                        },
            "areas": [
                {
                    "name": "area",
                    "depend_on": [
                        "car",
                        "truck"
                    ],
                    "area_point": [
                        [
                            0.156,
                            0.203
                        ],
                        [
                            0.468,
                            0.203
                        ],
                        [
                            0.468,
                            0.592
                        ],
                        [
                            0.156,
                            0.592
                        ]
                    ],
                    "line_point": {
                        "line_1": [
                            [
                                0.14666666666,
                                0.45074074074
                            ],
                            [
                                0.40291666666,
                                0.35962962963
                            ]
                        ],
                        "line_2": [
                            [
                                0.14666666666,
                                0.55074074074
                            ],
                            [
                                0.50291666666,
                                0.45962962963
                            ]
                        ],
                    },
                    "line_relation": [
                        {
                            "name": "To Taipei",
                            "start": "line_2",
                            "end": "line_1"
                        },
                        {
                            "name": "To Keelung",
                            "start": "line_1",
                            "end": "line_2"
                        }
                    ],
                    "events": {
                        "title": "Detect the traffic flow between Taipei and Keelung ",
                        "logic_operator": ">",
                        "logic_value": 2,
                    },
                }
            ],
            "draw_result":False,
            "draw_bbox":False
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
            "device": device
        },
        replace=True
    )


def init_intel_samples():
    intel_sample_cls()
    intel_sample_obj()
    intel_sample_detection_zone()
    intel_sample_tracking_zone()
    intel_sample_movement_zone()


if __name__ == "__main__":
    db_handler.init_sqlite()
    intel_sample_cls()
    intel_sample_obj()
