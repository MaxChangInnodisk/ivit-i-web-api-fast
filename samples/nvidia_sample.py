# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import gdown
import sqlite3
import os
import time

try:
    from .imagenet import IMAGE_NET_LABEL
    from .coco import COCO_LABEL
    from .utils import download_data, download_model, load_palette
except:
    from imagenet import IMAGE_NET_LABEL
    from coco import COCO_LABEL
    from utils import download_data, download_model, load_palette

try:
    from ..common import SERV_CONF
    from ..utils import gen_uid, json_to_str
    from ..handlers import db_handler, model_handler
except:
    from common import SERV_CONF
    from utils import gen_uid, json_to_str
    from handlers import db_handler, model_handler

from ivit_i.utils import iDevice

# Parameters
DEV = iDevice().get_available_device()[0]
CREATED_TIME = time.time()

def nv_sample_cls(db_path: str = SERV_CONF["DB_PATH"]):
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
    
    model_name = "resnet34"
    model_url = "https://drive.google.com/file/d/1VOL7jDNvehjoBwKFUcOYbFw_3O7W3oOW/view?usp=drive_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "classification-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = DEV

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
            "palette": {
                "airplane": [255, 255, 255],
                "warpalne": [0, 0, 0],
            },
            "areas": [
                {
                    "name": "default",
                    "depend_on": IMAGE_NET_LABEL
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


def nv_sample_obj_yolov4_tiny(db_path: str = SERV_CONF["DB_PATH"]):
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

    model_name = "yolov4-tiny-416"
    model_url = "https://drive.google.com/file/d/1k8ycjBQS3GiUdHNIkD1PpFOe0iu5fUKG/view?usp=drive_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "yolov4-tiny-416-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = DEV

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join(SERV_CONF["DATA_DIR"], source_name)
    source_uid = gen_uid(source_name)
    source_height, source_width = "720", "1280"

    model_uid = gen_uid(model_name)
    model_setting = {
        "confidence_threshold": 0.7
    }

    # Get label and palette
    data = model_handler.parse_model_folder(os.path.join(SERV_CONF["MODEL_DIR"], model_name))
    label_path = data["label_path"]
    trg_palette = load_palette(label_path=label_path)

    app_name = 'Basic_Object_Detection'
    app_type = 'OBJ'
    app_uid = task_uid
    app_setting = {
        "application": {
            "palette":trg_palette,
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


def nv_sample_obj_yolov4(db_path: str = SERV_CONF["DB_PATH"]):
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

    model_name = "yolov4-416"
    model_url = "https://drive.google.com/file/d/12PmVMv87887aYBOtK76BUed3ibieOVOD/view?usp=drive_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "yolov4-416-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = DEV

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join(SERV_CONF["DATA_DIR"], source_name)
    source_uid = gen_uid(source_name)
    source_height, source_width = "720", "1280"

    model_uid = gen_uid(model_name)
    model_setting = {
        "confidence_threshold": 0.7
    }

    # Get label and palette
    data = model_handler.parse_model_folder(os.path.join(SERV_CONF["MODEL_DIR"], model_name))
    label_path = data["label_path"]
    trg_palette = load_palette(label_path=label_path)

    app_name = 'Basic_Object_Detection'
    app_type = 'OBJ'
    app_uid = task_uid
    app_setting = {
        "application": {
            "palette":trg_palette,
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


def nv_sample_detection_zone(db_path: str = SERV_CONF["DB_PATH"]):
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

    model_name = "yolov4-416"
    model_url = "https://drive.google.com/file/d/12PmVMv87887aYBOtK76BUed3ibieOVOD/view?usp=drive_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "detection-zone-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = DEV

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
                    105,
                    125,
                    105
                ],
                "truck": [
                    125,
                    115,
                    105
                ]
            },
            "areas": [
                {
                    "name": "Area0",
                    "depend_on": [
                        "car",
                    ],
                    "area_point": [
                        [
                            0.256,
                            0.583
                        ],
                        [
                            0.658,
                            0.503
                        ],
                        [
                            0.848,
                            0.712
                        ],
                        [
                            0.356,
                            0.812
                        ]
                    ]
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


def nv_sample_tracking_zone(db_path: str = SERV_CONF["DB_PATH"]):
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

    model_name = "yolov4-416"
    model_url = "https://drive.google.com/file/d/12PmVMv87887aYBOtK76BUed3ibieOVOD/view?usp=drive_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "tracking-zone-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = DEV

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
                            105,
                            125,
                            105
                        ],
                        "truck": [
                            125,
                            115,
                            105
                        ]
                    },
            "areas": [
                {
                    "name": "Area0",
                    "depend_on": [ 'car', 'truck'
                    ],
                    "area_point": [
                        [
                            0.256,
                            0.583
                        ],
                        [
                            0.658,
                            0.503
                        ],
                        [
                            0.848,
                            0.712
                        ],
                        [
                            0.356,
                            0.812
                        ]
                    ]
                }
            ],
            "draw_result":True,
            "draw_bbox":True
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


def nv_sample_movement_zone(db_path: str = SERV_CONF["DB_PATH"]):
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

    model_name = "yolov4-416"
    model_url = "https://drive.google.com/file/d/12PmVMv87887aYBOtK76BUed3ibieOVOD/view?usp=drive_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "movement-zone-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = DEV

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
                    105,
                    125,
                    105
                ],
                "truck": [
                    125,
                    115,
                    105
                ]
            },
            "areas": [
                {
                    "name": "Area0",
                    "depend_on": [
                        'car', 'truck'
                    ],
                    "area_point": [
                        [
                            0.256,
                            0.583
                        ],
                        [
                            0.658,
                            0.503
                        ],
                        [
                            0.848,
                            0.712
                        ],
                        [
                            0.356,
                            0.812
                        ]
                    ],
                    "line_point": {
                        "line_1": [
                            [
                                0.36666666666,
                                0.64074074074
                            ],
                            [
                                0.67291666666,
                                0.52962962963
                            ]
                        ],
                        "line_2": [
                            [
                                0.36041666666,
                                0.83333333333
                            ],
                            [
                                0.72916666666,
                                0.62962962963
                            ]
                        ],
                    },
                    "line_relation": [
                        {
                            "name": "to Taipei",
                            "start": "line_2",
                            "end": "line_1"
                        },
                        {
                            "name": "To Keelung",
                            "start": "line_1",
                            "end": "line_2"
                        }
                    ],
                },
            ],
            "draw_result":True,
            "draw_bbox":True
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


def init_nvidia_samples():

    nv_sample_cls()
    # nv_sample_obj()
    nv_sample_obj_yolov4_tiny()
    nv_sample_obj_yolov4()
    nv_sample_detection_zone()
    nv_sample_tracking_zone()
    nv_sample_movement_zone()


if __name__ == "__main__":
    db_handler.init_sqlite()
    nv_sample_cls()
    nv_sample_obj()
