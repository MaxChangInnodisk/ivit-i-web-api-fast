# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import gdown, sqlite3, os, sys, time, json
import logging as log

from .imagenet import IMAGE_NET_LABEL
from .coco import COCO_LABEL

from ..common import SERV_CONF
from ..utils import gen_uid, json_to_str
from ..handlers import db_handler

def download_file(file_path, file_url):
    check_data = file_path
    if os.path.splitext(file_path)[1] == '.zip':
        check_data = os.path.splitext(file_path)[0]

    if os.path.exists(check_data):
        return
    
    gdown.download(url=file_url, output=file_path, quiet=False, fuzzy=True)

def download_model(file_name, file_url):
    file_path = os.path.join(SERV_CONF["MODEL_DIR"], file_name)
    download_file(file_path, file_url)

def download_data(file_name, file_url):
    file_path = os.path.join(SERV_CONF["DATA_DIR"], file_name)
    download_file(file_path, file_url)


def xlnx_sample_cls( db_path:str=SERV_CONF["DB_PATH"] ):
    """ Add xlnx sample information into database 
    ---
    1. Download Data from Google Drive, and Update into Database
    2. Download Model from Google Drive, and Update into Database
    3. Update AI Task Information into Database
    """

    # Download data and model
    data_name = 'cat.jpg'
    data_url = "https://drive.google.com/file/d/1hq2CvCT4SRTvvkHo3QVZSh85bdUuiXbR/view?usp=sharing"
    download_data(data_name, data_url)

    model_name = "mobilenet_1_1_224_tf2"
    model_url = "https://drive.google.com/file/d/1U6oZOtMEIynU_TI21RyuEvy-1OmkBgI3/view?usp=sharing"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "classification-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = 'DPU'

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join( SERV_CONF["DATA_DIR"], source_name)
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
            "areas":[
                {
                    "name": "default",
                    "depend_on": IMAGE_NET_LABEL,
                }
            ],
            "palette": {
                "airplane": [255, 255, 255],
                "warpalne": [0, 0, 0],
            }
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
        replace=True )

    # Application
    db_handler.insert_data(
        table="app",
        data={
            "uid": app_uid,
            "name": app_name,
            "type": app_type,
            "app_setting": json_to_str(app_setting)
        },
        replace = True
    )

    db_handler.insert_data(
        table="task",
        data={
            "uid": task_uid,
            "name": task_name,
            "source_uid": source_uid,
            "model_uid": model_uid,
            "model_setting": json_to_str(model_setting),
            "status": task_status,
            "device": device
        },
        replace=True
    )

def xlnx_sample_obj( db_path:str=SERV_CONF["DB_PATH"] ):
    """ Add xlnx sample information into database 
    ---
    1. Download Data from Google Drive, and Update into Database
    2. Download Model from Google Drive, and Update into Database
    3. Update AI Task Information into Database
    """

    # Download data and model
    data_name = 'car.mp4'
    data_url = "https://drive.google.com/file/d/15X2EwgdtNmLgaLOWlMyBytJ3gM1c1Wqd/view?usp=sharing"
    download_data(data_name, data_url)

    model_name = "yolov3_voc"
    model_url = "https://drive.google.com/file/d/117u1B3wIvljIky9lL90CxS_qJrwkX26x/view?usp=share_link"
    download_model(model_name, model_url)

    # Define Parameters
    task_name = "object-detection-sample"
    task_status = 'stop'
    task_uid = gen_uid()

    device = 'DPU'

    source_status = 'stop'
    source_name = data_name
    source_type = "IMAGE"
    source_input = os.path.join( SERV_CONF["DATA_DIR"], source_name)
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
        replace=True )

    # Application
    db_handler.insert_data(
        table="app",
        data={
            "uid": app_uid,
            "name": app_name,
            "type": app_type,
            "app_setting": json_to_str(app_setting)
        },
        replace = True
    )

    db_handler.insert_data(
        table="task",
        data={
            "uid": task_uid,
            "name": task_name,
            "source_uid": source_uid,
            "model_uid": model_uid,
            "model_setting": json_to_str(model_setting),
            "status": task_status,
            "device": device
        },
        replace=True
    )



def init_xlnx_samples():
    xlnx_sample_cls()
    xlnx_sample_obj()