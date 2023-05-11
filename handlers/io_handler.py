# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import time, os, shutil, cv2, copy
import logging as log
import numpy as np
from ..common import RT_CONF, SERV_CONF
from ..utils import gen_uid

from .db_handler import (
    select_data, 
    insert_data,
    update_data,
    delete_data,
    parse_source_data
)

# iVIT Libraray
from .ivit_handler import SourceV2, Displayer

K_SRC = 'SRC'


def get_src_status(source_uid):
    return select_data(table='source', data=['status'], condition=f"WHERE uid='{source_uid}'")[0][0]


def update_src_status(source_uid, status):
    update_data(table='source', data={'status': status}, condition=f"WHERE uid='{source_uid}'")

# Helper Function
def get_src_info(uid: str=None):
    """ Get Source Information from database """
    if uid == None:    
        data = select_data(table='source', data="*")
    else:
        data = select_data(table='source', data="*", condition=f"WHERE uid='{uid}'")
    if data==[]:
        raise KeyError('Could not find source: {}'.format(uid))
    ret = [ parse_source_data(src) for src in data ]
    return ret

def create_source(source_uid:str):
    """ Create source """

    # Source Information
    source = select_data(
        table='source', data="*",
        condition=f"WHERE uid='{source_uid}'" )
    assert len(source) >= 1, "Could not find source"
    source = source[0]
    src_info = parse_source_data(source)

    # Update Key    
    if RT_CONF.get(K_SRC) is None:
        RT_CONF.update( { K_SRC: {} })

    # Check Source
    timeout = 0
    while(True):

        status = get_src_status(source_uid)
        
        if status in [ 'running',  "loaded" ]:
            mesg = 'Source is running'
            src = RT_CONF[K_SRC].get(source_uid)
            if src is None:
                break
            return src
    
        elif status in ["stop", "error"]:
            update_src_status(source_uid, 'loading')
            break
        else:
            print('Waiting Source Object ...')
            timeout += 1
            time.sleep(1)
        
        if timeout >=10:
            raise RuntimeError('Initialize Source Object Timeout')

    # Initialize Source         
    src_object = SourceV2(
        input=src_info['input'], 
        resolution=src_info.get('resolution'), 
        fps=src_info.get('fps') )

    # Update into RT_CONF
    RT_CONF[K_SRC].update( { source_uid: src_object } )
    log.info('Update {} to {}'.format(source_uid, SERV_CONF.get_name))

    log.info('Initialized Source')
    return src_object


def add_source(files=None, input: str=None, option: dict=None) -> dict:
    """ Add new source function """

    def save_file(file, path):
        """ Save file from fastapi """
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        shutil.move(file.filename, path)

    # Got RTSP, V4L2
    if (files is None) or not (files[0].filename):
        name = input
    
    # Got Files
    else:
        
        # Multi File
        if len(files)>1:

            log.info('Add Multiple File')
            folder_name = "dataset-{}".format(gen_uid(files[0].filename))

            folder_path = os.path.join( SERV_CONF["DATA_DIR"], folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                log.warning('Create Folder: {}'.format(folder_path))
                
            for file in files:
                path = os.path.join( folder_path, file.filename )
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

    resolution = option.get('resolution')
    fps = option.get('fps')
    uid = gen_uid(name)
    
    src = SourceV2(
        input=input, 
        resolution=resolution, 
        fps=fps)
    
    type = src.get_type()
    height, width = src.get_shape()

    data = {
        "uid": uid,
        "name": name,
        "type": type,
        "input": input,
        "height": str(height), 
        "width": str(width),
        "status": "stop"
    }

    insert_data(table="source", data=data, replace=True)

    src.release()

    return data


def get_source_frame(source_uid:str, resolution:list=None) -> np.ndarray:
    """Get source frame with target resolution

    Args:
        source_uid (str): the uid of source
        resolution (list, optional): (height, width). Defaults to None.

    Returns:
        cv2.ndarray: _description_
    """
    src = create_source(source_uid=source_uid)
    frame = copy.deepcopy(src.frame)
    src.release()
    update_data(table="source", data={"status": "stop"}, condition='WHERE uid="{}"'.format(source_uid))
    
    if not (resolution is None):
        frame = cv2.resize( frame, (resolution[1], resolution[0]))
    
    return frame

create_displayer = Displayer