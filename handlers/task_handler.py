# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import threading, time
import logging as log
import asyncio

# Custom
from ..common import SERV_CONF, RT_CONF, WS_CONF
from ..utils import gen_uuid, json_to_str

from .ai_handler import get_ivit_api
from .io_handler import create_displayer, create_source, update_src_status
from .app_handler import create_app
from .mesg_handler import json_exception
from .db_handler import (
    select_data,
    update_data,
    delete_data,
    insert_data,
    parse_task_data,
    parse_model_data
)


def get_task_info(uid:str=None):
    """ Get AI Task Information from Database  """
    
    # Get Data
    if uid == None:    
        data = select_data(table='task', data="*")
    else:
        data = select_data(table='task', data="*", condition=f"WHERE uid='{uid}'")

    # Parse Data
    ret = []
    for task in data:

        # Task Information
        info = parse_task_data(task)
        task_uid = info['uid']
        source_uid = info['source_uid']
        model_uid = info['model_uid']
        
        # Another Information: Source, Model, App
        try:
            source_name = select_data(table='source', data=["name"], condition=f"WHERE uid='{source_uid}'")[0][0]
        except Exception as e:
            
            log.exception(e)
        try:
            model_name = select_data(table='model', data=["name"], condition=f"WHERE uid='{model_uid}'")[0][0]
        except Exception as e:
            log.exception(e)
        try:
            app_name = select_data(table='app', data=["name"], condition=f"WHERE uid='{task_uid}'")[0][0] 
        except Exception as e:
            log.exception(e)
        
        info.update({
            'source_name': source_name,
            'model_name': model_name,
            'app_uid': info['uid'],
            'app_name': app_name
        })
        
        ret.append(info)

    return ret


def get_task_status(uid):
    return select_data(table='task', data=['status'], condition=f"WHERE uid='{uid}'")[0][0]


def update_task_status(uid, status):
    update_data(table='task', data={'status':status}, condition=f"WHERE uid='{uid}'")



def verify_task_exist(uid):
    # Task Information
    task = select_data(table='task', data="*", condition=f"WHERE uid='{uid}'")
    
    # Not found AI Task
    if task == []:
        raise RuntimeError("Could't find AI Task ({})".format(uid))

    return task[0]


def verify_duplicate_task(task_name):
    # Check exist AI Task
    for name in select_data(table='task', data=['name']):
        if task_name in name:
            raise NameError('AI Task is already exist')


def verify_thres(threshold:float):
    if threshold < 0.1 and threshold > 1.0:
        raise ValueError('Threshold value should in range 0 to 1')


def run_ai_task(uid:str, data:dict=None):
    """ Run AI Task via uid """

    # Check Keyword in RT_CONF
    def check_rt_conf(uid):
        if RT_CONF.get(uid) is None:
            RT_CONF.update( {uid: {'EXEC': None}} )

    # Task Information
    task = verify_task_exist(uid=uid)
    task_info   = parse_task_data(task)
    source_uid  =  task_info['source_uid']
    model_uid   = task_info['model_uid']

    # Check Status: Wait Task if there is someone launching task
    while(True):

        status = get_task_status(uid)
        
        if status == 'running':
            mesg = 'AI Task is running'
            log.warn(mesg)
            return mesg    
        elif status == "stop":
            update_task_status(uid, 'loading')
            break
        elif status == 'error':
            update_task_status(uid, 'loading')
            break
        else:
            time.sleep(1)

    # Model Information
    model_data = select_data( table='model', data="*", 
                            condition=f"WHERE uid='{model_uid}'")[0]
    model_info = parse_model_data(model_data)

    # Load Model: Config
    model_info['meta_data'].update( 
        {**task_info['model_setting'], 'device': task_info['device'] } )
    
    # Load Model
    load_model = get_ivit_api(SERV_CONF["FRAMEWORK"])
    model = load_model( model_info['type'], model_info['meta_data'])
    
    # Source & Displayer
    try:
        src = create_source(source_uid=source_uid)
        height, width = src.get_shape()
        log.warning('{}, {}'.format(height, width))

    except Exception as e:
        log.exception(e)
        update_data(table='source', data={'status':'error'}, condition=f"WHERE uid='{source_uid}'")
        raise RuntimeError('Initialize Source Failed !')

    try:
        cv_flag = getattr(data, 'cv_display', None) 
        # Parse Data
        dpr = create_displayer(
            cv=cv_flag,
            rtsp=True, 
            height=height, 
            width=width, 
            fps=src.get_fps(),
            name=uid, 
            platform='intel')
            
    except Exception as e:
        log.exception(e)
        raise RuntimeError('Initialize Displayer Failed !')

    # Load Application
    try:
        app = create_app(app_uid=uid, label_path=model_info['label_path'])
    except Exception as e:
        log.exception(e)
        raise RuntimeError('Initialize Application Faild !')
    
    # Create Threading
    check_rt_conf(uid=uid)

    RT_CONF[uid].update({ 
        'EXEC': InferenceLoop(
        uid=uid,
        src=src,
        model=model,
        app=app,
        dpr=dpr ),
        'DATA': {} 
    })

    # Start each threading
    src.start()
    RT_CONF[uid]['EXEC'].start()
    
    # Change Status
    update_task_status(uid, 'running')
    update_src_status(source_uid, 'running')

    # End
    mesg = 'Run AI Task ( {}: {} )'.format(uid, task_info['name'] )
    return mesg


def stop_ai_task(uid:str, data:dict=None):
    """ Stop AI Task via uid """

    # Task Information
    task = verify_task_exist(uid=uid)
    task_info = parse_task_data(task)
    source_uid=  task_info['source_uid']

    if get_task_status(uid=uid)!='running':
        mesg = 'AI Task is stopped !'
        return mesg

    mesg = 'Stop AI Task ( {}: {} )'.format(uid, task_info['name'] )

    # Stop AI Task
    RT_CONF[uid]['EXEC'].stop()
    RT_CONF[uid]['EXEC'] = None
    update_task_status(uid, 'stop')
    
    # Stop Source if not using
    status_list = select_data(table='task', data=['status'], condition=f"WHERE source_uid='{source_uid}'")
    at_least_one_using = True in [ 'running' in status for status in status_list ]
    
    if not at_least_one_using:
        RT_CONF['SRC'][source_uid].release()
        update_src_status(source_uid, 'stop')

    log.info(mesg)
    return mesg


def update_ai_task(uid:str, data:dict=None):
    """ update AI Task via uid """
    
    RT_CONF[uid]['DATA'] = data
    log.info('Update AI Task: {}'.format(uid))    
    return 'Update success'


def add_ai_task(add_data):
    """ Add a AI Task 
    ---
    1. Add task information into database ( table: task )
    2. Add application information into database ( table: app )
    """
    
    # Generate Task UUID and Application UUID    
    task_uid = app_uid = gen_uuid()

    # CHECK: task exist
    verify_duplicate_task(add_data.name)

    # CHECK: threshold value is available
    verify_thres(threshold=add_data.model_setting['confidence_threshold'])

    # Add Task Information into Database
    insert_data(
        table="task",
        data={
            "uid": task_uid,
            "name": add_data.name,
            "source_uid": add_data.source_uid,
            "model_uid": add_data.model_uid,
            "model_setting": json_to_str(add_data.model_setting),
            "status": "stop",
            "device": add_data.device
        }
    )
    
    # Add App Information into Database
    app_type = select_data(table='app', data=['type'], condition=f"WHERE name='{add_data.app_name}'")[0][0]
    insert_data(
        table="app",
        data={
            "uid": app_uid,
            "name": add_data.app_name,
            "type": app_type,
            "app_setting": json_to_str(add_data.app_setting)
        }
    )
    
    return {
        "uid": task_uid,
        "status": "success"
    }


def edit_ai_task(edit_data):
    """ Edit AI Task
    ---
    
    """
    # Get Task UUID and Application UUID    
    task_uid = app_uid = edit_data.uid

    # CHECK: AI task is exist or not
    verify_task_exist(task_uid)

    # CHECK: task exist
    verify_duplicate_task(edit_data.name)

    # CHECK: threshold value is available
    verify_thres(threshold=edit_data.model_setting['confidence_threshold'])

    # Add Task Information into Database
    insert_data(
        table="task",
        data={
            "uid": task_uid,
            "name": edit_data.name,
            "source_uid": edit_data.source_uid,
            "model_uid": edit_data.model_uid,
            "model_setting": json_to_str(edit_data.model_setting),
            "status": "stop",
            "device": edit_data.device
        },
        replace=True
    )
    
    # Add App Information into Database
    app_type = select_data(table='app', data=['type'], condition=f"WHERE name='{edit_data.app_name}'")[0][0]
    insert_data(
        table="app",
        data={
            "uid": app_uid,
            "name": edit_data.app_name,
            "type": app_type,
            "app_setting": json_to_str(edit_data.app_setting)
        },
        replace=True
    )
    
    return {
        "uid": task_uid,
        "status": "success"
    }


def del_ai_task(uid:str):
    """ Delete a AI Task 
    1. Delete task information into database ( table: task )
    2. Delete application information into database ( table: app )
    """

    verify_task_exist(uid)

    # Del Task
    delete_data(
        table='task',
        condition=f"WHERE uid='{uid}'")
    
    # Del App
    delete_data(
        table='app',
        condition=f"WHERE uid='{uid}'")


class InferenceLoop:
    """ Inference Thread Helper """

    def __init__(self, uid, src, model, app, dpr=None) -> None:
        self.uid = uid
        self.src = src
        self.model = model
        self.app = app
        self.dpr = dpr        

        self.stop_thread = False
        self.thread_object = None
        
        self.draw = None
        self.results = None
        
        log.warning('Create a InferenceLoop')

    def create_thread(self) -> threading.Thread:
        return threading.Thread(target=self._inference_thread, daemon=True)

    def _inference_thread(self):
        
        prev_data, cur_data = None, None

        try:
            log.warning('Start AI Task Inference Stream')
            frame_idx = 0
            t_wait_fps = time.time()
            while(not self.stop_thread):
                
                # Limit Speed to fit FPS
                if 1/(time.time()-t_wait_fps) > self.src.get_fps():
                    time.sleep(0.033); continue

                # Dynamic Modify Varaible
                if getattr(RT_CONF[self.uid]['DATA'], 'raise', False):
                    raise RuntimeError('Raise Testing when AI Task Running')
                areas = getattr(RT_CONF[self.uid]['DATA'], 'area', None)
                if areas:
                    for area in areas:
                        palette = area.get('palette')
                        log.info(palette)
                        if palette:
                            for key, val in palette.items():
                                self.app.palette[key] = val
                                try:
                                    self.app.params['application']['areas'][0]['palette'][key]=val
                                except: pass
                RT_CONF[self.uid]['DATA'] = {}

                # Get data
                ret, frame = self.src.read()
                frame_idx += 1

                # Do Async Inference
                cur_data = self.model.inference(frame)
                
                # Update Inference Data if need
                prev_data = cur_data if cur_data else prev_data

                # Check Previous Data    
                print(cur_data)

                if not prev_data: continue

                # Run Application
                self.draw, self.results = self.app(frame, prev_data, draw=True)
            
                # Display
                if self.dpr: 
                    self.dpr.show(self.draw)

                # Send Data
                WS_CONF.update({ self.uid: self.results })

        except Exception as e:
            log.exception(e)
            if "WS" in WS_CONF:
                asyncio.run( WS_CONF["WS"].send_json({"ERROR": json_exception(e)}) )
            update_task_status(self.uid, 'error')

        finally:
            self.model.release()
            if self.dpr:
                self.dpr.release()
        log.warning('InferenceLoop ({}) is Stop'.format(self.uid))

    def get_results(self):
        return self.results

    def stop(self):
        self.stop_thread = True
        self.thread_object.join()
        self.create_thread()

    def start(self):
        if self.thread_object is None:
            self.thread_object = self.create_thread()
        
        if not self.thread_object.is_alive():
            self.thread_object.start()

    def release(self):
        self.stop()
        del self.thread_object
        self.thread_object = None
        









