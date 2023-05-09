# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import threading, time
import logging as log
import asyncio

# Custom
from ..common import SERV_CONF, RT_CONF, WS_CONF
from ..utils import gen_uid, json_to_str

from .ai_handler import get_ivit_api
from .io_handler import create_displayer, create_source, update_src_status
from . import task_handler
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
from .ivit_handler import Metric


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
        task_uid = app_uid = info['uid']
        task_name = info['name']
        source_uid = info['source_uid']
        model_uid = info['model_uid']
        
        # Another Information: Source, Model, App
        try:
            source_name = select_data(table='source', data=["name"], condition=f"WHERE uid='{source_uid}'")[0][0]
        except Exception as e:
            log.exception(e)

        try:
            model_name, model_type = select_data(table='model', data=["name", "type"], condition=f"WHERE uid='{model_uid}'")[0]
            
        except Exception as e:
            log.exception(e)

        try:
            app_uid, app_name = select_data(table='app', data=["uid", "name"], condition=f"WHERE uid='{task_uid}'")[0]
        except Exception as e:
            log.exception(e)
        
        info.update({
            'task_uid': task_uid,
            'task_name': task_name,
            'source_name': source_name,
            'model_name': model_name,
            'model_type': model_type,
            'app_uid': app_uid,
            'app_name': app_name,
            'error': info['error']
        })
        
        for key in [ 'uid', 'name' ]:
            info.pop(key, None)
        
        ret.append(info)

    return ret


def get_task_status(uid):
    return select_data(table='task', data=['status'], condition=f"WHERE uid='{uid}'")[0][0]


def update_task_status(uid, status, err_mesg:dict = {}):
    write_data = {
        'status':status,
        'error': err_mesg
    }
    update_data(table='task', data=write_data, condition=f"WHERE uid='{uid}'")


def verify_task_exist(uid):
    # Task Information
    task = select_data(table='task', data="*", condition=f"WHERE uid='{uid}'")
    
    # Not found AI Task
    if task == []:
        raise RuntimeError("Could not find AI Task ({})".format(uid))

    return task[0]


def verify_duplicate_task(task_name):
    # Check exist AI Task
    for uid, name in select_data(table='task', data=['uid','name']):
        if task_name in name:
            raise NameError('AI Task is already exist ( {}: {} )'.format(uid, name))


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
        raise e


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
            dpr=dpr 
        ),
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
    task_uid = app_uid = gen_uid()

    # CHECK: task exist
    verify_duplicate_task(add_data.task_name)

    # CHECK: threshold value is available
    verify_thres(threshold=add_data.model_setting['confidence_threshold'])

    # Add Task Information into Database
    insert_data(
        table="task",
        data={
            "uid": task_uid,
            "name": add_data.task_name,
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
    task_uid = app_uid = edit_data.task_uid

    # CHECK: AI task is exist or not
    verify_task_exist(task_uid)

    # CHECK: task exist
    verify_duplicate_task(edit_data.task_name)

    # CHECK: threshold value is available
    verify_thres(threshold=edit_data.model_setting['confidence_threshold'])

    # Add Task Information into Database
    insert_data(
        table="task",
        data={
            "uid": task_uid,
            "name": edit_data.task_name,
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


class FakeDisplayer:
    log.warning('Initialize Fake Displayer')
    show = lambda frame: frame
    release = lambda: log.warning('Release Fake Displayer')


class InferenceLoop:
    """ Inference Thread Helper """

    def __init__(self, uid, src, model, app, dpr=None) -> None:
        self.uid = uid
        self.src = src
        self.model = model
        self.app = app
        self.dpr = dpr if dpr else FakeDisplayer()

        self.is_ready = True
        self.thread_object = None
        
        self.draw = None
        self.results = None
        self.metric = Metric()
        self.latency_limitor = Metric()

        self.minimum_latency = 1/self.src.get_fps()

        self.icap_alive = 'ICAP' in SERV_CONF and not (SERV_CONF['ICAP'] is None)
        
        log.warning('Create a InferenceLoop')

    def create_thread(self) -> threading.Thread:
        return threading.Thread(target=self._inference_thread, daemon=True)


    def _dynamic_change_app(self):
        """ Dynamic Modify Varaible of the Application """

        # Return: No dynamic variable setting
        if getattr(RT_CONF[self.uid], 'DATA', {}):
            return
        
        # Debug: with raise key
        if getattr(RT_CONF[self.uid]['DATA'], 'raise', False):
            raise RuntimeError('Raise Testing when AI Task Running')

        try:
            # Area Event: Color, ... etc
            areas = getattr(RT_CONF[self.uid]['DATA'], 'area', None)
            if areas:
                for area in areas:
                    palette = area.get('palette')
                    if palette:
                        for key, val in palette.items():
                            self.app.palette[key] = val
                            try:
                                self.app.params['application']['areas'][0]['palette'][key]=val
                            except: pass
        
        except Exception as e:
            log.error('Setting Area Failed')
            log.exception(e)
        
        finally:
            # Clear DATA
            RT_CONF[self.uid]['DATA'] = {}


    def _inference_thread(self):
        
        prev_data, cur_data = None, None

        try:
            log.warning('Start AI Task Inference Stream')
            self.prev_time = time.time()

            # Make sure source is ready
            while( self.is_ready and self.src.is_ready):
                
                # Limit Speed to fit FPS
                if self.minimum_latency < self.latency_limitor.update():
                    time.sleep(1e-3); continue
                
                # Ready to calculate performance
                self.metric.update()

                # Setting Dynamic Variable
                self._dynamic_change_app()
 
                # Get data        
                ret, frame = self.src.read()

                # Do Sync Inference
                cur_data = self.model.inference(frame)

                # Run Application
                self.draw, self.results = self.app(frame, cur_data, draw=True)
            
                # Display
                self.dpr.show(self.draw)

                # Log
                log.debug(cur_data)

                # Send Data
                self.results.update({"fps": self.metric.get_fps()})
                WS_CONF.update({ self.uid: self.results })

                # Calculate FPS and Update spped_limitor
                self.metric.update()
                self.latency_limitor.update()

            # Check is error from source or not
            if (not self.src.is_ready) and len(self.src.errors) > 0:
                raise self.src.errors[-1]

            # Update Task Status to Stop
            update_task_status(self.uid, 'stop')
            log.warning('Stop InferenceLoop')

        # If Get Exception
        except Exception as e:
            
            # Write Log
            log.error('InferenceLoop Error!!!')
            log.exception(e)

            # Send and Store Error Message with Json Format
            json_exp = json_exception(e)
            if "WS" in WS_CONF:
                asyncio.run( WS_CONF["WS"].send_json({"ERROR": json_exp}) )
            update_task_status(
                uid=self.uid, status='error', err_mesg=json_exp)

        finally:

            self.model.release()
            self.dpr.release()
            if self.icap_alive:
                SERV_CONF['ICAP'].send_attr(data=task_handler.get_task_info())

        log.warning('InferenceLoop ({}) is Stop'.format(self.uid))

    def get_results(self):
        return self.results

    def stop(self):
        self.is_ready = False
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
        









