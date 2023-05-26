# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import threading, time, sqlite3, copy
import logging as log
import asyncio
from typing import Union
import numpy as np
from multiprocessing.pool import ThreadPool

# Custom
try:
    from ..common import SERV_CONF, RT_CONF, WS_CONF
    from ..utils import gen_uid, json_to_str
except:
    from common import SERV_CONF, RT_CONF, WS_CONF
    from utils import gen_uid, json_to_str

from .ai_handler import get_ivit_api, iModel
from .io_handler import (
    create_displayer, 
    create_source, 
    update_src_status, 
    start_source, 
    stop_source,
    is_source_using
)
from . import task_handler
from .app_handler import create_app
from .mesg_handler import json_exception, handle_exception, simple_exception
from .err_handler import InvalidError, InvalidUidError
from .db_handler import (
    select_data,
    update_data,
    delete_data,
    insert_data,
    parse_task_data,
    parse_model_data,
    db_to_list,
    is_list_empty,
    connect_db,
    close_db,
    select_column_by_uid,
    update_column_by_uid
)
from .ivit_handler import Metric

# --------------------------------------------------------

def get_task_info(uid:str=None):
    """ Get AI Task Information from Database
    1. Get All AI Task Information
    2. Parse Each AI Task Information
    """

    # Basic Params
    return_data, return_errors = [], []

    # Open DB
    con, cur = connect_db()

    # Move DB Cursor
    results = db_to_list( cur.execute(
        '''SELECT * FROM task''' if uid == None \
            else """SELECT * FROM task WHERE uid=\"{}\"""".format(uid)
    ))

    # Check DB Data
    if is_list_empty(results):
        raise InvalidUidError("Got invalid task uid: {}".format(uid))
    
    # Get Data
    for info in map(parse_task_data, results):
  
        task_uid = app_uid = info['uid']
        task_name = info['name']
        source_uid = info['source_uid']
        model_uid = info['model_uid']
        error = info['error']
        
        # Source
        results = select_column_by_uid(cur, 'source', source_uid, ["name"])

        if is_list_empty(results):
            source_name = None
            return_errors.append( InvalidUidError(f'Got invalid source uid: {source_uid}') )
        else:
            source_name = results[0]

        # Model
        results = select_column_by_uid(cur, 'model', model_uid, ["name", "type"])
        if is_list_empty(results):
            model_name, model_type = None, None
            return_errors.append( InvalidUidError(f'Got invalid source uid: {model_uid}') )
        else:
            model_name, model_type = results[0]
        
        # App
        results = select_column_by_uid(cur, "app", app_uid, ["name"])
        if is_list_empty(results):
            return_errors.append( InvalidUidError(f'Got invalid application uid: {app_uid}') )
            app_name = None
        else:
            app_name = results[0]

        # Add More Data
        info.update({
            'task_uid': task_uid,
            'task_name': task_name,
            'source_name': source_name,
            'model_name': model_name,
            'model_type': model_type,
            'app_uid': app_uid,
            'app_name': app_name,
            'error': return_errors if return_errors!=[] else error
        })

        # Pop unused data
        for key in [ 'uid', 'name' ]:
            info.pop(key, None)
        
        return_data.append(info)

    # Cloase DB
    close_db(con, cur)

    return return_data


def get_task_status(uid):
    """Get Status of AI Task"""
    return select_data(table='task', data=['status'], condition=f"WHERE uid='{uid}'")[0][0]


def update_task_status(uid, status, err_mesg:dict = {}):
    """Update Status of AI Task"""    
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


# --------------------------------------------------------
# Execute AI Task

def run_ai_task(uid:str, data:dict=None) -> str:
    """Run AI Task
    
    Workflow
    1. Check task is exist
    2. Parse task information
    3. Check task status
        If running then return message; 
        If stop or error then keep going; 
        If loading then wait for N seconds ( default is 20 ).
    4. Get model information and load model
    5. Get source object and displayer object
    6. Load Application
    7. Create InferenceLoop and keep in RT_CONF[<uid>]['EXEC']
    8. Start source and Start InferenceLoop
    """
    # Get Task Information
    task = select_data(table='task', data="*", condition=f"WHERE uid='{uid}'")

    # Not found AI Task
    if is_list_empty(task):
        raise RuntimeError("Could not find AI Task ({})".format(uid))
    
    # Parse Information
    task_info   = parse_task_data(task[0])
    source_uid  = task_info['source_uid']
    model_uid   = task_info['model_uid']

    # Check Status: Wait Task if there is someone launching task
    timeout = 0
    while(True):

        status = get_task_status(uid)
        
        if status == 'running':
            mesg = 'AI Task is running'
            log.warn(mesg)
            return mesg    
        elif status in [ "stop", "error" ]:
            update_task_status(uid, 'loading')
            break
        else:
            time.sleep(1)
            timeout += 1
            if timeout >= 20:
                raise RuntimeError('Waitting AI Task Timeout')

    # Prepare AI Model Config and Load Model
    try:
        model_data = select_data( table='model', data="*", 
                                condition=f"WHERE uid='{model_uid}'")[0]
        model_info = parse_model_data(model_data)
        
        model_info['meta_data'].update( 
            {**task_info['model_setting'], 'device': task_info['device'] } )
        
        # Load AI Model
        load_model = get_ivit_api(SERV_CONF["FRAMEWORK"])
        model = load_model( model_info['type'], model_info['meta_data'])
    
    except Exception as e:
        raise RuntimeError("Load AI Model Failed: {}".format(simple_exception(e)[1]))
        
    # Create Source Object
    try:
        src = create_source(source_uid=source_uid)
        height, width = src.get_shape()

    except Exception as e:
        update_src_status(source_uid, 'error')
        raise RuntimeError("Load Source Failed: {}".format(simple_exception(e)[1]))

    # Create Displayer Object
    try:
        cv_flag = getattr(data, 'cv_display', None) 
        dpr = create_displayer(
            cv=cv_flag,
            rtsp=True, 
            height=height, 
            width=width, 
            fps=src.get_fps(),
            name=uid, 
            platform='intel')
            
    except Exception as e:
        raise RuntimeError("Load Displayer Failed: {}".format(simple_exception(e)[1]))

    # Load Application
    try:
        app = create_app(app_uid=uid, label_path=model_info['label_path'])
    except Exception as e:
        raise RuntimeError("Load Application Failed: {}".format(simple_exception(e)[1]))
    
    # Create Threading
    try:
        # Check Keyword in RT_CONF
        if RT_CONF.get(uid) is None:
            RT_CONF.update( {uid: {'EXEC': None}} )
        
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
        start_source(source_uid)
        RT_CONF[uid]['EXEC'].start()

    except Exception as e:
        del RT_CONF[uid]['EXEC']
        raise RuntimeError("Launch AI Task Failed: {}".format(simple_exception(e)[1]))

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
    del RT_CONF[uid]['EXEC']
    
    # Stop Source if not using
    stop_source(source_uid)

    update_task_status(uid, 'stop')
    log.info(mesg)
    return mesg


def update_ai_task(uid:str, data:dict=None):
    """ update AI Task via uid """
    
    RT_CONF[uid]['DATA'] = data
    log.info('Update AI Task: {} With {}'.format(uid, data))    
    return 'Update success'


def add_ai_task(add_data):
    """ Add a AI Task 
    ---
    1. Add task information into database ( table: task )
    2. Add application information into database ( table: app )
    """
    
    # Generate Task UUID and Application UUID    
    errors = {}
    task_uid = app_uid = gen_uid()

    try:
        # Check Database Data
        con, cur = connect_db()

        # Task Name
        results = db_to_list(cur.execute("""SELECT name FROM task WHERE name=\"{}\" """.format(add_data.task_name)))
        if not is_list_empty(results):
            errors.update( {"task_name": "Duplicate task name: {}".format(add_data.task_name)} )

        # Source UID
        results = db_to_list(cur.execute("""SELECT uid FROM source WHERE uid=\"{}\" """.format(add_data.source_uid)))
        if is_list_empty(results):
            errors.update( {"source_uid": "Unkwon Source UID: {}".format(add_data.source_uid)} )

        # Model UID
        results = db_to_list(cur.execute("""SELECT uid FROM model WHERE uid=\"{}\" """.format(add_data.model_uid)))
        if is_list_empty(results):
            errors.update( {"model_uid": "Unkwon model UID: {}".format(add_data.model_uid)} )
    except Exception as e:
        log.exception(e)
    finally:
        # Close DB
        close_db(con, cur)

    # Model Setting
    threshold = add_data.model_setting['confidence_threshold']
    if threshold < 0.1 or threshold > 1.0:
        errors.update( {"confidence_threshold": "Invalid confidence threshold, should in range 0 to 1"})
    
    if len(errors.keys())>0:
        return {
        "uid": task_uid,
        "status": "failed",
        "data": errors
    }

    # Add Task Information into Database
    insert_data(
        table="task",
        data={
            "uid": task_uid,
            "name": add_data.task_name,
            "source_uid": add_data.source_uid,
            "model_uid": add_data.model_uid,
            "model_setting": add_data.model_setting,
            "status": "stop",
            "device": add_data.device
        }
    )
    
    # Add App Information into Database
    # app_type = select_data(table='app', data=['type'], condition=f"WHERE name='{add_data.app_name}'")[0][0]
    app_type = RT_CONF["IAPP"].get_app(add_data.app_name).get_type()
    insert_data(
        table="app",
        data={
            "uid": app_uid,
            "name": add_data.app_name,
            "type": app_type,
            "app_setting": add_data.app_setting
        }
    )
    
    return {
        "uid": task_uid,
        "status": "success",
        "data": errors
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


class AsyncInference:

    def __init__(   self, 
                    imodel:iModel,
                    pool_maximum:int=2,
                    exec_freq:float=0.033) -> None:
        """Asynchorize Inference Object

        Args:
            imodel (iModel): the iModel object for inference
            pool_maximum (int, optional): the maximum number of the thread. Defaults to 2.
            exec_freq (float, optional): the freqency of the inference. Defaults to 0.033.
        """
        self.imodel = imodel
        self.results = []

        self.pool_maximum = pool_maximum
        self.pool = ThreadPool(self.pool_maximum)
        
        self.exec_pools = []
        self.exec_time = time.time()
        self.exec_freq = exec_freq

    def infer_callback(self, result):
        """Callback function for threading pool

        Args:
            result (_type_): the result of the model inference.
        
        Workflow:

            1. Update `self.results`.
            2. Pop out the first one in `self.exec_pools`.
            3. Update timestamp `exec_freq`.
        """
        self.results = result
        self.exec_pools.pop(0)
        self.exec_time = time.time()

    def submit_data(self, frame:np.ndarray):
        """Create a threading for inference

        Args:
            frame (np.ndarray): the input image
        """
        
        full_pool = (len(self.exec_pools) > self.pool_maximum)
        too_fast = (time.time()-self.exec_time <= self.exec_freq)
        if full_pool or too_fast: return

        self.exec_pools.append(
            self.pool.apply_async(self.imodel.inference, (frame, ), callback=self.infer_callback)
        )
    
    def get_results(self) -> list:
        """Get results

        Returns:
            list: the results of ai inference
        """
        return self.results


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
        self.event = None

        self.metric = Metric()
        self.latency_limitor = Metric()

        self.minimum_latency = 1/self.src.get_fps()

        self.icap_alive = 'ICAP' in SERV_CONF and not (SERV_CONF['ICAP'] is None)
        
        self.async_infer = AsyncInference( imodel=self.model, pool_maximum=1)
        
        log.warning('Create a InferenceLoop')

    def create_thread(self) -> threading.Thread:
        return threading.Thread(target=self._inference_thread, daemon=True)

    def _app_setup_event(self, data):
        # Debug: with raise key
        if getattr(data, 'raise', False):
            raise RuntimeError('Raise Testing when AI Task Running')

        try:
            print(data)
            # Area Event: Color, ... etc
            palette = getattr(data, 'palette', None)
            if palette:
                for key, val in palette.items():
                    try:
                        self.app.palette[key] = val
                        log.warning('Update Color Successed ( {} -> {} )'.format(key, val))
                    except Exception as e: 
                        log.warning('Update Color Failed ... {}'.format(handle_exception(e)))

            # Draw Something
            app_setup = { key:getattr(data, key) \
                for key in [ "draw_bbox", "draw_result", "draw_area", "draw_tracking", "draw_line" ] \
                    if getattr(data, key, None) is not None
            }
            if app_setup:
                self.app.set_draw(app_setup)

            # AI Model Threshold
            thres = getattr(data, 'thres', None)
            if thres:
                self.model.set_thres(thres)
                log.warning('Set threshold')

        except Exception as e:
            log.error('Setting Parameters ... Failed')
            log.exception(e)
        
        finally:
            # Clear DATA
            RT_CONF[self.uid]['DATA'] = {}
            log.warning('App Setup Finished')

    def _dynamic_change_app_setup_event(self):
        """ Dynamic Modify Varaible of the Application """

        # Return: No dynamic variable setting
        data = RT_CONF[self.uid]['DATA']
        if data=={}: return
        t = threading.Thread(target=self._app_setup_event, args=(data, ), daemon=True).start()

    def _inference_thread(self):
        
        prev_data, cur_data = None, None

        try:
            log.warning('Start AI Task Inference Stream')
            self.prev_time = time.time()
            update_task_status(self.uid, 'running')

            # Make sure source is ready
            while( self.is_ready and self.src.is_ready):
                
                # Limit Speed to fit FPS
                if self.minimum_latency < self.latency_limitor.update():
                    time.sleep(1e-3); continue
                
                # Ready to calculate performance
                self.metric.update()

                # Setting Dynamic Variable
                self._dynamic_change_app_setup_event()
 
                # Get data        
                ret, frame = self.src.read()

                # # Do Sync Inference
                # cur_data = self.model.inference(frame)

                # Do Async Inference
                self.async_infer.submit_data(frame=frame)

                cur_data = self.async_infer.get_results()

                self.draw, self.results, self.event = self.app(frame, cur_data, draw=True)

                # Display
                self.dpr.show(self.draw)

                # Log
                # log.debug(cur_data)

                # Send Data
                self.results.update({
                    "fps": self.metric.get_fps(),
                    "live_time": self.metric.get_exec_time()
                })
                WS_CONF.update({ self.uid: self.results })

                # Calculate FPS and Update spped_limitor
                self.metric.update()
                self.latency_limitor.update()

                time.sleep(1e-5)

            # Check is error from source or not
            if (not self.src.is_ready) and len(self.src.errors) > 0:
                raise self.src.errors[-1]

            # Update Task Status to Stop
            update_task_status(self.uid, 'stop')

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

