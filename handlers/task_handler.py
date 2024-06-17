# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import asyncio
import copy
import glob
import json
import logging as log
import os
import shutil
import threading
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import Union

import numpy as np
from fastapi import File

# Custom
try:
    from ..common import EVENT_CONF, RT_CONF, SERV_CONF, WS_CONF, manager
    from ..utils import encode, file_helper, gen_uid, json_to_str
except:
    from common import EVENT_CONF, RT_CONF, SERV_CONF, WS_CONF, manager
    from utils import encode, file_helper, gen_uid, json_to_str

from . import (
    db_handler,
    event_handler,
    icap_handler,
    mesg_handler,
    model_handler,
    task_handler,
)
from .ai_handler import get_ivit_api, iModel
from .app_handler import create_app
from .db_handler import (
    close_db,
    connect_db,
    db_to_list,
    delete_data,
    insert_data,
    is_db_empty,
    is_list_empty,
    parse_app_data,
    parse_model_data,
    parse_source_data,
    parse_task_data,
    select_column_by_uid,
    select_data,
    update_data,
)
from .err_handler import InvalidUidError
from .io_handler import (
    create_displayer,
    create_source,
    start_source,
    stop_source,
    update_src_status,
)
from .ivit_handler import Metric
from .mesg_handler import handle_exception, json_exception, simple_exception, ws_msg

# --------------------------------------------------------


def get_task_info(uid: str = None):
    """Get AI Task Information from Database
    1. Get All AI Task Information
    2. Parse Each AI Task Information

    Return

        - Sample

            ```
            task_uid: task uid
            task_name: task name
            source_uid: source uid
            source_name: source name
            model_uid: model uid
            model_name: model name
            model_type: model type
            app_uid: app uid
            app_name: app name
            error: error message
            ```
    """

    # Basic Params
    return_data, return_errors = [], []

    # Open DB
    con, cur = connect_db()

    # Move DB Cursor
    sql_command = (
        """SELECT * FROM task"""
        if uid == None
        else """SELECT * FROM task WHERE uid=\"{}\"""".format(uid)
    )

    # Sort
    sql_command = sql_command + " ORDER BY created_time ASC, name ASC"

    # Get data
    results = db_to_list(cur.execute(sql_command))

    # Check BD is any task exist
    if is_db_empty(SERV_CONF["DB_PATH"]):
        return "No task setup"

    # Check DB Data
    if is_list_empty(results):
        raise InvalidUidError("Got invalid task uid: {}".format(uid))

    # Get Data
    print([(task[1], task[8]) for task in results])
    for info in map(parse_task_data, results):
        task_uid = app_uid = info["uid"]
        task_name = info["name"]
        source_uid = info["source_uid"]
        model_uid = info["model_uid"]
        error = info["error"]

        # Source
        results = select_column_by_uid(cur, "source", source_uid, ["name"])

        if is_list_empty(results):
            source_name = None
            return_errors.append(f"Got invalid source uid: {source_uid}")
        else:
            source_name = results[0][0]

        # Model
        results = select_column_by_uid(cur, "model", model_uid, ["name", "type"])
        if is_list_empty(results):
            model_name, model_type = None, None
            return_errors.append(f"Got invalid source uid: {model_uid}")
        else:
            model_name, model_type = results[0]

        # App
        results = select_column_by_uid(cur, "app", app_uid, ["name"])
        if is_list_empty(results):
            return_errors.append(f"Got invalid application uid: {app_uid}")
            app_name = None
        else:
            app_name = results[0][0]

        # Add More Data
        info.update(
            {
                "task_uid": task_uid,
                "task_name": task_name,
                "source_name": source_name,
                "model_name": model_name,
                "model_type": model_type,
                "app_uid": app_uid,
                "app_name": app_name,
                "error": return_errors if return_errors != [] else error,
            }
        )

        # Pop unused data
        for key in ["uid", "name"]:
            info.pop(key, None)

        return_data.append(info)

    # Cloase DB
    close_db(con, cur)

    return return_data


def get_task_status(uid):
    """Get Status of AI Task"""
    return select_data(table="task", data=["status"], condition=f"WHERE uid='{uid}'")[
        0
    ][0]


def update_task_status(uid, status, err_mesg: dict = {}):
    """Update Status of AI Task"""
    write_data = {"status": status, "error": err_mesg}
    update_data(table="task", data=write_data, condition=f"WHERE uid='{uid}'")


def verify_task_exist(uid) -> dict:
    # Task Information
    task = select_data(table="task", data="*", condition=f"WHERE uid='{uid}'")

    # Not found AI Task
    if task == []:
        raise RuntimeError("Could not find AI Task ({})".format(uid))

    return task[0]


def verify_duplicate_task(task_name: str, duplicate_limit=2):
    """Check duplicate AI task.

    Args:
        task_name (str): the ai task name
        edit_mode (bool, optional): if is edit mode. Defaults to False.

    Raises:
        NameError: if ai task is already exist.
    """
    # Calculate duplicate nums
    duplicate_nums = 0

    # Checking
    for uid, name in select_data(table="task", data=["uid", "name"]):
        # if duplicate then add 1
        if task_name == name:
            duplicate_nums += 1

        # more than 2 task name then raise error
        if duplicate_nums >= duplicate_limit:
            raise NameError("AI Task is already exist ( {}: {} )".format(uid, name))


def has_duplicate_task(task_name: str, duplicate_limit=2) -> bool:
    """Check duplicate AI task.

    Args:
        task_name (str): the ai task name
        edit_mode (bool, optional): if is edit mode. Defaults to False.

    Raises:
        NameError: if ai task is already exist.
    """
    # Calculate duplicate nums
    duplicate_nums = 0

    # Checking
    for uid, name in select_data(table="task", data=["uid", "name"]):
        # if duplicate then add 1
        if task_name == name:
            duplicate_nums += 1

        # more than 2 task name then raise error
        if duplicate_nums >= duplicate_limit:
            return 1
    return 0


def verify_thres(threshold: float):
    if threshold < 0.01 and threshold > 1.0:
        raise ValueError("Threshold value should in range 0 to 1")


def check_single_task_for_hailo():
    # NOTE: Hailo only support one task running
    if SERV_CONF["PLATFORM"] != "hailo":
        return

    running_tasks = select_data(
        table="task", data=["uid"], condition="WHERE status='run'"
    )
    if len(running_tasks) > 0:
        raise RuntimeError("Not support multiple AI task !!!")


def check_event_params(app_setting: dict, app_name: str = ""):
    """Check event parameters and update event status in application setting ( which key is 'enable')."""

    # NOTE: check event first because the setting was in app_setting
    # Add Event Information
    for area_info in app_setting["application"]["areas"]:
        if "events" not in area_info:
            continue
        log.info(f"Get event setting in area ( {area_info['name']} )")
        for key in ["enable", "uid", "title", "logic_operator", "logic_value"]:
            if key == "enable" and key not in area_info["events"]:
                area_info["events"].update({"enable": True})
            log.debug("\t* {}: {}".format(key, area_info["events"].get(key)))

            if key == "logic_operator" and (
                app_name in ["Tracking_Zone", "Movement_Zone"]
            ):
                if area_info["events"][key] == "<":
                    raise KeyError('Tracking and Movement not support "<" operator. ')


# --------------------------------------------------------
# Execute AI Task


def run_ai_task(uid: str, data: dict = None) -> str:
    """Run AI Task

    Workflow
    1. Check task is exist
    2. Parse task information
    3. Check task status
        If run then return message;
        If stop or error then keep going;
        If loading then wait for N seconds ( default is 20 ).
    4. Get model information and load model
    5. Get source object and displayer object
    6. Load Application
    7. Create InferenceLoop and keep in RT_CONF[<uid>]['EXEC']
    8. Start source and Start InferenceLoop
    """

    # Checking single task
    check_single_task_for_hailo()

    # Get Task Information
    task = select_data(table="task", data="*", condition=f"WHERE uid='{uid}'")

    # Not found AI Task
    if is_list_empty(task):
        raise RuntimeError("Could not find AI Task ({})".format(uid))

    # Parse Information
    task_info = parse_task_data(task[0])
    source_uid = task_info["source_uid"]
    model_uid = task_info["model_uid"]

    # Check Status: Wait Task if there is someone launching task
    timeout = 0
    while True:
        status = get_task_status(uid)

        if status == "run":
            mesg = "AI Task is running"
            log.warn(mesg)
            return mesg
        elif status in ["stop", "error"]:
            update_task_status(uid, "loading")
            break
        else:
            time.sleep(1)
            timeout += 1
            if timeout >= 20:
                raise RuntimeError("Waitting AI Task Timeout")

    # Prepare AI Model Config and Load Model
    try:
        model_data = select_data(
            table="model", data="*", condition=f"WHERE uid='{model_uid}'"
        )[0]
        model_info = parse_model_data(model_data)
    except Exception:
        raise RuntimeError("Load AI Model Failed: Can not found AI Model")

    # Load Application
    try:
        app = create_app(app_uid=uid, label_path=model_info["label_path"])
    except Exception as e:
        raise RuntimeError("Load Application Failed: {}".format(simple_exception(e)[1]))

    # Load Model
    try:
        model_info["meta_data"].update(
            {**task_info["model_setting"], "device": task_info["device"]}
        )

        # Load AI Model
        load_model = get_ivit_api(SERV_CONF["FRAMEWORK"])
        model = load_model(model_info["type"], model_info["meta_data"])

    except Exception as e:
        raise RuntimeError("Load AI Model Failed: {}".format(simple_exception(e)[1]))

    # Create Source Object
    try:
        src = create_source(source_uid=source_uid)
        height, width = src.get_shape()

    except Exception as e:
        update_src_status(source_uid, "error")
        raise RuntimeError("Load Source Failed: {}".format(simple_exception(e)[1]))

    # Create Displayer Object
    try:
        cv_flag = getattr(data, "cv_display", None)
        dpr = create_displayer(
            cv=cv_flag,
            rtsp=True,
            height=height,
            width=width,
            fps=30,
            name=uid,
            platform=SERV_CONF["platform"],
        )

        # dpr = create_rtsp_displayer(
        #     name = uid,
        #     width = width,
        #     height = height,
        #     fps = 30,
        #     server = 'rtsp://127.0.0.1:8554',
        #     platform = 'intel',
        # )

    except Exception as e:
        raise RuntimeError("Load Displayer Failed: {}".format(simple_exception(e)[1]))

    # Create Threading
    try:
        # Check Keyword in RT_CONF
        if RT_CONF.get(uid) is None:
            RT_CONF.update({uid: {"EXEC": None}})

        RT_CONF[uid].update(
            {
                "EXEC": InferenceLoop(
                    uid=uid,
                    src=src,
                    model=model,
                    app=app,
                    src_uid=source_uid,
                    model_uid=model_uid,
                    dpr=dpr,
                ),
                "DATA": {},
            }
        )

        # Start each threading
        start_source(source_uid)
        RT_CONF[uid]["EXEC"].start()

    except Exception as e:
        del RT_CONF[uid]["EXEC"]
        raise RuntimeError("Launch AI Task Failed: {}".format(simple_exception(e)[1]))

    # Add MQTT Topic
    SERV_CONF["MQTT"].subscribe_event(uid)
    SERV_CONF["MQTT"].publish(
        SERV_CONF["MQTT"].get_event_topic(uid), "iVIT Registered !"
    )

    # End
    mesg = "Run AI Task ( {}: {} )".format(uid, task_info["name"])
    return mesg


def stop_ai_task(uid: str, data: dict = None):
    """Stop AI Task via uid"""

    # Task Information
    task = verify_task_exist(uid=uid)
    task_info = parse_task_data(task)

    task_uid = app_uid = task_info["uid"]
    source_uid = task_info["source_uid"]

    if get_task_status(uid=uid) != "run":
        mesg = "AI Task is stopped !"
        return mesg

    mesg = "Stop AI Task ( {}: {} )".format(uid, task_info["name"])

    # Clear event
    try:
        events = select_data(
            table="event", data="uid", condition=f"WHERE app_uid='{app_uid}'"
        )
        [event_handler.del_event(event[0]) for event in events]
    except Exception:
        log.warning("Delete event fail when stopping AI task.")

    # Stop AI Task
    RT_CONF[uid]["EXEC"].stop()
    del RT_CONF[uid]["EXEC"]

    # Stop Source if not using
    stop_source(source_uid)

    update_task_status(uid, "stop")

    log.info(mesg)
    return mesg


def update_ai_task(uid: str, data: dict = None):
    """update AI Task via uid"""
    if uid not in RT_CONF:
        RT_CONF.update({uid.upper(): {}})
    if uid.upper() not in RT_CONF:
        raise RuntimeError(
            f"The AI task has not launch yet. ({uid}, {','.join(RT_CONF.keys())})"
        )

    RT_CONF[uid]["DATA"] = data
    log.info("Update AI Task: {} With {}".format(uid, data))
    return "Update success"


def add_ai_task(add_data):
    """Add a AI Task
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
        results = db_to_list(
            cur.execute(
                """SELECT uid, name FROM task WHERE name=\"{}\" """.format(
                    add_data.task_name
                )
            )
        )
        if not is_list_empty(results):
            uid, name = results[0]
            errors.update(
                {"task_name": "Duplicate task name: {} ({})".format(name, uid)}
            )

        # Source UID
        results = db_to_list(
            cur.execute(
                """SELECT uid FROM source WHERE uid=\"{}\" """.format(
                    add_data.source_uid
                )
            )
        )
        if is_list_empty(results):
            errors.update(
                {"source_uid": "Unkwon Source UID: {}".format(add_data.source_uid)}
            )

        # Model UID
        results = db_to_list(
            cur.execute(
                """SELECT uid FROM model WHERE uid=\"{}\" """.format(add_data.model_uid)
            )
        )
        if is_list_empty(results):
            errors.update(
                {"model_uid": "Unkwon model UID: {}".format(add_data.model_uid)}
            )

        # Check Event
        try:
            check_event_params(add_data.app_setting, add_data.app_name)
        except Exception as e:
            errors.update({"events": "Event setting error ({})".format(e.message)})

    except Exception as e:
        log.exception(e)

    finally:
        # Close DB
        close_db(con, cur)

    # Model Setting
    threshold = add_data.model_setting["confidence_threshold"]
    if threshold < 0.01 or threshold > 1.0:
        errors.update(
            {
                "confidence_threshold": "Invalid confidence threshold, should in range 0 to 1"
            }
        )

    # Find Error and return errors
    if len(errors.keys()) > 0:
        return {"uid": task_uid, "status": "failed", "data": errors}

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
            "device": add_data.device,
            "created_time": time.time(),
        },
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
            "app_setting": add_data.app_setting,
        },
    )

    return {"uid": task_uid, "status": "success", "data": errors}


def edit_ai_task(edit_data):
    """Edit AI Task
    ---
    1. Check task_uid and task_name is exist or not ( ignore itself )
    2. Check threshold is available
    3. Check source_uid, model_uid is exist or not
    4. Is there has any errors then return response with status='failed'
    5. insert data into database
        * task
        * app
    """
    # Get Task UUID and Application UUID
    task_uid = app_uid = edit_data.task_uid

    # CHECK: AI task is exist or not
    task_raw_data = verify_task_exist(task_uid)
    task_info = parse_task_data(task_raw_data)

    # Check status
    if task_info["status"] == "run":
        raise RuntimeError("The AI task is running, can not be edit.")

    # Name is exists
    verify_duplicate_task(edit_data.task_name)

    # CHECK: threshold value is available
    verify_thres(threshold=edit_data.model_setting["confidence_threshold"])

    # CHECK Others
    errors = {}
    try:
        # Check Database Data
        con, cur = connect_db()
        # Source UID
        results = db_to_list(
            cur.execute(
                """SELECT uid FROM source WHERE uid=\"{}\" """.format(
                    edit_data.source_uid
                )
            )
        )
        if is_list_empty(results):
            errors.update(
                {"source_uid": "Unkwon Source UID: {}".format(edit_data.source_uid)}
            )
        # Model UID
        results = db_to_list(
            cur.execute(
                """SELECT uid FROM model WHERE uid=\"{}\" """.format(
                    edit_data.model_uid
                )
            )
        )
        if is_list_empty(results):
            errors.update(
                {"model_uid": "Unkwon model UID: {}".format(edit_data.model_uid)}
            )

        # Check Event
        try:
            check_event_params(edit_data.app_setting, edit_data.app_name)
        except Exception as e:
            errors.update({"events": "Event setting error ({})".format(e)})

    except Exception as e:
        log.exception(e)
        errors.update({"unknown": "Unknown error ({})".format(e)})

    finally:
        # Close DB
        close_db(con, cur)

    # Find Error and return errors
    print(errors)
    if len(errors.keys()) > 0:
        return {"uid": task_uid, "status": "failed", "data": errors}

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
            "device": edit_data.device,
            "created_time": task_info["created_time"],
        },
        replace=True,
    )

    # check_event_params(edit_data.app_setting, edit_data.app_name)

    # Add App Information into Database
    app_type = RT_CONF["IAPP"].get_app(edit_data.app_name).get_type()

    insert_data(
        table="app",
        data={
            "uid": app_uid,
            "name": edit_data.app_name,
            "type": app_type,
            "app_setting": json_to_str(edit_data.app_setting),
        },
        replace=True,
    )

    return {"uid": task_uid, "status": "success"}


def del_ai_task(uid: str):
    """Delete a AI Task
    1. Delete task information into database ( table: task )
    2. Delete application information into database ( table: app )
    """

    verify_task_exist(uid)

    # Del Task
    delete_data(table="task", condition=f"WHERE uid='{uid}'")

    # Del App
    delete_data(table="app", condition=f"WHERE uid='{uid}'")


def export_ai_task(export_uids: list, to_icap: bool = False) -> list:
    exporter = TaskExporter(export_uids[0])
    exporter.start()

    # Waiting the parse status
    while exporter.status != exporter.S_PARS:
        if exporter.status == exporter.S_FAIL:
            raise exporter.error
        print(f"{exporter.status:20}", end="\r")

    return {"uid": exporter.uid, "status": exporter.status}


def import_ai_task(file: File, url: str = None):
    importer = TASK_ZIP_IMPORTER(file=file)
    importer.start()

    while importer.status != importer.S_PARS:
        if importer.status == importer.S_FAIL:
            raise importer.error

        print(f"{importer.status:20}", end="\r")

    return {"uid": importer.uid, "status": importer.status}


class EasyTimer:
    def __init__(self, limit: float) -> None:
        self._prev_time = time.time()
        self.limit = limit
        self._is_stop = False

        self._duration = time.time()

    def stop(self):
        self._is_stop = True
        # log.debug(f'Stop {self.__class__.__name__}')

    def start(self):
        self._prev_time = time.time()
        self._is_stop = False
        # log.debug(f'Start {self.__class__.__name__}')

    def update(self):
        self._duration = time.time() - self._prev_time
        return self._duration

    @property
    def is_timeup(self) -> bool:
        return (time.time() - self._prev_time) > self.limit

    @property
    def is_stop(self) -> bool:
        return self._is_stop


class FakeDisplayer:
    log.warning("Initialize Fake Displayer")

    def show(frame):
        return frame

    def release():
        return log.warning("Release Fake Displayer")


class AsyncInference:
    def __init__(
        self,
        imodel: iModel,
        workers: int = 1,
        freqency: float = 0.066,
        clean_duration: int = 1,
    ) -> None:
        """Asynchorize Inference Object

        Args:
            imodel (iModel): the iModel object for inference
            workers (int, optional): the maximum number of the thread. Defaults to 2.
            freqency (float, optional): the freqency of the inference. Defaults to 0.033.
            clean_duration (int, optional): the duration of cleanning results. Defaults to 5.
        """
        self.imodel = imodel
        self.results = []

        self.workers = workers
        self.pool = ThreadPool(self.workers)

        self.pools = []
        self.exec_time = time.time()
        self.freqency = freqency

        # Set Timer for clean result
        self.start_clean_timer = False
        self.clean_timer = EasyTimer(limit=clean_duration)

        self.lock = threading.RLock()

        # Metrics
        self.async_infer_fps = -1

    def _is_too_fast(self):
        """too fast"""
        return (time.time() - self.exec_time) <= self.freqency

    def _is_full_pool(self):
        """full pool"""
        return len(self.pools) > self.workers

    def _check_time_to_clean_results(self):
        # If no need to clean
        if not self.start_clean_timer:
            return

        # If need clean, then keep the start time
        if self.clean_timer.is_stop:
            self.clean_timer.start()

        # Calculate Time
        if not self.clean_timer.is_timeup:
            return

        self.results = []
        self.clean_timer.stop()

    def async_infer_wrapper(self, frame):
        prev_time = time.time()
        result = self.imodel.inference(frame)
        self.async_infer_fps = 1 // (time.time() - prev_time)

        return result

    def infer_callback(self, result):
        """Callback function for threading pool

        Args:
            result (_type_): the result of the model inference.

        Workflow:

            1. Update `self.results`.
            2. Pop out the first one in `self.workers`.
            3. Update timestamp `freqency`.
        """

        # Has results then update self.results
        if len(result) != 0:
            self.lock.acquire()
            self.results = result
            self.lock.release()
            self.start_clean_timer = False

        else:
            self.start_clean_timer = True

        # Check timer every time
        self._check_time_to_clean_results()

        # Keep update exec_time
        self.exec_time = time.time()

        # Pop out first exec
        self.pools.pop(0)

    def submit_data(self, frame: np.ndarray):
        """Create a threading for inference

        Args:
            frame (np.ndarray): the input image
        """

        if self._is_full_pool() or self._is_too_fast():
            return

        self.pools.append(
            self.pool.apply_async(
                func=self.async_infer_wrapper,
                args=(frame,),
                callback=self.infer_callback,
            )
        )

    def get_results(self) -> list:
        """Get results

        Returns:
            list: the results of ai inference
        """
        return self.results

    def get_fps(self):
        return self.async_infer_fps


class InferenceLoop:
    """Inference Thread Helper"""

    def __init__(self, uid, src, model, app, src_uid, model_uid, dpr=None) -> None:
        # Basic Parameter
        self.uid = uid
        self.src = src
        self.src_uid = src_uid
        self.model = model
        self.app = app
        self.dpr = dpr if dpr else FakeDisplayer()

        # Thread Parameter
        self.is_ready = True
        self.thread_object = None

        # Draw Parameter
        self.draw = None
        self.results = None
        self.event = None
        self.event_output = None

        # Metric
        self.stream_metric = Metric()
        self.latency_limitor = Metric()

        # RTSP Output
        self.display_latency = 1 / 30

        # For iCAP
        self.icap_alive = "ICAP" in SERV_CONF and SERV_CONF["ICAP"] is not None

        # Create AsyncInference Object
        self.async_infer = AsyncInference(
            imodel=self.model, workers=1, freqency=self.display_latency * 2
        )
        log.warning("Create a InferenceLoop")

        # FPS and Running Time
        self.fps = -1
        self.running_time = 0

    def create_thread(self) -> threading.Thread:
        return threading.Thread(target=self._infer_thread, daemon=True)

    def _update_app_setup_func(self, data):
        """Update the parameters of the application, like: palette, draw_bbox, etc."""
        try:
            # Display
            need_cv = getattr(data, "cv_display", False)
            if need_cv:
                self.dpr._setup_cv(
                    name=self.uid, width=self.src.width, height=self.src.height
                )
            else:
                self.dpr.dps.pop(self.dpr.CV)

            # Area Event: Color, ... etc
            palette = getattr(data, "palette", None)
            if palette:
                for key, val in palette.items():
                    try:
                        key = key.strip()
                        self.app.palette[key] = val
                        log.warning(
                            "Update Color Successed ( {} -> {} )".format(key, val)
                        )
                    except Exception as e:
                        log.warning(
                            "Update Color Failed ... {}".format(handle_exception(e))
                        )

            # Draw Something
            app_setup = {
                key: getattr(data, key)
                for key in [
                    "draw_bbox",
                    "draw_result",
                    "draw_area",
                    "draw_tracking",
                    "draw_line",
                ]
                if getattr(data, key, None) is not None
            }
            if app_setup:
                self.app.set_draw(app_setup)

            # AI Model Threshold
            thres = getattr(data, "thres", None)
            if thres:
                self.model.set_thres(thres)
                log.warning("Set threshold")

        except Exception as e:
            log.error("Setting Parameters ... Failed")
            log.exception(e)

        finally:
            log.warning("App Setup Finished")

    def _dynamic_change_app_setup(self):
        """Dynamic Modify Varaible of the Application"""

        # Return: No dynamic variable setting
        if RT_CONF[self.uid]["DATA"] == {}:
            return

        # Copy a data
        data = copy.deepcopy(RT_CONF[self.uid]["DATA"])

        # Create a thread to update task value
        threading.Thread(
            target=self._update_app_setup_func, args=(data,), daemon=True
        ).start()

        # Clear dara
        RT_CONF[self.uid]["DATA"] = {}

    def _launch_event(self, event_output: dict) -> None:
        """Launch event function"""

        if not event_output:
            return

        # Get value
        event_output = event_output.get("event")
        if not event_output:
            return

        # NOTE: store in database, event start if status is True else False
        for event in event_output:
            # Combine data
            data = {
                "uid": event["uid"],
                "title": event["title"],
                "app_uid": self.uid,
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "annotation": event["meta"],
            }

            # Event Trigger First Time: status=True and event_output!=[]
            if event["event_status"]:
                # Add new data
                insert_data(table="event", data=data)

            else:
                # Update old data ( update end_time )
                update_data(
                    table="event",
                    data=data,
                    condition=f"WHERE start_time={event['start_time']}",
                )

                # No need to send websocket
                continue

            # Tidy up data
            data["task_uid"] = self.uid
            data["annotation"].pop("detections")
            data["start_time"] = str(data["start_time"])
            data["end_time"] = str(data["end_time"])

            # Send data to front end
            event_message = ws_msg(type="EVENT", content=data)
            EVENT_CONF.update(data)

            # Send to front end via WebSocket
            try:
                asyncio.run(manager.send(self.uid, event_message))
            except Exception:
                log.warning("Send websocket failed!!!!")

            if "MQTT" in SERV_CONF:
                main_topic = SERV_CONF["MQTT"].get_event_topic()
                trg_topic = SERV_CONF["MQTT"].get_event_topic(data["task_uid"])
                for topic in [main_topic, trg_topic]:
                    SERV_CONF["MQTT"].publish(topic, event_message)
                    time.sleep(0.1)

    # NOTE: MAIN LOOP
    def _infer_loop(self):
        log.warning("Start AI Task Inference Stream")
        update_task_status(self.uid, "run")

        # Make sure source is ready
        while self.is_ready and self.src.is_ready:
            # Ready to calculate performance
            self.stream_metric.update()
            self.latency_limitor.update()

            # Setting Dynamic Variable ( thread )
            self._dynamic_change_app_setup()

            # Get data
            ret, frame = self.src.read()

            # Async Inference: Submit Data and Get Result
            self.async_infer.submit_data(frame=frame)
            cur_data = self.async_infer.get_results()

            # Run Application
            try:
                _draw, _results, _event = self.app(copy.deepcopy(frame), cur_data)

                # Not replace directly to avoid variable is replaced when interrupted
                self.draw, self.results, self.event_output = _draw, _results, _event
                # Trigger Event
                self._launch_event(_event)

            except Exception as e:
                log.warning("Run Application Error")
                log.exception(e)

            try:
                # Display
                self.dpr.show(self.draw)
            except Exception as e:
                log.exception(e)

            # Log
            # log.debug(cur_data)

            # Make sure Inference FPS is correct
            self.fps = self.async_infer.get_fps()

            # Send Data
            self.results.update(
                {
                    "fps": self.fps,
                    "stream_fps": self.stream_metric.get_fps(),
                    "live_time": self.stream_metric.get_exec_time(),
                }
            )
            WS_CONF.update({self.uid: self.results})

            # Limit FPS
            cur_latency = self.latency_limitor.update()
            t_delay = self.display_latency - cur_latency
            if 1 > t_delay > 0:
                # Sleep for correct FPS
                time.sleep(t_delay)

            # Calculate FPS and Update spped_limitor
            self.stream_metric.update()

        # Update Task Status to Stop
        update_task_status(self.uid, "stop")

    def _capture_src_error(self):
        # Means no error and some tasks is still running
        if self.src.is_ready and len(self.src.errors) == 0:
            return

        # Check is error from source or not
        if (not self.src.is_ready) and len(self.src.errors) > 0:
            # Update Status
            update_src_status(self.src_uid, "error")

            # Lastest error
            error = self.src.errors[-1]

            # Stop Source and update status
            self.src.release()

            # Add Source uid
            ret_mesg = ws_msg(type="ERROR", content=error.message)
            ret_mesg["data"] = {"source_uid": self.src_uid}
            asyncio.run(manager.send(uid=self.uid, message=ret_mesg))

    def _infer_thread(self):
        try:
            self._infer_loop()
            self._capture_src_error()

        # If Get Exception
        except Exception as e:
            # Write Log
            log.error("InferenceLoop Error!!!")
            log.exception(e)

            # Send and Store Error Message with Json Format
            json_exp = json_exception(e)
            asyncio.run(manager.send(self.uid, ws_msg(type="ERROR", content=e)))
            update_task_status(uid=self.uid, status="error", err_mesg=json_exp)

        finally:
            self.model.release()
            self.dpr.release()

            if self.icap_alive:
                SERV_CONF["ICAP"].send_attr(
                    data={"ivitTask": icap_handler.get_icap_task_info()}
                )

        log.warning("InferenceLoop ({}) is Stop".format(self.uid))

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
            try:
                self.thread_object.start()
            except Exception:
                log.warning("Initialize thread again !!!")
                self.thread_object = self.create_thread()
                self.thread_object.start()

    def release(self):
        self.stop()
        del self.thread_object
        self.thread_object = None


# Task Messager, Processors


class TaskProcessor(mesg_handler.TaskMessenger):
    S_FAIL = "Failure"
    S_INIT = "Waiting (0%)"
    S_FINISH = "Success (100%)"

    def __init__(self, uid: Union[str, None] = None):
        self.status = self.S_INIT
        self.uid = uid if uid else gen_uid()
        self.thread = self._create_thread()
        self.message = ""
        self.error = Exception()
        self.topic = "TASK"
        self.action = ""

    def update_status(
        self, status: str, message: Union[str, Exception] = "", push_mesg: bool = True
    ):
        """Update Status and push message to front end"""

        self.status = status

        if status == self.S_FAIL:
            if isinstance(message, Exception):
                self.error = message
            self.message = handle_exception(message)
        else:
            self.message = message

        # Add PROC
        if "PROC" not in SERV_CONF:
            print("Add PROC in SERV_CONF")
            SERV_CONF.update({"PROC": defaultdict(dict)})

        # Update Message
        if self.uid not in SERV_CONF["PROC"]:
            SERV_CONF["PROC"].update({self.uid: dict()})

        SERV_CONF["PROC"][self.uid].update(
            {"status": self.status, "message": self.message}
        )

        # Update Topic and Action
        if self.topic and self.action:
            SERV_CONF["PROC"][self.uid].update(
                {"topic": self.topic, "action": self.action}
            )

        # Push data
        if push_mesg:
            self.push_mesg()

        # Clear error, finish or unused data
        if status in [self.S_FAIL, self.S_FINISH]:
            SERV_CONF["PROC"].pop(self.uid, None)

    def main_event(self):
        pass

    def fail_event(self):
        pass

    def thread_event(self):
        try:
            self.main_event()

        except Exception as e:
            self.fail_event()
            log.exception(e)
            self.update_status(status=self.S_FAIL, message=e)
            time.sleep(1)

        finally:
            log.info("End of event")

    def _create_thread(self) -> threading.Thread:
        """Create a thread which will run self.export_event at once"""
        return threading.Thread(target=self.thread_event, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.thread.join()


# Task Exporter


class TaskExporter(TaskProcessor):
    S_PARS = "Verifying"
    S_CONV = "Packaging"

    def __init__(self, uid: str):
        super().__init__(self._valid_empty_uid(uid))
        self.performance = defaultdict(float)
        self.action = "EXPORT"

    def _valid_empty_uid(self, uid: str):
        if uid == "":
            raise NameError("UID should not be empty.")
        return uid

    def _parse_event(self) -> dict:
        # Checking
        verify_task_exist(self.uid)

        # Capture information from database
        con, cur = connect_db()

        # Task Table
        task_table = parse_task_data(
            db_to_list(
                cur.execute("""SELECT * FROM task WHERE uid=\"{}\"""".format(self.uid))
            )[0]
        )

        # Parse Data
        task_name = task_table["name"].replace(" ", "-")
        task_uid = task_table["uid"]
        app_uid = task_uid
        source_uid = task_table["source_uid"]
        model_uid = task_table["model_uid"]

        # Reset Status
        task_table["status"] = "stop"

        # Model Table
        model_table = parse_model_data(
            db_to_list(
                cur.execute(
                    """SELECT * FROM model WHERE uid=\"{}\"""".format(model_uid)
                )
            )[0]
        )

        # Source Table
        source_table = parse_source_data(
            db_to_list(
                cur.execute(
                    """SELECT * FROM source WHERE uid=\"{}\"""".format(source_uid)
                )
            )[0]
        )

        # App Table
        app_table = parse_app_data(
            db_to_list(
                cur.execute("""SELECT * FROM app WHERE uid=\"{}\"""".format(app_uid))
            )[0]
        )

        close_db(con, cur)

        # Update Return Info
        db_data = {
            "task": task_table,
            "model": model_table,
            "source": source_table,
            "app": app_table,
        }

        return db_data

    def _get_export_dir(self):
        """Create Export Folder if need"""
        exp_dir = SERV_CONF["EXPORT_DIR"]
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        return exp_dir

    def _package_event(self, db_data: dict):
        # Parse
        model_table = db_data["model"]
        source_table = db_data["source"]

        # Define Variable
        platform = SERV_CONF["PLATFORM"]
        exp_dir = self._get_export_dir()
        task_name = db_data["task"]["name"]
        task_uid = db_data["task"]["uid"]
        zip_files = []

        date_time = datetime.fromtimestamp(time.time())
        str_date_time = date_time.strftime("%Y%m%d")

        def file_name_formatter(name, ext):
            return f"{platform}_{task_name}_{str_date_time}.{ext}"

        # --------------------------------------------------
        # Write a Configuration
        cfg_name = file_name_formatter(task_name, "json")
        cfg_path = os.path.join(exp_dir, cfg_name)
        with open(cfg_path, "w") as f:
            json.dump(db_data, f, indent=4)
        zip_files.append(cfg_path)

        # --------------------------------------------------
        # Append file to compress
        zip_name = file_name_formatter(task_name, "zip")
        zip_path = os.path.join(exp_dir, zip_name)

        # Append model data
        model_dir = os.path.dirname(model_table["model_path"])
        [
            zip_files.append(os.path.join(model_dir, model_file))
            for model_file in os.listdir(model_dir)
        ]

        # Append Source data
        if source_table["type"] in ["IMAGE", "VIDEO", "DIR"]:
            zip_files.append(source_table["input"])

        # Compress
        first_zip_flag = file_helper.compress_files(
            zip_path=zip_path, zip_files=zip_files
        )
        assert first_zip_flag, "Fail to zip with all files!"

        # Remove temperate file
        os.remove(cfg_path)

        # --------------------------------------------------

        # Get checksum
        md5sums = encode.md5(zip_path)
        checksum_name = "MD5SUMS"
        checksum_path = os.path.join(exp_dir, checksum_name)
        with open(checksum_path, "w") as f:
            f.write(md5sums)

        # Compress again
        zip_files = [zip_path, checksum_path]
        temp_zip_path = "temperate.zip"
        second_zip_flag = file_helper.compress_files(
            zip_path=temp_zip_path, zip_files=zip_files, keep_dir=False
        )
        assert second_zip_flag, "Fail to zip with checksum! "

        os.rename(temp_zip_path, zip_path)
        assert os.path.exists(zip_path), "Fail rename zip."

        # Delete temperate checksum
        os.remove(checksum_path)

        # Return
        package_data = {
            "task_uid": task_uid,
            "task_name": task_name,
            "cfg_name": cfg_name,
            "zip_name": zip_name,
        }

        SERV_CONF["PROC"][self.uid].update(package_data)

    def main_event(self):
        log.info("Waiting")
        self.update_status(self.S_INIT)

        t_parse = time.time()
        log.info("Packaging")
        self.update_status(self.S_PARS)
        db_data = self._parse_event()
        self.performance["parse"] = time.time() - t_parse

        t_package = time.time()
        log.info("Packaging")
        self.update_status(self.S_CONV)
        packaged_data = self._package_event(db_data)
        self.performance["package"] = time.time() - t_parse

        self.update_status(self.S_FINISH)
        log.info("Finished !!!")


# Task Importer


class TaskImporterWrapper(TaskProcessor):
    """Model Helper
    1. Download ZIP
    2. Extract ZIP
    3. Convert Model if need
    4. Initailize AI Task into Database
    """

    S_DOWN = "Downloading (25%)"
    S_PARS = "Verifying (50%)"
    S_IMPO = "Importing (75%)"

    def __init__(self) -> None:
        """
        uid:
        platfrom:
        status:
        benchmark:
        """
        super().__init__()
        self.action = "IMPORT"

        # Task information
        self.task_platform = None
        self.task_name = None
        self.export_time = None

        # File Parameters
        self.file_name = ""
        self.file_path = ""
        self.file_folder = ""
        self.cfg_path = ""

        # Benchmark
        self.performance = defaultdict(float)

        # Init
        self._update_uid_into_share_env()

        self.tmp_dir = SERV_CONF["TEMP_DIR"]
        self._create_tmp_dir(self.tmp_dir)

    def _create_tmp_dir(self, tmp_dir: str):
        if os.path.exists(tmp_dir):
            return
        os.mkdir(tmp_dir)

    def _update_uid_into_share_env(self):
        """Updae Environment Object"""
        if SERV_CONF.get("PROC") is None:
            SERV_CONF.update({"PROC": defaultdict()})
        if SERV_CONF["PROC"].get(self.uid) is None:
            SERV_CONF["PROC"].update(
                {self.uid: {"status": self.status, "name": self.file_name}}
            )

    def _download_event(self):
        pass

    def _parse_event(self):
        """Parsing Event

        - Structure
            <task_name>/
                data/
                model/
                export/     # db

        - Workflow
            1. Extract ZIP.
            2. Remove ZIP.
            3. Move the file to correct path
        """

        def move_data(org_path, trg_path):
            if os.path.exists(trg_path):
                if os.path.isdir(trg_path):
                    shutil.rmtree(trg_path)
                else:
                    os.remove(trg_path)
                log.warning(
                    "\t- Import file exists, auto remove it. ({})".format(trg_path)
                )

            shutil.move(org_path, trg_path)
            log.info("\t- Move file from {} to {}".format(file_path, trg_path))

        # Parse ZIP Name
        self.task_platform, self.task_name, self.export_time = self._parse_zip_name(
            self.file_name
        )

        # Concate data
        self.file_folder = os.path.join(SERV_CONF["TEMP_DIR"], self.task_name)

        log.debug("Parsing File:")
        log.debug(f"\t- Platform: {self.task_platform}")
        log.debug(f"\t- Name: {self.task_name}")
        log.debug(f"\t- Export Time: {self.export_time}")
        log.debug(f"\t- Save Path: {self.file_path}")

        # Extract
        file_helper.extract_files(zip_path=self.file_path, trg_folder=self.file_folder)
        log.debug(f"Extract to {self.file_folder}")

        # Verify
        files = os.listdir(self.file_folder)
        if len(files) != 2:
            raise FileExistsError(f"Expected 2 file, but got {len(files)}")
        else:
            print(files)

        for file_name in files:
            file_path = os.path.join(self.file_folder, file_name)
            if "MD5" in file_name:
                checksum = encode.load_txt(file_path)[0]

        real_zip_path = os.path.join(self.file_folder, self.file_name)

        if checksum != encode.md5(real_zip_path):
            raise RuntimeError("File is broken...")
        flag = file_helper.extract_files(
            zip_path=real_zip_path, trg_folder=self.file_folder
        )
        if not flag:
            raise FileNotFoundError("Extract second file failed")

        # Remove Zip File
        os.remove(self.file_path)
        os.remove(real_zip_path)

        # Get Configuration
        _cfg_path = glob.glob(os.path.join(self.file_folder, "export/*"))
        if len(_cfg_path) != 1:
            raise FileNotFoundError(
                "Configuration only have one, but got {}".format(", ".join(_cfg_path))
            )
        self.cfg_path = _cfg_path[0]

        # Get All Path: [ temp/<task_name>/<file_type>/<files> ... ]
        for file_path in glob.glob(
            os.path.join(self.file_folder, "*/*"), recursive=True
        ):
            tmp_dir, task_name, file_type, file_name = file_path.split("/")

            if file_type in ["data", "model"]:
                trg_path = os.path.join(file_type, file_name)
                move_data(org_path=file_path, trg_path=trg_path)

    def _get_valid_task_name(self, task_name: str):
        # Check duplicate name
        if not has_duplicate_task(task_name, duplicate_limit=1):
            return task_name

        new_task_name = task_name
        idx = 1
        while True:
            new_task_name = f"{task_name} ({idx})"
            if not has_duplicate_task(new_task_name, duplicate_limit=1):
                break
            idx += 1

        log.warning(f"Task rename from {task_name} to {new_task_name}")
        return new_task_name

    def _add_into_db(self):
        # Get Config Data
        with open(self.cfg_path, "r") as f:
            cfg_data = json.load(f)

        task_table = cfg_data["task"]
        model_table = cfg_data["model"]
        source_table = cfg_data["source"]
        app_table = cfg_data["app"]

        # Check duplicate name
        task_table["name"] = self._get_valid_task_name(task_table["name"])

        # Source
        db_handler.insert_data(
            table="source",
            data={
                "uid": source_table["uid"],
                "name": source_table["name"],
                "type": source_table["type"],
                "input": source_table["input"],
                "status": source_table["status"],
                "height": source_table["height"],
                "width": source_table["width"],
            },
            replace=True,
        )

        # APP
        db_handler.insert_data(
            table="app",
            data={
                "uid": self.uid,
                "name": app_table["name"],
                "type": app_table["type"],
                "app_setting": app_table["app_setting"],
            },
            replace=True,
        )

        # Task
        db_handler.insert_data(
            table="task",
            data={
                "uid": self.uid,
                "name": task_table["name"],
                "source_uid": task_table["source_uid"],
                "model_uid": task_table["model_uid"],
                "model_setting": task_table["model_setting"],
                "status": task_table["status"],
                "device": task_table["device"],
                "created_time": time.time(),
            },
            replace=True,
        )

        # Model
        model_handler.init_db_model()

        log.info("Get the config of the import task , trying to add into database ...")

        # Updated
        SERV_CONF["PROC"][self.uid].update(task_handler.get_task_info(self.uid)[0])

        # Delete file
        file_need_to_clean = [self.file_path, self.file_folder]
        for file in file_need_to_clean:
            if not os.path.exists(file):
                continue

            try:
                if os.path.isfile(file):
                    os.remove(file)
                    print(f"Clear file: {file}")
                elif os.path.isdir(file):
                    shutil.rmtree(file)
                    print(f"Clear folder: {len(os.listdir(file))}")

            except Exception as e:
                print(e)

    def main_event(self):
        t_down = time.time()
        log.info(self.S_DOWN)
        self.update_status(self.S_DOWN)
        self._download_event()
        self.performance["download"] = time.time() - t_down

        t_parse = time.time()
        log.info(self.S_PARS)
        self.update_status(self.S_PARS)
        self._parse_event()
        self.performance["parse"] = time.time() - t_parse

        log.info(self.S_IMPO)
        self.update_status(self.S_IMPO)
        self._add_into_db()
        self.update_status(self.S_FINISH)

        # Wait a second to ensure the success message is arrived.
        time.sleep(1)
        log.info("Finished !!!")

    def fail_event(self):
        log.warning("Fail Event ..")
        # Clear files
        file_need_to_clean = [self.file_path, self.file_folder]
        for file in file_need_to_clean:
            if not os.path.exists(file):
                continue

            try:
                if os.path.isfile(file):
                    os.remove(file)
                    print(f"Clear file: {file}")
                elif os.path.isdir(file):
                    shutil.rmtree(file)
                    print(f"Clear folder: {len(os.listdir(file))}")

            except Exception as e:
                print(e)

    def _parse_zip_name(self, file_name: str) -> tuple:
        file_name = os.path.basename(file_name)
        name_formats = os.path.splitext(file_name)[0].split("_")
        task_platform = name_formats[0]
        task_name = "".join(name_formats[1:-1])
        export_time = name_formats[-1]

        support_platforms = ["intel", "nv", "nvidia", "hailo", "xilinx"]
        if task_platform not in support_platforms:
            raise NameError(
                "Got unexpected platform, expect platform is {}".format(
                    support_platforms
                )
            )
        if not export_time.isdigit():
            raise NameError(
                "Go invalidate export time, it should be as <year><month><day>"
            )

        return (task_platform, task_name, export_time)


class TASK_ZIP_IMPORTER(TaskImporterWrapper):
    """IMPORTER for ZIP Model"""

    def __init__(self, file: File) -> None:
        super().__init__()

        # Buffer
        self.file = file

        # Name
        self.zip_name = self.file.filename
        self.zip_path = os.path.join(SERV_CONF["TEMP_DIR"], self.zip_name)

    def _parse_event(self):
        """Parsing Event

        - Structure
            <task_name>/
                data/
                model/
                export/     # db

        - Workflow
            1. Extract ZIP.
            2. Remove ZIP.
            3. Move the file to correct path
        """

        def move_data(org_path, trg_path):
            if os.path.exists(trg_path):
                if os.path.isdir(trg_path):
                    shutil.rmtree(trg_path)
                else:
                    os.remove(trg_path)
                log.warning(
                    "\t- Import file exists, auto remove it. ({})".format(trg_path)
                )

            shutil.move(org_path, trg_path)
            log.debug("\t- Move file from {} to {}".format(org_path, trg_path))

        # Extract
        self.zip_folder = os.path.join(
            SERV_CONF["TEMP_DIR"], os.path.splitext(self.zip_name)[0]
        )
        file_helper.extract_files(zip_path=self.zip_path, trg_folder=self.zip_folder)

        # Parse the task zip ( file_name, file_path means the task zip name and path )
        task_zips = glob.glob(f"{self.zip_folder}/*.zip")
        assert len(task_zips) == 1, "ZIP file should only have one."
        self.file_path = task_zips[0]
        self.file_name = os.path.basename(self.file_path)

        # Get platform, name and timestamp
        self.task_platform, self.task_name, self.export_time = self._parse_zip_name(
            self.file_path
        )

        # Update PROC information
        SERV_CONF["PROC"][self.uid]["name"] = self.task_name

        # Concatenate data
        self.file_folder = os.path.join(SERV_CONF["TEMP_DIR"], self.task_name)

        # Rename and update file_path
        os.rename(self.zip_folder, self.file_folder)
        self.file_path = self.file_path.replace(self.zip_folder, self.file_folder)

        log.debug("Parsing File:")
        log.debug(f"\t- Platform: {self.task_platform}")
        log.debug(f"\t- Name: {self.task_name}")
        log.debug(f"\t- Export Time: {self.export_time}")
        log.debug(f"\t- Save Path: {self.zip_path}")
        log.debug(f"\t- Target Task Path: {self.file_path}")

        # Verify with MD5
        md5_path = glob.glob(f"{self.file_folder}/MD5SUMS")[0]
        if encode.load_txt(md5_path)[0] != encode.md5(self.file_path):
            raise RuntimeError("File is broken...")

        # Extract task zip
        try:
            file_helper.extract_files(
                zip_path=self.file_path, trg_folder=self.file_folder
            )
        except Exception:
            raise FileNotFoundError("Extract second file failed")

        # Remove Zip File
        os.remove(self.zip_path)
        os.remove(self.file_path)

        # Get Configuration
        export_files = glob.glob(os.path.join(self.file_folder, "export/*"))
        if len(export_files) != 1:
            raise FileNotFoundError(
                "Configuration only have one, but got {}".format(
                    ", ".join(export_files)
                )
            )
        self.cfg_path = export_files[0]

        # Get All Path: [ temp/<task_name>/<file_type>/<files> ... ]
        for file_path in glob.glob(
            os.path.join(self.file_folder, "*/*"), recursive=True
        ):
            tmp_dir, task_name, file_type, file_name = file_path.split("/")
            if file_type not in ["data", "model"]:
                continue

            trg_path = os.path.join(file_type, file_name)
            move_data(org_path=file_path, trg_path=trg_path)

    def _download_event(self):
        """Download file via FastAPI"""

        with open(self.zip_name, "wb") as buffer:
            shutil.copyfileobj(self.file.file, buffer)
        shutil.move(self.zip_name, self.zip_path)

        file_helper.valid_zip(self.zip_path)
