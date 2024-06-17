# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import logging as log
import os
import shutil
from typing import Optional

import cv2
import numpy as np

# Import iVIT-I
try:
    from ..common import RT_CONF
    from .app_handler import create_app
    from .db_handler import (
        close_db,
        connect_db,
        db_to_list,
        delete_data,
        is_list_empty,
        parse_event_data,
        select_data,
    )

except BaseException:
    from common import RT_CONF
    from handlers.app_handler import create_app
    from handlers.db_handler import (
        close_db,
        connect_db,
        db_to_list,
        delete_data,
        is_list_empty,
        parse_event_data,
        select_data,
    )


class DetectionWrapper:
    xmin = (None,)
    ymin = (None,)
    xmax = (None,)
    ymax = (None,)
    label = (None,)
    score = (None,)


def verify_event_exist(uid: str):
    # Task Information
    events = select_data(table="event", data="*", condition=f"WHERE uid='{uid}'")

    # Not found AI Task
    if events == []:
        raise RuntimeError(f"Could not find Event ({uid})")

    return events[0]


def get_all_events(condition: Optional[str] = None) -> list:
    events = select_data(table="event", data="*", condition=condition)
    event_nums = len(events)
    print(f"Get {event_nums} events")
    ret = []
    for event in events:
        data = parse_event_data(event)
        data["start_time"] = str(data["start_time"])
        data["end_time"] = str(data["end_time"])
        data["annotation"].pop("detections", None)
        ret.append(data)
    return ret


def del_all_events() -> None:
    events = select_data(table="event", data="*", condition=condition)
    event_nums = len(events)
    print(f"Get {event_nums} events")
    ret = []
    for event in events:
        data = parse_event_data(event)
        try:
            del_event(data["uid"])
        except Exception:
            log.warning("Delete event error.")


def get_cond_events(condition: str) -> list:
    events = select_data(table="event", data="*", condition=condition)
    event_nums = len(events)
    print(f"Get {event_nums} events")
    return [parse_event_data(event) for event in events]


def get_events(data):
    conditions = []

    # if event_uids is
    if data.event_uid:
        new_condition = f"uid='{data.event_uid}'"
        conditions.append(new_condition)

    # if data.start_time
    if data.start_time:
        new_condition = f"start_time >= {data.start_time}"
        conditions.append(new_condition)

    # if data.end_time
    if data.end_time:
        new_condition = f"end_time <= {data.end_time}"
        conditions.append(new_condition)

    if len(conditions) == 0:
        return get_all_events()

    raw_conditions = "WHERE "
    raw_conditions += " AND ".join(conditions)
    print(raw_conditions)
    return get_all_events(condition=raw_conditions)


def del_event(uid: str):
    """"""

    verify_event_exist(uid)

    # Del Screenshot
    del_event_screenshot(uid)

    # Del App
    delete_data(table="event", condition=f"WHERE uid='{uid}'")


def get_event_screenshot(timestamp: int, draw_result: bool = False) -> np.ndarray:
    """"""
    # ------------------------------------
    # Get data from db
    con, cur = connect_db()
    event_db_data = db_to_list(
        cur.execute(
            f"""SELECT * FROM event WHERE start_time={timestamp} OR end_time={timestamp}"""
        )
    )

    if is_list_empty(event_db_data):
        raise KeyError(f"Can not find the timestamp ({timestamp})")

    event_data = parse_event_data(event_db_data[0])

    # Parse event data
    event_uid = event_data["uid"]
    task_uid = app_uid = event_data["app_uid"]

    # Get detections from event data
    file_path = f"events/{event_uid}/{timestamp}.json"
    with open(file_path) as f:
        event_data = json.load(f)

    # Get another require data ( model_uid, label_uid )
    model_uid = db_to_list(
        cur.execute(f'''SELECT model_uid FROM task WHERE uid="{task_uid}"''')
    )[0][0]
    label_path = db_to_list(
        cur.execute(f'''SELECT label_path FROM model WHERE uid="{model_uid}"''')
    )[0][0]

    close_db(con, cur)

    # ------------------------------------
    # NOTE: Image
    file_path = f"events/{event_uid}/{timestamp}.jpg"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Can not find image: {file_path}")

    # Read Image
    frame = cv2.imread(file_path)
    if frame is None:
        raise RuntimeError("Can not open image")

    # ------------------------------------
    # Retur original frame if not draw_results
    if not draw_result:
        return frame

    # ------------------------------------
    # init app, need_clean
    if app_uid in RT_CONF:
        need_clean = False
        app = RT_CONF[app_uid].app

    else:
        # NOTE: create new app
        need_clean = True
        app = create_app(app_uid=app_uid, label_path=label_path)

    # Update Options
    app.draw_area = False
    app.draw_bbox = True
    app.draw_label = True
    app.draw_result = True
    app.force_close_event = True

    # from collections import namedtuple
    # Wrapper = namedtuple("Wrapper", "xmin, ymin, xmax, ymax, label, score")
    # wrap_dets = [ Wrapper(
    #     det["xmin"], det["ymin"], det["xmax"], det["ymax"],
    #     det["label"], det["score"]
    # ) for det in dets ]
    # draw, _, _ = app(frame, wrap_dets)

    # Remove event
    draw = app.draw_event_data(frame, event_data)

    # clear app
    if need_clean:
        del app

    return draw


def del_event_screenshot(uid: str) -> None:
    """Delete the screenshot of the event

    Args:
        uid (str): event uid
    """
    event_folder = os.path.join("events", uid)

    if not os.path.exists(event_folder):
        return

    shutil.rmtree(event_folder)

    log.warning(
        "Delete event screenshot folder ... {}".format(
            "FAIL" if os.path.exists(event_folder) else "PASS"
        )
    )
