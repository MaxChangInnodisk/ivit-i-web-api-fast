# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging as log
from typing import Union

# Import iVIT-I
from common import RT_CONF, SERV_CONF
from handlers.db_handler import parse_app_data, select_data
from handlers.ivit_handler import iAPP_CLS, iAPP_HANDLER, iAPP_OBJ, iAPP_SEG


def init_db_app() -> None:
    """Add iAPP_HANDLER into RT_CONF

    Workflow:
    1. Initialize iAPP_HANDLER
    2. Register SERV_CONF['APP_DIR']
    3. Add hanlder into RT_CONF with 'iAPP' key
    """
    app_handler = iAPP_HANDLER()
    app_handler.register_from_folder(app_folder=SERV_CONF["APP_DIR"])
    RT_CONF.update({"IAPP": app_handler})


def create_app(app_uid: str, label_path: str) -> Union[iAPP_CLS, iAPP_OBJ, iAPP_SEG]:
    """Create execute application

    Args:
        app_uid (str): the applicatoin uid
        label_path (str): the path to label file

    Returns:
        Union[iAPP_CLS, iAPP_OBJ, iAPP_SEG]: return application which inherit with iAPP_<TYPE>
    """

    # Application Information
    app_data = select_data(table="app", data="*", condition=f"WHERE uid='{app_uid}'")
    task_data = select_data(
        table="task", data=["uid", "name"], condition=f"WHERE uid='{app_uid}'"
    )

    task_uid, task_name = task_data[0]
    log.info("Creating application ( {}: {})".format(task_name, task_uid))

    app_info = parse_app_data(app_data[0])

    # NOTE: check enable is True
    # We have multiple area have to check
    for idx, area_info in enumerate(app_info["app_setting"]["application"]["areas"]):
        if "events" not in area_info:
            print("{} not setup events".format(area_info["name"]))
            continue
        # NOTE: default is True
        if not area_info["events"].get("enable", True):
            log.warning("Delete event object !")
            del app_info["app_setting"]["application"]["areas"][idx]["events"]

    # Instance App Object
    app = RT_CONF["IAPP"].get_app(app_info["name"])(
        params=app_info["app_setting"], label=label_path
    )

    log.info("Initialized application: {}".format(app_uid))
    return app


if __name__ == "__main__":
    init_db_app()
