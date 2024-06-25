# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import logging as log
import sqlite3
import time
from typing import Tuple, Union

from common import SERV_CONF

# --------------------------------------------------------
# Sqlite3 Command Helper


def db_to_list(cursor: sqlite3.Cursor) -> list:
    """Convert sqlite data to list"""
    return [row for row in cursor.fetchall()]


def is_list_empty(data: list) -> bool:
    """Check list is empty or not"""
    return len(data) == 0


def connect_db(
    dp_path: str = SERV_CONF["DB_PATH"],
) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    con = sqlite3.connect(dp_path)
    cur = con.cursor()
    return (con, cur)


def close_db(con: sqlite3.Connection, cur: sqlite3.Cursor) -> None:
    cur.close()
    con.close()


def select_column_by_uid(
    cursor: sqlite3.Cursor, table: str, uid: str, cols: Union[str, list]
) -> list:
    data = db_to_list(
        cursor.execute(
            """SELECT {} FROM {} WHERE uid=\"{}\"""".format(
                ", ".join(cols) if isinstance(cols, list) else cols, table, uid
            )
        )
    )
    return data


def update_column_by_uid(cursor: sqlite3.Cursor, table: str, uid: str, data: dict):
    set_list = [f'{key} = "{val}"' for key, val in data.items()]
    set_string = ", ".join(set_list)
    return db_to_list(
        cursor.execute(
            """UPDATE {} SET {} WHERE uid=\"{}\"""".format(table, set_string, uid)
        )
    )


# --------------------------------------------------------
# Sqlite3 Helper: The function below will connect sqlite and close each time


def is_db_empty(dp_path: str) -> bool:
    """Check database is empty or not

    Args:
        dp_path (str): path to sqlite

    Returns:
        bool: is empty or not
    """
    con, cur = connect_db(dp_path)
    nums = db_to_list(cur.execute("SELECT COUNT(*) FROM task "))[0][0]
    close_db(con, cur)
    return nums == 0


def insert_data(
    table: str, data: dict, db_path: str = SERV_CONF["DB_PATH"], replace: bool = False
) -> None:
    """Insert data into `ivit_i.sqlite`
    ---

    - arguments
        - table: name of the table
        - data: data you want to insert or replace into database
        - db_path: path to database, default is SERV_CONF["DB_PATH"]

    - mecs
        - sqlite insert sample
            - ```INSERT OR REPLACE INTO table_name (column1, column2, column3, ...)
            VALUES (value1, value2, value3, ...);```

    - workflow
        1. 將主要的語法組合好，包含 Table
        2. 將所有的 Key 給裝入 () 當中
        3. VALUES 的部份則根據有幾個 Key 就給幾個 ?
        4. 同時組裝參數 (<value_1>, <value_2>, <value_3> ... , )
    """

    head = f"INSERT {'OR REPLACE' if replace else ''} INTO {table}"
    cols, vals = list(data.keys()), []

    col_string = "({})".format(", ".join(cols))
    val_string = "({})".format(", ".join(["?" for _ in range(len(cols))]))

    command = "{} {} VALUES {}".format(head, col_string, val_string)

    for val in data.values():
        vals.append(val if isinstance(val, str) else json.dumps(val))

    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(command, tuple(vals))
    conn.commit()
    conn.close()


def update_data(
    table: str,
    data: dict,
    condition: Union[str, None] = None,
    db_path: str = SERV_CONF["DB_PATH"],
) -> list:
    """A Helper function to select database data
    ---

    -
    UPDATE employees
        SET city = 'Toronto',
            state = 'ON',
            postalcode = 'M5P 2N7'
        WHERE
            employeeid = 4;

    """
    head = f"UPDATE {table} SET "

    set_list = []
    for key, val in data.items():
        if isinstance(val, dict):
            data = f"{key} = '{json.dumps(val)}'"
        else:
            data = f"{key} = '{val}'"
        set_list.append(data)

    set_string = ", ".join(set_list)

    # Concatenate
    command = f"{head} {set_string}"

    # If got the condition
    if condition:
        command = command + " " + condition

    # Write into database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(command)
        conn.commit()
        conn.close()

    except Exception as e:
        log.exception(e)


def select_data(
    table: str,
    data: Union[str, list],
    condition: Union[str, None] = None,
    db_path: str = SERV_CONF["DB_PATH"],
) -> list:
    """A Helper function to select database data
    ---

    -
    SELECT aaa, bbb, ccc FROM users
    SELECT * FROM users WHERE email = 'user@example.com';
    """
    head, tail = "SELECT", f"from {table}"

    # Get correct data
    if isinstance(data, list):
        data = ", ".join(data)

    # Concatenate
    command = f"{head} {data} {tail}"

    # If got the condition
    if condition:
        command = command + " " + condition

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    data = db_to_list(cursor.execute(command))
    conn.close()

    return data


def delete_data(
    table: str, condition: Union[str, None] = None, db_path: str = SERV_CONF["DB_PATH"]
) -> list:
    """A Helper function to select database data
    ---

    -
    DELETE FROM COMPANY WHERE ID = 7;

    """
    head = f"DELETE FROM {table}"
    command = head

    # If got the condition
    if condition:
        command = f"{command} {condition}"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(command)
    conn.commit()
    conn.close()


def init_tables(db_path: str):
    """Initialize SQLite Table"""

    conn = sqlite3.connect(db_path, check_same_thread=False)
    ivit_db = conn.cursor()

    # Task
    ivit_db.execute(
        "CREATE TABLE IF NOT EXISTS task ("
        "uid TEXT PRIMARY KEY,"
        "name TEXT,"
        "source_uid TEXT,"
        "model_uid TEXT,"
        "model_setting TEXT,"
        "status TEXT,"
        "device TEXT,"
        "error TEXT,"
        "created_time TEXT"
        "annotation TEXT,"
        "FOREIGN KEY(source_uid) REFERENCES source(uid) ON DELETE CASCADE,"
        "FOREIGN KEY(model_uid) REFERENCES model(uid) ON DELETE CASCADE"
        ");"
    )
    # Model
    ivit_db.execute(
        "CREATE TABLE IF NOT EXISTS model ("
        "uid TEXT PRIMARY KEY,"
        "name TEXT,"
        "type TEXT,"
        "model_path TEXT,"
        "label_path TEXT,"
        "json_path TEXT,"
        "classes TEXT,"
        "input_size TEXT,"
        "preprocess TEXT,"
        "meta_data TEXT,"
        "default_model INTEGER,"
        "annotation TEXT"
        ");"
    )
    # Source
    ivit_db.execute(
        "CREATE TABLE IF NOT EXISTS source ("
        "uid TEXT PRIMARY KEY,"
        "name TEXT,"
        "status TEXT,"
        "type TEXT,"
        "input TEXT,"
        "width TEXT,"
        "height TEXT,"
        "annotation TEXT"
        ");"
    )
    # Application
    ivit_db.execute(
        "CREATE TABLE IF NOT EXISTS app ("
        "uid TEXT PRIMARY KEY,"
        "name TEXT,"
        "type TEXT,"
        "app_setting TEXT,"
        "event_uid TEXT,"
        "annotation TEXT,"
        "FOREIGN KEY(uid) REFERENCES task(uid) ON DELETE CASCADE"
        ");"
    )
    # Event
    ivit_db.execute(
        "CREATE TABLE IF NOT EXISTS event ("
        "uid TEXT,"
        "title TEXT,"
        "app_uid TEXT,"
        "start_time INTEGER PRIMARY KEY,"
        "end_time INTEGER,"
        "annotation TEXT,"
        "FOREIGN KEY(app_uid) REFERENCES app(uid) ON DELETE CASCADE"
        ");"
    )

    ivit_db.close()
    conn.commit()
    conn.close()


def init_sqlite(db_path=SERV_CONF["DB_PATH"]):
    """Initialize SQLite"""

    t_start = time.time()

    log.info(f"Connect to Database: {db_path}")

    init_tables(db_path)

    log.info(f"Initialized Database ( {round(time.time()-t_start, 3)}s )")


def reset_db(db_path=SERV_CONF["DB_PATH"]):
    """Reset each status to stop"""
    update_data(table="task", data={"status": "stop"})
    update_data(table="source", data={"status": "stop"})


# --------------------------------------------------------
# Sqlite3 Parser: Parse each table data


def parse_task_data(data: Union[dict, sqlite3.Cursor]) -> dict:
    """Parse all task data

    Args:
        data (Union[dict, sqlite3.Cursor]): _description_

    Returns:
        dict: Support key is: uid, name, source_uid, model_uid, model_setting, status, device, error
    """
    error_message = data[7] if data[7] != "{}" else None
    try:
        error_message = json.loads(error_message)
    except Exception:
        pass

    return {
        "uid": data[0],
        "name": data[1],
        "source_uid": data[2],
        "model_uid": data[3],
        "model_setting": json.loads(data[4]),
        "status": data[5],
        "device": data[6],
        "error": error_message,
        "created_time": data[8],
    }


def parse_model_data(data: Union[dict, sqlite3.Cursor]):
    return {
        "uid": data[0],
        "name": data[1],
        "type": data[2],
        "model_path": data[3],
        "label_path": data[4],
        "json_path": data[5],
        "classes": data[6],
        "input_size": data[7],
        "preprocess": data[8],
        "meta_data": json.loads(data[9]),
        "default_model": data[10],
        "annotation": data[11],
    }


def parse_source_data(data: dict):
    return {
        "uid": data[0],
        "name": data[1],
        "status": data[2],
        "type": data[3],
        "input": data[4],
        "width": data[5],
        "height": data[6],
        "annotation": data[7],
    }


def parse_app_data(data: dict):
    return {
        "uid": data[0],
        "name": data[1],
        "type": data[2],
        "app_setting": json.loads(data[3]),
        "event_uid": data[4],
        "annotation": data[5],
    }


def parse_event_data(data: dict):
    return {
        "uid": data[0],
        "title": data[1],
        "app_uid": data[2],
        "start_time": data[3],
        "end_time": data[4],
        "annotation": json.loads(data[5]),
    }
