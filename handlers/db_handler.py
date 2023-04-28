# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys, time, json, re, traceback, sqlite3
import logging as log
from typing import Union

from ..utils import load_db_json
from ..common import SERV_CONF


def check_db_is_empty(dp_path:str):
    """ Check Database is empty """
    is_empty = True
    conn = sqlite3.connect(dp_path)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM task ")
    result = c.fetchone()

    if result[0] == 0:
        is_empty = True
    else:
        is_empty = False
    conn.close()
    return is_empty


def insert_data(table:str, data:dict, db_path:str=SERV_CONF["DB_PATH"], replace:bool=False) -> None:
    """ Insert data into `ivit_i.sqlite`
    ---

    - arguments
        - table: name of the table
        - data: data you want to insert or replace into database 
        - db_path: path to database, default is SERV_CONF["DB_PATH"]
    
    - mecs
        - sqlite insert sample
            - ```INSERT OR REPLACE INTO table_name (column1, column2, column3, ...)
            VALUES (value1, value2, value3, ...);```
    """

    head = f"INSERT {'OR REPLACE' if replace else ''} INTO {table}"
    col, val = [], []
    
    for db_key, db_val in data.items():
        # Append Key
        col.append(f"\"{db_key}\"")
        
        # Process Value
        if isinstance(db_val, str) or db_val is None:
            db_val = f'"{db_val}\"'
        val.append(db_val)

    col_string = "({})".format( ', '.join(col) )
    val_string = "({})".format( ', '.join(val) )
    
    commond = "{} {} VALUES {}".format(
        head, col_string, val_string )

    # log.debug(commond)

    conn = sqlite3.connect( db_path, check_same_thread=False )
    ivit_db = conn.cursor()
    
    ivit_db.execute(commond)

    conn.commit()
    conn.close()


def update_data(table:str, data:dict, condition:Union[str, None]=None, db_path:str=SERV_CONF["DB_PATH"]) -> list:
    """ A Helper function to select database data
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
    
    set_list =[ f"{key} = '{val}'" for key, val in data.items() ]
    set_string = ', '.join(set_list)

    # Concatenate
    command = f"{head} {set_string}"

    # If got the condition
    if condition:
        command = f"{command} {condition}"

    # Write into database
    conn = sqlite3.connect(db_path)
    ivit_db = conn.cursor()
    ivit_db.execute(command)    
    conn.commit()
    conn.close()


def select_data(table:str, data:Union[str,list], condition:Union[str, None]=None, db_path:str=SERV_CONF["DB_PATH"]) -> list:
    """ A Helper function to select database data
    ---

    - 
    SELECT aaa, bbb, ccc FROM users
    SELECT * FROM users WHERE email = 'user@example.com';
    """
    head, tail = "SELECT", f"from {table}"
    
    # Get correct data
    if isinstance(data, list):
        data = ', '.join(data) 
    
    # Concatenate
    command = f"{head} {data} {tail}"

    # If got the condition
    if condition:
        command = f"{command} {condition}"

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    cursor = c.execute(command)     
    ret_data = [ row for row in cursor ]    
    conn.close()

    return ret_data


def delete_data(table:str, condition:Union[str, None]=None, db_path:str=SERV_CONF["DB_PATH"]) -> list:
    """ A Helper function to select database data
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
    c = conn.cursor()
    cursor = c.execute(command)     
    conn.commit()
    conn.close()


def init_tables(db_path:str):
    """ Initialize SQLite Table """

    conn = sqlite3.connect( db_path, check_same_thread=False )
    ivit_db = conn.cursor()

    # Task
    ivit_db.execute("CREATE TABLE IF NOT EXISTS task ("
                    "uid TEXT PRIMARY KEY,"
                    "name TEXT,"
                    "source_uid TEXT,"
                    "model_uid TEXT,"
                    "model_setting TEXT,"
                    "status TEXT,"
                    "device TEXT,"
                    "annotation TEXT,"
                    "FOREIGN KEY(source_uid) REFERENCES source(uid) ON DELETE CASCADE,"
                    "FOREIGN KEY(model_uid) REFERENCES model(uid) ON DELETE CASCADE"
                    ");" )
    # Model
    ivit_db.execute("CREATE TABLE IF NOT EXISTS model ("
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
                    "annotation TEXT"
                    ");" )
    # Source
    ivit_db.execute("CREATE TABLE IF NOT EXISTS source ("
                    "uid TEXT PRIMARY KEY,"
                    "name TEXT,"
                    "status TEXT,"
                    "type TEXT,"
                    "input TEXT,"
                    "annotation TEXT"
                    ");" )
    # Application
    ivit_db.execute("CREATE TABLE IF NOT EXISTS app ("
                    "uid TEXT PRIMARY KEY,"
                    "name TEXT,"
                    "type TEXT,"
                    "app_setting TEXT,"
                    "event_uid TEXT,"
                    "annotation TEXT,"
                    "FOREIGN KEY(uid) REFERENCES task(uid) ON DELETE CASCADE"
                    ");" )
    
    # Event
    ivit_db.execute("CREATE TABLE IF NOT EXISTS event ("
                    "uid TEXT PRIMARY KEY,"
                    "event_setting TEXT,"
                    "annotation TEXT"
                    ");" )
    
    ivit_db.close()
    conn.commit()
    conn.close()


def init_sqlite(db_path=SERV_CONF["DB_PATH"]):
    """ Initialize SQLite """
    
    t_start = time.time()

    log.info(f'Connect to Database: {db_path}')    

    init_tables(db_path)

    log.info(f'Initialized Database ( {round(time.time()-t_start, 3)}s )')


def reset_db(db_path=SERV_CONF["DB_PATH"]):
    """ Reset each status to stop """
    update_data(table="task", data={
        "status": "stop"
    })
    update_data(table="source", data={
        "status": "stop"
    })



def parse_task_data(data: dict):
    return {
        "uid": data[0],
        "name": data[1],
        "source_uid": data[2],
        "model_uid": data[3],
        "model_setting": load_db_json(data[4]),
        "status": data[5],
        "device": data[6] 
    }


def parse_model_data(data: dict):
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
        "meta_data": load_db_json(data[9]),
        "annotation": data[10]
    }


def parse_source_data(data: dict):
    return {
        "uid": data[0],
        "name": data[1],
        "status": data[2],
        "type": data[3],
        "input": data[4],
        "annotation": data[5]
    }


def parse_app_data(data: dict):
    return {
        "uid": data[0],
        "name": data[1],
        "type": data[2],
        "app_setting": load_db_json(data[3]),
        "event_uid": data[4],
        "annotation": data[5]
    }



