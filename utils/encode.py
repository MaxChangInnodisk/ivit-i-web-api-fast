# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import uuid, json, re

def gen_uid3(name):
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, name))[:8]

def gen_uid4():
    return str(uuid.uuid4())[:8]

def gen_uid(name:str=None):
    return gen_uid3(name) if name else gen_uid4() 

def load_json(path:str):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def json_to_str(data:dict) -> str:
    return re.sub('["]', '\"', json.dumps(data))

def load_db_json(db_json:str):
    return json.loads(re.sub('[\']', '"', db_json))

def check_json(s):
    try:
        json.decode(s)
        return True
    except json.JSONDecodeError:
        return False

def pure_jsonify_2(in_dict, ret_dict, exclude_key:list=[], include_key=[dict, list, str, int, float, bool]):    
    for key, val in in_dict.items():
        try:
            if (key in exclude_key):
                ret_dict[key]=str(type(val)) if val!=None else None
                continue
            if (type(val) in include_key ):
                ret_dict[key] = val
                pure_jsonify_2(in_dict[key], ret_dict[key])
            else:
                ret_dict[key] = str(type(val))
        except:
            continue

def get_pure_jsonify(in_dict:dict, json_format=True)-> dict:
    ret_dict = dict()
    # temp_in_dict = copy.deepcopy(in_dict)
    temp_in_dict = in_dict
    pure_jsonify_2(temp_in_dict, ret_dict)
    # return ret_dict
    ret = json.dumps(ret_dict, cls=NumpyEncoder, indent=4)
    return json.loads(ret) if json_format else ret

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
