import time
import os
import colorsys
from typing import Tuple, Callable, List
from collections import defaultdict
import numpy as np
import math

    
def rgb2hsv(rgb: tuple):
    return colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])

def rgb2hls(rgb: tuple):
    return colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])

def get_font_color(bg_color:tuple):
    (h,l,s) = rgb2hls(bg_color)
    black, white = (0,0,0), (255,255,255)
    if l > 130 or s < -0.9:
        return black 
    return white

def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print('[Timeit] Function', func.__name__, 'time:', round((te -ts)*1000,1), 'ms')
        return result
    return timed

def get_time(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print('[Timeit] Function', func.__name__, 'time:', round((te -ts)*1000,1), 'ms')
        return (result, te-ts)
    return timed


def update_palette_and_labels(custom_palette:dict,  default_palette:dict, label_path:str) -> Tuple[dict, list]:
    """update the color palette ( self.palette ) which key is label name and label list ( self.labels).

    Args:
        custom_palette (dict): the custom palette which key is the label name
        default_palette (dict): the default palette which key is integer with string type.
        label_path (str): the path to label file

    Raises:
        TypeError: if the default palette with wrong type
        FileNotFoundError: if not find label file

    Returns:
        Tuple[dict, list]: palette, labels
    """

    # check type and path is available
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Can not find label file in '{label_path}'.")
    
    if not isinstance(custom_palette, dict):
        raise TypeError(f"Expect custom palette type is dict, but get {type(custom_palette)}.")

    # Params
    ret_palette, ret_labels = {}, []

    # update palette and label
    idx = 1
    f = open(label_path, 'r')
    for raw_label in f.readlines():

        label = raw_label.strip()
        
        # if setup custom palette
        if label in custom_palette:
            color = custom_palette[label]
        
        # use default palette 
        else:
            color = default_palette[str(idx)]
        
        # update palette, labels and idx
        ret_palette[label] = color
        ret_labels.append(label)
        idx += 1

    f.close()

    return (ret_palette, ret_labels)


def get_logic_map() -> dict:
    greater = lambda x,y: x>y
    greater_or_equal = lambda x,y: x>=y
    less = lambda x,y: x<y
    less_or_equal = lambda x,y: x<=y
    equal = lambda x,y: x==y
    return {
        '>': greater,
        '>=': greater_or_equal,
        '<': less,
        '<=': less_or_equal,
        '=': equal,
    }



def denorm_area_points(width: int, height: int, area_points: list) -> list:
    return [ [ math.ceil(x*width), math.ceil(y*height) ] for [x, y] in area_points ]

def denorm_line_points(width: int, height: int, line_points: dict) -> dict:
    """Denormalize line points
    
    - Input Sample
        "line_point": {
            "line_1": [
                [ 0.36666666666, 0.64074074074 ],
                [ 0.67291666666, 0.52962962963 ]
            ],
            "line_2": [
                [ 0.36041666666, 0.83333333333 ],
                [ 0.72916666666, 0.62962962963 ]
            ],
        }
    """
    ret = defaultdict(list)
    for line_name, line_point in line_points.items():
        for [x, y] in line_point:
            ret[line_name].append( [ math.ceil(x*width), math.ceil(y*height) ] )
    return ret

def sort_area_points(point_list: list) -> list:
    """
    This function will help user to sort the point in the list counterclockwise.
    step 1 : We will calculate the center point of the cluster of point list.
    step 2 : calculate arctan for each point in point list.
    step 3 : sorted by arctan.

    Args:
        point_list (list): not sort point.

    Returns:
        sorted_point_list(list): after sort.
    
    """

    cen_x, cen_y = np.mean(point_list, axis=0)
    #refer_line = np.array([10,0]) 
    temp_point_list = []
    sorted_point_list = []
    for i in range(len(point_list)):

        o_x = point_list[i][0] - cen_x
        o_y = point_list[i][1] - cen_y
        atan2 = np.arctan2(o_y, o_x)
        # angle between -180~180
        if atan2 < 0:
            atan2 += np.pi * 2
        temp_point_list.append([point_list[i], atan2])
    
    temp_point_list = sorted(temp_point_list, key=lambda x:x[1])
    for x in temp_point_list:
        sorted_point_list.append(x[0])

    return sorted_point_list
        








