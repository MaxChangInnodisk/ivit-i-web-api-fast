import sys, os, cv2, logging, time
import numpy as np
import uuid
import os
import threading
import math
from typing import Any, Union, get_args
from datetime import datetime
from apps.palette import palette
from ivit_i.common.app import iAPP_OBJ
import os
import numpy as np
import time
from typing import Tuple, Callable, List
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import json

# Parameters
FRAME_SCALE     = 0.0005    # Custom Value which Related with Resolution
BASE_THICK      = 1         # Setup Basic Thick Value
BASE_FONT_SIZE  = 0.5   # Setup Basic Font Size Value
FONT_SCALE      = 0.2   # Custom Value which Related with the size of the font.
WIDTH_SPACE = 10
HEIGHT_SPACE = 10
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX
LINE_TYPE = cv2.LINE_AA

def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print('[Timeit] Function', func.__name__, 'time:', round((te -ts)*1000,1), 'ms')
        return result
    return timed


# ------------------------------------------------------------------------    

def denorm_area_points(width: int, height: int, area_points: list) -> list:
    print(width, height, area_points)

    return [ [ math.ceil(x*width), math.ceil(y*height) ] for [x, y] in area_points ]


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

# ------------------------------------------------------------------------    

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

# ------------------------------------------------------------------------    

class DrawTool:
    """ Draw Tool for Label, Bounding Box, Area ... etc. """

    def __init__(self, labels:list, palette:dict) -> None:
        # palette[cat] = (0,0,255), labels = [ cat, dog, ... ]
        self.palette, self.labels = palette, labels
        
        # Initialize draw params
        self.font_size = 1
        self.font_thick = 1
        
        # bbox
        self.bbox_line = 1
        
        # circle
        self.circle_radius = 3

        # area
        self.area_thick = 3
        self.area_color = [ 0,0,255 ]
        self.area_opacity = 0.4          
        self.draw_params_is_ready = False

        # output
        self.out_color = [0, 255, 255]
        self.out_font_color = [0, 0, 0]

    def update_draw_params(self, frame: np.ndarray) -> None:
        
        if self.draw_params_is_ready: return

        # Get Frame Size
        self.frame_size = frame.shape[:2]
        
        # Calculate the common scale
        scale = FRAME_SCALE * sum(self.frame_size)
        
        # Get dynamic thick and dynamic size 
        self.thick  = BASE_THICK + round( scale )
        self.font_thick = self.thick//2
        self.font_size = BASE_FONT_SIZE + ( scale*FONT_SCALE )
        self.width_space = int(scale*WIDTH_SPACE) 
        self.height_space = int(scale*HEIGHT_SPACE) 

        # Change Flag
        self.draw_params_is_ready = True
        print('Updated draw parameters !!!')

    def draw_areas(  self, 
                    frame: np.ndarray, 
                    areas: list, 
                    draw_point: bool= True,
                    draw_name: bool= True,
                    draw_line: bool= True,
                    name: str = None,
                    radius: int = None,
                    color: list = None, 
                    thick: int = None,
                    opacity: float = None) -> np.ndarray:

        radius = radius if radius else self.circle_radius
        color = color if color else self.area_color
        opacity = opacity if opacity else self.area_opacity
        thick = thick if thick else self.area_thick
        
        overlay = frame.copy()

        for area in areas:
            
            area_pts = area["area_point"]
            
            # draw poly
            cv2.fillPoly(overlay, pts=[ np.array(area_pts) ], color=color)

            # draw point and line if need
            prev_point_for_line = area_pts[-1]       # for line
            for point in area_pts:

                # draw point
                if draw_point:
                    cv2.circle(frame, tuple(point), radius, color, -1)
                
                # draw line
                if draw_line:
                    cv2.line(frame, point, prev_point_for_line, color, thick)
                    prev_point_for_line = point

                if draw_name and not name:
                    pass

        return cv2.addWeighted( frame, 1-opacity, overlay, opacity, 0 ) 

    def draw_bbox(self, frame: np.ndarray, left_top: list, right_bottom: list, color: list, thick: int= None) -> None:
        # Draw bbox
        thick = thick if thick else self.thick
        (xmin, ymin), (xmax, ymax) = map(int, left_top), map(int, right_bottom)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color , thick)

    def draw_label(self, frame: np.ndarray, label: str, left_bottom: list, color: list, thick: int=None) -> None:
        # Draw label
        xmin, ymin = left_bottom
        thick = thick if thick else self.thick

        (t_wid, t_hei), t_base = cv2.getTextSize(label, FONT_TYPE, self.font_size, self.font_thick)
        t_xmin, t_ymin, t_xmax, t_ymax = xmin, ymin-(t_hei+(t_base*2)), xmin+t_wid, ymin
        cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), color , -1)
        cv2.putText(
            frame, label, (xmin, ymin-(t_base)), FONT_TYPE,
            self.font_size, (255,255,255), self.font_thick, LINE_TYPE
        )

    def draw_area_results(self, frame: np.ndarray, areas: dict, color: list= None, font_color: list= None) -> None:
        color = color if color else self.out_color
        font_color = font_color if font_color else self.out_font_color
        
        cur_idx = 0
        for area in areas:
            area_name = area["name"]
            area_output = area.get("output", [])
            for (cur_label, cur_nums) in area_output.items():
                
                result = f"{area_name} : {cur_nums} {cur_label}"
                
                (t_wid, t_hei), t_base = cv2.getTextSize(result, FONT_TYPE, self.font_size, self.font_thick)
                
                t_xmin = WIDTH_SPACE
                t_ymin = HEIGHT_SPACE + ( HEIGHT_SPACE*cur_idx) + (cur_idx*(t_hei+t_base))
                t_xmax = t_wid + WIDTH_SPACE
                t_ymax = HEIGHT_SPACE + ( HEIGHT_SPACE*cur_idx) + ((cur_idx+1)*(t_hei+t_base))
                
                cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), color , -1)
                cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), (0,0,0) , 1)
                cv2.putText(
                    frame, result, (t_xmin, t_ymax), FONT_TYPE,
                    self.font_size, font_color, self.font_thick, LINE_TYPE
                )
                cur_idx += 1

    def get_color(self, label:str) -> list:
        return self.palette[label]

# ------------------------------------------------------------------------    

class EventHandler:

    def __init__(   self, 
                    title: str, 
                    folder: str,
                    drawer: DrawTool, 
                    operator: str, 
                    threshold: float, 
                    cooldown: float ):
        # Basic
        self.title = title
        self.drawer = drawer
        self.threshold = threshold
        self.cooldown = cooldown

        # Logic
        self.logic_map = get_logic_map()
        self.logic_event = self.logic_map[operator]
    
        # Timer
        self.trigger_time = time.time_ns()

        # Generate uuid
        self.folder = self.check_folder(folder)
        self.uuid = str(uuid.uuid4())[:8]
        self.uuid_folder = self.check_folder(os.path.join(self.folder, self.uuid))
        
        # Dynamic Variable
        self.current_time = time.time_ns()
        self.current_value = None

        # Custom Threading Pool
        self.exec_pool = []
        self.pools = ThreadPool(processes=4)

        # Flag
        self.event_status = False

    
    def check_folder(self, folder_path: str) -> None:
        if not os.path.exists(folder_path):        
            os.makedirs(folder_path)
        return folder_path

    def is_trigger(self) -> bool:
        return self.logic_event(self.current_value , self.threshold)

    def is_ready(self) -> bool:
        if (self.current_time - self.trigger_time) <= self.cooldown:
            print('Not ready, still have {} seconds'.format(round(self.current_time - self.trigger_time, 5)))
            return False
        self.trigger_time = self.current_time
        return True

    def get_pure_dets(self, detections: list):
        return [{
            "xmin": det.xmin,
            "ymin": det.ymin,
            "xmax": det.xmax,
            "ymax": det.ymax,
            "label": det.label,
            "score": det.score
        } for det in detections ]

    def get_pure_area(self, area:dict) -> dict:
        return {
            'name':area['name'],
            'area_point': area['area_point']
        }

    def save_image(self, save_path: str, image: np.ndarray) -> None:
        cv2.imwrite(save_path, image)

    def get_meta_data(self, pure_dets: list, pure_area: dict) -> dict:
        return {
            "detections": pure_dets,
            "area": pure_area
        }

    def save_meta_data(self, path: str, meta_data: dict) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=4)


    def event_behaviour(self, original: np.ndarray, meta_data: dict) -> dict:
        # timestamp is folder name
        timestamp = str(self.current_time)
        target_folder = self.uuid_folder
        # target_folder = self.check_folder(os.path.join(self.uuid_folder, timestamp))

        # Save Data: Image, Metadata
        image_path = os.path.join(target_folder, f"{timestamp}.jpg")
        data_path = os.path.join(target_folder, f"{timestamp}.json")

        self.pools.apply_async( self.save_image, args=(image_path, original))
        self.pools.apply_async( self.save_meta_data, args=(data_path, meta_data))
        
        if self.event_status:
            self.start_time = self.current_time
            self.end_time = None
        else:
            self.end_time = self.current_time

        return {
            "uid": self.uuid,
            "title": self.title,
            "timestamp": self.current_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "event_status": self.event_status,
            "meta": meta_data,
        }

    def _reset_timestamp(self):
        self.start_time = None
        self.end_time = None
        

    def __call__(self, original: np.ndarray, value: int, detections: list, area: dict) -> Union[None, dict]:
        # Update variable
        self.current_time = time.time_ns()
        self.current_value = value

        # Event triggered
        if self.is_trigger():
            # Event is on going
            if self.event_status: return

        # Event not triggered and event not started
        elif not self.event_status: 
            self._reset_timestamp()
            return

        # Update Status
        self.event_status = not self.event_status
        
        # Cooldown time: avoid trigger too fast
        if not self.is_ready(): return        

        print('Event {} ( Total: {})'.format('Start' if self.event_status else 'Stop', value))

        # get data
        pure_dets = self.get_pure_dets(detections)
        pure_area = self.get_pure_area(area)
        meta_data = self.get_meta_data(pure_dets=pure_dets, pure_area=pure_area)

        # Event
        return self.event_behaviour(
                original= original,
                meta_data= meta_data
        )

    def __del__(self):
        # Close threading pool iterminate
        self.pools.terminate()

# ------------------------------------------------------------------------    

class Detection_Zone(iAPP_OBJ):

    def __init__(self, params:dict, label:str, event_behavi_save_folder:str="event", palette:dict=palette):
        """_summary_

        Args:
            params (dict): _description_
            label (str): _description_
            event_save_folder (str, optional): _description_. Defaults to "event".
            palette (dict, optional): _description_. Defaults to palette.
        
        Workflows:
            1. Setup app_type.
            2. Initailize all parameters.
            3. Update palette and labels.
            4. Update each area informations into self.areas.
                * name
                * depend_on
                * norm_area_pts
                * output
        """
        # NOTE: Step 1. Define Application Type in __init__()
        self.app_type = 'obj' 
        
        # Initailize app params
        self.areas = []

        # Update setting
        self.params = params
        self.app_setting = self._get_app_setting(params)
        self.draw_area = self.app_setting.get("draw_area", True) 
        self.draw_bbox = self.app_setting.get("draw_bbox", True)
        self.draw_label = self.app_setting.get("draw_label", True)
        self.draw_result = self.app_setting.get("draw_result", True)
        
        # Event
        self.force_close_event = False

        # Palette and labels
        self.custom_palette = params.get("palette", {})
        self.palette, self.labels = update_palette_and_labels(
            custom_palette = self.custom_palette,
            default_palette = palette,
            label_path = label )
 
        # init draw tool that share the same palette and labels
        self.drawer = DrawTool(
            labels = self.labels,
            palette = self.palette
        )
       
        # NOTE: main parameter object
        self.areas = self.get_areas_setting(
            app_setting= self.app_setting,
            labels= self.labels
        )
        
        # helper
        self.frame_h, self.frame_w = None, None

    # ------------------------------------------------
    # Update parameter in __init__

    def _get_app_setting(self, input_params:dict) -> dict:
        """Check the input parameters is available.

        Args:
            input_params (dict): the input parameters.

        Raises:
            TypeError: the input parameter should be dict.
            KeyError: the input parameter must with key 'application'.
            TypeError: the application content should be dict ( input_params['application'] ).

        Returns:
            dict: the data of application.
        """


        #check config type
        if not isinstance( input_params, dict ):
            raise TypeError(f"The application parameter should be dict, but get {type(input_params)}")
        
        #check config key ( application ) is exist or not and the type is correct or not
        data =  input_params.get("application", None)
        if not data:
            raise KeyError("The app_config must have key 'application'.")
        if not isinstance( data, dict ):
            raise TypeError(f"The application information in application should be dict, but get {type(input_params)}")
        
        return data

    def _get_area_from_setting(self, app_setting: dict) -> list:
        areas  = app_setting.get("areas", None)
        if not areas:
            raise KeyError("Can not find areas key in application config.")
        if not isinstance(areas, list):
            raise TypeError("The expected type of the area setting shout be list.")
        return areas

    def _get_area_name(self, area_data: dict, default_name= "The defalt area") -> str:
        _name = area_data.get("name", None)

        if not _name:
            raise ValueError(f"Can not find 'name' in config.")

        elif _name == []:
            _name = default_name

        elif not isinstance(_name, str):
            raise TypeError("The type of area name should be str")

        return _name
    
    def _get_area_depend(self, area_data: dict, default_depend: list) -> list:
        # check has depend_on and type is correct
        _depend_on = area_data.get("depend_on", None)

        if not _depend_on:
            raise ValueError(f"Can not find 'depend_on' in area setting ")
        
        elif _depend_on==[]:
            _depend_on = default_depend

        elif not isinstance(_depend_on, list):
            raise TypeError("The type of area name should be list")

        return _depend_on

    def _get_norm_area_pts(self, area_data: dict, default_value: list= [[0,0],[1,0],[1,1],[0,1]]) -> list:
        """get normalized area point, if not setup means the whole area"""
        _area_point = area_data.get("area_point", None)

        if not _area_point:
            raise ValueError(f"Can not find 'area_point' in config.")
        
        elif _area_point == []:
            _area_point = default_value

        elif not isinstance(_area_point, list):
            raise TypeError("The type of area name should be list")

        return _area_point

    # NOTE: Init Event function
    def _get_event_obj(self, area_data: dict, event_func: EventHandler, drawer: DrawTool) -> Union[None, EventHandler]:
        events = area_data.get("events", None)
        if not events: return None 

        if not isinstance(area_data, dict):
            raise TypeError("Event setting should be dict.")
        
        event_obj = event_func(
            title = events["title"],
            folder = events.get("event_folder", "./events"),
            drawer = drawer,
            operator = events["logic_operator"],
            threshold = events["logic_value"],
            cooldown = events.get("cooldown", 15)
        )
        return event_obj

    # NOTE: Get Area Setting function
    def get_areas_setting(self, app_setting: dict, labels: list) -> List[dict]:
        """Parse each area setting and generate a new area setting with new items.

        Args:
            app_setting (dict): the intput app setting ( params['application'] )
            labels (list): the labels object

        Returns:
            List[dict]: the new area setting that include new key ( norm_area_pts, event )
        """
        # Parse each area data
        new_areas = []
        for area_data in self._get_area_from_setting(app_setting):
            new_areas.append({
                "name": self._get_area_name(area_data= area_data),
                "depend_on": self._get_area_depend(area_data= area_data, default_depend=labels),
                "norm_area_pts": self._get_norm_area_pts(area_data= area_data),
                "event": self._get_event_obj(area_data= area_data, drawer= self.drawer, event_func= EventHandler)
            })
        
        return new_areas

    def update_all_area_point(self, frame:np.ndarray, areas: list) -> None:
        """ Denormalize all area point. 

        Args:
            frame (np.ndarray): the input frame
            areas (list): the normalize area point
        """
        # denormalize area point and add new key into areas    
        # get shape
        h, w = frame.shape[:2]
        if self.frame_h == h and self.frame_w == w:
            return

        self.frame_h, self.frame_w = h, w
        
        def map_wraper(area):
            denorm_area_pts = denorm_area_points( self.frame_w, self.frame_h, area["norm_area_pts"])
            sorted_area_pts = sort_area_points(denorm_area_pts)
            area["area_point"] = sorted_area_pts

        # NOTE: map is faster than for loop
        # areas = list(map(map_wraper, areas))
        for area in areas:
            map_wraper(area)

        print("Updated area point !!!")
        
    def obj_in_area(self, px:int, py:int, poly:list) -> bool:
        """does the object in area or not

        Args:
            px (int): object center x
            py (int): object center y
            poly (list): area points

        Returns:
            bool: does object in area or not
        """
        is_in = False
        length = len(poly)

        for idx , corner in enumerate(poly):

            next_idx = idx + 1 if idx +1 < length else 0
            x1 , y1 = corner
            x2 , y2 = poly[next_idx]
            
            # if on the line
            if (x1 == px and y1 ==py) or (x2==px and y2 ==py):
                is_in = False
                break
            
            # if not in poly

            if min(y1,y2) <py <= max(y1 ,y2):
                
                x =x1+(py-y1)*(x2-x1)/(y2-y1)
                if x ==px:
                    is_in = False
                    break
                elif x > px:
                    is_in = not is_in
        
        return is_in

    def get_available_areas_and_update_output(self, label:str, xmin:int, ymin:int, xmax:int, ymax:int) -> list:
        """ parse all areas setting, check is label in depend_on and object in area,
        if available then update area["output"] 

        Args:
            label (str): the detected label
            xmin (int): the detected object position
            ymin (int): the detected object position
            xmax (int): the detected object position
            ymax (int): the detected object position

        Returns:
            list: the available areas
        """
        ret = []
        cnt_px, cnt_py = (xmin+xmax)//2, (ymin+ymax)//2
            
        for area_idx, area in enumerate(self.areas):
            area_pts = area["area_point"]
            depend = area["depend_on"]

            if label in depend and self.obj_in_area(cnt_px, cnt_py, area_pts):
                ret.append(area_idx)
                area["output"][label] += 1

        return ret

    # ------------------------------------------------

    def clear_app_output(self) -> None:
        """clear app output ( self.areas ) """
        for area in self.areas:
            area["output"] = defaultdict(int)

    # ------------------------------------------------

    # NOTE: __call__ function is requirements
    # @timeit
    def __call__(self, frame:np.ndarray, detections:list) -> Tuple[np.ndarray, list, list]:
        """
        1. Update basic parameters
        2. Parse detection and draw
        3. Check the event is trigger or not
        4. Return data
        """

        self.drawer.update_draw_params(frame=frame)
        self.update_all_area_point(frame=frame, areas=self.areas)
        self.clear_app_output()

        # Draw area
        original = frame.copy()
        overlay = frame
        
        if self.draw_area:
            overlay = self.drawer.draw_areas(overlay, self.areas )

        # Draw inference results
        for det in detections:
            
            # get parameters
            label, score = det.label, det.score
            xmin, ymin, xmax, ymax = \
                tuple(map(int, [ det.xmin, det.ymin, det.xmax, det.ymax ] ))            
            
            # get available areas and update area["output"]
            available_areas = self.get_available_areas_and_update_output(
                label, xmin, ymax, xmax, ymax)
            
            # no available area
            if len(available_areas)==0: continue
            
            # Draw bounding box and label
            color = self.drawer.get_color(label)
            if self.draw_bbox:
                self.drawer.draw_bbox(
                    overlay, [xmin, ymin], [xmax, ymax], color )
            
            if self.draw_label:
                info = f"{label}: {int(score*100)}%"
                self.drawer.draw_label(overlay, info, [xmin, ymin], color )
                

        # Draw app output
        if self.draw_result:
            self.drawer.draw_area_results(
                overlay, self.areas )

        # Combine all app output and get total object number
        app_output, total_obj_nums = [], 0
        event_output = []
        for area_idx, area in enumerate(self.areas):
            
            area_output = dict(area["output"])

            # Current area has nothing
            if not area_output: continue

            # Re-Combine new area output for v1.1 version
            new_area_output = []
            for label, num in area_output.items():
                new_area_output.append({
                    "label": label,
                    "num": num
                })

            # Combine app_output
            app_output.append({
                "id": area_idx,
                "name": area["name"],
                "data": new_area_output 
            })

            # Event
            total_obj_nums += sum(area_output.values())
            event = area.get("event", None)
            if self.force_close_event or not event: continue
            
            cur_output = event( original= original,
                                value= total_obj_nums,
                                detections= detections,
                                area= area )
            if not cur_output: continue

            event_output.append( cur_output )

        return overlay, {"areas": app_output}, {"event": event_output}


# ------------------------------------------------------------------------
# Test

def main():

    import logging as log
    import sys, cv2
    from argparse import ArgumentParser, SUPPRESS
    from typing import Union
    from ivit_i.io import Source, Displayer
    from ivit_i.core.models import iDetection
    from ivit_i.common import Metric

    def build_argparser():

        parser = ArgumentParser(add_help=False)
        basic_args = parser.add_argument_group('Basic options')
        basic_args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
        basic_args.add_argument('-m', '--model', required=True,
                        help='Required. Path to an .xml file with a trained model '
                            'or address of model inference service if using ovms adapter.')
        basic_args.add_argument('-i', '--input', required=True,
                        help='Required. An input to process. The input must be a single image, '
                            'a folder of images, video file or camera id.')
        available_model_wrappers = [name.lower() for name in iDetection.available_wrappers()]
        basic_args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                        type=str, required=True, choices=available_model_wrappers)
        basic_args.add_argument('-d', '--device', type=str,
                        help='Optional. `Intel` support [ `CPU`, `GPU` ] \
                                `Hailo` is support [ `HAILO` ]; \
                                `Xilinx` support [ `DPU` ]; \
                                `dGPU` support [ 0, ... ] which depends on the device index of your GPUs; \
                                `Jetson` support [ 0 ].' )

        model_args = parser.add_argument_group('Model options')
        model_args.add_argument('-l', '--label', help='Optional. Labels mapping file.', default=None, type=str)
        model_args.add_argument('-t', '--confidence_threshold', default=0.6, type=float,
                                    help='Optional. Confidence threshold for detections.')
        model_args.add_argument('--anchors', default=None, type=float, nargs='+',
                                    help='Optional. A space separated list of anchors. '
                                            'By default used default anchors for model. \
                                                Only for `Intel`, `Xilinx`, `Hailo` platform.')

        io_args = parser.add_argument_group('Input/output options')
        io_args.add_argument('-n', '--name', default='ivit', 
                            help="Optional. The window name and rtsp namespace.")
        io_args.add_argument('-r', '--resolution', type=str, default=None, 
                            help="Optional. Only support usb camera. The resolution you want to get from source object.")
        io_args.add_argument('-f', '--fps', type=int, default=None,
                            help="Optional. Only support usb camera. The fps you want to setup.")
        io_args.add_argument('--no_show', action='store_true',
                            help="Optional. Don't display any stream.")

        args = parser.parse_args()
        # Parse Resoltion
        if args.resolution:
            args.resolution = tuple(map(int, args.resolution.split('x')))

        return args
        
    # 1. Argparse
    args = build_argparser()

    # 2. Basic Parameters
    infer_metrx = Metric()

    # 3. Init Model
    model = iDetection(
        model_path = args.model,
        label_path = args.label,
        device = args.device,
        architecture_type = args.architecture_type,
        anchors = args.anchors,
        confidence_threshold = args.confidence_threshold )

    # 4. Init Source
    src = Source( 
        input = args.input, 
        resolution = (640, 480), 
        fps = 30 )

    # 5. Init Display
    if not args.no_show:
        dpr = Displayer( cv = True )

    # 6. Setting iApp
    app_config = {
        "application": {
            "palette": {
                "car": [ 105, 125, 105 ],
                "truck": [ 125, 115, 105 ]
            },
            "areas": [ 
                {
                    "name": "Area0",
                    "depend_on": [
                        "car", "truck"
                    ],
                    "area_point": [
                        [
                            0.256,
                            0.583
                        ],
                        [
                            0.658,
                            0.503
                        ],
                        [
                            0.848,
                            0.712
                        ],
                        [
                            0.356,
                            0.812
                        ]
                    ],
                    "events": {
                        "uid":"",
                        "title": "Traffic in Area 1 is very heavy",
                        "logic_operator": ">",
                        "logic_value": 3,
                    }
                },
                {
                    "name": "Area1",
                    "depend_on": [
                        "car",
                    ],
                    "area_point": [
                        [
                            0.256,
                            0.383
                        ],
                        [
                            0.538,
                            0.203
                        ],
                        [
                            0.268,
                            0.512
                        ],
                        [
                            0.456,
                            0.212
                        ]
                    ],
                    "events": {
                        "uid":"",
                        "title": "GGGGG",
                        "logic_operator": ">",
                        "logic_value": 3,
                    }
                }
            ]
        }
    }
    app = Detection_Zone(app_config ,args.label)

    # 7. Start Inference
    try:
        while True:
            # Get frame & Do infernece
            frame = src.read()
            
            results = model.inference(frame=frame)
            frame , app_output , event_output =app(frame,results)
            # infer_metrx.paint_metrics(frame)
            
            # Draw FPS: default is left-top                     
            dpr.show(frame=frame)

            # Display
            if dpr.get_press_key() == ord('+'):
                model.set_thres( model.get_thres() + 0.05 )
            elif dpr.get_press_key() == ord('-'):
                model.set_thres( model.get_thres() - 0.05 )
            elif dpr.get_press_key() == ord('q'):
                break

            # Update Metrix
            infer_metrx.update()

    except KeyboardInterrupt:
        log.info('Detected Key Interrupt !')

    finally:
        model.release()
        src.release()
        dpr.release()

if __name__=='__main__':
    """
    python3 apps/Detection_Zone.py \
    -m model/yolo-v3-tf/yolo-v3-tf.xml \
    -l model/yolo-v3-tf/coco.names \
    -i data/car.mp4 -at yolo
    """
    main()