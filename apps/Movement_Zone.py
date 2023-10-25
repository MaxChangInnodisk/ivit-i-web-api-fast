import os, cv2, time
import numpy as np
import uuid
import os
from typing import Any, Union, get_args
from datetime import datetime
from apps.palette import palette
from ivit_i.common.app import iAPP_OBJ
import os
import numpy as np
import time
from filterpy.kalman import KalmanFilter
from typing import Tuple, List
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import json

# iVIT-I 
from ivit_i.common.app import iAPP_OBJ

# App utilities
try:
    from .palette import palette
    from .utils import ( 
        timeit, get_time,
        update_palette_and_labels,
        get_logic_map,
        denorm_area_points,
        sort_area_points,
        denorm_line_points
    )
    from .drawer import DrawTool
    from .event_handler import EventHandler

except:
    from apps.palette import palette
    from apps.utils import ( 
        timeit, get_time,
        update_palette_and_labels,
        get_logic_map,
        denorm_area_points,
        sort_area_points,
        denorm_line_points
    )
    from apps.drawer import DrawTool
    from apps.event_handler import EventHandler

# ------------------------------------------------------------------------    

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    
    def __init__(self,bbox, count,label):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = count
    
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.current_idx=None
        self.label = label 

    def convert_x_to_bbox(self,x,score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

    def convert_bbox_to_z(self,bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    #scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)


class TrackerHanlder(object):

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.changeable_total = 0
        self.frame_count = 0

        self.tracked = defaultdict(int)
        self.next_idx = 0
  
    def iou_batch(self,bb_test, bb_gt):
        """
        From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
        return(o)  

    def linear_assignment(self,cost_matrix):
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i],i] for i in x if i >= 0]) #
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))

    def _count_object_num(self, label:str):
        self.tracked[label]+=1
        
    def _asign_new_id(self, label: str):
        return self.tracked[label]
    
    def get_total_nums(self, label: str) -> int:
        return self.tracked[label]

    def associate_detections_to_trackers(self,detections,trackers,iou_threshold = 0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    
        iou_matrix = self.iou_batch(detections, trackers)
    
        if min(iou_matrix.shape) > 0:

            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self.linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0,2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        #filter out matched with low IOU
        matches = []

        for m in matched_indices:
            if(iou_matrix[m[0], m[1]]<iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, dets, dets_label):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        
        self.frame_count += 1
        
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):

            # update position
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            
            # the position have to delete
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # find the index of invalid value
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
    
        matched, unmatched_dets, unmatched_trks = \
            self.associate_detections_to_trackers(dets, trks, self.iou_threshold)
   
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])
            self.trackers[m[1]].current_idx = m[0]
        
        # create and initialise new trackers for unmatched detections
    
        for i in unmatched_dets:
            # Count new object
            self._count_object_num(dets_label)
            
            # Add new object to track
            trk = KalmanBoxTracker(dets[i],self._asign_new_id(dets_label),dets_label)
            self.trackers.append(trk)
            self.trackers[-1].current_idx = i
            self.changeable_total+=1

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id],[trk.current_idx])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if(len(ret)>0):
            return np.concatenate(ret)
        
        return np.empty((0,5))

# ------------------------------------------------------------------------    

class Movement_Zone(iAPP_OBJ):

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
                * norm_area_point
                * output
        """
        # NOTE: Step 1. Define Application Type in __init__()
        self.app_type = 'obj' 
        
        # Initailize app params
        self.areas = []

        # Update setting
        self.params = params
        self.app_setting = self._get_app_setting(params)
        self.draw_area = self.app_setting.get("draw_area", False) 
        self.draw_bbox = self.app_setting.get("draw_bbox", True)
        self.draw_label = self.app_setting.get("draw_label", True)
        self.draw_result = self.app_setting.get("draw_result", True)
        self.draw_line = self.app_setting.get("draw_line", False)
        
        
        # Event
        self.force_close_event = False

        # Palette and labels
        self.custom_palette = self.app_setting.get("palette", {})
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

        # NOTE: Add tracker
        # self.mot_trackers= self._creat_mot_tracker_for_each_area()
        self.mot_trackers = self._ger_tracker_in_each_areas(self.areas)

        # NOTE: Movement
        self.tracked_status = defaultdict( lambda: defaultdict(dict) )

        # TEMP
        self.prev_area_output = {}

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

    def _get_norm_line_pts(sefl, area_data: dict) -> dict:
        
        _line_point = area_data.get("line_point", None)

        if not _line_point:
            raise ValueError(f"Can not find 'line_point' in config.")
        
        elif not isinstance(_line_point, dict):
            raise TypeError("The type of area name should be list")

        return _line_point

    def _get_line_relation(self, area_data: dict) -> list:
        _line_relation = area_data.get("line_relation", None)

        if not _line_relation:
            raise ValueError(f"Can not find 'line_relation' in config.")
        
        elif not isinstance(_line_relation, list):
            raise TypeError("The type of area name should be list")

        # Combine the line name
        ret_line_relation = defaultdict()
        for _relation in _line_relation:
            new_flag = f"{_relation['start']}+{_relation['end']}"
            ret_line_relation[new_flag] = _relation['name']

        print('Create new relation table: ', ret_line_relation)
        return ret_line_relation
    
    def _get_output_placeholder(self, area_data: dict) -> dict:
        _line_relation = area_data.get("line_relation", None)

        if not _line_relation:
            raise ValueError(f"Can not find 'line_relation' in config.")
        
        elif not isinstance(_line_relation, list):
            raise TypeError("The type of area name should be list")

        output = {}
        for _relation in _line_relation:
            output[_relation["name"]] = 0
        
        return output

    # NOTE: Init Event function
    def _get_event_obj(self, area_data: dict, event_func: EventHandler, drawer: DrawTool) -> Union[None, EventHandler]:
        events = area_data.get("events", None)
        if not events: return None 

        if not isinstance(area_data, dict):
            raise TypeError("Event setting should be dict.")

        if "<" in events["logic_operator"]:
            raise TypeError("Tracking, Movement not support \"<\" operator !")

        event_obj = event_func(
            title = events["title"],
            folder = events.get("event_folder", "./events"),
            drawer = drawer,
            operator = events["logic_operator"],
            threshold = events["logic_value"],
            cooldown = events.get("cooldown", 5)
        )
        return event_obj

    # NOTE: Get Area Setting function
    def get_areas_setting(self, app_setting: dict, labels: list) -> List[dict]:
        """Parse each area setting and generate a new area setting with new items.

        Args:
            app_setting (dict): the intput app setting ( params['application'] )
            labels (list): the labels object

        Returns:
            List[dict]: the new area setting that include new key ( norm_area_point, event )
        """
        # Parse each area data
        new_areas = []
        for area_data in self._get_area_from_setting(app_setting):
            new_areas.append({
                "name": self._get_area_name(area_data= area_data),
                "depend_on": self._get_area_depend(area_data= area_data, default_depend=labels),
                "norm_area_point": self._get_norm_area_pts(area_data= area_data),
                "norm_line_point": self._get_norm_line_pts(area_data= area_data),
                "line_relation": self._get_line_relation(area_data= area_data),
                "output": self._get_output_placeholder(area_data= area_data),
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
        
        for area in areas:
            # area
            denorm_area_pts = denorm_area_points( self.frame_w, self.frame_h, area["norm_area_point"] )
            sorted_area_pts = sort_area_points(denorm_area_pts)
            area["area_point"] = sorted_area_pts

            # line
            area["line_point"] = denorm_line_points( self.frame_w, self.frame_h, area["norm_line_point"] )

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

        return ret

    # ------------------------------------------------

    def _ger_tracker_in_each_areas(self, areas):

        trackers = defaultdict( lambda: defaultdict() )
        
        for area_idx, area in enumerate(areas):
            
            for label in area["depend_on"]:
                
                trackers[area_idx][label] = TrackerHanlder(
                    max_age=1,
                    min_hits=3,
                    iou_threshold=0.3
                )

        return trackers

    def _get_tracking_data_and_sorter(self, detections:list):
        
        areas_tracking_data = defaultdict(lambda: defaultdict(list))
        
        # Draw inference results
        for det in detections:
            
            # get parameters
            label, score = det.label, det.score
            xmin, ymin, xmax, ymax = \
                tuple(map(int, [ det.xmin, det.ymin, det.xmax, det.ymax ] ))            
            
            # get available areas and update area["output"]
            available_areas = self.get_available_areas_and_update_output(
                label, xmin, ymin, xmax, ymax)
            
            # no available area
            if len(available_areas)==0: continue
            
            # parse each area
            for area_id in available_areas:  
                areas_tracking_data[area_id][label].append( 
                    [xmin, ymin, xmax, ymax, score] )
        
        return areas_tracking_data

    # ------------------------------------------------

    def is_point_cross_trigger_line( self,
            current_center_point: tuple,
            now_center_point: tuple,
            line_point: dict ):
        """
            Judge the object whether cross the trigger line or not.
        Args:
            current_center_point (tuple): the object center point at last time.
            now_center_point (tuple): the object center point now.
            line_point (dict): trigger line . format is { area_id : {'line_1': [[704, 692], [1292, 573]], 'line_2': [[692, 900], [1400, 681]]}}
            area_id (int): area_id.

        Returns:
            _type_: _description_
        """
        temp_line_name = ""
        cross_flag=False
        
        for line_name,trigger_point in line_point.items():
            
            # print(" area total{} ,p1 :({},{}) , p2 :({},{})".format(trigger_point,trigger_point[0][0],trigger_point[0][1],trigger_point[1][0],trigger_point[1][1]))
            trigger_line=np.array([trigger_point[0][0]-trigger_point[1][0],trigger_point[0][1]-trigger_point[1][1]])
            trigger_line_side=np.array([trigger_point[0][0]-current_center_point[0],trigger_point[0][1]-current_center_point[1]])
            trigger_line_side2=np.array([trigger_point[0][0]-now_center_point[0],trigger_point[0][1]-now_center_point[1]])

            trigger_cross=np.cross(trigger_line,trigger_line_side)*np.cross(trigger_line,trigger_line_side2)
            
            if trigger_cross<0:
                temp_line_name=line_name
                cross_flag=True

        # if len(temp)>1:
        #     return True ,  [0]
        
        return cross_flag ,temp_line_name

    def _check_cross_trigger_line( 
            self, 
            area_id:int, 
            label:str, 
            tracking_tag:int, 
            center_point:tuple ):
        """
        self.tracked_status = defaultdict( lambda: defaultdict(dict) )

        self.tracked_status[area_id][label][tracked_idx] = {
            'center_point':center_point,
            'cross_line':[],
        }

        """
        
        if tracking_tag not in self.tracked_status[area_id][label]:
            self.tracked_status[area_id][label][tracking_tag] = {
                'center_point':center_point,
                'cross_line':[]
            }
            return
        
        tracked_obj = self.tracked_status[area_id][label][tracking_tag]
        
        is_cross, cross_line = \
            self.is_point_cross_trigger_line(
                tracked_obj["center_point"],
                center_point,
                self.areas[area_id]["line_point"]
            )
        
        # NOTE: update center_point
        tracked_obj["center_point"] = center_point

        # Condition: not corss or already crocessed
        if not is_cross or cross_line in tracked_obj["cross_line"]: return
        
        # Append new line
        tracked_obj["cross_line"].append(cross_line)

        # Only two line to track
        if len(tracked_obj["cross_line"])!=2: return 
        
        # Get flag & title
        cur_flag = "+".join(tracked_obj["cross_line"])
        cur_title = self.areas[area_id]["line_relation"].get(cur_flag)
        if not cur_title:
            return
        
        # Update output
        self.areas[area_id]["output"][cur_title] += 1
        
        # Clear object
        print()

    # ------------------------------------------------

    def clear_app_output(self) -> None:
        """clear app output ( self.areas ) """
        if self.prev_area_output != {}:
            return
        for area in self.areas:
            area["output"] = defaultdict(int)
            for label in area["depend_on"]:
                area["output"][label]+=0

    def copy_area_output(self) -> list:
        copy_areas = list()
        for area in self.areas:
            copy_areas.append({
              "name": area["name"],
              "output": area["output"]
            })
        return copy_areas    

    # ------------------------------------------------
    # Drawing Meta Data ( Tracking data )

    def draw_event_data(self, frame:np.ndarray, event_data:dict) -> np.ndarray:
        """ Draw Tracking Data from meta data
        1. Update Area Points
        2. Bounding Box with Tracking ID.
        3. Application Output.
        4. Return Draw Image
        """
        
        self.drawer.update_draw_params(frame=frame)
        self.update_all_area_point(frame=frame, areas=self.areas)
        self.clear_app_output()

        overlay = frame
        
        # Combine all app output and get total object number
        detections = event_data["detections"]
        areas = [ event_data["area"] ]
        
        # Get Detections with tracked_idx
        for det in detections:
            [ xmin, ymin, xmax, ymax, label, tracked_idx ] = \
                [ det[key] for key in [ "xmin", "ymin", "xmax", "ymax", "label", "tracked_idx" ] ]

            color = self.drawer.get_color(label)

            self.drawer.draw_bbox(
                overlay, [xmin, ymin], [xmax, ymax], color )
            
            self.drawer.draw_label(
                overlay, f"{label}: {tracked_idx:03}", (xmin, ymin), color )

        # Get Output
        self.drawer.draw_area_results(
            overlay, areas )

        return overlay

    # ------------------------------------------------

    @get_time
    def test(self,frame:np.ndarray, detections:list):
        return self.__call__(frame, detections)

    # NOTE: __call__ function is requirements
    def __call__(self, frame:np.ndarray, detections:list) -> Tuple[np.ndarray, list, list]:
        """
        1. Update basic parameters
        2. Parse detection and draw
        3. Check the event is trigger or not
        4. Return data
        """
        self.drawer.update_draw_params(frame=frame)
        self.update_all_area_point(frame=frame, areas=self.areas)

        # Draw area
        original = frame.copy()
        overlay = frame
        
        if self.draw_area:
            overlay = self.drawer.draw_areas(overlay, self.areas )

        if self.draw_line:
            for area in self.areas:
                for name, point in area["line_point"].items():
                    self.drawer.draw_line(overlay, point, name )

        """
        # NOTE: NEW!!!!
        修改 Area 的資訊才能讓 tracker 接到
        
        """
        # Combine all app output and get total object number
        app_output = []
        event_output = []

        # Get the tracking data ( which is different with detections )
        tracking_dets = self._get_tracking_data_and_sorter(detections)

        # track each areas
        for area_idx, tracking_data in tracking_dets.items():

            save_tracked_dets = []

            # track each labels
            for label, tracking_points in tracking_data.items():
                
                trg_label_tracker = self.mot_trackers[area_idx][label]
                
                # get the result of tracking object
                tracking_result = trg_label_tracker.update(
                    tracking_points, label)

                # Draw results
                color = self.drawer.get_color(label)
                for result in tracking_result:

                    xmin, ymin, xmax, ymax, tracked_idx, idx_in_dets = \
                        map(int, map(float, result))

                    if self.draw_bbox:                    
                        self.drawer.draw_bbox(
                            overlay, [xmin, ymin], [xmax, ymax], color )
                        
                    if self.draw_label:
                        self.drawer.draw_label(
                            overlay, f"{label}: {tracked_idx:03}", (xmin, ymin), color )

                    save_tracked_dets.append({
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "label": label,
                        "score": detections[idx_in_dets].score,
                        "tracked_idx": tracked_idx,
                    })

                    # NOTE: Movement
                    self._check_cross_trigger_line(
                        area_idx,
                        label,
                        tracked_idx,
                        ((xmin+xmax)//2,(ymin+ymax)//2))

            new_area_output = []
            for area in self.areas:
                for title, nums in area["output"].items():
                    new_area_output.append({ 
                        "label": title,
                        "nums": nums 
                    })
                        
        
            # Combine output
            app_output.append({
                "id": area_idx,
                "name": self.areas[area_idx]["name"],
                "data": new_area_output
            })

            # Trigger event
            event = self.areas[area_idx].get("event", None)
            if self.force_close_event or not event: continue
            cur_output = event( original= original,
                                value= sum([ item["nums"] for item in new_area_output ]),
                                detections= save_tracked_dets,
                                area= self.areas[area_idx] )
            if not cur_output: continue
            event_output.append( cur_output )

        # Update prev_area
        if len(tracking_dets)!=0:
            self.prev_area_output = self.copy_area_output()

        # Draw app output
        if self.draw_result:
            self.drawer.draw_area_results(
                overlay, self.prev_area_output )

        return (overlay, {"areas": app_output}, {"event": event_output})

# ------------------------------------------------------------------------    

if __name__=='__main__':
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
        model_args.add_argument('-t', '--confidence_threshold', default=0.4, type=float,
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
                    "car": [
                        105,
                        125,
                        105
                    ],
                    "truck": [
                        125,
                        115,
                        105
                    ]
                },
                "areas": [
                    {
                        "name": "Area0",
                        "depend_on": [
                            'car'
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
                        "line_point": {
                            "line_1": [
                                [
                                    0.26666666666,
                                    0.68074074074
                                ],
                                [
                                    0.67291666666,
                                    0.58962962963
                                ]
                            ],
                            "line_2": [
                                [
                                    0.26041666666,
                                    0.70333333333
                                ],
                                [
                                    0.72916666666,
                                    0.60062962963
                                ]
                            ]
                        },
                        "line_relation": [
                            {
                                "name": "to Taipei",
                                "start": "line_2",
                                "end": "line_1"
                            },
                            {
                                "name": "To Keelung",
                                "start": "line_1",
                                "end": "line_2"
                            }
                        ],
                        # "events": {
                        #         "uid":"cfd1f399",
                        #         "title": "Detect the traffic flow between Taipei and Xi Zhi ",
                        #         "logic_operator": ">",
                        #         "logic_value": 1,
                                
                        #     }
                    },
                    # {
                    #             "name": "second area",
                    #             "depend_on": [
                    #                 "car",
                    #             ],
                    #             "area_point": [
                    #                  [
                    #         0.468,
                    #         0.592
                    #     ],
                        
                    #     [
                    #         0.468,
                    #         0.203
                    #     ],
                        
                    #     [
                    #         0.156,
                    #         0.592
                    #     ],
                    #     [
                    #         0.156,
                    #         0.203
                    #     ]
                    #             ],
                    #             "line_point": {
                    #         "line_1": [
                    #             [
                    #                 0.16666666666,
                    #                 0.74074074074
                    #             ],
                    #             [
                    #                 0.57291666666,
                    #                 0.62962962963
                    #             ]
                    #         ],
                    #         "line_2": [
                    #             [
                    #                 0.26041666666,
                    #                 0.83333333333
                    #             ],
                    #             [
                    #                 0.72916666666,
                    #                 0.62962962963
                    #             ]
                    #         ],
                    #     },
                    #     "line_relation": [
                    #         {
                    #             "name": "Wrong Direction",
                    #             "start": "line_2",
                    #             "end": "line_1"
                    #         }
                    #     ],
                    #         }
                ],
                "draw_result":True,
                "draw_bbox":True,
                "draw_line":True,
                "draw_area": True
            }
        }
    app = Movement_Zone(app_config,args.label )

    # 7. Start Inference
    try:
        t_costs = []
        while True:
            # Get frame & Do infernece
            frame = src.read()
            
            results = model.inference(frame=frame)
          
            # frame , app_output , event_output = app(frame,results)
            (frame , app_output , event_output), t_cost = app.test(frame,results)
            t_costs.append(t_cost)
            if len(t_costs)==500:
                print('\n\n\n')
                print('Average Cost Time: ', round(sum(t_costs)/500*1000, 5))
                break

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

    except Exception as e:
        print ('error: %s: %r' % (e.strerror, e.filename))

    finally:
        model.release()
        src.release()
        dpr.release()

"""
python3 apps/Movement_Zone.py \
-m model/yolo-v3-tf/yolo-v3-tf.xml \
-l model/yolo-v3-tf/coco.names \
-i data/car.mp4 -at yolo -d GPU
"""
