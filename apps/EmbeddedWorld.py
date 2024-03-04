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
        sort_area_points
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
        sort_area_points
    )
    from apps.drawer import DrawTool
    from apps.event_handler import EventHandler

# Parameters
# FRAME_SCALE     = 0.0005    # Custom Value which Related with Resolution
# BASE_THICK      = 1         # Setup Basic Thick Value
# BASE_FONT_SIZE  = 0.5   # Setup Basic Font Size Value
# FONT_SCALE      = 0.2   # Custom Value which Related with the size of the font.
# WIDTH_SPACE = 10
# HEIGHT_SPACE = 10
FONT_TYPE = 5-1
LINE_TYPE = cv2.LINE_AA

# FONT_SIZE = 2
# FONT_COLOR = (0, 255, 255)
# BOX_COLOR = (0, 0, 0)

# ------------------------------------------------------------------------    

def custom_logic_event(value: int, base: int = 5):
    return value%base==0


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

        self.output = defaultdict(int)
        self.tracking_obj = defaultdict(int)


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


class EmbeddedWorld(iAPP_OBJ):

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

        # TEMP
        self.prev_area_output = {}

        # NOTE: NEW
        self.current_status = ""
        self.pass_fail_info = None

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
            cooldown = events.get("cooldown", 5)
        )

        event_obj.logic_event = custom_logic_event
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
            denorm_area_pts = denorm_area_points( self.frame_w, self.frame_h, area["norm_area_point"])
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

                # print(areas_tracking_data[area_id][label])
        
        return areas_tracking_data
        
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

        pass_fail_info = self.update_pass_fail_params(
            event_data["area"]
        )

        # NOTE: Draw current
        font_size =  self.drawer.font_size
        font_thick = int(font_size*2)
        pass_color, fail_color = (0,255,0), (0,0,255)
        bg_color = (0,0,0) 
        pass_text , fail_text = \
            f" {'Pass'}: {pass_fail_info['Pass']:03}  ", f" {'Fail'}: {pass_fail_info['Fail']:03}  "

        # Get size
        (p_wid, p_hei), t_base = cv2.getTextSize(pass_text, FONT_TYPE, font_size, font_thick)
        box_width = max(p_wid, cv2.getTextSize(fail_text, FONT_TYPE, font_size, font_thick)[0][0])
        
        # Get position
        p_xmin, p_ymin = 20, 20
        p_xmax, p_ymax = p_xmin + p_wid, p_ymin + p_hei + t_base
        f_xmin, f_ymin = 20, p_ymax + p_hei
        f_xmax, f_ymax = f_xmin + p_wid, f_ymin + p_hei + t_base
        
        # Background
        result = overlay.copy()
        cv2.rectangle( result, (p_xmin, p_ymin), (p_xmax, p_ymax+t_base), bg_color, -1)
        cv2.rectangle( result, (f_xmin, f_ymin), (f_xmax, f_ymax+t_base), bg_color, -1)
        
        # Right- Top
        # font_size = 3
        # pad = 20
        # (r_wid, r_hei), r_base = cv2.getTextSize("Pass", FONT_TYPE, font_size, font_size+1)
        # r_xmin, r_ymin = frame.shape[1] - r_wid - pad , pad
        # r_xmax, r_ymax = r_xmin + r_wid, pad + r_hei + r_base + pad
        # cv2.rectangle( result, (r_xmin, r_ymin), (r_xmax, r_ymax), BOX_COLOR , -1)

        # Add Weight
        opacity = 0.8
        overlay = cv2.addWeighted( overlay, 1-opacity, result, opacity, 0 ) 
        
        # Frontend
        for text, pos, color in zip([ pass_text, fail_text], [(p_xmin, p_ymax), (f_xmin, f_ymax)], [pass_color, fail_color]):
            cv2.putText( overlay, text, pos, FONT_TYPE,
                font_size, color, font_thick, LINE_TYPE )

        if self.current_status != "":
            font_size = 4
            pad = 20
            color = (0, 255, 0) if self.current_status == "Pass" else (0, 0, 255)
            (r_wid, r_hei), r_base = cv2.getTextSize(self.current_status, FONT_TYPE, font_size, font_thick)
            r_xmin, r_ymin = frame.shape[1]//2 - r_wid//2 , pad + r_hei + r_base
            cv2.putText( overlay, f"{self.current_status}", (r_xmin + pad//2, r_ymin ), FONT_TYPE,
                font_size, (255,255,255), font_thick, LINE_TYPE )
            cv2.putText( overlay, f"{self.current_status}", (r_xmin + pad//2, r_ymin ), FONT_TYPE,
                font_size, color, font_thick, LINE_TYPE )

        return overlay
    

    def get_storage_tag(self, label: str) -> bool:
        sd_pass = [ "front_left" ]
        # cfast_pass = [ "left" ]
        return label in (sd_pass )


    def update_pass_fail_params(self, area_output: dict):
        
        outs = area_output.get("output", None )
        if not outs: return
        
        # SD Card, Cfast
        cur_pass, cur_fail = 0, 0
        for label, nums in outs.items():

            if self.get_storage_tag(label):
                cur_pass += nums
            else:
                cur_fail += nums

        ret = {
            "Pass": cur_pass,
            "Fail": cur_fail
        }
        return ret

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
        self.clear_app_output()

        # Draw area
        original = frame.copy()
        overlay = frame
        
        if self.draw_area:
            overlay = self.drawer.draw_areas(overlay, self.areas )

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
        self.current_status = ""
        for area_idx, tracking_data in tracking_dets.items():
            
            new_area_output = []
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
                        "score": float(detections[idx_in_dets].score),
                        "tracked_idx": tracked_idx,
                    })
                
                # NOTE: Update self.current_status
                self.current_status = "Pass" if self.get_storage_tag(label) else "Fail"
                    
                # Update tracking label numbers in each area and send to event.
                # Re-Combine new area output for v1.1 version
                total_label_nums = trg_label_tracker.get_total_nums(label)
                self.areas[area_idx]["output"][label] = total_label_nums

                # Combine label informations
                # new_area_output.append({
                #     "labelssssss": label,
                #     "nums213123213123": total_label_nums,
                #     "asdsadasd": 123213123
                # })


            # Combine output
            self.pass_fail_info = self.update_pass_fail_params(self.areas[area_idx])
            app_output.append({
                "id": area_idx,
                "name": self.areas[area_idx]["name"],
                "data": [ self.pass_fail_info ]
            })

            # Trigger event
            if save_tracked_dets:
                event = self.areas[area_idx].get("event", None)
                if self.force_close_event or not event: continue
                cur_output = event( original= original,
                                    value= self.pass_fail_info["Fail"],
                                    detections= save_tracked_dets,
                                    area= self.areas[area_idx] )
                if not cur_output: continue
                event_output.append( cur_output )

        # Update prev_area
        if len(tracking_dets)!=0:
            self.prev_area_output = self.copy_area_output()

        if self.pass_fail_info:

            font_size =  self.drawer.font_size
            font_thick = int(font_size*2)
            

            # NOTE: Draw current
            pass_color, fail_color = (0,255,0), (0,0,255)
            bg_color = (0,0,0) 
            pass_text , fail_text = \
                f" {'Pass'}: {self.pass_fail_info['Pass']:03}  ", f" {'Fail'}: {self.pass_fail_info['Fail']:03}  "

            # Get size
            (p_wid, p_hei), t_base = cv2.getTextSize(pass_text, FONT_TYPE, font_size, font_thick)
            box_width = max(p_wid, cv2.getTextSize(fail_text, FONT_TYPE, font_size, font_thick)[0][0])
            
            # Get position
            p_xmin, p_ymin = 20, 20
            p_xmax, p_ymax = p_xmin + p_wid, p_ymin + p_hei + t_base
            f_xmin, f_ymin = 20, p_ymax + p_hei
            f_xmax, f_ymax = f_xmin + p_wid, f_ymin + p_hei + t_base
            
            # Background
            result = overlay.copy()
            cv2.rectangle( result, (p_xmin, p_ymin), (p_xmax, p_ymax+t_base), bg_color, -1)
            cv2.rectangle( result, (f_xmin, f_ymin), (f_xmax, f_ymax+t_base), bg_color, -1)
            
            # Add Weight
            opacity = 0.8
            overlay = cv2.addWeighted( overlay, 1-opacity, result, opacity, 0 ) 
            
            # Frontend
            for text, pos, color in zip([ pass_text, fail_text], [(p_xmin, p_ymax), (f_xmin, f_ymax)], [pass_color, fail_color]):
                cv2.putText( overlay, text, pos, FONT_TYPE,
                    font_size, color, font_thick, LINE_TYPE )

            if self.current_status != "":
                font_size *= 4
                font_thick = int(font_size+1)
                pad = 20
                color = (0, 255, 0) if self.current_status == "Pass" else (0, 0, 255)
                (r_wid, r_hei), r_base = cv2.getTextSize(self.current_status, FONT_TYPE, font_size, font_thick)
                r_xmin, r_ymin = frame.shape[1]//2 - r_wid//2 , pad + r_hei
                cv2.putText( overlay, f"{self.current_status}", (r_xmin + pad//2, r_ymin ), FONT_TYPE,
                    font_size, (255,255,255), font_thick*3, LINE_TYPE )
                cv2.putText( overlay, f"{self.current_status}", (r_xmin + pad//2, r_ymin ), FONT_TYPE,
                    font_size, color, font_thick, LINE_TYPE )

        return overlay, {"areas": app_output}, {"event": event_output}
