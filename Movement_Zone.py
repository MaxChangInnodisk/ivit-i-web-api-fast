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
from filterpy.kalman import KalmanFilter

class event_handle(threading.Thread):
  def __init__(self ,operator:dict,thres:dict,cooldown_time:dict,event_title:dict,area_id:int,event_save_folder:str,uid:str=None):
    threading.Thread.__init__(self)
    self.operator = operator
    self.thres = thres
    self.cooldown_time = cooldown_time
    self.event_title = event_title
    self.event_output={}
    self.area_id =area_id
    self.pass_time = self.cooldown_time+1
    self.event_time=datetime.now()
    self.trigger_time=datetime.now()        
    self.info =" "
    self.event_save_folder=event_save_folder
    self.uid=uid

  def get_logic_event(self, operator):
    """ Define the logic event """
    greater = lambda x,y: x>y
    greater_or_equal = lambda x,y: x>=y
    less = lambda x,y: x<y
    less_or_equal = lambda x,y: x<=y
    equal = lambda x,y: x==y
    logic_map = {
        '>': greater,
        '>=': greater_or_equal,
        '<': less,
        '<=': less_or_equal,
        '=': equal,
    }
    return logic_map.get(operator)

  def logic_event(self, value,thres):
    return self.operator(value,thres)    

  def __call__(self,frame,ori_frame,area_id,total_object_number,app_output):
    self.event_output={}
    self.event_time=datetime.now()
    self.info =""
    # area have operator
    # if self.operator.__contains__(area_id) == False:
    #   return
    
    # when object in area 
    # if total_object_number.__contains__(area_id) == False: 
    #   return
    
    if self.logic_event(total_object_number,self.thres) == False:
      return
    
    
    if self.pass_time > self.cooldown_time:
        
      self.eventflag=True
      self.trigger_time=datetime.now()
      self.pass_time = (int(self.event_time.minute)*60+int(self.event_time.second))-(int(self.trigger_time.minute)*60+int(self.trigger_time.second))
      
      uid=self.uid[area_id] if not (self.uid[area_id]==None) else str(uuid.uuid4())[:8]
      path='./'+self.event_save_folder+'/'+str(uid)+'/'+str(time.time())+'/'
      if not os.path.isdir(path):
          os.makedirs(path)
      cv2.imwrite(path+'original.jpg', frame)
      cv2.imwrite(path+'overlay.jpg', ori_frame)
      self.event_output.update({"uuid":uid,"title":self.event_title,"areas":app_output["areas"],\
                                "timesamp":self.trigger_time,"screenshot":{"overlay": path+str(self.trigger_time)+'.jpg',
      "original": path+str(self.trigger_time)+"_org"+'.jpg'}}) 
      # Draw Inforamtion
      
      self.info = "The {} area : ".format(area_id)+self.event_title+\
        ' , '.join([ 'total:{}  , cool down time:{}/{}'.format(total_object_number,0,self.cooldown_time)])
      
    else :
        
      self.pass_time = (int(self.event_time.minute)*60+int(self.event_time.second))-(int(self.trigger_time.minute)*60+int(self.trigger_time.second))    
      self.info = "The {} area : ".format(area_id)+self.event_title+\
        ' , '.join([ 'total:{}  , cool down time:{}/{}'.format(total_object_number,self.pass_time,self.cooldown_time)])

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

class Sort(object):

  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3,app_output_data:dict=None,area_id:int=None):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.area_id = area_id
    self.app_output_data = app_output_data[self.area_id]
    self.changeable_total = 0
    self.frame_count = 0
  
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

  def _count_object_num(self,label:str):
    """
      Get the max number as total tracking number.
    Args:
      track_object (tuple): format is (label , tracking tag). 
      area_id(int):area id.
    """
    for id,label_info in enumerate(self.app_output_data):
       if label_info['label']==label:
          self.app_output_data[id]['num'] += 1

  def _asign_new_id(self,label:str):
    for id,label_info in enumerate(self.app_output_data):
      if label_info['label']==label:
        return label_info['num']


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

  def update(self, dets,dets_label):
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
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    
    matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets,trks, self.iou_threshold)
   
    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0]])
      self.trackers[m[1]].current_idx = m[0]
      
    # create and initialise new trackers for unmatched detections
    
    for i in unmatched_dets:
        self._count_object_num(dets_label)

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

class Movement_Zone(iAPP_OBJ, event_handle):

  def __init__(self, params:dict, label:str,event_save_folder:str="event", palette:dict=palette):
      self.app_type = 'obj'

      #this step will check application and areas whether with standards or not .
      self.params = self._check_params(params)

      self.palette={}
      self.label_path = label
      self.label_list =[]
      
      # Update each Variable
      self._init_palette(palette)

      #put here reason is in there we need self.label_list but self.label_list get value after _init_palette
      self.depend_on , self.app_output_data = self._get_depend_on()

      self._init_draw_params()
      self._update_area_params()
      self._init_event_param(event_save_folder)
      self._update_event_param()
      self._init_event_object()
      self.MOT_tracker= self._creat_MOT_tracker_for_each_area()

  def _check_params(self,params:dict):
    """
      Ensure params with the standards.
    Args:
        params (dict): app config.
    """
    #judge type
    if not isinstance(params,dict):
      logging.error("app config type is dict! but your type is {} ,Please check and correct it.".format(type(params)))
      raise TypeError("app config type is dict! but your type is {} ,Please check and correct it.".format(type(params)))
    #judge container
    if not params.__contains__('application'):
      logging.error("app config must have key 'application'! Please check and correct it.")
      raise ValueError("app config must have key 'application'! Please check and correct it.")
    
    if not isinstance(params['application'],dict):
      logging.error("app config key 'application' type is dict! but your type is {} ,Please check and correct it.".format(type(params['application'])))
      raise TypeError("app config key 'application' type is dict! but your type is {} ,Please check and correct it.".format(type(params['application'])))
    
    if not params['application'].__contains__('areas'):
      logging.error("app config must have key 'areas'! Please check and correct it.")
      raise ValueError("app config must have key 'areas'! Please check and correct it.")
    
    if not isinstance(params['application']['areas'],list):
      logging.error("app config key 'areas' type is list! but your type is {} ,Please check and correct it.".format(type(params['application']['areas'])))
      raise TypeError("app config key 'areas' type is list! but your type is {} ,Please check and correct it.".format(type(params['application']['areas'])))
    
    return params

  def _get_depend_on(self):
    """
      Getd epend on info. 
    Returns:
      list : depend on object for each area.
    """
    temp_depend ={}
    temp_app_output_data={}
    for area_id in range(len(self.params['application']['areas'])):
      if not isinstance(self.params['application']['areas'][area_id],dict):
        logging.error("app config each areas info in key 'areas' type is dict! but your type is {} ,Please check and correct it."\
                      .format(type(self.params['application']['areas'][area_id])))
        raise TypeError("app config each areas info in key 'areas' type is dict! but your type is {} ,Please check and correct it."\
                        .format(type(self.params['application']['areas'][area_id])))
      
      if not self.params['application']['areas'][area_id].__contains__('depend_on'):
        logging.error("app config must have key 'depend_on'! Please check and correct it.")
        raise ValueError("app config must have key 'depend_on'! Please check and correct it.")

      if not isinstance(self.params['application']['areas'][area_id]['depend_on'],list):
        logging.error("app config depend_on type is dict! but your type is {} ,Please check and correct it."\
                      .format(type(self.params['application']['areas'][area_id]['depend_on'])))
        raise TypeError("app config depend_on type is dict! but your type is {} ,Please check and correct it."\
                        .format(type(self.params['application']['areas'][area_id]['depend_on'])))
      
      
      _label=[]
      if self.params['application']['areas'][area_id]['depend_on']==[]:
        
        temp_depend.update({area_id:self.label_list})
        for label in self.label_list:
           _label.append({
                          "label":label,
                          "num":0
                          })
        temp_app_output_data.update({area_id:_label})
      else:
        temp_depend.update({area_id:self.params['application']['areas'][area_id]['depend_on']})
        
        for label in self.params['application']['areas'][area_id]['depend_on']:
          if not (label in self.label_list):
            logging.error("The label {} you set not in the label path! Please check and correct it!".format(label))
            raise ValueError("The label '{}' you set not in the label file! Please check and correct it!".format(label))
          _label.append({
                          "label":label,
                          "num":0
                          })
        temp_app_output_data.update({area_id:_label})

    return temp_depend , temp_app_output_data

  def _init_draw_params(self):
      """ Initialize Draw Parameters """
      #for draw result and boundingbox
      self.frame_idx = 0
      self.frame_size = None
      self.font_size  = None
      self.font_thick = None
      self.thick      = None
      
      #for draw area
      self.area_name={}
      self.area_opacity=None
      self.area_color=None
      self.area_pts = {}
      self.area_cnt = {}
      self.normalize_area_pts = {}
      self.area_pts = {}

      #control draw
      self.draw_bbox=self.params['application']['draw_bbox'] if self.params['application'].__contains__('draw_bbox') else False
      self.draw_result=self.params['application']['draw_result'] if self.params['application'].__contains__('draw_result') else False
      self.draw_tracking=True
      self.draw_area=False
      self.is_draw_line=False
      self.draw_app_common_output=True

  def _creat_MOT_tracker_for_each_area(self):
    """
      Init MOT_tracker for each area.
    Returns:
        dict: init MOT_tracker for each area.
    """
    _temp = {}
    for area_id ,depend_info in self.depend_on.items():
      if not _temp.__contains__(area_id):
           _temp.update({area_id:{}})
      for idx , label in enumerate(depend_info):

        _temp[area_id].update({label:Sort(max_age=1, 
                                          min_hits=3,
                                          iou_threshold=0.3,
                                          app_output_data=self.app_output_data,
                                          area_id =area_id )})
    return _temp

  def _update_area_params(self):
    """
      Get area point from app config. defalt is full screen.
    """
    for area_id ,area_info in enumerate(self.params['application']['areas']):
      
      if not area_info.__contains__('area_point'): 
        logging.error("app config must have key 'area_point'! please correct it.")
        raise ValueError("app config must have key 'area_point'! please correct it.")
      if not isinstance(area_info['area_point'],list):
        logging.error("app config each areas info in key 'area_point' type is list! but your type is {} ,please correct it."\
                      .format(type(area_info['area_point'])))
        raise TypeError("app config each areas info in key 'area_point' type is list! but your type is {} ,please correct it."\
                        .format(type(area_info['area_point']))) 
      
      if not area_info.__contains__('name'): 
        logging.error("app config must have key 'name'! please correct it.")
        raise ValueError("app config must have key 'name'! please correct it.")
      if not isinstance(area_info['name'],str):
        logging.error("app config each areas info in key 'name' type is str! but your type is {} ,please correct it."\
                      .format(type(area_info['name'])))
        raise TypeError("app config each areas info in key 'name' type is str! but your type is {} ,please correct it."\
                        .format(type(area_info['name']))) 

      if area_info['area_point']!=[]:
        self.normalize_area_pts.update({area_id:area_info['area_point']})
        self.area_name.update({area_id:area_info['name']})
        # self.area_color.update({i:[random.randint(0,255),random.randint(0,255),random.randint(0,255)]})
      else:
        self.normalize_area_pts.update({area_id:[[0,0],[1,0],[1,1],[0,1]]})
        self.area_name.update({area_id:"The defalt area"})
      
      if area_info['name']!="":
        self.area_name.update({area_id:area_info['name']})
        # self.area_color.update({i:[random.randint(0,255),random.randint(0,255),random.randint(0,255)]})
      else:
        self.area_name.update({area_id:"The defalt area"})

  def _update_draw_param(self, frame:np.ndarray):
      """ Update the parameters of the drawing tool, which only happend at first time. """
      
      # if frame_size not None means it was already init 
      if( self.frame_idx > 1): return None

      # Parameters
      FRAME_SCALE     = 0.0005    # Custom Value which Related with Resolution
      BASE_THICK      = 1         # Setup Basic Thick Value
      BASE_FONT_SIZE  = 0.5   # Setup Basic Font Size Value
      FONT_SCALE      = 0.2   # Custom Value which Related with the size of the font.
      WIDTH_SPACE = 10
      HIGHT_SPACE = 10
      # Get Frame Size
      self.frame_size = frame.shape[:2]
      
      # Calculate the common scale
      scale = FRAME_SCALE * sum(self.frame_size)
      
      # Get dynamic thick and dynamic size 
      self.thick  = BASE_THICK + round( scale )
      self.font_thick = self.thick//2
      self.font_size = BASE_FONT_SIZE + ( scale*FONT_SCALE )

      self.area_color=[0,0,255]
      self.area_opacity=0.4  

      self.WIDTH_SPACE = int(scale*WIDTH_SPACE) 
      self.HIGHT_SPACE = int(scale*HIGHT_SPACE) 

  def _init_event_param(self,event_save_folder:str="event"):
    """ Initialize Event Parameters """
    #for tracking 
    self.logic_operator = {}
    self.logic_value = {}
    self.event_title = {}
    self.cooldown_time = {}
    self.sensitivity ={}
    self.event_handler={}
    self.event_save_folder=event_save_folder
    self.event_uid={}

    #for movement
    self.tracking_tag_status ={}
    self._update_tracking_tag_status()
    self.origin_line_point={}
    self.line_point={}
    self.line_relationship={}
    self.count_cross_line={}
    self.changeable_total_num_of_cross_line=0
    self._update_trigger_line_param()

  def _update_event_param(self):
    """ Update the parameters of the event, which only happend at first time. """    
    for area_id ,area_info in enumerate(self.params['application']['areas']):

      if area_info.__contains__('events'):
        if not isinstance(area_info['events'],dict):
          logging.error("Event type is dict! but your type is {} ,please correct it."\
                      .format(type(area_info['events'])))
          raise TypeError("Event type is dict! but your type is {} ,please correct it."\
                      .format(type(area_info['events'])))
        
        if not area_info['events'].__contains__('logic_operator'):
          logging.error("Events must have key 'logic_operator'! please correct it.")
          raise ValueError("Events must have key 'logic_operator'! please correct it.")
        
        self.logic_operator.update({area_id:self._get_logic_event(area_info['events']['logic_operator'])})

        if not area_info['events'].__contains__('logic_value'):
          logging.error("Events must have key 'logic_value'! please correct it.")
          raise ValueError("Events must have key 'logic_value'! please correct it.")
        
        self.logic_value.update({area_id: area_info['events']['logic_value']})

        if not area_info['events'].__contains__('title'):
          logging.error("Events must have key 'title'! please correct it.")
          raise ValueError("Events must have key 'title'! please correct it.")
        
        self.event_title.update({area_id:area_info['events']['title']})

        if area_info['events'].__contains__('cooldown_time'):
          self.cooldown_time.update({area_id:area_info['events']['cooldown_time']})
        else :
          self.cooldown_time.update({area_id:10})   
        if area_info['events'].__contains__('sensitivity'):
          self.sensitivity.update({area_id:self._get_sensitivity_event(area_info['events']['sensitivity'])})
        
        
        if area_info['events'].__contains__('uid'):
          if not isinstance(area_info['events']['uid'],str):
            logging.error("Event key uid type is str! but your type is {} ,please correct it."\
                        .format(type(area_info['events']['uid'])))
            raise TypeError("Event key uid type is str! but your type is {} ,please correct it."\
                        .format(type(area_info['events']['uid'])))
          self.event_uid.update({area_id:area_info['events']['uid']})
        else:
          self.event_uid.update({area_id:None})

      else:
        logging.warning("No set event!")
      
  def _get_logic_event(self, operator):
    """ Define the logic event """
    greater = lambda x,y: x>y
    greater_or_equal = lambda x,y: x>=y
    less = lambda x,y: x<y
    less_or_equal = lambda x,y: x<=y
    equal = lambda x,y: x==y
    logic_map = {
        '>': greater,
        '>=': greater_or_equal,
        '<': less,
        '<=': less_or_equal,
        '=': equal,
    }
    return logic_map.get(operator)

  def _get_sensitivity_event(self,sensitivity_str):
    """ Define the sensitivity of event """
    # sensitivity_map={
    #     "low":0.3,
    #     "medium" : 0.5,
    #     "high":0.7,
    # }
    sensitivity_map={
        "low":1,
        "medium" : 3,
        "high":5,
    }
    return sensitivity_map.get(sensitivity_str)

  def _sort_point_list(self,point_list:list):
        """
        This function will help user to sort the point in the list counterclockwise.
        step 1 : We will calculate the center point of the cluster of point list.
        step 2 : calculate arctan for each point in point list.
        step 3 : sorted by arctan.

        Args:
            pts_2ds (list): not sort point.


        Returns:
            point_list(list): after sort.
        
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

  def _convert_point(self,frame):
    #convert point value.

    temp_point=[]
    for area_id, area_point in self.normalize_area_pts.items():
              
        for point in area_point:
          if point[0]>1: return
          temp_point.append([math.ceil(point[0]*frame.shape[1]),math.ceil(point[1]*frame.shape[0])])
        temp_point = self._sort_point_list(temp_point)
        self.area_pts.update({area_id:temp_point})
        temp_point = []

        if not self.origin_line_point.__contains__(area_id) or self.origin_line_point[area_id]=={}: continue
        self.line_point.update({area_id:{}})
        for line_name,line_point in self.origin_line_point[area_id].items():
            self.line_point[area_id].update({line_name:[]})
            for point in  line_point:
                self.line_point[area_id][line_name].append([math.ceil(point[0]*frame.shape[1]),math.ceil(point[1]*frame.shape[0])])

  def _init_palette(self,palette:dict):
    """
      We will deal all color we need there.
      Step 1 : assign color for each label.
      Step 2 : if app_config have palette that user stting we will change the color follow user setting.
    Args:
        palette (dict): palette list.
    """
    color = None
    with open(self.label_path,'r') as f:
        # lines = f.read().splitlines()
        for idx, line in enumerate(f.readlines()):
            idx+=1
            if self.params['application'].__contains__('palette'):
                
                if self.params['application']['palette'].__contains__(line.strip()):
                    color = self.params['application']['palette'][line.strip()]
                else:
                    color = palette[str(idx)]
            else :         
                color = palette[str(idx)]
            
            self.palette.update({line.strip():color})
            self.label_list.append(line.strip())

  def _check_depend_and_area(self,dets:list):
    """
        According to the depend on to filter the detections.
    Args:
        dets : (list): output of model predict.
        traking_dets (list): output of MOT tracker.

    Returns:
        dict : According to the depend on to filter the detections. 
    """

    _dets = {}
    for area_id, depend in self.depend_on.items():
      _temp_dets={}
      for id,val in enumerate(dets):
        if self.inpolygon(((val.xmin+val.xmax)//2),((val.ymin+val.ymax)//2),self.area_pts[area_id]): 
          if val.label in depend:
            if not _temp_dets.__contains__(val.label):
              _temp_dets.update({val.label:[]})
            _temp_dets[val.label].append([val.xmin,val.ymin,val.xmax,val.ymax ,val.score])
            
      _dets.update({area_id:_temp_dets})
      
    return _dets 
  
  def _combined_data_from_SORT(self,data:list,area_id:int):
    """
      combined data from all area.
    Args:
        data (list): total object number .format [{'label': str, 'num': int},{'label': str, 'num': int},...]
        area_id (int): area_id
    """
    _temp_data=[]
    for id ,val in enumerate(data):
      if val['num']!=0:
        _temp_data.append(val)
    self.app_output_data[area_id]=_temp_data

  def _combined_app_output(self):
    app_output={"areas": []}
    for area_id,val in self.app_output_data.items():
      app_output["areas"].append({
                  "id":area_id,
                  "name":self.area_name[area_id],
                  "data":self.count_cross_line[area_id]
      })
    return app_output

  def _init_event_object(self):

   for area_id ,val in self.logic_operator.items():
      event_obj = event_handle(val,self.logic_value[area_id],self.cooldown_time[area_id]\
                               ,self.event_title[area_id],area_id,self.event_save_folder,self.event_uid) 
      self.event_handler.update( { area_id: event_obj }  )  

  def _deal_changeable_total(self,return_to_zero:bool=False):
    _temp_count = 0
    if return_to_zero:
      self.changeable_total_num_of_cross_line=0
    else : 
      for area_id , area_info in self.count_cross_line.items():
        for idx , val in enumerate(area_info):
          self.changeable_total_num_of_cross_line=self.changeable_total_num_of_cross_line+int(val['num'])
    return self.changeable_total_num_of_cross_line
  
  def _update_tracking_tag_status(self):
    """ Update the parameters of the tracking_tag_status , which only happend at first time. """
   
    for area_id,area_info in self.depend_on.items():
       if not self.tracking_tag_status.__contains__(area_id):
          self.tracking_tag_status.update({area_id:{}})
       for id,label in enumerate(area_info):
          if not  self.tracking_tag_status.__contains__(label):
            self.tracking_tag_status[area_id].update({label:[]})
    
  def _update_trigger_line_param(self):
    """ Update the parameters of the trigger line param , which only happend at first time. """
    _line_relationship={}
    for area_id,area_info in enumerate(self.params['application']["areas"]):

       if not area_info.__contains__('line_point'):
          logging.error("App config not set key line_point,Please check and correct it!")
          raise ValueError("App config not set key line_point,Please check and correct it!")
       
       if not area_info.__contains__('line_relation'):
          logging.error("App config not set key line_relation,Please check and correct it!")
          raise ValueError("App config not set key line_relation,Please check and correct it!")

       self.origin_line_point.update({area_id:area_info['line_point']})
       _line_relationship.update({area_id:area_info['line_relation']})

       if not self.line_relationship.__contains__(area_id):
          self.line_relationship.update({area_id:[]})
       if not self.count_cross_line.__contains__(area_id):
          self.count_cross_line.update({area_id:[]})
       for id,val in enumerate(_line_relationship[area_id]):
            
            self.line_relationship[area_id].append({ val['start']+val['end']: val['name'] })
            self.count_cross_line[area_id].append({
                                                 'label':val['name'],
                                                 'num':0 
                                                 })
  def _check_cross_trigger_line(self,area_id:int,label:str,tracking_tag:int,center_point:tuple):
    update_flag=False
    if len(self.tracking_tag_status[area_id][label])==0:
       self.tracking_tag_status[area_id][label].append({
                                                        'trackin_tag':tracking_tag,
                                                        'center_point':center_point,
                                                        'cross_line':[]
                                                    })
    else:
        for id,val in enumerate(self.tracking_tag_status[area_id][label]):
            
            if val['trackin_tag']== tracking_tag :
                check_is_cross, temp_cross_line = self.is_point_cross_trigger_line(val['center_point'],center_point,self.line_point,area_id)
                self.tracking_tag_status[area_id][label][id]['center_point'] = center_point
                update_flag=True
                if check_is_cross:
                    val['cross_line'].append(temp_cross_line)
                if len(val['cross_line'])==2:
                   if val['cross_line'][0]==val['cross_line'][1]:
                      self.tracking_tag_status[area_id][label][id]['cross_line']=[]
                      break
                   for line_info_id ,line_info in enumerate(self.line_relationship[area_id]):
                      if self.line_relationship[area_id][line_info_id].__contains__(val['cross_line'][0]+val['cross_line'][1]):
                        # line_info[val['cross_line'][0]+val['cross_line'][1]]
                        for idx in range(len(self.count_cross_line[area_id])):
                           if self.count_cross_line[area_id][idx]['label']==line_info[val['cross_line'][0]+val['cross_line'][1]]:
                                self.count_cross_line[area_id][idx]['num']=self.count_cross_line[area_id][idx]['num']+1
                                self.tracking_tag_status[area_id][label][id]['cross_line']=[]
                                return
        if not update_flag:
            self.tracking_tag_status[area_id][label].append({
                                                            'trackin_tag':tracking_tag,
                                                            'center_point':center_point,
                                                            'cross_line':[]
                                                        })
        
  def is_point_cross_trigger_line(self,current_center_point:tuple,now_center_point:tuple,line_point:dict,area_id:int):
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
    
    for line_name,trigger_point in line_point[area_id].items():
        
        # print(" area total{} ,p1 :({},{}) , p2 :({},{})".format(trigger_point,trigger_point[0][0],trigger_point[0][1],trigger_point[1][0],trigger_point[1][1]))
        trigger_line=np.array([trigger_point[0][0]-trigger_point[1][0],trigger_point[0][1]-trigger_point[1][1]])
        trigger_line_side=np.array([trigger_point[0][0]-current_center_point[0],trigger_point[0][1]-current_center_point[1]])
        trigger_line_side2=np.array([trigger_point[0][0]-now_center_point[0],trigger_point[0][1]-now_center_point[1]])

        trigger_cross=np.cross(trigger_line,trigger_line_side)*np.cross(trigger_line,trigger_line_side2)
        
        if trigger_cross<0:
            
            temp_line_name=line_name
            cross_flag=True
            # print(temp_id," ",trigger_cross)


    # if len(temp)>1:
    #     return True ,  [0]
    
    return cross_flag ,temp_line_name
  
  def draw_line(self,frame:np.ndarray,is_draw_line:bool=True):
    """
        Draw trigger line on the frame.
    Args:
        frame (np.ndarray): frame that we want to draw.
        is_draw_line (bool, optional): contral wether draw line on the frame or not. Defaults to True.
    """
    is_draw_line = self.is_draw_line
    if not is_draw_line : return
    for id, val in self.line_point.items():
        
        for idx ,valx in val.items():
            cv2.putText(
                    frame,str(idx) , valx[1], cv2.FONT_HERSHEY_TRIPLEX,
                    self.font_size, (0,255,255), self.font_thick, cv2.LINE_AA
                )
            
            cv2.line(frame, valx[0], valx[1], (0, 255, 255), 3)

  def draw_app_result(self,frame:np.ndarray,result:dict,outer_clor:tuple=(0,255,255),font_color:tuple=(0,0,0)):
    sort_id=0
    if self.draw_app_common_output == False:
      return
    for areas ,data in result.items():
      # print(result)
      for area_id ,area_info in enumerate(data):
        for label_id,val in enumerate(area_info['data']):
          temp_direction_result=" {} ({}) : {} ".format(self.area_name[area_id],str(val['label']),str(val['num']))
          
          
          (t_wid, t_hei), t_base = cv2.getTextSize(temp_direction_result, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
          
          t_xmin, t_ymin, t_xmax, t_ymax = self.WIDTH_SPACE, self.HIGHT_SPACE+self.HIGHT_SPACE*sort_id+(sort_id*(t_hei+t_base)), \
            self.WIDTH_SPACE+t_wid, self.HIGHT_SPACE+self.HIGHT_SPACE*sort_id+((sort_id+1)*(t_hei+t_base))
          
          cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), outer_clor , -1)
          cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), (0,0,0) , 1)
          cv2.putText(
              frame, temp_direction_result, (t_xmin, t_ymax), cv2.FONT_HERSHEY_SIMPLEX,
              self.font_size, font_color, self.font_thick, cv2.LINE_AA
          )
          sort_id=sort_id+1

  def draw_area_event(self, frame:np.ndarray, is_draw_area:bool, area_color:tuple=None, area_opacity:float=None, draw_points:bool=True):
    """ Draw Detecting Area and update center point if need.
    - args
        - frame: input frame
        - area_color: control the color of the area
        - area_opacity: control the opacity of the area
    """

    if not is_draw_area: return frame
    # Get Parameters
    area_color = self.area_color if area_color is None else area_color
    area_opacity = self.area_opacity if area_opacity is None else area_opacity
    
    # Parse All Area
    overlay = frame.copy()

    temp_area_next_point = []


    for area_idx, area_pts in self.area_pts.items():
        
                    
        if area_pts==[]: continue

        # draw area point
        if  draw_points: 
            
            [ cv2.circle(frame, tuple(pts), 3, area_color, -1) for pts in area_pts ]

        # if delet : referenced before assignment
        minxy,maxxy=(max(area_pts),min(area_pts))

        for pts in area_pts:
            if temp_area_next_point == []:
                cv2.line(frame,pts,area_pts[-1], (0, 0, 255), 3)
                
            else:
            
                cv2.line(frame, temp_area_next_point, pts, (0, 0, 255), 3)
            temp_area_next_point= pts
                
            if (tuple(pts)[0]+tuple(pts)[1])<(minxy[0]+minxy[1]): 
                minxy= pts

        #draw area name for each area            
        area_name =self.area_name[area_idx]     
        (t_wid, t_hei), t_base = cv2.getTextSize(area_name, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
        t_xmin, t_ymin, t_xmax, t_ymax = minxy[0], minxy[1], minxy[0]+t_wid, minxy[1]+(t_hei+t_base)
        
        cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), (0,0,255) , -1)
        cv2.putText(
            frame, area_name, (t_xmin, t_ymax), cv2.FONT_HERSHEY_SIMPLEX,
            self.font_size, (0,0,0), self.font_thick, cv2.LINE_AA
        )

        #draw area 
        cv2.fillPoly(overlay, pts=[ np.array(area_pts) ], color=area_color)
        temp_area_next_point = []
    
    return cv2.addWeighted( frame, 1-area_opacity, overlay, area_opacity, 0 ) 

  def draw_tracking_tag(self,frame:np.ndarray,tracking_tag:str,left_top:tuple, right_down:tuple,outer_clor:tuple,font_color:tuple=(255,255,255),draw_tracking:bool=True):

    draw_tracking = self.draw_tracking if self.draw_tracking is not None else draw_tracking
    xmin ,ymin=left_top
    xmax ,ymax=right_down
    if not draw_tracking: return
    (t_wid, t_hei), t_base = cv2.getTextSize(str(tracking_tag), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
    half_wid, half_hei = t_wid//2, t_hei//2
    [cnt_x, cnt_y] = [(xmin+xmax)//2,(ymin+ymax)//2]
    t_xmin, t_ymin, t_xmax, t_ymax = cnt_x-half_wid, cnt_y-half_hei, cnt_x+half_wid, cnt_y+half_hei
    cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), outer_clor , -1)
    cv2.putText(
        frame, str(tracking_tag), (t_xmin, t_ymin+t_hei), cv2.FONT_HERSHEY_SIMPLEX,
        self.font_size, font_color, self.font_thick, cv2.LINE_AA
    )
    
  def inpolygon(self,px,py,poly):
      is_in = False
      for i , corner in enumerate(poly):
          
          next_i = i +1 if i +1 < len(poly) else 0
          x1 ,y1 = corner
          x2 , y2=poly[next_i]
          if (x1 == px and y1 ==py) or (x2==px and y2 ==py):
            is_in = False
            
            break
          if min(y1,y2) <py <= max(y1 ,y2):
              
            x =x1+(py-y1)*(x2-x1)/(y2-y1)
            if x ==px:
              is_in = False
              break
            elif x > px:
              
              is_in = not is_in
      return is_in

  def get_color(self, label:str):
        """
            Get color of label.
        Args:
            label (str): label of object.

        Returns:
            list: (B,G,R).
        """
        return self.palette[label]
  
  def custom_function(self, frame:np.ndarray, color:tuple, label:str,score:float,\
                       left_top:tuple, right_down:tuple,draw_bbox:bool=True,draw_result:bool=True):
        
        """
        The draw method customize by user .

        Args:
            frame (np.ndarray): The img that we want to deal with.
            color (tuple): what color of label that you want to show.
            label (str): label of object.
            score (float): confidence of model predict.
            left_top (tuple): bbox left top.
            right_down (tuple): bbox right down
            draw_bbox (bool, optional): Draw bbox or not . Defaults to True.
            draw_result (bool, optional): Draw result on the bbox or not. Defaults to True.

        Returns:
            np.ndarray: frame that finished painting
        """
        (xmin, ymin), (xmax, ymax) = map(int, left_top), map(int, right_down)
        info = '{} {:.1%}'.format(label, score)
        # Draw bounding box
        draw_bbox = self.draw_bbox if self.draw_bbox is not None else draw_bbox
        if draw_bbox:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color , self.thick)

        draw_result = self.draw_result if self.draw_result is not None else draw_result
       
        # Draw Text
        if draw_result:
            (t_wid, t_hei), t_base = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
            t_xmin, t_ymin, t_xmax, t_ymax = xmin, ymin-(t_hei+(t_base*2)), xmin+t_wid, ymin
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), color , -1)
            cv2.putText(
                frame, info, (xmin, ymin-(t_base)), cv2.FONT_HERSHEY_SIMPLEX,
                self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
            )
        return frame

  def set_draw(self,params:dict):
    """
    Control anything about drawing.
    Which params you can contral :

    { 
        draw_area : bool , 
        draw_bbox : bool ,
        draw_result : bool ,
        draw_tracking : bool ,
        draw_line : bool ,
        draw_app_common_output : bool ,
        palette (dict) { label(str) : color(Union[tuple, list]) },
    }
    
    Args:
        params (dict): 
    """
    color_support_type = Union[tuple, list]
    if not isinstance(params, dict):
        logging.error("Input type is dict! but your type is {} ,please correct it.".format(type(params.get('draw_area', None))))
        return

    if isinstance(params.get('draw_area', self.draw_area) , bool):
        self.draw_area= params.get('draw_area', self.draw_area) 
        logging.info("Change draw_area mode , now draw_area mode is {} !".format(self.draw_area))
    else:
        logging.error("draw_area type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_area', self.draw_area))))

    if isinstance(params.get('draw_bbox', self.draw_bbox) , bool):
        self.draw_bbox= params.get('draw_bbox', self.draw_bbox)
        logging.info("Change draw_bbox mode , now draw_bbox mode is {} !".format(self.draw_bbox))
    else:
        logging.error("draw_bbox type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_bbox', self.draw_bbox))))
    
    if isinstance(params.get('draw_result', self.draw_result) , bool):    
        self.draw_result= params.get('draw_result', self.draw_result)
        logging.info("Change draw_result mode , now draw_result mode is {} !".format(self.draw_result))
    else:
        logging.error("draw_result type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_result', self.draw_result))))
    
    if isinstance(params.get('draw_tracking', self.draw_tracking) , bool):    
        self.draw_tracking= params.get('draw_tracking', self.draw_tracking)
        logging.info("Change draw_tracking mode , now draw_tracking mode is {} !".format(self.draw_tracking))
    else:
        logging.error("draw_tracking type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_tracking', self.draw_tracking))))

    if isinstance(params.get('draw_line', self.is_draw_line) , bool):    
        self.is_draw_line= params.get('draw_line', self.is_draw_line)
        logging.info("Change draw_line mode , now draw_line mode is {} !".format(self.is_draw_line))
    else:
        logging.error("draw_line type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_line', self.is_draw_line))))

    if isinstance(params.get('draw_app_common_output', self.draw_app_common_output) , bool):    
        self.draw_app_common_output= params.get('draw_app_common_output', self.draw_app_common_output)
        logging.info("Change draw_app_common_output mode , now draw_line mode is {} !".format(self.draw_app_common_output))
    else:
        logging.error("draw_app_common_output type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_line', self.draw_app_common_output))))

    palette = params.get('palette', None)
    if isinstance(palette, dict):
        if len(palette)==0:
            logging.warning("Not set palette!")
            pass
        else:
            for label,color in palette.items():

                if isinstance(label, str) and isinstance(color, get_args(color_support_type)):
                    if self.palette.__contains__(label):
                        self.palette.update({label:color})
                    else:
                        logging.error("Model can't recognition the label {} , please checkout your label!.".format(label))
                    logging.info("Label: {} , change color to {}.".format(label,color))
                else:
                    logging.error("Value in palette type must (label:str , color :Union[tuple , list] ),your type \
                                    label:{} , color:{} is error.".format(type(label),type(color)))
    else:
        logging.error("Not set palette or your type {} is error.".format(type(palette)))
  
  def __call__(self,frame:np.ndarray, detections:list):

    #step1 : Update param.
    self._update_draw_param(frame)
    self._convert_point(frame) #{0: [[0.256, 0.583], [0.658, 0.503], [0.848, 0.712], [0.356, 0.812]]} {0: [[1629, 769], [684, 877], [492, 630], [1264, 544]]}
    ori_frame = frame.copy()

    if cv2.waitKey(1) in [ ord('c'), 99 ]: self.draw_area= self.draw_area^1
    frame = self.draw_area_event(frame, self.draw_area)
    self.draw_line(frame) 
    #step2 : According to the depend on and area to filter the detections.
    dets = self._check_depend_and_area(detections)
    
    #step3 : Strat tracking for each label in each area.
    for area_id , area_tracker in self.MOT_tracker.items():
      for label , tracker in area_tracker.items():

        if not dets[area_id].__contains__(label):
          continue #this label no detect object 
        #dets format {0: {'car': [[405, 338, 595, 476, 0.9736913473111674], [370, 446, 560, 584, 0.9719230118526662], [730, 367, 926, 592, 0.9635377428470362], [749, 351, 890, 408, 0.49872639775276184]]}, 1: {'car': [[469, 220, 559, 277, 0.4864683151245117]]}}
        tracking_result = tracker.update(dets[area_id][label],label)
        
        self._combined_data_from_SORT(tracker.app_output_data,area_id)
        

        #step4 : draw result.  
        for id,val in enumerate(tracking_result):
          
          tracking_tag, idx_in_dets, xmin, ymin, xmax, ymax = int(float(val[4])), int(float(val[5])),int(float(val[0])),int(float(val[1])),int(float(val[2])),int(float(val[3]))
          
          #step5 : Check label whether cross trigger line or not. 
          self._check_cross_trigger_line(area_id,label,tracking_tag,((xmin+xmax)//2,(ymin+ymax)//2))

          frame = self.custom_function(
                          frame = frame,
                          color = self.get_color(label) ,
                          label = label,
                          score = dets[area_id][label][idx_in_dets][4],
                          left_top = (xmin, ymin),
                          right_down = (xmax, ymax)
                      )
          self.draw_tracking_tag(
                        frame = frame,
                        tracking_tag="{}:{}".format(label,str(tracking_tag)), 
                        left_top = (xmin, ymin),
                        right_down = (xmax, ymax),
                        outer_clor =self.get_color(label) ,
                      )
    #step6: combined app_output.

    app_output = self._combined_app_output()
    
    #step7: deal event.
    #if the area don't have set event. 
    event_output={'event':[]}
    for area_id , event_handler in self.event_handler.items():

      # self.pool.apply_async(event_handler,(frame,ori_frime,i,self.app_thread.total,self.app_thread.app_output))
      event_handler(frame,ori_frame,area_id,self._deal_changeable_total(False),app_output)
      if (event_handler.pass_time == event_handler.cooldown_time):
        self._deal_changeable_total(True)
      if event_handler.event_output !={}:
        event_output['event'].append(event_handler.event_output)

    #step8: draw total result on the left top.
    self.draw_app_result(frame,app_output)
    return (frame ,app_output,event_output)

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
                                    0.36666666666,
                                    0.64074074074
                                ],
                                [
                                    0.67291666666,
                                    0.52962962963
                                ]
                            ],
                            "line_2": [
                                [
                                    0.36041666666,
                                    0.83333333333
                                ],
                                [
                                    0.72916666666,
                                    0.62962962963
                                ]
                            ],
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
                "draw_result":False,
                "draw_bbox":False
            }
        }
    app = Movement_Zone(app_config,args.label )

    # 7. Start Inference
    try:
        while True:
            # Get frame & Do infernece
            frame = src.read()
            
            results = model.inference(frame=frame)
          
            frame , app_output , event_output =app(frame,results)
            # print(app_output,'\n')
            # print(event_output)
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