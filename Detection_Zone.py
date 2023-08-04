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
from typing import Tuple

class event_handle(threading.Thread):
  def __init__(self ,operator,thres:dict,cooldown_time:dict,event_title:dict,area_id:int,event_save_folder:str,uid:dict):
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
        
      self.trigger_time=datetime.now()
      self.pass_time = (int(self.event_time.minute)*60+int(self.event_time.second))-(int(self.trigger_time.minute)*60+int(self.trigger_time.second))
      uid=self.uid[area_id] if not (self.uid[area_id]==None) else str(uuid.uuid4())[:8]
      path='./'+self.event_save_folder+'/'+str(uid)+'/'+str(time.time())+'/'
      if not os.path.isdir(path):
          os.makedirs(path)
      cv2.imwrite(path+'original.jpg', frame)
      cv2.imwrite(path+'overlay.jpg', ori_frame)
      self.event_output.update({
        "uuid": uid,
        "title": self.event_title,
        "areas": app_output["areas"],
        "timesamp":str(self.trigger_time),
        "screenshot":
            { "overlay": path+str(self.trigger_time)+'.jpg',
      "original": path+str(self.trigger_time)+"_org"+'.jpg'}}) 
      # Draw Inforamtion
      
      self.info = "The {} area : ".format(area_id)+self.event_title+\
        ' , '.join([ 'total:{}  , cool down time:{}/{}'.format(total_object_number,0,self.cooldown_time)])
      
    else :
        
      self.pass_time = (int(self.event_time.minute)*60+int(self.event_time.second))-(int(self.trigger_time.minute)*60+int(self.trigger_time.second))    
      self.info = "The {} area : ".format(area_id)+self.event_title+\
        ' , '.join([ 'total:{}  , cool down time:{}/{}'.format(total_object_number,self.pass_time,self.cooldown_time)])

class Detection_Zone(iAPP_OBJ):

    def __init__(self, params:dict, label:str, event_save_folder:str="event", palette:dict=palette):
        
        # Define Application Type
        self.app_type = 'obj'

        # This step will check application and areas whether with standards or not .
        self.params = self._init_params(params)
        self.label_path = label

        self.palette={}
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

    def _init_params(self, params:dict) -> dict:
        """Initailize and check parameters

        Args:
            params (dict): the application config

        Raises:
            TypeError: params type error.
            ValueError: params['application'] not setup.
            TypeError: params['application'] type error.
            ValueError: params['application']['areas] not setup
            TypeError: params['application']['areas] type error

        Returns:
            dict: params after checking
        """
        #check config type
        if not isinstance(params, dict):
            raise TypeError(f"The app_config should be dictionaray ! but get {type(params)}.")
        
        #check config key ( application ) is exist or not and the type is correct or not
        app_info =  params.get("application", None)
        if not app_info:
           raise ValueError("The app_config must have key 'application'.")
        
        if not isinstance( app_info, dict):
            raise TypeError(f"The app_config['application'] should be dictionaray ! but get {type(app_info)}.")
        
        #check area setting is exist or not and the type is correct or not
        areas_info = app_info.get("areas", None)
        if not areas_info:
           raise ValueError("The app_config['application'] must have key 'areas'.")
        
        if not isinstance( areas_info, list):
            raise TypeError(f"The app_config['application']['areas'] should be list ! but get {type(areas_info)}.")
        
        return params

    def _init_palette_and_label(self, palette:dict, label_path:str) -> Tuple[ dict, list ]:
        """
        We will deal all color we need there.
        Step 1 : assign color for each label.
        Step 2 : if app_config have palette that user stting we will change the color follow user setting.
        Args:
            palette (dict): palette list.

        palette -> index: list ( bgr )
        """

        #check type and path is available
        if not isinstance(palette, dict):
            raise TypeError(f"Expect palette type is dict, but get {type(palette)}.")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Can not find label file in '{label_path}'.")
        
        #update custom color
        custom_palette = self.params["application"].get("palette", None)
        
        #paramerters
        ret_palette = {}
        ret_label = []

        #update palette and label
        idx = 1
        f = open(label_path, 'r')
        for raw_label in f.readlines():
            label = raw_label.strip()
            
            if custom_palette and label in custom_palette:
                color = custom_palette[label]
            else:
                color = palette[str(idx)]
            
            ret_palette[label] = color
            ret_label.append(label)
            idx += 1
        
        f.close()
        return ( ret_palette, ret_label )


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
      self.draw_area=False
      self.draw_app_common_output = True

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
    self.logic_operator = {}
    self.logic_value = {}
    self.event_title = {}
    self.cooldown_time = {}
    self.sensitivity ={}
    self.event_handler={}
    self.event_save_folder=event_save_folder
    self.event_uid={}

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
            self.cooldown_time.update({area_id:self.params['application']['areas'][i]['events']['cooldown_time']})
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

  def _convert_area_point(self,frame):
    #convert point value.

    temp_point=[]
    for area_id, area_point in self.normalize_area_pts.items():
              
        for point in area_point:
          if point[0]>1: return
          temp_point.append([math.ceil(point[0]*frame.shape[1]),math.ceil(point[1]*frame.shape[0])])
        temp_point = self._sort_point_list(temp_point)
        self.area_pts.update({area_id:temp_point})
        temp_point = []

  def _check_depend(self, label:str):
        """
            Check label whether in the depend on or not.
        Args:
            label (str): label of model predict.

        Returns:
            bool : label whether in the depend on or not.
            int : area_id.
        """
        ret = True
        area_id = []
        for _temp_area_id , depend_on_list in self.depend_on.items():
            if len(depend_on_list)>0:
                ret = (label in depend_on_list)
                area_id.append(_temp_area_id)
        
        return ret ,area_id
  
  def _cal_app_output_data(self,label:str,area_id:int):
    """
      combined data from all area.
    Args:
        label (str): output from model.
        area_id (int): area_id
    """
    
    for id ,val in enumerate(self.app_output_data[area_id]):
      if val['label']==label:
        self.app_output_data[area_id][id]['num']+=1
    

  def _combined_app_output(self):
    app_output={"areas": []}
    for area_id,val in self.app_output_data.items():
      app_output["areas"].append({
                  "id":area_id,
                  "name":self.area_name[area_id],
                  "data":val
      })
    return app_output

  def _init_event_object(self):

   for area_id ,val in self.logic_operator.items():
      event_obj = event_handle(val,self.logic_value[area_id],self.cooldown_time[area_id]\
                               ,self.event_title[area_id],area_id,self.event_save_folder,self.event_uid) 
      self.event_handler.update( { area_id: event_obj }  )  

  def draw_app_result(self,frame:np.ndarray,result:dict,outer_clor:tuple=(0,255,255),font_color:tuple=(0,0,0)):
    sort_id=0
    if self.draw_app_common_output == False:
      return
    for areas ,data in result.items():
      # print(result)
      for area_id ,area_info in enumerate(data):
        for label_id,val in enumerate(area_info['data']):
        #   if val['num']==0:
        #      continue
          temp_direction_result=" {} : {} object ".format(self.area_name[area_id],str(val['num']))
          
          
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
        
        if isinstance(params.get('draw_app_common_output', self.draw_app_common_output) , bool):    
            self.draw_app_common_output= params.get('draw_app_common_output', self.draw_app_common_output)
            logging.info("Change draw_app_common_output mode , now draw_app_common_output mode is {} !".format(self.draw_app_common_output))
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


  def _init_app_output(self):
    """
        return count to 0
    """
    for area_id ,val in self.app_output_data.items():
        for id ,val in enumerate(val):
            self.app_output_data[area_id][id]['num']=0
   
  def _get_total(self,area_id:int):
    """
        get total object num
    """
    _temp=0
    
    for id ,val in enumerate(self.app_output_data[area_id]):
        _temp = _temp +val['num']

    return _temp

  def __call__(self,frame:np.ndarray, detections:list):

    #step1 : Update draw param.
    self._update_draw_param(frame)
    self._convert_area_point(frame) #{0: [[0.256, 0.583], [0.658, 0.503], [0.848, 0.712], [0.356, 0.812]]} {0: [[1629, 769], [684, 877], [492, 630], [1264, 544]]}
    ori_frame = frame.copy()
    self._init_app_output()

    if cv2.waitKey(1) in [ ord('c'), 99 ]: self.draw_area= self.draw_area^1
    frame = self.draw_area_event(frame, self.draw_area)
    
    #step2 : Strat detection for each area.

    for detection in detections:
        # Check Label is what we want
        ( label, score, xmin, ymin, xmax, ymax ) \
                = detection.label, detection.score, detection.xmin, detection.ymin, detection.xmax, detection.ymax  
        # if user have set depend on
        ret , _total_area =self._check_depend(label)
        if ret:
            for id ,area_id in enumerate(_total_area):
                if self.inpolygon(((xmin+xmax)//2),((ymin+ymax)//2),self.area_pts[area_id]): 
                    #step3 : draw bbox and result. 
                    frame = self.custom_function(
                        frame = frame,
                        color = self.get_color(label) ,
                        label = label,
                        score=score,
                        left_top = (xmin, ymin),
                        right_down = (xmax, ymax)
                    )
                    #update app_output
                    self._cal_app_output_data(label,area_id)

    #step4: combined app_output.
    app_output = self._combined_app_output()

    #step5: draw total result on the left top.
    self.draw_app_result(frame,app_output)

    #step6: deal event.
    #if the area don't have set event. 
    event_output=[]
    for area_id , event_handler in self.event_handler.items():
      event_handler(frame,ori_frame,area_id,self._get_total(area_id),app_output)
      if event_handler.event_output !={}:
        event_output.append(event_handler.event_output)

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
                        "car",
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
                        "uid":"cfd1f399",
                        "title": "Traffic is very heavy",
                        "logic_operator": ">",
                        "logic_value": 1,
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
                        "uid":"cfd1f399",
                        "title": "Traffic is very heavy",
                        "logic_operator": ">",
                        "logic_value": 1,
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