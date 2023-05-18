import sys, os, cv2, logging, time
import numpy as np
import uuid
import os
import threading
import math
import random
from typing import Union, get_args
from datetime import datetime
sys.path.append( os.getcwd() )
from apps.palette import palette
from multiprocessing.pool import ThreadPool
from ivit_i.common.app import iAPP_OBJ
class event_handle(threading.Thread):
    def __init__(self ,operator:dict,thres:dict,cooldown_time:dict,event_title:dict,area_id:int):
        threading.Thread.__init__(self)
        self.operator = operator
        self.thres = thres
        self.cooldown_time = cooldown_time
        self.event_title = event_title
        self.event_output={}
        self.area_id =area_id
        self.pass_time = self.cooldown_time[self.area_id]+1
        self.event_time=datetime.now()
        self.trigger_time=datetime.now()        
        self.info =" "
       
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

    def logic_event(self, value,thres,area_id):
        return self.operator[area_id](value,thres)

    def __call__(self,frame,ori_frame,area_id,total_object_number,app_output):
        self.event_output={}
        self.event_time=datetime.now()
        self.info =""
        # area have operator
        if self.operator.__contains__(area_id) == False:
            return
        
        # when object in area 
        if total_object_number.__contains__(area_id) == False: 
            return
        
        if self.logic_event(total_object_number[area_id],self.thres[area_id],area_id) == False:
            return
        
       
        if self.pass_time > self.cooldown_time[area_id]:
            
            self.eventflag=True
            self.trigger_time=datetime.now()
            self.pass_time = (int(self.event_time.minute)*60+int(self.event_time.second))-(int(self.trigger_time.minute)*60+int(self.trigger_time.second))
            uid=str(uuid.uuid4())[:9]
            path='./'+str(uid)+'/'
            if not os.path.isdir(path):
                os.mkdir(path)
            cv2.imwrite(path+str(self.trigger_time)+'.jpg', frame)
            cv2.imwrite(path+str(self.trigger_time)+"_org"+'.jpg', ori_frame)
            self.event_output.update({"uuid":uid,"title":self.event_title[area_id],"areas":app_output["areas"][area_id],"timesamp":self.trigger_time,"screenshot":{"overlay": path+str(self.trigger_time)+'.jpg',
            "original": path+str(self.trigger_time)+"_org"+'.jpg'}}) 
            # Draw Inforamtion
            
            self.info = "The {} area : ".format(area_id)+self.event_title[area_id]+' , '.join([ 'total:{}  , cool down time:{}/{}'.format(total_object_number[area_id],0,self.cooldown_time[area_id])])
            
        else :
            
            self.pass_time = (int(self.event_time.minute)*60+int(self.event_time.second))-(int(self.trigger_time.minute)*60+int(self.trigger_time.second))    
            self.info = "The {} area : ".format(area_id)+self.event_title[area_id]+' , '.join([ 'total:{}  , cool down time:{}/{}'.format(total_object_number[area_id],self.pass_time,self.cooldown_time[area_id])])

class app_common_handle(threading.Thread):
    def __init__(self ,params:dict,area_mask:dict,depend_on:dict,total:dict,app_output:dict,area_pts:dict,area_name:dict,sensitivity:dict,tracking_distance):
        threading.Thread.__init__(self)
        self.params=params
        self.area_mask=area_mask
        self.depend_on=depend_on
        self.total={}
        self.app_output={}
        self.area_pts2=area_pts
        self.area_name2=area_name
        self.sensitivity=sensitivity
        self.track_object={}
        self.tracking_distance=tracking_distance
        self.total_object={}
        self.object_id=int
        self.object_buffer={}
        self.is_draw=False
        self.show_object_info=""
        self.show_object =[]
        
        
    def update_tracking_distance(self,frame):
        """

        """
        if frame.shape[0]<1080:
            self.tracking_distance = 60*(2/(math.pow(frame.shape[0]/1080,2)+math.pow(frame.shape[1]/1920,2))) 
        else:
            self.tracking_distance = 60*(2/(math.pow(1080/frame.shape[0],2)+math.pow(1920/frame.shape[1],2)))
    
    def update_tracking_distance(self,new_tracking_distance):
        self.tracking_distance = new_tracking_distance


    def init_tracking(self,xmin,xmax,ymin,ymax,area_id): 
        
        if len(self.track_object)!=len(self.depend_on):
            for i in range(len(self.depend_on)):
                center_x,center_y=(xmin+xmax)//2,(ymin+ymax)//2
                self.track_object.update({i:{0:{'x':center_x,'y':center_y,'frame_time':time.time()}}})
                self.object_buffer.update({i:{}})
                self.total_object.update({i:0})
        if self.app_output.__contains__("areas")==False:
            self.app_output.update({"areas": []})
            
        if len(self.app_output["areas"])==area_id:
            self.app_output["areas"].append({"id":area_id,"name":self.area_name2[area_id],"data":[]}) 

    def inpolygon_mask(self,left_up:list,right_down:list,area_id,frame):
        """
        mapping algorithm : Determine the object is in the area.
    
        """
        l_r = left_up
        m_up = [(right_down[0]+left_up[0])//2,left_up[1]]
        r_up = [right_down[0],left_up[1]]
        r_m = [right_down[0],(right_down[1]+left_up[1])//2]
        m_d = [int((right_down[0]+left_up[0])/2),right_down[1]]
        l_d = [ left_up[0],right_down[1]]
        l_m = [left_up[0],(right_down[1]+left_up[1])//2]
        r_d = right_down
        bb_point={1:l_r,2:m_up,3:r_up,4:r_m,5:m_d,6:l_d,7:l_m,8:r_d}

        num = 0
        x , y = 0,0
        for i , v in bb_point.items():
            
            if v[0]>= self.area_mask[area_id].shape[0]:
                x =self.area_mask[area_id].shape[0]-1
            else:
                x = v[0]  
            if v[1]>= self.area_mask[area_id].shape[1]:
                y = self.area_mask[area_id].shape[1]-1
            else:
                y = v[1]    
                  
            if self.area_mask[area_id][x][y]==1:
                num=num+1
        return num    
    
    def countiog_IOU(self,area_mask,object_mask):
        """
        Determine the object is in the area.
        perfomance is bad so we don't use this algorithm. 
    
        """

        sum_original = (object_mask==1).sum()
        if sum_original: return 0.0
        overlapping=area_mask+object_mask
        sum_overlay = (overlapping==2).sum()
        return sum_overlay/sum_original
    
    def cal_distance(self,p1x,p1y,p2x,p2y):
        return round(math.sqrt(math.pow(p1x-p2x,2))+math.sqrt(math.pow(p1y-p2y,2)))

    def delete_object_point(self,xmin,xmax,ymin,ymax,area_id):

        temp_xy = self.tracking_distance
        temp_id = int

        for object_id , object_value in self.track_object[area_id].items():
            if temp_xy > self.cal_distance(object_value['x'],object_value['y'],(xmin+xmax)//2,(ymin+ymax)//2) :
                temp_xy = self.cal_distance(object_value['x'],object_value['y'],(xmin+xmax)//2,(ymin+ymax)//2)
                temp_id = object_id


        # detect object which leaving area ,we move its info from track_object to object_buffer    
        if (temp_xy != self.tracking_distance):
            self.object_buffer[area_id].update({
                temp_id:{
                        'x': (xmin+xmax)//2,
                        'y': (ymin+ymax)//2,
                        'frame_time': time.time()
                }})
        
            del self.track_object[area_id][temp_id]
 
    def update_object_point(self,frame,xmin,xmax,ymin,ymax,area_id):
        
        temp_xy = self.tracking_distance
        tracked = 0
        buffer_distance=60
        coby_track_object=self.track_object[area_id].copy()
        coby_track_object_buffer=self.object_buffer[area_id].copy()
        for i , v in coby_track_object_buffer.items():
            # print(coby_track_object_buffer)
            if buffer_distance > self.cal_distance(v['x'],v['y'],(xmin+xmax)//2,(ymin+ymax)//2):
                self.object_buffer[area_id][i]['frame_time']=time.time()
                # print("id {} , val {} :".format(i,v))
                return  tracked  
            if time.time()-v['frame_time']>3:
                del self.object_buffer[area_id][i] 
                
        for object_id , object_value in coby_track_object.items():
            
            # cv2.circle(frame,(object_value['x'],object_value['y']),1,[255,0,0],3)
            if time.time()-object_value['frame_time']>3:
                del self.track_object[area_id][object_id]
                continue
               
            """ distance less than juge distance, update center point """
            if temp_xy > self.cal_distance(object_value['x'],object_value['y'],(xmin+xmax)//2,(ymin+ymax)//2):
                # keep update minimum distance
                temp_xy = self.cal_distance(object_value['x'],object_value['y'],(xmin+xmax)//2,(ymin+ymax)//2)
                self.object_id = object_id
                tracked = 1

       
        if not tracked:
            
            self.total_object[area_id]+=1
            self.object_id = self.total_object[area_id]
        
        self.track_object[area_id].update({ 
            self.object_id: { 
                    
                    'x': (xmin+xmax)//2,
                    'y': (ymin+ymax)//2,
                    'frame_time': time.time() }})
        
        self.show_object_info="Area{}: {}".format(str(area_id),str(self.object_id))

        return tracked
  
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

    def update_object_number_this_frame(self,area_id):
        if  self.total.__contains__(area_id):
            _total=self.total[area_id]+1    
            self.total.update({area_id:_total})
        else:
            self.total.update({area_id:1})   

    def check_depend_on(self,area_id):
        if len(self.depend_on[area_id])!=0:
            return True
        return False
    
    def __call__(self,area_id:int,label:str, score:float, xmin:int, ymin:int, xmax:int, ymax:int,area,frame):
        """
        Update track_obj

        """
        #In the bigin of tracking , do once.  
        self.init_tracking(xmin,xmax,ymin,ymax,area_id)
        # #self.update_tracking_distance( frame )
        # Delete object which leaving area.
        # if user have set sensitivity we use mapping algorithm , else we use ray casting algorithm.
        self.area_pts2 = area
        if self.sensitivity.__contains__(area_id):
            if self.inpolygon_mask([xmin, ymin],[xmax,ymax],area_id,frame)<self.sensitivity[area_id]:
                self.delete_object_point(xmin,xmax,ymin,ymax,area_id)
                return
        else:
            if not  self.inpolygon(((xmin+xmax)/2),((ymin+ymax)/2),self.area_pts2[area_id]): 
                self.delete_object_point(xmin,xmax,ymin,ymax,area_id)
                return


        if self.check_depend_on(area_id):
            if  (label in self.depend_on[area_id]):
                
                # update tracking object
                 
                #according to area and depend on to creat app output. Do once.
                if len(self.depend_on[area_id])!=len(self.app_output["areas"][area_id]["data"]):
                    for key in self.depend_on[area_id]:
                        self.app_output["areas"][area_id]["data"].append({"label":key,"num":0})
                
                # is new object have detection  
                if self.update_object_point(frame,xmin,xmax,ymin,ymax,area_id): 
                    self.update_object_number_this_frame(area_id)

                
                for d in range(len(self.app_output["areas"][area_id]["data"])):    
                    if self.app_output["areas"][area_id]["data"][d]["label"]==label:

                        if self.total.__contains__(area_id): 
                            self.app_output["areas"][area_id]["data"][d].update({"num":self.total_object[area_id]+1})
                
                
                if self.show_object==[]:
                    
                    self.is_draw=True
                elif self.object_id in self.show_object:
                    self.is_draw=False  
                else :
                    self.is_draw=True      
                  
                # print(self.app_output,'\n')
        else:
            
            #according to area and user setting to creat app output. Do once.
            if (len(self.app_output["areas"][area_id]["data"])==0): 
                self.app_output["areas"][area_id]["data"].append({"label":label,"num":1})

            
            
            if self.update_object_point(frame,xmin,xmax,ymin,ymax,area_id): 
                self.update_object_number_this_frame(area_id)
                for d in range(len(self.app_output["areas"][area_id]["data"])): 
                        
                    if self.app_output["areas"][area_id]["data"][d]["label"]==label:
                        _=self.app_output["areas"][area_id]["data"][d]["num"]+1
                        self.app_output["areas"][area_id]["data"][d].update({"num":_})
                        break
                    #app output doesn't have this label , new object !
                    if len(self.app_output["areas"][area_id]["data"])-1==d and self.app_output["areas"][area_id]["data"][d]["label"]!=label: 
                        self.app_output["areas"][area_id]["data"].append({"label":label,"num":1})

            
            self.is_draw=True
                 

class Tracking_Zone(iAPP_OBJ,event_handle,app_common_handle ):
    def __init__(self, params=None, label=None, palette=palette, log=True):
        
        self.params = params
        self.check_params()
        self.app_type = 'obj'
        self.depend_on ={}
        self.palette={}
        self.event_title={}
        self.operator={}
        self.thres={}
        self.app_output={}
        self.total={}
        self.cooldown_time={}
        self.sensitivity ={}
        # self.init_area()
        self.event_output ={}
        self.event_handler = {}
        self.area_mask={}


        self.track_object={}
        self.tracking_distance=60
        self.total_object=0

        self.model_label = label
        self.model_label_list =[]

        # self.pool = ThreadPool(os.cpu_count() )
        self.init_palette(palette)

        self.collect_depand_info()
        self.init_draw_params()
        self.init_logic_param()


        self.app_common_start()
        self.init_event_object()
        self.event_start()
        # parse area amount
        # assume areas means all area

    def check_params(self):
        if not self.params:
            logging.error('App config is None , plz set app config')
            sys.exit(0) 
        if not self.params.__contains__('application') or not self.params['application'] :
            logging.error('Key application value is None , plz set application in app config')
            sys.exit(0) 
        if not (dict==type(self.params['application'])):
            logging.error('Key application value is not support , plz check application in app config')
            sys.exit(0) 


    def update_tracking_distance(self,new_tracking_distance):
        self.tracking_distance = new_tracking_distance

    def init_palette(self,palette):

        color = None
        with open(self.model_label,'r') as f:
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
                self.model_label_list.append(line.strip())


    def init_draw_params(self):
        """ Initialize Draw Parameters """
        #for draw result and boundingbox
        self.frame_idx = 0
        self.frame_size = None
        self.font_size  = None
        self.font_thick = None
        self.thick      = None
        self.draw_result =self.params['application'].get('draw_result',True)

        #for draw area
        self.area_name={}
        self.draw_bbox =self.params['application'].get('draw_bbox',True)
        self.draw_tracking = True
        self.draw_area= True
        self.area_opacity=None
        self.area_color=None
        self.area_pts = {}
        self.normalize_area_pts = {}
        self.change_resulutuon = 1
        self.area_cnt = {}

    def update_draw_param(self, frame):
        """ Update the parameters of the drawing tool, which only happend at first time. """
        
        # if frame_size not None means it was already init 
        if( self.frame_idx > 1): return None

        # Parameters
        FRAME_SCALE     = 0.0005    # Custom Value which Related with Resolution
        BASE_THICK      = 1         # Setup Basic Thick Value
        BASE_FONT_SIZE  = 0.5   # Setup Basic Font Size Value
        FONT_SCALE      = 0.2   # Custom Value which Related with the size of the font.

        # Get Frame Size
        self.frame_size = frame.shape[:2]
        
        # Calculate the common scale
        scale = FRAME_SCALE * sum(self.frame_size)
        
        # Get dynamic thick and dynamic size 
        self.thick  = BASE_THICK + round( scale )
        self.font_thick = self.thick//2
        self.font_size = BASE_FONT_SIZE + ( scale*FONT_SCALE )
        # self.draw_result = self.params['application']['draw_result'] if self.params['application'].__contains__('draw_result') else True
        # self.draw_bbox = self.params['application']['draw_bbox'] if self.params['application'].__contains__('draw_bbox') else True
        
        self.area_color=[0,0,255]
        self.area_opacity=0.2
        logging.info('Frame: {} ({}), Get Border Thick: {}, Font Scale: {}, Font Thick: {}'
            .format(self.frame_size, scale, self.thick, self.font_size, self.font_thick))        
        for i in range(len(self.params['application']['areas'])):
            if self.params['application']['areas'][i].__contains__('area_point'):
                if self.params['application']['areas'][i]['area_point']!=[]:
                    self.normalize_area_pts.update({i:self.params['application']['areas'][i]['area_point']})
                    self.area_name.update({i:self.params['application']['areas'][i]['name']})
                    # self.area_color.update({i:[random.randint(0,255),random.randint(0,255),random.randint(0,255)]})
                else:
                    self.normalize_area_pts.update({i:[[0,0],[1,0],[1,1],[0,1]]})
                    self.area_name.update({i:"The defalt area"})
            else:
                self.normalize_area_pts.update({i:[[0,0],[1,0],[1,1],[0,1]]})
                self.area_name.update({i:"The defalt area"})

    def collect_depand_info(self):
        for i in range(len(self.params['application']['areas'])): 
           
            if len(self.params['application']['areas'][i]['depend_on'])>0:
                        
                self.depend_on.update({i:self.params['application']['areas'][i]['depend_on']})
            else:
                self.depend_on.update({i:[]})    
        
        # temp_palette ={}
        # for area , value in self.depend_on.items():
        #     temp_palette.update({area:{}})
        #     if not self.depend_on.__contains__(area): 
        #         temp_palette.update({area:self.palette})
        #         continue
        #     if self.depend_on[area]==[]:
        #         temp_palette.update({area:self.palette})
        #         continue
                
        #     for id in range(len(value)):
        #         if not (value[id] in self.model_label_list): continue
        #         if not (self.params['application']['areas'][area].__contains__('palette')): 
        #             temp_palette[area].update({value[id]:self.palette[value[id]]})
        #             continue
        #         if not (self.params['application']['areas'][area]['palette'].__contains__(value[id])): 
        #             temp_palette[area].update({value[id]:self.palette[value[id]]})
        #             continue  
        #         # if self.palette.__contains__(value[id]):
        #         temp_palette[area].update({value[id]:self.params['application']['areas'][area]['palette'][value[id]]})
        
        # self.palette = temp_palette
        
    def init_logic_param(self):
        
        for i in range(len(self.params['application']['areas'])):
            if self.params['application']['areas'][i].__contains__('events'):
                
                self.operator.update({i:self.get_logic_event(self.params['application']['areas'][i]['events']['logic_operator'])})
                self.thres.update({i: self.params['application']['areas'][i]['events']['logic_value']})
                self.event_title.update({i:self.params['application']['areas'][i]['events']['title']})
                if self.params['application']['areas'][i]['events'].__contains__('cooldown_time'):
                    self.cooldown_time.update({i:self.params['application']['areas'][i]['events']['cooldown_time']})
                else :
                    self.cooldown_time.update({i:10})   
                if self.params['application']['areas'][i]['events'].__contains__('sensitivity'):
                    self.sensitivity.update({i:self.get_sensitivity_event(self.params['application']['areas'][i]['events']['sensitivity'])})
            else:
                logging.warning("No set event!")
    def get_sensitivity_event(self,sensitivity_str):
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
  
    def init_event_object(self):
        for i , v   in self.operator.items():
            event_obj = event_handle(self.operator,self.thres,self.cooldown_time,self.event_title,i) 
            self.event_handler.update( { i: event_obj }  )        
    
    def init_scale(self,frame):
        return frame.shape[0]/1920,frame.shape[1]/1080
                   
    def init_area_mask(self,frame):
       
        for i in range(len(self.area_pts)):
            
            poly = np.array(self.area_pts[i], dtype= np.int32)
            tmp = np.zeros([frame.shape[1],frame.shape[0]],dtype=np.uint8)
            cv2.polylines(tmp , [poly] ,1, 1)
            cv2.fillPoly(tmp,[poly],1)
            # print(tmp)
            
            # cv2.waitKey(0)
            self.area_mask.update({i:tmp})
    
    def app_common_start(self)    :
        
        self.app_thread=app_common_handle(self.params, self.area_mask, self.depend_on, self.total, self.app_output, self.area_pts, self.area_name, self.sensitivity,self.tracking_distance)
        self.app_thread.start()  

    def event_start(self):
        for i in range(len(self.event_handler)):
            self.event_handler[i].start()       
    
    @staticmethod
    def sort_point_list(point_list:list):
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

    def draw_area_event(self, frame, is_draw_area, area_color=None, area_opacity=None, draw_points=True, draw_polys=True):
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

    def custom_function(self, frame, color:tuple, label,score, left_top:tuple, right_down:tuple,draw_bbox=True,draw_result=True):
        """ The draw method customize by user 
        """
        (xmin, ymin), (xmax, ymax) = left_top, right_down
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
    

    def get_color(self, label):
       
       return self.palette[label]      

    def check_input(self,frame,data):

        if frame is None : return False
        # if not data.__contains__('detections') or data['detections'] is None   : return False 
        if data is None   : return False 
        return True         
         
    def draw_tag(self,tracking_tag,xmin, ymin, xmax, ymax,outer_clor,font_color,frame,draw_tracking=True):
        draw_tracking = self.draw_tracking
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

    def convert_point_value(self,frame):
        #convert point value.
        
        if not self.change_resulutuon : return
        temp_point=[]

        
        for area in range(len(self.normalize_area_pts)):
                 
            for point in self.normalize_area_pts[area]:
                if point[0]>1: return
                temp_point.append([math.ceil(point[0]*frame.shape[1]),math.ceil(point[1]*frame.shape[0])])
            temp_point = self.sort_point_list(temp_point)
            self.area_pts.update({area:temp_point})
            temp_point = []
            self.change_resulutuon = 0
    
    
    def set_draw(self,params:dict):
        """
        Control anything about drawing.
        Which params you can contral :

        { 
            draw_area : bool , 
            draw_bbox : bool ,
            draw_result : bool ,
            draw_tracking : bool ,
            palette: list[ tuple:( label:str , color:Union[tuple , list] ) ]
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

        palette = params.get('palette', None)
        if isinstance(palette, list):
            if len(palette)==0:
                logging.warning("Not set palette!")
                pass
            else:
                for info in palette:
                    (label , color) = info
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


    def __call__(self, frame, detections, draw=True):
        self.app_thread.update_tracking_distance(self.tracking_distance)
        ori_frame = frame.copy()
        if not self.check_input(frame,detections) :
            logging.info("Input frame or input data format is error !!")
            return frame
        self.frame_idx += 1
        self.update_draw_param( frame )
        self.convert_point_value(frame)
        self.init_area_mask(frame)
       
        if cv2.waitKey(1) in [ ord('c'), 99 ]: self.draw_area= self.draw_area^1
        frame = self.draw_area_event(frame, self.draw_area)

        # each frame we need return to zero
        self.app_thread.total={}
        self.app_thread.app_output={}
        self.event_output={'event':[]}
        self.app_thread.show_object =[]
        # for id,det in enumerate(data['detections']):
        for detection in detections:
            # Check Label is what we want
            ( label, score, xmin, ymin, xmax, ymax ) \
                 = detection.label, detection.score, detection.xmin, detection.ymin, detection.xmax, detection.ymax                  
            
            # N Area
            for i in range(len(self.depend_on)):
                    
                # Delete un-tracked object

                # self.pool.apply_async(self.app_thread,(i,label, score, xmin, ymin, xmax, ymax,self.area_pts,frame))
                self.app_thread(i,label, score, xmin, ymin, xmax, ymax,self.area_pts,frame)
            
                 # app common result display
                if (self.app_thread.is_draw): 
                    if len(self.depend_on[i])>0:
                        if  not (label in self.depend_on[i]): continue
                    #draw the tracking tag for each object
                    outer_clor = self.get_color(label)
                    font_color = (255,255,255)
                    self.draw_tag(self.app_thread.show_object_info, xmin, ymin, xmax, ymax,outer_clor ,font_color,frame)
                    #draw bbox and result
                    frame = self.custom_function(
                            frame = frame,
                            color = self.get_color(label),
                            label = label,
                            score=score,
                            left_top = (xmin, ymin),
                            right_down = (xmax, ymax)
                        )
                    self.app_thread.show_object.append(self.app_thread.object_id) 
                    self.app_thread.is_draw=False


                #if the area don't have set event. 
                if self.event_handler.__contains__(i)==False: continue
                
                # self.pool.apply_async(self.event_handler[i],(frame,ori_frime,i,self.app_thread.total,self.app_thread.app_output))
                self.event_handler[i](frame,ori_frame,i,self.app_thread.total,self.app_thread.app_output)
                if (self.event_handler[i].pass_time == self.event_handler[i].cooldown_time[i]):
                    self.app_thread.total[i]=0
                if self.event_handler[i].event_output !={}:
                    self.event_output['event'].append(self.event_handler[i].event_output)
                    #ivit-i-hailo
        # if frame and self.app_thread.app_output and self.event_output:
        return (frame , self.app_thread.app_output , self.event_output)
