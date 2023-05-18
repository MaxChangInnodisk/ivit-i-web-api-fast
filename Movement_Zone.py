import sys, os, cv2, logging, time
import numpy as np
import uuid
import os
import threading
import math
from . import types as Error
from datetime import datetime
from typing import Union, get_args
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
        self.area_id = area_id
        
        self.pass_time = self.cooldown_time[self.area_id] +1
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

        if not self.logic_event(total_object_number[area_id],self.thres[area_id],area_id):
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
            
            self.info = "The {} area ".format(area_id)+self.event_title[area_id]+" "+' , '.join([ 'total:{}  ,cooldown time:{}/{}'.format(total_object_number[area_id],0,self.cooldown_time[area_id])])

           
            # print(self.event_output,'\n')
        else :
            
            self.pass_time = (int(self.event_time.minute)*60+int(self.event_time.second))-(int(self.trigger_time.minute)*60+int(self.trigger_time.second))    
            self.info = "The {} area ".format(area_id)+self.event_title[area_id]+" "+' , '.join([ 'total:{}  ,cooldown time:{}/{}'.format(total_object_number[area_id],self.pass_time,self.cooldown_time[area_id])])

class app_common_handle(threading.Thread):
    # def __init__(self ,params:dict,area_mask:dict,depend_on:dict,total:dict,app_output:dict,area_pts:dict,area_name:dict,sensitivity:dict,tracking_distance,line_point:dict,line_relationship:dict):
    def __init__(self ,params:dict,area_mask:dict,depend_on:dict,total:dict,app_output:dict,area_pts:dict,area_name:dict,sensitivity:dict,tracking_distance,line_relationship:dict):
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
        # self.line_point=line_point
        self.line_relationship=line_relationship
        self.show_object=[]
        self.show_object_info=""
        self.is_draw=False
    
    
    def update_tracking_distance(self,frame):
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
                self.track_object.update({i:{0:{'x':center_x,'y':center_y,'frame_time':time.time(),'torch_line':{}}}})
                self.object_buffer.update({i:{}})
               
                self.total_object.update({i:0})
        if self.app_output.__contains__("areas")==False:
            self.app_output.update({"areas": []})
        if len(self.app_output["areas"])==area_id:
            self.app_output["areas"].append({"id":area_id,"name":self.area_name2[area_id],"data":[]})      

    def inpolygon_mask(self,left_up:list,right_down:list,area_id,frame):
        l_r = left_up
        m_up = [(right_down[0]+left_up[0])//2,left_up[1]]
        r_up = [right_down[0],left_up[1]]
        r_m = [right_down[0],(right_down[1]+left_up[1])//2]
        m_d = [(right_down[0]+left_up[0])//2,right_down[1]]
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
        sum_original = (object_mask==1).sum()
        if sum_original: return 0.0

        overlapping=area_mask+object_mask
        sum_overlay = (overlapping==2).sum()
        
        # print(" countiog_IOU :{} = {} / {}".format((overlapping==2).sum()/(object_mask).sum(),(overlapping==2).sum(),(object_mask).sum()))
        
        return sum_overlay/sum_original
    
    def cal_distance(self,p1x,p1y,p2x,p2y):
        # print("x {} , y{}".format(math.sqrt(math.pow(p1x-p2x,2)),math.sqrt(math.pow(p1y-p2y,2))))
        return round(math.sqrt(math.pow(p1x-p2x,2))+math.sqrt(math.pow(p1y-p2y,2)))

    def delete_object_point(self,xmin,xmax,ymin,ymax,area_id):
        temp_xy = self.tracking_distance
        temp_id = int
        for object_id , object_value in self.track_object[area_id].items():
            if temp_xy > self.cal_distance(object_value['x'],object_value['y'],(xmin+xmax)//2,(ymin+ymax)//2) :
                temp_xy = self.cal_distance(object_value['x'],object_value['y'],(xmin+xmax)//2,(ymin+ymax)//2)
                temp_id = object_id
        if temp_xy != self.tracking_distance:
            self.object_buffer[area_id].update({
                temp_id:{
                        'x': (xmin+xmax)//2,
                        'y': (ymin+ymax)//2,
                        'frame_time': time.time()
             }})
            
            del self.track_object[area_id][temp_id]

    def is_point_cross_trigger_line(self,p1,p2,line_point,area_id):
        temp_id = ""
        cross_flag=False
        
        for trigger_id,trigger_point in line_point[area_id].items():
            
            # print(" area total{} ,p1 :({},{}) , p2 :({},{})".format(trigger_point,trigger_point[0][0],trigger_point[0][1],trigger_point[1][0],trigger_point[1][1]))
            trigger_line=np.array([trigger_point[0][0]-trigger_point[1][0],trigger_point[0][1]-trigger_point[1][1]])
            trigger_line_side=np.array([trigger_point[0][0]-p1[0],trigger_point[0][1]-p1[1]])
            trigger_line_side2=np.array([trigger_point[0][0]-p2[0],trigger_point[0][1]-p2[1]])

            trigger_cross=np.cross(trigger_line,trigger_line_side)*np.cross(trigger_line,trigger_line_side2)
            
            if trigger_cross<0:
                
                temp_id=trigger_id
                cross_flag=True
                # print(temp_id," ",trigger_cross)


        # if len(temp)>1:
        #     return True ,  [0]
        
        return cross_flag ,temp_id
        
    def map_line_relationship(self,area_id, all_cross_line:dict):
        temp_cross_line=""
        
        for id , val in all_cross_line.items():
            temp_cross_line=temp_cross_line+val
        for i in range(len(self.line_relationship[area_id])) :
            if self.line_relationship[area_id][i].__contains__(temp_cross_line):
                
                return self.line_relationship[area_id][i][temp_cross_line]

    def check_tarch_the_line(self,object_dict,lineID):
        if object_dict!=lineID:
            return True
        return False
    
    def check_object_have_direction_info(self,object_dict):
        if len(object_dict) %2 ==0:
            return True
        return False
    
    def update_object_point(self,frame,xmin,xmax,ymin,ymax,area_id,line_point):
        temp_xy = self.tracking_distance
        
        tracked = 0
        buffer_distance=60
        temp_direction=""

        #store last frame point 
        temp_last_point=[]
        
        coby_track_object=self.track_object[area_id].copy()
        coby_track_object_buffer=self.object_buffer[area_id].copy()
        
        for i , v in coby_track_object_buffer.items():
            
            
            if buffer_distance > self.cal_distance(v['x'],v['y'],(xmin+xmax)//2,(ymin+ymax)//2):
                self.object_buffer[area_id][i]['frame_time']=time.time()
                return temp_direction , tracked 
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
                
                temp_last_point=[object_value['x'],object_value['y']]
                tracked = 1

        if not tracked:
            self.total_object[area_id]=self.total_object[area_id]+1
            
            self.track_object[area_id].update({self.total_object[area_id]:{'x':(xmin+xmax)//2,'y':(ymin+ymax)//2,'frame_time':time.time(),'torch_line':{}}})    
            self.show_object_info="Area{}: {}".format(str(area_id),str(self.object_id))
            return temp_direction , tracked
        else:
            
            temp_torch_line=self.track_object[area_id][self.object_id]['torch_line']
            
            self.track_object[area_id].update({self.object_id:{'name':str(),'x':(xmin+xmax)//2,'y':(ymin+ymax)//2,'frame_time':time.time() ,'torch_line':temp_torch_line}}) 
        self.show_object_info="Area{}: {}".format(str(area_id),str(self.object_id))
        
        
        check_is_cross, temp_cross_line =self.is_point_cross_trigger_line(temp_last_point,[(xmin+xmax)//2,(ymin+ymax)//2],line_point,area_id)
        if not check_is_cross: return temp_direction , tracked 

        if len(self.track_object[area_id][self.object_id]['torch_line'])==0:
            self.track_object[area_id][self.object_id]['torch_line'].update({1:temp_cross_line})
        else:    
            
            # check object is tarch the line
            if self.check_tarch_the_line(self.track_object[area_id][self.object_id]['torch_line'][len(self.track_object[area_id][self.object_id]['torch_line'])],temp_cross_line):
                self.track_object[area_id][self.object_id]['torch_line'].update({len(self.track_object[area_id][self.object_id]['torch_line'])+1:temp_cross_line})
            # print("temp id {} ,update {}:".format(temp_id,self.track_object[temp_id]['torch_line']))
        
        # chaeck have direction info
        if self.check_object_have_direction_info(self.track_object[area_id][self.object_id]['torch_line']):
            
            temp_direction=self.map_line_relationship(area_id,self.track_object[area_id][self.object_id]['torch_line'])
            
            del self.track_object[area_id][self.object_id]['torch_line']
            self.track_object[area_id][self.object_id].update({'torch_line':{}})
            # print(" Objecdt:{} is {} .".format(temp_id,temp_direction))
        
        
        return temp_direction , tracked
    @staticmethod
    def inpolygon(px,py,poly):
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

    def check_direction_result(self,direction):
        if direction!="":
            return True
        return False
    
    def check_depend_on(self,area_id):
        if len(self.depend_on[area_id])!=0:
            return True
        return False

    def __call__(self,area_id:int,label:str, score:float, xmin:int, ymin:int, xmax:int, ymax:int,area,frame,line_point):
        
        self.init_tracking(xmin,xmax,ymin,ymax,area_id)      
        """
        Update track_obj
        """
        self.area_pts2 = area
        if self.sensitivity.__contains__(area_id):
            
            if self.inpolygon_mask([xmin, ymin],[xmax,ymax],area_id,frame)<self.sensitivity[area_id]:
                self.delete_object_point(xmin,xmax,ymin,ymax,area_id)
                return
        else:
            
            if not  self.inpolygon(((xmin+xmax)/2),((ymin+ymax)/2),self.area_pts2[area_id]): 
                
                self.delete_object_point(xmin,xmax,ymin,ymax,area_id)
                return

        # if not self.area_mask[area_id][(xmin+xmax)//2, (ymin+ymax)//2]: return
       
        if self.check_depend_on(area_id):

            if label in self.depend_on[area_id]:  
                
     
                direction , new_object_detection =self.update_object_point(frame,xmin,xmax,ymin,ymax,area_id,line_point)  
                  
                if self.check_direction_result(direction):
                    self.update_object_number_this_frame(area_id)
                    # print(self.total)
                    #according to area and depend on to creat app output. Do once.
                    if len(self.line_relationship[area_id])!=len(self.app_output["areas"][area_id]["data"]):
                        for i in range(len(self.line_relationship[area_id])):
                            for key in self.line_relationship[area_id][i].values():
                                self.app_output["areas"][area_id]["data"].append({"label":key,"num":0})

                    for d in range(len(self.app_output["areas"][area_id]["data"])): 

                        if self.app_output["areas"][area_id]["data"][d]["label"]==direction :
                            _=self.app_output["areas"][area_id]["data"][d]["num"]+1
                            
                            self.app_output["areas"][area_id]["data"][d].update({"num":_})
                            
                if self.show_object==[]:
                    self.is_draw=True
                elif self.object_id in self.show_object:
                    self.is_draw=False  
                else :
                    self.is_draw=True      
                # print(self.app_output,'\n')    
        else:

            direction , new_object_detection =self.update_object_point(frame,xmin,xmax,ymin,ymax,area_id,line_point)

            if self.check_direction_result(direction):
                
                self.update_object_number_this_frame(area_id) 
                
                
                #according to area and depend on to creat app output. Do once.
                if len(self.line_relationship[area_id])!=len(self.app_output["areas"][area_id]["data"]):
                    for i in range(len(self.line_relationship[area_id])):
                        for key in self.line_relationship[area_id][i].values():
                            self.app_output["areas"][area_id]["data"].append({"label":key,"num":0})

                for d in range(len(self.app_output["areas"][area_id]["data"])): 

                    if self.app_output["areas"][area_id]["data"][d]["label"]==direction :
                        _=self.app_output["areas"][area_id]["data"][d]["num"]+1
                        
                        self.app_output["areas"][area_id]["data"][d].update({"num":_})
            if self.show_object==[]:
                self.is_draw=True
            elif self.object_id in self.show_object:
                self.is_draw=False  
            else :
                self.is_draw=True  
            # print(self.app_output,'\n')  

class Movement_Zone(iAPP_OBJ,event_handle,app_common_handle):

    def __init__(self, params=None, label=None, palette=palette, log=True):
    
        
        self.params = params  

        self.check_params()
        # ( correct , error_list )=self.verify_params(params)
        # if not correct :
        #     raise Error.LinePointIncorrect(error_list)
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
        self.event_handler = {}
        self.event_output ={}
        self.area_mask={}


        self.track_object={}
        self.tracking_distance=60
        self.total_object=0



        self.line_point ={}
        self.normalize_line_point={}
        self.line_relation={}
        self.line_relationship={}

        self.model_label = label
        self.model_label_list =[]

        # self.pool = ThreadPool(os.cpu_count() )
    
        self.init_palette(palette)

        self.collect_depand_info()
        self.init_draw_params()
        self.init_logic_param()
        
        self.init_line_relation()
        self.app_common_start()
        self.init_event_object()
        self.event_start()

    @staticmethod
    def verify_params(params:dict):
        """

        To verify the line point whether or not in area point.
     
        Args:
            params (dict): app config.

        Returns:
            Tuple:( bool , error:list )
        """
        error = []
        error_temp=[]
        if not params.__contains__('application'):
            logging.error('App config is not set application, plz set application in app config')
            return False
        elif not params['application'].__contains__('areas'):
            logging.error('App config is not set area , plz set area in app config')
            return False
        
        num_area = len(params['application']['areas'])

        if num_area==0:
            logging.error('App config is not set area info , plz set area info in app config')
            return False

        for area_idx in range(num_area):
            
            for line_name,line_value in params['application']['areas'][area_idx]['line_point'].items():
                for point in line_value:
                    if not app_common_handle.inpolygon(point[0],point[1],params['application']['areas'][area_idx]['area_point']):
                
                        logging.error('Area name : {}  line_point is out of area_point !'.format(params['application']['areas'][area_idx]['name']\
                                                                       ,point[0],point[1] ))
                        error_temp.append(line_name)
                        break
            
            error.append((params['application']['areas'][area_idx]['name'],error_temp))
            error_temp=[]

        return ( len(error)==0 , error)

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
        self.draw_result =True
        self.draw_tracking = True
        #for draw area
        self.area_name={}
        self.draw_bbox =True
        self.draw_area=True
        self.is_draw_line = True
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
        self.draw_result = self.params['application'].get('draw_result',True)
        self.draw_bbox = self.params['application'].get('draw_bbox',True)
        self.area_color=[0,0,255]
        self.area_opacity=0.2
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
        logging.info('Frame: {} ({}), Get Border Thick: {}, Font Scale: {}, Font Thick: {}'
            .format(self.frame_size, scale, self.thick, self.font_size, self.font_thick))    
 
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

    def set_color(self,label:str,color:tuple):
        """
        set color :

        sample of paremeter : 
            label = "dog"
            color = (0,0,255)
        """
        self.palette.update({label:color})
        logging.info("Label: {} , change color to {}.".format(label,color))

    def get_color(self, label):
       
       return self.palette[label] 
   
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
            if self.params['application']['areas'][i].__contains__('line_point'):
                self.normalize_line_point.update({i:self.params['application']['areas'][i]['line_point']})
            if self.params['application']['areas'][i].__contains__('line_relation'):
                self.line_relation.update({i:self.params['application']['areas'][i]['line_relation']})


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
    def init_line_relation(self):
        for area_id ,val in self.line_relation.items():
            if not self.line_relationship.__contains__(area_id):
                self.line_relationship.update({area_id:[]})
               
            for i in range(len(val)):
                self.line_relationship[area_id].append(
                    
                    {
                        val[i]['start']+val[i]['end']: val[i]['name']
                        
                        
                    
                    }

                )
   
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
            event_obj = event_handle(self.operator,self.thres,self.cooldown_time,self.event_title ,i) 
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
                     
    def draw_line(self,frame,is_draw_line=True):
        is_draw_line = self.is_draw_line
        if not is_draw_line : return
        for id, val in self.line_point.items():
            
            for idx ,valx in val.items():
                cv2.putText(
                        frame,str(idx) , valx[1], cv2.FONT_HERSHEY_TRIPLEX,
                        self.font_size, (0,255,255), self.font_thick, cv2.LINE_AA
                    )
                
                cv2.line(frame, valx[0], valx[1], (0, 255, 255), 3)

    def app_common_start(self):
        
        # self.app_thread=app_common_handle(self.params, self.area_mask, self.depend_on, self.total, self.app_output, self.area_pts, self.area_name, self.sensitivity,self.tracking_distance,self.line_point,self.line_relationship)
        self.app_thread=app_common_handle(self.params, self.area_mask, self.depend_on, self.total, self.app_output, self.area_pts, self.area_name, self.sensitivity,self.tracking_distance,self.line_relationship)
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

        is_draw_area = self.draw_area
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


    

    def check_input(self,frame,data):

        if frame is None : return False
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

    def draw_direction_result(self,direction_result,outer_clor,font_color,frame):
        for id in  range(len(direction_result)):
            temp_direction_result="There are {} object {}.".format(str(direction_result[id]["num"]),direction_result[id]["label"])
            
            
            (t_wid, t_hei), t_base = cv2.getTextSize(temp_direction_result, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
            
            t_xmin, t_ymin, t_xmax, t_ymax = 10, 10*id+(id*(t_hei+t_base)), 10+t_wid, 10*id+((id+1)*(t_hei+t_base))
            
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), outer_clor , -1)
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), (0,0,0) , 1)
            cv2.putText(
                frame, temp_direction_result, (t_xmin, t_ymax), cv2.FONT_HERSHEY_SIMPLEX,
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
            
            #for convert line point
            if not self.normalize_line_point.__contains__(area) or self.normalize_line_point[area]=={}: continue
            self.line_point.update({area:{}})
            for line_name,line_point in self.normalize_line_point[area].items():
                self.line_point[area].update({line_name:[]})
                for point in  line_point:
                    self.line_point[area][line_name].append([math.ceil(point[0]*frame.shape[1]),math.ceil(point[1]*frame.shape[0])])

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
            draw_line : bool ,
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

        if isinstance(params.get('draw_line', self.is_draw_line) , bool):    
            self.is_draw_line= params.get('draw_line', self.is_draw_line)
            logging.info("Change draw_line mode , now draw_line mode is {} !".format(self.is_draw_line))
        else:
            logging.error("draw_line type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_line', self.is_draw_line))))

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
        


        if not self.check_input(frame,detections) :
            logging.info("Input frame or input data format is error !!")
            sys.exit(0)
        ori_frame= frame.copy()    
        self.frame_idx += 1
        self.update_draw_param( frame )
        self.convert_point_value(frame)
        self.draw_line(frame) 
        self.init_area_mask(frame)
        self.event_output={'event':[]}


        if cv2.waitKey(1) in [ ord('c'), 99 ]: self.draw_area= self.draw_area^1
        frame = self.draw_area_event(frame, self.draw_area)  
        
        self.app_thread.show_object =[]


        # for id,det in enumerate(data['detections']):
        for detection in detections:
            # Check Label is what we want
            ( label, score, xmin, ymin, xmax, ymax ) \
                 = detection.label, detection.score, detection.xmin, detection.ymin, detection.xmax, detection.ymax                  
            for i in range(len(self.depend_on)):
                
                self.app_thread(i,label, score, xmin, ymin, xmax, ymax,self.area_pts,frame,self.line_point)
                # self.pool.apply_async(self.app_thread,(i,label, score, xmin, ymin, xmax, ymax,self.area_pts,frame,self.line_point))
                
                
                if self.app_thread.is_draw:

                    
                    outer_clor = self.get_color(label)
                    font_color = (255,255,255)
                    self.draw_tag(self.app_thread.show_object_info, xmin, ymin, xmax, ymax,outer_clor ,font_color,frame)
                    
                    outer_clor = (0,255,255)
                    font_color = (0,0,0)
                    
                    self.draw_direction_result(self.app_thread.app_output["areas"][i]["data"],outer_clor,font_color,frame)
                
                    frame = self.custom_function(
                            frame = frame,
                            color = self.get_color(label) ,
                            label = label,
                            score=score,
                            left_top = (xmin, ymin),
                            right_down = (xmax, ymax)
                        ) 
                    self.app_thread.show_object.append(self.app_thread.object_id) 

                self.app_thread.is_draw=False

                if self.event_handler.__contains__(i)==False:
                    continue
                
                self.event_handler[i](frame,ori_frame,i,self.app_thread.total,self.app_thread.app_output)
                # self.pool.apply_async(self.event_handler[i],(frame,ori_frame,i,self.app_thread.total,self.app_thread.app_output))

                

                #cooltime end we recount .
                if (self.event_handler[i].pass_time == self.event_handler[i].cooldown_time[i]):
                    self.app_thread.total[i]=0

                if self.event_handler[i].event_output !={}:
                    self.event_output['event'].append(self.event_handler[i].event_output)
                    #ivit-i-hailo
        return(frame,self.app_thread.app_output,self.event_output)       
                