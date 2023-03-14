import sys, os, cv2, logging, time
import numpy as np
import uuid
import os
import threading
import math
import random
from datetime import datetime
sys.path.append( os.getcwd() )
from ivit_i.app.common import ivitApp    

#add jay 20230313 for using defalt palette.
from ivit_i.app.palette import palette

class event_handle(threading.Thread):
    def __init__(self ,operator:dict,thres:dict,cooldown_time:dict,event_title:dict):
        threading.Thread.__init__(self)
        self.operator = operator
        self.thres = thres
        self.cooldown_time = cooldown_time
        self.event_title = event_title
        self.event_output={}
        self.pass_time = 59
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

    def __call__(self,frame,area_id,total_object_number,app_output):
        self.event_output={}
        self.event_time=datetime.now()
        self.info =""
        self.pass_time = (int(self.event_time.minute)*60+int(self.event_time.second))-(int(self.trigger_time.minute)*60+int(self.trigger_time.second))
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
            # if not os.path.isdir(path):
            #     os.mkdir(path)
            # cv2.imwrite(path+str(self.trigger_time)+'.jpg', frame)
            # cv2.imwrite(path+str(self.trigger_time)+"_org"+'.jpg', ori_frame)
            self.event_output.update({"uuid":uid,"title":self.event_title[area_id],"areas":app_output["areas"][area_id],"timesamp":self.trigger_time,"screenshot":{"overlay": path+str(self.trigger_time)+'.jpg',
            "original": path+str(self.trigger_time)+"_org"+'.jpg'}}) 
            # Draw Inforamtion
            
            self.info = "The {} area : ".format(area_id)+self.event_title[area_id]+' , '.join([ 'total:{}  , cool down time:{}/{}'.format(total_object_number[area_id],0,self.cooldown_time[area_id])])

            
            print(self.event_output,'\n')
            
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
        self.show_object=""

    def update_tracking_distance(self,frame):
        """

        """
        if frame.shape[0]<1080:
            self.tracking_distance = 60*(2/(math.pow(frame.shape[0]/1080,2)+math.pow(frame.shape[1]/1920,2))) 
        else:
            self.tracking_distance = 60*(2/(math.pow(1080/frame.shape[0],2)+math.pow(1920/frame.shape[1],2)))
    
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
            if time.time()-v['frame_time']>3:
                    del self.object_buffer[area_id][i] 
            if buffer_distance > self.cal_distance(v['x'],v['y'],(xmin+xmax)//2,(ymin+ymax)//2):
                self.object_buffer[area_id][i]['frame_time']=time.time()
                # print("id {} , val {} :".format(i,v))
                return  tracked      
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
        self.show_object="Area{}: {}".format(str(area_id),str(self.object_id))

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
                 
                
                # is new object have detection  
                if self.update_object_point(frame,xmin,xmax,ymin,ymax,area_id): 
                    self.update_object_number_this_frame(area_id)

                #according to area and depend on to creat app output. Do once.
                if len(self.depend_on[area_id])!=len(self.app_output["areas"][area_id]["data"]):
                    for key in self.depend_on[area_id]:
                        self.app_output["areas"][area_id]["data"].append({"label":key,"num":0})
                
                for d in range(len(self.app_output["areas"][area_id]["data"])):    
                    if self.app_output["areas"][area_id]["data"][d]["label"]==label:
                        #mod 20230310 jay for trancking is no 
                        # _=self.app_output["areas"][area_id]["data"][d]["num"]+1
                        if self.total.__contains__(area_id): 
                            #mod jay 2023 0313 . 
                            # self.app_output["areas"][area_id]["data"][d].update({"num":self.total[area_id]})
                            self.app_output["areas"][area_id]["data"][d].update({"num":self.total_object[area_id]+1})

                self.is_draw=True
                # print(self.app_output,'\n')
        else:
            #according to area and user setting to creat app output. Do once.
            if (len(self.app_output["areas"][area_id]["data"])==0): 
                self.app_output["areas"][area_id]["data"].append({"label":label,"num":1})

            
            
            if self.update_object_point(frame,xmin,xmax,ymin,ymax,area_id): 
                
                for d in range(len(self.app_output["areas"][area_id]["data"])): 
                        
                    if self.app_output["areas"][area_id]["data"][d]["label"]==label:
                        _=self.app_output["areas"][area_id]["data"][d]["num"]+1
                        self.app_output["areas"][area_id]["data"][d].update({"num":_})
                        break
                    #app output doesn't have this label , new object !
                    if len(self.app_output["areas"][area_id]["data"])-1==d and self.app_output["areas"][area_id]["data"][d]["label"]!=label: 
                        self.app_output["areas"][area_id]["data"].append({"label":label,"num":1})
            

            self.is_draw=True
            # print(self.app_output,'\n')                      

class Tracking(event_handle,app_common_handle ,ivitApp):
    #mod jay 20230313 for using defalt palette. 
    def __init__(self, params=None, label=None, palette=palette, log=True):
       
        self.app_type = 'obj'
        self.params = params
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
        self.tracking_distance=400
        self.total_object=0

        
        #add jay 20230313 for using defalt palette.
        self.model_label = label
        self.model_label_list =[]
        self.init_palette(palette)

        self.collect_depand_info()
        self.init_draw_params()
        self.init_logic_param()


        self.app_common_start()
        self.init_event_object()
        self.event_start()
        # parse area amount
        # assume areas means all area

    #add jay 20230313 for using defalt palette.
    def init_palette(self,palette):
        temp_id=1
        with open(self.model_label,'r') as f:
            line = f.read().splitlines()
            for i in line:
                self.palette.update({i:palette[str(temp_id)]})
                self.model_label_list.append(i)
                temp_id+=1

    def init_draw_params(self):
        """ Initialize Draw Parameters """
        #for draw result and boundingbox
        self.frame_idx = 0
        self.frame_size = None
        self.font_size  = None
        self.font_thick = None
        self.thick      = None
        self.draw_result =True
        
        #for draw area
        self.area_name={}
        self.draw_bbox =True
        self.draw_area= 1
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
        logging.info('Frame: {} ({}), Get Border Thick: {}, Font Scale: {}, Font Thick: {}'
            .format(self.frame_size, scale, self.thick, self.font_size, self.font_thick))        
        for i in range(len(self.params['application']['areas'])):
            if self.params['application']['areas'][i]['area_point']!=[]:
                self.normalize_area_pts.update({i:self.params['application']['areas'][i]['area_point']})
                self.area_name.update({i:self.params['application']['areas'][i]['name']})
                # self.area_color.update({i:[random.randint(0,255),random.randint(0,255),random.randint(0,255)]})
            else:
                self.normalize_area_pts.update({i:[[0,0],[1,0],[1,1],[0,1]]})
                self.area_name.update({i:"The defalt area"})

    def collect_depand_info(self):
        for i in range(len(self.params['application']['areas'])): 
           
            if len(self.params['application']['areas'][i]['depend_on'])>0:
                        
                self.depend_on.update({i:self.params['application']['areas'][i]['depend_on']})
            else:
                self.depend_on.update({i:[]})    
        #add jay 20230313 for using defalt palette. 
        
        temp_palette ={}
        for area , value in self.depend_on.items():
            temp_palette.update({area:{}})
            if not self.depend_on.__contains__(area): 
                temp_palette.update({area:self.palette})
                continue
            if self.depend_on[area]==[]:
                temp_palette.update({area:self.palette})
                continue
                
            for id in range(len(value)):
                if not (value[id] in self.model_label_list): continue
                if not (self.params['application']['areas'][area].__contains__('palette')): 
                    temp_palette[area].update({value[id]:self.palette[value[id]]})
                    continue
                if not (self.params['application']['areas'][area]['palette'].__contains__(value[id])): 
                    temp_palette[area].update({value[id]:self.palette[value[id]]})
                    continue  
                # if self.palette.__contains__(value[id]):
                temp_palette[area].update({value[id]:self.params['application']['areas'][area]['palette'][value[id]]})
        
        self.palette = temp_palette
        
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
            event_obj = event_handle(self.operator,self.thres,self.cooldown_time,self.event_title) 
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
  
    def draw_area_event(self, frame, is_draw_area, area_color=None, area_opacity=None, draw_points=True, draw_polys=True):
        """ Draw Detecting Area and update center point if need.
        - args
            - frame: input frame
            - area_color: control the color of the area
            - area_opacity: control the opacity of the area
        """
        #add 20230309 jay for control area display.
        is_draw_area = self.draw_area
        if not is_draw_area: return frame
        # Get Parameters
        area_color = self.area_color if area_color is None else area_color
        area_opacity = self.area_opacity if area_opacity is None else area_opacity
        
        # Parse All Area
        overlay = frame.copy()
        #add 20230309 jay for draw area outline , it can draw irregular rectangle.
        temp_area_next_point = []


        for area_idx, area_pts in self.area_pts.items():
            
                        
            if area_pts==[]: continue

            # draw area point
            if  draw_points: 
                [ cv2.circle(frame, tuple(pts), 3, area_color, -1) for pts in area_pts ]

            # if delet : referenced before assignment
            minxy,maxxy=(max(area_pts),min(area_pts))

            for pts in area_pts:
                #add 20230309 jay for draw area outline , it can draw irregular rectangle.
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

    #mod jay 20230313 for using defalt palette. 
    def get_color(self, label,area_id):
       
       return self.palette[area_id][label]      

    def check_input(self,frame,data):

        if frame is None : return False
        if not data.__contains__('detections') or data['detections'] is None   : return False 
        return True         
         
    def draw_tag(self,tracking_tag,xmin, ymin, xmax, ymax,outer_clor,font_color,frame):
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
            self.area_pts.update({area:temp_point})
            temp_point = []
            self.change_resulutuon = 0
    

    def __call__(self, frame, data, draw=True):
        #add 20230309 jay for check user input is correct.
        if not self.check_input(frame,data) :
            logging.info("Input frame or input data format is error !!")
            return frame
        self.frame_idx += 1
        self.update_draw_param( frame )
        self.convert_point_value(frame)
        self.init_area_mask(frame)
        #FIXME: 
        #add 20230309 jay for control area display.
        if cv2.waitKey(1) in [ ord('c'), 99 ]: self.draw_area= self.draw_area^1
        frame = self.draw_area_event(frame, self.draw_area)

        # each frame we need return to zero
        self.app_thread.total={}
        self.app_thread.app_output={}
        self.event_output={'event':[]}

        for id,det in enumerate(data['detections']):
            # Parsing output
            ( label, score, xmin, ymin, xmax, ymax ) \
                 = [ det[key] for key in [ 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax' ] ]                  
            
            # N Area
            for i in range(len(self.depend_on)):
                    
                # Delete un-tracked object
                
                self.app_thread(i,label, score, xmin, ymin, xmax, ymax,self.area_pts,frame)
            
                 # app common result display
                if (self.app_thread.is_draw): 
                    
                    #draw the tracking tag for each object
                    outer_clor = self.get_color(label,i)
                    font_color = (255,255,255)
                    self.draw_tag(self.app_thread.show_object, xmin, ymin, xmax, ymax,outer_clor ,font_color,frame)

                    #draw bbox and result
                    frame = self.custom_function(
                            frame = frame,
                            color = self.get_color(label,i),
                            label = label,
                            score=score,
                            left_top = (xmin, ymin),
                            right_down = (xmax, ymax)
                        ) 
                self.app_thread.is_draw=False


                #if the area don't have set event. 
                if self.event_handler.__contains__(i)==False: continue
                
                self.event_handler[i](frame,i,self.app_thread.total,self.app_thread.app_output)
                if self.event_handler[i].event_output !={}:
                    self.event_output['event'].append(self.event_handler[i].event_output)
                    #ivit-i-hailo
        return(frame,self.app_thread.app_output,self.event_output)     
                           
if __name__ == "__main__":

    import cv2
    from ivit_i.common.model import get_ivit_model

    # Define iVIT Model
    model_type = 'obj'
    model_anchor = [ 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 ]
    model_conf = { 
        "tag": model_type,
        "openvino": {
            "model_path": "./model/yolo-v3-tf/FP32/yolo-v3-tf.xml",
            "label_path": "./model/yolo-v3-tf/coco.names",
            "anchors": model_anchor,
            "architecture_type": "yolo",
            "device": "CPU",
            "thres": 0.9
        }
    }

    ivit = get_ivit_model(model_type)
    ivit.load_model(model_conf)
    ivit.set_async_mode()
    
    # Def Application
#     app_conf = {"application": {
# 		"name": "TrackingArea",
# 		"areas": [
# 				{
# 						"name": "The intersection of Datong Rd",
# 						"depend_on": [ "car"],
						
#                         "area_point": [ ], 
						
# 				}
# 		],
#         "draw_result":False,
#         "draw_bbox":False
# }
#  }   
    app_conf = {"application": {
		"name": "TrackingArea",
		"areas": [
				{
						"name": "The intersection of Datong Rd",
						"depend_on": [ "car", "truck" ],
						"area_point": [ [0,0], [640, 0], [480, 640], [0, 480] ], 
						"events": {
										"title": "The daily traffic is over 1000",
										"logic_operator": ">",
										"logic_value": 1000,
						}
				}
		],
        "draw_result":True,
        "draw_bbox":True
}





    }

    app_conf = {"application": {
		"name": "TrackingArea",

		"areas": [
				{
						"name": "Datong Rd",
						"depend_on": ["car", "truck", "motocycle"  ],
                        
						"area_point": [  ], 
						"events": {
										"title": "1111",
										"logic_operator": ">",
										"logic_value": 0,
						}
				},
                
		],
        "draw_result":False,
        "draw_bbox":False
}
}
    
    app = Tracking( 
        params=app_conf, 
        label=model_conf['openvino']['label_path']
    )

    # Get Source
    src_path = './data/car.mp4'   # 1280x720/
    # src_path = './data/4-corner-downtown.mp4' # 1920x1080
    src_path = './data/video.mp4'   # 1920x1080
    
    cap = cv2.VideoCapture(src_path)

    # Set up Area
    ret, frame = cap.read()
    # app.set_area(frame)

    output = None
    a=0
    totalt=0
    # frame_counter = 0
    # play=1
    while(cap.isOpened()):
        
        # if (play):
        ret, frame = cap.read()
        # frame_counter +=1
        # if frame_counter==int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        #     frame_counter=0
        #     cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        if not ret: break
        _output = ivit.inference(frame=frame)
        output = _output if _output is not None else output 
        
        if (output):
            start = time.time()
            frame , app_output , event_output = app(frame, output)
            print(" app_output : {} , event_output {} ".format(app_output , event_output))
            # print(info)
            # print('\n')
            end = time.time()
            totalt=totalt+(end-start)
            a=a+1
            # if a %30==0:
            #     cv2.imwrite('./'+str(a)+'.jpg',frame)
            #     print(a/totalt)
            
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) in [ ord('q'), 27 ]: break
        # key = cv2.waitKey(0)
        # if key in [ ord('q'), 27 ]: break

        # if cv2.waitKey(1) in [ ord('q'), 27 ]: play=play^1
        
    cap.release()
    ivit.release()
