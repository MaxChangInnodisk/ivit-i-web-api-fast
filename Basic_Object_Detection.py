import sys, os, cv2
sys.path.append( os.getcwd() )
from apps.palette import palette
from ivit_i.utils.logger import ivit_logger
from ivit_i.app import iAPP_OBJ
class Basic_Object_Detection(iAPP_OBJ):    
    """ Basic Object Detection Application
    * Parameters
        1. depend_onarea_opacity
    * Function
        1. depend_label()
    """
    def __init__(self, params=None, label=None, palette=palette,log=True):
        self.app_type = 'obj'
        self.params = params
        
        if self.params:
            self.depend_on =self.params['application']['areas'][0]['depend_on']

        self.depend_on = []
        self.palette={}
        self.model_label = label
        self.model_label_list =[]
        self.logger = ivit_logger()
        
        self.init_palette(palette)
        self.init_draw_params()

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
        self.draw_area=True
        self.area_opacity=None
        self.area_color=None
        self.area_pts = {}
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
        if self.params:
            self.draw_result = self.params['application']['draw_result'] if self.params['application'].__contains__('draw_result') else True
            self.draw_bbox = self.params['application']['draw_bbox'] if self.params['application'].__contains__('draw_bbox') else True
        else :
            self.draw_result = True
            self.draw_bbox = True

        self.area_color=[0,0,255]
        self.area_opacity=0.4  
    
    
 
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

    def set_color(self,label:str,color:tuple):
        """
        set color :

        sample of paremeter : 
            label = "dog"
            color = (0,0,255)
        """
        self.palette.update({label:color})
        self.logger.info("Label: {} , change color to {}.".format(label,color))

    def get_color(self, label):

        return self.palette[label] 
       

    def __call__(self, frame, detections, draw=True) -> tuple:
        #collect depend_on for each area from config
        app_output={"areas":[{"id":0,"name":"default","data":[]}]}

        self.update_draw_param(frame=frame)

        # for id,det in enumerate(data['detections']):
        for detection in detections:
            # Check Label is what we want
            ( label, score, xmin, ymin, xmax, ymax ) \
                 = detection.label, detection.score, detection.xmin, detection.ymin, detection.xmax, detection.ymax  
            # if user have set depend on
            if len(self.depend_on)>0:
                              
                if label in self.depend_on[0] :

                    app_output['areas'][0]['data'].append({'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax,'label':label,'score':score,'id':detection.id})
                    frame = self.custom_function(
                        frame = frame,
                        color = self.get_color(label) ,
                        label = label,
                        score=score,
                        left_top = (xmin, ymin),
                        right_down = (xmax, ymax)
                    )
            else:
                app_output['areas'][0]['data'].append({'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax,'label':label,'score':score,'id':detection.id})
                
                frame = self.custom_function(
                            frame = frame,
                            color = self.get_color(label) ,
                            label = label,
                            score=score,
                            left_top = (xmin, ymin),
                            right_down = (xmax, ymax)
                        ) 
                                        
        return ( frame, app_output, {})

